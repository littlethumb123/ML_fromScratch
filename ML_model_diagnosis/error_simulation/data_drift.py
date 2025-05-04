import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                            f1_score, roc_auc_score, average_precision_score,
                            roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def plot_metrics_comparison(metrics_dict, title):
    """Plot metrics comparison between original and shifted distributions"""
    metrics = list(metrics_dict.keys())
    original_values = [metrics_dict[m]['original'] for m in metrics]
    shifted_values = [metrics_dict[m]['shifted'] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original')
    plt.bar(x + width/2, shifted_values, width, label='Shifted')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Metrics Comparison - {title}')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(cm_original, cm_shifted, title):
    """Plot confusion matrices for original and shifted distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Original Distribution')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(cm_shifted, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Shifted Distribution')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.suptitle(f'Confusion Matrices - {title}')
    plt.tight_layout()
    plt.show()
    
def plot_curves(y_true_orig, y_pred_prob_orig, y_true_shift, y_pred_prob_shift, title):
    """Plot ROC and PR curves for original and shifted distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr_orig, tpr_orig, _ = roc_curve(y_true_orig, y_pred_prob_orig)
    fpr_shift, tpr_shift, _ = roc_curve(y_true_shift, y_pred_prob_shift)
    
    ax1.plot(fpr_orig, tpr_orig, label=f'Original (AUC = {roc_auc_score(y_true_orig, y_pred_prob_orig):.3f})')
    ax1.plot(fpr_shift, tpr_shift, label=f'Shifted (AUC = {roc_auc_score(y_true_shift, y_pred_prob_shift):.3f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # PR Curve
    precision_orig, recall_orig, _ = precision_recall_curve(y_true_orig, y_pred_prob_orig)
    precision_shift, recall_shift, _ = precision_recall_curve(y_true_shift, y_pred_prob_shift)
    
    ax2.plot(recall_orig, precision_orig, label=f'Original (AP = {average_precision_score(y_true_orig, y_pred_prob_orig):.3f})')
    ax2.plot(recall_shift, precision_shift, label=f'Shifted (AP = {average_precision_score(y_true_shift, y_pred_prob_shift):.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.suptitle(f'{title}')
    plt.tight_layout()
    plt.show()

def calculate_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def evaluate_model(model, X, y):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    specificity = calculate_specificity(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_prob)
    pr_auc = average_precision_score(y, y_pred_prob)
    
    return {
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }, y_pred_prob

def simulate_label_shift():
    """Simulate label shift - P(Y) changes but P(X|Y) remains the same"""
    print("\n=== Simulating Label Shift ===")
    
    # Generate original data
    n_samples = 5000
    
    # Create features for class 0 and class 1
    n_class0 = int(0.8 * n_samples)  # Original distribution: 80% class 0, 20% class 1
    n_class1 = n_samples - n_class0
    
    X_class0 = np.random.normal(loc=0, scale=1, size=(n_class0, 2))
    X_class1 = np.random.normal(loc=2, scale=1, size=(n_class1, 2))
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
    
    # Train a model on the original distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on original distribution
    metrics_orig, y_pred_prob_orig = evaluate_model(model, X_test, y_test)
    print("Original distribution metrics:")
    print(f"Precision: {metrics_orig['precision']:.3f}")
    print(f"Recall: {metrics_orig['recall']:.3f}")
    print(f"Specificity: {metrics_orig['specificity']:.3f}")
    print(f"F1 Score: {metrics_orig['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_orig['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_orig['pr_auc']:.3f}")
    
    # Create shifted data with different class distribution (label shift)
    # P(Y) changes but P(X|Y) remains the same
    n_samples_shift = 5000
    n_class0_shift = int(0.4 * n_samples_shift)  # Shifted distribution: 40% class 0, 60% class 1
    n_class1_shift = n_samples_shift - n_class0_shift
    
    X_class0_shift = np.random.normal(loc=0, scale=1, size=(n_class0_shift, 2))
    X_class1_shift = np.random.normal(loc=2, scale=1, size=(n_class1_shift, 2))
    
    X_shift = np.vstack([X_class0_shift, X_class1_shift])
    y_shift = np.hstack([np.zeros(n_class0_shift), np.ones(n_class1_shift)])
    
    # Evaluate on shifted distribution
    metrics_shift, y_pred_prob_shift = evaluate_model(model, X_shift, y_shift)
    print("\nShifted distribution metrics:")
    print(f"Precision: {metrics_shift['precision']:.3f}")
    print(f"Recall: {metrics_shift['recall']:.3f}")
    print(f"Specificity: {metrics_shift['specificity']:.3f}")
    print(f"F1 Score: {metrics_shift['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_shift['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_shift['pr_auc']:.3f}")
    
    # Plot comparisons
    metrics_dict = {
        'Precision': {'original': metrics_orig['precision'], 'shifted': metrics_shift['precision']},
        'Recall': {'original': metrics_orig['recall'], 'shifted': metrics_shift['recall']},
        'Specificity': {'original': metrics_orig['specificity'], 'shifted': metrics_shift['specificity']},
        'F1': {'original': metrics_orig['f1_score'], 'shifted': metrics_shift['f1_score']},
        'ROC-AUC': {'original': metrics_orig['roc_auc'], 'shifted': metrics_shift['roc_auc']},
        'PR-AUC': {'original': metrics_orig['pr_auc'], 'shifted': metrics_shift['pr_auc']}
    }
    
    plot_metrics_comparison(metrics_dict, "Label Shift")
    plot_confusion_matrices(metrics_orig['confusion_matrix'], metrics_shift['confusion_matrix'], "Label Shift")
    plot_curves(y_test, y_pred_prob_orig, y_shift, y_pred_prob_shift, "Label Shift - ROC and PR Curves")
    
    return metrics_orig, metrics_shift

def simulate_covariate_shift():
    """Simulate covariate shift - P(X) changes but P(Y|X) remains the same"""
    print("\n=== Simulating Covariate Shift ===")
    
    # Generate original data
    n_samples = 5000
    
    # Create features with a specific distribution
    X1 = np.random.normal(loc=0, scale=1, size=n_samples)
    X2 = np.random.normal(loc=0, scale=1, size=n_samples)
    X = np.column_stack([X1, X2])
    
    # Create target based on a fixed relationship
    # y = 1 if X1 + X2 > 0, else 0
    y = (X1 + X2 > 0).astype(int)
    
    # Train a model on the original distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on original distribution
    metrics_orig, y_pred_prob_orig = evaluate_model(model, X_test, y_test)
    print("Original distribution metrics:")
    print(f"Precision: {metrics_orig['precision']:.3f}")
    print(f"Recall: {metrics_orig['recall']:.3f}")
    print(f"Specificity: {metrics_orig['specificity']:.3f}")
    print(f"F1 Score: {metrics_orig['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_orig['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_orig['pr_auc']:.3f}")
    
    # Create shifted data with different feature distribution (covariate shift)
    # P(X) changes but P(Y|X) remains the same
    n_samples_shift = 5000
    
    # Shifted feature distribution (different means and scales)
    X1_shift = np.random.normal(loc=1, scale=1.5, size=n_samples_shift)
    X2_shift = np.random.normal(loc=-0.5, scale=0.8, size=n_samples_shift)
    X_shift = np.column_stack([X1_shift, X2_shift])
    
    # Same target function
    y_shift = (X1_shift + X2_shift > 0).astype(int)
    
    # Evaluate on shifted distribution
    metrics_shift, y_pred_prob_shift = evaluate_model(model, X_shift, y_shift)
    print("\nShifted distribution metrics:")
    print(f"Precision: {metrics_shift['precision']:.3f}")
    print(f"Recall: {metrics_shift['recall']:.3f}")
    print(f"Specificity: {metrics_shift['specificity']:.3f}")
    print(f"F1 Score: {metrics_shift['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_shift['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_shift['pr_auc']:.3f}")
    
    # Plot comparisons
    metrics_dict = {
        'Precision': {'original': metrics_orig['precision'], 'shifted': metrics_shift['precision']},
        'Recall': {'original': metrics_orig['recall'], 'shifted': metrics_shift['recall']},
        'Specificity': {'original': metrics_orig['specificity'], 'shifted': metrics_shift['specificity']},
        'F1': {'original': metrics_orig['f1_score'], 'shifted': metrics_shift['f1_score']},
        'ROC-AUC': {'original': metrics_orig['roc_auc'], 'shifted': metrics_shift['roc_auc']},
        'PR-AUC': {'original': metrics_orig['pr_auc'], 'shifted': metrics_shift['pr_auc']}
    }
    
    plot_metrics_comparison(metrics_dict, "Covariate Shift")
    plot_confusion_matrices(metrics_orig['confusion_matrix'], metrics_shift['confusion_matrix'], "Covariate Shift")
    plot_curves(y_test, y_pred_prob_orig, y_shift, y_pred_prob_shift, "Covariate Shift - ROC and PR Curves")
    
    return metrics_orig, metrics_shift

def simulate_concept_shift():
    """Simulate concept shift - P(Y|X) changes but P(X) remains the same"""
    print("\n=== Simulating Concept Shift ===")
    
    # Generate original data
    n_samples = 5000
    
    # Create features with a specific distribution
    X1 = np.random.normal(loc=0, scale=1, size=n_samples)
    X2 = np.random.normal(loc=0, scale=1, size=n_samples)
    X = np.column_stack([X1, X2])
    
    # Original concept: y = 1 if X1 + X2 > 0, else 0
    y = (X1 + X2 > 0).astype(int)
    
    # Train a model on the original distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on original distribution
    metrics_orig, y_pred_prob_orig = evaluate_model(model, X_test, y_test)
    print("Original distribution metrics:")
    print(f"Precision: {metrics_orig['precision']:.3f}")
    print(f"Recall: {metrics_orig['recall']:.3f}")
    print(f"Specificity: {metrics_orig['specificity']:.3f}")
    print(f"F1 Score: {metrics_orig['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_orig['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_orig['pr_auc']:.3f}")
    
    # Create shifted data with same feature distribution but different relationship (concept shift)
    # P(X) remains the same but P(Y|X) changes
    n_samples_shift = 5000
    
    # Same feature distribution
    X1_shift = np.random.normal(loc=0, scale=1, size=n_samples_shift)
    X2_shift = np.random.normal(loc=0, scale=1, size=n_samples_shift)
    X_shift = np.column_stack([X1_shift, X2_shift])
    
    # Different target function - changed from X1 + X2 > 0 to X1 - X2 > 0
    y_shift = (X1_shift - X2_shift > 0).astype(int)
    
    # Evaluate on shifted distribution
    metrics_shift, y_pred_prob_shift = evaluate_model(model, X_shift, y_shift)
    print("\nShifted distribution metrics:")
    print(f"Precision: {metrics_shift['precision']:.3f}")
    print(f"Recall: {metrics_shift['recall']:.3f}")
    print(f"Specificity: {metrics_shift['specificity']:.3f}")
    print(f"F1 Score: {metrics_shift['f1_score']:.3f}")
    print(f"ROC-AUC: {metrics_shift['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics_shift['pr_auc']:.3f}")
    
    # Plot comparisons
    metrics_dict = {
        'Precision': {'original': metrics_orig['precision'], 'shifted': metrics_shift['precision']},
        'Recall': {'original': metrics_orig['recall'], 'shifted': metrics_shift['recall']},
        'Specificity': {'original': metrics_orig['specificity'], 'shifted': metrics_shift['specificity']},
        'F1': {'original': metrics_orig['f1_score'], 'shifted': metrics_shift['f1_score']},
        'ROC-AUC': {'original': metrics_orig['roc_auc'], 'shifted': metrics_shift['roc_auc']},
        'PR-AUC': {'original': metrics_orig['pr_auc'], 'shifted': metrics_shift['pr_auc']}
    }
    
    plot_metrics_comparison(metrics_dict, "Concept Shift")
    plot_confusion_matrices(metrics_orig['confusion_matrix'], metrics_shift['confusion_matrix'], "Concept Shift")
    plot_curves(y_test, y_pred_prob_orig, y_shift, y_pred_prob_shift, "Concept Shift - ROC and PR Curves")
    
    return metrics_orig, metrics_shift

if __name__ == "__main__":
    print("Data Drift Simulation and Analysis")
    print("==================================")
    
    # Simulate all three types of drift
    label_orig, label_shift = simulate_label_shift()
    covariate_orig, covariate_shift = simulate_covariate_shift()
    concept_orig, concept_shift = simulate_concept_shift()
    
    # Summary of impacts
    print("\n=== Summary of Drift Impacts ===")
    
    print("\nLabel Shift Impact (changes in metric):")
    for metric in ['precision', 'recall', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']:
        change = label_shift[metric] - label_orig[metric]
        print(f"{metric}: {change:.3f} ({'+' if change >= 0 else ''}{change*100:.1f}%)")
    
    print("\nCovariate Shift Impact (changes in metric):")
    for metric in ['precision', 'recall', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']:
        change = covariate_shift[metric] - covariate_orig[metric]
        print(f"{metric}: {change:.3f} ({'+' if change >= 0 else ''}{change*100:.1f}%)")
    
    print("\nConcept Shift Impact (changes in metric):")
    for metric in ['precision', 'recall', 'specificity', 'f1_score', 'roc_auc', 'pr_auc']:
        change = concept_shift[metric] - concept_orig[metric]
        print(f"{metric}: {change:.3f} ({'+' if change >= 0 else ''}{change*100:.1f}%)")