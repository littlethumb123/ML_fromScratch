import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                            roc_auc_score, precision_recall_curve, auc,
                            roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from scipy.special import expit

class ErrorAnalysis:
    """
    Class to analyze and visualize error scenarios
    """
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculate classification metrics
        """
        metrics = {}
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        # Precision, recall
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)
        
        # False positive rate
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # ROC AUC and PR AUC (if probabilities provided)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = auc(recall, precision)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm, title='Confusion Matrix'):
        """
        Plot a confusion matrix
        """
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob, title='ROC Curve'):
        """
        Plot ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pr_curve(y_true, y_prob, title='Precision-Recall Curve'):
        """
        Plot precision-recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.axhline(y=np.mean(y_true), color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, title='Metrics Comparison'):
        """
        Plot metrics comparison between train, validation, and test sets
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'roc_auc', 'pr_auc']
        available_metrics = [m for m in metrics if m in train_metrics and m in val_metrics and m in test_metrics]
        
        x = np.arange(len(available_metrics))
        width = 0.25
        
        plt.figure(figsize=(10, 5))
        plt.bar(x - width, [train_metrics[m] for m in available_metrics], width, label='Train')
        plt.bar(x, [val_metrics[m] for m in available_metrics], width, label='Validation')
        plt.bar(x + width, [test_metrics[m] for m in available_metrics], width, label='Test')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(title)
        plt.xticks(x, available_metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(X, y, model, title='Decision Boundary'):
        """
        Plot decision boundary for a 2D dataset
        """
        if X.shape[1] != 2:
            print("Decision boundary plotting requires 2D data.")
            return
        
        plt.figure(figsize=(8, 6))
        
        # Define mesh grid
        h = 0.02  # step size in mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh grid
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
        plt.colorbar()
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_high_bias(results):
        """
        Analyze high bias scenario
        """
        print("\n===== HIGH BIAS SCENARIO ANALYSIS =====\n")
        
        # Calculate metrics
        train_metrics = ErrorAnalysis.calculate_metrics(
            results['y_train'], results['train_preds'], results['train_probs']
        )
        val_metrics = ErrorAnalysis.calculate_metrics(
            results['y_val'], results['val_preds'], results['val_probs']
        )
        test_metrics = ErrorAnalysis.calculate_metrics(
            results['y_test'], results['test_preds'], results['test_probs']
        )
        
        # Print metrics
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
        
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
        
        # Plot confusion matrices
        print("\nConfusion Matrices:")
        ErrorAnalysis.plot_confusion_matrix(train_metrics['confusion_matrix'], title='Training Confusion Matrix')
        ErrorAnalysis.plot_confusion_matrix(val_metrics['confusion_matrix'], title='Validation Confusion Matrix')
        ErrorAnalysis.plot_confusion_matrix(test_metrics['confusion_matrix'], title='Test Confusion Matrix')
        
        # Plot ROC curves
        print("\nROC Curves:")
        ErrorAnalysis.plot_roc_curve(results['y_train'], results['train_probs'], title='Training ROC Curve')
        ErrorAnalysis.plot_roc_curve(results['y_val'], results['val_probs'], title='Validation ROC Curve')
        ErrorAnalysis.plot_roc_curve(results['y_test'], results['test_probs'], title='Test ROC Curve')
        
        # Plot PR curves
        print("\nPrecision-Recall Curves:")
        ErrorAnalysis.plot_pr_curve(results['y_train'], results['train_probs'], title='Training PR Curve')
        ErrorAnalysis.plot_pr_curve(results['y_val'], results['val_probs'], title='Validation PR Curve')
        ErrorAnalysis.plot_pr_curve(results['y_test'], results['test_probs'], title='Test PR Curve')
        
        # Plot metrics comparison
        print("\nMetrics Comparison:")
        ErrorAnalysis.plot_metrics_comparison(train_metrics, val_metrics, test_metrics, 
                                            title='High Bias Scenario: Metrics Comparison')
        
        # Plot decision boundary (if 2D)
        if results['X_train'].shape[1] == 2:
            print("\nDecision Boundary:")
            ErrorAnalysis.plot_decision_boundary(results['X_train'], results['y_train'], 
                                               results['model'], title='Training Decision Boundary')
            ErrorAnalysis.plot_decision_boundary(results['X_test'], results['y_test'], 
                                               results['model'], title='Test Decision Boundary')
        
        print("\nHigh Bias Analysis:")
        print("- We observe that both training and test performance are poor.")
        print("- This indicates the model is underfitting the data.")
        print("- The model's capacity is too low to capture the complexity of the data pattern.")
        print("- This is evidenced by similar (poor) performance across train/val/test sets.")
        print("- The ROC curves are likely close to the diagonal, showing low discriminative power.")
        print("- Suggested solutions include using a more complex model, adding features, or reducing regularization.")
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
    # Similar analyze_* functions for other scenarios would follow
    # Each would extract relevant metrics and visualizations for the particular error type
    # For brevity, I'm showing just the high_bias analysis function as an example
    
    def analyze_scenario(self, results):
        """
        Analyze a scenario based on its type
        """
        scenario = results['scenario']
        
        if scenario == 'high_bias':
            return self.analyze_high_bias(results)
        elif scenario == 'high_variance':
            return self.analyze_high_variance(results)
        elif scenario == 'class_imbalance':
            return self.analyze_class_imbalance(results)
        elif scenario == 'noisy_labels':
            return self.analyze_noisy_labels(results)
        elif scenario == "poor_optimization":
            return self.analyze_poor_optimization(results)
        elif scenario == "":
