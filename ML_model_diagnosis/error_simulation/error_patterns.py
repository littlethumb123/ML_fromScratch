import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           roc_auc_score, precision_recall_curve, auc,
                           roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
from scipy.special import expit
from data_generator import DatasetGenerator

class ErrorPattern:
    """
    Class to simulate various error scenarios in machine learning
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data_generator = DatasetGenerator(random_state=random_state)
        
    def simulate_high_bias(self):
        """
        Simulate high bias (underfitting) by:
        1. Using a complex dataset with non-linear relationships
        2. Training a simple linear model on it
        """
        print("Simulating High Bias (Underfitting) Scenario...")
        
        # 1. Generate complex, non-linear data
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=2,  # Using fewer features makes visualization easier
            class_sep=0.5,  # Low separation makes it more complex
        )
        
        # Make the data more non-linear by applying a transformation
        X[:, 0] = np.sin(X[:, 0] * 3)
        X[:, 1] = np.cos(X[:, 1] * 2)
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Train a simple linear model (high bias)
        model = LogisticRegression(C=1.0, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 4. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # 5. Calculate metrics
        results = {
            "scenario": "high_bias",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results
    
    def simulate_high_variance(self):
        """
        Simulate high variance (overfitting) by:
        1. Using a small dataset
        2. Training a complex model with minimal regularization
        """
        print("Simulating High Variance (Overfitting) Scenario...")
        
        # 1. Generate a small dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,  # Small sample
            n_features=300,  # Many features (increases variance)
            class_sep=1.0,  # Reasonable separation
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Train a complex model with minimal regularization (high variance)
        model = SVC(C=100.0, kernel='rbf', gamma='auto', probability=True, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 4. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # 5. Calculate metrics
        results = {
            "scenario": "high_variance",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results
    
    def simulate_imbalanced_dataset(self):
        """
        Simulate issues with imbalanced data
        """
        print("Simulating Imbalanced Dataset Scenario...")
        
        # 1. Generate an imbalanced dataset (90% class 0, 10% class 1)
        X, y = self.data_generator.generate_base_dataset(
            n_samples=10000,
            n_features=10,
            class_sep=1.0,
            class_weight={0: 0.95, 1: 0.05}
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test, random_state=self.random_state
        )
        
        # 3. Train model without class weights (will be biased toward majority class)
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 4. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # 5. Calculate metrics
        results = {
            "scenario": "imbalanced_dataset",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results

    def simulate_covariate_drift(self):
        """
        Simulate covariate drift by:
        1. Training on one distribution of X
        2. Testing on a different distribution of X (with same P(Y|X))
        """
        print("Simulating Covariate Drift Scenario...")
        
        # 1. Generate base dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Apply covariate shift to validation and test sets
        X_val_shifted = self.data_generator.apply_covariate_shift(X_val, shift_scale=1.5)
        X_test_shifted = self.data_generator.apply_covariate_shift(X_test, shift_scale=1.5)
        
        # 4. Train model on original distribution
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        # Predictions on original validation data
        val_preds_orig = model.predict(X_val)
        val_probs_orig = model.predict_proba(X_val)[:, 1]
        
        # Predictions on shifted validation data (with drift)
        val_preds = model.predict(X_val_shifted)
        val_probs = model.predict_proba(X_val_shifted)[:, 1]
        
        # Predictions on original test data
        test_preds_orig = model.predict(X_test)
        test_probs_orig = model.predict_proba(X_test)[:, 1]
        
        # Predictions on shifted test data (with drift)
        test_preds = model.predict(X_test_shifted)
        test_probs = model.predict_proba(X_test_shifted)[:, 1]
        
        # 6. Calculate metrics
        results = {
            "scenario": "covariate_drift",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_val_shifted": X_val_shifted,
            "X_test": X_test, "y_test": y_test,
            "X_test_shifted": X_test_shifted,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds_orig": val_preds_orig, "val_probs_orig": val_probs_orig,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds_orig": test_preds_orig, "test_probs_orig": test_probs_orig,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results

    def simulate_concept_drift(self):
        """
        Simulate concept drift by:
        1. Training on one P(Y|X) relationship
        2. Testing on a different P(Y|X) relationship
        """
        print("Simulating Concept Drift Scenario...")
        
        # 1. Generate base dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Apply concept shift to validation and test sets (change P(Y|X))
        _, y_val_shifted = self.data_generator.apply_concept_shift(X_val, y_val, shift_scale=0.5)
        _, y_test_shifted = self.data_generator.apply_concept_shift(X_test, y_test, shift_scale=0.5)
        
        # 4. Train model on original concept
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        # Predictions on original validation data (no drift)
        val_preds_orig = model.predict(X_val)
        val_probs_orig = model.predict_proba(X_val)[:, 1]
        
        # Evaluate against shifted concept (with drift)
        val_preds = val_preds_orig  # Same predictions but comparing to shifted labels
        val_probs = val_probs_orig  # Same probabilities but comparing to shifted labels
        
        # Predictions on original test data (no drift)
        test_preds_orig = model.predict(X_test)
        test_probs_orig = model.predict_proba(X_test)[:, 1]
        
        # Evaluate against shifted concept (with drift)
        test_preds = test_preds_orig  # Same predictions but comparing to shifted labels
        test_probs = test_probs_orig  # Same probabilities but comparing to shifted labels
        
        # 6. Calculate metrics
        results = {
            "scenario": "concept_drift",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "y_val_shifted": y_val_shifted,
            "X_test": X_test, "y_test": y_test,
            "y_test_shifted": y_test_shifted,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds_orig": val_preds_orig, "val_probs_orig": val_probs_orig,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds_orig": test_preds_orig, "test_probs_orig": test_probs_orig,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model,
            "original_y_val": y_val,
            "original_y_test": y_test
        }
        
        return results

    def simulate_label_drift(self):
        """
        Simulate label drift by:
        1. Training on one distribution of Y
        2. Testing on a different distribution of Y (with same P(X|Y))
        """
        print("Simulating Label Drift Scenario...")
        
        # 1. Generate base dataset with balanced classes
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Apply label drift to test set by generating more of class 1
        X_test_shifted, y_test_shifted = self.data_generator.apply_label_shift(
            X_test, y_test, target_ratio=0.7
        )
        
        # Also apply to validation set
        X_val_shifted, y_val_shifted = self.data_generator.apply_label_shift(
            X_val, y_val, target_ratio=0.7
        )
        
        # 4. Train model on original label distribution
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        # Predictions on original validation data
        val_preds_orig = model.predict(X_val)
        val_probs_orig = model.predict_proba(X_val)[:, 1]
        
        # Predictions on shifted validation data (with drift)
        val_preds = model.predict(X_val_shifted)
        val_probs = model.predict_proba(X_val_shifted)[:, 1]
        
        # Predictions on original test data
        test_preds_orig = model.predict(X_test)
        test_probs_orig = model.predict_proba(X_test)[:, 1]
        
        # Predictions on shifted test data (with drift)
        test_preds = model.predict(X_test_shifted)
        test_probs = model.predict_proba(X_test_shifted)[:, 1]
        
        # 6. Calculate metrics
        results = {
            "scenario": "label_drift",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_val_shifted": X_val_shifted, "y_val_shifted": y_val_shifted,
            "X_test": X_test, "y_test": y_test,
            "X_test_shifted": X_test_shifted, "y_test_shifted": y_test_shifted,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds_orig": val_preds_orig, "val_probs_orig": val_probs_orig,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds_orig": test_preds_orig, "test_probs_orig": test_probs_orig,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results

    def simulate_mislabeled_training_data(self):
        """
        Simulate issues with mislabeled training data
        """
        print("Simulating Mislabeled Training Data Scenario...")
        
        # 1. Generate base dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Introduce label noise in the training data
        y_train_noisy = self.data_generator.flip_labels(y_train, flip_ratio=0.2)
        
        # 4. Train model on noisy labels
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train_noisy)
        
        # 5. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # 6. Calculate metrics (against true labels to show the effect)
        results = {
            "scenario": "mislabeled_training",
            "X_train": X_train, "y_train": y_train,
            "y_train_noisy": y_train_noisy,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results

    def simulate_mislabeled_test_data(self):
        """
        Simulate issues with mislabeled test data
        """
        print("Simulating Mislabeled Test Data Scenario...")
        
        # 1. Generate base dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Introduce label noise in the test data
        y_test_noisy = self.data_generator.flip_labels(y_test, flip_ratio=0.2)
        y_val_noisy = self.data_generator.flip_labels(y_val, flip_ratio=0.2)
        
        # 4. Train model on clean training data
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        
        # 6. Calculate metrics (against noisy labels for test data)
        results = {
            "scenario": "mislabeled_test",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "y_val_noisy": y_val_noisy,
            "X_test": X_test, "y_test": y_test,
            "y_test_noisy": y_test_noisy,
            "train_preds": train_preds, "train_probs": train_probs,
            "val_preds": val_preds, "val_probs": val_probs,
            "test_preds": test_preds, "test_probs": test_probs,
            "model": model
        }
        
        return results

    def simulate_poor_optimization(self):
        """
        Simulate issues with optimization approach
        """
        print("Simulating Poor Optimization Scenario...")
        
        # 1. Generate a dataset with highly separable clusters (easy for SVM)
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=2,
            class_sep=2.0,
            random_state=self.random_state
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=self.random_state
        )
        
        # 3. Train logistic regression (suboptimal for this data)
        lr_model = LogisticRegression(random_state=self.random_state)
        lr_model.fit(X_train, y_train)
        
        # 4. Train SVM (better suited for this data)
        svm_model = SVC(kernel='rbf', probability=True, random_state=self.random_state)
        svm_model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        lr_train_preds = lr_model.predict(X_train)
        lr_train_probs = lr_model.predict_proba(X_train)[:, 1]
        
        lr_val_preds = lr_model.predict(X_val)
        lr_val_probs = lr_model.predict_proba(X_val)[:, 1]
        
        lr_test_preds = lr_model.predict(X_test)
        lr_test_probs = lr_model.predict_proba(X_test)[:, 1]
        
        svm_train_preds = svm_model.predict(X_train)
        svm_train_probs = svm_model.predict_proba(X_train)[:, 1]
        
        svm_val_preds = svm_model.predict(X_val)
        svm_val_probs = svm_model.predict_proba(X_val)[:, 1]
        
        svm_test_preds = svm_model.predict(X_test)
        svm_test_probs = svm_model.predict_proba(X_test)[:, 1]
        
        # 6. Calculate metrics
        results = {
            "scenario": "poor_optimization",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "lr_train_preds": lr_train_preds, "lr_train_probs": lr_train_probs,
            "lr_val_preds": lr_val_preds, "lr_val_probs": lr_val_probs,
            "lr_test_preds": lr_test_preds, "lr_test_probs": lr_test_probs,
            "svm_train_preds": svm_train_preds, "svm_train_probs": svm_train_probs,
            "svm_val_preds": svm_val_preds, "svm_val_probs": svm_val_probs,
            "svm_test_preds": svm_test_preds, "svm_test_probs": svm_test_probs,
            "lr_model": lr_model,
            "svm_model": svm_model
        }
        
        return results

    def simulate_inappropriate_cost_function(self):
        """
        Simulate issues with using an inappropriate cost function
        """
        print("Simulating Inappropriate Cost Function Scenario...")
        
        # 1. Generate a highly imbalanced dataset
        X, y = self.data_generator.generate_base_dataset(
            n_samples=1000,
            n_features=10,
            class_sep=1.0,
            class_weight={0: 0.95, 1: 0.05}  # Very imbalanced
        )
        
        # 2. Split into train/val/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, stratify=y_test, random_state=self.random_state
        )
        
        # 3. Train model with default cost function (inappropriate for imbalance)
        default_model = LogisticRegression(random_state=self.random_state)
        default_model.fit(X_train, y_train)
        
        # 4. Train model with balanced class weights (appropriate cost function)
        balanced_model = LogisticRegression(class_weight='balanced', random_state=self.random_state)
        balanced_model.fit(X_train, y_train)
        
        # 5. Get predictions and probabilities
        default_train_preds = default_model.predict(X_train)
        default_train_probs = default_model.predict_proba(X_train)[:, 1]
        
        default_val_preds = default_model.predict(X_val)
        default_val_probs = default_model.predict_proba(X_val)[:, 1]
        
        default_test_preds = default_model.predict(X_test)
        default_test_probs = default_model.predict_proba(X_test)[:, 1]
        
        balanced_train_preds = balanced_model.predict(X_train)
        balanced_train_probs = balanced_model.predict_proba(X_train)[:, 1]
        
        balanced_val_preds = balanced_model.predict(X_val)
        balanced_val_probs = balanced_model.predict_proba(X_val)[:, 1]
        
        balanced_test_preds = balanced_model.predict(X_test)
        balanced_test_probs = balanced_model.predict_proba(X_test)[:, 1]
        
        # 6. Calculate metrics
        results = {
            "scenario": "inappropriate_cost_function",
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "default_train_preds": default_train_preds, "default_train_probs": default_train_probs,
            "default_val_preds": default_val_preds, "default_val_probs": default_val_probs,
            "default_test_preds": default_test_preds, "default_test_probs": default_test_probs,
            "balanced_train_preds": balanced_train_preds, "balanced_train_probs": balanced_train_probs,
            "balanced_val_preds": balanced_val_preds, "balanced_val_probs": balanced_val_probs,
            "balanced_test_preds": balanced_test_preds, "balanced_test_probs": balanced_test_probs,
            "default_model": default_model,
            "balanced_model": balanced_model
        }
        
        return results