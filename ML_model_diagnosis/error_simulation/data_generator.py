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

class DatasetGenerator:
    """
    Flexible dataset generator to simulate various ML error scenarios
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def generate_base_dataset(self, n_samples=10000, n_features=20, class_sep=1.0, 
                             class_weight=None, random_state=None):
        """
        Generate a base classification dataset with controllable parameters
        
        Parameters:
        -----------
        n_samples : int
            Number of samples (can be small or large)
        n_features : int
            Number of features (small or large feature set)
        class_sep : float
            Class separation (higher values make classification easier)
        class_weight : dict or None
            Class weights for imbalanced data, e.g., {0:0.9, 1:0.1}
        """
        if random_state is None:
            random_state = self.random_state
            
        # Calculate class weights if provided
        if class_weight:
            weights = [class_weight[0], class_weight[1]]
            ratio = class_weight[1] / sum(class_weight.values())  # Proportion of class 1
        else:
            weights = None
            ratio = 0.5  # Balanced
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(3, n_features // 2),  # Ensure some informative features
            n_redundant=max(2, n_features // 5),    # Some redundant features
            n_repeated=0,
            n_classes=2,
            class_sep=class_sep,
            weights=[1-ratio, ratio] if weights else None,
            random_state=random_state
        )
        
        return X, y
    
    def flip_labels(self, y, flip_ratio=0.1, random_state=None):
        """
        Introduce label noise by flipping some labels
        
        Parameters:
        -----------
        y : array-like
            Original labels
        flip_ratio : float
            Proportion of labels to flip
        """
        if random_state is None:
            random_state = self.random_state
            
        rng = np.random.RandomState(random_state)
        y_noisy = y.copy()
        
        # Randomly select indices to flip
        flip_indices = rng.choice(
            np.arange(len(y)), 
            size=int(flip_ratio * len(y)), 
            replace=False
        )
        
        # Flip the selected labels
        y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
        
        return y_noisy
    
    def apply_covariate_shift(self, X, shift_scale=1.0):
        """
        Apply a covariate shift to features (P(X) changes)
        """
        X_shifted = X.copy()
        
        # Apply a non-linear transformation to create a shift
        X_shifted[:, :2] = X_shifted[:, :2] + shift_scale  # Shift first two features
        
        return X_shifted
    
    def apply_concept_shift(self, X, y, shift_scale=1.0, random_state=None):
        """
        Apply a concept shift (P(Y|X) changes)
        """
        if random_state is None:
            random_state = self.random_state
            
        rng = np.random.RandomState(random_state)
        X_shifted = X.copy()
        y_shifted = y.copy()
        
        # Create a concept shift by changing decision boundary
        # We'll do this by selecting a subset of points near the decision boundary
        # and flipping their labels
        
        # Get a model to find points near decision boundary
        model = LogisticRegression(random_state=random_state)
        model.fit(X, y)
        
        # Get probabilities
        probs = model.predict_proba(X)[:, 1]
        
        # Find points close to decision boundary
        boundary_points = np.where((probs > 0.4) & (probs < 0.6))[0]
        
        # Randomly select a portion of these points based on shift scale
        num_to_flip = int(len(boundary_points) * shift_scale)
        if num_to_flip > 0:
            to_flip = rng.choice(boundary_points, size=num_to_flip, replace=False)
            y_shifted[to_flip] = 1 - y[to_flip]  # Flip labels
            
        return X_shifted, y_shifted
    
    def apply_label_shift(self, X, y, target_ratio=0.7, random_state=None):
        """
        Apply a label shift (P(Y) changes but P(X|Y) remains the same)
        """
        if random_state is None:
            random_state = self.random_state
            
        # Calculate current class ratio
        current_ratio = np.mean(y)
        
        # If we need more positive examples
        if target_ratio > current_ratio:
            # Oversample positive class
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            # Calculate how many positive samples we need
            n_total = len(y)
            n_pos_needed = int(target_ratio * n_total)
            n_pos_current = len(pos_indices)
            n_to_add = n_pos_needed - n_pos_current
            
            rng = np.random.RandomState(random_state)
            
            if n_to_add > 0:
                # Sample with replacement from positive class
                sampled_indices = rng.choice(pos_indices, size=n_to_add, replace=True)
                
                # Combine with original data
                X_new = np.vstack([X, X[sampled_indices]])
                y_new = np.hstack([y, y[sampled_indices]])
                
                return X_new, y_new
            
        # If we need more negative examples
        elif target_ratio < current_ratio:
            # Oversample negative class
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            # Calculate how many negative samples we need
            n_total = len(y)
            n_neg_needed = int((1-target_ratio) * n_total)
            n_neg_current = len(neg_indices)
            n_to_add = n_neg_needed - n_neg_current
            
            rng = np.random.RandomState(random_state)
            
            if n_to_add > 0:
                # Sample with replacement from negative class
                sampled_indices = rng.choice(neg_indices, size=n_to_add, replace=True)
                
                # Combine with original data
                X_new = np.vstack([X, X[sampled_indices]])
                y_new = np.hstack([y, y[sampled_indices]])
                
                return X_new, y_new
        
        # If no change needed
        return X, y

if __name__ == '__main__':
    # Example usage
    generator = DatasetGenerator()
    
    # Generate a base dataset
    X, y = generator.generate_base_dataset(n_samples=1000, n_features=20, class_sep=1.0)
    
    # Introduce label noise
    y_noisy = generator.flip_labels(y, flip_ratio=0.1)
    
    # Apply covariate shift
    X_shifted = generator.apply_covariate_shift(X, shift_scale=1.0)
    
    # Apply concept shift
    X_concept_shifted, y_concept_shifted = generator.apply_concept_shift(X, y, shift_scale=0.2)
    
    # Apply label shift
    X_label_shifted, y_label_shifted = generator.apply_label_shift(X, y, target_ratio=0.7)