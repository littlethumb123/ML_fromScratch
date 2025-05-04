## Simulation Scenarios for ML Model Diagnosis

### Senarios:
* High bias: Model is too simple to capture the underlying data patterns.
* High variance: Model is too complex and overfits the training data.
* Data Imbalanced: Class distribution is skewed, leading to poor performance on minority classes.
* Covariate Drift: The distribution of features changes over time, affecting model performance.
* Label Drift: The distribution of labels changes over time, affecting model performance.
* Concept Drift: The underlying relationship between features and labels changes over time.
* Mislabelled Samples: Incorrect labels in the training data lead to poor model performance.
* Optimization Not Fit: The optimization algorithm is not suitable for the problem at hand.
* Objective function not fit: The loss function used is not appropriate for the task (e.g., using regression loss for classification).
### Dataset:
We’ll use sklearn.datasets.make_classification for synthetic data, and manipulate it for the following parameters
* Sample size: small/large
* Feature size: small/large
* Label noise: flip_y
* Class separation (easy or hard to classify): class_sep
* Imbalance: weights
* Data shift: manual manipulation
* SVM vs Logistic Regression: non-linear separability
#### Manipulation:
Let’s define each scenario and how to simulate it:
| Scenario | How to Simulate |
|----------|-----------------|
| High bias | Small model, low features, high regularization, low class_sep |
| High variance | Complex model, high features, low regularization, small dataset |
| Imbalanced | weights=[0.9, 0.1] |
| Covariate drift | Shift test set features |
| Label drift | Change label distribution in test set |
| Concept drift | Change label rules in test set |
| Mislabelled samples | Increase flip_y or flip labels manually |
| Optimization not fit | Use wrong optimizer (e.g., SGD with bad learning rate) |
| Wrong objective | Use regression loss for classification |