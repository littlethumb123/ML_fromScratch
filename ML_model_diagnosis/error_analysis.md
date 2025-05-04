# ML model diagnosis
## Error analysis
### Model Capacity issues:
1. Overfitting (high variance):
    -  Overfitting occurs when model performs well on training data but poorly on dev/test data.
    All metrics (accuracy, F1, etc.) stay low. The ROC curve will lie near the diagonal (AUC≈0.5) and PR curve near the bottom-left, reflecting no discriminative power.
    - The model is too complex and captures noise in the training data.
2. Underfitting (high bias):
    - The model performs poorly on both the training and test data. In a confusion matrix, an overfit model may show many TPs on training data but produce many more FPs and FNs on test data (so test recall and precision both degrade).
    - The model is too simple and fails to capture the underlying patterns in the data.

### Optimization issues:
1. Poor convergence:
    - The model fails to converge to a good solution during training, leading to suboptimal performance. This normally happens when the learning rate is too high or too low, or when the objective function is not suitable for the problem.
    - Diagnosis:
        - The loss function does not decrease over time or oscillates wildly.
        - The model's performance metrics (accuracy, F1, etc.) do not improve significantly during training.
2. Local minima or saddle points:
    - The model gets stuck in a local minimum or saddle point during training, preventing it from finding the global optimum.
    - Diagnosis:
        - The loss function plateaus or oscillates around a certain value without significant improvement.
        - The model's performance metrics do not improve significantly over time.

### Data quality:
1. Data imbalance:
    - The model is trained on imbalanced data, leading to biased predictions.
    - Diagnosis:
        - Accuracy, specificity, or auc_roc are high but pr-auc, precision and recall is low.
        - 
2. Noisy labels:
    - Inconsistent or incorrect ground truth labels mislead the model during training or testing
    - Diagnosis:
        - When incorrect labels in training set: This usually lowers all performance metrics on a clean test set. For instance, mislabeled positives (i.e., positive cases labeled as negative) in training will cause the model to under-recognize real positives (lower recall) and possibly misidentify negatives (lower precision). In short, label noise in training degrades model quality and yields lower precision/recall/F1 on test.
        - When incorrect labels in test set: A flipped positive in the test set will make a true positive prediction look like a false positive, which will lower precision. A flipped positive classified correctly by the model would count as an FN, lowering recall. In short, label noise in test degrades model quality and yields lower precision/recall/F1 on test.

3. Missing or corrupted data:
    -  Incomplete information leads to systematic prediction errors.

### Flawed pipeline 
1. Data leakage:
    - The model performs nearly perfectly on both the training and test data.
    - Diagnosis:
        - Leakage often manifests as abnormally high validation performance that doesn’t hold up in a truly separated test
2. Data labeling errors:
    - The model is trained or test on incorrect labels, leading to poor performance or bias
    - Diagnosis:

### Data drift:
1. Covariate shift: 
    - When P(X) changes but P(Y|X) remains the same. This refers to the first decomposition of the joint distribution, i.e. distribution of input changes, but the conditional probability of an output given an input remains the same. For example, you are trying to predict whether a person would default or not. You have a variable education in your model, and let's say people with lower education typically default more often. In your training dataset, suppose you have many examples of people with higher education, but in your inference (testing) dataset you have many examples of lower education.
    - Diagnosis:
        - Under covariate shift the model’s true performance (ability to predict Y from X) could remain the same in principle, but realized performance often drops because the model sees unfamiliar patterns. In practice, studies note that “almost all real-world ML models gradually degrade in performance due to covariate shift”
2. Concept drift:
    - When P(Y|X) changes but P(X) remains the same. This refers to the first decomposition of the joint distribution. This is also known as posterior shift. For example, in house price prediction model, house's area is an input parameter, suppose before covid-19, the house price is 200K, but after covid the house price has come down to 150K. So even though, the house features remain the same, the conditional distribution of the price of a house given its features has changed.
    - Diagnosis:
        - Because the relationship between features and outcome has shifted, all metrics can suddenly degrade. A model that learned P(Y|X) from training will no longer match reality: the confusion matrix on new data will reflect many more errors (new FNs or FPs depending on how the concept changed). Quantitatively, one would see drops in accuracy, recall, precision, and AUC after the shift. (Concept shift is essentially a change in the decision boundary – accuracy and ROC-AUC drop unless the model is updated.)
3. Label shift:
    - When P(Y) changes but P(X|Y) remains the same. This refers to the second decomposition of the joint distribution. This is also known as prior shift, prior probability shift or target shift. Let's build up on earlier example itself, suppose government starts providing direct cash transfer to all people, this may reduce the probability of defaulting P(Y) for everyone, yet the conditional probability P(X|Y), meaning the probability that a person has a lower education, given it defaulted, hasn't changed.
    - Diagnosis:
        - Under label shift, sensitivity (recall) and specificity for a given threshold usually stay the same (because P(X|Y) is unchanged), but PPV/precision changes
        - With higher prevalence in the test set, the same model yields a higher TP count but also more FNs

## Strategy
### Model Capacity issues:
1. Overfitting:
    - Apply regularization (e.g. L2 or L1 penalties, dropout in neural nets) to constrain the model’s complexity and prevent memorizing noise.
    - Use cross-validation to ensure the model isn’t overly tuned to one particular train/val split. This also helps catch issues like data leakage.
    - Gather more training data or augment existing data if possible. More data helps the model generalize and smooth out idiosyncrasies.
    - Consider a simpler model or fewer features if appropriate. A less complex model has lower variance. For instance, if using a very deep neural network on a small dataset, try a smaller network or early stopping.
    - Use ensembling techniques (combine multiple models) which often generalize better than a single model and reduce variance by averaging out quirks.
2. Underfitting:
    - Use a more complex model or architecture. For example, try a deeper network, or add polynomial features for a linear model, etc. The current model is too restricted.
    - Perform feature engineering to expose more predictive signals. In a medical context, this could mean incorporating additional patient features (e.g., family history, genetic markers) or creating composite features (like BMI from height and weight) that have a more linear relation with the outcome.
    - Decrease regularization if it was too strong (a very large lambda can overly simplify the model).
    - Ensure the training process is effective: e.g., train longer or with a better optimizer if the model simply hasn’t converged yet.
    - If feasible, obtain more informative data. Sometimes underfitting means the phenomenon is just hard to predict with available features – adding a new data modality (like adding imaging data to a model that currently only uses lab test results) can improve the fit.
### Data quality:
1. Data imbalance:
    - Use stratified sampling to ensure balanced representation of classes in training and test sets.
    - Apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.
    - Use cost-sensitive learning or class weights to penalize misclassifications of the minority class more heavily.
    - Consider using ensemble methods like Random Forest or Gradient Boosting, which can handle imbalanced data better than single models.
2. Noisy labels:
    - Use robust loss functions that are less sensitive to label noise (e.g., Huber loss instead of MSE).
    - Implement data cleaning techniques to identify and correct mislabeled instances.
    - Use ensemble methods or model averaging to reduce the impact of noisy labels.
    - Consider using semi-supervised learning or active learning to improve label quality by leveraging unlabeled data or querying for labels on uncertain instances.
### Flawed pipeline:
1. Data leakage:
    - Ensure that the training and test data are properly separated and that no information from the test data is used during training.
    - Use cross-validation to detect potential leakage by evaluating the model on different subsets of the data.
    - Regularly audit the data pipeline to ensure that no unintended leakage occurs during preprocessing or feature engineering.
    - Remove any that won’t be available at test time. For example, if predicting cancer outcome, ensure you don’t include a feature like “treatment given” which is decided after the diagnosis (leakage of future information).


## Evaluation metrics and thresholds:
### Threshold based metrics in classification:
1. Confusion matrix:
|--------Threshold adjustment-----|---------Precision-------|---------Recall---------|
Threshold Adjustment	Precision	Recall	Specificity	FPR	ROC-AUC	PR-AUC
Lower Threshold	↓	↑	↓	↑	–	–
Raise Threshold	↑	↓	↑	↓	–	–
2. Accuracy will vary non-monotonically and is maximized at an optimal threshold (which may not be 0.5 if classes are imbalanced). Importantly, metrics like ROC-AUC and PR-AUC measure performance across all thresholds and thus are invariant to any single threshold choice (they summarize the full curve).
### Define threshold: