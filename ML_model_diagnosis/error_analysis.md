# ML model diagnosis
## Error analysis
### Model Capacity issues:
1. Overfitting (high variance):
    - The model performs well on the training data but poorly on the test data.
    - The model is too complex and captures noise in the training data.
2. Underfitting (high bias):
    - The model performs poorly on both the training and test data.
    - The model is too simple and fails to capture the underlying patterns in the data.
### Data quality:
1. Data imbalance:
    - The model is trained on imbalanced data, leading to biased predictions.
    - Diagnosis:
        - Accuracy or auc_roc are high but pr-auc is low.
        - 
2. Noisy labels:
    - Inconsistent or incorrect ground truth labels mislead the model during training.

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
2. Concept drift:
    - When P(Y|X) changes but P(X) remains the same. This refers to the first decomposition of the joint distribution. This is also known as posterior shift. For example, in house price prediction model, house's area is an input parameter, suppose before covid-19, the house price is 200K, but after covid the house price has come down to 150K. So even though, the house features remain the same, the conditional distribution of the price of a house given its features has changed.
3. Label shift:
    - When P(Y) changes but P(X|Y) remains the same. This refers to the second decomposition of the joint distribution. This is also known as prior shift, prior probability shift or target shift. Let's build up on earlier example itself, suppose government starts providing direct cash transfer to all people, this may reduce the probability of defaulting P(Y) for everyone, yet the conditional probability P(X|Y), meaning the probability that a person has a lower education, given it defaulted, hasn't changed.

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

### Flawed pipeline:
1. Data leakage:
    - Ensure that the training and test data are properly separated and that no information from the test data is used during training.
    - Use cross-validation to detect potential leakage by evaluating the model on different subsets of the data.
    - Regularly audit the data pipeline to ensure that no unintended leakage occurs during preprocessing or feature engineering.
    - Remove any that won’t be available at test time. For example, if predicting cancer outcome, ensure you don’t include a feature like “treatment given” which is decided after the diagnosis (leakage of future information).


## Evaluation metrics and thresholds:
### Threshold based metrics in classification:
1. Confusion matrix:


### Define threshold: