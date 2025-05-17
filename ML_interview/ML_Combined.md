# Machine Learning Interview Preparation

## 1. Generic Knowledge

### 1.1 Overfitting and Underfitting

*   **Overfitting:** A model learns the training data too well, including noise and outliers, leading to poor generalization on unseen data. High variance, low bias.
A model that overfits not only learns underlying patterns but also memorizes random noise in the training data. Methods to avoid this issue include collecting more data or performing data augmentation, adding regularization (L1, L2, or other constraints), applying early stopping, using cross-validation to better estimate generalization error, and reducing model complexity or pruning (in decision trees).

*   **Underfitting:** A model is too simple to capture the underlying patterns in the data. High bias, low variance.
A model that underfits is too simple to capture the data’s complexity and therefore exhibits high bias. Common remedies include increasing model complexity by adding more parameters or switching to a more sophisticated model, identifying and incorporating more relevant features, lowering regularization strength, and examining data quality and distribution for possible feature engineering or transformations.

### 1.2 Bias-Variance Tradeoff

*   **Bias:** The difference between the expected prediction of our model and the true value. High bias implies the model is making strong assumptions about the data.
    * Inductive bias: The assumptions made by the learning algorithm to generalize from the training data to unseen data. 
*   **Variance:** The variability in the model's predictions for different training datasets. High variance implies the model is very sensitive to the training data.
*   **Decomposition:**
    $$
    \begin{aligned}
    E[(y-\hat{y})^2] &= Bias^2 + Var + \text{Irreducible Error} \\
    Bias^2 &= (y - E[\hat{y}])^2 \\
    Var &= E[(E[\hat{y}] - \hat{y})^2]
    \end{aligned}
    $$
    Where:
    *   $y$ is the true value.
    *   $\hat{y}$ is the predicted value.
*   **Tradeoff:** Reducing bias often increases variance, and vice versa. The goal is to find a balance that minimizes the overall error.
A model with high bias underfits by ignoring relevant details in the data, while an overly flexible model has high variance and tends to overfit by memorizing noise in the training set. Common solutions include adjusting model complexity, gathering more data or applying data augmentation, and using cross-validation to select models more robustly.

### 1.3 Regularization

*   **Purpose:** To prevent overfitting by adding a penalty term to the loss function, discouraging complex models.
#### L1 Regularization (Lasso):
* Adds the sum of the absolute values of the coefficients to the loss function. In lasso, we add a **absolute** magnitude of the coefficient to the loss function as a penalty term with a scale $\lambda$; as the penalty $\lambda$ increases, the $\beta$ becomes 0. Thus the lasso term can be used for feature selection. 
    $$
    J(\theta) = \text{Loss} + \lambda \sum_{i=1}^n |\theta_i|
    $$
    *   Encourages sparsity (feature selection) by driving some coefficients to zero.
*   **L2 Regularization (Ridge):** Adds the sum of the squares of the coefficients to the loss function.
    $$
    J(\theta) = \text{Loss} + \lambda \sum_{i=1}^n \theta_i^2
    $$
    *   Reduces the magnitude of coefficients but rarely sets them to zero. Helps with multicollinearity.
*   **Why L1 leads to sparsity:** L1 regularization has corners at zero, while L2 regularization is smooth and continuously differentiable. The L1 norm penalty creates diamond-shaped constraint regions in the coefficient space, centered around the origin. As a result, the optimization process may drive some coefficients exactly to zero, leading to a sparse solution.
L1 (Lasso) encourages sparse solutions by driving some coefficients to zero, while L2 (Ridge) shrinks coefficients toward zero without necessarily eliminating them. Elastic Net combines L1 and L2 to balance both sparseness and coefficient shrinkage. Regularization is crucial to prevent overfitting in models such as linear/logistic regression or neural networks, and the strength of the penalty term (lambda) is usually set by cross-validation.

### 1.4 Loss Functions

*   **Mean Squared Error (MSE):**
    $$
    MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
    $$
    *   Used in regression problems. Sensitive to outliers due to the squared term.
*   **Cross-Entropy Loss (Log Loss):**
    *   Binary Classification:
        $$
        L = -\frac{1}{N} \sum_{i=1}^N [y_i \log(p_i) + (1 - y_i) \log (1 - p_i)]
        $$
    *   Multi-class Classification:
        $$
        L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^M [y_{ic} \log (p_{ic})]
        $$
    *   Used in classification problems, especially with logistic regression.
*   **Hinge Loss:** Used in SVM.
    *   $L(y, \hat{y}) = \max(0, 1 - y\hat{y})$ where $y \in \{-1, 1\}$ and $\hat{y}$ is the predicted value.
MSE is widely used but can overemphasize large errors; MAE is more robust to outliers but not differentiable at zero. Cross-Entropy (Log Loss) is tied to the idea of maximum likelihood in classification. Hinge Loss for SVM emphasizes a large margin for separable classes.

### 1.5 Optimization: Gradient Descent

*   **Goal:** To find the minimum of the cost function by iteratively adjusting the model parameters.
*   **Batch Gradient Descent:** Computes the gradient using the entire training dataset in each iteration.
    *   Pros: Stable convergence.
    *   Cons: Slow for large datasets.
*   **Stochastic Gradient Descent (SGD):** Computes the gradient using a single randomly selected data point in each iteration.
    *   Pros: Fast updates, can escape local minima.
    *   Cons: Noisy updates, requires careful tuning of the learning rate.
*   **Mini-Batch Gradient Descent:** Computes the gradient using a small random subset (mini-batch) of the training data in each iteration.
    *   Pros: Balances the advantages of batch and stochastic gradient descent.
    *   Cons: Requires tuning of the batch size and learning rate.
*   **Algorithm (Mini-Batch):**
    1.  Initialize parameters $\theta$.
    2.  Repeat until convergence:
        *   Randomly sample a mini-batch of data points.
        *   Compute the gradient of the cost function with respect to $\theta$ using the mini-batch.
        *   Update parameters: $\theta = \theta - \eta \nabla J(\theta)$, where $\eta$ is the learning rate.
Learning rate (η) size strongly affects convergence, as too large a value may diverge while too small converges slowly. Extensions like Momentum, RMSProp, and Adam adapt the learning rate or include momentum. Convergence often relies on set iteration counts, small parameter updates, or monitoring validation error for early stopping. Shuffling data before forming mini-batches also helps avoid local minima or poor convergence.

## 2. Supervised Learning Models

### 2.1 Logistic Regression

*   **Purpose:** Binary classification.
*   **Hypothesis:** Models the probability of a binary outcome using the sigmoid function.
    $$
    P(y=1|x) = \frac{1}{1 + e^{-(\theta^T x)}}
    $$
*   **Loss Function:** Log Loss (Cross-Entropy).
*   **Optimization:** Gradient Descent or other optimization algorithms to maximize the likelihood.
*   **Decision Boundary:** Linear.
*   **Example:** Predicting whether a customer will click on an ad.
*   **Key points:**
    *   Assumes a linear relationship between the log-odds of the outcome and the predictors.
    *   Outputs probabilities.
Logistic regression coefficients can be interpreted as changes in the log-odds per unit increase in a feature. Typical performance measures are accuracy, precision, recall, F1, and ROC AUC. Extensions include multinomial logistic (Softmax) for multi-class tasks and regularized variants (L1, L2, or both).

### 2.2 Linear Regression

*   **Purpose:** Regression.
*   **Hypothesis:** Models the relationship between the dependent variable and independent variables as a linear function.
    $$
    y = \theta^T x + \epsilon
    $$
    where $\epsilon$ is the error term.
*   **Loss Function:** Mean Squared Error (MSE).
*   **Optimization:** Ordinary Least Squares (OLS) or Maximum Likelihood Estimation (MLE).
*   **Assumptions:**
    *   Linearity: Linear relationship between independent and dependent variables.
    *   Independence: Errors are independent.
    *   Homoscedasticity: Constant variance of errors.
    *   Normality: Errors are normally distributed (for MLE).
    *   No Multicollinearity: No high correlation between features.
*   **Example:** Predicting house prices based on features like size and location.
*   **Handling Multicollinearity:**
    *   Remove correlated variables.
    *   Use PCA to reduce dimensionality.
    *   Use Ridge Regression (L2 regularization).
OLS has a closed-form solution (θ = (XᵀX)⁻¹ Xᵀy) unless XᵀX is non-invertible, in which case regularization or dimensionality reduction applies. Large datasets often require gradient-based methods. Inference on coefficients assumes somewhat ideal conditions, though the model can still predict if assumptions are relaxed.

### 2.3 Decision Tree

*   **Purpose:** Classification or Regression.
*   **Algorithm:** Recursively partitions the data based on feature values to create a tree-like structure.
*   **Splitting Criteria:**
    *   Classification: Information Gain, Gini Impurity.
    *   Regression: Mean Squared Error (MSE).
*   **Example:**
    *   Imagine you want to predict if a person will play tennis. You start with the entire dataset at the root node.
    *   The algorithm looks at each feature (e.g., Outlook, Temperature, Humidity, Wind) and calculates which feature provides the best split based on a chosen criterion (e.g., Information Gain).
    *   Suppose "Outlook" provides the best split. The tree branches into three nodes: Sunny, Overcast, and Rainy.
    *   The algorithm continues recursively for each branch until a stopping criterion is met (e.g., all data points in a node belong to the same class, or a maximum depth is reached).
*   **Preventing Overfitting:**
    *   Pruning: Removing branches that do not improve performance on a validation set.
    *   Limiting tree depth.
    *   Setting minimum sample requirements for splitting nodes.
Common impurity measures for classification include Gini (1 - Σpᵢ²) and Entropy (-Σpᵢ log₂ pᵢ). Pre-pruning can stop splits early when improvement is minimal, and post-pruning removes branches that don’t lower validation error. Decision trees typically have low bias but high variance, mitigated by ensemble methods.

### 2.4 Random Forest

*   **Purpose:** Classification or Regression.
*   **Algorithm:** Ensemble of decision trees trained on random subsets of the data and features (Bagging).
*   **Key Concepts:**
    *   **Bagging (Bootstrap Aggregating):** Randomly sampling the training data with replacement to create multiple subsets. Each subset is used to train a separate decision tree.
    *   **Random Subspace:** Randomly selecting a subset of features for each tree.
*   **Prediction:**
    *   Classification: Majority voting among the trees.
    *   Regression: Averaging the predictions of the trees.
*   **Feature Importance:** Calculated based on how much each feature reduces impurity across all trees.
*   **Example:**
    1.  **Bootstrap Sampling:** Create multiple (e.g., 100) bootstrap samples from the original dataset.
    2.  **Feature Subset Selection:** For each tree, randomly select a subset of features (e.g., $\sqrt{\text{number of features}}$).
    3.  **Tree Training:** Train a decision tree on each bootstrap sample using the selected feature subset.
    4.  **Prediction:** For a new data point, each tree makes a prediction. The final prediction is the majority vote (classification) or average (regression) of all tree predictions.
*   **Why Random Forest Reduces Variance:** By averaging the predictions of multiple trees, each trained on a different subset of the data and features, the random forest reduces the variance of the model.
*   **Out-of-Bag (OOB) Error:** The average prediction error on each training sample $x_i$ , using only the trees that did not have $x_i$ in their bootstrap sample.
Usually made of CART trees grown to significant depth, with bagging to reduce variance. Out-of-Bag (OOB) samples deliver an unbiased performance estimate without a separate test set. Random forests resist outliers and non-linearities, handle many features, but are less interpretable than a single tree.

### 2.5 XGBoost

*   **Purpose:** Classification or Regression.
*   **Algorithm:** Gradient boosting algorithm that builds an ensemble of decision trees sequentially, with each tree correcting the errors of its predecessors.
*   **Key Concepts:**
    *   **Boosting:** Sequentially adding weak learners to improve the model's performance.
    *   **Gradient Descent:** Using gradient descent to minimize the loss function.
    *   **Regularization:** L1 and L2 regularization to prevent overfitting.
*   **Objective Function:**
    $$
    Obj = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
    $$
    where:
    *   $l$ is the loss function.
    *   $\hat{y}_i$ is the prediction for instance $i$.
    *   $\Omega$ is the regularization term.
    *   $f_k$ is the $k$-th tree.
*   **Example:**
    1.  **Initial Prediction:** Start with an initial prediction (e.g., the average of the target variable).
    2.  **Residual Calculation:** Calculate the residuals (the difference between the true values and the initial prediction).
    3.  **Tree Training:** Train a decision tree to predict the residuals.
    4.  **Prediction Update:** Update the predictions by adding the predictions of the new tree, scaled by a learning rate.
    5.  **Repeat:** Repeat steps 2-4 for a specified number of trees.
*   **GBDT vs. AdaBoost:**
    *   GBDT learns the residuals from the previous learner, while AdaBoost modifies the weights of the instances.
    *   GBDT can use different loss functions, while AdaBoost typically uses exponential loss.
It is a popular gradient boosting approach optimized for speed and performance, including direct regularization, tree-based learning, and approximations for large data. XGBoost supports custom loss functions, multiple hyperparameters (learning_rate, max_depth, etc.), handles sparse data, and often performs well on tabular tasks.

### 2.6 Support Vector Machine (SVM)

*   **Purpose:** Classification or Regression.
*   **Algorithm:** Finds the optimal hyperplane that maximizes the margin between classes.
*   **Key Concepts:**
    *   **Support Vectors:** Data points closest to the hyperplane that influence its position and orientation.
    *   **Margin:** The distance between the hyperplane and the nearest data points from each class.
    *   **Kernel Trick:** Mapping data to a higher-dimensional space to handle non-linear relationships.
*   **Kernels:**
    *   Linear: $K(x_i, x_j) = x_i^T x_j$
    *   Polynomial: $K(x_i, x_j) = (x_i^T x_j + c)^d$
    *   Radial Basis Function (RBF): $K(x_i, x_j) = \exp(-\frac{||x_i - x_j||^2}{2\sigma^2})$
*   **Example:** Classifying images of cats and dogs.
*   **Loss Function:** Hinge Loss.
*   **Optimization:** Convex optimization techniques (e.g., Quadratic Programming).
This method involves a convex quadratic problem, with the kernel trick enabling non-linear boundaries. The regularization parameter C balances margin maximization against classification errors. SVMs can be extended for probability estimates (e.g., Platt scaling), work well in high-dimensional spaces, but may be slower for very large datasets.

### 2.7 K-Nearest Neighbors (KNN)

*   **Purpose:** Classification or Regression.
*   **Algorithm:** Classifies a data point based on the majority class (classification) or average value (regression) of its k nearest neighbors in the feature space.
*   **Distance Metrics:** Euclidean distance, Manhattan distance, etc.
*   **Example:** Recommending movies based on the preferences of similar users.
*   **Algorithm:**
    1.  Calculate the distances between the new data point and all data points in the training set.
    2.  Select the k nearest neighbors based on the distance metric.
    3.  Assign the class label (classification) or predict the value (regression) based on the neighbors.
KNN does minimal “training” and classifies or regresses by majority vote or averaging among the k nearest neighbors at prediction time. Feature scaling is important to avoid unfair distance weighting, and a well-chosen k helps prevent over- or underfitting. Data structures like KD-trees can speed searches.

## 3. Unsupervised Learning Models

### 3.1 K-Means Clustering

*   **Purpose:** Partitioning data into k clusters based on similarity.
*   **Algorithm:**
    1.  Initialize k centroids randomly.
    2.  Assign each data point to the nearest centroid.
    3.  Recalculate the centroids as the mean of the data points in each cluster.
    4.  Repeat steps 2 and 3 until convergence.
*   **Loss Function:** Euclidean distance between each data point and its centroid.
    $$
    J = \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2
    $$
    where:
    *   $C_i$ is the $i$-th cluster.
    *   $\mu_i$ is the centroid of the $i$-th cluster.
*   **Example:** Segmenting customers based on their purchasing behavior.
*   **Determining the Number of Clusters (k):**
    *   Elbow Method: Plot the within-cluster sum of squares (WCSS) as a function of k and choose the k value at the "elbow" point.
    *   Silhouette Analysis: Calculate the silhouette coefficient for different k values and select the one with the highest average coefficient.
*   **K-Means++ Initialization:** Choose initial centroids that are far apart from each other to improve convergence. K-Means++ helps good initialization. This algorithm fits spherical clusters well because it uses Euclidean distance. Iterative refinement might get stuck in local optima. The performance measure known as “inertia” is the sum of squared distances to centroids.
*  **Limitations:** 
    * Sensitive to the initial placement of centroids, requires specifying k in advance, and assumes spherical clusters of similar sizes.
    * Sensitive to outliers, which can skew the centroids.
*  **Improvement:** 
    * Use K-Means++ for better initialization of centroids. How it works? 

### 3.2 EM Algorithm

*   **Purpose:** Estimating parameters of statistical models with latent variables.
*   **Algorithm:** Iteratively alternates between two steps:
    *   **Expectation (E) Step:** Compute the expected values of the latent variables given the current parameter estimates.
    *   **Maximization (M) Step:** Update the parameter estimates by maximizing the expected log-likelihood calculated in the E-step.
*   **Example:** Estimating the parameters of a Gaussian Mixture Model (GMM).
*   **GMM:** A probabilistic model that assumes the data is generated from a mixture of Gaussian distributions.
*   **Relationship to K-Means:** K-Means can be seen as a special case of GMM where the covariance matrices are spherical and equal.
Often used in Gaussian Mixture Models, where the E-step computes the expected cluster assignments given parameters and the M-step updates distributions to maximize the expected log-likelihood. Converges to a local maximum but can handle missing data or incomplete observations with flexibility.

### 3.3 Agglomerative Clustering

*   **Purpose:** Hierarchical clustering that builds a cluster hierarchy by iteratively merging the closest clusters.
*   **Algorithm:**
    1.  Start with each data point as a separate cluster.
    2.  Find the two closest clusters and merge them into a single cluster.
    3.  Repeat step 2 until all data points belong to a single cluster.
*   **Linkage Criteria:**
    *   Single Linkage: Minimum distance between points in the two clusters.
    *   Complete Linkage: Maximum distance between points in the two clusters.
    *   Average Linkage: Average distance between all pairs of points in the two clusters.
    *   Ward Linkage: Minimizes the variance within the clusters being merged.
*   **Example:** Grouping documents based on their content.
It generates a dendrogram describing successive merges based on linkage selections (single, complete, average, Ward), suitable for clusters not necessarily spherical. This can be computationally expensive (O(n² log n)) for large data.

## 4. Evaluation Metrics

### 4.1 Classification Metrics

*   **Accuracy:** The proportion of correctly classified instances.
    $$
    Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
    $$
*   **Precision:** The proportion of correctly predicted positive instances among all instances predicted as positive.
    $$
    Precision = \frac{TP}{TP + FP}
    $$
*   **Recall (Sensitivity):** The proportion of correctly predicted positive instances among all actual positive instances.
    $$
    Recall = \frac{TP}{TP + FN}
    $$
*   **F1-Score:** The harmonic mean of precision and recall.
    $$
    F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
    $$
*   **F2-Score:** A weighted harmonic mean of precision and recall, giving more weight to recall.
    $$
    F2 = (1 + 2^2) \cdot \frac{Precision \cdot Recall}{2^2 \cdot Precision + Recall}
    $$
*   **ROC AUC:** Area Under the Receiver Operating Characteristic curve, which plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds.
*   **PR AUC:** Area Under the Precision-Recall curve, which plots precision against recall at various classification thresholds.
Accuracy may mislead for imbalanced datasets, so consider precision (when false positives are costly) or recall (when false negatives are costly). The Fβ-score generalizes F1 by weighting recall more or less. ROC AUC remains high even with imbalance, whereas PR AUC is more informative for rare positive classes. Probability calibration can be checked with Brier scores or calibration curves.

### 4.2 Regression Metrics

*   **Mean Squared Error (MSE):**
    $$
    MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
    $$
*   **Root Mean Squared Error (RMSE):**
    $$
    RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
    $$
*   **R-squared:** The proportion of variance in the dependent variable that is explained by the model.
    $$
    R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    $$
*   **Adjusted R-squared:** Adjusts the R-squared value based on the number of predictors in the model.
    $$
    Adjusted \ R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
    $$
    where $p$ is the number of predictors.
While MSE and RMSE heavily penalize large errors, MAE is more robust but not differentiable at zero. MAPE is expressed in relative percentages but fails near zero targets. R-squared reflects the fraction of variance explained, and Adjusted R-squared corrects for excessive feature inclusion.

## 5. ML Practice

### 5.1 Imbalanced Data Handling

*   **Problem:** When the classes in the dataset are not represented equally.
*   **Impact:** Standard classifiers may be biased towards the majority class.
*   **Techniques:**
    *   **Resampling:**
        *   Oversampling: Increasing the number of instances in the minority class (e.g., SMOTE).
        *   Undersampling: Decreasing the number of instances in the majority class.
    *   **Cost-Sensitive Learning:** Assigning different misclassification costs to different classes.
    *   **Evaluation Metrics:** Precision, Recall, F1-score, AUPRC.
Methods like SMOTE generate synthetic minority samples. Ensemble approaches like EasyEnsemble combine multiple undersampled or oversampled subsets. Focal Loss helps highlight difficult examples in tasks like object detection. Inspecting the confusion matrix gives insights into prediction distribution across classes.

### 5.2 Missing Data Imputation

*   **Problem:** When some values are missing in the dataset.
*   **Techniques:**
    *   **Deletion:** Removing rows or columns with missing values.
    *   **Imputation:** Replacing missing values with estimated values.
        *   Mean/Median Imputation: Replacing missing values with the mean or median of the feature.
        *   K-Nearest Neighbors (KNN) Imputation: Replacing missing values with the average of the k-nearest neighbors.
*   **Impact on Algorithms:** Naive Bayes is robust to missing values.
Possible mechanisms include MCAR, MAR, and MNAR, each requiring different assumptions. Regression or EM-based methods can estimate missing values, while simpler approaches include mean substitution or nearest-neighbor strategies. Familiarity with domain context remains key for correct handling.

### 5.3 Outlier Handling

*   **Problem:** When some data points are significantly different from the other data points in the dataset.
*   **Impact:** Outliers can affect the performance of many machine learning algorithms, especially linear models.
*   **Detection Techniques:**
    *   Interquartile Range (IQR): Identifying data points outside the range $[Q1 - 1.5 \cdot IQR, Q3 + 1.5 \cdot IQR]$.
    *   Z-scores: Identifying data points with a Z-score greater than a threshold (e.g., 3).
    *   Boxplots: Visualizing the distribution of the data and identifying outliers.
*   **Handling Techniques:**
    *   Trimming: Removing outlier data points.
    *   Capping/Flooring: Replacing outlier values with a maximum or minimum value.
    *   Transformation: Applying a transformation to reduce the impact of outliers (e.g., log transformation).
Outliers may represent data entry errors or genuine anomalies. Eliminating them risks losing rare but potentially critical data points. Domain knowledge clarifies whether outliers are noise or important signals. Robust estimation methods, such as RANSAC, reduce the impact of outliers on linear models.

