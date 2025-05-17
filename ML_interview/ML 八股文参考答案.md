# ML基础概念类1
1. overfitting/underfiting是指的什么
   - Underfitting: Machine learning model is too simple to capture the underlying patterns in the data. Model performs poorly on both training and new unseen data
     - training and validation error both high
     - more complex model, adding more features
   - Overfitting: machine learning model becomes too complex and starts to memorize the trianing data instead of learning generalizable patterns.
     - training error significantly lower than the validation error/model perform poorly on new data
     - reducing the complexity of the model
     - regularizing the model (l1, l2 regularization, dropout, cross-validation to choose best model)
     - collecting more training data
1. bias/variance trade off 是指的什么
   - bias: difference between predicted value and the expectation of the real data. Bias happens when the model oversimplifies the underlying patterns in the data and makes strong assumptions. This can lead to underfitting, where the model fails to capture the true relationships between the features and the target variable
   - Variance: measures how spread the predicted values are from the expected value. A model with high variance is sensitive to the specific data points and memorize noise or outliers. This can lead to overfitting.
   - Trade-off: Low variance models tend to be less complex, with simple structure, this can lead to high bias. Low bias models tend to be more complex, with flexible undering structure, which can leads to high variance. decreasing one component often leads to an increase in the other. Achieving low bias and low variance simultaneously is challenging. The goal is to find the right balance between bias and variance for optimal model performance.
3. Overfitting一般有哪些预防手段
   - reducing the complexity of the model
   - regularizing the model (l1, l2 regularization, dropout, cross-validation to choose best model)
   - Early stop
   - collecting more training data
   - data augmentation
4. Give a set of ground truths and 2 models, how do you be confident that one model is better than another? Model Selection
    - Evaluation Metrics
    - Cross-Validation (split data into multi-fold，train each model on different fold and test on alternating set, evaluate their average performance)
    - Hypothesis Testing A/B testing
    - Domain Experties
## Regression
1. Linear Regression的基础假设是什么
    - There is a linear relationship between the independent variables(X) and the dependent variables (y)
    - Independence: Independence assumes that there is no relationship or correlation between the errors (residuals) of different observations.
    - Normality: The residuals of the linear regression model are assumed to be normally distributed.
    - Homoscedasticity: Homoscedasticity assumes that the variability of the errors (residuals) is constant across all levels of the independent variables.
    - No Multicollinearity between features
2. what will happen when we have correlated variables, how to solve
    - Outcome of correlated variables: unstable coefficient estimates, unreliable significance tests, difficulties in interpreting the individual contributions of the correlated variables
    - Solve: feature selection, ridge regression, PCA…
3. explain regression coefficient
    Coefficients represent the change in the dependent variable associated with a one-unit change in the corresponding independent variable, while holding other variables constant. It is essential to note that the interpretation of regression coefficients should be done with caution and within the context of the specific regression model and dataset.
4. what is the relationship between minimizing squared error and maximizing the likelihood
    - In linear regression, when the assumption of Gaussian errors holds, minimizing the squared error is equivalent to maximizing the likelihood of the observed data. This connection arises because the squared error can be derived from the likelihood function assuming Gaussian errors.
    - In cases where the assumption of Gaussian errors is not appropriate, such as when dealing with non-Gaussian or heteroscedastic errors, the relationship between minimizing squared error and maximizing likelihood might not hold.
5. How could you minimize the inter-correlation between variables with Linear Regression?
    - Feature Selection
    - PCA
    - Ridge Regression
    - Feeature Engineering
6. If the relationship between y and x is no linear, can linear regression solve that
    - simple linear regression may not accurately capture the underlying relationship.
    - Solve:
        - interaction terms
        - piecewise linear regression
        - non-linear regression
7. why use interaction variables
   - Capture Non-Additive Effects:
   - Improved Model Fit
   - Context-Specific Relationships
   - Avoiding Omitted Variable Bias
   - Enhanced Interpretability
## Reguarlization
1. L1 vs L2 **regularization**:, which one is which and difference
    - Add a term of L1 norm of the parameters in the loss function (sum of absolute values)
    - Add a term of L2 norm of the parameters in the loss function ($||\beta||_2 = (\sum \beta_i^2)^{1/2}$)
2. Lasso Regression
   - Least Absolute Shrinkage and Selection Operator
   - Introduces an additional penalty term based on the absolute values of the coefficients, L1 norm of the coefficients
   - objective: find the value of the coefficients that minimize the sum of the squared differences between the predicted values and the actual values, while also minimizing the L1 regularization term
   - $L=|| \hat{y} - y ||_2 + \lambda || \beta ||_1$,
   - where $\hat{y} = f_{\beta}(x)$
   - Lasso regression can shrink the coefficients towards zero. when $\lambda$ is sufficiently large, some coefficients are driven to zero. Useful for feature selection
3. Ridge Regression
   - Linear Regression with L2 Regularization
   - $L = ||\hat{y} - y||_2 + \lambda||\beta||_2$
   - Higher values of lambda result in more aggressive shrinkage of the coefficient estimators.
4. 为什么L1比L2稀疏
   - L1 norm has corners at zero, while L2 norm is smooth and continuously differentiable
   - L1 norm penalty creates diamond-shaped constraint regions in the coefficient space, centered around the origin. As a result, the optimization process may drive some coefficient exactly to zero, leading to sparsity (the optimum solution/plain usually hits the vertex of the dimond) Whereas L2 norm is a ball, the optimum solution usually hits a point where the coefficients are non zero.
5. 为什么regularization works
    Regularization works by introducing penalty term into the objective function of a machine learning model. This penalty term encourage the model to have certain desirable properties, such as simplicity, sparsity, or smoothness (Adding more constrains to the coefficient). Reduce the variance.
6. 为什么regularization用L1 L2，而不是L3, L4..
    - mathematical properties, L1 L2 norms have well-studied mathematical properties that make them particularly useful for regularization. Their properties align with the goals for reducing model complexity, handling Multicollinearity and identifying important features
    - Computational simplicity, high order can introduce additional computational complexity without providing significant advantages over L1 and L2 norm
    - Interpretability
## Metrics
1. precision and recall, trade-off
    - Precision is a measure of how many of the **positively predicted instances are actually true** positives. =
        - true positive / (true positive + false positive).
        - Precision focuses on the quality of the positive predictions, high precision aiming for **a low number of false positives**.
    - Recall is a measure of how many of the actual positive instances are correctly identified
        - True positive / (true positive + false negative).
        - Recall emphasizes the completeness of the positive predictions, high recall aiming for a **low number of false negatives.**
    - Trade-off:
        - improving one metric might lead to a decrease in the other.
        - high precision, low recall: tuned to prioritize precision. more conservative in predicting positive instances. result in a **low number of false positives but may lead to missing some true positive instances, resulting in a low recall.**
        - low precision, high recall: be more liberal in predicting positive instances, lead to a **high number of true positives, but may also generate more false positives, reducing precision.**
        - the consequences of false positives and false negatives
        - the desired balance between avoiding mis-classification errors and capturing all relevant positive instances
        - It's important to consider precision and recall together and select the appropriate balance
2. label 不平衡时用什么metric
    - Precision and Recall
    - F1-score (harmonic mean of precision and recall) provides a balanced evaluation metric for imbalanced dataset
    - Area Under the Precision-Recall Curve (AUPRC) always use when the focus is on the positive class robust to class imbalance
    - Receiver Operating Characteristic (ROC) curve and the Area Under The Curve (AUC). ROC curve plots the true positive rate (recall) against the false positive rate at different classification thresholds. AUC is the area under the ROC curve it is widely used metric that quantifies the model’s discriminative power and is suitable for imbalanced dataset
3. 分类问题该选用什么metric，and why
    - Understand the problem, identify the importance of correctly classifying each class and whether there is a class imbalance in the dataset
    - Define evaluation goals, consider false positive vs false negative, different impacts? Decide whether the emphasis is on overall accuracy, precision or recall, or a balanced trade-off
    - class imbalance
    - domain knowledge
    - multiple metrics
4. confusion matrix
    A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for each class. By examining the values in the confusion matrix, you can gain insights into different performance aspects of the classification model, such as accuracy, precision, recall, and F1-score.
5. true positive rate, false positive rate, ROC (for binary classification)
    - **True Positive Rate (TPR) or Sensitivity or Recall**: The TPR measures the proportion of actual positive instances that are correctly classified as positive by the classifier. It is calculated as the ratio of true positives (TP) to the sum of true positives and false negatives (FN). **TPR = TP / (TP + FN) = TP / (ALL Actual positive examples)**
    - **False Positive Rate (FPR):** The FPR measures the proportion of actual negative instances that are incorrectly classified as positive by the classifier. It is calculated as the ratio of false positives (FP) to the sum of false positives and true negatives (TN). **FPR = FP / (FP + TN) = FP / (All Actual Negative examples).** The TPR indicates the classifier's ability to correctly identify positive instances from the actual positive class. A higher TPR suggests better sensitivity or recall.
    - **Receiver Operating Characteristic (ROC) Curve**: plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds
6. AUC的解释 (for binary classification)
    - The AUC represents the area under the receiver operating characteristic (ROC) curve, which plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds.
    - It represents the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance.
    - The AUC value ranges from 0 to 1. A model with an AUC of 0.5 performs no better than random guessing, as the ROC curve coincides with the diagonal line connecting (0,0) and (1,1). A perfect classifier achieves an AUC of 1, as it can perfectly separate positive and negative instances.
7. Ranking metrics
    - **Mean reciprocal rank (MRR)**: This metric measures the quality of the model by considering the rank of the first relevant item in each output list produced by the model, and then averaging them.
        - $$MRR = \frac{1}{m} \sum_{i=1}^m \frac{1}{\text{rank}_i}$$
        - shortcoming: only considers the first relevant item and ignores other relevant items in the list, it does not measure the precision and ranking quality of a ranked list.
    - **Recall@k:** This metric measures **the ratio between the number of relevant items** **in the output list** and **the total number or relevant items available in the entire dataset**. The formula is
        - $$\text{recall \@ k} = \frac{\text{number of relevant items among the top $k$ items in the output list}}{\text{total relevant items}}$$
        - measures how many relevant items the model failed to include in the output list
        - shortcoming: in some systems, the total number of relevant items can be very high. This negatively affects the recall as the denominator is very large. For example, if we want to find a list of image the close to a query image of dog, when the databse may contain millions of dog images. The goal is not to return every dog image but to retreve a handful of the most similar dog images.
    - **Precision@k:** measures the **proportion** of **relevant items among the top k items in the output list**. The formula is:
        - $$\text{precision\@k} = \frac{\text{number of relevant items among the top $k$ items in the output list}}{k}$$
        - measures **how precise the output lists are**, but **it doesn’t consider the ranking quality**.
    - **Average Precision (AP):** computes average precision@K for each k. AP is high if more relevant items are located at the top of the list.
    - **mAP**: first computes the average precision (AP) for each output list, and then averages AP values.
        - mAP is designed for binary relevances; in other words, it works well when each item is either relevant or irrelevant. For continuous relevance scores, nDCG is a better choice.
    - Normalized discounted cumulative gain (nDCG)
        - DCG calculates the cumulative gain of items in a list by summing up the relevance score of each item
        - $$\text{DCG}p = \sum_{i=1}^p\frac{rel_i}{\log_2(i+1)}$$
        - **nDCG divides the DCG by the DCG of an ideal ranking. The formula is:**
        - $$nDCG_p = \frac{DCG_p}{IDCG_p}$$
        - Its primary shortcoming is that **deriving ground truth relevance scores is not always possible**.
8. Recommender System Metrics
    - **Precision@k: proportion of relevant content among the top k recommended items**
    - MRR: focuses on the rank of the first relevant item in the list, suitable in system where only one relevant item is expected
    - mAP: average of all recommended items AP, measures the ranking quality of recommended items. mAP works only when the relevance scores are binary (if the score is ether relevant vs irrelevant, mAP is a better fit)
    - nDCG: relevance score between a user and an item is non-binary ( [relevant vs irrelevant case(mAP)] vs [how relevant case(nDCG)]
    - Diversity: This metric measures how dissimilar recommended videos are to each other. This metric is important to track, as users are more interested in diversified videos. To measure diversity, we calculate the average pairwise similarity (e.g., cosine similarity or dot product) between videos in the list. A low average pairwise similarity score indicates the list is diverse.
    mAP, MRR, nDCG are commonly used to measure ranking quality




## Loss
1. 用MSE做loss的Logistic Rregression是convex problem吗
    No, not convex,

2. 解释并写出MSE的公式, 什么时候用到MSE?
    - Mean Square Error $\frac{1}{N}\sum_{i=1}^N(Y_i - \hat{Y}_i)^2$
    - average of the squares of the error
    - Regression, normally distributed errors, Emphasizing larger errors(more weight to larger error, due to square)
3. Linear Regression最小二乘法和MSE关系
    - MSE is the objective function of Least Squares Method for linear regression
    - Least Square Method finds the coefficients (global min) that minimize the sum of squared residuals which is equivalent to minimizing the MSE.
4. 什么是relative entropy, What is KL divergence
    - Relative entropy, also known as Kullback-Leibler divergence or cross entropy, is a measure that quantifies the difference between two probability distributions.
    - $D_{KL}(P||Q) = \sum_{x\in\mathcal{X}}P(x) \log \left( \frac{P(x)}{Q(x)}\right)$
    - Expectation with respect to the distribution $p$ of the logarithmic difference between the probabilities P and Q
    - Cross Entropy $H(P, Q) = -\sum_{x\in\mathcal{X}}P(x) \log Q(x) = H(p) +D_{KL}(P||Q)$, expected value of logq with respect to the distribution p
5. Logistic Regression的loss是什么
   

6. Logistic Regression的 Loss 推导
    use Maximum Likelihood Estimation
7. SVM的loss是什么
    - goal is to find a hyperplane that separates the data points of different classes with a maximum margin.
    - hinge loss formulation and the squared hinge loss formulation.

8. Multiclass Logistic Regression, 为什么用cross entropy做cost function
    - softmax regression/Multinomial logistic regression
    - extension of binary logistic regression to handle problems of multiple classes
    - The softmax function takes the linear combination of the feature values and weights for each class and applies the exponential function to them. It then normalizes the exponentiated values by dividing them by the sum of all exponentiated values. This normalization ensures that the resulting probabilities sum up to 1.
    - Multiclass logistic regression uses a specific loss function called the cross-entropy loss or log loss to measure the discrepancy between the predicted class probabilities and the true class labels. The goal is to minimize this loss by adjusting the model parameters.
    - log-likelihood (provides MLE)
9. Decision Tree split node的时候优化目标是啥
    - To find the best split that maximize the separation between different classes or reduces the impurity with each resulting node.
    - different decision tree algorithm has different specific objectives
        - Gini impurity
        - Entropy
        - Information Gain
10. Log-loss是什么，什么时候用logloss
    - Log loss, also known as logarithmic loss or logistic loss, is a loss function commonly used in binary and multiclass classification problems. It measures the performance of a classification model by quantifying the discrepancy between predicted probabilities and the true class labels.
    - based on the principles of maximum likelihood estimation and information theory.
    - The formula for log loss in binary classification is: Log Loss = -[y * log(p) + (1 - y) * log(1 - p)]
    - Classification problem usually uses log loss
Logistic Regression
1. logistic regression和svm的差别 （我想这个主要是想问两者的loss的不同以及输出的不同，一个是概率输出一个是score）
    - Objective:
        - logistic regression: model the probability of an instance belonging to a certain class using a logistic/sigmoid function. It estimates the probabilities and makes predictions based on a threshold
        - SVM: the objective of SVM is to find a hyperplane that maximally separates the instance of different classes. SVM focuses on maximizing the margin between the classes, rather than directly estimating class probabilities.
    - Decision boundary
        - logistic regression: linear decision bounday, dividing the feature space into two regions for binary classificaiton
        - SVM: also uses linear decision bounday, but it can employ non-linear decision boundaries through kernel
    - Loss function:
        - logistic regression: log loss: quantifies the discrepancy between predicted probabilities and true class labels.
        - SVM: hinge loss penalizes misclassifications and encourages maximizing the margin
    - Optimization method:
        - logistic regression: gradient descent, newton’s method, or other iterative method
        - SVM: convex optimization techniques such as quadratic programming or sequential minimal optimization
    - Robust
        - logistic regression is sensitive to outliers
        - SVM is more robust to outliers due to the margin maximization objective.

Decision Tree
1. How regression/classification DT split nodes
    - Regression: find the split that minimizes the variance or the sum of squared differences between the predicted and actual values within each resulting node.
    - Classification: find the split that maximizes the separation or purity between different classes within each resulting node
2. How to prevent overfitting in DT? How to do regularization in DT?
    - pruning
    - Set minimum Sample Requirements for splittting nodes during tree construction
    - Feature selection
    - Ensemble methods
    - cross-validation
    - maximum depth
    - minimum impurity decrease
Clustering and EM
1. K-means clustering (explain the algorithm in detail; whether it will converge, 收敛到global or local optimums; how to stop
    K-means clustering is an unsupervised machine learning algorithm used for partitioning a dataset into k distinct clusters based on similarity measures. It aims to minimize the within-cluster sum of squares (WCSS) or the average squared distance between data points and their assigned cluster centroids. Here's an explanation of the k-means clustering process:
    - Initialization:
    The algorithm starts by randomly selecting k initial cluster centroids. These centroids can be randomly chosen from the data points or initialized using other strategies.
    - Assignment:
    Each data point is assigned to the nearest centroid based on a distance metric, commonly the Euclidean distance. This step creates initial clusters based on the closest centroids.
    - Update:
    The centroids are recalculated as the mean of all the data points assigned to each cluster. This step updates the centroid positions, adjusting them to be closer to the data points in their respective clusters.
    - Iteration:
    Steps 2 and 3 are repeated iteratively until convergence. In each iteration, data points are reassigned to the closest centroids, and centroids are updated based on the new assignments.
    Convergence: K-means clustering algorithm will eventually converge to a solution. The convergence occurs when the assignments and centroids no longer change significantly between iterations or when a predefined stopping criterion is met.
    Global vs. Local Optima: K-means clustering is **sensitive to the initial positions of the centroids**. It can converge to a local optimum rather than the global optimum, which means that the clustering result depends on the initial centroid locations. Multiple runs with different initializations are commonly performed to mitigate the risk of getting trapped in a poor local optimum.
    Stopping Criterion: Several stopping criteria can be used to stop the training iterations of k-means clustering. Common approaches include:
    - Convergence criterion: Stop when the centroids and assignments no longer change significantly.
    - Maximum number of iterations: Set a fixed number of iterations to limit computation time.
    - Threshold on WCSS improvement: Stop when the improvement in WCSS falls below a specified threshold.
    Selecting the Number of Clusters (k): Choosing the optimal number of clusters (k) is a critical step in k-means clustering. Some methods for determining the appropriate k value include:
    - Elbow method: Plot the WCSS as a function of k and choose the k value at the "elbow" point, where further increasing k does not significantly reduce WCSS.
    - Silhouette coefficient: Calculate the silhouette coefficient for different k values and select the one with the highest average coefficient, indicating better cluster quality.
    In summary, k-means clustering is an iterative algorithm that partitions a dataset into k clusters by minimizing the WCSS. It eventually converges, but it can be sensitive to initializations, leading to local optima. Stopping criteria are used to determine when to stop the iterations, and techniques such as the elbow method or silhouette coefficient aid in selecting an appropriate value for k.
2. EM算法是什么
    The EM (Expectation-Maximization) algorithm is an **iterative optimization algorithm used to estimate the parameters of statistical models with latent or unobserved variables**. It is particularly useful in situations where there are missing data, incomplete data, or when the data is **generated from a mixture of probability distributions**. The EM algorithm alternates between an expectation step (E-step) and a maximization step (M-step) to update the parameter estimates. Here's an overview of how the EM algorithm works:
    - Initialization:
    Initialize the parameters of the model with some initial values.
    - Expectation Step (E-step):
    In the E-step, the algorithm computes the expected values of the latent variables or missing data given the current parameter estimates. It calculates the posterior probabilities or membership probabilities of each latent variable.
    - Maximization Step (M-step):
    In the M-step, the algorithm updates the parameter estimates by maximizing the expected log-likelihood calculated in the E-step. It treats the expected values of the latent variables as complete data and performs a standard maximum likelihood estimation or other optimization methods to find the parameter values that maximize the log-likelihood.
    - Iteration:
    Steps 2 and 3 are repeated iteratively until convergence, where the parameter estimates no longer change significantly or a stopping criterion is met. The convergence can be determined by monitoring the change in the log-likelihood or other convergence criteria.
    - Final Parameter Estimates:
    After convergence, the final parameter estimates are obtained, representing the maximum likelihood estimates or maximum a posteriori estimates given the observed and latent variables.
    The EM algorithm is widely used in various statistical modeling and machine learning applications, including clustering, mixture models, hidden Markov models, and more. It provides a framework for estimating parameters when dealing with incomplete or missing data. The E-step computes the expectation of the latent variables given the current parameters, and the M-step maximizes the log-likelihood using these expected values to update the parameter estimates.
    It's important to note that the EM algorithm finds a **local optimum** and **may not guarantee convergence to the global optimum**. Multiple runs with different initializations are often performed to mitigate the risk of being trapped in a poor local optimum.
3. GMM是什么，和Kmeans的关系
    GMM stands for Gaussian Mixture Model, which is a probabilistic model used for **representing and analyzing data that can be modeled as a combination of multiple Gaussian distributions**. It is a powerful technique for **density estimation**, **clustering**, and **generating synthetic data**. The GMM **assumes that the observed data points are generated from a mixture of Gaussian** distributions, each representing a different underlying component or cluster. Here's an explanation of the GMM model:
    - Model Representation:
    The GMM represents the data as a weighted sum of K Gaussian distributions. Each Gaussian distribution corresponds to a component or cluster within the data. The GMM model is defined by the following parameters:
        - Component weights (π): The proportions or probabilities of each component in the mixture.
        - Mean vectors (μ): The mean values of the Gaussian distributions.
        - Covariance matrices (Σ): The variance and covariance of the Gaussian distributions.
    - Probability Density Function:
    The GMM calculates the probability density function (PDF) of a data point x as the weighted sum of the PDFs of the individual Gaussian components:
    PDF(x) = ∑(π_k * N(x | μ_k, Σ_k))
        Here, N(x | μ_k, Σ_k) represents the Gaussian distribution with mean μ_k and covariance Σ_k, and π_k is the weight or proportion of the k-th component.
    - Parameter Estimation:
    The parameters of the GMM, including the component weights, mean vectors, and covariance matrices, are estimated from the observed data using the Expectation-Maximization (EM) algorithm. The EM algorithm iteratively updates the parameter estimates by alternating between an E-step and an M-step, as explained in the previous response about the EM algorithm.
    - Applications:
    GMMs have various applications, including:
        - Density Estimation: GMMs can estimate the underlying probability density function of the data, enabling the generation of synthetic data samples.
        - Clustering: GMMs can perform soft clustering, where each data point is assigned a probability of belonging to each cluster, allowing for fuzzy boundaries between clusters.
        - Anomaly Detection: GMMs can detect anomalies by modeling the normal behavior of the data and identifying instances with low likelihoods under the model.
    The GMM model provides a flexible framework for capturing complex data distributions by combining multiple Gaussian components. It is a versatile tool for density estimation, clustering, and other statistical modeling tasks, offering probabilistic interpretations and handling data with varying cluster sizes and shapes.
    GMM vs K-means: GMM and K-means have different assumptions, objectives, and cluster representations. GMM provides probabilistic soft assignment and accommodates clusters with various shapes and orientations. K-means, on the other hand, performs hard assignment and assumes spherical clusters with equal variance. The choice between GMM and K-means depends on the nature of the data and the desired characteristics of the clusters being sought.
