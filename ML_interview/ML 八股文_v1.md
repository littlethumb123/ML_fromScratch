## ML基础概念类
### Basic concepts
#### What is overfitting/underfiting
   1. Underfitting occurs when the training dataset cannot get a low error. In this case, both the training error and the test error will be high, as the classifier does not account for relevant information present in the training set. Underfitting is often due to the assumption (model) of the data is oversimplified
   2. Overfitting occurs when the generalization error is larger than training errors so that the model cannot be generalized to other samples; It often occurs when the sample size is small and feature dimension is high.
#### Bias/variance trade off
   0. Decomposition of bias and variance from the mean squared value
      * The squared errors can be decomposed into three parts, bias, variance and irreducible errors. Now assume y is the true value (constant) and $\hat{y}$ is the predicted value (a random variable). The squared error can be express in the following
        $$
        \begin{aligned}
          S &= (y-\hat{y})^2 \\
            &= (y- E[\hat{y}] + E[\hat{y}] - \hat{y})^2 \\
            &= (y- E[\hat{y}])^2 + (E[\hat{y}] - \hat{y})^2
                -2(y- E[\hat{y}])(E[\hat{y}] - \hat{y}) \\
          E[S] &= E[(y- E[\hat{y}])^2 + (E[\hat{y}] - \hat{y})^2
              - 2(y- E[\hat{y}])(E[\hat{y}] - \hat{y})] \\
          E[S] &= (y- E[\hat{y}])^2 + E[(E[\hat{y}] - \hat{y})^2]\\
                &= Bias^2 + Var
        \end{aligned}
        $$
      * Explanations:
        * here you can think of $y$ as $\theta$ in the following expression in bias and var
        *  $E[(y- E[\hat{y}])^2] = (y- E[\hat{y}])^2$ because $y$ and $E[\hat{y}]$ are all constant values so expectation are the original value
        *  The last item $E[2(y- E[\hat{y}])(E[\hat{y}] - \hat{y})] = 0$. This is because $E[E[\hat{y}] - \hat{y}] = E[\hat{y}] - E[\hat{y}] = 0$
   1. Bias errors
      * $Bias[\theta] = E[\hat{\theta}] - \theta$
        - $\theta$ is the true value and $\hat{\theta}$ is a predicted value based on a single training set. Then $E[\hat{\theta}]$ is the expected (average) predicted value of the true value at a particular point, across different training sets (or across different estimators).
      * **Bias refers to the deviations of predicted values from true values.**
      * Bias is caused because the model is over-simplified, in another word, we made a simple hypothesis on the mapping function between predictors and outcomes. So called **underfitting**
      * The bias error was on training set
      * Symptom: the model with high bias have high errors in both training and test data.
      * High bias model: Linear regression, logistic regression
      * Low bias model: decision tree, KNN, support vector machines
   2. Variance
      * $Var[\hat{\theta}] = E[(E[\hat{\theta}] - \hat{\theta})^2]$
        - $\hat{\theta}$ is a single prediction of the true value at one point $E[\hat{\theta}]$ is the expected (average) predicted value of that true value across training set. The Var in fact is the expectation of the variance of the predicted value across training sets for a single true value.
      * **Variance refers to the variability of the predicted values from a model trained with different training sets**; Or the variability of the prediction in test set due to the variations in the training dataset. A model with high variance is sensitive to changes in training dataset and cannot be generalized well to other dataset. So called **Overfitting**
      * High variance model has low errors in training set but high errors in testing set. On average the prediction was right in training set (low training errors) but the variability of each model's prediction is high (high variance)
      * The variance was on the testing set
      * High variance model includes non-linear models, decision tree
      * Low variance model includes Linear models
   3. Trade off
      * The trade off between the two is that we cannot reduce both errors at the same time. A model with low bias tends to have high variance and vice versa.
#### Overfitting:
  1. Overfitting is a scenario where your model performs well on training data but performs poorly on validation data or testing data. This basically means that the model has memorized the training data instead of learning the relationships between features and labels. Correspondingly, the model has low bias but high variance.
  2. How to reduce overfitting?  
     1. Data-based methods:
        1. Increase sample size; **The impacts of increase sample size**
             1. In general, increasing sample size **usually** reduce the chance of overfitting, and the generalization errors. Because the model can be trained with data with more diversity.
             2. **NOTICE**: the new data added should be having the same distribution with the old dataset (from the same population)
        2. Data augmentation:
        3. Reduce number of features: PCA, feature selection
     2. Algorithm methods:
         1. Regularization: L1, L2 Norm penalty, for linear, logistic regression models
         2. Ensemble: boosting, bagging and stacking
            1. Boosting:
               1. multiple independent estimators on different random samples of the original data set and average or vote across all the predictions.
               2. Reduce bias; thus good for complex model
               3. In case of regression problem we take mean of the output and in case of classification we take the majority vote.
            2. Bagging
               1. Multiple independent estimators are used sequentially, to boost the performance of the previous one by overcoming or reducing the error of the previous model.
               2. Reduce variance: it reduce the variance and meanwhile retaining the bias
               3. Bagging provides an unbiased estimate of the test error, which we refer to as the out-of-bag error.
            3. Stacking
               1. Multiple independent estimators were applied to the whole dataset. The predictions of each individual model are stacked together and used as input to a final estimator to compute the prediction
         3. For deep learning methods
            1. fewer layers (shallower networks), fewer neurons per layer, sparser connections between the layers (as in convolutional nets), or regularization techniques like dropout.

#### Underfitting

1. Underfitting: model is oversimplified and has high bias and low variance, but high training and testing errors.
2. How to solve underfitting? Reduce bias?
    - Use more complex data
    - Use more model with more complex assumption
    - add features
    - Ensemble: boosting, bagging and stacking models
    - Reduce regularization terms





### L1 and L2 Regularization:
  There are many regularization method to reduce overfitting (reduce generalization errors). Parameter norm penalties are typical ones (L1 and L2).
  1. What is a norm:
     1. Vector norms refer to the different ways the magnitude of a vector can be measured, Eucliean distance is an example. It measures the cartesian distance from the origin to the tip of the vector.
  2. How regularization works:
    1. Simply put, minimizing L1 and L2 regularization encourages the weights to be small, which in turns gives “simpler” functions. L1 and L2 shrink the learnt parameters to 0 to panelize the magnitude of coefficient; as the penalty $\lambda$ increases, the $\beta$ becomes lower. Adding terms can reduce variance.
#### L1 Regularization (Lasso)
   1. L1 norm:  The L1 norm is calculated as the sum of the absolute vector values, $||v||_1 = |a_1| + |a_2| + |a_3|$
   2. In lasso, we add a **absolute** magnitude of the coefficient to the loss function as a penalty term with a scale $\lambda$; as the penalty $\lambda$ increases, the $\beta$ becomes 0. Thus the lasso term can be used for feature selection. The following formula shows the lasso regression
   3. How the penalty term influences the weight
      $$
        J(\theta) = RSS - \frac{\lambda}{m} \sum^q_{i=1}{|w|}\\
        w = w - \eta \frac{\partial J}{\partial w} - \eta \frac{\lambda w}{m}\\
        = (1 -  \eta \frac{\lambda}{m})w - \eta \frac{\partial J}{\partial w}
      $$
      * $m$ is the batch size to train in mini-batch
      * $q$ is the number of weigtht
      * $\eta$ is the learning rate

#### L2 Regularization
  1. L2 norm: The L2 norm is calculated as the sum of the absolute vector values, $||v||_2 = a^2_1 + a^2_2 + a^2_3$
  2. As the penalty $\lambda$ increases, the $\beta$ becomes lower to 0 but not equal to zero
  $$RSS - \lambda \sum^q_{j=1}{\beta^2}$$

  5. The difference between L1 and L2 regularization
     * L1 regularization penalizes the sum of absolute values of the weights, whereas L2 regularization penalizes the sum of squares of the weights.
     * The L1 regularization solution is sparse. The L2 regularization solution is non-sparse.
     * L2 regularization doesn’t perform feature selection, since weights are only reduced to values near 0 instead of L1 regularization has built-in feature selection.
     * L1 regularization is robust to outliers, L2 regularization is not.
  6. pros and cons of each
      1. L1 reduce the weight of unimportant features to 0; whereas L2 tends to balance weights and increase the important ones whereas L1 reduce the weight close to zero, thus L1 is useful for feature selection and L2 is useful for collinear data
      3. L1 can be used to high-dimensional data with millions of features; L2 is primarily used due to its generally outperforms L1 regularization. Meanwhile L2 suffers from expensive computational sources when dimensions go higher.
  7. When to use which
     1. L1 regularization is sometimes used as a feature selection method. Suppose you have some kind of hard cap on the number of features you can use (because data collection for all features is expensive, or you have tight engineering constraints on how many values you can store, etc.). You can try to tune the L1 penalty to hit your desired number of non-zero features.
     2. L2 regularization can address the multicollinearity problem by constraining the coefficient norm and keeping all the variables. It's unlikely to estimate a coefficient to be exactly 0. This isn't necessarily a drawback, unless a sparse coefficient vector is important for some reason. This makes it useful in some traditional regression issues where number of features is more than observations.



  8. Why Lasso has sparsity/why L1 can be used for feature selection
     1. For L1, the coefficient constrain shape is a square where the angle lies on the axis. Without the constraints, the weight/coefficients will head to the center of contours of the loss function. With the L1-norm constrains, the coefficient is limited in that square shape. It turns out that the solution of coefficient is more likely to find a touch point on a spike tip and thus have zero beta coefficient, or a sparse solution.
     2. Bayesian : L1 --> assume $\theta$ follows laplace distribution
     3. L2 --> assume $\theta$ follows gaussian distribution and use MAP to derive a formula that has a L2 regularization term [here learn about the connection between regularization, MLE and MAP]()

reference [1](https://www.bilibili.com/video/BV1aE411L7sj?p=6)


2. Lasso/Ridge的解释 (prior分别是什么）
3. Lasso/Ridge的推导
4. 为什么L1比L2稀疏
5. 为什么regularization works
6. 为什么regularization用L1 L2，而不是L3, L4..
### Model Evaluation Metric:
1. Classification
   1. Accuracy, precision, recall, specificity, F1
   2. Logloss:
      1. Cross entropy: This metric captures the extent to which predicted probabilities diverge from class labels.
      2. Logloss can be any value greater than or equal to 0, with 0 it means all labels were correctly assigned.
   2. Precision-recall curve vs. ROC curve
      1. ROC-AUC: recall vs 1-specificity
      2. PR-AUC: precision vs recall
      3. Key concepts:
          * Both curves are determined by setting various threshold
          * ROC Curves summarize the trade-off between the recall true positive rate and false positive rate for a predictive model using different probability thresholds. These are two probabilities conditioned on the true class label. Therefore, they will be the same regardless of what $P(Y=1)$ is
          * Precision-Recall curves summarize the trade-off between the precision and recall for a predictive model using different probability thresholds.
          * ROC curves are appropriate when the observations are balanced between each class, whereas precision-recall curves are appropriate for **imbalanced datasets**. This is because ROC curve does not condition on the prediction, instead, it conditions on the true values and therefore imbalance dataset cannot be reflected in the formula (in denominator). However, the performance of model in predicting imbalanced dataset can be reflected in PR curve.
          * 注意TPR用到的TP和FN同属P列，FPR用到的FP和TN同属N列，所以即使P或N的整体数量发生了改变，也不会影响到另一列。也就是说，即使正例与负例的比例发生了很大变化，ROC曲线也不会产生大的变化，而像Precision使用的TP和FP就分属两列，则易受类别分布改变的影响。
          ![img](https://pic2.zhimg.com/80/v2-5b2e1966e5a1c06f050ad5954de9a1f5_1440w.jpg)
          * example: ROC曲线的缺点
          在类别不平衡的背景下，负例的数目众多致使FPR的增长不明显，导致ROC曲线呈现一个过分乐观的效果估计。ROC曲线的横轴采用FPR，根据$FPR = \frac{FP}{\text{All Neg case}} = \frac{FP}{FP+TN}$，当负例N的数量远超正例P时，FP的大幅增长只能换来FPR的微小改变。结果是虽然大量负例被错判成正例，在ROC曲线上却无法直观地看出来。
          举个例子，假设一个数据集有正例20，负例10000，开始时有20个负例被错判，FPR = 0.002，接着又有20个负例错判，FPR = 0.004，在ROC曲线上这个变化是很细微的。而与此同时Precision则从原来的0.5下降到了0.33，在PR曲线上将会是一个大幅下降。
          * ROC曲线由于兼顾正例与负例，所以适用于评估分类器的整体性能，相比而言PR曲线完全聚焦于正例。
          * How to choose?
            If true negative is not much valuable to the problem, or negative examples are abundant. Then, PR-curve is typically more appropriate. For example, if the class is highly imbalanced and positive samples are very rare, then use PR-curve. One example may be fraud detection, where non-fraud sample may be 10000 and fraud sample may be below 100.


2. Regression
   1. Mean squared error (MSE)
      1.
   2. Root Mean Squared Error (RMSE)
   3. Mean Absolute Error
   4. Residual of sum of squares
   5. R squares
      $$R = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum{(y_i - \hat{y})^2}}{\sum{(y_i - \bar{y})^2}}$$
      * RSS is residual sum of squared, is the same as SSR (Sum of Squared Residuals), SSE (Sum of Squared Errors)
        - $RSS = N*MSE$
      * TSS is Total sum of squared
        - Total Sum of Squares (TSS) is related with variance and not a metric on regression models
      * This metric does not consider the overfitting problem. As variable size increased, the R will
   6. Adjusted R squares
      * adjust the R square with number of predictors. $p$ is the number of predictors
      $$
        \begin{align}
        				 & R^2 = 1 - \frac{RSS}{TSS} \newline
        \text{Adjusted } & R^2 = 1 - \frac{RSS/(m-p-1)}{TSS/(m-1)} = 1 - \frac{m-1}{m-p-1} \frac{RSS}{TSS}
        \end{align}

      $$
    7. **Selection of metrics**: R Square/Adjusted R Square is better used to explain the model to other people because you can explain the number as a percentage of the output variability. MSE, RMSE, or MAE are better be used to compare performance between different regression models. Personally, I would prefer using RMSE and I think Kaggle also uses it to assess the submission. However, it makes total sense to use MSE if the value is not too big and MAE if you do not want to penalize large prediction errors.




* For the optimization, MSE computationally efficient is MAE
* Outliers.
Explained variance = within-class sum of error



2. label 不平衡时用什么metric
3. 分类问题该选用什么metric，and why


### Confusion matrix
1. ROC-AUC and PR-AUC

2. Imbalanced dataset:
In the imbalanced dataset, ROC AUC tended to


Variance







2. true positive rate, false positive rate, ROC
3.
   1.
4. 还有一些和场景比较相关的问题，比如ranking design的时候用什么metric，推荐的时候用什么.等.不在这个讨论范围内
### Optimization and loss function
1. 用MSE做loss的Logistic Rregression是convex problem吗
2. 解释并写出MSE的公式, 什么时候用到MSE?
3. Linear Regression最小二乘法和MLE关系
4. 什么是relative entropy/crossentropy,  以及K-L divergence 他们intuition
5. Logistic Regression的loss是什么
6. Logistic Regression的 Loss 推导
7. SVM的loss是什么: hinge loss
8. Multiclass Logistic Regression然后问了一个为什么用cross entropy做cost function
9. Decision Tree split node 的时候优化目标是啥
10. OLS vs. MLE
   1. OLS and MLE are essentially the same

Reference: https://towardsdatascience.com/understanding-sigmoid-logistic-softmax-functions-and-cross-entropy-loss-log-loss-dbbbe0a17efb







#### Difference between generative and discriminative model
    * Discriminative model
      - Discriminative models learn about the boundary between classes within a dataset.
      - Mathmatically, a D model tries to train a model by learning the parameters that maximize the conditional probability $p(Y|X)$
      - The goal is to identify the decision boundary between classes in the dataset, examples includes support vector machines, logistic regression, decision trees, and random forests.
    * Generative model
      - Generative models learn about the distribution of the individual class, i.e., how the data is produced
      - Mathmatically, a D model tries to train a model by learning the parameters that maximize the joint probability $p(X, Y)$
      - Naïve Bayes
    * Reference: http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf

    * Generative和Discrimitive的区别
        5. Give a set of ground truths and 2 models, how do you be confident that one model is better than another?
        6. Our model performs poorly in predicting both the development and production data. Is this due to Underfitting? Explain how. (development data --> )














#### General loss functions
1. Loss function, cost function and objective function
   1. Loss function: for single sample
   2. Cost function: for the whole dataset
   3. Objective function: a more general function in the context of optimization
   4. A loss function is a part of a cost function which is a type of an objective function.
2. Typical loss function

   ![img](https://pic3.zhimg.com/80/v2-ba0466b8a4ca9a7aa36c567cfc9443e9_1440w.jpg?source=1940ef5c)


### Data processing

#### Imbalanced data
   0. Moving threshold
      1. The process involves first fitting the model on a training dataset and making predictions on a test dataset. The predictions are in the form of normalized probabilities or scores that are transformed into normalized probabilities. Different threshold values are then tried and the resulting crisp labels are evaluated using a chosen evaluation metric. The threshold that achieves the best evaluation metric is then adopted for the model when making predictions on new data in the future
      2. In a study, this procedure can be done at the end of model evaluation, i.e., choose the most proper threshold for each trained model
   1. How imbalanced data affected model performance
      1. 90 positive - 10 negative:
         1. Random guess on all positive gives 90% accuracy
         2. Model will learn nothing about the negative dataset
         3. Example: flight failures, financial fraud, rare diseases.
      2. Model sensitive to imbalanced datasets
         1. Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances.
   2. How to deal with imbalanced dataset:
      1. Data engineering\
         This process focused on training dataset, manipulate the distribution of the positive vs. negative samples.
          1. Down-sampling: sample the majority class
             1. cons: lose information
          2. Over-sampling: oversampling the minority class using bootstrapping
              1. Cons:
                 1. Decrease the variance of the dataset, because a lot of datasets are duplicated.
                 2. Lead to overfitting
          3. SMOTE (Synthetic minority oversampling technique)
              1. SMOTE generated synthesized minor data points based on the existing points. It uses KNN to determine the synthesized
      2. ML model that address missing values
         1. Cost-sensitive algorithm
      3. Evaluation metrics:
         1. Using Precision recall curve
         2. Balanced accuracy
            1. If the classifier performs equally well on either class, this term reduces to the conventional accuracy (i.e., the number of correct predictions divided by the total number of predictions). In contrast, if the conventional accuracy is above chance only because the classifier takes advantage of an imbalanced test set, then the balanced accuracy, as appropriate, will drop to chance (see sketch below).
      4. Reference
         1. https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/

#### Missing data
   1. Impacts of missing values on algorithm
      1. Naive bayesian is robust to missing value see [video](https://www.youtube.com/watch?v=EqjyLfpv5oA)
      2.
#### Outliers/anomaly detection

   1. Impacts of outliers on each algorithm
      1. Linear regression
         1. Linear regression is very sensitive to outliers. It will shift the slope towards the outliers
      2. Logistic regression
         1. Similarly, logistic regression is also very sensitive to outliers.
      3. KNN
         1.KNN is sensitive to outliers because the outliers can significantly affect how the boundary is drawn.
      4. Decision tree/random forest tree
         1. Tree-based models not sensitive to the outliers in predictor variables(feature)
         2. However, tree based models may be sensitive to the outliers in outcome variables (continuous target value) because the value in each leaf node is the average of all values and may affect the predictions. However, if we use median, then tree-based regressor may not be affected by the outliers.
      5. SVM
         1.SVM is not very robust to outliers. If the outliers just lie on the margin, that will significantly impact how the boundary is determined. In some cases, if the outliers are just away from the boundary, it will not affect how the boundary is drawn.
      6. Naive bayesian:
         1. Naive bayesian is sensitive to outliers. (1) unseen data in training set can lead to zero probability (or zero frequency issue); (2) Outliers will affect the shape of the Gaussian distribution and have the usual effects on the mean etc.
      7. RNN
   2. How to detect outliers
      1. Inter quartile range (IQR) scores
         1. Mechanism: $IQR = Q3-Q1$. Consider the data points outside the lower bound and upperbound $(Q1-1.5*IQR, Q3+1.5IQR)$ as outliers.
      2. Z-scores:
         1. Normally the values out of 3 standard deviation can be considered as outliers How far a point is from the mean, how many std it is distanced, $\frac{i-\text{mean}}{\text{std}}$
      3. Boxplot: visual display
      4. Cluster:
         * The anomaly will be identified by exploring the trends or patterns within the dataset itself, then detect anomalies that sit outside these patterns. For example, a model may cluster unlabelled data into a specific count of groupings or categorisations based on relationship between data points. Individual data points that sit beyond a threshold of a cluster are identified as anomalies or outliers.

   3. How to solve:
      1. The outlier detection and handling are all in the training dataset. So all techniques are wrong, in a sense of testing dataset. These techniques are just for quick processing. They don't assume will not see the outliers in the future. CONSIDER analyzing the mechanism of the outliers and may develop customized model for that. When having observed outliers: consider the following **questions**
          * why there are outliers? measure or data entry errors or natural data trend?
          * what do they represent? less representative samples? or simply errors
          * What are the implications of the following solutions to the outcomes?
      2. Common solutions
          1. Trimming
             1. Drop the extreme values 95 percentile or 5 percentile. This makes the histogram rising
             2. Cons: lose information
          2. Flooring or capping
             1. Cap the outliers with cert with the max or min value in the dataset (e.g., 95 percentile or 5 percentile).
             2. Cons: Many data are the same on the extreme; Pros: not missing data
          3. Median or mean imputation:
             1. Replace the ouliers with median or mean value
             2. Pros: not missing data; Cons: the variance reduces and many data are the same on the extreme
          4. Transformation
             1. Square root or log transformation: Both methods will pull in greater numbers.

#### Noises in prediction
  1. Label noises: mislabeled dataset


  2. Handling mislabeled dataset
       - Noise elimination (filtering)
         - Algorithm-based filter
         - Dataset is marked as "Mislabeled" or "Correctly labelled" and use one or multiple models to predict the marks.
         - Cross-validation
       - Noise Tolerance (Robust algorithm handling overfitting)

  * Reference:
    - https://longjp.github.io/statcomp/projects/mislabeled.pdf
#### Feature scaling and standardization
   1. When to scale features?
      1. Algorithms with gradient descent algorithm
         * Machine learning algorithms like linear regression, logistic regression, neural network, etc. that use gradient descent as an optimization technique require data to be scaled. Because the scale of the variable will affect the step size of the gradient descent.
      2. Distance-Based Algorithms; BUT not tree based models, Naive Bayes, Linear Discriminant Analysis
         * Distance algorithms like KNN, K-means, PCA and SVM are most affected by the range of features. This is because behind the scenes they are using distances between data points to determine their similarity. If not scale, the feature with a higher value range starts dominating when calculating distances

      1. Normalization(MinMax):
         1. $\frac{X - Min}{Max - Min}$
      2. Normalization(Quantile)
      3. Standardization: The Standard Scaler assumes data is normally distributed within each feature and scales them such that the distribution centered around 0, with a standard deviation of 1.
   2. When to use standardization and when to use normalization?
      1. Normalization will remove the outliers. If the outliers are important, then don't use
      2. Standardization can be used when the data distribution is gaussian


#### Dimension reduction
  1. PCA
     1. How PCA works
        1. PCA is a dimension reduction technique that transform a high-dimensional dataset to a low dimensional dataset. The dimensions in the transformed dataset will have the maximum variance explained in the first dimension and second maximum in the second dimension. This makes the variance and covariance between features represented in more efficient way.
        2. It is **NOT** feature selection because it tweaked the data and condense the features into principle components, in this sense, we are not selecting a subset of features
        3. Technically, this is achieved by eigen decomposition. PCA is conducted based on the **covariance matrix** of the original dataset. and we find the top eigen value of that covariance matrix. Check [singular decompensation](https://guzintamath.com/textsavvy/2018/05/26/eigenvalues-and-eigenvectors/)
           1. Basically each squared matrix can be considered as a transformation for a vector, stretching or spinning the vector somehow. There will be exist vectors that don’t change direction as a result of the transformation—either staying the same or just getting scaled. That is, eigen vectors. They are vectors $(r_1, r_2)$ , such that
           $$
           \begin{bmatrix}\mathtt{2} & \mathtt{-3}\\\mathtt{0} & \mathtt{\,\,\,\,5}\end{bmatrix}\begin{bmatrix}\mathtt{r_1}\\\mathtt{r_2}\end{bmatrix} = \mathtt{\lambda}\begin{bmatrix}\mathtt{r_1}\\\mathtt{r_2}\end{bmatrix}
           $$
           2. The eigenvalue tells us that any vector of this form (eigen vector) will be stretched by a factor of $\lambda$ in the transformation.
           3. PCA apply eigen-decomposition to the covariance matrix of the feature and get the corresponding eigen vectors and eigen values.

     2. Why standardization is required before a PCA
        1.
     3. How to conduct PCA to categorical variables
        1. PCA cannot be applied to nominal variable because nominal variables does not have meaningful variance and covariance structure.
        2. PCA either not works for non-linear data.
     3. Pros and cons
        1. Pros:
           * Dimension reduction &rarr; less computing, less data storage
           * Remove collinearity
        2. Cons:
           * lose information
           * Makes feature hard to interpret
           * Not working well for non-linear data
     4. Usage and evaluation:
        1. Standardization is **needed** if some features are measured at different scales
        2. A dimensionality reduction algorithm performs well if it eliminates a lot of dimensions from the dataset without losing too much information.

  2. Reference:
       * https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643
       * https://guzintamath.com/textsavvy/2019/02/02/eigenvalue-decomposition/

#### Feature selection
  1. Filter methods:
     1. Definition: Filter techniques assess the relevance of features by looking only at the intrinsic properties of the data. In most cases a feature relevance score is calculated, and low-scoring features are removed.
     2. Method examples
       1. Fisher's score
       2. Chi-square
       3. Information gain
       4. Correlation
       5. Variance Threshold: features with higher variance had a higher level contains more useful information
     3. Pros and cons
        1. Pros: independent of classifier, computationally cheap
        2. Cons: ignore the interaction with the classifiers; features lacks interactions with each other
  2. Wrapper Methods
      * Definition: Wrappers require some method to search the space of all possible subsets of features, assessing their quality by learning and evaluating a classifier with each feature subset (e.g., Naive Bayes or SVM).
      * method examples:
        1. Forward feature select
        2. Backward selection
        3. Recursive feature elimination
            1. The estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute.
            2. The least important features are pruned from the current set of features.
     Logistic regression feature extraction
  3. Embedded: In embedded techniques, the feature selection algorithm is integrated as part of the learning algorithm.
     1. Method examples
       1. LASSO regularization
          1. Regularization added penalty to parameters and L1 regularization can shrink the parameters of unimportant features to 0
       2. Random forest importance
          1. The importance is calculated using the decrease in impurity measurement (Gini)
       3. Choose between Lasso and random forest in feature selection: Lasso works for linear function and thus if want to uncover the linear relationship between features with response variables, then Lasso works; otherwise random forest works.
  4. Hybrid method


#### Feature engineering:
   1. Feature encoding
       1. Identify types of categorical data
           1. Nominal: Country, gender, age, product type, seller, manufacturer
           2. Ordinal: low, medium and high,
           3. Continuous
       2. Encoding to dummy variable
           1. Label encoding: change every categorical data to a number.
           2. One hot encoding: takes each category value and turns it into a binary vector of size |i|(number of values in category i) where all columns are equal to zero besides the category column.
       3. Sparse categorical variables
          1. Frequency encoding
          2. Target encoding
       4. Feature hashing: represent categories in a “one hot encoding style” as a sparse matrix but with a much lower dimensions.

       5. Word embeddings
          1. Using COBW (Continuous Bag of Words) and SKIP-gram, word embedding encode the meaning
#### Feature colinearity

#### Data leakage



### Training procedure

#### Gradient descent
* Batch gradient descent
  - Parameters are updated after computing the gradient of the error with respect to the entire training set
* Sochastic gradient descent
  - The parameters is updated after computing the gradient of error with respect to single sample
  - It makes smooth updates in the model parameters
* Mini-batch gradient descent
  - Parameters are updated after computing the gradient of the error with respect to a **randomly sampled** subset of the training set
  - The errors will be accumulated/summed for each mini-batch and get derivatives of loss function with respect to parameters (see colab example [here]())


![img](https://media.geeksforgeeks.org/wp-content/uploads/20220615041457/resizedImage-660x187.png)

Reference:
https://kenndanielso.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_6_Stochastic_and_minibatch_gradient_descent.html





### 项目经验类
1. 训练好的模型在现实中不work,问你可能的原因
2. Loss趋于Inf或者NaN的可能的原因
3. 生产和开发时候data发生了一些shift应该如何detect和补救
4. annotation有限的情況下你要怎麼Train model
5. 假设有个model要放production了但是发现online one important feature missing 不能重新train model 你怎么办
6. How to determine which models to use?
Recommendation system/search engine











### Learning material
[Course](https://dasepli.github.io/nndl-materials/)\
[Additional ML questions](https://analyticsarora.com/quickly-master-l1-vs-l2-regularization-ml-interview-qa/)
