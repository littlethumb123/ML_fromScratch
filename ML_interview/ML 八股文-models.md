### KNN model
KNN is a distance-based algorithm
* Algorithm:
  * Determine the $k$ neighbors
  * Predict the class of a given data based on the existing data points
    * Calculate the total distances between the X observable and all the data points.
    * Sort the distance in an increasing order
    * Select the top $k$ points
  * Assign labels to the given point.
    * If KNN is used for regression tasks, the predictions will be based on the mean or median of the K closest observations.
    * If KNN is used for classification purposes, the mode of the closest observations will serve for prediction.
* Pros and cons
  - Pros:
    - Easy to understand and implement
    - No training involved. So called lazy learning.
    - Interpretability
    - Naturally handle the multi-classes
  - Cons
    - Sensitive to outliers, Scalability is needed before use
    - Curse of dimensionality: perform poorly on high dimensional datasets
    - Perform poorly on imbalanced dataset
### Linear Regression:
1. Assumption.
   1. Dependent y and independent variables follow the linear forms, or $y = \theta^T + \epsilon$
   2. The error term is independently and identically distributed. That is, there is no correlation between the errors $\epsilon$ with the dependent variable.
   3. (optional) The error term follows gaussian distribution. This is not a must but can make OLS as the most efficient estimator
2. Cost function
   1. Ordinary least square.
      1. m is the feature number and n is the instance number
   $$
   h(x) = \sum^m_{j=0}\theta_j x_j = \theta^Tx\\
   J(\theta) = \sum^n_{i=1}(h(x^{(i)}) - y^{(i)})^2
   $$

   2. Using MLE estimation to proof that the OLS is equivalent to the loss function derived from those three assumptions
   $$
    p(\epsilon) = \frac{1}{\sigma\sqrt{2\pi}}\exp(\frac{-\epsilon^2}{2\sigma^2})
   $$
3. Multicolinearity
   1. Result of collinearity
      1. It affects the reliability of the estimation of the beta coefficient. The standard errors of the regression coefficient tended to be large. That is the estimate of beta coefficient can swing widely.
      2. It reduces the precision of estimated coefficient
      3. However, it the collinearity does not affect the how well the model fits/predicted outcomes
   2. How to test it: variance inflation factors
   3. How to solve it
      1. colinearity only affected the correlated variables, so don't have to resolve it if the interested variable is ok
      2. Drop or combine correlated variables
      3. Use PCA to reduce uncessary dimensions.
   4. How the number $k$ affect the model?
4. explain regression coefficient
5. what is the relationship between minimizing squared error and maximizing the likelihood
5. How could you minimize the inter-correlation between variables with Linear Regression?
   1. L1 Regularization
6. if the relationship between y and x is no linear, can linear regression solve that
7. why use interaction variables

8. Two aspects to understand OLS regression
   1. Ordinary least square perspective: this is a optimization method
   2. probability perspective: MLE. This is based on three basic statistic assumptions, mentioned in the assumptions


### Logistic Regression
1. Assumption
    1. Outcome is binary variable and follows Bernoulli distribution
    2. Linear relationship between the logit of the outcome variables and each predictor variables
    3.
2. Prediction formula and decision boundary
   1. Hypothesis Formula
       * 2. The LR assumes the probability of y given x follows bernoulli distribution. It also assumes a linear relationship between dependent variables with logit of the probability. $h_\theta(x)$ is the probability of a given x has an output of 1, given the parameter $\theta$
       * Sigmoid function $\frac{1}{1 + e^{-z}}$
         - This function convert a linear data to non-linear (0,1)
         - Pros and cons:
       * Softmax for multiple classification problem
      $$
        h(x) = P(y=1|x; \theta) = g(\theta^Tx)\\
        g(z) = \frac{1}{1 + e^{-z}}\\
        h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}
      $$
   2. Decision boundary: this is a line $\theta^Tx = 0$ and thus $h(x) = 0.5$, that is the probability of getting 1 is 0.5
3. Loss function:
   1. Cost function:
      * Why using MLE to derive parameters. Because if pass the logit to the linear cost function, this will result in [non-convexfunction](https://miro.medium.com/max/2084/1*3o9_XoQP4TaceYPRZVHlxg.png).
          $$
            J(\theta) = \frac{1}{m} \sum{( \frac{1}{1 + e^{-\theta^Tx}} - y)^2}
          $$
      * logloss
        - $\text{Cost}(h_\theta(x), y) = - \log(h_\theta(x))$ if $y = 1$
        - $\text{Cost}(h_\theta(x), y) = - \log(1 - h_\theta(x))$ if $y = 0$
        - $\text{Cost}(h_\theta(x), y) = - y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x))$
        - Benefit of using logloss
          - When the difference between prediction $h_\theta(x)$ and y is different, the cost function will be infinitely large to penalize the model

          ![img](https://images.ctfassets.net/pzhspng2mvip/mS99cBYmDSfGonrteJbMW/8e77c2f6258f382caba8941fe6d15f7f/log_loss_error_on_a_single_object.png)

    2. Loss function
       $$
        J(\theta) =\frac{1}{m} \sum^n_{i=1}  [- y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x))]
       $$

    3. Optimization using MLE and gradient descent (from a MLE perspective)
       1. According to the hypothesis formula we have the probability of having an outcome as 1 or 0
          * $P(y=0|x,\theta) = h_\theta(x)$;
          * $P(y=1|x,\theta) = 1 - h_\theta(x)$
          * $P(y|x,\theta) = h_\theta(x)^y (1 - h_\theta(x))^{1-y}$
       2. The probability of all instances can be writen as a $L(\theta)$
          $$
            \begin{align}
            P(y|x;\theta) &= h_\theta(x)^y (1 - h_\theta(x))^{1-y} \\
            L(\theta)     &= \prod^n_{i=1}{P(y|x;\theta)} \\
                          &= \prod^n_{i=1}{h_\theta(x)^y (1 - h_\theta(x))^{1-y}} \\
            \log{L(\theta)} &= \log{\prod^n_{i=1}{h_\theta(x)^y (1 - h_\theta(x))^{1-y}}}\\
                          &= \sum^n_{i=1}{[- y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x))]}
            \end{align}
          $$
       3. Gradient descent

            ![img](https://img2020.cnblogs.com/blog/1309518/202004/1309518-20200430102603245-740966432.png)

1. Difference with SVM
  1. Different loss function
     1. LR: logloss function
     2. SVM: hinge loss function
  2. SVM only consider the support vector and LR considers all data points
  3. Kernel function, SVM is common but LR is not common
  4. SVM depends on the computation of distance and thus needs normalization. LR does not needs normalization
  5. SVM has internal regularization (hinge loss) and logistic regression requires external regularization
1. logistic regression和svm的差别 （我想这个主要是想问两者的loss的不同以及输出的不同，一个是概率输出一个是score）
  1. SVM try to maximize the margin between the closest support vectors while LR the posterior class probability.
2. LR大部分面经集中在logloss和regularization，相关的问题在上个帖子有了这里就不重复了。


### Logistic and linear regression
* 找一个合适的预测函数（Andrew Ng的公开课中称为hypothesis），一般表示为h函数，该函数就是我们需要找的分类函数，它用来预测输入数据的判断结果。这个过程时非常关键的，需要对数据有一定的了解或分析，知道或者猜测预测函数的“大概”形式，比如是线性函数还是非线性函数。
* 构造一个Cost函数（损失函数），该函数表示预测的输出（h）与训练数据类别（y）之间的偏差，可以是二者之间的差（h-y）或者是其他的形式。综合考虑所有训练数据的“损失”，将Cost求和或者求平均，记为J(θ)函数，表示所有训练数据预测值与实际类别的偏差。
* 显然，J(θ)函数的值越小表示预测函数越准确（即h函数越准确），所以这一步需要做的是找到J(θ)函数的最小值。找函数的最小值有不同的方法，Logistic Regression实现时有的是梯度下降法（Gradient Descent）


### Clustering and EM:

#### K-means
1. K-means clustering (explain the algorithm in detail; whether it will converge, 收敛到global or local optimums;  how to stop)
   * Definition: K-means is a cluster algorithm that assigns observations to predefined centerics in a way that makes the distance between the observations with their assigned centrics maximum.
   * Mechanism:
      - K means first identify $k$ centroids $\mu^0 = (\mu_1^0... \mu_k^0)$
      - Allocates each sample point $x_j$ from N sample points $(j \subset {1,2...N})$ to a the nearest centeroid $\mu_i$ as a cluster $C_i$. The cost function now is the Eucliean distance bewteen each point with all given centeroids, getting the min value.
      - recalculate the centeroid by averaging to minimize the total squared errors betwee each training sample with their centroid.
   * Optimization function:
      - Intuition: the cluster itself should be tight and the points within this cluster should be close to each other; Meanwhile the
      - Loss function: Euclidean distance between each training sample with its centeroid\
      - Cost function
        $$
          \sum_{i\subset{K}}{\sum_{j\subset{N}}{(x_j-\mu_i)^2}}
        $$
   * Local and global optimization
      - Why k-means get stuck easily in local minima?
        - sdf
   * When to stop:
      - The centroids becomes stablized and the decrease in the objective function is lower than a threshold
      - Reach the defined number of iterations
   * Advantage and disadvantage
     1. Advantage
        - Scalability
     2. Disadvantage
       - need to predefine k
       - different initial points will result in different clusters &rarr; pick different initial points to get start.
       - sensitive to outliers: the presence of outliers can distort the clustering outcomes.
       - sensitive to scale: need standardization before doing kn means
       - stuck in local monimim &rarr; Nearby points may not end up in the same cluster
       - Curse of dimensionality
       - not fit to imbalanced dataset

2. How to determine the number of $k$ clusters
   * Elbow method
     - Sum of squared distance between points and their assigned centroid
     - run k means on the dataset for a range of values of k and calculate the sum of distance for all $k$ points
   * Silhouette Analysis
     - For a given cluster sample, Compute the average distance from all data points in the same cluster $\alpha_i$
     - Compute the average distance from all data points in the closest cluster $\beta_i$
     - Calculate $\frac{\beta_i - \alpha_i}{\max{(\alpha_i, \beta_i)}}$
   * DunnIndex
     - The numerator of the above function measures the maximum distance between every two points $(x_i, x_j)$ belonging to two different clusters. This represents the intracluster distance.
     - The denominator of the above function measures the maximum distance between every two points $(y_i, y_j)$ belonging to the same cluster. This represents the intercluster distance.
      ![img](https://miro.medium.com/max/612/1*Ml1cUinYf_H2jBqJq77hFg.png)
3. How to improve the k-means
   * Preprocessing
     - standardization and normalization PCA
     - outlier detection
   * Choose the right $k$ value: see point 2
   * Use Kmeans ++:
     - Kmeans++ choose the farthest point from the current centroid as the next centroid and repeat previous steps until find K centroid
   * Select multiple set of initial centeroids to avoid local optimum and select the one with lowest cost function

4. The difference between k-means with KNN
   1. K-means is an unsupervised learning algorithm and KNN is a supervised algorithm





5. How to deal with outliers


#### EM




4. GMM是什么，和Kmeans的关系
5. Curse of dimensionality:
   1. Definition: As the features dimensions increase, the
   2. Impacts on algorihtm
      Generally, as dimensions go up, it is harder to calculate the distance between samples.
      1. KNN: as dimension increases, the observations become very far from each other and cannot help classify new observations.
      2. K-means: the distance bewteen observations with the centeric point converge to a constant value.
### Decision Tree
1. Basic algorithm
   1. CART tree
      * CART分类树在每次分枝时，是穷举每一个feature的每一个阈值，根据GINI系数找到使不纯性降低最大的的feature以及其阀值，然后按照feature<=阈值，和feature>阈值分成的两个分枝，每个分支包含符合分支条件的样本。用同样方法继续分枝直到该分支下的所有样本都属于统一类别，或达到预设的终止条件，若最终叶子节点中的类别不唯一，则以多数人的类别作为该叶子节点的性别。回归树总体流程也是类似，不过在每个节点（不一定是叶子节点）都会得一个预测值，以年龄为例，该预测值等于属于这个节点的所有人年龄的平均值。分枝时穷举每一个feature的每个阈值找最好的分割点，但衡量最好的标准不再是GINI系数，而是最小化均方差--即（每个人的年龄-预测年龄）^2 的总和 / N，或者说是每个人的预测误差平方和 除以
      * Regression:
        - Least Square Deviation (LSD)
   2. C4.5
   3. ID3
   4.
2. How regression/classification DT split nodes?
   The original CART algorithm uses Gini impurity as the splitting criterion; The later ID3, C4.5, and C5.0 use entropy.
   1. Classification: information gain
      1. binary classification
      2. multi-classification
   2. Regression tree:
      1. Sum of Squared Error (SSE)
2. What is the difference between gini index vs. entropy when to use which?
   1. The difference between gini index and entropy in formula and diagram
   2. gini index is normally used for computational efficiency.
3. How to split the featuer with numerical values?
   1. Sort the value as well as the true classification
   2. each numerical value will be the threshold to split the data at the node; and the one with maximum information gain will be chosen for that node
   3. This is why random forest is slow when features are many. The computational cost will be huge
4. When the decision tree stops growing?
   1. The minimum number of data points in each node is reached
   2. The predefine depth of decision tree is reached
5. How to prevent overfitting in DT?
   1. **Control leaf size**:
      * The max depth of the tree: The maximum depth of the tree.
      * THe min sample required to split in an internal node: Minimum split size is a limit to stop the further splitting of nodes when the number of observations in the node is lower the minimum split size.
   2. Pruning
      1. Pre-pruning and post pruning

6. Imbalanced dataset
   1. Decision tree is very sensitive to imbalanced dataset because Gini index or Entropy are maximized when the classes in a node are perfectly balanced.
   2. Solution: weighted Gini, weighted entropy
7. Missing value
   Decision Tree can automatically handle missing values.

7 pros and cons
  * Pros:
    - Scale invariant
    - Robust to irrelevant features
    - Robust to missing value
    - Interpretable
    - Robust to outliers
  * Cons
    - Tend to overfitting
    - Sensitive to inbalanced dataset
### Ensemble Learning
#### Bagging
* Bagging chooses a random sample from the data set with **replacement**. Hence each model is generated from the samples (Bootstrap Samples) provided by the Original Data, known as row sampling.
* Sample with replacement makes sure the data trained on each tree was independent to each other. With this sampling method; around 36% data instances will not be sampled bec
* Each model is trained independently with a sample and the final result is based on majority voting.
* Challenge:
  - The trees would become similar because each tree use the same set of features. An important feature will be critical in every tree so that it makes trees highly correlated.

##### Random Forest

1. Definition:
   * Random forest is an implementation of bagging technology. Additionally, RF also sample the features.
2. Algorithm:
   1. Construct M CART trees
   1. Bootstrap: we generate a bootstrap sample (n) with random draw with replecement (can have duplicate data) from the entire dataset (N). Redo the bootstrap M times (because M tree, each tree has one bootstrap)
   3. For each tree training, select a random number of the features (classification $\sqrt{feature}$ and regression) $\frac{feature}{3}$) WITHOUT replacement for splitting each node
   4. Trained tree with the trainset and test it with test set
   3. Each tree generate an output and the final output is derived from all trees.
      1. Classification - majority voting
      2. Regression - average
3. tree construction criteria
   1. Node impurity criteria
   ![img](https://miro.medium.com/max/1400/1*eES0Bh8jTB73P3ad_U2aCA.png)

4. OOB errors
   1.  consider that our training set has n samples. Then, the probability of selecting one particular sample from the training set is $\frac{1}{n}$. Similarly, the probability of not selecting one particular sample is $1-\frac{1}{n}$. Since we select the bootstrap samples with replacement, the probability of one particular sample not being selected n times is equal to $(1-\frac{1}{n})^{n}$. Now, if the number n is pretty big or if it tends to infinity, we’ll get a limit below:
    ![img](https://www.baeldung.com/wp-content/ql-cache/quicklatex.com-cb3fa3b715a11452b45d7124d68d443e_l3.svg)
  2. The OOB errors will be the average of the test errors or performance of all the trees.
4. Feature importance calculation
   * Calculate the feature importance with permutation
     -
   * Calculate the node importance of each tree for each node
     - the node importance on each node is calculated as the decrease in node impurity weighted by the probability of reaching that node. The node probability can be calculated by the number of samples that reach the node, divided by the total number of samples.
       $$
          \text{ni}_j = w_jC_j - w_\text{left(j)}C_\text{left(j)} - w_\text{right(j)}C_\text{right(j)}
       $$
          * $\text{ni}_j$ = the importance of node j
          * $w_j$ = weighted number of samples reaching node j
          * $C_j$ = the impurity value of node j
          * $\text{left(j)}$ = child node from left split on node j
          * $\text{right(j)}$ = child node from right split on node j
   * Calculate the feature importance of each tree for each feature based on node importance
     - There can be multiple nodes split on a single feature, thus use a weighted importance for each feature on a particular tree
       $$
          {f_i}_j = \frac{\sum_{\text{j: node j splits on feature i}}ni_j}{\sum_{k\subset \text{all nodes}} ni_k}
       $$
          * i refers to the $i_{th}$ feature
          * j refers to the $j_{th}$ tree
          * ${f_i}_j$ refers to the feature $i$ on the $j$ tree.
   * Calculate the feature importance across all trees by averaging the $f_i$ at the random forest level
     - These can then be normalized to a value between 0 and 1 by dividing by the sum of all feature importance values
       $$
          f_i = \frac{\sum_{\text{j: all trees}}{f_i}_j}{T}
       $$
5. Pros and Cons
   1. Pros:
      1. Large dataset
      2. Work for high-dimensional feature (no need to feature reduction)
      2. Low variance
      2. No need to use cross validation
      3. No need for feature selection
   2. Cons
      1. Computationally cost
      2. lack of interpretability; but feature importance helps
      3. Random forest is not
6. Missing value with random forest
   1. The random forest with CART tree has missing value handling mechanism
      1. RF trees handles missing values either by imputation with average, either by rough average/mode, either by an averaging/mode based on proximities.

7. Unbalance dataset with random forest
   1.

   例如，在模型的优缺点中，我们提到了随机森林可以对变量重要性进行排序。相应地，我们应该能够解释随机森林是如何对变量重要性进行排序，有哪几种常见的排序指标，比如利用OOB误分率的改变或者分裂时信息增益的变化等。当然，问题并非到此终止，基于上面提到的两种常见的变量重要性排序指标，又可以衍生出新的问题。例如，针对OOB误分率这个指标，解释一下什么是OOB，随机森林中OOB是如何计算的，它有什么样的优缺点；针对信息增益，同样会有很多与之有关的问题，如什么是信息增益，如何计算信息增益，什么是熵，什么是GINI指数，他们之间的区别是什么，他们之间的区别会对建树产生怎样的影响等。
   再如，在垃圾邮件分类器项目中，有一部分数据存在缺失，而随机森林具有处理缺失数据的优点，建模的过程中我们充分利用了这一特性。那么，与之相关的问题可能会是，随机森林为什么会有这个优点？随机森林是怎样对缺失数据进行训练及预测？



1. difference between bagging and boosting
2. gbdt和random forest 区别，pros and cons
3. explain gbdt/random forest
4. will random forest help reduce bias or variance/why random forest can help reduce variance
5. Compare boosting approach
6. How random forest help feature selection. What are the disadvantage of using that for feature selection?
   1.



##### Adaboosting
 Adjust the weight of each sample in each round




#### Boosting tree

* Boosting starts from a weak learner and select the best estimator from the rest of trees
Learn F(x) as sum of M **weak learns**
$$F(x) = \sum{f_i(x)}$$
* Use residual




##### Gradient boosting decision tree
* A series of weak learners are fit sequentially to improve the error of previous weak learner. Differently from the boosting tree, GBDT uses gradient as the residual
* Assumption:
* Loss function:
    * Regression
    * Binary classification
    * mulshould be **Differentiable**
* Algorithm
  - Create a weak model which output is the average of the y value
  -
* Parameters: number of trees, depth of trees and learning rate
* Pros:
  - Deal with unbalanced data
    - How?
  - Flexible in objective function (But need to be differentiable)
* Cons:
  - Gradient boosting tree are more sensitive to overfitting and impacted by noisy
  - Computing inefficiency and take longer time to train
  - Poor interpretability


* Difference between adaboosting vs gradient boosting



##### Stochastic gradient Boosting



##### XGboosting

##### **GBDT vs Adaboost**

* Similarity
  - Both use boosting method to build weak learners in a sequential fashion
  - gradient boosting is a special case of Adaboost.
* Difference
  - What is learnt from previous learner is difference
    - Adaboost: adaptive boosting changes the sample distribution by modifying the weights attached to each of the instances. So that incorrectly classified instances are given more attention in the next round
    - GBDT: The residual is learnt from the previous learner and the next learner tries to predict the residual, rather than original value
  - Tree structure
    - Adaboost: decision stump and each stump has its own weight based on performance
    - GBDT: only regression tree as the weak learner. use sigmoid and softmax function to output classification output
  - Loss function
    - Adaboost: Exponential loss, there is no learning rate
    - GBDT: Gradient descent and thus learning rate can be specified.





##### Random forest vs GBDT
* Similarity:
  - Both use ensemble method to build individual estimators to give a final decision
1. Bagging
   1. Bootstrapped samples
   2. Base tree created independently
   3. Each data points are considered
   4. Avoid overfitting and reduce variance
2. Boosting
   1. Fit entire dataset (GBDT can also use bootstrap to sample data and features)
   2. Base tree created successively
   3. Model the residuals, rather than each data y
   4. Boosting is possible to overfitting

* Difference
  - How estimators are constructed
    - RF:
      * bagging parallel; the models are independent of each other.
      * Tree can be deep.
      * Individual tree has a low bias but high variance
      * The estimator can be both regression tree and classification tree
    - GBDT:
      * boosting uses sequential decision trees to model the residual of the previous trees
      * Tree is a weak learner (normally fewer than 5 layers)
      * Individual tree has a high bias but low variance (because it is weak learner)
      * GBDT can be only regression tree.

- Other operations:
   - Handling missing data

      - GB: Gradient Boosting Trees uses CART trees. CART handles missing values either by imputation with average, either by rough average/mode, either by an averaging/mode based on proximities.
      - RF: CART trees are also used in RF but it can have other types of tree, such as C4.5.
      Both models need long time to train


### Generative Model
1. 和Discrimitive模型比起来，Generative 更容易overfitting还是underfitting
2. Naïve Bayes model
   1. What is Naive Bayes:
      1. The bayesian method estimates the joint probability for the target variable $y$ and features $x_1, x_2, \dots, x_m$ and then classifies by choosing the classes that got the greatest probability.
      2. Assumption: The features are independent to each other.
   2. Type of naive bayesian model
      1. Bernoulli:
      2. Multinomial:
      3. Gaussian
      4. Multiclass
   3. Pros and cons:
      1. pros
        1. If the assumption holds, it requires less data and give better results.
        2. simple and easy to implement;
        3. it didn't go through an optimziation process so fast to make real time prediction
      2. cons:
         1. zero probability problem: some features in testing data might be missign in training data and thus leads to zero probability
         2. simple assumptions which is not the case
   4.
3. LDA/QDA是什么，假设是什么
### 其他模型
1. SVM
  1. Explain SVM, 如何引入非线性
     * SVM uses support vectors to find out a hyperplane in N dimensional space to maximize the distance between two classifications. '
     * **Support vector** is the instances located on the margin of the hyperplane, to be used to find out the optimum decision boundary. Other instances have no impacts on the decision boundary
     * It can be used for both classification and regression problems. In classification, it can solve both linear and non-linear classification
     *
  2. Explain kernel methods, why to use
     *
  3.

4. what kernels do you know
5. 怎么把SVM的output按照概率输出
6. Explain KNN
### !所有模型的pros and cons （最高频的一个问题）
