######### Implement Logistic regression from the scratch ##############
# Loss function: L = -1/N · Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] where y = true labels, ŷ = predicted probabilities
# Linear: z = W*X + b
# Sigmoid: y_pred = 1 / (1 + exp(-z))

# Gradient of the loss function: dL/dw, dL/db

# 1. Get dL/dw
# dL/dw = dL/dy_pred * dy_pred/dz * dz/dw
# dL/dy_pred = --1/N · [y/y_pred - (1-y)/(1-y_pred)] (derivative of the loss function)
# dy_pred/dz = y_pred * (1 - y_pred) (derivative of the sigmoid function: The derivative of the sigmoid function, σ(x) = 1 / (1 + e^(-x)), is given by σ(x) * (1 - σ(x)))
# dz/dw = X
# dL/dw = -1/N * sum(y_true/y_pred - (1 - y_true)/(1 - y_pred)) * y_pred * (1 - y_pred) * X
# (turn to vectorization)dL/dw = -1/N * (y_true - y_pred) dot X      # (y_pred and y_true: Both are vectors containing all N samples' values)

# 2. Get dL/db
# dL/db = dL/dy_pred * dy_pred/dz * dz/db
# dz/db = 1
# dL/db = -1/N * sum(y_true - y_pred)   # this is scaler value so sum is needed



from math import exp
import numpy as np
class LogisitRegression:

    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.W = None  # shape (n_features,)
        self.b = None  # shape (1,)
    
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z)) # formula: 

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        for i in range(self.n_iter):
            z = X.dot(self.W) + self.b # shape (n_samples,)
            y_pred = self.__sigmoid(z)

            # Compute gradients
            dw = (1/n_samples) * X.T.dot(y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.W -= self.lr * dw
            self.b -= self.lr * db
        print("Traning completed")

    def predict(self, X):
        z = X.dot(self.W) + self.b
        y_pred = self.__sigmoid(z)
        return
        

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000,
                               n_features=5,
                               n_classes=2,
                               random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    my_lr = LogisitRegression(n_iter = 100, lr = 0.0001)
    my_lr.fit(X_train, y_train)
    y_pred = my_lr.predict(X_test)

