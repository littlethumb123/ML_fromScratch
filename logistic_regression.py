######### Implement Logistic regression from the scratch ##############
# Loss function: L = -1/N * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
# Linear: z = W*X + b
# Sigmoid: y_pred = 1 / (1 + exp(-z))

# Gradient of the loss function: dL/dw, dL/db

# 1. Get dL/dw
# dL/dw = dL/dy_pred * dy_pred/dz * dz/dw
# dL/dy_pred = -1/N * sum(y_true/y_pred - (1 - y_true)/(1 - y_pred))
# dy_pred/dz = y_pred * (1 - y_pred) (derivative of the sigmoid function)
# dz/dw = X
# dL/dw = -1/N * sum(y_true/y_pred - (1 - y_true)/(1 - y_pred)) * y_pred * (1 - y_pred) * X
# dL/dw = -1/N * sum(y_true - y_pred) * X

# 2. Get dL/db
# dL/db = dL/dy_pred * dy_pred/dz * dz/db
# dz/db = 1
# dL/db = -1/N * sum(y_true - y_pred)



import numpy as np
class LogisitRegression:

    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iter):
            z = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            
            # first derivatives of the loss function
            dw = (-1/n_samples) * np.dot(X.T, (y - y_pred))
            db = (-1/n_samples) * np.sum(y - y_pred)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_pred = self.sigmoid(z)
        return [1 if i > 0.5 else 0 for i in y_pred]
        
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

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

