########## Implement linear regression from the scratch ################

# Loss function: L = 1/N * Σ(y_true - y_pred)^2
# Linear: y_pred = W * X + b (W: (feature_dim, ), X: (n_samples, feature_dim), b: (1,))
# Gradient of the loss function: dL/dw, dL/db
# dL/dw = dL/dy_pred * dy_pred/dw
# dL/dy_pred = -2/N * (y_true - y_pred) (derivative of the loss function)
# dy_pred/dw = X (derivative of the linear function)
# dL/dw = -2/N * X.T dot (y_true - y_pred) # (y_pred, y_true: (n_samples, 1), X: (n_samples, feature_dim))
# dL/db = -2/N * np.sum(y_true - y_pred)

import numpy as np

class LinearRegression():
    def __init__(self, iter = 100, lr = 0.001):
        self.iter = iter
        self.lr = lr
        self.W = None   # Shape (n_features, )
        self.b = None   # Shape (1, )
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0
        for _ in range(self.iter):
            y_pred = X.dot(self.W) + self.b
            # Correct gradient calculation
            dW = -(2/n_samples) * X.T.dot(y - y_pred)
            db = -(2/n_samples) * np.sum(y - y_pred)
            # Update parameters (subtract because we want to minimize)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return X.dot(self.W) + self.b  # y_pred shape (n_samples, )
    

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression(iter=1000, lr=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error: ", mse)
    print("Weights: ", model.W)
    print("Bias: ", model.b)
    plt.scatter(y_test, predictions)
    plt.xlabel("True Values")   
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values")
    plt.show()


