################# Implement KNN from the scratch ##########################

# Algorithm:
# 1. Given a training set of data points, create a space
# 2. For each given test point, calculate the distance to all training points (e.g., using Euclidean distance, Manhattan distance, etc.)
# 3. Sort the distances and select the k nearest neighbors
# 4. Assign the class label based on the majority vote of the k neighbors

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from collections import Counter
class KNN:
    def __init__(self, k, distance = 'euclidean'):
        self.k = k
        self.distance = distance
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

    def calculate_distance(self, x1, x2):
        if self.distance == 'euclidean':
            return np.linalg.norm(np.array(x2) - np.array(x1))


    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            distances = []
            for j in range(self.n_samples):
                dist = self.calculate_distance(X[i], self.X[j])
                distances.append((self.y[j], dist))
            # get the top k clostest i
            distances.sort(key = lambda x: x[1])
            neighbors = [label for label, _ in distances[:self.k]]
            print(neighbors)
            y.append(Counter(neighbors).most_common(1)[0][0])

        return np.array(y)


if __name__ == "__main__":
    # Create a sample dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=3, random_state=42)
    
    # Visualize the dataset
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.title('Toy Classification Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.show()
    
    # Create a KNN classifier
    knn = KNN(k=5, distance='euclidean')
    
    # Fit the model
    knn.fit(X, y)
    
    # Test on individual points
    test_points = np.array([[0, 0, 1], [1, 1, 2], [-4, -2, 4]])
    for point in test_points:
        prediction = knn.predict(point)
        print(f"Predicted class for {point} is {prediction}")
    
    # Test on multiple points at once
    predictions = knn.predict(test_points)
    print(f"Predictions for all test points: {predictions}")


            
