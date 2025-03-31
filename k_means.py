import numpy as np
class kMeans():
    """
    Algorithms 
    1. set k centroids within the dataset (the data points are already on the plane)
    2. calculate distance bewteen the points with each centroid and color each one to that centroid
    3. Update the centroid of each contempory cluster
    4. iterate the step 1, 2, 3, to make the assignment of cluster stable

    assume data = {(x1, y1),... (xn, yn)}

    
    calculate distance alternatives:https://towardsdatascience.com/log-book-guide-to-distance-measuring-approaches-for-k-means-clustering-f137807e8e21
    """
    def __init__(self, 
                 k = 3, 
                 iterations=300,
                 distance_method = "euclidean",
                 tolerance = 1e-2,
                 random_state=0):
        """
        k: topic number
        _labels: the final assigned label
        iterations: how many times the centroids are calculated
        distance_method: default is "euclidean" method
        tolerance: convergence threshold
        random_state: value for random.seed(), make sure the resutls are replicable

        """
        self.k = k
        self._labels = None
        self.iterations = iterations
        self.distance_method = distance_method
        self.tolerance = tolerance
        self.centroids = None
        self.random_state = random_state
        
        
    def _distance(self, arr1, arr2):
        """
        calculate distance between two arrays, default method is euclidean distance
        return distance values
        """
        if self.distance_method == 'manhattan':
            return np.sum(arr1 - arr2)

        return np.linalg.norm(arr1 - arr2)
    
    
    def _init_cent(self, X):
        
        """
        centroids initialization strategy: 
        random sample points from the dataset as the initial centroids
        
        param:
            X: (sample_size, feature_size)        
        return: 
            centroids 2d array (k, feature_size) 
        """
        np.random.seed(self.random_state)
        X = np.array(X)
        sample_size = X.shape[0]
        # get k random sample points from the dataset as the initial centroids
        random_index = np.random.choice(sample_size, self.k, replace=False)
        centroids = X[random_index]
        return centroids
    
def _init_cent_plus_plus(self, X):
    """
    K-means++ initialization strategy:
    1. Choose first centroid randomly
    2. Choose subsequent centroids with probability proportional to distance²
    3. The goal is to spread out the initial centroids, reducing the chance of poor clustering
    
    param:
        X: (sample_size, feature_size)        
    return: 
        centroids 2d array (k, feature_size) 
    """
    np.random.seed(self.random_state)
    X = np.array(X)
    n_samples, n_features = X.shape
    centroids = np.zeros((self.k, n_features))
    
    # Choose first centroid randomly
    first_centroid_idx = np.random.choice(n_samples)
    centroids[0] = X[first_centroid_idx]
    
    # Choose remaining centroids
    for c in range(1, self.k):
        # Calculate distance to closest centroid for each point
        min_distances = np.zeros(n_samples)
        for i in range(n_samples):
            # Find distance to closest existing centroid
            point_distances = [self._distance(X[i], centroids[j]) for j in range(c)]
            min_distances[i] = min(point_distances)
            
        # Square distances to increase weight for distant points
        weights = min_distances ** 2
        
        # Choose next centroid with probability proportional to distance²
        next_centroid_idx = np.random.choice(
            n_samples,
            p=weights/np.sum(weights)
        )
        centroids[c] = X[next_centroid_idx]
    
    return centroids
    
    def assign_label_data(self, X, centroids):
        
        """
        Assign labels to each data point based on distance
        param:
            X: (sample_size, feature_size)
            centroids: (k, feature_size)
        
        return: 
            the label_index array (sample_size, 1)
        """
        # Calculate the distance between each sample with teh centroid
        X = np.array(X)
        centroids = np.array(centroids)
        labels = np.zeros((X.shape[0], 1))
        # iterate each sample point and calculate the distance with each centroid
        for i in range(X.shape[0]):
            i_distances = []
            for j in range(centroids.shape[0]):
                i_distances.append(self._distance(X[i, :], centroids[j, :]))
            # get the index of the closest centroid
            labels[i] = np.argmin(i_distances)
        return labels

            
    def reassign_centroids(self, X, label):
        """
        calculate the new centroid for each cluster by average all x and y of each point respectively
        x_new = mean(x_i)
        y_new = mean(y_i)
        param:
            X (sample_size, feature_size)
            label (sample_size, 1(cluster_index))
        return: 
            updated centroids array, the same shape to centroids (k, feature_size)
        """
        X = np.array(X)
        label = np.array(label)
        new_centroids = []
        # calculate the new centroid based on the X and labels
        for l in np.unique(label):
            new_centroids.append(X[label.flatten() == l].mean(axis = 0))
        return np.array(new_centroids)
        
    
    def _converge(self, new_centroids, pre_centroids):
        return np.sum((new_centroids - pre_centroids)**2) < self.tolerance
        
        
    def fit(self, X):
        """
        1. assess if algorithm converge?
        2. assign label to dataset for each iteration
        3. after the assignment, recalculate the centroid of each cluster
        
        
        """
        centroids = self._init_cent(X)
        labels = None
        
        for i in range(self.iterations):
            
            pre_centroids = centroids
            
            # assign label based on the distance
            labels = self.assign_label_data(X, centroids)
            
            # update centroid of each cluster
            centroids = self.reassign_centroids(X, labels)
            
            # when the move distance of each centroid is smaller than tolerance, then stop the iterations
            if self._converge(centroids, pre_centroids):
                break        
        
        self._labels = labels
        self.centroids = centroids

def main():
  X = np.array([[1, 2], 
                [1, 4], 
                [1, 0],
                [3, 12], 
                [3, 14], 
                [3, 10],
                [10, 2], 
                [10, 4], 
                [10, 0]])
  km = kMeans(k=3)
  km.fit(X)
  print(km._labels)
  # output: 
    # array([
    #     [2],
    #     [2],
    #     [2],
    #     [0],
    #     [0],
    #     [0],
    #     [1],
    #     [1],
    #     [1]], dtype=int64)

if __name__ == "__main__":
    main()



