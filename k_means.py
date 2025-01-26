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
        centroids initialization strategy: random sample points between the min and max value in each feature the training dataset
        
        param:
            X: (sample_size, feature_size)        
        return: 
            centroids 2d array (k, feature_size) 
        """
        np.random.seed(self.random_state)
        maxmin_range = X.max(axis=0) - X.min(axis=0)
        
        # Here check Data validity
        return np.array([X.min(axis=0) + maxmin_range*np.random.rand(X.shape[1]) for _ in range(self.k)])
    
    def _label_data(self, X, centroids):
        
        """
        Assign labels to each data point based on distance
        param:
            X: (sample_size, feature_size)
            centroids: (k, feature_size)
        
        return: 
            the label_index array (sample_size, 1)
        """
        # calculate the distance and figure out the label for each point
        distance_of_each_cluster = []  # (sample, distance with each centroid for each sample)
        for X_i in X:
            distance_per_point = []
            for cent_i in centroids:
                distance_per_point.append(self._distance(X_i, cent_i))
            distance_of_each_cluster.append(distance_per_point)
        
        # get the index of the min distance from the distance of each cluster data set. 
        # get label_index array and convert from 1d to 2d array (sample_size, 1)
        sample_index_of_cluster = np.argmin(distance_of_each_cluster, axis=1).reshape(X.shape[0], 1)
        
        return sample_index_of_cluster
        
    
    def _reassign_centroids(self, X, label):
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
        new_centroids = []
        X_label = np.append(X, label, axis=1)
        for cluster_i in range(self.k):
            # get the mean value of feature dimension as the position of the new cluster (1, feature_size)
            # the last column [:, -1] is label while the rest [:, :-1] are the features
            new_centroids.append(np.mean(X_label[X_label[:, -1] == cluster_i, :-1], axis = 0))
        
        return np.array(new_centroids)
    
    
    def _converge(self, new_centroids, pre_centroids):
        
        difference = 0
        for new, pre in zip(new_centroids, pre_centroids):
            difference += self._distance(new, pre)
        return difference < self.tolerance
        
        
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
            labels = self._label_data(X, centroids)
            
            # update centroid of each cluster
            centroids = self._reassign_centroids(X, labels)
            
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
    # array([[2],
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



