"""Mock sklearn module for testing."""

class preprocessing:
    class StandardScaler:
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X

class cluster:
    class KMeans:
        def __init__(self, n_clusters=3, random_state=None):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.labels_ = None
            self.inertia_ = 100.0
        
        def fit(self, X):
            import numpy as np
            self.labels_ = np.random.randint(0, self.n_clusters, len(X))
            return self
        
        def predict(self, X):
            import numpy as np
            return np.random.randint(0, self.n_clusters, len(X))
