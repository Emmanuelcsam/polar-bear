# tests/test_clustering.py
import unittest
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.clustering import cluster_defects

class TestClustering(unittest.TestCase):

    def setUp(self):
        """Set up a dummy DataFrame for testing."""
        data = {
            'area_px': [100, 110, 105, 500, 510, 505],
            'aspect_ratio': [1.2, 1.1, 1.3, 5.0, 5.2, 4.9],
            'solidity': [0.9, 0.95, 0.92, 0.5, 0.55, 0.52],
            'mean_intensity': [128, 130, 125, 50, 55, 52],
            'contrast': [10, 12, 11, 100, 105, 102],
            'dissimilarity': [1, 1.1, 1.05, 10, 10.5, 10.2],
            'homogeneity': [0.8, 0.82, 0.81, 0.3, 0.32, 0.31],
            'energy': [0.1, 0.11, 0.105, 0.01, 0.011, 0.0105],
            'correlation': [0.9, 0.91, 0.905, 0.2, 0.21, 0.205]
        }
        self.features_df = pd.DataFrame(data)
        self.config = {'num_clusters': 2}

    def test_cluster_defects(self):
        """Test the cluster_defects function."""
        clustered_df = cluster_defects(self.features_df, self.config)
        
        self.assertIn('cluster', clustered_df.columns)
        self.assertEqual(len(clustered_df['cluster'].unique()), 2)

if __name__ == '__main__':
    unittest.main()
