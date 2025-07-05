#!/usr/bin/env python3
"""
Comprehensive tests for clustering.py
Tests K-Means clustering functionality for defect grouping.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestClusterDefects(unittest.TestCase):
    """Test cluster_defects function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample defect features DataFrame
        self.sample_features = pd.DataFrame({
            'image_path': [f'image_{i}.png' for i in range(20)],
            'defect_id': list(range(20)),
            'area': np.random.randint(10, 100, 20),
            'perimeter': np.random.randint(20, 200, 20),
            'circularity': np.random.rand(20),
            'aspect_ratio': np.random.rand(20) * 2,
            'mean_intensity': np.random.randint(50, 200, 20),
            'std_intensity': np.random.rand(20) * 50,
            'contrast': np.random.rand(20),
            'homogeneity': np.random.rand(20),
            'energy': np.random.rand(20),
            'correlation': np.random.rand(20),
            'zone': np.random.choice(['core', 'cladding', 'ferrule'], 20)
        })
        
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        from clustering import cluster_defects
        
        # Run clustering
        clustered_df, kmeans_model = cluster_defects(
            self.sample_features,
            n_clusters=3
        )
        
        # Check output DataFrame
        self.assertEqual(len(clustered_df), len(self.sample_features))
        self.assertIn('cluster', clustered_df.columns)
        self.assertEqual(clustered_df['cluster'].nunique(), 3)
        
        # Check cluster assignments
        self.assertTrue(all(0 <= c < 3 for c in clustered_df['cluster']))
        
        # Check model
        self.assertIsNotNone(kmeans_model)
        self.assertEqual(kmeans_model.n_clusters, 3)
    
    def test_automatic_cluster_selection(self):
        """Test automatic cluster number selection."""
        from clustering import cluster_defects
        
        # Run with automatic selection
        clustered_df, kmeans_model = cluster_defects(
            self.sample_features,
            n_clusters=None  # Should trigger automatic selection
        )
        
        # Check that clusters were assigned
        self.assertIn('cluster', clustered_df.columns)
        self.assertGreater(clustered_df['cluster'].nunique(), 0)
        self.assertLessEqual(clustered_df['cluster'].nunique(), 10)  # Max clusters
    
    def test_feature_selection(self):
        """Test that correct features are used for clustering."""
        from clustering import cluster_defects
        
        # Add some non-feature columns
        features_with_extra = self.sample_features.copy()
        features_with_extra['extra_col'] = 'test'
        features_with_extra['another_col'] = 123
        
        # Run clustering
        clustered_df, kmeans_model = cluster_defects(features_with_extra, n_clusters=3)
        
        # Check that extra columns are preserved
        self.assertIn('extra_col', clustered_df.columns)
        self.assertIn('another_col', clustered_df.columns)
        
        # Check that clustering worked
        self.assertIn('cluster', clustered_df.columns)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from clustering import cluster_defects
        
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        with self.assertRaises((ValueError, KeyError)):
            cluster_defects(empty_df, n_clusters=3)
    
    def test_single_sample(self):
        """Test handling of single sample."""
        from clustering import cluster_defects
        
        single_sample = self.sample_features.iloc[[0]]
        
        # Should handle single sample
        clustered_df, kmeans_model = cluster_defects(single_sample, n_clusters=1)
        
        self.assertEqual(len(clustered_df), 1)
        self.assertEqual(clustered_df['cluster'].iloc[0], 0)
    
    @patch('sklearn.preprocessing.StandardScaler')
    def test_feature_scaling(self, mock_scaler):
        """Test that features are properly scaled."""
        from clustering import cluster_defects
        
        # Mock scaler
        mock_scaler_instance = Mock()
        mock_scaler.return_value = mock_scaler_instance
        mock_scaler_instance.fit_transform.return_value = np.random.rand(20, 10)
        
        # Run clustering
        cluster_defects(self.sample_features, n_clusters=3)
        
        # Check that scaler was used
        mock_scaler_instance.fit_transform.assert_called_once()
    
    def test_cluster_consistency(self):
        """Test that clustering is consistent with same data."""
        from clustering import cluster_defects
        
        # Run clustering multiple times with same random seed
        np.random.seed(42)
        result1, model1 = cluster_defects(self.sample_features, n_clusters=3)
        
        np.random.seed(42)
        result2, model2 = cluster_defects(self.sample_features, n_clusters=3)
        
        # Results should be similar (not exactly same due to K-Means initialization)
        # But with same seed, should be very close
        similarity = (result1['cluster'] == result2['cluster']).sum() / len(result1)
        self.assertGreater(similarity, 0.8)  # At least 80% same assignments
    
    def test_elbow_method(self):
        """Test elbow method for optimal cluster selection."""
        from clustering import cluster_defects
        
        # Create data with clear clusters
        clear_clusters = pd.DataFrame()
        for i in range(3):
            cluster_data = pd.DataFrame({
                'area': np.random.normal(50 + i*50, 5, 20),
                'perimeter': np.random.normal(100 + i*50, 10, 20),
                'circularity': np.random.normal(0.5 + i*0.2, 0.05, 20),
                'aspect_ratio': np.random.normal(1.0 + i*0.3, 0.1, 20),
                'mean_intensity': np.random.normal(100 + i*30, 5, 20),
                'std_intensity': np.random.normal(20 + i*10, 2, 20),
                'contrast': np.random.normal(0.5, 0.1, 20),
                'homogeneity': np.random.normal(0.7, 0.1, 20),
                'energy': np.random.normal(0.3, 0.05, 20),
                'correlation': np.random.normal(0.8, 0.1, 20),
                'zone': 'cladding'
            })
            clear_clusters = pd.concat([clear_clusters, cluster_data])
        
        clear_clusters.reset_index(drop=True, inplace=True)
        clear_clusters['image_path'] = [f'image_{i}.png' for i in range(len(clear_clusters))]
        clear_clusters['defect_id'] = list(range(len(clear_clusters)))
        
        # Run with automatic selection
        clustered_df, model = cluster_defects(clear_clusters, n_clusters=None)
        
        # Should detect approximately 3 clusters
        n_clusters = clustered_df['cluster'].nunique()
        self.assertIn(n_clusters, [2, 3, 4])  # Allow some flexibility
    
    def test_missing_features(self):
        """Test handling of missing required features."""
        from clustering import cluster_defects
        
        # Remove required features
        incomplete_features = self.sample_features.drop(columns=['area', 'perimeter'])
        
        # Should raise error for missing features
        with self.assertRaises(KeyError):
            cluster_defects(incomplete_features, n_clusters=3)
    
    def test_output_preservation(self):
        """Test that all original data is preserved in output."""
        from clustering import cluster_defects
        
        # Add custom columns
        custom_features = self.sample_features.copy()
        custom_features['custom_field'] = 'test_value'
        custom_features['numeric_field'] = 42
        
        # Run clustering
        clustered_df, _ = cluster_defects(custom_features, n_clusters=3)
        
        # Check all original columns are preserved
        for col in custom_features.columns:
            self.assertIn(col, clustered_df.columns)
            
        # Check values are preserved
        self.assertTrue(all(clustered_df['custom_field'] == 'test_value'))
        self.assertTrue(all(clustered_df['numeric_field'] == 42))

class TestClusteringEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_nan_handling(self):
        """Test handling of NaN values in features."""
        from clustering import cluster_defects
        
        # Create features with NaN values
        features_with_nan = pd.DataFrame({
            'image_path': ['image_1.png', 'image_2.png', 'image_3.png'],
            'defect_id': [0, 1, 2],
            'area': [50, np.nan, 70],
            'perimeter': [100, 120, np.nan],
            'circularity': [0.8, 0.7, 0.9],
            'aspect_ratio': [1.2, 1.1, 1.3],
            'mean_intensity': [100, 110, 120],
            'std_intensity': [20, 25, 30],
            'contrast': [0.5, 0.6, 0.7],
            'homogeneity': [0.8, 0.7, 0.9],
            'energy': [0.3, 0.4, 0.5],
            'correlation': [0.8, 0.7, 0.9],
            'zone': ['core', 'cladding', 'ferrule']
        })
        
        # Should handle NaN values (either by dropping or imputing)
        try:
            clustered_df, _ = cluster_defects(features_with_nan, n_clusters=2)
            # If successful, check that clustering was performed
            self.assertIn('cluster', clustered_df.columns)
        except ValueError:
            # It's okay if it raises an error for NaN values
            pass
    
    def test_categorical_zone_handling(self):
        """Test that categorical 'zone' feature is handled properly."""
        from clustering import cluster_defects
        
        # Create features with different zone values
        features = pd.DataFrame({
            'image_path': [f'image_{i}.png' for i in range(9)],
            'defect_id': list(range(9)),
            'area': [50, 55, 52, 100, 105, 102, 150, 155, 152],
            'perimeter': [100, 105, 102, 150, 155, 152, 200, 205, 202],
            'circularity': [0.8] * 9,
            'aspect_ratio': [1.2] * 9,
            'mean_intensity': [100, 105, 102, 150, 155, 152, 200, 205, 202],
            'std_intensity': [20] * 9,
            'contrast': [0.5] * 9,
            'homogeneity': [0.8] * 9,
            'energy': [0.3] * 9,
            'correlation': [0.8] * 9,
            'zone': ['core', 'core', 'core', 'cladding', 'cladding', 'cladding', 'ferrule', 'ferrule', 'ferrule']
        })
        
        # Run clustering
        clustered_df, _ = cluster_defects(features, n_clusters=3)
        
        # Check that zone information is preserved
        self.assertIn('zone', clustered_df.columns)
        self.assertEqual(set(clustered_df['zone']), {'core', 'cladding', 'ferrule'})
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        from clustering import cluster_defects
        import time
        
        # Create larger dataset
        n_samples = 1000
        large_features = pd.DataFrame({
            'image_path': [f'image_{i}.png' for i in range(n_samples)],
            'defect_id': list(range(n_samples)),
            'area': np.random.randint(10, 100, n_samples),
            'perimeter': np.random.randint(20, 200, n_samples),
            'circularity': np.random.rand(n_samples),
            'aspect_ratio': np.random.rand(n_samples) * 2,
            'mean_intensity': np.random.randint(50, 200, n_samples),
            'std_intensity': np.random.rand(n_samples) * 50,
            'contrast': np.random.rand(n_samples),
            'homogeneity': np.random.rand(n_samples),
            'energy': np.random.rand(n_samples),
            'correlation': np.random.rand(n_samples),
            'zone': np.random.choice(['core', 'cladding', 'ferrule'], n_samples)
        })
        
        # Time the clustering
        start_time = time.time()
        clustered_df, _ = cluster_defects(large_features, n_clusters=5)
        end_time = time.time()
        
        # Check that it completes in reasonable time
        execution_time = end_time - start_time
        self.assertLess(execution_time, 10.0)  # Should complete within 10 seconds
        
        # Check results
        self.assertEqual(len(clustered_df), n_samples)
        self.assertIn('cluster', clustered_df.columns)

if __name__ == '__main__':
    unittest.main()