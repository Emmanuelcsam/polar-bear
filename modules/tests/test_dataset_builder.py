#!/usr/bin/env python3
"""
Comprehensive tests for dataset_builder.py
Tests dataset creation and organization functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os

class TestBuildDatasets(unittest.TestCase):
    """Test build_datasets function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample clustered features DataFrame
        self.clustered_features = pd.DataFrame({
            'image_path': [f'/path/to/image_{i}.png' for i in range(30)],
            'defect_id': list(range(30)),
            'area': np.random.randint(10, 100, 30),
            'perimeter': np.random.randint(20, 200, 30),
            'circularity': np.random.rand(30),
            'aspect_ratio': np.random.rand(30) * 2,
            'mean_intensity': np.random.randint(50, 200, 30),
            'std_intensity': np.random.rand(30) * 50,
            'contrast': np.random.rand(30),
            'homogeneity': np.random.rand(30),
            'energy': np.random.rand(30),
            'correlation': np.random.rand(30),
            'zone': np.random.choice(['core', 'cladding', 'ferrule'], 30),
            'cluster': np.random.randint(0, 5, 30)
        })
        
        # Create dummy images for some defects
        self.image_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        for i in range(5):
            img_path = os.path.join(self.image_dir, f'image_{i}.png')
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(img_path, dummy_img)
            
            # Update paths in dataframe
            self.clustered_features.loc[i, 'image_path'] = img_path
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_dataset_building(self):
        """Test basic dataset building functionality."""
        from dataset_builder import build_datasets
        
        # Build datasets
        output_dir = Path(self.temp_dir) / 'output'
        build_datasets(self.clustered_features, output_dir)
        
        # Check output directory structure
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / 'image_dataset.csv').exists())
        self.assertTrue((output_dir / 'region_dataset.csv').exists())
        self.assertTrue((output_dir / 'defect_library').exists())
    
    def test_image_level_dataset(self):
        """Test image-level dataset creation."""
        from dataset_builder import build_datasets
        
        output_dir = Path(self.temp_dir) / 'output'
        build_datasets(self.clustered_features, output_dir)
        
        # Load image dataset
        image_dataset = pd.read_csv(output_dir / 'image_dataset.csv')
        
        # Check structure
        expected_columns = [
            'image_path', 'num_defects',
            'area_mean', 'area_std', 'area_min', 'area_max',
            'circularity_mean', 'mean_intensity_mean'
        ]
        
        for col in expected_columns:
            self.assertIn(col, image_dataset.columns)
        
        # Check aggregation
        unique_images = self.clustered_features['image_path'].nunique()
        self.assertEqual(len(image_dataset), unique_images)
        
        # Check defect counts
        for _, row in image_dataset.iterrows():
            img_path = row['image_path']
            actual_count = len(self.clustered_features[self.clustered_features['image_path'] == img_path])
            self.assertEqual(row['num_defects'], actual_count)
    
    def test_region_level_dataset(self):
        """Test region-level dataset creation."""
        from dataset_builder import build_datasets
        
        output_dir = Path(self.temp_dir) / 'output'
        build_datasets(self.clustered_features, output_dir)
        
        # Load region dataset
        region_dataset = pd.read_csv(output_dir / 'region_dataset.csv')
        
        # Should have same number of rows as input
        self.assertEqual(len(region_dataset), len(self.clustered_features))
        
        # Check all columns preserved
        for col in self.clustered_features.columns:
            self.assertIn(col, region_dataset.columns)
        
        # Check cluster column exists
        self.assertIn('cluster', region_dataset.columns)
    
    def test_defect_library_organization(self):
        """Test defect library directory structure."""
        from dataset_builder import build_datasets
        
        output_dir = Path(self.temp_dir) / 'output'
        build_datasets(self.clustered_features, output_dir)
        
        defect_lib = output_dir / 'defect_library'
        
        # Check cluster directories created
        cluster_dirs = list(defect_lib.glob('cluster_*'))
        unique_clusters = self.clustered_features['cluster'].nunique()
        self.assertEqual(len(cluster_dirs), unique_clusters)
        
        # Check images copied to appropriate clusters
        for cluster in self.clustered_features['cluster'].unique():
            cluster_dir = defect_lib / f'cluster_{cluster}'
            self.assertTrue(cluster_dir.exists())
            
            # Count images in cluster
            cluster_images = list(cluster_dir.glob('*.png'))
            
            # Count defects with valid image paths in this cluster
            cluster_df = self.clustered_features[self.clustered_features['cluster'] == cluster]
            valid_images = cluster_df[cluster_df['image_path'].apply(os.path.exists)]
            
            # Should have copied valid images
            if len(valid_images) > 0:
                self.assertGreater(len(cluster_images), 0)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from dataset_builder import build_datasets
        
        empty_df = pd.DataFrame()
        output_dir = Path(self.temp_dir) / 'empty_output'
        
        # Should handle empty input gracefully
        try:
            build_datasets(empty_df, output_dir)
            # Check that output files exist but are empty/minimal
            self.assertTrue(output_dir.exists())
        except Exception as e:
            # Or it might raise an exception - both are acceptable
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        from dataset_builder import build_datasets
        
        # Remove important columns
        incomplete_df = self.clustered_features.drop(columns=['cluster', 'area'])
        output_dir = Path(self.temp_dir) / 'incomplete_output'
        
        # Should either handle gracefully or raise meaningful error
        with self.assertRaises((KeyError, ValueError)):
            build_datasets(incomplete_df, output_dir)
    
    def test_invalid_image_paths(self):
        """Test handling of invalid image paths."""
        from dataset_builder import build_datasets
        
        # Set all image paths to non-existent files
        invalid_df = self.clustered_features.copy()
        invalid_df['image_path'] = [f'/non/existent/path_{i}.png' for i in range(len(invalid_df))]
        
        output_dir = Path(self.temp_dir) / 'invalid_output'
        build_datasets(invalid_df, output_dir)
        
        # Should create datasets even if images don't exist
        self.assertTrue((output_dir / 'image_dataset.csv').exists())
        self.assertTrue((output_dir / 'region_dataset.csv').exists())
        
        # Defect library might be empty
        defect_lib = output_dir / 'defect_library'
        if defect_lib.exists():
            # Check no images were copied
            all_images = list(defect_lib.rglob('*.png'))
            self.assertEqual(len(all_images), 0)
    
    def test_overwrite_existing_output(self):
        """Test overwriting existing output directory."""
        from dataset_builder import build_datasets
        
        output_dir = Path(self.temp_dir) / 'overwrite_output'
        
        # Create existing output
        output_dir.mkdir(parents=True)
        (output_dir / 'existing_file.txt').write_text('existing content')
        
        # Build datasets (should overwrite)
        build_datasets(self.clustered_features, output_dir)
        
        # Check new files created
        self.assertTrue((output_dir / 'image_dataset.csv').exists())
        self.assertTrue((output_dir / 'region_dataset.csv').exists())
    
    def test_cluster_statistics(self):
        """Test cluster-level statistics in datasets."""
        from dataset_builder import build_datasets
        
        output_dir = Path(self.temp_dir) / 'stats_output'
        build_datasets(self.clustered_features, output_dir)
        
        # Load datasets
        image_dataset = pd.read_csv(output_dir / 'image_dataset.csv')
        
        # Check for cluster distribution columns
        for cluster in self.clustered_features['cluster'].unique():
            cluster_col = f'cluster_{cluster}_count'
            if cluster_col in image_dataset.columns:
                # Verify counts are correct
                for _, row in image_dataset.iterrows():
                    img_path = row['image_path']
                    actual_count = len(self.clustered_features[
                        (self.clustered_features['image_path'] == img_path) &
                        (self.clustered_features['cluster'] == cluster)
                    ])
                    self.assertEqual(row[cluster_col], actual_count)
    
    def test_zone_distribution(self):
        """Test zone distribution in datasets."""
        from dataset_builder import build_datasets
        
        # Ensure we have defects in each zone
        balanced_df = self.clustered_features.copy()
        zones = ['core', 'cladding', 'ferrule']
        for i, zone in enumerate(zones):
            balanced_df.loc[i*10:(i+1)*10-1, 'zone'] = zone
        
        output_dir = Path(self.temp_dir) / 'zone_output'
        build_datasets(balanced_df, output_dir)
        
        # Check region dataset preserves zone information
        region_dataset = pd.read_csv(output_dir / 'region_dataset.csv')
        self.assertIn('zone', region_dataset.columns)
        
        # Verify zone distribution
        for zone in zones:
            zone_count = len(region_dataset[region_dataset['zone'] == zone])
            self.assertGreater(zone_count, 0)

class TestDatasetBuilderUtils(unittest.TestCase):
    """Test utility functions and helpers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_handling(self):
        """Test Path object handling."""
        from dataset_builder import build_datasets
        
        # Create test data
        test_df = pd.DataFrame({
            'image_path': [str(Path('/path/to/image.png'))],
            'defect_id': [0],
            'cluster': [0],
            'area': [100]
        })
        
        # Test with Path object
        output_path = Path(self.temp_dir) / 'path_test'
        build_datasets(test_df, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Test with string path
        output_str = os.path.join(self.temp_dir, 'str_test')
        build_datasets(test_df, output_str)
        
        self.assertTrue(os.path.exists(output_str))
    
    def test_csv_formatting(self):
        """Test CSV output formatting."""
        from dataset_builder import build_datasets
        
        # Create test data with various types
        test_df = pd.DataFrame({
            'image_path': ['image1.png', 'image2.png'],
            'defect_id': [0, 1],
            'cluster': [0, 1],
            'area': [100.5, 200.7],
            'zone': ['core', 'cladding'],
            'float_value': [1.23456789, 2.3456789]
        })
        
        output_dir = Path(self.temp_dir) / 'format_test'
        build_datasets(test_df, output_dir)
        
        # Check CSV formatting
        region_dataset = pd.read_csv(output_dir / 'region_dataset.csv')
        
        # Verify data types preserved
        self.assertEqual(region_dataset['zone'].dtype, object)
        self.assertTrue(np.issubdtype(region_dataset['area'].dtype, np.floating))
        
        # Check no data loss
        self.assertAlmostEqual(region_dataset['area'].iloc[0], 100.5, places=1)

if __name__ == '__main__':
    unittest.main()