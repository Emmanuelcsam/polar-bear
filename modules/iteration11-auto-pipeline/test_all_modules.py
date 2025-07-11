#!/usr/bin/env python3
# test_all_modules.py
# Comprehensive unit tests for all modules in the image analysis system

import unittest
import os
import shutil
import json
import numpy as np
import cv2
import torch
import tempfile
from unittest.mock import patch, MagicMock

# Import the modules we'll be testing
import shared_config
from shared_config import DATA_DIR, IMAGE_INPUT_DIR


class TestSharedConfig(unittest.TestCase):
    """Test the shared configuration module"""
    
    def test_device_selection(self):
        """Test that device is either 'cuda' or 'cpu'"""
        self.assertIn(shared_config.DEVICE, ['cuda', 'cpu'])
    
    def test_paths_exist(self):
        """Test that all paths are properly defined"""
        self.assertIsNotNone(shared_config.DATA_DIR)
        self.assertIsNotNone(shared_config.IMAGE_INPUT_DIR)
        self.assertIsNotNone(shared_config.GENERATOR_PARAMS_PATH)
        self.assertIsNotNone(shared_config.ANALYSIS_RESULTS_PATH)
        self.assertIsNotNone(shared_config.ANOMALIES_PATH)
    
    def test_image_size(self):
        """Test that generated image size is a valid tuple"""
        self.assertIsInstance(shared_config.GENERATED_IMAGE_SIZE, tuple)
        self.assertEqual(len(shared_config.GENERATED_IMAGE_SIZE), 2)
        self.assertGreater(shared_config.GENERATED_IMAGE_SIZE[0], 0)
        self.assertGreater(shared_config.GENERATED_IMAGE_SIZE[1], 0)


class TestSetupDirectories(unittest.TestCase):
    """Test the directory setup module"""
    
    def setUp(self):
        """Create a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_input_dir = shared_config.IMAGE_INPUT_DIR
        
        # Override the directories for testing
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.IMAGE_INPUT_DIR = os.path.join(self.test_dir, 'input_images')
    
    def tearDown(self):
        """Clean up the test directory"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.IMAGE_INPUT_DIR = self.original_input_dir
    
    def test_directory_creation(self):
        """Test that directories are created properly"""
        # Import and run the setup module
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup", "1_setup_directories.py")
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)
        
        # Check directories exist
        self.assertTrue(os.path.exists(shared_config.DATA_DIR))
        self.assertTrue(os.path.exists(shared_config.IMAGE_INPUT_DIR))
        
        # Check placeholder file exists
        placeholder_path = os.path.join(shared_config.IMAGE_INPUT_DIR, "place_your_images_here.txt")
        self.assertTrue(os.path.exists(placeholder_path))


class TestAnalyzeImages(unittest.TestCase):
    """Test the image analysis module"""
    
    def setUp(self):
        """Create test environment with sample images"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_input_dir = shared_config.IMAGE_INPUT_DIR
        self.original_analysis_path = shared_config.ANALYSIS_RESULTS_PATH
        
        # Override paths
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.IMAGE_INPUT_DIR = os.path.join(self.test_dir, 'input_images')
        shared_config.ANALYSIS_RESULTS_PATH = os.path.join(shared_config.DATA_DIR, 'analysis_results.json')
        
        # Create directories
        os.makedirs(shared_config.DATA_DIR)
        os.makedirs(shared_config.IMAGE_INPUT_DIR)
        
        # Create test images with known properties
        self.create_test_image('test1.png', mean=100, std=20)
        self.create_test_image('test2.png', mean=150, std=30)
        self.create_test_image('test3.jpg', mean=200, std=40)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.IMAGE_INPUT_DIR = self.original_input_dir
        shared_config.ANALYSIS_RESULTS_PATH = self.original_analysis_path
    
    def create_test_image(self, filename, mean, std):
        """Create a test image with specific statistical properties"""
        # Generate random image data with specified mean and std
        image = np.random.normal(mean, std, (100, 100))
        image = np.clip(image, 0, 255).astype(np.uint8)
        path = os.path.join(shared_config.IMAGE_INPUT_DIR, filename)
        cv2.imwrite(path, image)
    
    def test_image_analysis(self):
        """Test that images are analyzed correctly"""
        # Run the analysis module
        exec(open('2_analyze_images.py').read())
        
        # Check that results file was created
        self.assertTrue(os.path.exists(shared_config.ANALYSIS_RESULTS_PATH))
        
        # Load and verify results
        with open(shared_config.ANALYSIS_RESULTS_PATH, 'r') as f:
            results = json.load(f)
        
        # Check all images were processed
        self.assertEqual(len(results), 3)
        self.assertIn('test1.png', results)
        self.assertIn('test2.png', results)
        self.assertIn('test3.jpg', results)
        
        # Verify each result has required fields
        for filename, data in results.items():
            self.assertIn('mean_intensity', data)
            self.assertIn('std_dev_intensity', data)
            self.assertIsInstance(data['mean_intensity'], (int, float))
            self.assertIsInstance(data['std_dev_intensity'], (int, float))
    
    def test_empty_directory(self):
        """Test behavior with no images"""
        # Remove all images
        for f in os.listdir(shared_config.IMAGE_INPUT_DIR):
            os.remove(os.path.join(shared_config.IMAGE_INPUT_DIR, f))
        
        # The module should exit gracefully
        with self.assertRaises(SystemExit):
            exec(open('2_analyze_images.py').read())


class TestLearnFromAnalysis(unittest.TestCase):
    """Test the learning module"""
    
    def setUp(self):
        """Create test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_analysis_path = shared_config.ANALYSIS_RESULTS_PATH
        self.original_params_path = shared_config.GENERATOR_PARAMS_PATH
        
        # Override paths
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.ANALYSIS_RESULTS_PATH = os.path.join(shared_config.DATA_DIR, 'analysis_results.json')
        shared_config.GENERATOR_PARAMS_PATH = os.path.join(shared_config.DATA_DIR, 'generator_params.json')
        
        os.makedirs(shared_config.DATA_DIR)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.ANALYSIS_RESULTS_PATH = self.original_analysis_path
        shared_config.GENERATOR_PARAMS_PATH = self.original_params_path
    
    def test_learning_from_analysis(self):
        """Test learning from analysis data"""
        # Create mock analysis data
        analysis_data = {
            'img1.png': {'mean_intensity': 100, 'std_dev_intensity': 20},
            'img2.png': {'mean_intensity': 150, 'std_dev_intensity': 30},
            'img3.png': {'mean_intensity': 200, 'std_dev_intensity': 40}
        }
        
        with open(shared_config.ANALYSIS_RESULTS_PATH, 'w') as f:
            json.dump(analysis_data, f)
        
        # Run the learning module
        exec(open('3_learn_from_analysis.py').read())
        
        # Verify parameters were saved
        self.assertTrue(os.path.exists(shared_config.GENERATOR_PARAMS_PATH))
        
        with open(shared_config.GENERATOR_PARAMS_PATH, 'r') as f:
            params = json.load(f)
        
        # Check learned parameters
        expected_mean = np.mean([100, 150, 200])
        expected_std = np.mean([20, 30, 40])
        
        self.assertAlmostEqual(params['target_mean'], expected_mean, places=2)
        self.assertAlmostEqual(params['target_std'], expected_std, places=2)
    
    def test_no_analysis_file(self):
        """Test behavior when analysis file doesn't exist"""
        # Run the learning module without analysis data
        exec(open('3_learn_from_analysis.py').read())
        
        # Should use default parameters
        with open(shared_config.GENERATOR_PARAMS_PATH, 'r') as f:
            params = json.load(f)
        
        self.assertEqual(params['target_mean'], 128.0)
        self.assertEqual(params['target_std'], 50.0)


class TestGenerateNewImage(unittest.TestCase):
    """Test the image generation module"""
    
    def setUp(self):
        """Create test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_params_path = shared_config.GENERATOR_PARAMS_PATH
        
        # Override paths
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.GENERATOR_PARAMS_PATH = os.path.join(shared_config.DATA_DIR, 'generator_params.json')
        
        os.makedirs(shared_config.DATA_DIR)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.GENERATOR_PARAMS_PATH = self.original_params_path
    
    def test_image_generation(self):
        """Test that image is generated with correct properties"""
        # Create test parameters
        params = {'target_mean': 150.0, 'target_std': 25.0}
        with open(shared_config.GENERATOR_PARAMS_PATH, 'w') as f:
            json.dump(params, f)
        
        # Run the generator
        exec(open('4_generate_new_image.py').read())
        
        # Check image was created
        output_path = os.path.join(shared_config.DATA_DIR, 'generated_image.png')
        self.assertTrue(os.path.exists(output_path))
        
        # Load and verify image properties
        image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, shared_config.GENERATED_IMAGE_SIZE)
        
        # Check that values are in valid range
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(image <= 255))
    
    def test_generation_without_params(self):
        """Test generation with default parameters"""
        # Run without parameter file
        exec(open('4_generate_new_image.py').read())
        
        # Should still create an image
        output_path = os.path.join(shared_config.DATA_DIR, 'generated_image.png')
        self.assertTrue(os.path.exists(output_path))


class TestDetectAnomalies(unittest.TestCase):
    """Test the anomaly detection module"""
    
    def setUp(self):
        """Create test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_analysis_path = shared_config.ANALYSIS_RESULTS_PATH
        self.original_anomalies_path = shared_config.ANOMALIES_PATH
        
        # Override paths
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.ANALYSIS_RESULTS_PATH = os.path.join(shared_config.DATA_DIR, 'analysis_results.json')
        shared_config.ANOMALIES_PATH = os.path.join(shared_config.DATA_DIR, 'anomalies.json')
        
        os.makedirs(shared_config.DATA_DIR)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.ANALYSIS_RESULTS_PATH = self.original_analysis_path
        shared_config.ANOMALIES_PATH = self.original_anomalies_path
    
    def test_anomaly_detection(self):
        """Test that anomalies are detected correctly"""
        # Create analysis data with one clear anomaly
        analysis_data = {
            'normal1.png': {'mean_intensity': 100, 'std_dev_intensity': 20},
            'normal2.png': {'mean_intensity': 105, 'std_dev_intensity': 22},
            'normal3.png': {'mean_intensity': 98, 'std_dev_intensity': 19},
            'anomaly.png': {'mean_intensity': 250, 'std_dev_intensity': 50}  # Clear outlier
        }
        
        with open(shared_config.ANALYSIS_RESULTS_PATH, 'w') as f:
            json.dump(analysis_data, f)
        
        # Run anomaly detection
        exec(open('5_detect_anomalies.py').read())
        
        # Check results
        self.assertTrue(os.path.exists(shared_config.ANOMALIES_PATH))
        
        with open(shared_config.ANOMALIES_PATH, 'r') as f:
            anomalies = json.load(f)
        
        # Should detect the anomaly
        self.assertIn('anomaly.png', anomalies)
        self.assertNotIn('normal1.png', anomalies)
        
        # Verify anomaly data
        self.assertIn('mean_intensity', anomalies['anomaly.png'])
        self.assertIn('deviation_score', anomalies['anomaly.png'])
        self.assertGreater(anomalies['anomaly.png']['deviation_score'], 2.0)
    
    def test_no_anomalies(self):
        """Test when no anomalies exist"""
        # Create similar images
        analysis_data = {
            'img1.png': {'mean_intensity': 100, 'std_dev_intensity': 20},
            'img2.png': {'mean_intensity': 102, 'std_dev_intensity': 21},
            'img3.png': {'mean_intensity': 98, 'std_dev_intensity': 19}
        }
        
        with open(shared_config.ANALYSIS_RESULTS_PATH, 'w') as f:
            json.dump(analysis_data, f)
        
        exec(open('5_detect_anomalies.py').read())
        
        with open(shared_config.ANOMALIES_PATH, 'r') as f:
            anomalies = json.load(f)
        
        # Should be empty
        self.assertEqual(len(anomalies), 0)
    
    def test_insufficient_data(self):
        """Test with too few images"""
        # Only one image
        analysis_data = {
            'single.png': {'mean_intensity': 100, 'std_dev_intensity': 20}
        }
        
        with open(shared_config.ANALYSIS_RESULTS_PATH, 'w') as f:
            json.dump(analysis_data, f)
        
        # Should exit
        with self.assertRaises(SystemExit):
            exec(open('5_detect_anomalies.py').read())


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire pipeline"""
    
    def setUp(self):
        """Create test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = shared_config.DATA_DIR
        self.original_input_dir = shared_config.IMAGE_INPUT_DIR
        self.original_analysis_path = shared_config.ANALYSIS_RESULTS_PATH
        self.original_params_path = shared_config.GENERATOR_PARAMS_PATH
        self.original_anomalies_path = shared_config.ANOMALIES_PATH
        
        # Override all paths
        shared_config.DATA_DIR = os.path.join(self.test_dir, 'data')
        shared_config.IMAGE_INPUT_DIR = os.path.join(self.test_dir, 'input_images')
        shared_config.ANALYSIS_RESULTS_PATH = os.path.join(shared_config.DATA_DIR, 'analysis_results.json')
        shared_config.GENERATOR_PARAMS_PATH = os.path.join(shared_config.DATA_DIR, 'generator_params.json')
        shared_config.ANOMALIES_PATH = os.path.join(shared_config.DATA_DIR, 'anomalies.json')
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir)
        shared_config.DATA_DIR = self.original_data_dir
        shared_config.IMAGE_INPUT_DIR = self.original_input_dir
        shared_config.ANALYSIS_RESULTS_PATH = self.original_analysis_path
        shared_config.GENERATOR_PARAMS_PATH = self.original_params_path
        shared_config.ANOMALIES_PATH = self.original_anomalies_path
    
    def create_test_images(self):
        """Create test images for integration testing"""
        os.makedirs(shared_config.IMAGE_INPUT_DIR)
        
        # Normal images
        for i in range(3):
            image = np.random.normal(128, 30, (100, 100))
            image = np.clip(image, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(shared_config.IMAGE_INPUT_DIR, f'normal{i}.png'), image)
        
        # Anomaly
        anomaly = np.random.normal(250, 10, (100, 100))
        anomaly = np.clip(anomaly, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(shared_config.IMAGE_INPUT_DIR, 'anomaly.png'), anomaly)
    
    def test_full_pipeline(self):
        """Test the entire pipeline end-to-end"""
        # 1. Setup directories
        exec(open('1_setup_directories.py').read())
        self.assertTrue(os.path.exists(shared_config.DATA_DIR))
        self.assertTrue(os.path.exists(shared_config.IMAGE_INPUT_DIR))
        
        # Add test images
        self.create_test_images()
        
        # 2. Analyze images
        exec(open('2_analyze_images.py').read())
        self.assertTrue(os.path.exists(shared_config.ANALYSIS_RESULTS_PATH))
        
        # 3. Learn from analysis
        exec(open('3_learn_from_analysis.py').read())
        self.assertTrue(os.path.exists(shared_config.GENERATOR_PARAMS_PATH))
        
        # 4. Generate new image
        exec(open('4_generate_new_image.py').read())
        generated_path = os.path.join(shared_config.DATA_DIR, 'generated_image.png')
        self.assertTrue(os.path.exists(generated_path))
        
        # 5. Detect anomalies
        exec(open('5_detect_anomalies.py').read())
        self.assertTrue(os.path.exists(shared_config.ANOMALIES_PATH))
        
        # Verify anomaly was detected
        with open(shared_config.ANOMALIES_PATH, 'r') as f:
            anomalies = json.load(f)
        self.assertIn('anomaly.png', anomalies)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)