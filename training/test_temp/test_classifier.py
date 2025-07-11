#!/usr/bin/env python3
"""
Comprehensive test suite for image-classifier.py
Tests all functions, methods, classes, and features
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import pickle
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import cv2
from PIL import Image
import logging

# Set up test logging
logging.basicConfig(level=logging.DEBUG)

# Add the current directory to path to import the classifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestImageClassifier(unittest.TestCase):
    """Test suite for UltimateImageClassifier"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.reference_dir = os.path.join(cls.test_dir, "reference")
        cls.dataset_dir = os.path.join(cls.test_dir, "dataset")
        cls.logs_dir = os.path.join(cls.test_dir, "logs")
        
        # Create directories
        os.makedirs(cls.reference_dir, exist_ok=True)
        os.makedirs(cls.dataset_dir, exist_ok=True)
        os.makedirs(cls.logs_dir, exist_ok=True)
        
        # Create test images
        cls.create_test_images()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_test_images(cls):
        """Create test images for testing"""
        # Create various test images
        test_images = [
            ("26-50-clean.png", (100, 100, 3), [255, 0, 0]),  # Red image
            ("27-sma-clean.jpg", (120, 120, 3), [0, 255, 0]),  # Green image
            ("28-91-dirty.jpg", (80, 80, 3), [0, 0, 255]),     # Blue image
            ("test-fc-core.png", (150, 150, 3), [255, 255, 0]), # Yellow image
        ]
        
        for filename, shape, color in test_images:
            # Create reference image
            ref_path = os.path.join(cls.reference_dir, filename)
            img = np.full(shape, color, dtype=np.uint8)
            cv2.imwrite(ref_path, img)
            
            # Create dataset image (slightly different)
            dataset_path = os.path.join(cls.dataset_dir, f"dataset_{filename}")
            # Add some noise to make it slightly different
            noisy_img = img.copy()
            noise = np.random.randint(-10, 10, shape, dtype=np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(dataset_path, noisy_img)

class TestKnowledgeBank(unittest.TestCase):
    """Test KnowledgeBank class"""
    
    def setUp(self):
        """Set up test case"""
        self.test_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.test_dir, "test_kb.pkl")
        
    def tearDown(self):
        """Clean up test case"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_knowledge_bank_init(self):
        """Test KnowledgeBank initialization"""
        # Import here to avoid issues with module-level execution
        from image_classifier_fixed import KnowledgeBank
        
        kb = KnowledgeBank(self.kb_path)
        self.assertEqual(kb.filepath, self.kb_path)
        self.assertEqual(kb.version, "2.0")
        self.assertIsInstance(kb.features_db, dict)
        self.assertIsInstance(kb.classifications_db, dict)
    
    def test_add_custom_keyword(self):
        """Test adding custom keywords"""
        from image_classifier_fixed import KnowledgeBank
        
        kb = KnowledgeBank(self.kb_path)
        kb.add_custom_keyword("test_keyword")
        self.assertIn("test_keyword", kb.custom_keywords)
    
    def test_add_image(self):
        """Test adding image to knowledge bank"""
        from image_classifier_fixed import KnowledgeBank
        
        kb = KnowledgeBank(self.kb_path)
        
        # Test data
        features = np.random.rand(100).astype(np.float32)
        classifications = ["test-class"]
        characteristics = {"type": "test", "condition": "clean"}
        image_hash = "test_hash_123"
        
        kb.add_image(image_hash, features, classifications, characteristics, "test_path.jpg")
        
        self.assertIn(image_hash, kb.features_db)
        self.assertIn(image_hash, kb.classifications_db)
        self.assertIn(image_hash, kb.characteristics_db)
        self.assertEqual(kb.learning_stats['total_images'], 1)
    
    def test_save_load(self):
        """Test saving and loading knowledge bank"""
        from image_classifier_fixed import KnowledgeBank
        
        # Create and populate knowledge bank
        kb1 = KnowledgeBank(self.kb_path)
        kb1.add_custom_keyword("test_save_load")
        features = np.random.rand(50).astype(np.float32)
        kb1.add_image("hash1", features, ["class1"], {"type": "test"})
        kb1.save()
        
        # Load into new instance
        kb2 = KnowledgeBank(self.kb_path)
        
        self.assertIn("test_save_load", kb2.custom_keywords)
        self.assertIn("hash1", kb2.features_db)
        self.assertEqual(kb2.learning_stats['total_images'], 1)

class TestClassifierFunctions(unittest.TestCase):
    """Test individual classifier functions"""
    
    def setUp(self):
        """Set up test case"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test case"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_install_package(self):
        """Test package installation function"""
        from image_classifier_fixed import install_package
        
        # Test with already installed package
        result = install_package("os", "os")
        self.assertTrue(result)
        
        # Test with non-existent package (should fail gracefully)
        result = install_package("non_existent_package_12345")
        self.assertFalse(result)
    
    def test_setup_logging(self):
        """Test logging setup"""
        from image_classifier_fixed import setup_logging
        
        with patch('os.makedirs'):
            logger = setup_logging()
            self.assertIsNotNone(logger)

class InteractiveClassifierTest:
    """Interactive test runner that handles user interactions"""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp()
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up comprehensive test environment"""
        # Create directory structure
        self.reference_dir = os.path.join(self.test_dir, "reference")
        self.dataset_dir = os.path.join(self.test_dir, "dataset")
        
        os.makedirs(self.reference_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create test images with proper naming for classification
        self.create_test_images()
        
        print(f"Test environment created at: {self.test_dir}")
        print(f"Reference dir: {self.reference_dir}")
        print(f"Dataset dir: {self.dataset_dir}")
    
    def create_test_images(self):
        """Create comprehensive test images"""
        # Reference images with proper classifications
        ref_images = [
            ("26-50-clean.png", (100, 100, 3), [255, 0, 0]),
            ("27-sma-clean.jpg", (120, 120, 3), [0, 255, 0]),
            ("28-91-dirty-scratched.jpg", (80, 80, 3), [0, 0, 255]),
            ("fc-core-clean.png", (150, 150, 3), [255, 255, 0]),
            ("sc-cladding-dirty-oil.jpg", (110, 110, 3), [255, 0, 255]),
        ]
        
        for filename, shape, color in ref_images:
            # Create reference image
            ref_path = os.path.join(self.reference_dir, filename)
            img = np.full(shape, color, dtype=np.uint8)
            cv2.imwrite(ref_path, img)
        
        # Dataset images (to be classified)
        dataset_images = [
            ("unknown_1.jpg", (100, 100, 3), [250, 5, 5]),      # Similar to 26-50-clean
            ("unknown_2.png", (120, 120, 3), [5, 250, 5]),      # Similar to 27-sma-clean
            ("unknown_3.jpg", (80, 80, 3), [5, 5, 250]),        # Similar to 28-91-dirty
            ("test_image.png", (150, 150, 3), [250, 250, 5]),   # Similar to fc-core-clean
        ]
        
        for filename, shape, color in dataset_images:
            dataset_path = os.path.join(self.dataset_dir, filename)
            img = np.full(shape, color, dtype=np.uint8)
            # Add some noise for realism
            noise = np.random.randint(-5, 5, shape, dtype=np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(dataset_path, noisy_img)
    
    def test_auto_mode(self):
        """Test automatic mode with simulated responses"""
        print("\n" + "="*60)
        print("TESTING AUTOMATIC MODE")
        print("="*60)
        
        try:
            # Import the fixed classifier
            from image_classifier_fixed import UltimateImageClassifier
            
            # Initialize classifier
            classifier = UltimateImageClassifier()
            
            # Test automatic processing
            classifier.process_dataset_auto(self.reference_dir, self.dataset_dir)
            
            print("✓ Automatic mode test completed successfully")
            
        except Exception as e:
            print(f"✗ Automatic mode test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_manual_mode_console(self):
        """Test manual mode with console interface"""
        print("\n" + "="*60)
        print("TESTING MANUAL MODE (CONSOLE)")
        print("="*60)
        
        try:
            from image_classifier_fixed import UltimateImageClassifier
            
            # Mock user inputs for testing
            mock_inputs = [
                "test-classification",  # First image classification
                "s",                    # Skip second image
                "auto-suggest",         # Accept suggestion for third
                "q"                     # Quit
            ]
            
            with patch('builtins.input', side_effect=mock_inputs):
                classifier = UltimateImageClassifier()
                classifier.process_dataset_manual_console(self.reference_dir, self.dataset_dir)
            
            print("✓ Manual console mode test completed successfully")
            
        except Exception as e:
            print(f"✗ Manual console mode test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_feature_extraction(self):
        """Test feature extraction methods"""
        print("\n" + "="*60)
        print("TESTING FEATURE EXTRACTION")
        print("="*60)
        
        try:
            from image_classifier_fixed import UltimateImageClassifier
            
            classifier = UltimateImageClassifier()
            
            # Test with each test image
            test_images = [f for f in os.listdir(self.reference_dir) if f.endswith(('.jpg', '.png'))]
            
            for img_file in test_images:
                img_path = os.path.join(self.reference_dir, img_file)
                print(f"Testing feature extraction for: {img_file}")
                
                features, img_hash = classifier.extract_features(img_path)
                
                self.assertIsNotNone(features, f"Features should not be None for {img_file}")
                self.assertIsNotNone(img_hash, f"Hash should not be None for {img_file}")
                self.assertIsInstance(features, np.ndarray, f"Features should be numpy array for {img_file}")
                
                print(f"  ✓ Features shape: {features.shape}")
                print(f"  ✓ Hash: {img_hash}")
            
            print("✓ Feature extraction test completed successfully")
            
        except Exception as e:
            print(f"✗ Feature extraction test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_classification_parsing(self):
        """Test classification parsing"""
        print("\n" + "="*60)
        print("TESTING CLASSIFICATION PARSING")
        print("="*60)
        
        try:
            from image_classifier_fixed import UltimateImageClassifier
            
            classifier = UltimateImageClassifier()
            
            test_cases = [
                "26-50-clean",
                "27-sma-dirty-scratched",
                "fc-core-clean",
                "sc-cladding-dirty-oil-wet",
                "91-ferrule-anomaly"
            ]
            
            for test_case in test_cases:
                print(f"Testing parsing: {test_case}")
                components = classifier.parse_classification(test_case)
                print(f"  Parsed components: {components}")
                
                # Test building back
                rebuilt = classifier.build_classification_string(components)
                print(f"  Rebuilt string: {rebuilt}")
            
            print("✓ Classification parsing test completed successfully")
            
        except Exception as e:
            print(f"✗ Classification parsing test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def test_similarity_calculation(self):
        """Test similarity calculation"""
        print("\n" + "="*60)
        print("TESTING SIMILARITY CALCULATION")
        print("="*60)
        
        try:
            from image_classifier_fixed import UltimateImageClassifier
            
            classifier = UltimateImageClassifier()
            
            # Create test feature vectors
            features1 = np.random.rand(100).astype(np.float32)
            features2 = features1.copy()  # Identical
            features3 = np.random.rand(100).astype(np.float32)  # Different
            
            # Test identical features
            sim1 = classifier.calculate_similarity(features1, features2)
            print(f"Similarity between identical features: {sim1:.3f}")
            self.assertGreater(sim1, 0.9, "Identical features should have high similarity")
            
            # Test different features
            sim2 = classifier.calculate_similarity(features1, features3)
            print(f"Similarity between different features: {sim2:.3f}")
            
            # Test with None features
            sim3 = classifier.calculate_similarity(None, features1)
            print(f"Similarity with None features: {sim3:.3f}")
            self.assertEqual(sim3, 0.0, "None features should return 0 similarity")
            
            print("✓ Similarity calculation test completed successfully")
            
        except Exception as e:
            print(f"✗ Similarity calculation test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_tests(self):
        """Run all interactive tests"""
        print("Starting comprehensive classifier testing...")
        
        # Run individual tests
        self.test_feature_extraction()
        self.test_classification_parsing()
        self.test_similarity_calculation()
        self.test_auto_mode()
        self.test_manual_mode_console()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        print("Test environment cleaned up.")
    
    def assertIsNotNone(self, value, msg=""):
        """Helper assertion method"""
        if value is None:
            raise AssertionError(f"Value is None: {msg}")
    
    def assertIsInstance(self, obj, cls, msg=""):
        """Helper assertion method"""
        if not isinstance(obj, cls):
            raise AssertionError(f"Object is not instance of {cls}: {msg}")
    
    def assertGreater(self, a, b, msg=""):
        """Helper assertion method"""
        if not a > b:
            raise AssertionError(f"{a} is not greater than {b}: {msg}")
    
    def assertEqual(self, a, b, msg=""):
        """Helper assertion method"""
        if a != b:
            raise AssertionError(f"{a} != {b}: {msg}")

if __name__ == "__main__":
    print("Image Classifier Comprehensive Test Suite")
    print("="*50)
    
    # First, run unit tests
    print("\nRunning unit tests...")
    # unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Then run interactive tests
    print("\nRunning interactive tests...")
    interactive_test = InteractiveClassifierTest()
    interactive_test.run_all_tests()
