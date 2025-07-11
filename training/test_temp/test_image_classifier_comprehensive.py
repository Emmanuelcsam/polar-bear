#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Ultimate Image Classifier
Tests every function, method, class, and feature with automated interactions
"""

import unittest
import os
import sys
import shutil
import tempfile
import json
import pickle
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import logging
import io
from contextlib import redirect_stdout, redirect_stderr
import argparse
from PIL import Image
import cv2
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module to test (will trigger dependency installation)
print("Starting comprehensive test suite...")
print("This will test dependency installation, all classes, methods, and interactive features")
print("-" * 80)

class TestDependencies(unittest.TestCase):
    """Test dependency installation and imports"""
    
    def test_imports(self):
        """Test all required imports work"""
        try:
            import PIL
            import numpy
            import sklearn
            import cv2
            import imagehash
            import tqdm
            import scipy
            import matplotlib
            self.assertTrue(True, "All dependencies imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import dependency: {e}")

class TestLogging(unittest.TestCase):
    """Test logging functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """Test logging setup creates log file and returns logger"""
        # Import after changing directory
        from image_classifier import setup_logging
        
        logger = setup_logging()
        self.assertIsNotNone(logger)
        self.assertTrue(os.path.exists("logs"))
        
        # Check log file was created
        log_files = os.listdir("logs")
        self.assertEqual(len(log_files), 1)
        self.assertTrue(log_files[0].startswith("image_classifier_"))
        self.assertTrue(log_files[0].endswith(".log"))
        
        # Test logging works
        logger.info("Test message")
        logger.debug("Debug message")
        logger.error("Error message")

class TestKnowledgeBank(unittest.TestCase):
    """Test KnowledgeBank class functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "test_kb.pkl")
        
        # Import after setup
        from image_classifier import KnowledgeBank
        self.KnowledgeBank = KnowledgeBank
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test KnowledgeBank initialization"""
        kb = self.KnowledgeBank(self.kb_path)
        self.assertEqual(kb.filepath, self.kb_path)
        self.assertEqual(kb.version, "2.0")
        self.assertIsInstance(kb.features_db, dict)
        self.assertIsInstance(kb.classifications_db, dict)
        self.assertIsInstance(kb.characteristics_db, dict)
        self.assertIsInstance(kb.file_paths_db, dict)
        
    def test_add_custom_keyword(self):
        """Test adding custom keywords"""
        kb = self.KnowledgeBank(self.kb_path)
        kb.add_custom_keyword("test_keyword")
        self.assertIn("test_keyword", kb.custom_keywords)
        
    def test_add_folder_structure(self):
        """Test adding folder structure"""
        kb = self.KnowledgeBank(self.kb_path)
        path_parts = ["folder1", "folder2", "folder3"]
        kb.add_folder_structure(path_parts)
        self.assertIn("folder1/folder2/folder3", kb.folder_structure)
        
    def test_add_image(self):
        """Test adding image to knowledge bank"""
        kb = self.KnowledgeBank(self.kb_path)
        
        # Create test data
        image_hash = "test_hash_12345"
        features = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        classifications = ["test-class-1", "test-class-2"]
        characteristics = {
            "core_diameter": "50",
            "connector_type": "fc",
            "condition": "clean"
        }
        file_path = "/test/path/image.jpg"
        
        # Add image
        kb.add_image(image_hash, features, classifications, characteristics, file_path)
        
        # Verify storage
        self.assertIn(image_hash, kb.features_db)
        self.assertIn(image_hash, kb.classifications_db)
        self.assertIn(image_hash, kb.characteristics_db)
        self.assertIn(image_hash, kb.file_paths_db)
        self.assertEqual(kb.file_paths_db[image_hash], file_path)
        
        # Verify relationships
        for classification in classifications:
            self.assertIn(image_hash, kb.relationships[classification])
            
    def test_add_feedback(self):
        """Test adding user feedback"""
        kb = self.KnowledgeBank(self.kb_path)
        
        image_hash = "test_hash"
        correct_class = "correct-class"
        wrong_class = "wrong-class"
        
        kb.add_feedback(image_hash, correct_class, wrong_class)
        
        self.assertIn(image_hash, kb.user_feedback)
        self.assertEqual(kb.user_feedback[image_hash][0], (correct_class, wrong_class))
        self.assertEqual(kb.learning_stats['user_corrections'], 1)
        
    def test_get_statistics(self):
        """Test getting statistics"""
        kb = self.KnowledgeBank(self.kb_path)
        
        # Add some test data
        kb.add_image("hash1", np.array([1, 2, 3]), ["class1"], {"type": "test"}, "path1")
        kb.add_image("hash2", np.array([4, 5, 6]), ["class1", "class2"], {"type": "test2"}, "path2")
        
        stats = kb.get_statistics()
        
        self.assertIn('total_images', stats)
        self.assertEqual(stats['total_images'], 2)
        self.assertIn('classifications', stats)
        self.assertIn('characteristics', stats)
        
    def test_save_and_load(self):
        """Test saving and loading knowledge bank"""
        kb1 = self.KnowledgeBank(self.kb_path)
        
        # Add test data
        kb1.add_custom_keyword("persistent_keyword")
        kb1.add_image("hash1", np.array([1, 2, 3]), ["class1"], {"type": "test"}, "path1")
        kb1.add_feedback("hash1", "correct", "wrong")
        
        # Save
        kb1.save()
        self.assertTrue(os.path.exists(self.kb_path))
        
        # Load in new instance
        kb2 = self.KnowledgeBank(self.kb_path)
        
        # Verify data persisted
        self.assertIn("persistent_keyword", kb2.custom_keywords)
        self.assertIn("hash1", kb2.features_db)
        self.assertEqual(kb2.learning_stats['user_corrections'], 1)
        
    def test_feature_dimension_handling(self):
        """Test feature dimension mismatch handling"""
        kb = self.KnowledgeBank(self.kb_path)
        
        # Add first image sets dimensions
        features1 = np.array([1, 2, 3, 4, 5])
        kb.add_image("hash1", features1, ["class1"], {})
        self.assertEqual(kb.feature_dimensions, (5,))
        
        # Add image with different dimensions
        features2 = np.array([1, 2, 3])  # Too short
        kb.add_image("hash2", features2, ["class2"], {})
        
        # Should be padded to match
        stored_features = kb.features_db["hash2"]
        self.assertEqual(len(stored_features), 5)
        
        # Add image with too many dimensions
        features3 = np.array([1, 2, 3, 4, 5, 6, 7])  # Too long
        kb.add_image("hash3", features3, ["class3"], {})
        
        # Should be truncated
        stored_features = kb.features_db["hash3"]
        self.assertEqual(len(stored_features), 5)

class TestImageClassifier(unittest.TestCase):
    """Test UltimateImageClassifier class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test images
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        self.create_test_image(self.test_image_path)
        
        # Import after setup
        from image_classifier import UltimateImageClassifier
        self.UltimateImageClassifier = UltimateImageClassifier
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_test_image(self, path, size=(128, 128), color=(255, 0, 0)):
        """Create a test image"""
        img = Image.new('RGB', size, color)
        img.save(path)
        return path
        
    def test_init_with_defaults(self):
        """Test classifier initialization with default values"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        self.assertEqual(classifier.config['similarity_threshold'], 0.65)
        self.assertEqual(classifier.config['auto_create_folders'], True)
        self.assertIsInstance(classifier.knowledge_bank, object)
        self.assertIsInstance(classifier.image_extensions, set)
        
    def test_init_with_custom_values(self):
        """Test classifier initialization with custom values"""
        custom_keywords = ["custom1", "custom2"]
        classifier = self.UltimateImageClassifier(
            config_file=self.config_file,
            similarity_threshold=0.75,
            auto_create_folders=False,
            custom_keywords=custom_keywords
        )
        
        self.assertEqual(classifier.config['similarity_threshold'], 0.75)
        self.assertEqual(classifier.config['auto_create_folders'], False)
        self.assertIn("custom1", classifier.config['custom_keywords'])
        self.assertIn("custom2", classifier.config['custom_keywords'])
        
    def test_save_and_load_config(self):
        """Test configuration save and load"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Modify config
        classifier.config['test_value'] = "test"
        classifier.save_config()
        
        # Verify file exists
        self.assertTrue(os.path.exists(self.config_file))
        
        # Load config
        with open(self.config_file, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config['test_value'], "test")
        
    def test_is_image_file(self):
        """Test image file detection"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        self.assertTrue(classifier.is_image_file("test.jpg"))
        self.assertTrue(classifier.is_image_file("test.JPEG"))
        self.assertTrue(classifier.is_image_file("test.png"))
        self.assertTrue(classifier.is_image_file("test.bmp"))
        self.assertFalse(classifier.is_image_file("test.txt"))
        self.assertFalse(classifier.is_image_file("test.pdf"))
        
    def test_parse_classification(self):
        """Test classification parsing"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Test various classification formats
        test_cases = [
            ("50-fc-core-clean", {
                "core_diameter": "50",
                "connector_type": "fc",
                "region": "core",
                "condition": "clean"
            }),
            ("91_sma_cladding_dirty_scratched", {
                "core_diameter": "91",
                "connector_type": "sma",
                "region": "cladding",
                "condition": "dirty",
                "defect_type": "scratched"
            }),
            ("fc-50-ferrule-oil-wet", {
                "core_diameter": "50",
                "connector_type": "fc",
                "region": "ferrule",
                "defect_type": "oil-wet"
            }),
            ("custom_keyword_test", {
                "additional_characteristics": "custom-keyword-test"
            })
        ]
        
        for text, expected in test_cases:
            result = classifier.parse_classification(text)
            for key, value in expected.items():
                self.assertIn(key, result)
                self.assertEqual(result[key], value)
                
    def test_build_classification_string(self):
        """Test building classification string from components"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        components = {
            "core_diameter": "50",
            "connector_type": "fc",
            "region": "core",
            "condition": "clean",
            "defect_type": "scratched"
        }
        
        result = classifier.build_classification_string(components)
        
        # Should contain all components in configured order
        self.assertIn("50", result)
        self.assertIn("fc", result)
        self.assertIn("core", result)
        self.assertIn("clean", result)
        self.assertIn("scratched", result)
        
    def test_extract_features(self):
        """Test feature extraction from image"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        features, img_hash = classifier.extract_features(self.test_image_path)
        
        self.assertIsNotNone(features)
        self.assertIsNotNone(img_hash)
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(img_hash, str)
        self.assertGreater(len(features), 0)
        
        # Test caching
        features2, img_hash2 = classifier.extract_features(self.test_image_path)
        np.testing.assert_array_equal(features, features2)
        self.assertEqual(img_hash, img_hash2)
        
    def test_feature_extraction_methods(self):
        """Test individual feature extraction methods"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Create test image array
        img = np.ones((128, 128, 3), dtype=np.uint8) * 100
        
        # Test color features
        color_features = classifier._extract_color_features(img)
        self.assertIsInstance(color_features, list)
        self.assertGreater(len(color_features), 0)
        
        # Test texture features
        texture_features = classifier._extract_texture_features(img)
        self.assertIsInstance(texture_features, list)
        self.assertGreater(len(texture_features), 0)
        
        # Test edge features
        edge_features = classifier._extract_edge_features(img)
        self.assertIsInstance(edge_features, list)
        self.assertGreater(len(edge_features), 0)
        
        # Test shape features
        shape_features = classifier._extract_shape_features(img)
        self.assertIsInstance(shape_features, list)
        self.assertGreater(len(shape_features), 0)
        
        # Test statistical features
        stat_features = classifier._extract_statistical_features(img)
        self.assertIsInstance(stat_features, list)
        self.assertGreater(len(stat_features), 0)
        
    def test_normalize_features(self):
        """Test feature normalization"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Test with various feature arrays
        features1 = np.array([1, 2, 3, 4, 5])
        normalized1 = classifier._normalize_features(features1)
        self.assertTrue(np.all(normalized1 >= 0))
        self.assertTrue(np.all(normalized1 <= 1))
        
        # Test with extreme values
        features2 = np.array([0, 100, -100, 1000, -1000])
        normalized2 = classifier._normalize_features(features2)
        self.assertTrue(np.all(normalized2 >= 0))
        self.assertTrue(np.all(normalized2 <= 1))
        
        # Test with all same values
        features3 = np.array([5, 5, 5, 5, 5])
        normalized3 = classifier._normalize_features(features3)
        self.assertEqual(len(normalized3), len(features3))
        
    def test_calculate_similarity(self):
        """Test similarity calculation"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Test identical features
        features1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        similarity1 = classifier.calculate_similarity(features1, features1)
        self.assertAlmostEqual(similarity1, 1.0, places=2)
        
        # Test different features
        features2 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        similarity2 = classifier.calculate_similarity(features1, features2)
        self.assertGreater(similarity2, 0)
        self.assertLess(similarity2, 1)
        
        # Test with None
        similarity3 = classifier.calculate_similarity(None, features1)
        self.assertEqual(similarity3, 0.0)
        
        # Test different dimensions
        features3 = np.array([0.1, 0.2, 0.3])
        similarity4 = classifier.calculate_similarity(features1, features3)
        self.assertGreater(similarity4, 0)
        
    def test_create_folder_structure(self):
        """Test folder structure creation"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        base_path = self.temp_dir
        components = {
            "connector_type": "fc",
            "core_diameter": "50",
            "region": "core",
            "condition": "clean"
        }
        
        result_path = classifier.create_folder_structure(base_path, components)
        
        self.assertTrue(os.path.exists(result_path))
        self.assertIn("fc", result_path)
        self.assertIn("50", result_path)
        self.assertIn("core", result_path)
        self.assertIn("clean", result_path)
        
    def test_get_unique_filename(self):
        """Test unique filename generation"""
        classifier = self.UltimateImageClassifier(config_file=self.config_file)
        
        # Create existing file
        existing_file = os.path.join(self.temp_dir, "test.jpg")
        open(existing_file, 'w').close()
        
        # Test getting unique name
        unique_name = classifier.get_unique_filename(self.temp_dir, "test", ".jpg")
        self.assertEqual(unique_name, "test_1.jpg")
        
        # Create that file too
        open(os.path.join(self.temp_dir, unique_name), 'w').close()
        
        # Test again
        unique_name2 = classifier.get_unique_filename(self.temp_dir, "test", ".jpg")
        self.assertEqual(unique_name2, "test_2.jpg")

class TestImageProcessing(unittest.TestCase):
    """Test image processing workflows"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.reference_dir = os.path.join(self.temp_dir, "reference")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        os.makedirs(self.reference_dir)
        os.makedirs(self.dataset_dir)
        
        # Create reference structure
        self.create_reference_structure()
        
        # Import after setup
        from image_classifier import UltimateImageClassifier
        self.UltimateImageClassifier = UltimateImageClassifier
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def create_reference_structure(self):
        """Create reference folder structure with images"""
        # Create folder hierarchy
        structures = [
            "fc/50/core/clean",
            "fc/50/core/dirty",
            "fc/91/cladding/clean",
            "sma/50/ferrule/scratched",
            "sma/91/core/oil"
        ]
        
        for structure in structures:
            path = os.path.join(self.reference_dir, structure)
            os.makedirs(path, exist_ok=True)
            
            # Create test image in each folder
            img_path = os.path.join(path, f"ref_{structure.replace('/', '_')}.jpg")
            self.create_test_image(img_path)
            
    def create_test_image(self, path, size=(128, 128)):
        """Create a test image with random colors"""
        # Create image with some variation
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(path)
        return path
        
    def test_analyze_reference_folder(self):
        """Test reference folder analysis"""
        classifier = self.UltimateImageClassifier()
        
        reference_data = classifier.analyze_reference_folder(self.reference_dir)
        
        self.assertIsInstance(reference_data, dict)
        self.assertGreater(len(reference_data), 0)
        
        # Check knowledge bank was populated
        self.assertGreater(len(classifier.knowledge_bank.features_db), 0)
        self.assertGreater(len(classifier.knowledge_bank.classifications_db), 0)
        
    def test_find_similar_images(self):
        """Test finding similar images"""
        classifier = self.UltimateImageClassifier()
        
        # Analyze reference first
        classifier.analyze_reference_folder(self.reference_dir)
        
        # Create test features
        test_features = np.random.rand(100)
        
        similar = classifier.find_similar_images(test_features, top_k=5)
        
        self.assertIsInstance(similar, list)
        self.assertLessEqual(len(similar), 5)
        
        if similar:
            # Check structure of results
            for item in similar:
                self.assertIn('hash', item)
                self.assertIn('similarity', item)
                self.assertIn('classifications', item)
                self.assertIn('characteristics', item)
                self.assertIsInstance(item['similarity'], float)
                
    def test_classify_image(self):
        """Test image classification"""
        classifier = self.UltimateImageClassifier()
        
        # Analyze reference first
        classifier.analyze_reference_folder(self.reference_dir)
        
        # Create test image
        test_img_path = os.path.join(self.dataset_dir, "test_classify.jpg")
        self.create_test_image(test_img_path)
        
        # Classify
        classification, components, confidence = classifier.classify_image(test_img_path)
        
        # May or may not find a match depending on random image
        if classification:
            self.assertIsInstance(classification, str)
            self.assertIsInstance(components, dict)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            
    def test_apply_classification(self):
        """Test applying classification (renaming/moving file)"""
        classifier = self.UltimateImageClassifier()
        
        # Create test image
        test_img_path = os.path.join(self.dataset_dir, "test_apply.jpg")
        self.create_test_image(test_img_path)
        
        # Apply classification
        classification = "50-fc-core-clean"
        components = {
            "core_diameter": "50",
            "connector_type": "fc",
            "region": "core",
            "condition": "clean"
        }
        
        success = classifier._apply_classification(
            test_img_path, classification, components, self.reference_dir
        )
        
        self.assertTrue(success)
        
        # Check file was moved/renamed
        self.assertFalse(os.path.exists(test_img_path))
        
        # Check new file exists
        expected_files = [
            os.path.join(self.dataset_dir, "fc", "50", "core", "clean", f"{classification}.jpg"),
            os.path.join(self.dataset_dir, f"{classification}.jpg")
        ]
        
        file_exists = any(os.path.exists(f) for f in expected_files)
        self.assertTrue(file_exists)

class TestInteractiveMode(unittest.TestCase):
    """Test interactive mode and user inputs"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
        
    @patch('sys.argv', ['image-classifier.py', '--mode', 'auto'])
    def test_main_auto_mode(self):
        """Test main function in auto mode"""
        # Create directories
        os.makedirs('reference')
        os.makedirs('dataset')
        
        # Create a reference image
        img = Image.new('RGB', (100, 100), color='red')
        img.save('reference/test_ref.jpg')
        
        # Create a dataset image
        img.save('dataset/test_data.jpg')
        
        # Import and run main
        from image_classifier import main
        
        # Should complete without error
        main()
        
    @patch('sys.argv', ['image-classifier.py', '--mode', 'exit'])
    def test_main_exit_mode(self):
        """Test main function in exit mode"""
        # Create reference directory
        os.makedirs('reference')
        
        # Create a reference image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save('reference/test_ref.jpg')
        
        from image_classifier import main
        
        # Should complete without error
        main()
        
    @patch('builtins.input', side_effect=['y'])
    @patch('sys.argv', ['image-classifier.py'])
    def test_main_create_reference_folder(self, mock_input):
        """Test main function creating reference folder on prompt"""
        from image_classifier import main
        
        # Should prompt to create folder and exit
        main()
        
        # Check folder was created
        self.assertTrue(os.path.exists('reference'))
        
    def test_command_line_arguments(self):
        """Test command line argument parsing"""
        test_args = [
            'image-classifier.py',
            '--reference_folder', '/custom/ref',
            '--dataset_folder', '/custom/data', 
            '--mode', 'manual',
            '--similarity_threshold', '0.8',
            '--auto_create_folders', 'False',
            '--custom_keywords', 'keyword1', 'keyword2'
        ]
        
        with patch('sys.argv', test_args):
            parser = argparse.ArgumentParser(description="Ultimate Image Classifier")
            parser.add_argument("--reference_folder", type=str, default='reference')
            parser.add_argument("--dataset_folder", type=str, default='dataset')
            parser.add_argument("--mode", type=str, choices=["auto", "manual", "exit"], default="auto")
            parser.add_argument("--similarity_threshold", type=float, default=0.65)
            parser.add_argument("--auto_create_folders", type=bool, default=True)
            parser.add_argument("--custom_keywords", nargs='*', default=[])
            
            args = parser.parse_args()
            
            self.assertEqual(args.reference_folder, '/custom/ref')
            self.assertEqual(args.dataset_folder, '/custom/data')
            self.assertEqual(args.mode, 'manual')
            self.assertEqual(args.similarity_threshold, 0.8)
            self.assertEqual(args.custom_keywords, ['keyword1', 'keyword2'])

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        from image_classifier import UltimateImageClassifier
        self.UltimateImageClassifier = UltimateImageClassifier
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_invalid_image_path(self):
        """Test handling of invalid image paths"""
        classifier = self.UltimateImageClassifier()
        
        features, img_hash = classifier.extract_features("/nonexistent/image.jpg")
        self.assertIsNone(features)
        self.assertIsNone(img_hash)
        
    def test_corrupted_image(self):
        """Test handling of corrupted image files"""
        classifier = self.UltimateImageClassifier()
        
        # Create corrupted image file
        corrupted_path = os.path.join(self.temp_dir, "corrupted.jpg")
        with open(corrupted_path, 'w') as f:
            f.write("This is not an image")
            
        features, img_hash = classifier.extract_features(corrupted_path)
        self.assertIsNone(features)
        self.assertIsNone(img_hash)
        
    def test_empty_reference_folder(self):
        """Test handling of empty reference folder"""
        classifier = self.UltimateImageClassifier()
        
        empty_ref = os.path.join(self.temp_dir, "empty_ref")
        os.makedirs(empty_ref)
        
        reference_data = classifier.analyze_reference_folder(empty_ref)
        
        self.assertIsInstance(reference_data, dict)
        self.assertEqual(len(classifier.knowledge_bank.features_db), 0)
        
    def test_classification_with_no_references(self):
        """Test classification when no references exist"""
        classifier = self.UltimateImageClassifier()
        
        # Create test image
        test_img = os.path.join(self.temp_dir, "test.jpg")
        img = Image.new('RGB', (100, 100), color='green')
        img.save(test_img)
        
        classification, components, confidence = classifier.classify_image(test_img)
        
        self.assertIsNone(classification)
        self.assertIsNone(components)
        self.assertEqual(confidence, 0.0)
        
    def test_special_characters_in_classification(self):
        """Test handling of special characters in classification"""
        classifier = self.UltimateImageClassifier()
        
        # Test parsing with special characters
        special_cases = [
            "test@#$%",
            "test/with/slashes",
            "test\\with\\backslashes",
            "test with spaces",
            "test.with.dots"
        ]
        
        for case in special_cases:
            result = classifier.parse_classification(case)
            self.assertIsInstance(result, dict)
            
    def test_very_large_image(self):
        """Test handling of very large images"""
        classifier = self.UltimateImageClassifier()
        
        # Create large image (2000x2000)
        large_img_path = os.path.join(self.temp_dir, "large.jpg")
        large_img = Image.new('RGB', (2000, 2000), color='yellow')
        large_img.save(large_img_path)
        
        # Should handle without error
        features, img_hash = classifier.extract_features(large_img_path)
        
        if features is not None:
            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(len(features), 0)
            
    def test_concurrent_knowledge_bank_access(self):
        """Test concurrent access to knowledge bank"""
        from image_classifier import KnowledgeBank
        
        kb_path = os.path.join(self.temp_dir, "concurrent_kb.pkl")
        
        # Create and save initial data
        kb1 = KnowledgeBank(kb_path)
        kb1.add_image("hash1", np.array([1, 2, 3]), ["class1"], {})
        kb1.save()
        
        # Load in another instance
        kb2 = KnowledgeBank(kb_path)
        
        # Both should have the data
        self.assertIn("hash1", kb1.features_db)
        self.assertIn("hash1", kb2.features_db)
        
        # Add different data to each
        kb1.add_image("hash2", np.array([4, 5, 6]), ["class2"], {})
        kb2.add_image("hash3", np.array([7, 8, 9]), ["class3"], {})
        
        # Save both (second save will overwrite)
        kb1.save()
        kb2.save()
        
        # Load fresh instance
        kb3 = KnowledgeBank(kb_path)
        
        # Should have data from kb2 (last save)
        self.assertIn("hash1", kb3.features_db)
        self.assertIn("hash3", kb3.features_db)
        # May or may not have hash2 depending on timing


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    # Create test suite
    test_classes = [
        TestDependencies,
        TestLogging,
        TestKnowledgeBank,
        TestImageClassifier,
        TestImageProcessing,
        TestInteractiveMode,
        TestEdgeCases
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"\n- {test}")
                print(traceback)
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"\n- {test}")
                print(traceback)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)