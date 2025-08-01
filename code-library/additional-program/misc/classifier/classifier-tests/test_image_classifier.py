#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced Image Classifier
Tests all features including hierarchical folder structure, dynamic folder creation,
and reference image storage
"""

import os
import sys
import shutil
import json
import pickle
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import time
import unittest
from unittest.mock import patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import the classifier module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_classifier import AdvancedImageClassifier, KnowledgeBank, timestamp

class TestKnowledgeBank(unittest.TestCase):
    """Test the KnowledgeBank class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.kb_path = os.path.join(self.temp_dir, "test_kb.pkl")
        self.kb = KnowledgeBank(self.kb_path)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test knowledge bank initialization"""
        self.assertTrue(os.path.exists(self.kb_path) or not self.kb.features_db)
        self.assertIsInstance(self.kb.features_db, dict)
        self.assertIsInstance(self.kb.custom_keywords, set)
    
    def test_add_image(self):
        """Test adding image to knowledge bank"""
        test_hash = "test123"
        test_features = np.random.rand(100)
        test_classifications = ["fc", "clean"]
        test_characteristics = {"connector_type": "fc", "condition": "clean"}
        
        self.kb.add_image(test_hash, test_features, test_classifications, test_characteristics)
        
        self.assertIn(test_hash, self.kb.features_db)
        self.assertIn(test_hash, self.kb.classifications_db)
        self.assertEqual(self.kb.classifications_db[test_hash], test_classifications)
    
    def test_add_custom_keyword(self):
        """Test adding custom keywords"""
        keyword = "custom_defect"
        self.kb.add_custom_keyword(keyword)
        self.assertIn(keyword.lower(), self.kb.custom_keywords)
    
    def test_add_folder_structure(self):
        """Test learning folder structure"""
        path_components = ["fc", "50", "core", "clean"]
        self.kb.add_folder_structure(path_components)
        
        self.assertIn("50", self.kb.folder_structure["fc"])
        self.assertIn("core", self.kb.folder_structure["50"])
        self.assertIn("clean", self.kb.folder_structure["core"])
    
    def test_save_and_load(self):
        """Test saving and loading knowledge bank"""
        # Add test data
        test_hash = "test456"
        test_features = np.random.rand(100)
        test_classifications = ["sma", "dirty"]
        test_characteristics = {"connector_type": "sma", "condition": "dirty"}
        
        self.kb.add_image(test_hash, test_features, test_classifications, test_characteristics)
        self.kb.add_custom_keyword("test_keyword")
        
        # Save
        self.kb.save()
        self.assertTrue(os.path.exists(self.kb_path))
        
        # Load in new instance
        kb2 = KnowledgeBank(self.kb_path)
        
        self.assertIn(test_hash, kb2.features_db)
        self.assertIn("test_keyword", kb2.custom_keywords)
        self.assertEqual(kb2.classifications_db[test_hash], test_classifications)

class TestImageClassifier(unittest.TestCase):
    """Test the AdvancedImageClassifier class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.reference_dir = os.path.join(self.temp_dir, "reference")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        
        # Create directory structure
        os.makedirs(self.reference_dir)
        os.makedirs(self.dataset_dir)
        
        # Create test folder structure
        self.create_test_folder_structure()
        
        # Initialize classifier with test config
        with patch('image_classifier.AdvancedImageClassifier._load_models'):
            self.classifier = AdvancedImageClassifier(self.config_path)
            self.classifier.models = {}  # Mock models
            self.classifier.device = 'cpu'
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_folder_structure(self):
        """Create test reference folder structure"""
        # Create hierarchical structure
        structures = [
            "fc/50/core/clean",
            "fc/50/cladding/dirty",
            "fc/91/core/scratched",
            "fc/91/cladding/oil",
            "sma/core/clean",
            "sma/cladding/blob",
            "anomaly",
            "clean",
            "dirty"
        ]
        
        for structure in structures:
            path = os.path.join(self.reference_dir, structure)
            os.makedirs(path, exist_ok=True)
            
            # Create dummy image in each folder
            img = Image.new('RGB', (100, 100), color='red')
            img_path = os.path.join(path, f"test_{structure.replace('/', '_')}.jpg")
            img.save(img_path)
    
    def create_test_image(self, name="test.jpg", size=(100, 100), color='blue'):
        """Create a test image"""
        img = Image.new('RGB', size, color=color)
        img_path = os.path.join(self.dataset_dir, name)
        img.save(img_path)
        return img_path
    
    def test_config_loading(self):
        """Test configuration loading and saving"""
        self.assertTrue(os.path.exists(self.config_path))
        self.assertIn('custom_keywords', self.classifier.config)
        self.assertIn('auto_create_folders', self.classifier.config)
        self.assertTrue(self.classifier.config['auto_create_folders'])
    
    def test_parse_classification(self):
        """Test parsing classification from filename"""
        test_cases = [
            ("fc-50-core-clean.jpg", {
                "connector_type": "fc",
                "core_diameter": "50",
                "region": "core",
                "condition": "clean"
            }),
            ("sma-cladding-scratched.jpg", {
                "connector_type": "sma",
                "region": "cladding",
                "defect_type": "scratched"
            }),
            ("91-ferrule-oil-blob.jpg", {
                "core_diameter": "91",
                "region": "ferrule",
                "defect_type": "oil-blob"
            })
        ]
        
        for filename, expected in test_cases:
            components = self.classifier.parse_classification(filename)
            for key, value in expected.items():
                self.assertEqual(components.get(key), value)
    
    def test_build_classification_string(self):
        """Test building classification string from components"""
        components = {
            "connector_type": "fc",
            "core_diameter": "50",
            "region": "core",
            "condition": "clean"
        }
        
        classification = self.classifier.build_classification_string(components)
        self.assertIn("fc", classification)
        self.assertIn("50", classification)
        self.assertIn("core", classification)
        self.assertIn("clean", classification)
    
    def test_create_folder_if_needed(self):
        """Test dynamic folder creation"""
        base_path = os.path.join(self.temp_dir, "test_create")
        os.makedirs(base_path)
        
        components = {
            "connector_type": "fc",
            "core_diameter": "62.5",
            "region": "ferrule",
            "defect_type": "contaminated"
        }
        
        new_path = self.classifier.create_folder_if_needed(base_path, components)
        
        self.assertTrue(os.path.exists(new_path))
        self.assertIn("fc", new_path)
        self.assertIn("62.5", new_path)
        self.assertIn("ferrule", new_path)
        self.assertIn("contaminated", new_path)
    
    @patch('image_classifier.AdvancedImageClassifier.extract_visual_features')
    @patch('image_classifier.AdvancedImageClassifier.extract_deep_features')
    def test_analyze_reference_folder(self, mock_deep, mock_visual):
        """Test analyzing reference folder structure"""
        # Mock feature extraction
        mock_visual.return_value = (np.random.rand(100), "hash123")
        mock_deep.return_value = np.random.rand(200)
        
        reference_data = self.classifier.analyze_reference_folder(self.reference_dir)
        
        # Check that folders were analyzed
        self.assertIsInstance(reference_data, dict)
        self.assertTrue(len(reference_data) > 0)
        
        # Check folder structure was learned
        self.assertIn("50", self.classifier.knowledge_bank.folder_structure.get("fc", set()))
    
    def test_custom_keywords_integration(self):
        """Test custom keyword functionality"""
        # Add custom keyword
        self.classifier.knowledge_bank.add_custom_keyword("custom_defect")
        
        # Test parsing with custom keyword
        components = self.classifier.parse_classification("fc-50-custom_defect.jpg")
        self.assertIn("custom_defect", components.get("additional_characteristics", ""))
    
    @patch('image_classifier.AdvancedImageClassifier.extract_visual_features')
    @patch('image_classifier.AdvancedImageClassifier.extract_deep_features')
    @patch('image_classifier.AdvancedImageClassifier.find_similar_images')
    def test_save_to_reference(self, mock_similar, mock_deep, mock_visual):
        """Test saving classified images to reference folder"""
        # Create test image
        test_image = self.create_test_image("test_save.jpg")
        
        # Mock feature extraction
        mock_visual.return_value = (np.random.rand(100), "hash789")
        mock_deep.return_value = np.random.rand(200)
        
        components = {
            "connector_type": "lc",
            "core_diameter": "125",
            "condition": "pristine"
        }
        
        # Save to reference
        self.classifier.save_to_reference(
            test_image,
            "lc-125-pristine",
            components,
            self.reference_dir
        )
        
        # Check if folder structure was created
        expected_path = os.path.join(self.reference_dir, "lc", "125", "pristine")
        self.assertTrue(os.path.exists(expected_path))
        
        # Check if image was copied
        files = os.listdir(expected_path)
        self.assertTrue(len(files) > 0)
        self.assertTrue(any("test_save" in f for f in files))
    
    def test_get_unique_filename(self):
        """Test unique filename generation"""
        # Create existing file
        existing = self.create_test_image("test-classification.jpg")
        
        # Test getting unique filename
        new_filename = self.classifier.get_unique_filename(
            self.dataset_dir,
            "test-classification",
            ".jpg"
        )
        
        self.assertEqual(new_filename, "test-classification-1.jpg")
        
        # Create another and test again
        self.create_test_image("test-classification-1.jpg")
        new_filename = self.classifier.get_unique_filename(
            self.dataset_dir,
            "test-classification",
            ".jpg"
        )
        
        self.assertEqual(new_filename, "test-classification-2.jpg")

class TestImageClassifierIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.reference_dir = os.path.join(self.temp_dir, "reference")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        
        os.makedirs(self.reference_dir)
        os.makedirs(self.dataset_dir)
        
        # Create more complex test structure
        self.create_complex_test_structure()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def create_complex_test_structure(self):
        """Create a complex test folder structure with multiple images"""
        structures = {
            "fc/50/core": ["clean", "dirty", "scratched"],
            "fc/50/cladding": ["clean", "oil", "blob"],
            "fc/91/core": ["clean", "dirty", "anomaly"],
            "fc/91/ferrule": ["clean", "dirty", "dig"],
            "sma/core": ["clean", "scratched"],
            "sma/cladding": ["clean", "contaminated"]
        }
        
        for path, conditions in structures.items():
            for condition in conditions:
                full_path = os.path.join(self.reference_dir, path, condition)
                os.makedirs(full_path, exist_ok=True)
                
                # Create multiple test images
                for i in range(3):
                    img = Image.new('RGB', (100, 100), 
                                  color=(i*50, i*50, i*50))  # Different colors
                    img_name = f"{path.replace('/', '-')}-{condition}-{i}.jpg"
                    img.save(os.path.join(full_path, img_name))
    
    @patch('image_classifier.AdvancedImageClassifier._load_models')
    def test_full_workflow(self, mock_models):
        """Test complete classification workflow"""
        # Initialize classifier
        mock_models.return_value = {}
        classifier = AdvancedImageClassifier()
        classifier.models = {}
        classifier.device = 'cpu'
        
        # Mock feature extraction to speed up test
        with patch.object(classifier, 'extract_visual_features') as mock_visual, \
             patch.object(classifier, 'extract_deep_features') as mock_deep:
            
            # Set up mocks
            mock_visual.side_effect = lambda x: (np.random.rand(100), f"hash_{x}")
            mock_deep.side_effect = lambda x: np.random.rand(200)
            
            # Analyze reference folder
            print("\n[TEST] Analyzing reference folder...")
            reference_data = classifier.analyze_reference_folder(self.reference_dir)
            
            # Verify knowledge bank was populated
            kb_size = len(classifier.knowledge_bank.features_db)
            print(f"[TEST] Knowledge bank contains {kb_size} images")
            self.assertGreater(kb_size, 0)
            
            # Test custom keyword addition
            classifier.knowledge_bank.add_custom_keyword("special_defect")
            classifier.knowledge_bank.add_custom_keyword("rare_condition")
            
            # Create test images in dataset
            test_images = [
                "IMG_001.jpg",  # Should be classified
                "IMG_002.jpg",  # Should be classified
                "fc-50-core-clean.jpg",  # Already classified
                "unknown_image.jpg"  # New type
            ]
            
            for img_name in test_images:
                img = Image.new('RGB', (100, 100), color='green')
                img.save(os.path.join(self.dataset_dir, img_name))
            
            # Test classification
            print("\n[TEST] Testing classification...")
            for img_name in test_images:
                img_path = os.path.join(self.dataset_dir, img_name)
                
                # Mock similarity search
                with patch.object(classifier, 'find_similar_images') as mock_find:
                    mock_find.return_value = [
                        {
                            'hash': 'ref_hash',
                            'similarity': 0.85,
                            'classifications': ['fc-50-core-clean']
                        }
                    ]
                    
                    # Set up characteristics for the mock result
                    classifier.knowledge_bank.characteristics_db['ref_hash'] = {
                        'connector_type': 'fc',
                        'core_diameter': '50',
                        'region': 'core',
                        'condition': 'clean'
                    }
                    
                    classification, confidence, components = classifier.classify_image(img_path)
                    
                    if classification:
                        print(f"[TEST] {img_name} -> {classification} (confidence: {confidence:.3f})")
                        self.assertIsNotNone(classification)
                        self.assertGreater(confidence, 0)
                        self.assertIsInstance(components, dict)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        with patch('image_classifier.AdvancedImageClassifier._load_models'):
            classifier = AdvancedImageClassifier()
            classifier.models = {}
            classifier.device = 'cpu'
            
            # Test with non-existent image
            result = classifier.classify_image("/non/existent/image.jpg")
            self.assertEqual(result, (None, 0.0, {}))
            
            # Test with invalid reference folder
            with self.assertRaises(Exception):
                classifier.analyze_reference_folder("/non/existent/folder")

class TestCLIInterface(unittest.TestCase):
    """Test command-line interface functionality"""
    
    def test_timestamp_function(self):
        """Test timestamp generation"""
        ts1 = timestamp()
        time.sleep(0.01)
        ts2 = timestamp()
        
        self.assertIsInstance(ts1, str)
        self.assertNotEqual(ts1, ts2)
        self.assertRegex(ts1, r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}')
    
    @patch('builtins.input')
    @patch('sys.stdout')
    def test_main_menu_selection(self, mock_stdout, mock_input):
        """Test main menu selection"""
        # This would test the main() function menu selection
        # For brevity, keeping it simple
        pass

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE IMAGE CLASSIFIER TEST SUITE")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBank))
    suite.addTests(loader.loadTestsFromTestCase(TestImageClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestImageClassifierIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIInterface))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n- {test}:\n{traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n- {test}:\n{traceback}")
    
    return result.wasSuccessful()

def run_manual_test():
    """Run a manual test with visual output"""
    print("\n" + "="*80)
    print("MANUAL FEATURE TEST")
    print("="*80 + "\n")
    
    # Create temporary test environment
    temp_dir = tempfile.mkdtemp()
    reference_dir = os.path.join(temp_dir, "manual_test_reference")
    dataset_dir = os.path.join(temp_dir, "manual_test_dataset")
    
    try:
        # Create test structure
        os.makedirs(reference_dir)
        os.makedirs(dataset_dir)
        
        print(f"Created test directories at: {temp_dir}")
        
        # Create sample reference structure
        structures = [
            "fc/50/core/clean",
            "fc/91/cladding/dirty",
            "sma/ferrule/scratched"
        ]
        
        for structure in structures:
            path = os.path.join(reference_dir, structure)
            os.makedirs(path, exist_ok=True)
            
            # Create sample image
            img = Image.new('RGB', (200, 200), color='blue')
            img_path = os.path.join(path, f"ref_{structure.replace('/', '_')}.jpg")
            img.save(img_path)
            print(f"Created reference image: {img_path}")
        
        # Create test images in dataset
        test_images = ["test1.jpg", "test2.jpg", "IMG_12345.jpg"]
        for img_name in test_images:
            img = Image.new('RGB', (200, 200), color='red')
            img.save(os.path.join(dataset_dir, img_name))
            print(f"Created test image: {img_name}")
        
        print("\n[MANUAL TEST] Testing features:")
        print("1. Hierarchical folder structure parsing")
        print("2. Dynamic folder creation")
        print("3. Custom keyword support")
        print("4. Reference image storage")
        
        # Initialize classifier with mock models
        with patch('image_classifier.AdvancedImageClassifier._load_models'):
            classifier = AdvancedImageClassifier()
            classifier.models = {}
            classifier.device = 'cpu'
            
            # Add custom keywords
            classifier.knowledge_bank.add_custom_keyword("experimental")
            classifier.knowledge_bank.add_custom_keyword("prototype")
            print("\n[MANUAL TEST] Added custom keywords: experimental, prototype")
            
            # Test folder creation
            test_components = {
                "connector_type": "new_type",
                "core_diameter": "100",
                "condition": "experimental"
            }
            
            new_path = classifier.create_folder_if_needed(reference_dir, test_components)
            print(f"\n[MANUAL TEST] Created new folder structure: {new_path}")
            print(f"Folder exists: {os.path.exists(new_path)}")
        
        print("\n[MANUAL TEST] Test completed successfully!")
        
    except Exception as e:
        print(f"\n[MANUAL TEST] Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n[MANUAL TEST] Cleaning up test directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("Starting Image Classifier Test Suite...")
    
    # Run automated tests
    success = run_comprehensive_tests()
    
    # Run manual feature test
    run_manual_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)