#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Automated Processing Studio
========================================================
Tests all components and functionality of the automated image processing system.
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automated_processing_studio import (
    DependencyManager, ImageProcessor, ScriptManager,
    ReinforcementLearner, AnomalyDetector, AutomatedProcessingStudio
)


class TestDependencyManager(unittest.TestCase):
    """Test dependency management functionality"""
    
    def test_check_dependencies(self):
        """Test that required packages are checked"""
        # Just verify the method runs without error
        # In real test environment, all deps should be installed
        try:
            DependencyManager.check_and_install_dependencies()
        except Exception as e:
            self.fail(f"Dependency check failed: {e}")
    
    def test_required_packages_list(self):
        """Test that required packages are properly defined"""
        self.assertIn('opencv-python', DependencyManager.REQUIRED_PACKAGES)
        self.assertIn('numpy', DependencyManager.REQUIRED_PACKAGES)
        self.assertIn('scikit-learn', DependencyManager.REQUIRED_PACKAGES)


class TestImageProcessor(unittest.TestCase):
    """Test image processing functionality"""
    
    def setUp(self):
        self.processor = ImageProcessor()
        # Create test images
        self.color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.float_image = np.random.random((100, 100)).astype(np.float32)
    
    def test_normalize_image(self):
        """Test image normalization"""
        # Test uint8 image (should remain unchanged)
        normalized = self.processor.normalize_image(self.color_image)
        self.assertEqual(normalized.dtype, np.uint8)
        np.testing.assert_array_equal(normalized, self.color_image)
        
        # Test float image
        normalized = self.processor.normalize_image(self.float_image)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(normalized.min() >= 0)
        self.assertTrue(normalized.max() <= 255)
    
    def test_to_grayscale(self):
        """Test grayscale conversion"""
        # Test color to gray
        gray = self.processor.to_grayscale(self.color_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))
        
        # Test already gray
        gray2 = self.processor.to_grayscale(self.gray_image)
        np.testing.assert_array_equal(gray2, self.gray_image)
    
    def test_extract_features(self):
        """Test feature extraction"""
        features = self.processor.extract_features(self.color_image)
        
        # Check all expected features are present
        expected_keys = ['shape', 'mean', 'std', 'histogram', 'edges', 'corners', 'gabor']
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Check feature properties
        self.assertEqual(features['shape'], self.color_image.shape)
        self.assertEqual(len(features['histogram']), 256)
        self.assertIsInstance(features['mean'], (float, np.floating))
        self.assertIsInstance(features['gabor'], np.ndarray)
    
    def test_calculate_similarity(self):
        """Test image similarity calculation"""
        # Identical images should have low similarity score (0 = identical)
        similarity = self.processor.calculate_similarity(self.gray_image, self.gray_image)
        self.assertLess(similarity, 0.1)
        
        # Different images should have higher score
        different_image = np.ones_like(self.gray_image) * 255
        similarity = self.processor.calculate_similarity(self.gray_image, different_image)
        self.assertGreater(similarity, 0.5)
        
        # Test with different sized images
        small_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        similarity = self.processor.calculate_similarity(self.gray_image, small_image)
        self.assertIsInstance(similarity, (float, np.floating))
    
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        # Test internal SSIM method
        ssim = self.processor._calculate_ssim(self.gray_image, self.gray_image)
        self.assertGreater(ssim, 0.9)  # Should be close to 1 for identical images
        
        # Test with different images
        noise = np.random.randint(0, 50, self.gray_image.shape, dtype=np.uint8)
        noisy_image = np.clip(self.gray_image.astype(int) + noise, 0, 255).astype(np.uint8)
        ssim = self.processor._calculate_ssim(self.gray_image, noisy_image)
        self.assertLess(ssim, 0.9)
        self.assertGreater(ssim, 0)


class TestScriptManager(unittest.TestCase):
    """Test script management functionality"""
    
    def setUp(self):
        # Create temporary scripts directory
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.scripts_dir.mkdir()
        
        # Create test scripts
        self._create_test_scripts()
        
        self.manager = ScriptManager(str(self.scripts_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_scripts(self):
        """Create test processing scripts"""
        # Simple blur script
        blur_script = '''
import numpy as np
import cv2

def process_image(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(image, (5, 5), 0)
'''
        with open(self.scripts_dir / "gaussian_blur_test.py", 'w') as f:
            f.write(blur_script)
        
        # Edge detection script
        edge_script = '''
import numpy as np
import cv2

def process_image(image: np.ndarray) -> np.ndarray:
    """Detect edges"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, 50, 150)
'''
        with open(self.scripts_dir / "edge_detection_test.py", 'w') as f:
            f.write(edge_script)
        
        # Invalid script (no process_image function)
        invalid_script = '''
def some_other_function():
    pass
'''
        with open(self.scripts_dir / "invalid_test.py", 'w') as f:
            f.write(invalid_script)
    
    def test_load_scripts(self):
        """Test script loading"""
        # Should load valid scripts
        self.assertIn("gaussian_blur_test.py", self.manager.functions)
        self.assertIn("edge_detection_test.py", self.manager.functions)
        
        # Should not load invalid script
        self.assertNotIn("invalid_test.py", self.manager.functions)
        
        # Check function info
        self.assertIn("gaussian_blur_test.py", self.manager.function_info)
        info = self.manager.function_info["gaussian_blur_test.py"]
        self.assertEqual(info['category'], 'filtering')
    
    def test_categorize_script(self):
        """Test script categorization"""
        categories = {
            'blur_script.py': 'filtering',
            'edge_detect.py': 'edge_detection',
            'threshold_image.py': 'thresholding',
            'morph_ops.py': 'morphology',
            'enhance_contrast.py': 'enhancement',
            'detect_circles.py': 'detection',
            'grayscale_convert.py': 'color',
            'unknown_script.py': 'other'
        }
        
        for filename, expected_category in categories.items():
            category = self.manager._categorize_script(filename)
            self.assertEqual(category, expected_category)
    
    def test_get_scripts_by_category(self):
        """Test getting scripts organized by category"""
        categorized = self.manager.get_scripts_by_category()
        
        self.assertIn('filtering', categorized)
        self.assertIn('edge_detection', categorized)
        self.assertIn("gaussian_blur_test.py", categorized['filtering'])
        self.assertIn("edge_detection_test.py", categorized['edge_detection'])
    
    def test_script_execution(self):
        """Test that loaded scripts can be executed"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Test blur function
        blur_func = self.manager.functions["gaussian_blur_test.py"]
        result = blur_func(test_image)
        self.assertEqual(result.shape, test_image.shape)
        
        # Test edge function
        edge_func = self.manager.functions["edge_detection_test.py"]
        result = edge_func(test_image)
        self.assertEqual(len(result.shape), 2)  # Should be grayscale


class TestReinforcementLearner(unittest.TestCase):
    """Test reinforcement learning functionality"""
    
    def setUp(self):
        self.learner = ReinforcementLearner(state_size=10, action_size=5)
        self.test_state = np.random.random(10)
        self.test_next_state = np.random.random(10)
    
    def test_initialization(self):
        """Test learner initialization"""
        self.assertEqual(self.learner.state_size, 10)
        self.assertEqual(self.learner.action_size, 5)
        self.assertEqual(self.learner.epsilon, 1.0)
        self.assertEqual(len(self.learner.q_table), 0)  # Empty initially
    
    def test_state_key_generation(self):
        """Test state to key conversion"""
        key1 = self.learner.get_state_key(self.test_state)
        key2 = self.learner.get_state_key(self.test_state)
        self.assertEqual(key1, key2)  # Same state should give same key
        
        different_state = np.random.random(10)
        key3 = self.learner.get_state_key(different_state)
        self.assertNotEqual(key1, key3)  # Different states should give different keys
    
    def test_choose_action(self):
        """Test action selection"""
        # With epsilon = 1.0, should always explore (random action)
        actions = [self.learner.choose_action(self.test_state) for _ in range(10)]
        self.assertTrue(all(0 <= a < 5 for a in actions))
        
        # After learning, should exploit
        self.learner.epsilon = 0
        state_key = self.learner.get_state_key(self.test_state)
        self.learner.q_table[state_key][2] = 10  # Make action 2 best
        
        action = self.learner.choose_action(self.test_state)
        self.assertEqual(action, 2)
    
    def test_remember(self):
        """Test experience memory"""
        self.assertEqual(len(self.learner.memory), 0)
        
        self.learner.remember(self.test_state, 1, 0.5, self.test_next_state, False)
        self.assertEqual(len(self.learner.memory), 1)
        
        # Test memory limit
        for i in range(3000):
            self.learner.remember(self.test_state, i % 5, 0.1, self.test_next_state, False)
        
        self.assertEqual(len(self.learner.memory), 2000)  # Max size
    
    def test_learn(self):
        """Test Q-value updates"""
        action = 1
        reward = 1.0
        
        # Initial Q-value should be 0
        state_key = self.learner.get_state_key(self.test_state)
        self.assertEqual(self.learner.q_table[state_key][action], 0)
        
        # Learn from experience
        self.learner.learn(self.test_state, action, reward, self.test_next_state, False)
        
        # Q-value should be updated
        self.assertGreater(self.learner.q_table[state_key][action], 0)
        
        # Epsilon should decay
        self.assertLess(self.learner.epsilon, 1.0)
    
    def test_replay(self):
        """Test experience replay"""
        # Add experiences
        for i in range(50):
            self.learner.remember(
                np.random.random(10), i % 5, np.random.random(),
                np.random.random(10), False
            )
        
        # Replay should work without error
        initial_epsilon = self.learner.epsilon
        self.learner.replay(batch_size=32)
        
        # Epsilon should have decayed
        self.assertLess(self.learner.epsilon, initial_epsilon)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection functionality"""
    
    def setUp(self):
        self.detector = AnomalyDetector()
        self.processor = ImageProcessor()
        
        # Create test images
        self.normal_image = np.ones((100, 100), dtype=np.uint8) * 128
        self.anomaly_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Extract features
        self.normal_features = self.processor.extract_features(self.normal_image)
        self.anomaly_features = self.processor.extract_features(self.anomaly_image)
    
    def test_add_to_library(self):
        """Test adding entries to libraries"""
        self.assertEqual(len(self.detector.anomaly_library), 0)
        self.assertEqual(len(self.detector.similarity_library), 0)
        
        # Add anomaly
        self.detector.add_to_library(self.anomaly_image, self.anomaly_features, is_anomaly=True)
        self.assertEqual(len(self.detector.anomaly_library), 1)
        
        # Add normal
        self.detector.add_to_library(self.normal_image, self.normal_features, is_anomaly=False)
        self.assertEqual(len(self.detector.similarity_library), 1)
        
        # Check entry structure
        anomaly_entry = self.detector.anomaly_library[0]
        self.assertIn('timestamp', anomaly_entry)
        self.assertIn('features', anomaly_entry)
        self.assertIn('shape', anomaly_entry)
        self.assertIn('hash', anomaly_entry)
    
    def test_flatten_features(self):
        """Test feature flattening"""
        flat_features = self.detector._flatten_features(self.normal_features)
        
        self.assertIsInstance(flat_features, np.ndarray)
        self.assertEqual(len(flat_features.shape), 1)  # Should be 1D
        self.assertGreater(len(flat_features), 0)
    
    def test_compare_features(self):
        """Test feature comparison"""
        # Same features should have high similarity
        similarity = self.detector._compare_features(
            self.normal_features, self.normal_features
        )
        self.assertGreater(similarity, 0.9)
        
        # Different features should have lower similarity
        similarity = self.detector._compare_features(
            self.normal_features, self.anomaly_features
        )
        self.assertLess(similarity, 0.9)
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        # Add some normal samples
        for i in range(20):
            normal_variant = self.normal_image + np.random.randint(-10, 10, (100, 100))
            normal_variant = np.clip(normal_variant, 0, 255).astype(np.uint8)
            features = self.processor.extract_features(normal_variant)
            self.detector.add_to_library(normal_variant, features, is_anomaly=False)
        
        # Test anomaly detection
        result = self.detector.detect_anomalies(self.anomaly_image, self.anomaly_features)
        
        self.assertIn('is_anomaly', result)
        self.assertIn('anomaly_score', result)
        self.assertIn('reasons', result)
        self.assertIsInstance(result['is_anomaly'], bool)
        self.assertIsInstance(result['anomaly_score'], (float, np.floating))
    
    def test_find_similar(self):
        """Test finding similar images"""
        # Add some samples
        for i in range(10):
            variant = self.normal_image + np.random.randint(-5, 5, (100, 100))
            variant = np.clip(variant, 0, 255).astype(np.uint8)
            features = self.processor.extract_features(variant)
            self.detector.add_to_library(variant, features, is_anomaly=False)
        
        # Find similar
        similar = self.detector.find_similar(self.normal_features, top_k=5)
        
        self.assertLessEqual(len(similar), 5)
        if similar:
            self.assertIn('entry', similar[0])
            self.assertIn('similarity', similar[0])
            # Should be sorted by similarity
            for i in range(1, len(similar)):
                self.assertLessEqual(similar[i]['similarity'], similar[i-1]['similarity'])


class TestAutomatedProcessingStudio(unittest.TestCase):
    """Test the main automated processing studio"""
    
    def setUp(self):
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.scripts_dir.mkdir()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Create minimal test script
        test_script = '''
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Identity function for testing"""
    return image.copy()
'''
        with open(self.scripts_dir / "identity.py", 'w') as f:
            f.write(test_script)
        
        # Create studio instance
        self.studio = AutomatedProcessingStudio(
            scripts_dir=str(self.scripts_dir),
            cache_dir=str(self.cache_dir)
        )
        
        # Create test images
        self.input_image = np.ones((50, 50), dtype=np.uint8) * 100
        self.target_image = np.ones((50, 50), dtype=np.uint8) * 150
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test studio initialization"""
        self.assertIsNotNone(self.studio.processor)
        self.assertIsNotNone(self.studio.script_manager)
        self.assertIsNotNone(self.studio.anomaly_detector)
        self.assertIsNotNone(self.studio.learner)
        
        self.assertTrue(self.cache_dir.exists())
        self.assertEqual(len(self.studio.processing_history), 0)
    
    def test_state_persistence(self):
        """Test saving and loading state"""
        # Add some data
        self.studio.successful_combinations.append({'test': 'data'})
        self.studio.learner.epsilon = 0.5
        
        # Save state
        self.studio._save_state()
        
        # Create new instance and verify state is loaded
        new_studio = AutomatedProcessingStudio(
            scripts_dir=str(self.scripts_dir),
            cache_dir=str(self.cache_dir)
        )
        
        self.assertEqual(len(new_studio.successful_combinations), 1)
        self.assertEqual(new_studio.learner.epsilon, 0.5)
    
    def test_process_to_match_basic(self):
        """Test basic image matching process"""
        # Use very similar images for quick convergence
        input_img = np.ones((30, 30), dtype=np.uint8) * 100
        target_img = np.ones((30, 30), dtype=np.uint8) * 105  # Very close
        
        results = self.studio.process_to_match(
            input_img, target_img,
            max_iterations=10,
            similarity_threshold=0.5
        )
        
        self.assertIn('success', results)
        self.assertIn('iterations', results)
        self.assertIn('final_similarity', results)
        self.assertIn('pipeline', results)
        self.assertIn('processing_log', results)
        self.assertIn('final_image', results)
        
        self.assertGreater(results['iterations'], 0)
        self.assertIsInstance(results['final_image'], np.ndarray)
    
    def test_find_anomalies_and_similarities(self):
        """Test anomaly and similarity detection"""
        result = self.studio.find_anomalies_and_similarities(self.input_image)
        
        self.assertIn('anomaly_analysis', result)
        self.assertIn('similar_images', result)
        self.assertIn('features', result)
        
        anomaly_analysis = result['anomaly_analysis']
        self.assertIn('is_anomaly', anomaly_analysis)
        self.assertIn('anomaly_score', anomaly_analysis)
        self.assertIn('reasons', anomaly_analysis)
    
    def test_report_generation(self):
        """Test report generation"""
        results = {
            'success': True,
            'iterations': 5,
            'final_similarity': 0.05,
            'pipeline': ['script1', 'script2'],
            'processing_time': 2.5,
            'anomalies_detected': [],
            'processing_log': []
        }
        
        # Generate report
        self.studio._generate_report(
            results, self.input_image, self.target_image, self.input_image
        )
        
        # Check report files exist
        report_dirs = list(self.cache_dir.glob("report_*"))
        self.assertEqual(len(report_dirs), 1)
        
        report_dir = report_dirs[0]
        self.assertTrue((report_dir / "input.png").exists())
        self.assertTrue((report_dir / "target.png").exists())
        self.assertTrue((report_dir / "output.png").exists())
        self.assertTrue((report_dir / "comparison.png").exists())
        self.assertTrue((report_dir / "report.txt").exists())
        self.assertTrue((report_dir / "processing_log.json").exists())
    
    @patch('builtins.input')
    def test_interactive_setup(self, mock_input):
        """Test interactive configuration"""
        # Mock user inputs
        mock_input.side_effect = [
            'test_input.png',  # Input path
            'test_target.png',  # Target path
            '50',  # Max iterations
            '0.2',  # Similarity threshold
            'y'  # Anomaly detection
        ]
        
        # Create dummy files
        Path('test_input.png').touch()
        Path('test_target.png').touch()
        
        try:
            config = self.studio.interactive_setup()
            
            self.assertEqual(config['input_path'], 'test_input.png')
            self.assertEqual(config['target_path'], 'test_target.png')
            self.assertEqual(config['max_iterations'], 50)
            self.assertEqual(config['similarity_threshold'], 0.2)
            self.assertTrue(config['check_anomalies'])
        finally:
            # Cleanup
            Path('test_input.png').unlink(missing_ok=True)
            Path('test_target.png').unlink(missing_ok=True)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        # Create temporary setup
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.scripts_dir.mkdir()
        
        # Create a variety of test scripts
        self._create_test_script_library()
        
        # Create studio
        self.studio = AutomatedProcessingStudio(
            scripts_dir=str(self.scripts_dir),
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _create_test_script_library(self):
        """Create a library of test processing scripts"""
        scripts = {
            'brighten.py': '''
import numpy as np
def process_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(int) + 20, 0, 255).astype(np.uint8)
''',
            'darken.py': '''
import numpy as np
def process_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(int) - 20, 0, 255).astype(np.uint8)
''',
            'blur.py': '''
import numpy as np
import cv2
def process_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), 0)
''',
            'sharpen.py': '''
import numpy as np
import cv2
def process_image(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)
'''
        }
        
        for name, content in scripts.items():
            with open(self.scripts_dir / name, 'w') as f:
                f.write(content)
    
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline"""
        # Create input and target images
        input_image = np.ones((50, 50), dtype=np.uint8) * 100
        # Target is brighter
        target_image = np.ones((50, 50), dtype=np.uint8) * 120
        
        # Process
        results = self.studio.process_to_match(
            input_image, target_image,
            max_iterations=20,
            similarity_threshold=0.1
        )
        
        # Verify results
        self.assertIn('final_image', results)
        self.assertLess(results['final_similarity'], 0.5)
        
        # The system should have found that 'brighten.py' helps
        self.assertIn('brighten.py', results['pipeline'])
        
        # Check learning occurred
        self.assertLess(self.studio.learner.epsilon, 1.0)
    
    def test_anomaly_detection_integration(self):
        """Test anomaly detection in full pipeline"""
        # Create normal image
        normal_image = np.ones((50, 50), dtype=np.uint8) * 128
        
        # Add several normal samples
        for i in range(10):
            variant = normal_image + np.random.randint(-5, 5, (50, 50))
            variant = np.clip(variant, 0, 255).astype(np.uint8)
            features = self.studio.processor.extract_features(variant)
            self.studio.anomaly_detector.add_to_library(variant, features, is_anomaly=False)
        
        # Create anomalous image (very different pattern)
        anomaly_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # Detect anomalies
        results = self.studio.find_anomalies_and_similarities(anomaly_image)
        
        # Should detect as anomaly (depending on randomness)
        self.assertIn('anomaly_analysis', results)
        self.assertIn('similar_images', results)


class MasterTestSuite(unittest.TestCase):
    """Master test suite that runs all tests and validates the system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for master test suite"""
        print("\n" + "="*60)
        print("MASTER TEST SUITE - Automated Processing Studio")
        print("="*60)
    
    def test_all_components_exist(self):
        """Verify all required components exist"""
        required_classes = [
            DependencyManager,
            ImageProcessor,
            ScriptManager,
            ReinforcementLearner,
            AnomalyDetector,
            AutomatedProcessingStudio
        ]
        
        for cls in required_classes:
            self.assertTrue(
                hasattr(cls, '__init__'),
                f"{cls.__name__} class not properly defined"
            )
    
    def test_system_requirements(self):
        """Test that all system requirements are met"""
        # Test 1: RAM-based processing (no disk I/O during processing)
        processor = ImageProcessor()
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # All operations should work in memory
        normalized = processor.normalize_image(test_image)
        gray = processor.to_grayscale(test_image)
        features = processor.extract_features(test_image)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertIsInstance(gray, np.ndarray)
        self.assertIsInstance(features, dict)
    
    def test_learning_capability(self):
        """Test that the system has learning capabilities"""
        learner = ReinforcementLearner(state_size=10, action_size=5)
        
        # Test that it can learn from experience
        initial_epsilon = learner.epsilon
        state = np.random.random(10)
        
        for i in range(10):
            action = learner.choose_action(state)
            learner.learn(state, action, 1.0, state, False)
        
        self.assertLess(learner.epsilon, initial_epsilon)
        self.assertGreater(len(learner.q_table), 0)
    
    def test_comprehensive_functionality(self):
        """Test that all required functionality is present"""
        with tempfile.TemporaryDirectory() as temp_dir:
            studio = AutomatedProcessingStudio(
                scripts_dir=temp_dir,
                cache_dir=os.path.join(temp_dir, "cache")
            )
            
            # Test all required methods exist
            required_methods = [
                'process_to_match',
                'find_anomalies_and_similarities',
                '_generate_report',
                'interactive_setup',
                '_save_state',
                '_load_state'
            ]
            
            for method in required_methods:
                self.assertTrue(
                    hasattr(studio, method),
                    f"Required method '{method}' not found"
                )


def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDependencyManager,
        TestImageProcessor,
        TestScriptManager,
        TestReinforcementLearner,
        TestAnomalyDetector,
        TestAutomatedProcessingStudio,
        TestIntegration,
        MasterTestSuite
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)