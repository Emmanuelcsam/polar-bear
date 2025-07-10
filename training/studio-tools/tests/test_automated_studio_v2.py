#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Automated Image Processing Studio V2
================================================================
"""

import unittest
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the fixed module
from automated_processing_studio_v2_fixed import (
    EnhancedImageProcessor,
    ImprovedScriptManager,
    SmartLearner,
    EnhancedProcessingStudio,
    check_dependencies
)


class TestEnhancedImageProcessor(unittest.TestCase):
    """Test cases for EnhancedImageProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.processor = EnhancedImageProcessor()
        # Create test images
        self.img_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.img_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.img_small = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

    def test_normalize_images_same_size(self):
        """Test normalizing images of same size"""
        img1 = self.img_color.copy()
        img2 = self.img_color.copy()
        norm1, norm2 = self.processor.normalize_images(img1, img2)
        self.assertEqual(norm1.shape, norm2.shape)

    def test_normalize_images_different_sizes(self):
        """Test normalizing images of different sizes"""
        img1 = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        norm1, norm2 = self.processor.normalize_images(img1, img2)
        self.assertEqual(norm1.shape, norm2.shape)
        self.assertEqual(norm1.shape[:2], (100, 100))

    def test_normalize_images_gray_to_color(self):
        """Test normalizing grayscale and color images"""
        norm1, norm2 = self.processor.normalize_images(self.img_gray, self.img_color)
        self.assertEqual(len(norm1.shape), len(norm2.shape))
        self.assertEqual(norm1.shape[2], norm2.shape[2])

    def test_normalize_images_none_input(self):
        """Test error handling for None inputs"""
        with self.assertRaises(ValueError):
            self.processor.normalize_images(None, self.img_color)
        with self.assertRaises(ValueError):
            self.processor.normalize_images(self.img_color, None)

    def test_normalize_images_empty_input(self):
        """Test error handling for empty images"""
        empty = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.processor.normalize_images(empty, self.img_color)

    def test_calculate_perceptual_hash(self):
        """Test perceptual hash calculation"""
        hash1 = self.processor.calculate_perceptual_hash(self.img_color)
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in '01' for c in hash1))

        # Same image should produce same hash
        hash2 = self.processor.calculate_perceptual_hash(self.img_color)
        self.assertEqual(hash1, hash2)

    def test_calculate_perceptual_hash_empty(self):
        """Test perceptual hash for empty image"""
        empty = np.array([], dtype=np.uint8)
        hash_val = self.processor.calculate_perceptual_hash(empty)
        self.assertEqual(hash_val, "0" * 64)

    def test_calculate_similarity_score(self):
        """Test similarity score calculation"""
        # Identical images should have low similarity score (near 0)
        score = self.processor.calculate_similarity_score(self.img_color, self.img_color)
        self.assertLess(score, 0.1)
        
        # Different images should have higher score
        img_different = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = self.processor.calculate_similarity_score(self.img_color, img_different)
        self.assertGreater(score, 0.1)

    def test_calculate_ssim(self):
        """Test SSIM calculation"""
        ssim = self.processor._calculate_ssim(self.img_color, self.img_color)
        self.assertGreaterEqual(ssim, 0.9)  # Identical images should have high SSIM

    def test_calculate_ssim_small_images(self):
        """Test SSIM for small images"""
        ssim = self.processor._calculate_ssim(self.img_small, self.img_small)
        self.assertGreaterEqual(ssim, 0.0)
        self.assertLessEqual(ssim, 1.0)

    def test_histogram_correlation(self):
        """Test histogram correlation"""
        corr = self.processor._histogram_correlation(self.img_color, self.img_color)
        self.assertAlmostEqual(corr, 1.0, places=2)

    def test_edge_similarity(self):
        """Test edge similarity calculation"""
        sim = self.processor._edge_similarity(self.img_color, self.img_color)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_extract_features(self):
        """Test feature extraction"""
        features = self.processor.extract_features(self.img_color)
        
        # Check required features exist
        self.assertIn('mean', features)
        self.assertIn('std', features)
        self.assertIn('min', features)
        self.assertIn('max', features)
        self.assertIn('edge_density', features)
        self.assertIn('corner_density', features)
        
        # Check feature values are reasonable
        self.assertGreaterEqual(features['mean'], 0)
        self.assertLessEqual(features['mean'], 255)
        self.assertGreaterEqual(features['edge_density'], 0)
        self.assertLessEqual(features['edge_density'], 1)

    def test_create_anomaly_map(self):
        """Test anomaly map creation"""
        # Create slightly different image
        modified = self.img_color.copy()
        modified[40:60, 40:60] = 255  # Add white square
        
        anomaly_map, diff_map, heatmap = self.processor.create_anomaly_map(
            self.img_color, modified
        )
        
        # Check outputs have correct shape
        self.assertEqual(anomaly_map.shape, (*self.img_color.shape[:2], 3))
        self.assertEqual(diff_map.shape, self.img_color.shape[:2])
        self.assertEqual(heatmap.shape, (*self.img_color.shape[:2], 3))


class TestImprovedScriptManager(unittest.TestCase):
    """Test cases for ImprovedScriptManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "test_scripts"
        self.scripts_dir.mkdir()
        
        # Create test script
        self.test_script_path = self.scripts_dir / "test_blur.py"
        self.test_script_content = """
import numpy as np

def process_image(image):
    return image * 0.5  # Simple dimming
"""
        self.test_script_path.write_text(self.test_script_content)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_script_loading(self):
        """Test loading scripts from directory"""
        manager = ImprovedScriptManager([str(self.scripts_dir)])
        self.assertGreater(len(manager.functions), 0)
        self.assertIn("test_blur.py", manager.functions)

    def test_script_categorization(self):
        """Test script category determination"""
        # Create scripts with category keywords
        blur_script = self.scripts_dir / "gaussian_blur.py"
        blur_script.write_text(self.test_script_content)
        
        edge_script = self.scripts_dir / "canny_edges.py"
        edge_script.write_text(self.test_script_content)
        
        manager = ImprovedScriptManager([str(self.scripts_dir)])
        
        self.assertIn("gaussian_blur.py", manager.get_scripts_by_category('filtering'))
        self.assertIn("canny_edges.py", manager.get_scripts_by_category('edge_detection'))

    def test_get_random_scripts(self):
        """Test getting random scripts"""
        manager = ImprovedScriptManager([str(self.scripts_dir)])
        random_scripts = manager.get_random_scripts(5)
        self.assertLessEqual(len(random_scripts), 5)
        self.assertGreater(len(random_scripts), 0)

    def test_empty_directory(self):
        """Test handling empty directory"""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        manager = ImprovedScriptManager([str(empty_dir)])
        self.assertEqual(len(manager.functions), 0)

    def test_invalid_script(self):
        """Test handling invalid scripts"""
        # Create script with syntax error
        bad_script = self.scripts_dir / "bad_script.py"
        bad_script.write_text("def process_image(image):\n    return image +")
        
        # Should still load other scripts
        manager = ImprovedScriptManager([str(self.scripts_dir)])
        self.assertNotIn("bad_script.py", manager.functions)
        self.assertIn("test_blur.py", manager.functions)


class TestSmartLearner(unittest.TestCase):
    """Test cases for SmartLearner class"""

    def setUp(self):
        """Set up test fixtures"""
        self.learner = SmartLearner(10)
        
        # Create test states
        self.current_state = {
            'mean': 100,
            'std': 30,
            'edge_density': 0.2,
            'gabor': np.array([0.1, 0.2, 0.3])
        }
        
        self.target_state = {
            'mean': 150,
            'std': 40,
            'edge_density': 0.4,
            'gabor': np.array([0.2, 0.3, 0.4])
        }

    def test_analyze_needs(self):
        """Test needs analysis"""
        needs = self.learner._analyze_needs(self.current_state, self.target_state)
        
        self.assertIn('brightness', needs)
        self.assertIn('edges', needs)
        self.assertGreater(needs['brightness'], 0)

    def test_get_scripts_for_need(self):
        """Test getting scripts for specific needs"""
        scripts = self.learner._get_scripts_for_need('brightness')
        self.assertGreater(len(scripts), 0)
        self.assertIn('gamma_correction.py', scripts)

    def test_get_recommended_scripts(self):
        """Test script recommendation"""
        recommendations = self.learner.get_recommended_scripts(
            self.current_state, self.target_state
        )
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 10)

    def test_record_result(self):
        """Test recording processing results"""
        sequence = ['blur.py', 'sharpen.py']
        self.learner.record_result(sequence, True, 0.5)
        
        self.assertEqual(len(self.learner.successful_sequences), 1)
        self.assertEqual(self.learner.script_usage_count['blur.py'], 1)
        self.assertGreater(self.learner.script_success_rate['blur.py'], 0)

    def test_save_load_knowledge(self):
        """Test saving and loading knowledge"""
        # Add some data
        self.learner.record_result(['test.py'], True, 0.8)
        
        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            self.learner.save_knowledge(temp_file)
            
            # Create new learner and load
            new_learner = SmartLearner(10)
            new_learner.load_knowledge(temp_file)
            
            self.assertEqual(len(new_learner.successful_sequences), 1)
            self.assertEqual(new_learner.script_usage_count['test.py'], 1)
        finally:
            os.unlink(temp_file)


class TestEnhancedProcessingStudio(unittest.TestCase):
    """Test cases for EnhancedProcessingStudio class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.scripts_dir.mkdir()
        
        # Create minimal test script
        test_script = self.scripts_dir / "test_processor.py"
        test_script.write_text("""
import numpy as np

def process_image(image):
    return np.clip(image * 1.1, 0, 255).astype(np.uint8)
""")
        
        # Create test images
        self.input_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.target_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_studio_initialization(self):
        """Test studio initialization"""
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        self.assertIsNotNone(studio.processor)
        self.assertIsNotNone(studio.script_manager)
        self.assertIsNotNone(studio.learner)
        self.assertTrue(studio.cache_dir.exists())

    def test_process_to_match_target_basic(self):
        """Test basic target matching"""
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        # Process with very limited iterations
        results = studio.process_to_match_target(
            self.input_image,
            self.target_image,
            max_iterations=5,
            similarity_threshold=0.01,
            verbose=False
        )
        
        self.assertIn('success', results)
        self.assertIn('final_similarity', results)
        self.assertIn('iterations', results)
        self.assertIn('pipeline', results)
        self.assertIn('final_image', results)
        self.assertIsInstance(results['final_image'], np.ndarray)

    def test_process_with_none_images(self):
        """Test error handling for None images"""
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        with self.assertRaises(ValueError):
            studio.process_to_match_target(None, self.target_image)

    def test_generate_anomaly_visualization(self):
        """Test anomaly visualization generation"""
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        visualizations = studio.generate_anomaly_visualization(
            self.input_image,
            self.target_image
        )
        
        self.assertIn('anomaly_map', visualizations)
        self.assertIn('difference_map', visualizations)
        self.assertIn('heatmap', visualizations)
        self.assertIn('edge_changes', visualizations)

    def test_interactive_setup_mocked(self):
        """Test interactive setup with mocked input"""
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        # Create dummy image files
        input_path = Path(self.temp_dir) / "input.png"
        target_path = Path(self.temp_dir) / "target.png"
        cv2.imwrite(str(input_path), self.input_image)
        cv2.imwrite(str(target_path), self.target_image)
        
        # Mock user input
        with patch('builtins.input', side_effect=[
            str(input_path),
            str(target_path),
            '100',
            '0.1',
            'y'
        ]):
            config = studio.interactive_setup()
        
        self.assertEqual(config['input_path'], str(input_path))
        self.assertEqual(config['target_path'], str(target_path))
        self.assertEqual(config['max_iterations'], 100)
        self.assertEqual(config['similarity_threshold'], 0.1)
        self.assertTrue(config['verbose'])


class TestDependencyManagement(unittest.TestCase):
    """Test dependency checking and installation"""

    @patch('subprocess.check_call')
    @patch('builtins.__import__')
    def test_check_dependencies_all_present(self, mock_import, mock_subprocess):
        """Test when all dependencies are present"""
        mock_import.return_value = MagicMock()
        result = check_dependencies()
        self.assertTrue(result)
        mock_subprocess.assert_not_called()

    @patch('subprocess.check_call')
    @patch('builtins.__import__')
    def test_check_dependencies_missing(self, mock_import, mock_subprocess):
        """Test when dependencies are missing"""
        def import_side_effect(name):
            if name == 'cv2':
                raise ImportError()
            return MagicMock()
        
        mock_import.side_effect = import_side_effect
        result = check_dependencies()
        
        # Should try to install opencv-python
        mock_subprocess.assert_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = Path(self.temp_dir) / "scripts"
        self.scripts_dir.mkdir()
        
        # Create multiple test scripts
        scripts = {
            'blur.py': """
import numpy as np
import cv2

def process_image(image):
    return cv2.GaussianBlur(image, (5, 5), 1.0)
""",
            'brighten.py': """
import numpy as np

def process_image(image):
    return np.clip(image * 1.2, 0, 255).astype(np.uint8)
""",
            'darken.py': """
import numpy as np

def process_image(image):
    return np.clip(image * 0.8, 0, 255).astype(np.uint8)
"""
        }
        
        for name, content in scripts.items():
            (self.scripts_dir / name).write_text(content)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline"""
        # Create test images
        input_img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        target_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
        
        # Create studio
        studio = EnhancedProcessingStudio(
            scripts_dirs=[str(self.scripts_dir)],
            cache_dir=str(Path(self.temp_dir) / "cache")
        )
        
        # Process
        results = studio.process_to_match_target(
            input_img,
            target_img,
            max_iterations=20,
            similarity_threshold=0.1,
            verbose=False
        )
        
        # Verify results
        self.assertIsNotNone(results['final_image'])
        self.assertGreater(len(results['pipeline']), 0)
        
        # Final image should be brighter than input
        self.assertGreater(
            np.mean(results['final_image']),
            np.mean(input_img)
        )


def run_specific_test(test_class=None, test_method=None):
    """Run specific test class or method"""
    if test_class and test_method:
        suite = unittest.TestLoader().loadTestsFromName(
            f'{test_class}.{test_method}',
            module=sys.modules[__name__]
        )
    elif test_class:
        suite = unittest.TestLoader().loadTestsFromTestCase(
            getattr(sys.modules[__name__], test_class)
        )
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)