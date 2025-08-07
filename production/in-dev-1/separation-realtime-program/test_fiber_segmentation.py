"""
Unit tests for the fiber segmentation real-time system.
Tests all major components and functionality.
"""

import unittest
import numpy as np
import cv2
import tempfile
import json
from pathlib import Path
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fiber_segmentation_realtime import (
    NumpyEncoder, SegmentationResult, EnhancedConsensusSystem,
    UnifiedSegmentationSystem, RealtimeSegmentationProcessor
)


class TestNumpyEncoder(unittest.TestCase):
    """Test the custom JSON encoder for numpy types."""
    
    def test_numpy_integer(self):
        """Test encoding numpy integers."""
        encoder = NumpyEncoder()
        result = encoder.default(np.int32(42))
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)
    
    def test_numpy_float(self):
        """Test encoding numpy floats."""
        encoder = NumpyEncoder()
        result = encoder.default(np.float64(3.14))
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)
    
    def test_numpy_array(self):
        """Test encoding numpy arrays."""
        encoder = NumpyEncoder()
        arr = np.array([[1, 2], [3, 4]])
        result = encoder.default(arr)
        self.assertEqual(result, [[1, 2], [3, 4]])
        self.assertIsInstance(result, list)
    
    def test_other_types(self):
        """Test that other types fall back to default behavior."""
        encoder = NumpyEncoder()
        with self.assertRaises(TypeError):
            encoder.default("string")


class TestSegmentationResult(unittest.TestCase):
    """Test the SegmentationResult class."""
    
    def test_initialization(self):
        """Test result initialization."""
        result = SegmentationResult("test_method", "test_image.png")
        self.assertEqual(result.method_name, "test_method")
        self.assertEqual(result.image_path, "test_image.png")
        self.assertIsNone(result.center)
        self.assertIsNone(result.core_radius)
        self.assertIsNone(result.cladding_radius)
        self.assertIsNone(result.masks)
        self.assertEqual(result.confidence, 0.5)
        self.assertEqual(result.execution_time, 0.0)
        self.assertIsNone(result.error)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SegmentationResult("test_method", "test_image.png")
        result.center = (100, 200)
        result.core_radius = 50
        result.cladding_radius = 100
        result.confidence = 0.8
        result.execution_time = 1.5
        result.error = "Test error"
        
        dict_result = result.to_dict()
        expected = {
            'method_name': 'test_method',
            'center': (100, 200),
            'core_radius': 50,
            'cladding_radius': 100,
            'confidence': 0.8,
            'execution_time': 1.5,
            'error': 'Test error',
            'has_masks': False
        }
        self.assertEqual(dict_result, expected)


class TestEnhancedConsensusSystem(unittest.TestCase):
    """Test the consensus system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.consensus_system = EnhancedConsensusSystem()
    
    def test_calculate_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[2:8, 2:8] = 1
        mask2 = mask1.copy()
        
        iou = self.consensus_system._calculate_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 1.0, places=6)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[0:5, 0:5] = 1
        mask2 = np.zeros((10, 10), dtype=np.uint8)
        mask2[5:10, 5:10] = 1
        
        iou = self.consensus_system._calculate_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[0:6, 0:6] = 1
        mask2 = np.zeros((10, 10), dtype=np.uint8)
        mask2[3:9, 3:9] = 1
        
        iou = self.consensus_system._calculate_iou(mask1, mask2)
        # Calculate actual expected value: intersection=9, union=36+9=45
        # IoU = 9/(45+1e-6) ≈ 0.143
        self.assertAlmostEqual(iou, 0.143, places=3)
    
    def test_calculate_iou_none_masks(self):
        """Test IoU calculation with None masks."""
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        iou = self.consensus_system._calculate_iou(mask1, None)
        self.assertEqual(iou, 0.0)
    
    def test_create_masks_from_params(self):
        """Test mask creation from geometric parameters."""
        center = (50, 50)
        core_radius = 20
        cladding_radius = 40
        image_shape = (100, 100)
        
        masks = self.consensus_system.create_masks_from_params(
            center, core_radius, cladding_radius, image_shape
        )
        
        self.assertIn('core', masks)
        self.assertIn('cladding', masks)
        self.assertIn('ferrule', masks)
        
        # Check that masks are binary
        self.assertTrue(np.all(np.unique(masks['core']) == [0, 1]))
        self.assertTrue(np.all(np.unique(masks['cladding']) == [0, 1]))
        self.assertTrue(np.all(np.unique(masks['ferrule']) == [0, 1]))
        
        # Check that masks are mutually exclusive
        intersection = np.logical_and(masks['core'], masks['cladding'])
        self.assertEqual(np.sum(intersection), 0)
        
        intersection = np.logical_and(masks['core'], masks['ferrule'])
        self.assertEqual(np.sum(intersection), 0)
        
        intersection = np.logical_and(masks['cladding'], masks['ferrule'])
        self.assertEqual(np.sum(intersection), 0)


class TestUnifiedSegmentationSystem(unittest.TestCase):
    """Test the unified segmentation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary methods directory
        self.temp_dir = tempfile.mkdtemp()
        self.methods_dir = Path(self.temp_dir)
        
        # Create a mock method file
        mock_method_content = '''
def adaptive_segment_image(image_path, output_dir):
    return {
        'success': True,
        'center': [100, 100],
        'core_radius': 30,
        'cladding_radius': 60,
        'confidence': 0.8
    }
'''
        with open(self.methods_dir / 'adaptive_intensity.py', 'w') as f:
            f.write(mock_method_content)
        
        self.system = UnifiedSegmentationSystem(str(self.methods_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.system.methods_dir)
        self.assertIsNotNone(self.system.output_dir)
        self.assertIsNotNone(self.system.consensus_system)
        self.assertIsInstance(self.system.dataset_stats, dict)
    
    def test_detect_and_inpaint_anomalies(self):
        """Test anomaly detection and inpainting."""
        # Create a test image with a dark spot
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image[40:60, 40:60] = 50  # Create a dark region
        
        inpainted, defect_mask = self.system.detect_and_inpaint_anomalies(image)
        
        self.assertEqual(inpainted.shape, image.shape)
        self.assertEqual(defect_mask.shape, (100, 100))
        self.assertIsInstance(defect_mask, np.ndarray)
    
    def test_load_knowledge(self):
        """Test knowledge loading and saving."""
        # Test saving knowledge
        self.system.dataset_stats['method_scores']['test_method'] = 0.8
        self.system.save_knowledge()
        
        # Test loading knowledge
        new_system = UnifiedSegmentationSystem(str(self.methods_dir))
        self.assertIn('test_method', new_system.dataset_stats['method_scores'])
        self.assertEqual(new_system.dataset_stats['method_scores']['test_method'], 0.8)


class TestRealtimeSegmentationProcessor(unittest.TestCase):
    """Test the real-time segmentation processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.methods_dir = Path(self.temp_dir)
        
        # Create a mock method file
        mock_method_content = '''
def adaptive_segment_image(image_path, output_dir):
    return {
        'success': True,
        'center': [100, 100],
        'core_radius': 30,
        'cladding_radius': 60,
        'confidence': 0.8
    }
'''
        with open(self.methods_dir / 'adaptive_intensity.py', 'w') as f:
            f.write(mock_method_content)
        
        self.processor = RealtimeSegmentationProcessor(str(self.methods_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor.segmentation_system)
        self.assertIsNotNone(self.processor.frame_queue)
        self.assertIsNotNone(self.processor.result_queue)
        self.assertIsInstance(self.processor.frame_counter, int)
        self.assertIsInstance(self.processor.processed_counter, int)
    
    def test_create_display_frame(self):
        """Test display frame creation."""
        # Create a test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Test without result
        display_frame = self.processor._create_display_frame(frame, None)
        self.assertEqual(display_frame.shape, frame.shape)
        
        # Test with result
        result = {
            'consensus': {
                'center': (320, 240),
                'core_radius': 50,
                'cladding_radius': 100,
                'contributing_methods': ['test_method']
            }
        }
        display_frame = self.processor._create_display_frame(frame, result)
        self.assertEqual(display_frame.shape, frame.shape)
    
    def test_adjust_processing_interval(self):
        """Test adaptive processing interval adjustment."""
        initial_interval = self.processor.process_every_n_frames
        
        # Test with fast processing
        self.processor._adjust_processing_interval(0.1)
        self.assertLessEqual(self.processor.process_every_n_frames, initial_interval)
        
        # Test with slow processing
        self.processor._adjust_processing_interval(5.0)
        self.assertGreaterEqual(self.processor.process_every_n_frames, initial_interval)
    
    def test_get_performance_stats(self):
        """Test performance statistics calculation."""
        # Add some performance history
        self.processor.performance_history = [
            {'processing_time': 1.0, 'frame_number': 1},
            {'processing_time': 2.0, 'frame_number': 2},
            {'processing_time': 1.5, 'frame_number': 3}
        ]
        self.processor.frame_counter = 100
        self.processor.processed_counter = 10
        
        stats = self.processor.get_performance_stats()
        
        self.assertIn('total_frames', stats)
        self.assertIn('processed_frames', stats)
        self.assertIn('processing_rate', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertEqual(stats['total_frames'], 100)
        self.assertEqual(stats['processed_frames'], 10)
        self.assertEqual(stats['processing_rate'], 0.1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.methods_dir = Path(self.temp_dir)
        
        # Create multiple mock method files
        methods = {
            'geometric_approach.py': '''
def segment_with_geometric(image_path, output_dir):
    return {
        'success': True,
        'center': [100, 100],
        'core_radius': 25,
        'cladding_radius': 55,
        'confidence': 0.9
    }
''',
            'threshold_separation.py': '''
def segment_with_threshold(image_path, output_dir):
    return {
        'success': True,
        'center': [105, 95],
        'core_radius': 30,
        'cladding_radius': 60,
        'confidence': 0.8
    }
''',
            'hough_separation.py': '''
def segment_with_hough(image_path, output_dir):
    return {
        'success': True,
        'center': [95, 105],
        'core_radius': 28,
        'cladding_radius': 58,
        'confidence': 0.7
    }
'''
        }
        
        for filename, content in methods.items():
            with open(self.methods_dir / filename, 'w') as f:
                f.write(content)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_processing_pipeline(self):
        """Test the complete processing pipeline."""
        system = UnifiedSegmentationSystem(str(self.methods_dir))
        
        # Create a test image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Add a circular region to simulate fiber
        cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
        
        # Save test image
        test_image_path = self.methods_dir / "test_image.png"
        cv2.imwrite(str(test_image_path), image)
        
        # Process the image
        consensus = system._process_image_lightweight(
            test_image_path, 
            self.methods_dir, 
            image.shape[:2]
        )
        
        # Verify results
        self.assertIsNotNone(consensus)
        self.assertIn('center', consensus)
        self.assertIn('core_radius', consensus)
        self.assertIn('cladding_radius', consensus)
        self.assertIn('masks', consensus)
        self.assertIn('contributing_methods', consensus)
        
        # Clean up
        test_image_path.unlink()
    
    def test_consensus_generation(self):
        """Test consensus generation with multiple methods."""
        consensus_system = EnhancedConsensusSystem()
        
        # Create mock results
        results = []
        for i, method_name in enumerate(['method1', 'method2', 'method3']):
            result = SegmentationResult(method_name, "test.png")
            result.center = (100 + i*5, 100 + i*5)
            result.core_radius = 25 + i*2
            result.cladding_radius = 55 + i*2
            result.confidence = 0.8 - i*0.1
            result.masks = consensus_system.create_masks_from_params(
                result.center, result.core_radius, result.cladding_radius, (200, 200)
            )
            results.append(result)
        
        # Generate consensus
        method_scores = {'method1': 1.0, 'method2': 1.0, 'method3': 1.0}
        consensus = consensus_system.generate_consensus(results, method_scores, (200, 200))
        
        # Verify consensus
        self.assertIsNotNone(consensus)
        self.assertIn('center', consensus)
        self.assertIn('core_radius', consensus)
        self.assertIn('cladding_radius', consensus)
        self.assertIn('masks', consensus)
        self.assertIn('contributing_methods', consensus)


def run_performance_test():
    """Run a performance test to measure processing speed."""
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    
    # Create test system
    temp_dir = tempfile.mkdtemp()
    methods_dir = Path(temp_dir)
    
    # Create mock method
    mock_method_content = '''
def adaptive_segment_image(image_path, output_dir):
    import time
    time.sleep(0.1)  # Simulate processing time
    return {
        'success': True,
        'center': [100, 100],
        'core_radius': 30,
        'cladding_radius': 60,
        'confidence': 0.8
    }
'''
    with open(methods_dir / 'adaptive_intensity.py', 'w') as f:
        f.write(mock_method_content)
    
    system = UnifiedSegmentationSystem(str(methods_dir))
    
    # Create test image
    image = np.ones((200, 200, 3), dtype=np.uint8) * 128
    cv2.circle(image, (100, 100), 50, (255, 255, 255), -1)
    
    # Test processing speed
    import time
    start_time = time.time()
    
    test_image_path = methods_dir / "test_image.png"
    cv2.imwrite(str(test_image_path), image)
    
    consensus = system._process_image_lightweight(
        test_image_path, 
        methods_dir, 
        image.shape[:2]
    )
    
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Consensus achieved: {consensus is not None}")
    
    if consensus:
        print(f"Contributing methods: {consensus['contributing_methods']}")
        print(f"Center: {consensus['center']}")
        print(f"Core radius: {consensus['core_radius']}")
        print(f"Cladding radius: {consensus['cladding_radius']}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return processing_time < 5.0  # Should complete within 5 seconds


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    print("\nRunning performance test...")
    performance_ok = run_performance_test()
    
    if performance_ok:
        print("✅ Performance test passed")
    else:
        print("❌ Performance test failed")
    
    print("\nAll tests completed!") 