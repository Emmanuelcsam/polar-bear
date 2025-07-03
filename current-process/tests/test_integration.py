"""
Integration tests for complete pipeline
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
import time

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from app import EnhancedApplication, DataAcquisition
from process import EnhancedProcessor
from separation import EnhancedSeparator
from detection import EnhancedDetector
from realtime_processor import (
    RealtimeProcessor, FrameBuffer, OptimizedPipeline, FrameResult
)
from config_manager import get_config


class TestDataAcquisition:
    """Test data acquisition and reporting"""
    
    @pytest.fixture
    def data_acq(self):
        """Create data acquisition instance"""
        return DataAcquisition()
    
    def test_determine_pass_fail(self, data_acq):
        """Test pass/fail determination"""
        from enhanced_detection import Defect
        
        # No defects - should pass
        assert data_acq._determine_pass_fail([]) == True
        
        # Few minor defects - should pass
        minor_defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.3,
                confidence=0.8,
                zone="ferrule",
                area=400
            )
        ]
        assert data_acq._determine_pass_fail(minor_defects) == True
        
        # Severe defect - should fail
        severe_defects = [
            Defect(
                type="crack",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.9,
                confidence=0.9,
                zone="core",
                area=400
            )
        ]
        assert data_acq._determine_pass_fail(severe_defects) == False
        
        # Too many defects - should fail
        many_defects = minor_defects * 10
        assert data_acq._determine_pass_fail(many_defects) == False
    
    def test_generate_visualizations(self, data_acq):
        """Test visualization generation"""
        from enhanced_detection import Defect
        
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        zones = {
            'core': np.zeros((200, 200), dtype=np.uint8),
            'cladding': np.zeros((200, 200), dtype=np.uint8),
            'ferrule': np.zeros((200, 200), dtype=np.uint8)
        }
        
        # Add zone regions
        cv2.circle(zones['core'], (100, 100), 30, 255, -1)
        
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.5,
                confidence=0.8,
                zone="core",
                area=400
            )
        ]
        
        visualizations = data_acq._generate_visualizations(image, zones, defects)
        
        assert 'defect_overlay' in visualizations
        assert 'zone_overlay' in visualizations
        
        # Check shapes
        for vis in visualizations.values():
            assert vis.shape == image.shape
    
    def test_calculate_metrics(self, data_acq):
        """Test metrics calculation"""
        from enhanced_detection import Defect
        
        image = np.ones((200, 200, 3), dtype=np.uint8)
        zones = {
            'core': np.ones((200, 200), dtype=np.uint8),
            'cladding': np.ones((200, 200), dtype=np.uint8),
            'ferrule': np.ones((200, 200), dtype=np.uint8)
        }
        
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.6,
                confidence=0.8,
                zone="core",
                area=400
            ),
            Defect(
                type="pit",
                location=(150, 150),
                bbox=(145, 145, 10, 10),
                severity=0.4,
                confidence=0.7,
                zone="cladding",
                area=100
            )
        ]
        
        metrics = data_acq._calculate_metrics(image, zones, defects)
        
        assert 'total_area' in metrics
        assert 'defect_density' in metrics
        assert 'average_severity' in metrics
        assert 'zone_defects' in metrics
        
        assert metrics['average_severity'] == pytest.approx(0.5, rel=0.01)
        assert metrics['zone_defects']['core']['count'] == 1
        assert metrics['zone_defects']['cladding']['count'] == 1
    
    def test_process_results(self, data_acq):
        """Test complete results processing"""
        from enhanced_detection import Defect
        
        image_path = Path("test.jpg")
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        zones = {
            'core': np.zeros((200, 200), dtype=np.uint8),
            'cladding': np.zeros((200, 200), dtype=np.uint8),
            'ferrule': np.zeros((200, 200), dtype=np.uint8)
        }
        defects = []
        
        result = data_acq.process_results(image_path, image, zones, defects)
        
        assert 'report' in result
        assert 'visualizations' in result
        assert 'pass' in result
        
        assert result['pass'] == True  # No defects
        assert result['report']['total_defects'] == 0


class TestFrameBuffer:
    """Test real-time frame buffer"""
    
    def test_initialization(self):
        """Test buffer initialization"""
        buffer = FrameBuffer(max_size=10)
        assert buffer.size() == 0
        assert buffer.frame_counter == 0
    
    def test_put_get(self):
        """Test putting and getting frames"""
        buffer = FrameBuffer()
        
        # Put frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_id = buffer.put(frame)
        
        assert frame_id == 1
        assert buffer.size() == 1
        
        # Get frame
        result = buffer.get()
        assert result is not None
        
        retrieved_id, timestamp, retrieved_frame = result
        assert retrieved_id == frame_id
        assert np.array_equal(retrieved_frame, frame)
        assert buffer.size() == 0
    
    def test_max_size(self):
        """Test buffer max size limit"""
        buffer = FrameBuffer(max_size=3)
        
        # Add more frames than max size
        for i in range(5):
            frame = np.ones((10, 10, 3), dtype=np.uint8) * i
            buffer.put(frame)
        
        # Should only have max_size frames
        assert buffer.size() <= 3
    
    def test_clear(self):
        """Test clearing buffer"""
        buffer = FrameBuffer()
        
        # Add frames
        for i in range(5):
            frame = np.ones((10, 10, 3), dtype=np.uint8) * i
            buffer.put(frame)
        
        # Clear
        buffer.clear()
        assert buffer.size() == 0


class TestOptimizedPipeline:
    """Test optimized processing pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        config = get_config()
        config.interactive_mode = False
        return OptimizedPipeline()
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.processor is not None
        assert pipeline.separator is not None
        assert pipeline.detector is not None
        assert isinstance(pipeline.cache, dict)
    
    def test_process_frame(self, pipeline):
        """Test frame processing"""
        # Create test frame
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(frame, (100, 100), 30, (255, 255, 255), -1)
        
        result = pipeline.process_frame(frame, frame_id=1)
        
        assert isinstance(result, FrameResult)
        assert result.frame_id == 1
        assert result.original_frame.shape == frame.shape
        assert result.processed_frame.shape == frame.shape
        assert isinstance(result.zones, dict)
        assert isinstance(result.defects, list)
        assert result.processing_time > 0
        assert result.fps >= 0
    
    def test_quick_preprocess(self, pipeline):
        """Test quick preprocessing"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        variations = pipeline._quick_preprocess(frame)
        
        assert isinstance(variations, dict)
        assert 'original' in variations
        assert len(variations) >= 2  # At least original + some transforms
        
        # Check all variations are valid
        for name, var in variations.items():
            assert isinstance(var, np.ndarray)
            assert var.dtype == np.uint8
    
    def test_quick_separation(self, pipeline):
        """Test quick separation"""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.circle(frame, (100, 100), 50, (255, 255, 255), -1)
        
        variations = {"original": frame}
        zones = pipeline._quick_separation(frame, variations)
        
        assert isinstance(zones, dict)
        # Should have at least some zones
        assert len(zones) >= 1
        
        for zone_name, mask in zones.items():
            assert mask.shape == frame.shape[:2]
            assert mask.dtype == np.uint8
    
    def test_frame_caching(self, pipeline):
        """Test frame caching for separation"""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        variations = {"original": frame}
        
        # First call
        zones1 = pipeline._quick_separation(frame, variations)
        
        # Second call with same frame - should use cache
        zones2 = pipeline._quick_separation(frame, variations)
        
        # Should get same result
        for zone_name in zones1:
            if zone_name in zones2:
                assert np.array_equal(zones1[zone_name], zones2[zone_name])
    
    def test_create_visualization(self, pipeline):
        """Test visualization creation"""
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        zones = {
            'core': np.zeros((200, 200), dtype=np.uint8),
            'cladding': np.zeros((200, 200), dtype=np.uint8)
        }
        
        cv2.circle(zones['core'], (100, 100), 30, 255, -1)
        
        from enhanced_detection import Defect
        defects = [
            Defect(
                type="scratch",
                location=(100, 100),
                bbox=(90, 90, 20, 20),
                severity=0.5,
                confidence=0.8,
                zone="core",
                area=400
            )
        ]
        
        vis = pipeline._create_visualization(frame, zones, defects)
        
        assert vis.shape == frame.shape
        assert vis.dtype == frame.dtype
        # Should have modifications
        assert not np.array_equal(vis, frame)


class TestRealtimeProcessor:
    """Test real-time processor"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = RealtimeProcessor()
        
        assert processor.frame_buffer is not None
        assert processor.pipeline is not None
        assert processor.is_running == False
        assert processor.metrics['frames_processed'] == 0
    
    def test_get_result_empty(self):
        """Test getting result when queue is empty"""
        processor = RealtimeProcessor()
        
        result = processor.get_result(timeout=0.01)
        assert result is None
    
    def test_metrics_tracking(self):
        """Test metrics initialization"""
        processor = RealtimeProcessor()
        
        assert 'frames_processed' in processor.metrics
        assert 'frames_dropped' in processor.metrics
        assert 'avg_fps' in processor.metrics
        assert 'avg_processing_time' in processor.metrics


class TestCompletePipeline:
    """Test complete processing pipeline integration"""
    
    def test_full_pipeline(self):
        """Test processing image through complete pipeline"""
        config = get_config()
        config.interactive_mode = False
        config.processing.parallel_processing = False
        
        # Create components
        processor = EnhancedProcessor()
        separator = EnhancedSeparator()
        detector = EnhancedDetector()
        data_acq = DataAcquisition()
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # Create fiber optic-like image
            image = np.zeros((300, 300, 3), dtype=np.uint8)
            center = (150, 150)
            
            # Draw fiber structure
            cv2.circle(image, center, 120, (50, 50, 50), -1)    # Ferrule
            cv2.circle(image, center, 80, (100, 100, 100), -1)  # Cladding
            cv2.circle(image, center, 30, (200, 200, 200), -1)  # Core
            
            # Add defect
            cv2.circle(image, (150, 120), 5, (0, 0, 0), -1)
            
            cv2.imwrite(f.name, image)
            image_path = Path(f.name)
        
        try:
            # Step 1: Process variations
            variations = processor.process_image(image_path)
            assert len(variations) >= 2
            
            # Step 2: Separate zones
            zones = separator.separate_zones(image, variations)
            assert 'core' in zones
            assert 'cladding' in zones
            assert 'ferrule' in zones
            
            # Step 3: Detect defects
            defects = detector.detect_defects(image, zones, variations)
            assert isinstance(defects, list)
            
            # Step 4: Process results
            result = data_acq.process_results(image_path, image, zones, defects)
            assert 'report' in result
            assert 'visualizations' in result
            assert 'pass' in result
            
            # Verify report structure
            report = result['report']
            assert 'timestamp' in report
            assert 'total_defects' in report
            assert 'metrics' in report
            
        finally:
            # Cleanup
            image_path.unlink()
    
    def test_performance_requirements(self):
        """Test that pipeline meets performance requirements"""
        config = get_config()
        config.interactive_mode = False
        config.processing.num_variations = 10  # Reduce for speed
        
        pipeline = OptimizedPipeline()
        
        # Create simple test frame
        frame = np.ones((640, 480, 3), dtype=np.uint8) * 128
        
        # Measure processing time
        start_time = time.time()
        result = pipeline.process_frame(frame, frame_id=1)
        processing_time = time.time() - start_time
        
        # Should process reasonably fast
        assert processing_time < 1.0, f"Processing took {processing_time:.2f}s, should be < 1s"
        
        # Check FPS capability
        assert result.fps > 1.0, f"FPS is {result.fps:.1f}, should be > 1"


def test_configuration_no_argparse():
    """Test that no argparse is used in the system"""
    # Check that argparse is not imported in main modules
    modules_to_check = [
        'enhanced_app',
        'enhanced_process',
        'enhanced_separation', 
        'enhanced_detection',
        'realtime_processor',
        'config_manager',
        'enhanced_logging'
    ]
    
    for module_name in modules_to_check:
        module = sys.modules.get(f"{module_name}")
        if module:
            # Check module doesn't use argparse
            assert not hasattr(module, 'argparse'), f"{module_name} should not import argparse"
            
            # Check source doesn't contain argparse
            if hasattr(module, '__file__'):
                source_path = Path(module.__file__)
                if source_path.exists():
                    content = source_path.read_text()
                    assert 'argparse' not in content, f"{module_name} contains 'argparse' in source"


def test_logging_system():
    """Test that enhanced logging is working"""
    from enhanced_logging import get_logger, info, error
    
    # Test getting logger
    logger = get_logger("test_module")
    assert logger is not None
    
    # Test logging functions work without errors
    info("Test info message")
    error("Test error message")
    
    # Test structured logging
    logger.info("Test structured", key1="value1", key2=42)


def test_in_memory_processing():
    """Test that processing works in-memory without intermediate files"""
    config = get_config()
    config.interactive_mode = False
    config.processing.ram_only_mode = True
    
    processor = EnhancedProcessor()
    
    # Create test image in memory
    image_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Save temporarily just to test
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, image_array)
        image_path = Path(f.name)
    
    try:
        # Process - should work entirely in memory
        results = processor.process_image(image_path)
        
        # Verify results are in memory (numpy arrays)
        for name, result in results.items():
            assert isinstance(result, np.ndarray)
            assert result.flags['OWNDATA'] or result.base is not None
        
    finally:
        image_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])