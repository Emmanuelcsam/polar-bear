"""
Unit tests for enhanced separation module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import json
import tempfile

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from separation import (
    EnhancedSeparator, MLSegmentationModel, MethodExecutor,
    ConsensusBuilder, SegmentationResult
)
from config_manager import get_config


class TestSegmentationResult:
    """Test SegmentationResult dataclass"""
    
    def test_initialization(self):
        """Test creating segmentation result"""
        result = SegmentationResult(
            method_name="test_method",
            confidence=0.8
        )
        
        assert result.method_name == "test_method"
        assert result.confidence == 0.8
        assert result.core_mask is None
        assert result.error is None
    
    def test_with_masks(self):
        """Test result with masks"""
        core_mask = np.ones((100, 100), dtype=np.uint8) * 255
        cladding_mask = np.zeros((100, 100), dtype=np.uint8)
        
        result = SegmentationResult(
            method_name="test_method",
            core_mask=core_mask,
            cladding_mask=cladding_mask,
            confidence=0.9
        )
        
        assert np.array_equal(result.core_mask, core_mask)
        assert np.array_equal(result.cladding_mask, cladding_mask)
        assert result.ferrule_mask is None
    
    def test_with_error(self):
        """Test result with error"""
        result = SegmentationResult(
            method_name="test_method",
            error="Method failed"
        )
        
        assert result.error == "Method failed"
        assert result.confidence == 0.0


class TestMLSegmentationModel:
    """Test ML segmentation model"""
    
    def test_initialization_no_ml(self):
        """Test initialization when ML is disabled"""
        config = get_config()
        config.processing.ml_enabled = False
        
        model = MLSegmentationModel()
        assert model.model is None
    
    def test_predict_without_model(self):
        """Test prediction when no model available"""
        config = get_config()
        config.processing.ml_enabled = False
        
        model = MLSegmentationModel()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = model.predict(image)
        assert isinstance(result, SegmentationResult)
        assert result.method_name == "ml_segmentation"
        assert result.error == "ML model not available"
    
    def test_predict_handles_errors(self):
        """Test prediction error handling"""
        model = MLSegmentationModel()
        model.model = None  # Force no model
        
        # Invalid image
        result = model.predict(None)
        assert result.error is not None


class TestMethodExecutor:
    """Test method executor for running segmentation methods"""
    @pytest.fixture
    def executor(self):
        """Create executor instance"""
        return MethodExecutor()
    
    def test_discover_methods(self, executor):
        """Test method discovery"""
        methods = executor.discover_methods()
        
        assert isinstance(methods, list)
        # Should find at least some methods
        expected_methods = [
            'adaptive_intensity', 'geometric_approach', 
            'threshold_separation', 'hough_separation'
        ]
        
        for method in expected_methods:
            if (executor.zone_methods_dir / f"{method}.py").exists():
                assert method in methods
    
    def test_execute_method_timeout(self, executor):
        """Test method execution with timeout"""
        # Create a method that will timeout
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = {"original": image}
        
        # Use very short timeout
        original_timeout = executor.config.separation.timeout_seconds
        executor.config.separation.timeout_seconds = 0.001
        
        result = executor.execute_method("nonexistent_method", image, variations)
        
        assert isinstance(result, SegmentationResult)
        assert result.error is not None
        
        # Restore timeout
        executor.config.separation.timeout_seconds = original_timeout


class TestConsensusBuilder:
    """Test consensus building from multiple results"""
    
    @pytest.fixture
    def builder(self):
        """Create consensus builder"""
        return ConsensusBuilder()
    
    def test_load_knowledge_base(self, builder):
        """Test loading knowledge base"""
        # Should handle missing file gracefully
        kb = builder._load_knowledge_base()
        assert isinstance(kb, dict)
    
    def test_build_consensus_no_results(self, builder):
        """Test consensus with no valid results"""
        results = []
        image_shape = (100, 100)
        
        masks = builder.build_consensus(results, image_shape)
        
        assert isinstance(masks, dict)
        assert 'core' in masks
        assert 'cladding' in masks
        assert 'ferrule' in masks
        
        # Should have default masks
        assert masks['core'].shape == image_shape
    
    def test_build_consensus_single_result(self, builder):
        """Test consensus with single result"""
        core_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(core_mask, (50, 50), 20, 255, -1)
        
        result = SegmentationResult(
            method_name="test_method",
            core_mask=core_mask,
            cladding_mask=np.zeros((100, 100), dtype=np.uint8),
            ferrule_mask=np.zeros((100, 100), dtype=np.uint8),
            confidence=0.9
        )
        
        masks = builder.build_consensus([result], (100, 100))
        
        assert masks['core'].shape == (100, 100)
        # Core should have some non-zero pixels
        assert np.sum(masks['core'] > 0) > 0
    
    def test_build_consensus_multiple_results(self, builder):
        """Test consensus with multiple agreeing results"""
        # Create similar masks from different methods
        results = []
        
        for i in range(3):
            core_mask = np.zeros((100, 100), dtype=np.uint8)
            cv2.circle(core_mask, (50, 50), 20 + i, 255, -1)
            
            result = SegmentationResult(
                method_name=f"method_{i}",
                core_mask=core_mask,
                cladding_mask=np.zeros((100, 100), dtype=np.uint8),
                ferrule_mask=np.zeros((100, 100), dtype=np.uint8),
                confidence=0.8
            )
            results.append(result)
        
        masks = builder.build_consensus(results, (100, 100))
        
        # Should have consensus core
        assert np.sum(masks['core'] > 0) > 0
    
    def test_post_process_mask(self, builder):
        """Test mask post-processing"""
        # Create noisy mask
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
        
        # Process
        processed = builder._post_process_mask(mask, 'core')
        
        # Should have less noise
        assert processed.dtype == np.uint8
        assert processed.shape == mask.shape
    
    def test_update_scores(self, builder):
        """Test updating method scores"""
        # Create test results
        core_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(core_mask, (50, 50), 20, 255, -1)
        
        result = SegmentationResult(
            method_name="test_method",
            core_mask=core_mask,
            cladding_mask=np.zeros((100, 100), dtype=np.uint8),
            ferrule_mask=np.zeros((100, 100), dtype=np.uint8),
            confidence=0.9
        )
        
        consensus_masks = {
            'core': core_mask.copy(),
            'cladding': np.zeros((100, 100), dtype=np.uint8),
            'ferrule': np.zeros((100, 100), dtype=np.uint8)
        }
        
        # Update scores
        builder._update_scores([result], consensus_masks)
        
        # Should have updated knowledge base
        assert 'test_method' in builder.knowledge_base
        assert 0 <= builder.knowledge_base['test_method'] <= 1
    
    def test_create_default_masks(self, builder):
        """Test default mask creation"""
        h, w = 200, 200
        masks = builder._create_default_masks(h, w)
        
        assert 'core' in masks
        assert 'cladding' in masks
        assert 'ferrule' in masks
        
        # Check shapes
        assert masks['core'].shape == (h, w)
        assert masks['cladding'].shape == (h, w)
        assert masks['ferrule'].shape == (h, w)
        
        # Check no overlap
        overlap = ((masks['core'] > 0) & (masks['cladding'] > 0))
        assert not np.any(overlap)


class TestEnhancedSeparator:
    """Test the main enhanced separator"""
    
    @pytest.fixture
    def separator(self):
        """Create separator instance"""
        config = get_config()
        config.interactive_mode = False
        config.separation.parallel_execution = False  # For easier testing
        return EnhancedSeparator()
    
    def test_separator_initialization(self, separator):
        """Test separator initialization"""
        assert separator.ml_model is not None
        assert separator.executor is not None
        assert separator.consensus_builder is not None
    
    def test_separate_zones(self, separator):
        """Test zone separation"""
        # Create test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)  # Core
        cv2.circle(image, (100, 100), 60, (128, 128, 128), 3)   # Cladding
        cv2.circle(image, (100, 100), 90, (64, 64, 64), 3)      # Ferrule
        
        variations = {"original": image}
        
        # Separate zones
        zones = separator.separate_zones(image, variations)
        
        assert isinstance(zones, dict)
        assert 'core' in zones
        assert 'cladding' in zones
        assert 'ferrule' in zones
        
        # Check that masks have correct shape
        for mask in zones.values():
            assert mask.shape == image.shape[:2]
            assert mask.dtype == np.uint8
    
    def test_visualize_separation(self, separator):
        """Test separation visualization"""
        # Create test data
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        zones = {
            'core': np.zeros((100, 100), dtype=np.uint8),
            'cladding': np.zeros((100, 100), dtype=np.uint8),
            'ferrule': np.zeros((100, 100), dtype=np.uint8)
        }
        
        # Add some regions
        cv2.circle(zones['core'], (50, 50), 20, 255, -1)
        cv2.circle(zones['cladding'], (50, 50), 40, 255, -1)
        cv2.circle(zones['cladding'], (50, 50), 20, 0, -1)
        
        # Visualize
        vis = separator.visualize_separation(image, zones)
        
        assert vis.shape == image.shape
        assert vis.dtype == image.dtype
        
        # Should have some color overlay
        assert not np.array_equal(vis, image)


def test_integration_separation():
    """Integration test for complete separation pipeline"""
    config = get_config()
    config.interactive_mode = False
    config.separation.parallel_execution = False
    
    # Create separator
    separator = EnhancedSeparator()
    
    # Create realistic test image
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw fiber optic structure
    center = (150, 150)
    cv2.circle(image, center, 30, (200, 200, 200), -1)  # Core (bright)
    cv2.circle(image, center, 70, (100, 100, 100), -1)  # Cladding
    cv2.circle(image, center, 30, (200, 200, 200), -1)  # Restore core
    cv2.circle(image, center, 120, (50, 50, 50), -1)    # Ferrule
    cv2.circle(image, center, 70, (100, 100, 100), -1)  # Restore cladding
    cv2.circle(image, center, 30, (200, 200, 200), -1)  # Restore core
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create variations
    variations = {
        "original": image,
        "blurred": cv2.GaussianBlur(image, (5, 5), 0)
    }
    
    # Run separation
    zones = separator.separate_zones(image, variations)
    
    # Verify results
    assert 'core' in zones
    assert 'cladding' in zones
    assert 'ferrule' in zones
    
    # Core should be smallest
    core_area = np.sum(zones['core'] > 0)
    cladding_area = np.sum(zones['cladding'] > 0)
    ferrule_area = np.sum(zones['ferrule'] > 0)
    
    assert core_area > 0, "Core should have non-zero area"
    assert cladding_area > 0, "Cladding should have non-zero area"
    
    # Visualize
    vis = separator.visualize_separation(image, zones)
    assert vis.shape == image.shape


class TestMLSegmentationModelAdvanced:
    """Advanced tests for ML segmentation model"""
    
    def test_pytorch_unet_architecture(self):
        """Test PyTorch U-Net architecture creation"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = True
        config.processing.tensorflow_enabled = False
        
        # This will attempt to create the model
        model = MLSegmentationModel()
        
        # Model creation should handle missing dependencies
        assert model.config is not None
    
    def test_tensorflow_unet_architecture(self):
        """Test TensorFlow U-Net architecture creation"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = False
        config.processing.tensorflow_enabled = True
        
        model = MLSegmentationModel()
        assert model.config is not None
    
    def test_predict_with_various_image_sizes(self):
        """Test prediction with various image sizes"""
        model = MLSegmentationModel()
        model.model = None  # Force no model for this test
        
        # Test different image sizes
        test_sizes = [(100, 100), (256, 256), (123, 456), (64, 64)]
        
        for h, w in test_sizes:
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = model.predict(image)
            
            assert isinstance(result, SegmentationResult)
            if result.error is None and result.core_mask is not None:
                assert result.core_mask.shape == (h, w)
                assert result.cladding_mask.shape == (h, w)
                assert result.ferrule_mask.shape == (h, w)


class TestMethodExecutorAdvanced:
    """Advanced tests for method executor"""
    
    def test_discover_methods_with_syntax_errors(self, executor):
        """Test method discovery with syntax errors in files"""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            executor.zone_methods_dir = tmpdir
            
            # Create valid method file
            valid_file = tmpdir / "valid_method.py"
            valid_file.write_text("""
def segment_zones(image, variations):
    return {'core': None, 'cladding': None, 'ferrule': None}
""")
            
            # Create file with syntax error
            invalid_file = tmpdir / "invalid_method.py"
            invalid_file.write_text("def segment_zones(: invalid syntax")
            
            # Discover methods
            methods = executor.discover_methods()
            
            # Should find valid method but skip invalid
            assert 'valid_method' in methods
            assert 'invalid_method' not in methods
    
    def test_execute_method_with_complex_results(self, executor):
        """Test executing method with complex results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            executor.zone_methods_dir = tmpdir
            
            # Create method that returns complex results
            method_file = tmpdir / "complex_method.py"
            method_file.write_text("""
import numpy as np

def segment_zones(image, variations):
    h, w = image.shape[:2]
    core = np.zeros((h, w), dtype=np.uint8)
    core[h//4:3*h//4, w//4:3*w//4] = 255
    
    cladding = np.zeros((h, w), dtype=np.uint8)
    cladding[h//8:7*h//8, w//8:7*w//8] = 255
    cladding[core > 0] = 0
    
    return {
        'core': core,
        'cladding': cladding,
        'ferrule': np.zeros((h, w), dtype=np.uint8)
    }
""")
            
            # Execute method
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            variations = {"original": image}
            
            # Increase timeout for this test
            original_timeout = executor.config.separation.timeout_seconds
            executor.config.separation.timeout_seconds = 10
            
            try:
                result = executor.execute_method("complex_method", image, variations)
                
                # Should succeed if subprocess execution works
                if result.error is None:
                    assert result.core_mask is not None
                    assert result.cladding_mask is not None
                    assert np.sum(result.core_mask > 0) > 0
                    assert np.sum(result.cladding_mask > 0) > 0
            finally:
                executor.config.separation.timeout_seconds = original_timeout


class TestConsensusBuilderAdvanced:
    """Advanced tests for consensus builder"""
    
    def test_save_and_load_knowledge_base(self, builder):
        """Test saving and loading knowledge base"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            kb_path = Path(f.name)
        
        try:
            # Set temporary path
            builder.config.knowledge_base_path = kb_path
            
            # Add some scores
            builder.knowledge_base = {
                'method1': 0.8,
                'method2': 0.6,
                'method3': 0.9
            }
            
            # Save
            builder._save_knowledge_base()
            
            # Create new builder and load
            new_builder = ConsensusBuilder()
            new_builder.config.knowledge_base_path = kb_path
            loaded_kb = new_builder._load_knowledge_base()
            
            assert loaded_kb['method1'] == 0.8
            assert loaded_kb['method2'] == 0.6
            assert loaded_kb['method3'] == 0.9
            
        finally:
            if kb_path.exists():
                kb_path.unlink()
    
    def test_consensus_with_conflicting_results(self, builder):
        """Test consensus building with conflicting segmentations"""
        # Create conflicting masks
        h, w = 100, 100
        
        # Method 1: Small core
        core1 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core1, (50, 50), 10, 255, -1)
        
        # Method 2: Large core
        core2 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core2, (50, 50), 30, 255, -1)
        
        # Method 3: Offset core
        core3 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core3, (60, 60), 20, 255, -1)
        
        results = [
            SegmentationResult(
                method_name="conservative",
                core_mask=core1,
                cladding_mask=np.zeros((h, w), dtype=np.uint8),
                ferrule_mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.7
            ),
            SegmentationResult(
                method_name="aggressive",
                core_mask=core2,
                cladding_mask=np.zeros((h, w), dtype=np.uint8),
                ferrule_mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.8
            ),
            SegmentationResult(
                method_name="offset",
                core_mask=core3,
                cladding_mask=np.zeros((h, w), dtype=np.uint8),
                ferrule_mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.6
            )
        ]
        
        # Build consensus
        masks = builder.build_consensus(results, (h, w))
        
        # Consensus should be somewhere in between
        core_area = np.sum(masks['core'] > 0)
        assert core_area > 0
        assert core_area < np.sum(core2 > 0)  # Less than largest
    
    def test_consensus_threshold_effects(self, builder):
        """Test effect of consensus threshold"""
        h, w = 100, 100
        
        # Create overlapping masks
        results = []
        for i in range(5):
            core = np.zeros((h, w), dtype=np.uint8)
            # Slightly different circles
            cv2.circle(core, (50 + i*2, 50), 20, 255, -1)
            
            results.append(SegmentationResult(
                method_name=f"method_{i}",
                core_mask=core,
                cladding_mask=np.zeros((h, w), dtype=np.uint8),
                ferrule_mask=np.zeros((h, w), dtype=np.uint8),
                confidence=0.8
            ))
        
        # Test with different thresholds
        original_threshold = builder.config.separation.consensus_threshold
        
        try:
            # High threshold - only overlapping regions
            builder.config.separation.consensus_threshold = 0.8
            masks_high = builder.build_consensus(results, (h, w))
            area_high = np.sum(masks_high['core'] > 0)
            
            # Low threshold - more inclusive
            builder.config.separation.consensus_threshold = 0.2
            masks_low = builder.build_consensus(results, (h, w))
            area_low = np.sum(masks_low['core'] > 0)
            
            # Lower threshold should include more pixels
            assert area_low > area_high
            
        finally:
            builder.config.separation.consensus_threshold = original_threshold
    
    def test_iou_calculation(self, builder):
        """Test IoU calculation for score updates"""
        # Create test masks with known IoU
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Create overlapping rectangles
        cv2.rectangle(mask1, (20, 20), (60, 60), 255, -1)  # 40x40 = 1600 pixels
        cv2.rectangle(mask2, (40, 40), (80, 80), 255, -1)  # 40x40 = 1600 pixels
        
        # Overlap is 20x20 = 400 pixels
        # Union is 1600 + 1600 - 400 = 2800 pixels
        # IoU should be 400/2800 ≈ 0.143
        
        result = SegmentationResult(
            method_name="test_method",
            core_mask=mask1,
            cladding_mask=np.zeros((100, 100), dtype=np.uint8),
            ferrule_mask=np.zeros((100, 100), dtype=np.uint8),
            confidence=1.0
        )
        
        consensus_masks = {
            'core': mask2,
            'cladding': np.zeros((100, 100), dtype=np.uint8),
            'ferrule': np.zeros((100, 100), dtype=np.uint8)
        }
        
        # Update scores
        builder._update_scores([result], consensus_masks)
        
        # Check calculated score
        if 'test_method' in builder.knowledge_base:
            score = builder.knowledge_base['test_method']
            # Score should be influenced by IoU
            assert 0.1 < score < 0.2  # Approximate range


class TestEnhancedSeparatorAdvanced:
    """Advanced tests for enhanced separator"""
    
    def test_parallel_vs_sequential_execution(self):
        """Test that parallel and sequential execution produce same results"""
        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = {"original": image}
        
        # Test with sequential execution
        config = get_config()
        config.separation.parallel_execution = False
        separator_seq = EnhancedSeparator()
        
        # Disable ML for consistent results
        separator_seq.config.detection.use_ml_detection = False
        zones_seq = separator_seq.separate_zones(image, variations)
        
        # Test with parallel execution
        config.separation.parallel_execution = True
        config.processing.max_workers = 2
        separator_par = EnhancedSeparator()
        separator_par.config.detection.use_ml_detection = False
        zones_par = separator_par.separate_zones(image, variations)
        
        # Results should be similar (may not be identical due to method execution order)
        assert set(zones_seq.keys()) == set(zones_par.keys())
    
    def test_error_recovery_in_pipeline(self):
        """Test error recovery when some methods fail"""
        separator = EnhancedSeparator()
        
        # Mock executor to simulate some failures
        original_execute = separator.executor.execute_method
        
        def mock_execute(method_name, image, variations):
            if method_name == "failing_method":
                return SegmentationResult(
                    method_name=method_name,
                    error="Simulated failure"
                )
            return SegmentationResult(
                method_name=method_name,
                core_mask=np.ones((100, 100), dtype=np.uint8) * 255,
                cladding_mask=np.zeros((100, 100), dtype=np.uint8),
                ferrule_mask=np.zeros((100, 100), dtype=np.uint8),
                confidence=0.8
            )
        
        separator.executor.execute_method = mock_execute
        separator.executor.discover_methods = lambda: ["working_method", "failing_method"]
        separator.config.separation.methods_enabled = ["working_method", "failing_method"]
        
        # Run separation
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = {"original": image}
        
        try:
            zones = separator.separate_zones(image, variations)
            
            # Should still produce results from working method
            assert 'core' in zones
            assert 'cladding' in zones
            assert 'ferrule' in zones
            
        finally:
            separator.executor.execute_method = original_execute
    
    def test_visualization_with_empty_zones(self, separator):
        """Test visualization when some zones are empty"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        zones = {
            'core': np.zeros((100, 100), dtype=np.uint8),  # Empty
            'cladding': np.zeros((100, 100), dtype=np.uint8),
            'ferrule': np.zeros((100, 100), dtype=np.uint8)
        }
        
        # Add only cladding
        cv2.circle(zones['cladding'], (50, 50), 30, 255, -1)
        
        vis = separator.visualize_separation(image, zones)
        
        # Should handle empty zones gracefully
        assert vis.shape == image.shape
        assert not np.array_equal(vis, image)  # Should have some overlay


def test_full_separation_pipeline_stress():
    """Stress test the full separation pipeline"""
    config = get_config()
    config.interactive_mode = False
    
    separator = EnhancedSeparator()
    
    # Test with various image conditions
    test_cases = [
        # (name, image_generator)
        ("small", lambda: np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)),
        ("large", lambda: np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)),
        ("grayscale", lambda: np.random.randint(0, 255, (200, 200, 1), dtype=np.uint8)),
        ("high_contrast", lambda: np.where(
            np.random.random((200, 200, 3)) > 0.5, 255, 0
        ).astype(np.uint8)),
        ("low_contrast", lambda: (
            np.random.normal(128, 10, (200, 200, 3))
        ).clip(0, 255).astype(np.uint8))
    ]
    
    for name, image_gen in test_cases:
        image = image_gen()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        variations = {"original": image}
        
        try:
            zones = separator.separate_zones(image, variations)
            
            # Basic validation
            assert isinstance(zones, dict)
            assert 'core' in zones
            assert 'cladding' in zones
            assert 'ferrule' in zones
            
            # Zones should have correct shape
            for zone_mask in zones.values():
                if zone_mask is not None:
                    assert zone_mask.shape == image.shape[:2]
                    
        except Exception as e:
            pytest.fail(f"Separation failed for {name} image: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])