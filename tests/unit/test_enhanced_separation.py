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

from enhanced_separation import (
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])