"""
Unit tests for enhanced processing module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from process import (
    EnhancedProcessor, ProcessingCache, MLVariationSelector,
    TRANSFORM_FUNCTIONS, get_kernel
)
from config_manager import get_config


class TestProcessingCache:
    """Test the in-memory processing cache"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = ProcessingCache(max_size_mb=10)
        assert cache.max_size_bytes == 10 * 1024 * 1024
        assert cache.current_size == 0
        assert len(cache.cache) == 0
    
    def test_cache_put_get(self):
        """Test storing and retrieving from cache"""
        cache = ProcessingCache()
        
        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transform_name = "test_transform"
        result = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Store in cache
        cache.put(image, transform_name, result)
        
        # Retrieve from cache
        cached_result = cache.get(image, transform_name)
        assert cached_result is not None
        assert np.array_equal(cached_result, result)
    
    def test_cache_miss(self):
        """Test cache miss"""
        cache = ProcessingCache()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = cache.get(image, "nonexistent")
        assert result is None
    
    def test_cache_eviction(self):
        """Test cache eviction when size limit reached"""
        cache = ProcessingCache(max_size_mb=1)  # Small cache
        
        # Fill cache with large images
        for i in range(10):
            image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            result = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
            cache.put(image, f"transform_{i}", result)
        
        # Check that cache size is within limits
        assert cache.current_size <= cache.max_size_bytes
    
    def test_cache_clear(self):
        """Test clearing cache"""
        cache = ProcessingCache()
        
        # Add some items
        for i in range(5):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cache.put(image, f"transform_{i}", result)
        
        # Clear cache
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.current_size == 0


class TestMLVariationSelector:
    """Test ML-based variation selection"""
    
    def test_initialization_no_ml(self):
        """Test initialization when ML is disabled"""
        config = get_config()
        config.processing.ml_enabled = False
        
        selector = MLVariationSelector()
        assert selector.model is None
    
    def test_predict_without_model(self):
        """Test prediction when no model is available"""
        config = get_config()
        config.processing.ml_enabled = False
        
        selector = MLVariationSelector()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        variations = selector.predict_useful_variations(image)
        # Should return all variations when no model
        assert len(variations) == len(TRANSFORM_FUNCTIONS)
    
    def test_predict_with_grayscale(self):
        """Test prediction with grayscale image"""
        selector = MLVariationSelector()
        selector.model = None  # Force no model
        
        # Grayscale image
        image = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        variations = selector.predict_useful_variations(image)
        
        assert isinstance(variations, list)
        assert len(variations) > 0


class TestTransformFunctions:
    """Test individual transform functions"""
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def test_image_gray(self):
        """Create grayscale test image"""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_threshold_transforms(self, test_image):
        """Test thresholding transforms"""
        threshold_funcs = [
            'threshold_otsu', 'threshold_adaptive_mean', 'threshold_adaptive_gaussian',
            'threshold_binary', 'threshold_binary_inv', 'threshold_trunc',
            'threshold_tozero', 'threshold_tozero_inv'
        ]
        
        for func_name in threshold_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert len(result.shape) == 2  # Should be grayscale
            assert result.dtype == np.uint8
    
    def test_color_transforms(self, test_image):
        """Test color space transforms"""
        color_funcs = ['color_hsv', 'color_lab']
        
        for func_name in color_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert result.shape == test_image.shape
            assert len(result.shape) == 3  # Should be color
    
    def test_colormap_transforms(self, test_image):
        """Test colormap transforms"""
        colormap_funcs = [
            'colormap_jet', 'colormap_hot', 'colormap_cool', 'colormap_hsv',
            'colormap_rainbow', 'colormap_ocean', 'colormap_summer', 'colormap_spring',
            'colormap_winter', 'colormap_autumn', 'colormap_bone', 'colormap_pink'
        ]
        
        for func_name in colormap_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert len(result.shape) == 3  # Should be color
            assert result.shape[2] == 3
    
    def test_blur_transforms(self, test_image):
        """Test blur transforms"""
        blur_funcs = ['blur_gaussian', 'blur_median', 'blur_bilateral']
        
        for func_name in blur_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert result.shape == test_image.shape
    
    def test_morphological_transforms(self, test_image):
        """Test morphological transforms"""
        morph_funcs = [
            'morph_open', 'morph_close', 'morph_gradient',
            'morph_tophat', 'morph_blackhat'
        ]
        
        for func_name in morph_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert result.shape == test_image.shape
    
    def test_edge_transforms(self, test_image):
        """Test edge detection transforms"""
        edge_funcs = ['edge_canny', 'edge_sobel', 'edge_laplacian']
        
        for func_name in edge_funcs:
            func = TRANSFORM_FUNCTIONS[func_name]
            result = func(test_image)
            assert result is not None
            assert len(result.shape) == 2  # Should be grayscale
    
    def test_resize_transforms(self, test_image):
        """Test resize transforms"""
        # Half size
        result = TRANSFORM_FUNCTIONS['resize_half'](test_image)
        assert result.shape[0] == test_image.shape[0] // 2
        assert result.shape[1] == test_image.shape[1] // 2
        
        # Double size
        result = TRANSFORM_FUNCTIONS['resize_double'](test_image)
        assert result.shape[0] == test_image.shape[0] * 2
        assert result.shape[1] == test_image.shape[1] * 2
    
    def test_circular_mask(self, test_image):
        """Test circular mask transform"""
        result = TRANSFORM_FUNCTIONS['circular_mask'](test_image)
        assert result is not None
        assert result.shape == test_image.shape
        
        # Check that corners are masked (should be black)
        assert np.all(result[0, 0] == 0)
        assert np.all(result[-1, -1] == 0)
    
    def test_get_kernel(self):
        """Test kernel generation"""
        # Ellipse kernel
        kernel = get_kernel(5, 'ellipse')
        assert kernel.shape == (5, 5)
        assert kernel.dtype == np.uint8
        
        # Cross kernel
        kernel = get_kernel(5, 'cross')
        assert kernel.shape == (5, 5)
        
        # Rectangle kernel
        kernel = get_kernel(5, 'rect')
        assert kernel.shape == (5, 5)


class TestEnhancedProcessor:
    """Test the main enhanced processor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        config = get_config()
        config.interactive_mode = False
        config.processing.parallel_processing = False  # For easier testing
        return EnhancedProcessor()
    
    @pytest.fixture
    def test_image_file(self):
        """Create temporary test image file"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, image)
            yield Path(f.name)
            os.unlink(f.name)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.config is not None
        assert processor.cache is not None or not processor.config.processing.cache_enabled
        assert processor.ml_selector is not None or not processor.config.processing.ml_enabled
    
    def test_process_image(self, processor, test_image_file):
        """Test processing single image"""
        results = processor.process_image(test_image_file)
        
        assert isinstance(results, dict)
        assert 'original' in results
        assert len(results) > 1  # Should have variations
        
        # Check that all results are numpy arrays
        for name, image in results.items():
            assert isinstance(image, np.ndarray)
            assert image.dtype == np.uint8
    
    def test_process_batch(self, processor):
        """Test batch processing"""
        # Create multiple test images
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_files = []
            
            for i in range(3):
                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                path = tmpdir / f"test_{i}.jpg"
                cv2.imwrite(str(path), image)
                image_files.append(path)
            
            # Process batch
            results = processor.process_batch(image_files)
            
            assert isinstance(results, dict)
            assert len(results) == 3
            
            for path in image_files:
                assert path in results
                assert isinstance(results[path], dict)
                assert 'original' in results[path]
    
    def test_apply_transform_with_cache(self, processor):
        """Test transform application with caching"""
        if not processor.cache:
            pytest.skip("Cache not enabled")
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transform_name = 'blur_gaussian'
        
        # First call - should compute
        result1 = processor._apply_transform(image, transform_name)
        assert result1 is not None
        
        # Second call - should use cache
        result2 = processor._apply_transform(image, transform_name)
        assert result2 is not None
        assert np.array_equal(result1, result2)
    
    def test_apply_unknown_transform(self, processor):
        """Test applying unknown transform"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = processor._apply_transform(image, 'unknown_transform')
        assert result is None
    
    def test_transform_error_handling(self, processor):
        """Test error handling in transforms"""
        # Create invalid image
        invalid_image = np.array([])
        
        result = processor._apply_transform(invalid_image, 'blur_gaussian')
        assert result is None


def test_all_transforms_exist():
    """Test that all transforms in dictionary are valid"""
    for name, func in TRANSFORM_FUNCTIONS.items():
        assert callable(func), f"Transform {name} is not callable"
        assert hasattr(func, '__call__'), f"Transform {name} has no __call__ method"


def test_transform_count():
    """Test that we have approximately 49 transforms as expected"""
    assert len(TRANSFORM_FUNCTIONS) >= 45, "Should have at least 45 transforms"
    assert len(TRANSFORM_FUNCTIONS) <= 55, "Should have at most 55 transforms"


class TestAdditionalTransformFunctions:
    """Additional tests for transform functions edge cases"""
    
    def test_intensity_gamma_transform(self):
        """Test gamma intensity transform"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        from process import intensity_gamma
        
        # Test with different gamma values
        result1 = intensity_gamma(image, gamma=1.5)
        result2 = intensity_gamma(image, gamma=0.5)
        
        assert result1 is not None
        assert result2 is not None
        assert result1.shape == image.shape
        assert result2.shape == image.shape
        
        # Results should be different for different gamma values
        assert not np.array_equal(result1, result2)
    
    def test_intensity_log_transform(self):
        """Test log intensity transform"""
        image = np.random.randint(1, 255, (100, 100, 3), dtype=np.uint8)
        from process import intensity_log
        
        result = intensity_log(image)
        assert result is not None
        assert result.shape == image.shape
        assert result.dtype == np.uint8
    
    def test_binary_skeleton_transform(self):
        """Test binary skeleton transform"""
        # Create image with thick line
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(image, (20, 50), (80, 50), (255, 255, 255), 10)
        
        from process import binary_skeleton
        result = binary_skeleton(image)
        
        assert result is not None
        assert len(result.shape) == 2  # Should be grayscale
        # Skeleton should be thinner than original
        assert np.sum(result > 0) < np.sum(image[:,:,0] > 0)
    
    def test_binary_distance_transform(self):
        """Test binary distance transform"""
        # Create image with object
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 20, (255, 255, 255), -1)
        
        from process import binary_distance
        result = binary_distance(image)
        
        assert result is not None
        assert len(result.shape) == 2
        # Center should have highest distance value
        assert result[50, 50] > result[50, 35]
    
    def test_transform_with_empty_image(self):
        """Test transforms with empty/black image"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test a few transforms that might have issues with empty images
        from process import edge_canny, threshold_otsu, gradient_magnitude
        
        result1 = edge_canny(empty_image)
        assert result1 is not None
        assert np.all(result1 == 0)  # Should be all zeros
        
        result2 = threshold_otsu(empty_image)
        assert result2 is not None
        
        result3 = gradient_magnitude(empty_image)
        assert result3 is not None
        assert np.all(result3 == 0)  # No gradients in empty image
    
    def test_transform_with_single_channel(self):
        """Test color transforms with grayscale input"""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # These should handle grayscale gracefully
        from process import color_hsv, color_lab
        
        # Should convert to color first internally
        result1 = TRANSFORM_FUNCTIONS['threshold_otsu'](gray_image)
        assert result1 is not None
        
        # Edge detection should work with grayscale
        result2 = TRANSFORM_FUNCTIONS['edge_canny'](gray_image)
        assert result2 is not None


class TestProcessingCacheAdvanced:
    """Advanced tests for ProcessingCache"""
    
    def test_cache_key_generation(self):
        """Test cache key generation is deterministic"""
        cache = ProcessingCache()
        
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transform_name = "test_transform"
        
        key1 = cache._get_key(image, transform_name)
        key2 = cache._get_key(image, transform_name)
        
        assert key1 == key2
        
        # Different image should have different key
        image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        key3 = cache._get_key(image2, transform_name)
        assert key1 != key3
    
    def test_cache_access_counting(self):
        """Test cache access counting"""
        cache = ProcessingCache()
        
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # Put item in cache
        cache.put(image, "transform1", result)
        
        # Access multiple times
        for _ in range(3):
            cache.get(image, "transform1")
        
        key = cache._get_key(image, "transform1")
        assert cache.access_count[key] == 4  # 1 from put + 3 from get
    
    def test_cache_eviction_lru(self):
        """Test LRU eviction policy"""
        cache = ProcessingCache(max_size_mb=1)  # Very small cache
        
        # Add items until cache is full
        items = []
        for i in range(5):
            image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            result = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
            cache.put(image, f"transform_{i}", result)
            items.append((image, f"transform_{i}"))
        
        # Access first item multiple times (make it most recently used)
        for _ in range(5):
            cache.get(items[0][0], items[0][1])
        
        # Add new large item
        new_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        new_result = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        cache.put(new_image, "new_transform", new_result)
        
        # First item should still be in cache (most accessed)
        assert cache.get(items[0][0], items[0][1]) is not None


class TestMLVariationSelectorAdvanced:
    """Advanced tests for ML variation selector"""
    
    def test_ml_model_pytorch_fallback(self):
        """Test PyTorch model loading and fallback"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = True
        config.processing.tensorflow_enabled = False
        
        selector = MLVariationSelector()
        
        # Should handle missing PyTorch gracefully
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = selector.predict_useful_variations(image)
        
        assert isinstance(variations, list)
        assert len(variations) > 0
    
    def test_ml_model_tensorflow_fallback(self):
        """Test TensorFlow model loading and fallback"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.pytorch_enabled = False
        config.processing.tensorflow_enabled = True
        
        selector = MLVariationSelector()
        
        # Should handle missing TensorFlow gracefully
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = selector.predict_useful_variations(image)
        
        assert isinstance(variations, list)
        assert len(variations) > 0
    
    def test_variation_selection_minimum_count(self):
        """Test that minimum number of variations are selected"""
        selector = MLVariationSelector()
        selector.model = None  # Force no model
        
        # Even without model, should return reasonable number
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        variations = selector.predict_useful_variations(image)
        
        assert len(variations) >= 10  # Minimum variations


class TestEnhancedProcessorAdvanced:
    """Advanced tests for EnhancedProcessor"""
    
    def test_parallel_processing_disabled(self):
        """Test processor with parallel processing disabled"""
        config = get_config()
        config.processing.parallel_processing = False
        
        processor = EnhancedProcessor()
        assert processor.executor is None
    
    def test_parallel_processing_enabled(self):
        """Test processor with parallel processing enabled"""
        config = get_config()
        config.processing.parallel_processing = True
        config.processing.max_workers = 2
        
        processor = EnhancedProcessor()
        assert processor.executor is not None
        
        # Cleanup
        processor.__del__()
    
    def test_process_image_with_ml_selection(self):
        """Test image processing with ML variation selection"""
        config = get_config()
        config.processing.ml_enabled = True
        config.processing.num_variations = 5
        
        processor = EnhancedProcessor()
        
        # Mock ML selector
        processor.ml_selector = Mock()
        processor.ml_selector.predict_useful_variations = Mock(
            return_value=['blur_gaussian', 'edge_canny', 'threshold_otsu']
        )
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, image)
            
            try:
                results = processor.process_image(Path(f.name))
                
                # Should only have selected variations plus original
                assert 'original' in results
                assert 'blur_gaussian' in results
                assert 'edge_canny' in results
                assert 'threshold_otsu' in results
                assert len(results) == 4
                
            finally:
                os.unlink(f.name)
    
    def test_process_image_invalid_path(self):
        """Test processing with invalid image path"""
        processor = EnhancedProcessor()
        
        with pytest.raises(ValueError, match="Failed to load image"):
            processor.process_image(Path("nonexistent.jpg"))
    
    def test_process_batch_error_handling(self):
        """Test batch processing with some failing images"""
        processor = EnhancedProcessor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create one valid and one invalid path
            valid_path = tmpdir / "valid.jpg"
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(valid_path), image)
            
            invalid_path = tmpdir / "invalid.jpg"
            
            # Process batch
            results = processor.process_batch([valid_path, invalid_path])
            
            # Should have results for valid image
            assert valid_path in results
            assert isinstance(results[valid_path], dict)
            assert 'original' in results[valid_path]
            
            # Should have empty dict for invalid image
            assert invalid_path in results
            assert results[invalid_path] == {}
    
    def test_apply_transform_exception_handling(self):
        """Test transform application with exceptions"""
        processor = EnhancedProcessor()
        
        # Create a transform that will fail
        def failing_transform(img):
            raise RuntimeError("Transform failed")
        
        # Temporarily add failing transform
        original_transform = TRANSFORM_FUNCTIONS.get('test_failing', None)
        TRANSFORM_FUNCTIONS['test_failing'] = failing_transform
        
        try:
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = processor._apply_transform(image, 'test_failing')
            
            assert result is None  # Should return None on failure
            
        finally:
            # Restore original state
            if original_transform is None:
                del TRANSFORM_FUNCTIONS['test_failing']
            else:
                TRANSFORM_FUNCTIONS['test_failing'] = original_transform


def test_transform_consistency():
    """Test that transforms produce consistent results"""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test a few transforms for consistency
    test_transforms = ['blur_gaussian', 'threshold_otsu', 'edge_canny']
    
    for transform_name in test_transforms:
        transform = TRANSFORM_FUNCTIONS[transform_name]
        
        # Apply transform multiple times
        result1 = transform(image.copy())
        result2 = transform(image.copy())
        
        # Results should be identical
        assert np.array_equal(result1, result2), f"{transform_name} is not deterministic"


def test_all_transforms_handle_edge_cases():
    """Test all transforms with edge case images"""
    edge_cases = {
        'empty': np.zeros((50, 50, 3), dtype=np.uint8),
        'full': np.ones((50, 50, 3), dtype=np.uint8) * 255,
        'single_pixel': np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8),
        'thin': np.random.randint(0, 255, (100, 1, 3), dtype=np.uint8),
        'grayscale_3d': np.random.randint(0, 255, (50, 50, 1), dtype=np.uint8),
    }
    
    for case_name, test_image in edge_cases.items():
        for transform_name, transform in TRANSFORM_FUNCTIONS.items():
            try:
                result = transform(test_image)
                assert result is not None, f"{transform_name} returned None for {case_name}"
                assert isinstance(result, np.ndarray), f"{transform_name} didn't return ndarray for {case_name}"
                assert result.dtype == np.uint8, f"{transform_name} wrong dtype for {case_name}"
            except Exception as e:
                # Some transforms might not handle single pixel images
                if case_name == 'single_pixel' and 'convolution' in str(e).lower():
                    continue
                pytest.fail(f"{transform_name} failed on {case_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])