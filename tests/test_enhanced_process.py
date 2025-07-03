"""
Unit tests for enhanced processing module
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "current-process"))

from enhanced_process import (
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])