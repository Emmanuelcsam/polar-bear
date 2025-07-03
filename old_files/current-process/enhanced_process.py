"""
Enhanced Image Processing Module
- In-memory processing by default
- ML-powered variation selection
- Multi-variation preprocessing from old-processes
- No argparse, uses config manager
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import concurrent.futures
from functools import lru_cache
import hashlib

from config_manager import get_config
from enhanced_logging import get_logger, log_execution, log_performance, ProgressLogger

logger = get_logger(__name__)


class ProcessingCache:
    """In-memory cache for processed images"""
    
    def __init__(self, max_size_mb: int = 1024):
        self.cache = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size = 0
        self.access_count = {}
    
    def _get_key(self, image: np.ndarray, transform_name: str) -> str:
        """Generate cache key from image and transform"""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        return f"{image_hash}_{transform_name}"
    
    def get(self, image: np.ndarray, transform_name: str) -> Optional[np.ndarray]:
        """Get cached result"""
        key = self._get_key(image, transform_name)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            logger.debug(f"Cache hit for {transform_name}", key=key)
            return self.cache[key].copy()
        return None
    
    def put(self, image: np.ndarray, transform_name: str, result: np.ndarray):
        """Store result in cache"""
        key = self._get_key(image, transform_name)
        size = result.nbytes
        
        # Evict least accessed items if needed
        while self.current_size + size > self.max_size_bytes and self.cache:
            evict_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            evicted = self.cache.pop(evict_key)
            self.current_size -= evicted.nbytes
            self.access_count.pop(evict_key)
            logger.debug(f"Evicted from cache", key=evict_key)
        
        self.cache[key] = result.copy()
        self.current_size += size
        self.access_count[key] = 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
        self.current_size = 0


class MLVariationSelector:
    """ML-powered selection of useful variations"""
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ML model for variation selection"""
        if not self.config.processing.ml_enabled:
            return
        
        try:
            if self.config.processing.pytorch_enabled:
                self._load_pytorch_model()
            elif self.config.processing.tensorflow_enabled:
                self._load_tensorflow_model()
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            self.model = None
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        try:
            import torch
            import torch.nn as nn
            
            class VariationSelector(nn.Module):
                def __init__(self, num_variations: int = 49):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(64, num_variations)
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x).squeeze(-1).squeeze(-1)
                    x = self.sigmoid(self.fc(x))
                    return x
            
            self.model = VariationSelector()
            
            # Try to load pre-trained weights
            model_path = self.config.model_dir / "variation_selector.pth"
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                logger.info("Loaded PyTorch variation selector model")
            else:
                logger.info("Using untrained PyTorch model")
                
        except ImportError:
            logger.warning("PyTorch not available")
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model"""
        try:
            import tensorflow as tf
            
            model_path = self.config.model_dir / "variation_selector.h5"
            if model_path.exists():
                self.model = tf.keras.models.load_model(model_path)
                logger.info("Loaded TensorFlow variation selector model")
            else:
                # Create simple model
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(None, None, 3)),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(49, activation='sigmoid')
                ])
                logger.info("Using untrained TensorFlow model")
                
        except ImportError:
            logger.warning("TensorFlow not available")
    
    def predict_useful_variations(self, image: np.ndarray) -> List[str]:
        """Predict which variations will be useful for this image"""
        if self.model is None:
            # Return all variations if no model
            return list(TRANSFORM_FUNCTIONS.keys())
        
        try:
            # Prepare image
            if image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize for model
            input_size = (224, 224)
            resized = cv2.resize(image, input_size)
            
            if self.config.processing.pytorch_enabled:
                import torch
                x = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    scores = self.model(x).squeeze().numpy()
            else:
                x = resized[np.newaxis, ...] / 255.0
                scores = self.model.predict(x, verbose=0)[0]
            
            # Select variations with score > threshold
            threshold = 0.5
            variation_names = list(TRANSFORM_FUNCTIONS.keys())
            selected = [name for name, score in zip(variation_names, scores) if score > threshold]
            
            # Ensure minimum variations
            if len(selected) < 10:
                # Add top scoring variations
                indices = np.argsort(scores)[-10:]
                selected = [variation_names[i] for i in indices]
            
            logger.info(f"ML selected {len(selected)} variations")
            return selected
            
        except Exception as e:
            logger.error(f"Variation prediction failed: {e}")
            return list(TRANSFORM_FUNCTIONS.keys())


class EnhancedProcessor:
    """Enhanced image processor with 49 variations and ML integration"""
    
    def __init__(self):
        self.config = get_config()
        self.cache = ProcessingCache() if self.config.processing.cache_enabled else None
        self.ml_selector = MLVariationSelector() if self.config.processing.ml_enabled else None
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.processing.max_workers
        ) if self.config.processing.parallel_processing else None
    
    @log_performance
    def process_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        """Process image with multiple variations"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get variations to apply
        if self.ml_selector:
            variations_to_apply = self.ml_selector.predict_useful_variations(image)
        else:
            variations_to_apply = list(TRANSFORM_FUNCTIONS.keys())[:self.config.processing.num_variations]
        
        # Process variations
        results = {"original": image}
        
        if self.config.processing.parallel_processing and self.executor:
            # Parallel processing
            futures = {}
            for name in variations_to_apply:
                future = self.executor.submit(self._apply_transform, image, name)
                futures[future] = name
            
            # Collect results
            progress = ProgressLogger("processing", len(futures))
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[name] = result
                    progress.update(1, f"Completed {name}")
                except Exception as e:
                    logger.error(f"Transform {name} failed: {e}")
        else:
            # Sequential processing
            for name in variations_to_apply:
                try:
                    result = self._apply_transform(image, name)
                    if result is not None:
                        results[name] = result
                except Exception as e:
                    logger.error(f"Transform {name} failed: {e}")
        
        logger.info(f"Generated {len(results)} variations", count=len(results))
        return results
    
    def _apply_transform(self, image: np.ndarray, transform_name: str) -> Optional[np.ndarray]:
        """Apply a single transform with caching"""
        # Check cache
        if self.cache:
            cached = self.cache.get(image, transform_name)
            if cached is not None:
                return cached
        
        # Apply transform
        try:
            transform_func = TRANSFORM_FUNCTIONS.get(transform_name)
            if transform_func is None:
                logger.warning(f"Unknown transform: {transform_name}")
                return None
            
            result = transform_func(image)
            
            # Store in cache
            if self.cache and result is not None:
                self.cache.put(image, transform_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Transform {transform_name} failed: {e}")
            return None
    
    def process_batch(self, image_paths: List[Path]) -> Dict[Path, Dict[str, np.ndarray]]:
        """Process multiple images"""
        results = {}
        progress = ProgressLogger("batch_processing", len(image_paths))
        
        for path in image_paths:
            try:
                results[path] = self.process_image(path)
                progress.update(1, f"Processed {path.name}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results[path] = {}
        
        progress.complete()
        return results
    
    def __del__(self):
        """Cleanup"""
        if self.executor:
            self.executor.shutdown(wait=True)


# Transform functions (49 variations based on old-processes)
@lru_cache(maxsize=32)
def get_kernel(size: int, shape: str = 'ellipse') -> np.ndarray:
    """Get morphological kernel"""
    if shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


# Thresholding variations (8)
def threshold_otsu(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def threshold_adaptive_mean(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def threshold_adaptive_gaussian(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def threshold_binary(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return result

def threshold_binary_inv(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return result

def threshold_trunc(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    return result

def threshold_tozero(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    return result

def threshold_tozero_inv(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
    return result

# Masking (1)
def circular_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, int(radius * 0.9), 255, -1)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# Color space transformations (14)
def color_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def color_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def colormap_jet(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def colormap_hot(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

def colormap_cool(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_COOL)

def colormap_hsv_map(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_HSV)

def colormap_rainbow(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)

def colormap_ocean(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

def colormap_summer(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)

def colormap_spring(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)

def colormap_winter(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)

def colormap_autumn(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_AUTUMN)

def colormap_bone(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)

def colormap_pink(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_PINK)

# Preprocessing operations (16)
def blur_gaussian(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), 0)

def blur_median(img: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(img, 5)

def blur_bilateral(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, 9, 75, 75)

def morph_open(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(5)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def morph_close(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(5)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def morph_gradient(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(3)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def morph_tophat(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(9)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def morph_blackhat(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(9)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def edge_canny(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Canny(gray, 50, 150)

def edge_sobel(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return (magnitude * 255 / magnitude.max()).astype(np.uint8)

def edge_laplacian(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

def denoise_nlmeans(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def enhance_clahe(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def enhance_histogram(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        # Apply to each channel
        channels = cv2.split(img)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)
    else:
        return cv2.equalizeHist(img)

def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradx, grady)
    return (magnitude * 255 / magnitude.max()).astype(np.uint8)

def gradient_direction(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    direction = cv2.phase(gradx, grady, angleInDegrees=True)
    return (direction * 255 / 360).astype(np.uint8)

# Resizing operations (2)
def resize_half(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def resize_double(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

# Intensity/Binary manipulations (8)
def intensity_invert(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)

def intensity_normalize(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def intensity_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def intensity_log(img: np.ndarray) -> np.ndarray:
    img_float = img.astype(np.float32) + 1
    log_img = np.log(img_float)
    return (log_img * 255 / log_img.max()).astype(np.uint8)

def binary_erode(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(3)
    return cv2.erode(img, kernel, iterations=1)

def binary_dilate(img: np.ndarray) -> np.ndarray:
    kernel = get_kernel(3)
    return cv2.dilate(img, kernel, iterations=1)

def binary_skeleton(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    skeleton = np.zeros(binary.shape, np.uint8)
    kernel = get_kernel(3, 'cross')
    
    while True:
        eroded = cv2.erode(binary, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    
    return skeleton

def binary_distance(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return (dist * 255 / dist.max()).astype(np.uint8)


# Dictionary of all transform functions
TRANSFORM_FUNCTIONS = {
    # Thresholding (8)
    "threshold_otsu": threshold_otsu,
    "threshold_adaptive_mean": threshold_adaptive_mean,
    "threshold_adaptive_gaussian": threshold_adaptive_gaussian,
    "threshold_binary": threshold_binary,
    "threshold_binary_inv": threshold_binary_inv,
    "threshold_trunc": threshold_trunc,
    "threshold_tozero": threshold_tozero,
    "threshold_tozero_inv": threshold_tozero_inv,
    
    # Masking (1)
    "circular_mask": circular_mask,
    
    # Color transformations (14)
    "color_hsv": color_hsv,
    "color_lab": color_lab,
    "colormap_jet": colormap_jet,
    "colormap_hot": colormap_hot,
    "colormap_cool": colormap_cool,
    "colormap_hsv": colormap_hsv_map,
    "colormap_rainbow": colormap_rainbow,
    "colormap_ocean": colormap_ocean,
    "colormap_summer": colormap_summer,
    "colormap_spring": colormap_spring,
    "colormap_winter": colormap_winter,
    "colormap_autumn": colormap_autumn,
    "colormap_bone": colormap_bone,
    "colormap_pink": colormap_pink,
    
    # Preprocessing (16)
    "blur_gaussian": blur_gaussian,
    "blur_median": blur_median,
    "blur_bilateral": blur_bilateral,
    "morph_open": morph_open,
    "morph_close": morph_close,
    "morph_gradient": morph_gradient,
    "morph_tophat": morph_tophat,
    "morph_blackhat": morph_blackhat,
    "edge_canny": edge_canny,
    "edge_sobel": edge_sobel,
    "edge_laplacian": edge_laplacian,
    "denoise_nlmeans": denoise_nlmeans,
    "enhance_clahe": enhance_clahe,
    "enhance_histogram": enhance_histogram,
    "gradient_magnitude": gradient_magnitude,
    "gradient_direction": gradient_direction,
    
    # Resizing (2)
    "resize_half": resize_half,
    "resize_double": resize_double,
    
    # Intensity/Binary (8)
    "intensity_invert": intensity_invert,
    "intensity_normalize": intensity_normalize,
    "intensity_gamma": lambda img: intensity_gamma(img, 1.5),
    "intensity_log": intensity_log,
    "binary_erode": binary_erode,
    "binary_dilate": binary_dilate,
    "binary_skeleton": binary_skeleton,
    "binary_distance": binary_distance,
}


def main():
    """Main function for testing"""
    config = get_config()
    config.interactive_mode = False  # Disable for testing
    
    processor = EnhancedProcessor()
    
    # Test with a sample image
    test_image = config.input_dir / "test.jpg"
    if test_image.exists():
        results = processor.process_image(test_image)
        logger.info(f"Generated variations: {list(results.keys())}")
    else:
        logger.warning("No test image found")


if __name__ == "__main__":
    main()