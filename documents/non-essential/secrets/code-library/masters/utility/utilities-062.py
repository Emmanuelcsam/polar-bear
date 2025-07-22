#!/usr/bin/env python3
"""
Unified Helper Module for Fiber Optic Analysis
=============================================
This module combines all helper functions from various scripts into a single,
comprehensive utility module for image processing, analysis, and computer vision tasks.
"""

# Standard library imports
import os
import json
import time
import logging
import warnings
import argparse
import gc
import zipfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from functools import wraps

# Third-party imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import requests
from torch import nn
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(name: str = __name__, log_file: str = None, level: int = logging.INFO):
    """Configure comprehensive logging system."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class Logger:
    """Custom logger class with JSON file support."""
    
    def __init__(self, module_name):
        self.module = module_name
        self.log_file = 'system_log.json'
        self.load_log()
    
    def load_log(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
        else:
            self.logs = []
    
    def log(self, message, level='INFO'):
        entry = {
            'timestamp': time.time(),
            'module': self.module,
            'level': level,
            'message': message
        }
        
        # Print to terminal
        print(f"[{self.module}] {message}")
        
        # Add to log
        self.logs.append(entry)
        
        # Keep only last 1000 entries
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
        
        # Save log
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f)
    
    def error(self, message):
        self.log(message, 'ERROR')
    
    def warning(self, message):
        self.log(message, 'WARNING')
    
    def info(self, message):
        self.log(message, 'INFO')


def log_message(message: str, level: str = "INFO"):
    """Prints a timestamped log message to the console."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [{level.upper()}] {message}")


def quick_log(module, message):
    """Utility function for quick logging."""
    logger = Logger(module)
    logger.log(message)


# Configure default logger
logger = setup_logging(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ZoneDefinition:
    """Data structure to define parameters for a fiber zone."""
    name: str
    r_min_factor_or_um: float
    r_max_factor_or_um: float
    color_bgr: Tuple[int, int, int]
    max_defect_size_um: Optional[float] = None
    defects_allowed: bool = True


# =============================================================================
# JSON ENCODING
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy data types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_numpy_data(data, filepath, indent=2):
    """Save data containing numpy types to JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)


def dumps_numpy(obj, **kwargs):
    """Serialize numpy-containing data to JSON string."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

def performance_timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper


def memory_monitor(func: Callable) -> Callable:
    """Decorator to monitor memory usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        result = func(*args, **kwargs)
        gc.collect()
        return result
    return wrapper


def start_timer() -> float:
    """Returns the current time to start a timer."""
    return time.perf_counter()


def log_duration(operation_name: str, start_time: float, image_result: Optional[Any] = None):
    """Logs the duration of an operation."""
    duration = time.perf_counter() - start_time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{current_time}] [INFO] Operation '{operation_name}' completed in {duration:.4f} seconds.")
    
    if image_result and hasattr(image_result, 'timing_log') and isinstance(image_result.timing_log, dict):
        image_result.timing_log[operation_name] = duration
    
    return duration


def time_function(func: Callable, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Times a function call and returns both result and execution time."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time


def print_train_time(start, end, device=None):
    """Prints difference between start and end time."""
    total_time = end - start
    logger.info(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def load_image(image_path):
    """Loads an image from a file path."""
    return cv2.imread(str(image_path))


def list_images(folder_path):
    """Lists all images in a folder."""
    path = Path(folder_path)
    return list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.bmp"))


def resize_for_processing(image: np.ndarray, max_dimension: int = 1024, 
                         maintain_aspect_ratio: bool = True) -> Tuple[np.ndarray, float]:
    """Resize image for optimal processing while maintaining aspect ratio."""
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dimension:
        return image, 1.0
    
    if maintain_aspect_ratio:
        scale_factor = max_dimension / max(height, width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
    else:
        scale_factor = max_dimension / max(height, width)
        new_width = new_height = max_dimension
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def scale_coordinates(coordinates: np.ndarray, scale_factor: float, 
                     inverse: bool = False) -> np.ndarray:
    """Scale coordinates based on image resize factor."""
    if inverse:
        scale_factor = 1.0 / scale_factor
    return coordinates * scale_factor


def optimize_kernel_size(image_size: Tuple[int, int], base_kernel_size: int = 5) -> int:
    """Optimize kernel size based on image dimensions."""
    height, width = image_size
    image_diagonal = np.sqrt(height**2 + width**2)
    scale_factor = image_diagonal / 1000.0
    optimized_size = int(base_kernel_size * scale_factor)
    
    if optimized_size % 2 == 0:
        optimized_size += 1
    
    return max(3, optimized_size)


def create_processing_roi(image_shape: Tuple[int, int], center: Tuple[int, int], 
                         radius: int, padding: float = 0.2) -> Tuple[slice, slice]:
    """Create a region of interest (ROI) for focused processing."""
    height, width = image_shape
    center_x, center_y = center
    
    roi_radius = int(radius * (1 + padding))
    
    y_min = max(0, center_y - roi_radius)
    y_max = min(height, center_y + roi_radius)
    x_min = max(0, center_x - roi_radius)
    x_max = min(width, center_x + roi_radius)
    
    return slice(y_min, y_max), slice(x_min, x_max)


def optimize_image_dtype(image: np.ndarray, target_dtype: np.dtype = np.uint8) -> np.ndarray:
    """Optimize image data type for processing efficiency."""
    if image.dtype == target_dtype:
        return image
    
    if target_dtype == np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            return (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return (image >> 8).astype(np.uint8)
    elif target_dtype == np.float32:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
    
    return image.astype(target_dtype)


def check_gpu_availability() -> bool:
    """Check if GPU acceleration is available for OpenCV."""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("GPU acceleration available")
            return True
        else:
            logger.info("No CUDA-enabled devices found")
            return False
    except AttributeError:
        logger.info("OpenCV not built with CUDA support")
        return False


# =============================================================================
# ADAPTIVE THRESHOLDING
# =============================================================================

def niblack_threshold(image, window_size=15, k=-0.2, mask=None):
    """Niblack's adaptive thresholding method."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    threshold_map = mean + k * std
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'k': k,
        'window_size': window_size
    }


def sauvola_threshold(image, window_size=15, k=0.5, R=128, mask=None):
    """Sauvola's adaptive thresholding method."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    threshold_map = mean * (1 + k * ((std / R) - 1))
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'k': k,
        'R': R,
        'window_size': window_size
    }


def local_contrast_threshold(image, window_size=15, contrast_threshold=0.3, mask=None):
    """Local contrast-based thresholding for detecting regions with high local variation."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    kernel = np.ones((window_size, window_size), dtype=np.uint8)
    
    local_min = cv2.erode(img_float, kernel)
    local_max = cv2.dilate(img_float, kernel)
    
    denominator = local_max + local_min + 1e-7
    local_contrast = (local_max - local_min) / denominator
    
    contrast_defects = (local_contrast > contrast_threshold) & mask
    
    return {
        'contrast_map': local_contrast,
        'binary_result': contrast_defects,
        'defects': contrast_defects,
        'local_min': local_min,
        'local_max': local_max,
        'contrast_threshold': contrast_threshold,
        'window_size': window_size
    }


def phansalkar_threshold(image, window_size=15, k=0.25, p=2.0, q=10.0, mask=None):
    """Phansalkar's adaptive thresholding method."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    img_float = image.astype(np.float64)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float**2, (window_size, window_size))
    std = np.sqrt(np.maximum(0, mean_sq - mean**2))
    
    global_mean = np.mean(img_float[mask]) if np.any(mask) else np.mean(img_float)
    threshold_map = mean * (1 + p * np.exp(-q * mean) + k * ((std / global_mean) - 1))
    binary_result = (img_float < threshold_map) & mask
    
    return {
        'threshold_map': threshold_map,
        'binary_result': binary_result,
        'defects': binary_result,
        'local_mean': mean,
        'local_std': std,
        'global_mean': global_mean,
        'k': k,
        'p': p,
        'q': q,
        'window_size': window_size
    }


def multiscale_adaptive_threshold(image, window_sizes=[5, 10, 15, 20], method='sauvola', mask=None):
    """Multi-scale adaptive thresholding combining results from different window sizes."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    scale_results = {}
    combined_defects = np.zeros_like(mask, dtype=bool)
    
    for window_size in window_sizes:
        if method == 'niblack':
            result = niblack_threshold(image, window_size=window_size, mask=mask)
        elif method == 'sauvola':
            result = sauvola_threshold(image, window_size=window_size, mask=mask)
        elif method == 'contrast':
            result = local_contrast_threshold(image, window_size=window_size, mask=mask)
        elif method == 'phansalkar':
            result = phansalkar_threshold(image, window_size=window_size, mask=mask)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scale_results[f'scale_{window_size}'] = result
        combined_defects |= result['defects']
    
    total_pixels = np.sum(mask)
    defect_count = np.sum(combined_defects)
    defect_percentage = (defect_count / total_pixels * 100) if total_pixels > 0 else 0.0
    
    return {
        'scale_results': scale_results,
        'combined_defects': combined_defects,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'method': method,
        'window_sizes': window_sizes
    }


def adaptive_threshold_ensemble(image, mask=None, voting_threshold=0.5):
    """Ensemble of multiple adaptive thresholding methods with majority voting."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if mask is None:
        mask = image > 0
    
    methods = {
        'niblack': niblack_threshold(image, mask=mask),
        'sauvola': sauvola_threshold(image, mask=mask),
        'contrast': local_contrast_threshold(image, mask=mask),
        'phansalkar': phansalkar_threshold(image, mask=mask)
    }
    
    vote_count = np.zeros_like(mask, dtype=int)
    
    for method_name, result in methods.items():
        vote_count += result['defects'].astype(int)
    
    n_methods = len(methods)
    min_votes = int(voting_threshold * n_methods)
    ensemble_defects = vote_count >= min_votes
    
    total_pixels = np.sum(mask)
    defect_count = np.sum(ensemble_defects)
    defect_percentage = (defect_count / total_pixels * 100) if total_pixels > 0 else 0.0
    
    return {
        'methods': methods,
        'vote_count': vote_count,
        'ensemble_defects': ensemble_defects,
        'defect_count': defect_count,
        'defect_percentage': defect_percentage,
        'voting_threshold': voting_threshold
    }


def adaptive_threshold_parameters(image: np.ndarray, base_block_size: int = 11, 
                                base_c: float = 2.0) -> Tuple[int, float]:
    """Adaptively determine threshold parameters based on image characteristics."""
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    height, width = image.shape
    image_size_factor = np.sqrt(height * width) / 500.0
    
    optimized_block_size = int(base_block_size * image_size_factor)
    if optimized_block_size % 2 == 0:
        optimized_block_size += 1
    optimized_block_size = max(3, optimized_block_size)
    
    contrast_factor = std_intensity / 128.0
    optimized_c = base_c * (1 + contrast_factor)
    
    return optimized_block_size, optimized_c


# =============================================================================
# MORPHOLOGICAL OPERATIONS
# =============================================================================

def safe_thinning(binary_image: np.ndarray, method: str = "zhang_suen") -> np.ndarray:
    """Safe thinning implementation with multiple fallback methods."""
    if method == "opencv":
        try:
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                return cv2.ximgproc.thinning(binary_image)
            else:
                logger.debug("OpenCV ximgproc not available, using fallback")
                return morphological_skeleton(binary_image)
        except AttributeError:
            logger.debug("OpenCV contrib not available, using fallback skeleton")
            return morphological_skeleton(binary_image)
    elif method == "zhang_suen":
        return zhang_suen_thinning(binary_image)
    elif method == "morphological":
        return morphological_skeleton(binary_image)
    else:
        logger.warning(f"Unknown thinning method '{method}', using morphological")
        return morphological_skeleton(binary_image)


def morphological_skeleton(binary_image: np.ndarray) -> np.ndarray:
    """Morphological skeleton using iterative erosion and opening."""
    skeleton = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    binary_copy = binary_image.copy()
    iteration = 0
    max_iterations = 100
    
    while True:
        eroded = cv2.erode(binary_copy, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(binary_copy, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_copy = eroded.copy()
        
        iteration += 1
        if cv2.countNonZero(binary_copy) == 0 or iteration >= max_iterations:
            break
    
    return skeleton


def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning algorithm implementation."""
    img = (binary_image > 0).astype(np.uint8)
    h, w = img.shape
    
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = img
    
    iteration = 0
    max_iterations = 100
    
    while iteration < max_iterations:
        changed = False
        
        # Sub-iteration 1
        to_delete = []
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if padded[i, j] == 1:
                    p = [padded[i-1, j], padded[i-1, j+1], padded[i, j+1], 
                         padded[i+1, j+1], padded[i+1, j], padded[i+1, j-1],
                         padded[i, j-1], padded[i-1, j-1]]
                    
                    if zhang_suen_conditions(p, 1):
                        to_delete.append((i, j))
                        changed = True
        
        for i, j in to_delete:
            padded[i, j] = 0
        
        # Sub-iteration 2
        to_delete = []
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                if padded[i, j] == 1:
                    p = [padded[i-1, j], padded[i-1, j+1], padded[i, j+1], 
                         padded[i+1, j+1], padded[i+1, j], padded[i+1, j-1],
                         padded[i, j-1], padded[i-1, j-1]]
                    
                    if zhang_suen_conditions(p, 2):
                        to_delete.append((i, j))
                        changed = True
        
        for i, j in to_delete:
            padded[i, j] = 0
        
        if not changed:
            break
        
        iteration += 1
    
    result = padded[1:-1, 1:-1] * 255
    return result.astype(np.uint8)


def zhang_suen_conditions(neighbors: List[int], sub_iter: int) -> bool:
    """Check Zhang-Suen thinning conditions."""
    n = sum(neighbors)
    
    transitions = 0
    for i in range(8):
        if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
            transitions += 1
    
    if not (2 <= n <= 6):
        return False
    if transitions != 1:
        return False
    
    if sub_iter == 1:
        if neighbors[0] * neighbors[2] * neighbors[4] != 0:
            return False
        if neighbors[2] * neighbors[4] * neighbors[6] != 0:
            return False
    else:  # sub_iter == 2
        if neighbors[0] * neighbors[2] * neighbors[6] != 0:
            return False
        if neighbors[0] * neighbors[4] * neighbors[6] != 0:
            return False
    
    return True


def advanced_opening(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                    kernel_shape: str = "ellipse", iterations: int = 1) -> np.ndarray:
    """Advanced morphological opening with various kernel shapes."""
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def advanced_closing(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                    kernel_shape: str = "ellipse", iterations: int = 1) -> np.ndarray:
    """Advanced morphological closing with various kernel shapes."""
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    elif kernel_shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def top_hat_transform(image: np.ndarray, kernel_size: Tuple[int, int] = (15, 15),
                     transform_type: str = "white") -> np.ndarray:
    """Top-hat transformation for enhancing bright or dark features."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    if transform_type == "white":
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif transform_type == "black":
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    else:
        logger.warning(f"Unknown transform type '{transform_type}', using white")
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def morphological_gradient(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3),
                          gradient_type: str = "standard") -> np.ndarray:
    """Morphological gradient for edge detection."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    if gradient_type == "standard":
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    elif gradient_type == "external":
        dilated = cv2.dilate(image, kernel)
        return cv2.subtract(dilated, image)
    elif gradient_type == "internal":
        eroded = cv2.erode(image, kernel)
        return cv2.subtract(image, eroded)
    else:
        logger.warning(f"Unknown gradient type '{gradient_type}', using standard")
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def hit_or_miss_transform(image: np.ndarray, kernel_fg: np.ndarray,
                         kernel_bg: Optional[np.ndarray] = None) -> np.ndarray:
    """Hit-or-miss transformation for pattern matching."""
    if kernel_bg is None:
        kernel_bg = 1 - kernel_fg
    
    eroded_fg = cv2.erode(image, kernel_fg)
    complement = cv2.bitwise_not(image)
    eroded_bg = cv2.erode(complement, kernel_bg)
    result = cv2.bitwise_and(eroded_fg, eroded_bg)
    
    return result


def distance_transform_watershed(binary_image: np.ndarray, 
                                min_distance: int = 10) -> Tuple[np.ndarray, int]:
    """Watershed segmentation using distance transform."""
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    local_maxima = ndimage.maximum_filter(dist_transform, size=min_distance) == dist_transform
    local_maxima = local_maxima & (dist_transform > 0)
    
    markers, num_markers = ndimage.label(local_maxima)
    labels = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
    
    return labels, num_markers


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray,
                               method: str = "dilation") -> np.ndarray:
    """Morphological reconstruction by dilation or erosion."""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    if method == "dilation":
        reconstructed = marker.copy()
        while True:
            previous = reconstructed.copy()
            reconstructed = cv2.dilate(reconstructed, kernel)
            reconstructed = cv2.bitwise_and(reconstructed, mask)
            
            if np.array_equal(reconstructed, previous):
                break
    
    elif method == "erosion":
        reconstructed = marker.copy()
        while True:
            previous = reconstructed.copy()
            reconstructed = cv2.erode(reconstructed, kernel)
            reconstructed = cv2.bitwise_or(reconstructed, mask)
            
            if np.array_equal(reconstructed, previous):
                break
    
    else:
        logger.warning(f"Unknown method '{method}', using dilation")
        return morphological_reconstruction(marker, mask, "dilation")
    
    return reconstructed


def remove_small_objects(binary_image: np.ndarray, min_size: int = 50,
                        connectivity: int = 8) -> np.ndarray:
    """Remove small connected components from binary image."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    
    result = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            result[labels == i] = 255
    
    return result


def fill_holes(binary_image: np.ndarray, max_hole_size: Optional[int] = None) -> np.ndarray:
    """Fill holes in binary objects."""
    h, w = binary_image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    filled = binary_image.copy()
    cv2.floodFill(filled, mask, (0, 0), 255)
    
    filled_inv = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(binary_image, filled_inv)
    
    if max_hole_size is not None:
        holes = cv2.bitwise_and(filled_inv, cv2.bitwise_not(binary_image))
        filtered_holes = remove_small_objects(holes, max_hole_size + 1, 8)
        large_holes = cv2.bitwise_and(holes, cv2.bitwise_not(filtered_holes))
        result = cv2.bitwise_and(result, cv2.bitwise_not(large_holes))
    
    return result


def memory_efficient_morphology(image: np.ndarray, kernel: np.ndarray,
                               operation: str = 'open', iterations: int = 1) -> np.ndarray:
    """Perform morphological operations with memory optimization."""
    if operation == 'erode':
        result = cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'dilate':
        result = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == 'open':
        result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        result = image
    
    return result


# =============================================================================
# CALIBRATION
# =============================================================================

class CalibrationProcessor:
    """Handles calibration measurements and conversions."""
    
    def __init__(self):
        self.calibration_data = {}
        self.default_um_per_px = 0.5
    
    def detect_calibration_features(self, image: np.ndarray, 
                                   feature_type: str = "auto") -> List[Tuple[float, float]]:
        """Detect calibration features in the image."""
        logger.info(f"Detecting calibration features of type: {feature_type}")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        centroids = []
        
        if feature_type in ["dots", "auto"]:
            centroids = self._detect_dots(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} dot features")
                return centroids
        
        if feature_type in ["lines", "auto"]:
            centroids = self._detect_line_intersections(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} line intersection features")
                return centroids
        
        if feature_type in ["circles", "auto"]:
            centroids = self._detect_circles(gray)
            if len(centroids) > 0:
                logger.info(f"Found {len(centroids)} circle features")
                return centroids
        
        logger.warning("No calibration features detected")
        return []
    
    def _detect_dots(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect dots using SimpleBlobDetector."""
        try:
            params = cv2.SimpleBlobDetector.Params()
            
            params.minThreshold = 10
            params.maxThreshold = 200
            params.thresholdStep = 10
            
            params.filterByArea = True
            params.minArea = 20
            params.maxArea = 5000
            
            params.filterByCircularity = True
            params.minCircularity = 0.6
            
            params.filterByConvexity = True
            params.minConvexity = 0.8
            
            params.filterByInertia = True
            params.minInertiaRatio = 0.1
            
            detector = cv2.SimpleBlobDetector.create(params)
            keypoints = detector.detect(gray)
            
            centroids = [kp.pt for kp in keypoints]
            
        except Exception as e:
            logger.warning(f"SimpleBlobDetector failed: {e}, using fallback")
            centroids = []
        
        if not centroids:
            centroids = self._detect_circles(gray)
        
        return centroids
    
    def _detect_circles(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect circles using HoughCircles."""
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=60, param2=30, minRadius=5, maxRadius=50
        )
        
        centroids = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                centroids.append((float(x), float(y)))
        
        return centroids
    
    def _detect_line_intersections(self, gray: np.ndarray) -> List[Tuple[float, float]]:
        """Detect line intersections in grid patterns."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) < 4:
            return []
        
        line_equations = []
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            line_equations.append((a, b, rho))
        
        intersections = []
        for i in range(len(line_equations)):
            for j in range(i+1, len(line_equations)):
                intersection = self._line_intersection(line_equations[i], line_equations[j])
                if intersection is not None:
                    x, y = intersection
                    if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                        intersections.append((x, y))
        
        unique_intersections = []
        for point in intersections:
            is_unique = True
            for existing in unique_intersections:
                if np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2) < 10:
                    is_unique = False
                    break
            if is_unique:
                unique_intersections.append(point)
        
        return unique_intersections
    
    def _line_intersection(self, line1: Tuple[float, float, float], 
                          line2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """Calculate intersection of two lines."""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:  # Lines are parallel
            return None
        
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (x, y)
    
    def calculate_um_per_px(self, centroids: List[Tuple[float, float]], 
                           known_spacing_um: float, 
                           method: str = "nearest_neighbor") -> Optional[float]:
        """Calculate um_per_px from detected features."""
        if len(centroids) < 2:
            logger.error("Need at least 2 features for calibration")
            return None
        
        logger.info(f"Calculating um_per_px using {method} method")
        
        if method == "nearest_neighbor":
            return self._calculate_nearest_neighbor(centroids, known_spacing_um)
        elif method == "grid":
            return self._calculate_grid_spacing(centroids, known_spacing_um)
        elif method == "average":
            return self._calculate_average_spacing(centroids, known_spacing_um)
        else:
            logger.error(f"Unknown calculation method: {method}")
            return None
    
    def _calculate_nearest_neighbor(self, centroids: List[Tuple[float, float]], 
                                  known_spacing_um: float) -> float:
        """Calculate spacing using nearest neighbor distances."""
        distances = []
        
        for i, point1 in enumerate(centroids):
            min_dist = float('inf')
            for j, point2 in enumerate(centroids):
                if i != j:
                    dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                    if dist < min_dist:
                        min_dist = dist
            if min_dist != float('inf'):
                distances.append(min_dist)
        
        if not distances:
            return self.default_um_per_px
        
        avg_distance_px = np.mean(distances)
        um_per_px = known_spacing_um / avg_distance_px
        
        logger.info(f"Average nearest neighbor distance: {avg_distance_px:.2f} px")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def _calculate_grid_spacing(self, centroids: List[Tuple[float, float]], 
                               known_spacing_um: float) -> float:
        """Calculate spacing assuming regular grid pattern."""
        if len(centroids) < 4:
            return self._calculate_nearest_neighbor(centroids, known_spacing_um)
        
        centroids = sorted(centroids, key=lambda p: (p[1], p[0]))
        
        h_spacings = []
        v_spacings = []
        
        rows = []
        current_row = [centroids[0]]
        
        for point in centroids[1:]:
            if abs(point[1] - current_row[0][1]) < 20:
                current_row.append(point)
            else:
                rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [point]
        rows.append(sorted(current_row, key=lambda p: p[0]))
        
        for row in rows:
            for i in range(len(row) - 1):
                h_spacings.append(row[i+1][0] - row[i][0])
        
        for i in range(len(rows) - 1):
            if len(rows[i]) > 0 and len(rows[i+1]) > 0:
                v_spacings.append(rows[i+1][0][1] - rows[i][0][1])
        
        all_spacings = h_spacings + v_spacings
        if all_spacings:
            avg_spacing_px = np.median(all_spacings)
            um_per_px = known_spacing_um / avg_spacing_px
            
            logger.info(f"Grid spacing: {avg_spacing_px:.2f} px")
            logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
            
            return um_per_px
        
        return self._calculate_nearest_neighbor(centroids, known_spacing_um)
    
    def _calculate_average_spacing(self, centroids: List[Tuple[float, float]], 
                                  known_spacing_um: float) -> float:
        """Calculate average of all pairwise distances."""
        distances = []
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                              (centroids[i][1] - centroids[j][1])**2)
                distances.append(dist)
        
        if not distances:
            return self.default_um_per_px
        
        hist, bin_edges = np.histogram(distances, bins=20)
        mode_idx = np.argmax(hist)
        modal_distance = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        um_per_px = known_spacing_um / modal_distance
        
        logger.info(f"Modal distance: {modal_distance:.2f} px")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def calibrate_from_image(self, image_path: str, known_spacing_um: float, 
                            feature_type: str = "auto", 
                            method: str = "nearest_neighbor") -> Optional[float]:
        """Perform complete calibration from calibration image."""
        logger.info(f"Calibrating from image: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        centroids = self.detect_calibration_features(image, feature_type)
        
        if not centroids:
            logger.error("No calibration features detected")
            return None
        
        um_per_px = self.calculate_um_per_px(centroids, known_spacing_um, method)
        
        if um_per_px is not None:
            self.calibration_data = {
                'um_per_px': um_per_px,
                'known_spacing_um': known_spacing_um,
                'feature_count': len(centroids),
                'feature_type': feature_type,
                'method': method,
                'image_path': image_path,
                'centroids': centroids
            }
            
            logger.info(f"Calibration successful: {um_per_px:.4f} um/px")
        
        return um_per_px
    
    def calibrate_from_fiber_dimensions(self, cladding_diameter_px: float, 
                                       cladding_diameter_um: float = 125.0) -> float:
        """Calibrate using known fiber cladding diameter."""
        if cladding_diameter_px <= 0:
            logger.error("Invalid cladding diameter in pixels")
            return self.default_um_per_px
        
        um_per_px = cladding_diameter_um / cladding_diameter_px
        
        self.calibration_data = {
            'um_per_px': um_per_px,
            'cladding_diameter_px': cladding_diameter_px,
            'cladding_diameter_um': cladding_diameter_um,
            'method': 'fiber_cladding',
        }
        
        logger.info(f"Fiber calibration: {cladding_diameter_px:.2f} px = {cladding_diameter_um} um")
        logger.info(f"Calculated um_per_px: {um_per_px:.4f}")
        
        return um_per_px
    
    def save_calibration(self, filename: str = "calibration.json") -> bool:
        """Save calibration data to file."""
        try:
            save_data = self.calibration_data.copy()
            if 'centroids' in save_data:
                save_data['centroids'] = [list(c) for c in save_data['centroids']]
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Calibration data saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filename: str = "calibration.json") -> bool:
        """Load calibration data from file."""
        try:
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            
            logger.info(f"Calibration data loaded from {filename}")
            logger.info(f"um_per_px: {self.calibration_data.get('um_per_px', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def get_um_per_px(self) -> float:
        """Get the current um_per_px value."""
        return self.calibration_data.get('um_per_px', self.default_um_per_px)
    
    def convert_px_to_um(self, pixels: float) -> float:
        """Convert pixels to microns."""
        return pixels * self.get_um_per_px()
    
    def convert_um_to_px(self, microns: float) -> float:
        """Convert microns to pixels."""
        um_per_px = self.get_um_per_px()
        return microns / um_per_px if um_per_px > 0 else 0


# =============================================================================
# INTENSITY MAPPING AND ANALYSIS
# =============================================================================

def analyze_grayscale_image(image_path, output_dir='output'):
    """
    Analyzes an image by converting it to grayscale, generating a black and white
    pixel intensity map, a JSON file of grayscale pixel values, and a histogram
    of these intensities.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image from '{image_path}'. Check the file format.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Successfully loaded and converted image to grayscale: {image_path}")
    print(f"Grayscale image dimensions (Height, Width): {gray_image.shape}")

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Generate JSON file of all grayscale pixel values and coordinates
    height, width = gray_image.shape
    pixel_data = []
    for y in range(height):
        for x in range(width):
            pixel_value = int(gray_image.item(y, x))
            pixel_data.append({
                'coordinates': {'x': x, 'y': y},
                'intensity': pixel_value
            })

    json_path = os.path.join(output_dir, f'{base_filename}_grayscale_pixel_values.json')
    save_numpy_data(pixel_data, json_path)
    print(f"Successfully generated JSON file of grayscale pixel values at: {json_path}")

    # Generate a Black and White Pixel Intensity Map
    intensity_map_path = os.path.join(output_dir, f'{base_filename}_intensity_map.png')
    cv2.imwrite(intensity_map_path, gray_image)
    print(f"Successfully generated black and white pixel intensity map at: {intensity_map_path}")

    # Generate a Histogram of Grayscale Intensities
    plt.figure(figsize=(10, 6))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Histogram of Grayscale Pixel Intensities')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True)

    histogram_path = os.path.join(output_dir, f'{base_filename}_intensity_histogram.png')
    plt.savefig(histogram_path)
    plt.close()
    print(f"Successfully generated histogram of grayscale intensities at: {histogram_path}")
    print("\nGrayscale image analysis complete.")


# =============================================================================
# DATA CLUSTERING
# =============================================================================

def cluster_defects(features_df, config):
    """
    Clusters the detected defects based on their features.
    
    Args:
        features_df: A pandas DataFrame of defect features.
        config: The configuration dictionary.
        
    Returns:
        The input DataFrame with an added 'cluster' column.
    """
    if features_df.empty:
        return features_df
        
    features_for_clustering = [
        'area_px', 'aspect_ratio', 'solidity', 'mean_intensity',
        'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation'
    ]
    
    missing_cols = [col for col in features_for_clustering if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for clustering: {missing_cols}")

    features_df[features_for_clustering] = features_df[features_for_clustering].fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df[features_for_clustering])
    
    num_clusters = config.get('num_clusters', 4)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return features_df


# =============================================================================
# PYTORCH HELPER FUNCTIONS
# =============================================================================

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        logger.info(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y."""
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """Plots linear training data and test data and compares predictions."""
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def plot_loss_curves(results):
    """Plots training curves of a results dictionary."""
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: List[str] = None,
                       transform=None, device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """Makes a prediction on a target image with a trained model and plots the image."""
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image / 255.0

    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.imshow(target_image.squeeze().permute(1, 2, 0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination."""
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        logger.info(f"{image_path} directory exists, skipping download.")
    else:
        logger.info(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            logger.info(f"Downloading {target_file} from {source}...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            logger.info(f"Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def multi_scale_processing(image: np.ndarray, processing_func: Callable,
                          scales: List[float] = [0.5, 1.0, 2.0],
                          combine_method: str = 'max') -> np.ndarray:
    """Apply processing function at multiple scales and combine results."""
    original_shape = image.shape[:2]
    results = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            scaled_image = image
        
        processed = processing_func(scaled_image)
        
        if scale != 1.0:
            processed = cv2.resize(processed, (original_shape[1], original_shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        results.append(processed)
    
    if combine_method == 'max':
        combined = np.maximum.reduce(results)
    elif combine_method == 'mean':
        combined = np.mean(results, axis=0).astype(results[0].dtype)
    elif combine_method == 'weighted':
        weights = np.array(scales) / np.sum(scales)
        combined = np.zeros_like(results[0], dtype=np.float32)
        for result, weight in zip(results, weights):
            combined += result.astype(np.float32) * weight
        combined = combined.astype(results[0].dtype)
    else:
        combined = results[0]
    
    return combined


def batch_process_images(images: List[np.ndarray], processing_func: Callable,
                        batch_size: int = 4, show_progress: bool = True) -> List[Any]:
    """Process a batch of images efficiently."""
    results = []
    total_images = len(images)
    
    for i in range(0, total_images, batch_size):
        batch_end = min(i + batch_size, total_images)
        batch = images[i:batch_end]
        
        if show_progress:
            logger.info(f"Processing batch {i//batch_size + 1} "
                       f"({i+1}-{batch_end} of {total_images})")
        
        batch_results = []
        for image in batch:
            result = processing_func(image)
            batch_results.append(result)
        
        results.extend(batch_results)
        gc.collect()
    
    return results


def parallel_region_processing(image: np.ndarray, processing_func: Callable,
                              num_regions: int = 4, overlap: int = 50) -> np.ndarray:
    """Process image in parallel regions (simulated parallel processing)."""
    height, width = image.shape[:2]
    region_height = height // num_regions + overlap
    
    results = []
    for i in range(num_regions):
        start_y = max(0, i * (height // num_regions) - overlap // 2)
        end_y = min(height, start_y + region_height)
        
        region = image[start_y:end_y, :]
        processed_region = processing_func(region)
        results.append((start_y, end_y, processed_region))
    
    output = np.zeros_like(image)
    for start_y, end_y, processed_region in results:
        if start_y > 0 and overlap > 0:
            blend_start = overlap // 2
            for y in range(blend_start):
                alpha = y / blend_start
                blend_y = start_y + y
                if blend_y < height:
                    output[blend_y] = (alpha * processed_region[y] + 
                                     (1 - alpha) * output[blend_y])
            copy_start = start_y + blend_start
            copy_end = min(end_y, height)
            region_start = blend_start
            region_end = region_start + (copy_end - copy_start)
            output[copy_start:copy_end] = processed_region[region_start:region_end]
        else:
            output[start_y:end_y] = processed_region
    
    return output


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_adaptive_threshold_results(image, results, save_path=None):
    """Visualize adaptive thresholding results."""
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    if 'methods' in results:  # Ensemble results
        n_methods = len(results['methods'])
        fig, axes = plt.subplots(2, (n_methods + 2) // 2, figsize=(16, 8))
        fig.suptitle('Adaptive Thresholding Ensemble Results', fontsize=16)
        
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        for i, (method_name, method_result) in enumerate(results['methods'].items(), 1):
            if i < len(axes):
                axes[i].imshow(method_result['defects'], cmap='hot')
                count = np.sum(method_result['defects'])
                axes[i].set_title(f'{method_name.title()}\n{count} defects')
                axes[i].axis('off')
        
        if len(axes) > len(results['methods']) + 1:
            axes[-1].imshow(results['ensemble_defects'], cmap='hot')
            axes[-1].set_title(f'Ensemble\n{results["defect_count"]} defects')
            axes[-1].axis('off')
        
    elif 'scale_results' in results:  # Multi-scale results
        n_scales = len(results['scale_results'])
        fig, axes = plt.subplots(2, (n_scales + 2) // 2, figsize=(16, 8))
        fig.suptitle(f'Multi-scale {results["method"].title()} Thresholding', fontsize=16)
        
        axes = axes.flatten() if n_scales > 1 else [axes]
        
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        for i, (scale_name, scale_result) in enumerate(results['scale_results'].items(), 1):
            if i < len(axes):
                axes[i].imshow(scale_result['defects'], cmap='hot')
                count = np.sum(scale_result['defects'])
                window_size = scale_result['window_size']
                axes[i].set_title(f'Window {window_size}\n{count} defects')
                axes[i].axis('off')
        
        if len(axes) > len(results['scale_results']) + 1:
            axes[-1].imshow(results['combined_defects'], cmap='hot')
            axes[-1].set_title(f'Combined\n{results["defect_count"]} defects')
            axes[-1].axis('off')
    
    else:  # Single method result
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('Adaptive Thresholding Results', fontsize=16)
        
        axes[0].imshow(display_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        if 'threshold_map' in results:
            axes[1].imshow(results['threshold_map'], cmap='viridis')
            axes[1].set_title('Threshold Map')
            axes[1].axis('off')
        elif 'contrast_map' in results:
            axes[1].imshow(results['contrast_map'], cmap='viridis')
            axes[1].set_title('Contrast Map')
            axes[1].axis('off')
        
        axes[2].imshow(results['defects'], cmap='hot')
        count = np.sum(results['defects'])
        axes[2].set_title(f'Defects\n{count} pixels')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN TESTING FUNCTION
# =============================================================================

def test_unified_module():
    """Test the unified helper module functionality."""
    logger.info("Testing Unified Helper Module...")
    
    # Test image creation
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    # Test performance timer
    @performance_timer
    def test_func():
        return cv2.GaussianBlur(test_gray, (15, 15), 0)
    
    blurred = test_func()
    logger.info(f"Performance timer test completed")
    
    # Test adaptive thresholding
    sauvola_result = sauvola_threshold(test_gray)
    logger.info(f"Sauvola thresholding found {np.sum(sauvola_result['defects'])} defect pixels")
    
    # Test morphological operations
    binary_test = (test_gray > 128).astype(np.uint8) * 255
    skeleton = safe_thinning(binary_test, "morphological")
    logger.info(f"Morphological skeleton: {np.sum(skeleton > 0)} pixels")
    
    # Test calibration
    calibrator = CalibrationProcessor()
    um_per_px = calibrator.calibrate_from_fiber_dimensions(250, 125.0)
    logger.info(f"Calibration test: {um_per_px:.4f} um/px")
    
    # Test JSON encoding
    test_data = {'array': np.array([1, 2, 3]), 'float': np.float32(3.14)}
    json_str = dumps_numpy(test_data)
    logger.info(f"JSON encoding test completed")
    
    logger.info("All tests completed successfully!")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Helper Module for Fiber Optic Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test all functionality:
    python unified_helper_module.py --test
    
  Analyze grayscale image:
    python unified_helper_module.py --analyze-image path/to/image.jpg
    
  Calibrate from image:
    python unified_helper_module.py --calibrate path/to/calibration.jpg --spacing 50.0
"""
    )
    
    parser.add_argument('--test', action='store_true', 
                       help='Run module tests')
    parser.add_argument('--analyze-image', type=str,
                       help='Analyze grayscale properties of an image')
    parser.add_argument('--calibrate', type=str,
                       help='Calibrate from calibration image')
    parser.add_argument('--spacing', type=float, default=50.0,
                       help='Known spacing for calibration (microns)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.test:
        test_unified_module()
    elif args.analyze_image:
        analyze_grayscale_image(args.analyze_image, args.output)
    elif args.calibrate:
        calibrator = CalibrationProcessor()
        um_per_px = calibrator.calibrate_from_image(
            args.calibrate, args.spacing, "auto", "nearest_neighbor"
        )
        if um_per_px:
            calibrator.save_calibration(os.path.join(args.output, "calibration.json"))
    else:
        parser.print_help()