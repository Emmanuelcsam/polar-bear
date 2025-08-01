#!/usr/bin/env python3
"""
GPU Utilities for Fiber Optic Analysis Pipeline
Provides GPU acceleration with automatic CPU fallback and comprehensive logging
"""

import os
import sys
import time
import logging
import functools
from typing import Any, Callable, Optional, Union, Tuple, Dict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Try to import GPU libraries
GPU_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import ndimage as gpu_ndimage
    GPU_AVAILABLE = True
    logging.info("CuPy successfully imported - GPU acceleration available")
except ImportError:
    cp = None
    logging.warning("CuPy not available - will use CPU fallback")

try:
    # Check if OpenCV was compiled with CUDA support
    import cv2
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
        logging.info(f"OpenCV CUDA available - {cv2.cuda.getCudaEnabledDeviceCount()} devices found")
    else:
        logging.warning("OpenCV CUDA not available - will use CPU OpenCV")
except:
    logging.warning("Could not check OpenCV CUDA availability")

# Try to import RAPIDS cuML for clustering
RAPIDS_AVAILABLE = False
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    RAPIDS_AVAILABLE = True
    logging.info("RAPIDS cuML available for GPU clustering")
except ImportError:
    logging.warning("RAPIDS cuML not available - will use CPU clustering")


class GPUManager:
    """Manages GPU resources and provides automatic CPU fallback"""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize GPU manager
        
        Args:
            force_cpu: Force CPU mode even if GPU is available (for testing)
        """
        self.use_gpu = GPU_AVAILABLE and not force_cpu
        self.use_cuda = CUDA_AVAILABLE and not force_cpu
        self.use_rapids = RAPIDS_AVAILABLE and not force_cpu
        self.logger = logging.getLogger('GPUManager')
        
        if self.use_gpu:
            self.device = cp.cuda.Device(0)
            self.logger.info(f"Using GPU: {self.device.name}")
            self.logger.info(f"GPU Memory: {self.device.mem_info[1] / 1e9:.2f} GB")
        else:
            self.logger.info("Running in CPU mode")
    
    def array_to_gpu(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer numpy array to GPU if available"""
        if self.use_gpu and cp is not None:
            start_time = time.time()
            gpu_array = cp.asarray(array)
            transfer_time = time.time() - start_time
            self.logger.debug(f"Transferred array shape {array.shape} to GPU in {transfer_time:.3f}s")
            return gpu_array
        return array
    
    def array_to_cpu(self, array: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
        """Transfer GPU array back to CPU"""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            start_time = time.time()
            cpu_array = cp.asnumpy(array)
            transfer_time = time.time() - start_time
            self.logger.debug(f"Transferred array shape {array.shape} to CPU in {transfer_time:.3f}s")
            return cpu_array
        return array
    
    def get_array_module(self, array: Union[np.ndarray, 'cp.ndarray']):
        """Get appropriate module (numpy or cupy) for array"""
        if self.use_gpu and cp is not None and isinstance(array, cp.ndarray):
            return cp
        return np
    
    def synchronize(self):
        """Synchronize GPU operations"""
        if self.use_gpu and cp is not None:
            cp.cuda.Stream.null.synchronize()


def gpu_accelerated(func: Callable) -> Callable:
    """
    Decorator to automatically handle GPU acceleration with CPU fallback
    
    Usage:
        @gpu_accelerated
        def process_image(self, image, *args, **kwargs):
            # Function will receive GPU array if available
            # Return GPU array, will be converted back to CPU automatically
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        # Get GPU manager from self or create one
        if hasattr(self, 'gpu_manager'):
            gpu_manager = self.gpu_manager
        else:
            gpu_manager = GPUManager()
        
        # Log function entry
        logger = logging.getLogger(f"{self.__class__.__name__}.{func.__name__}")
        logger.debug(f"Starting {func.__name__} with GPU={gpu_manager.use_gpu}")
        
        try:
            # Convert numpy arrays in args to GPU arrays
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    gpu_args.append(gpu_manager.array_to_gpu(arg))
                else:
                    gpu_args.append(arg)
            
            # Convert numpy arrays in kwargs to GPU arrays
            gpu_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    gpu_kwargs[key] = gpu_manager.array_to_gpu(value)
                else:
                    gpu_kwargs[key] = value
            
            # Call the function with GPU arrays
            result = func(self, *gpu_args, **gpu_kwargs)
            
            # Convert result back to CPU if needed
            if isinstance(result, (cp.ndarray if cp else type(None), np.ndarray)):
                result = gpu_manager.array_to_cpu(result)
            elif isinstance(result, tuple):
                result = tuple(
                    gpu_manager.array_to_cpu(r) if isinstance(r, (cp.ndarray if cp else type(None), np.ndarray)) else r
                    for r in result
                )
            elif isinstance(result, dict):
                result = {
                    k: gpu_manager.array_to_cpu(v) if isinstance(v, (cp.ndarray if cp else type(None), np.ndarray)) else v
                    for k, v in result.items()
                }
            
            # Synchronize GPU
            gpu_manager.synchronize()
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.warning(f"Falling back to CPU for {func.__name__}")
            
            # Fallback to CPU
            cpu_args = []
            for arg in args:
                if hasattr(arg, '__array__'):
                    cpu_args.append(np.asarray(arg))
                else:
                    cpu_args.append(arg)
            
            cpu_kwargs = {}
            for key, value in kwargs.items():
                if hasattr(value, '__array__'):
                    cpu_kwargs[key] = np.asarray(value)
                else:
                    cpu_kwargs[key] = value
            
            # Temporarily disable GPU
            original_use_gpu = gpu_manager.use_gpu
            gpu_manager.use_gpu = False
            
            try:
                result = func(self, *cpu_args, **cpu_kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Completed {func.__name__} in CPU mode in {elapsed_time:.3f}s")
                return result
            finally:
                gpu_manager.use_gpu = original_use_gpu
    
    return wrapper


class GPUImageProcessor:
    """GPU-accelerated image processing operations"""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger('GPUImageProcessor')
        self.xp = cp if self.gpu_manager.use_gpu else np
    
    def gaussian_blur(self, image: Union[np.ndarray, 'cp.ndarray'], kernel_size: Tuple[int, int], 
                     sigma: float = 0) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated Gaussian blur"""
        if self.gpu_manager.use_cuda and CUDA_AVAILABLE:
            # Use OpenCV CUDA
            if isinstance(image, cp.ndarray):
                image = cp.asnumpy(image)
            
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            if sigma == 0:
                sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8
            
            gpu_result = cv2.cuda.bilateralFilter(gpu_image, -1, sigma, sigma)
            result = gpu_result.download()
            
            if self.gpu_manager.use_gpu:
                result = cp.asarray(result)
            
            return result
        else:
            # Use CuPy or NumPy
            xp = self.gpu_manager.get_array_module(image)
            if xp == cp:
                return cupyx.scipy.ndimage.gaussian_filter(image, sigma)
            else:
                from scipy import ndimage
                return ndimage.gaussian_filter(image, sigma)
    
    def morphological_operation(self, image: Union[np.ndarray, 'cp.ndarray'], 
                               operation: str, kernel: Union[np.ndarray, 'cp.ndarray'],
                               iterations: int = 1) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated morphological operations"""
        xp = self.gpu_manager.get_array_module(image)
        
        if self.gpu_manager.use_cuda and CUDA_AVAILABLE and isinstance(image, np.ndarray):
            # Use OpenCV CUDA
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            if isinstance(kernel, cp.ndarray):
                kernel = cp.asnumpy(kernel)
            
            if operation == 'erode':
                gpu_result = cv2.cuda.erode(gpu_image, kernel, iterations=iterations)
            elif operation == 'dilate':
                gpu_result = cv2.cuda.dilate(gpu_image, kernel, iterations=iterations)
            elif operation == 'open':
                gpu_result = cv2.cuda.morphologyEx(gpu_image, cv2.MORPH_OPEN, kernel)
            elif operation == 'close':
                gpu_result = cv2.cuda.morphologyEx(gpu_image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                gpu_result = cv2.cuda.morphologyEx(gpu_image, cv2.MORPH_GRADIENT, kernel)
            elif operation == 'tophat':
                gpu_result = cv2.cuda.morphologyEx(gpu_image, cv2.MORPH_TOPHAT, kernel)
            elif operation == 'blackhat':
                gpu_result = cv2.cuda.morphologyEx(gpu_image, cv2.MORPH_BLACKHAT, kernel)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = gpu_result.download()
            if self.gpu_manager.use_gpu:
                result = cp.asarray(result)
            return result
        else:
            # Use CuPy or SciPy
            if xp == cp:
                if operation == 'erode':
                    return cupyx.scipy.ndimage.grey_erosion(image, footprint=kernel)
                elif operation == 'dilate':
                    return cupyx.scipy.ndimage.grey_dilation(image, footprint=kernel)
                elif operation == 'open':
                    temp = cupyx.scipy.ndimage.grey_erosion(image, footprint=kernel)
                    return cupyx.scipy.ndimage.grey_dilation(temp, footprint=kernel)
                elif operation == 'close':
                    temp = cupyx.scipy.ndimage.grey_dilation(image, footprint=kernel)
                    return cupyx.scipy.ndimage.grey_erosion(temp, footprint=kernel)
                elif operation == 'gradient':
                    dilated = cupyx.scipy.ndimage.grey_dilation(image, footprint=kernel)
                    eroded = cupyx.scipy.ndimage.grey_erosion(image, footprint=kernel)
                    return dilated - eroded
                elif operation == 'tophat':
                    opened = cupyx.scipy.ndimage.grey_opening(image, footprint=kernel)
                    return image - opened
                elif operation == 'blackhat':
                    closed = cupyx.scipy.ndimage.grey_closing(image, footprint=kernel)
                    return closed - image
            else:
                # CPU fallback using OpenCV
                if operation == 'erode':
                    return cv2.erode(image, kernel, iterations=iterations)
                elif operation == 'dilate':
                    return cv2.dilate(image, kernel, iterations=iterations)
                elif operation == 'open':
                    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
                elif operation == 'close':
                    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                elif operation == 'gradient':
                    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
                elif operation == 'tophat':
                    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
                elif operation == 'blackhat':
                    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
                
            raise ValueError(f"Unknown operation: {operation}")
    
    def threshold(self, image: Union[np.ndarray, 'cp.ndarray'], 
                 threshold_value: float, max_value: float,
                 threshold_type: str = 'binary') -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated thresholding"""
        xp = self.gpu_manager.get_array_module(image)
        
        if threshold_type == 'binary':
            return xp.where(image > threshold_value, max_value, 0).astype(image.dtype)
        elif threshold_type == 'binary_inv':
            return xp.where(image > threshold_value, 0, max_value).astype(image.dtype)
        elif threshold_type == 'trunc':
            return xp.minimum(image, threshold_value).astype(image.dtype)
        elif threshold_type == 'tozero':
            return xp.where(image > threshold_value, image, 0).astype(image.dtype)
        elif threshold_type == 'tozero_inv':
            return xp.where(image > threshold_value, 0, image).astype(image.dtype)
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
    
    def adaptive_threshold(self, image: Union[np.ndarray, 'cp.ndarray'],
                          max_value: float, block_size: int,
                          C: float, method: str = 'mean') -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated adaptive thresholding"""
        xp = self.gpu_manager.get_array_module(image)
        
        # Calculate local threshold
        if method == 'mean':
            if xp == cp:
                local_mean = cupyx.scipy.ndimage.uniform_filter(image.astype(xp.float32), block_size)
            else:
                from scipy import ndimage
                local_mean = ndimage.uniform_filter(image.astype(np.float32), block_size)
            threshold = local_mean - C
        elif method == 'gaussian':
            if xp == cp:
                local_mean = cupyx.scipy.ndimage.gaussian_filter(image.astype(xp.float32), block_size/6)
            else:
                from scipy import ndimage
                local_mean = ndimage.gaussian_filter(image.astype(np.float32), block_size/6)
            threshold = local_mean - C
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return xp.where(image > threshold, max_value, 0).astype(image.dtype)


def log_gpu_memory():
    """Log current GPU memory usage"""
    if GPU_AVAILABLE and cp is not None:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        logging.info(f"GPU Memory - Used: {mempool.used_bytes() / 1e9:.2f} GB, "
                    f"Total: {mempool.total_bytes() / 1e9:.2f} GB")
        logging.info(f"Pinned Memory - Used: {pinned_mempool.used_bytes() / 1e9:.2f} GB, "
                    f"Total: {pinned_mempool.total_bytes() / 1e9:.2f} GB")


def clear_gpu_memory():
    """Clear GPU memory pools"""
    if GPU_AVAILABLE and cp is not None:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        logging.info("Cleared GPU memory pools")