#!/usr/bin/env python3
"""
GPU-Accelerated Image Processing Module
Performs various image preprocessing operations using GPU acceleration when available
"""

import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from gpu_utils import GPUManager, gpu_accelerated, GPUImageProcessor, log_gpu_memory

# Configure logging
logger = logging.getLogger('ProcessGPU')


class ImageProcessorGPU:
    """GPU-accelerated image processor for fiber optic analysis"""
    
    def __init__(self, config: Dict, force_cpu: bool = False):
        """
        Initialize GPU-accelerated image processor
        
        Args:
            config: Configuration dictionary
            force_cpu: Force CPU mode for testing
        """
        self.config = config
        self.gpu_manager = GPUManager(force_cpu=force_cpu)
        self.gpu_processor = GPUImageProcessor(self.gpu_manager)
        self.logger = logging.getLogger('ImageProcessorGPU')
        self.xp = self.gpu_manager.array_to_gpu(np.array([0])).xp if self.gpu_manager.use_gpu else np
        
        self.logger.info(f"Initialized ImageProcessorGPU with GPU={self.gpu_manager.use_gpu}")
    
    @gpu_accelerated
    def process_single_image(self, image_path: str, output_folder: str, 
                           return_arrays: bool = False) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single image with various transformations
        
        Args:
            image_path: Path to input image
            output_folder: Path to output folder (unused if return_arrays=True)
            return_arrays: If True, return dictionary of processed arrays instead of saving
        
        Returns:
            Dictionary of processed arrays if return_arrays=True, else None
        """
        start_time = time.time()
        self.logger.info(f"Processing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Failed to read image: {image_path}")
            return None
        
        # Transfer to GPU
        img_gpu = self.gpu_manager.array_to_gpu(img)
        
        # Dictionary to store results
        results = {} if return_arrays else None
        
        # Process all variations
        processed_images = self._generate_all_variations(img_gpu, Path(image_path).stem)
        
        # Save or return results
        if return_arrays:
            # Convert GPU arrays back to CPU
            for name, img_array in processed_images.items():
                results[name] = self.gpu_manager.array_to_cpu(img_array)
        else:
            # Save to disk
            os.makedirs(output_folder, exist_ok=True)
            for name, img_array in processed_images.items():
                output_path = os.path.join(output_folder, f"{name}.png")
                cpu_array = self.gpu_manager.array_to_cpu(img_array)
                cv2.imwrite(output_path, cpu_array)
                self.logger.debug(f"Saved: {output_path}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Processed image in {elapsed_time:.2f}s with {len(processed_images)} variations")
        log_gpu_memory()
        
        return results
    
    def _generate_all_variations(self, img: Union[np.ndarray, 'cp.ndarray'], 
                                base_name: str) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Generate all image variations using GPU acceleration"""
        xp = self.gpu_manager.get_array_module(img)
        results = {}
        
        # Convert to different color spaces
        if len(img.shape) == 3:
            # Original
            results[f"{base_name}_original"] = img.copy()
            
            # Grayscale
            gray = self._rgb_to_grayscale_gpu(img)
            results[f"{base_name}_grayscale"] = gray
            
            # Color channels
            results[f"{base_name}_blue_channel"] = img[:, :, 0]
            results[f"{base_name}_green_channel"] = img[:, :, 1]
            results[f"{base_name}_red_channel"] = img[:, :, 2]
            
            # HSV processing
            hsv = self._rgb_to_hsv_gpu(img)
            results[f"{base_name}_hue"] = hsv[:, :, 0]
            results[f"{base_name}_saturation"] = hsv[:, :, 1]
            results[f"{base_name}_value"] = hsv[:, :, 2]
            
            # LAB processing
            lab = self._rgb_to_lab_gpu(img)
            results[f"{base_name}_l_channel"] = lab[:, :, 0]
            results[f"{base_name}_a_channel"] = lab[:, :, 1]
            results[f"{base_name}_b_channel"] = lab[:, :, 2]
        else:
            gray = img
            results[f"{base_name}_grayscale"] = gray
        
        # Apply various filters and transformations to grayscale
        self._apply_filters(gray, results, base_name)
        self._apply_thresholds(gray, results, base_name)
        self._apply_morphological_operations(gray, results, base_name)
        self._apply_edge_detection(gray, results, base_name)
        self._apply_frequency_domain_filters(gray, results, base_name)
        
        return results
    
    @gpu_accelerated
    def _rgb_to_grayscale_gpu(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Convert RGB to grayscale using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        # Use standard RGB to grayscale conversion weights
        return xp.dot(img[..., :3], xp.array([0.299, 0.587, 0.114])).astype(xp.uint8)
    
    @gpu_accelerated
    def _rgb_to_hsv_gpu(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Convert RGB to HSV using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        # Normalize to [0, 1]
        img_float = img.astype(xp.float32) / 255.0
        
        # Get RGB channels
        r, g, b = img_float[:, :, 2], img_float[:, :, 1], img_float[:, :, 0]
        
        # Calculate Value (max of RGB)
        v = xp.maximum(xp.maximum(r, g), b)
        
        # Calculate Saturation
        min_rgb = xp.minimum(xp.minimum(r, g), b)
        delta = v - min_rgb
        s = xp.where(v != 0, delta / v, 0)
        
        # Calculate Hue
        h = xp.zeros_like(v)
        
        # When max is red
        mask = (v == r) & (delta != 0)
        h = xp.where(mask, (g - b) / delta, h)
        
        # When max is green
        mask = (v == g) & (delta != 0)
        h = xp.where(mask, 2 + (b - r) / delta, h)
        
        # When max is blue
        mask = (v == b) & (delta != 0)
        h = xp.where(mask, 4 + (r - g) / delta, h)
        
        h = h * 60  # Convert to degrees
        h = xp.where(h < 0, h + 360, h)
        
        # Scale to uint8
        hsv = xp.stack([
            (h / 2).astype(xp.uint8),  # Hue [0-180]
            (s * 255).astype(xp.uint8),  # Saturation [0-255]
            (v * 255).astype(xp.uint8)   # Value [0-255]
        ], axis=2)
        
        return hsv
    
    @gpu_accelerated
    def _rgb_to_lab_gpu(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Convert RGB to LAB using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        # Convert to float and normalize
        img_float = img.astype(xp.float32) / 255.0
        
        # RGB to XYZ conversion matrix
        rgb_to_xyz = xp.array([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ])
        
        # Apply matrix transformation
        xyz = xp.dot(img_float, rgb_to_xyz.T)
        
        # Normalize by white point (D65)
        xyz[:, :, 0] /= 0.95047
        xyz[:, :, 1] /= 1.00000
        xyz[:, :, 2] /= 1.08883
        
        # Apply f(x) transformation
        def f(t):
            delta = 6/29
            return xp.where(t > delta**3, xp.cbrt(t), t / (3 * delta**2) + 4/29)
        
        fxyz = f(xyz)
        
        # Calculate LAB values
        l = 116 * fxyz[:, :, 1] - 16
        a = 500 * (fxyz[:, :, 0] - fxyz[:, :, 1])
        b = 200 * (fxyz[:, :, 1] - fxyz[:, :, 2])
        
        # Scale to uint8
        lab = xp.stack([
            xp.clip(l * 2.55, 0, 255).astype(xp.uint8),
            xp.clip(a + 128, 0, 255).astype(xp.uint8),
            xp.clip(b + 128, 0, 255).astype(xp.uint8)
        ], axis=2)
        
        return lab
    
    def _apply_filters(self, gray: Union[np.ndarray, 'cp.ndarray'], 
                      results: Dict, base_name: str):
        """Apply various filters using GPU acceleration"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # Gaussian blur with different kernel sizes
        for kernel_size in [3, 5, 7, 9]:
            blurred = self.gpu_processor.gaussian_blur(gray, (kernel_size, kernel_size))
            results[f"{base_name}_gaussian_{kernel_size}"] = blurred
        
        # Median filter (approximate using repeated small median filters for GPU)
        if xp == self.xp and self.gpu_manager.use_gpu:
            # GPU median approximation
            median = gray.copy()
            for _ in range(3):
                median = self._median_filter_gpu(median, 3)
            results[f"{base_name}_median"] = median
        else:
            # CPU median filter
            results[f"{base_name}_median"] = cv2.medianBlur(self.gpu_manager.array_to_cpu(gray), 5)
        
        # Bilateral filter
        if self.gpu_manager.use_cuda:
            bilateral = self._bilateral_filter_cuda(gray)
        else:
            bilateral = self._bilateral_filter_gpu(gray)
        results[f"{base_name}_bilateral"] = bilateral
    
    @gpu_accelerated
    def _median_filter_gpu(self, img: Union[np.ndarray, 'cp.ndarray'], 
                          kernel_size: int) -> Union[np.ndarray, 'cp.ndarray']:
        """Approximate median filter using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        # Simple approximation using erosion and dilation
        kernel = xp.ones((kernel_size, kernel_size), dtype=xp.uint8)
        eroded = self.gpu_processor.morphological_operation(img, 'erode', kernel)
        dilated = self.gpu_processor.morphological_operation(img, 'dilate', kernel)
        
        # Average of erosion and dilation approximates median
        return ((eroded.astype(xp.float32) + dilated.astype(xp.float32)) / 2).astype(xp.uint8)
    
    @gpu_accelerated
    def _bilateral_filter_gpu(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Bilateral filter using GPU"""
        # For simplicity, use Gaussian blur as approximation
        # In production, implement full bilateral filter
        return self.gpu_processor.gaussian_blur(img, (9, 9), sigma=75)
    
    def _bilateral_filter_cuda(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Bilateral filter using OpenCV CUDA"""
        cpu_img = self.gpu_manager.array_to_cpu(img)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(cpu_img)
        
        gpu_result = cv2.cuda.bilateralFilter(gpu_img, -1, 50, 50)
        result = gpu_result.download()
        
        if self.gpu_manager.use_gpu:
            result = self.gpu_manager.array_to_gpu(result)
        
        return result
    
    def _apply_thresholds(self, gray: Union[np.ndarray, 'cp.ndarray'], 
                         results: Dict, base_name: str):
        """Apply various thresholding techniques"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # Global thresholds
        for thresh_val in [50, 100, 127, 150, 200]:
            binary = self.gpu_processor.threshold(gray, thresh_val, 255, 'binary')
            results[f"{base_name}_threshold_{thresh_val}"] = binary
        
        # Otsu's threshold
        if xp == np:
            # CPU Otsu
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results[f"{base_name}_otsu"] = otsu
        else:
            # GPU Otsu approximation
            hist = xp.histogram(gray.ravel(), bins=256, range=(0, 256))[0]
            cumsum = xp.cumsum(hist)
            cumsum_sq = xp.cumsum(hist * xp.arange(256))
            
            mean = cumsum_sq[-1] / cumsum[-1]
            
            # Calculate between-class variance
            w0 = cumsum[:-1]
            w1 = cumsum[-1] - w0
            
            mu0 = cumsum_sq[:-1] / (w0 + 1e-10)
            mu1 = (cumsum_sq[-1] - cumsum_sq[:-1]) / (w1 + 1e-10)
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            thresh = xp.argmax(variance)
            
            otsu = self.gpu_processor.threshold(gray, float(thresh), 255, 'binary')
            results[f"{base_name}_otsu"] = otsu
        
        # Adaptive thresholds
        for block_size in [11, 21, 31]:
            adaptive_mean = self.gpu_processor.adaptive_threshold(
                gray, 255, block_size, 2, 'mean'
            )
            results[f"{base_name}_adaptive_mean_{block_size}"] = adaptive_mean
            
            adaptive_gaussian = self.gpu_processor.adaptive_threshold(
                gray, 255, block_size, 2, 'gaussian'
            )
            results[f"{base_name}_adaptive_gaussian_{block_size}"] = adaptive_gaussian
    
    def _apply_morphological_operations(self, gray: Union[np.ndarray, 'cp.ndarray'], 
                                      results: Dict, base_name: str):
        """Apply morphological operations"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # Create kernels
        kernel3 = xp.ones((3, 3), dtype=xp.uint8)
        kernel5 = xp.ones((5, 5), dtype=xp.uint8)
        
        # Basic operations
        for kernel_size, kernel in [(3, kernel3), (5, kernel5)]:
            # Erosion and dilation
            eroded = self.gpu_processor.morphological_operation(gray, 'erode', kernel)
            dilated = self.gpu_processor.morphological_operation(gray, 'dilate', kernel)
            
            results[f"{base_name}_eroded_{kernel_size}"] = eroded
            results[f"{base_name}_dilated_{kernel_size}"] = dilated
            
            # Opening and closing
            opened = self.gpu_processor.morphological_operation(gray, 'open', kernel)
            closed = self.gpu_processor.morphological_operation(gray, 'close', kernel)
            
            results[f"{base_name}_opened_{kernel_size}"] = opened
            results[f"{base_name}_closed_{kernel_size}"] = closed
            
            # Gradient, tophat, blackhat
            gradient = self.gpu_processor.morphological_operation(gray, 'gradient', kernel)
            tophat = self.gpu_processor.morphological_operation(gray, 'tophat', kernel)
            blackhat = self.gpu_processor.morphological_operation(gray, 'blackhat', kernel)
            
            results[f"{base_name}_gradient_{kernel_size}"] = gradient
            results[f"{base_name}_tophat_{kernel_size}"] = tophat
            results[f"{base_name}_blackhat_{kernel_size}"] = blackhat
    
    @gpu_accelerated
    def _apply_edge_detection(self, gray: Union[np.ndarray, 'cp.ndarray'], 
                            results: Dict, base_name: str):
        """Apply edge detection algorithms"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # Sobel edge detection
        if xp == np:
            # CPU Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # GPU Sobel
            sobelx = self._sobel_gpu(gray, axis=0)
            sobely = self._sobel_gpu(gray, axis=1)
        
        sobel_magnitude = xp.sqrt(sobelx**2 + sobely**2)
        results[f"{base_name}_sobel"] = xp.clip(sobel_magnitude, 0, 255).astype(xp.uint8)
        
        # Canny edge detection
        if xp == np:
            # CPU Canny
            for low, high in [(50, 150), (100, 200)]:
                canny = cv2.Canny(gray, low, high)
                results[f"{base_name}_canny_{low}_{high}"] = canny
        else:
            # GPU Canny approximation using gradients
            for low, high in [(50, 150), (100, 200)]:
                # Simple Canny approximation
                edges = xp.where(
                    (sobel_magnitude > low) & (sobel_magnitude < high),
                    255,
                    0
                ).astype(xp.uint8)
                results[f"{base_name}_canny_{low}_{high}"] = edges
        
        # Laplacian
        if xp == np:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        else:
            laplacian = self._laplacian_gpu(gray)
        
        results[f"{base_name}_laplacian"] = xp.clip(xp.abs(laplacian), 0, 255).astype(xp.uint8)
    
    @gpu_accelerated
    def _sobel_gpu(self, img: Union[np.ndarray, 'cp.ndarray'], 
                   axis: int) -> Union[np.ndarray, 'cp.ndarray']:
        """Sobel filter using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        if axis == 0:  # X direction
            kernel = xp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=xp.float32)
        else:  # Y direction
            kernel = xp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=xp.float32)
        
        # Convolve using FFT for large images
        return self._convolve_gpu(img.astype(xp.float32), kernel)
    
    @gpu_accelerated
    def _laplacian_gpu(self, img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Laplacian filter using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        kernel = xp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=xp.float32)
        
        return self._convolve_gpu(img.astype(xp.float32), kernel)
    
    @gpu_accelerated
    def _convolve_gpu(self, img: Union[np.ndarray, 'cp.ndarray'], 
                     kernel: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """2D convolution using GPU"""
        xp = self.gpu_manager.get_array_module(img)
        
        if xp == self.xp and self.gpu_manager.use_gpu:
            # Use CuPy's convolve
            from cupyx.scipy import ndimage
            return ndimage.convolve(img, kernel, mode='constant')
        else:
            # Use OpenCV for CPU
            return cv2.filter2D(self.gpu_manager.array_to_cpu(img), -1, 
                              self.gpu_manager.array_to_cpu(kernel))
    
    @gpu_accelerated
    def _apply_frequency_domain_filters(self, gray: Union[np.ndarray, 'cp.ndarray'], 
                                      results: Dict, base_name: str):
        """Apply frequency domain filters using GPU"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # FFT
        f_transform = xp.fft.fft2(gray.astype(xp.float32))
        f_shift = xp.fft.fftshift(f_transform)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # High-pass filter
        mask_hp = xp.ones((rows, cols), dtype=xp.float32)
        r = 30
        y, x = xp.ogrid[:rows, :cols]
        mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
        mask_hp[mask_area] = 0
        
        f_hp = f_shift * mask_hp
        img_hp = xp.fft.ifft2(xp.fft.ifftshift(f_hp))
        img_hp = xp.abs(img_hp)
        
        results[f"{base_name}_highpass"] = xp.clip(img_hp, 0, 255).astype(xp.uint8)
        
        # Low-pass filter
        mask_lp = xp.zeros((rows, cols), dtype=xp.float32)
        mask_lp[mask_area] = 1
        
        f_lp = f_shift * mask_lp
        img_lp = xp.fft.ifft2(xp.fft.ifftshift(f_lp))
        img_lp = xp.abs(img_lp)
        
        results[f"{base_name}_lowpass"] = xp.clip(img_lp, 0, 255).astype(xp.uint8)
    
    def process_batch(self, image_paths: List[str], output_folder: str, 
                     return_arrays: bool = False) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image paths
            output_folder: Output folder path
            return_arrays: If True, return dictionary of processed arrays
            
        Returns:
            Dictionary mapping image names to processed arrays if return_arrays=True
        """
        batch_start = time.time()
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        all_results = {} if return_arrays else None
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(image_path, output_folder, return_arrays)
            
            if return_arrays and result:
                image_name = Path(image_path).stem
                all_results[image_name] = result
            
            # Clear GPU memory periodically
            if i % 10 == 0:
                self.gpu_manager.synchronize()
                if self.gpu_manager.use_gpu:
                    import cupy as cp
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
        
        batch_time = time.time() - batch_start
        self.logger.info(f"Batch processing completed in {batch_time:.2f}s")
        
        return all_results


def process_images_folder_gpu(input_folder: str, output_folder: str, 
                            config: Optional[Dict] = None, 
                            force_cpu: bool = False) -> None:
    """
    Process all images in a folder using GPU acceleration
    
    Args:
        input_folder: Input folder path
        output_folder: Output folder path  
        config: Configuration dictionary
        force_cpu: Force CPU mode for testing
    """
    if config is None:
        config = {}
    
    # Create processor
    processor = ImageProcessorGPU(config, force_cpu=force_cpu)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(input_folder).glob(f'*{ext}'))
        image_paths.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    if not image_paths:
        logger.warning(f"No images found in {input_folder}")
        return
    
    # Process batch
    processor.process_batch([str(p) for p in image_paths], output_folder)


if __name__ == "__main__":
    # Test the GPU processor
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python process_gpu.py <input_image_or_folder> <output_folder> [--cpu]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    force_cpu = '--cpu' in sys.argv
    
    if os.path.isfile(input_path):
        # Process single image
        processor = ImageProcessorGPU({}, force_cpu=force_cpu)
        processor.process_single_image(input_path, output_folder)
    else:
        # Process folder
        process_images_folder_gpu(input_path, output_folder, force_cpu=force_cpu)