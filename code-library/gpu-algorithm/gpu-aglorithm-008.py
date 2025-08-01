#!/usr/bin/env python3
"""
GPU-Accelerated Fiber Zone Separation Module
Identifies core, cladding, and ferrule regions using consensus-based segmentation
"""

import os
import json
import cv2
import numpy as np
import tempfile
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import multiprocessing as mp

from gpu_utils import GPUManager, gpu_accelerated, log_gpu_memory

# Configure logging
logger = logging.getLogger('SeparationGPU')


@dataclass
class MethodResult:
    """Result from a single segmentation method"""
    name: str
    center: Optional[Tuple[float, float]]
    core_radius: Optional[float]
    cladding_radius: Optional[float]
    confidence: float
    execution_time: float
    parameters: Dict[str, Any]
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None


@dataclass
class SeparationResult:
    """Result from the separation module"""
    masks: Dict[str, np.ndarray]  # core, cladding, ferrule masks
    regions: Dict[str, np.ndarray]  # core, cladding, ferrule image regions
    consensus: Dict[str, Any]  # consensus parameters
    defect_mask: np.ndarray  # detected defects from anomaly detection
    metadata: Dict[str, Any]  # timing, contributing methods, etc.


class UnifiedSeparationGPU:
    """GPU-accelerated unified fiber zone separation using consensus-based approach"""
    
    def __init__(self, config: Optional[Dict] = None, force_cpu: bool = False):
        """Initialize the separation module with GPU support"""
        self.config = config or {}
        self.gpu_manager = GPUManager(force_cpu=force_cpu)
        self.logger = logging.getLogger('UnifiedSeparationGPU')
        
        # Initialize method registry
        self.methods = self._initialize_methods()
        
        # Performance tracking
        self.dataset_stats = {
            'total_processed': 0,
            'consensus_achieved': 0,
            'method_accuracy': {name: {'success': 0, 'total': 0} for name in self.methods}
        }
        
        # Consensus system
        self.consensus_system = ConsensusSystemGPU(self.gpu_manager)
        
        # Methods vulnerable to defects
        self.vulnerable_methods = {
            'adaptive_intensity', 'bright_core', 'guess_approach',
            'hough_separation', 'threshold_separation'
        }
        
        self.logger.info(f"Initialized UnifiedSeparationGPU with GPU={self.gpu_manager.use_gpu}")
    
    def _initialize_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available segmentation methods"""
        methods = {
            'adaptive_intensity': {'score': 0.85, 'module': 'zones_methods.adaptive_intensity'},
            'bright_core': {'score': 0.80, 'module': 'zones_methods.bright_core_extractor'},
            'computational': {'score': 0.90, 'module': 'zones_methods.computational_separation'},
            'geometric': {'score': 0.88, 'module': 'zones_methods.geometric_approach'},
            'gradient': {'score': 0.87, 'module': 'zones_methods.gradient_approach'},
            'guess_approach': {'score': 0.75, 'module': 'zones_methods.guess_approach'},
            'hough_separation': {'score': 0.82, 'module': 'zones_methods.hough_separation'},
            'segmentation': {'score': 0.86, 'module': 'zones_methods.segmentation'},
            'threshold_separation': {'score': 0.78, 'module': 'zones_methods.threshold_separation'},
            'unified_detector': {'score': 0.91, 'module': 'zones_methods.unified_core_cladding_detector'}
        }
        return methods
    
    @gpu_accelerated
    def process_image(self, image_path: str, processed_arrays: Optional[Dict[str, np.ndarray]] = None) -> SeparationResult:
        """
        Process an image to separate fiber zones
        
        Args:
            image_path: Path to the original image
            processed_arrays: Optional pre-processed image arrays from process_gpu
            
        Returns:
            SeparationResult containing masks, regions, and metadata
        """
        start_time = time.time()
        self.logger.info(f"Processing image for separation: {image_path}")
        
        # Load original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Transfer to GPU
        original_img_gpu = self.gpu_manager.array_to_gpu(original_img)
        image_shape = original_img.shape[:2]
        
        # Anomaly detection and inpainting
        self.logger.info("Running anomaly detection and inpainting...")
        inpainted_img_gpu, defect_mask_gpu = self._detect_and_inpaint_anomalies_gpu(original_img_gpu)
        
        # Save inpainted image temporarily for methods that need file paths
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            inpainted_path = Path(tmp_f.name)
            cv2.imwrite(str(inpainted_path), self.gpu_manager.array_to_cpu(inpainted_img_gpu))
        
        # Run all segmentation methods
        all_results = []
        for method_name in self.methods:
            use_inpainted = method_name in self.vulnerable_methods
            current_image_path = inpainted_path if use_inpainted else image_path
            
            self.logger.info(f"Running {method_name} (using {'inpainted' if use_inpainted else 'original'} image)...")
            result = self._run_method_safe(method_name, str(current_image_path), image_shape)
            all_results.append(result)
            
            if result.error:
                self.logger.warning(f"{method_name} failed: {result.error}")
            else:
                self.logger.info(f"{method_name} completed - Confidence: {result.confidence:.2f}")
        
        # Generate consensus
        consensus = self.consensus_system.generate_consensus(
            all_results,
            {name: info['score'] for name, info in self.methods.items()},
            image_shape
        )
        
        if not consensus:
            raise RuntimeError("Failed to achieve consensus on fiber zones")
        
        # Update performance tracking
        self._update_learning(consensus, all_results)
        
        # Generate regions from masks
        masks = consensus['masks']
        regions = self._apply_masks_to_image_gpu(original_img_gpu, masks)
        
        # Clean up temporary file
        os.remove(inpainted_path)
        
        # Prepare result
        elapsed_time = time.time() - start_time
        metadata = {
            'processing_time': elapsed_time,
            'contributing_methods': consensus['contributing_methods'],
            'confidence': consensus['confidence'],
            'center': consensus['center'],
            'core_radius': consensus['core_radius'],
            'cladding_radius': consensus['cladding_radius']
        }
        
        # Convert GPU arrays back to CPU for compatibility
        result = SeparationResult(
            masks={k: self.gpu_manager.array_to_cpu(v) for k, v in masks.items()},
            regions={k: self.gpu_manager.array_to_cpu(v) for k, v in regions.items()},
            consensus=consensus,
            defect_mask=self.gpu_manager.array_to_cpu(defect_mask_gpu),
            metadata=metadata
        )
        
        self.logger.info(f"Separation completed in {elapsed_time:.2f}s")
        log_gpu_memory()
        
        return result
    
    @gpu_accelerated
    def _detect_and_inpaint_anomalies_gpu(self, image: Union[np.ndarray, 'cp.ndarray']) -> Tuple[Union[np.ndarray, 'cp.ndarray'], Union[np.ndarray, 'cp.ndarray']]:
        """GPU-accelerated anomaly detection and inpainting"""
        xp = self.gpu_manager.get_array_module(image)
        
        # Convert to grayscale
        gray = xp.dot(image[..., :3], xp.array([0.299, 0.587, 0.114])).astype(xp.uint8)
        
        # Detect anomalies using adaptive thresholding
        blur = self._gaussian_blur_gpu(gray, 5)
        
        # Calculate local statistics
        mean = self._box_filter_gpu(blur.astype(xp.float32), 15)
        mean_sq = self._box_filter_gpu((blur.astype(xp.float32) ** 2), 15)
        variance = mean_sq - mean ** 2
        std_dev = xp.sqrt(xp.maximum(variance, 0))
        
        # Detect anomalies (pixels that deviate significantly from local mean)
        z_score = xp.abs(blur.astype(xp.float32) - mean) / (std_dev + 1e-6)
        defect_mask = (z_score > 3.0).astype(xp.uint8) * 255
        
        # Dilate defect mask to ensure complete coverage
        kernel = xp.ones((5, 5), xp.uint8)
        defect_mask = self._morphology_gpu(defect_mask, 'dilate', kernel, iterations=2)
        
        # Inpaint defects
        if xp == np:
            # CPU inpainting
            inpainted = cv2.inpaint(image, defect_mask, 3, cv2.INPAINT_TELEA)
        else:
            # GPU inpainting (simplified version)
            inpainted = self._inpaint_gpu(image, defect_mask)
        
        return inpainted, defect_mask
    
    @gpu_accelerated
    def _gaussian_blur_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], kernel_size: int) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated Gaussian blur"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp == np:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            # GPU Gaussian blur using separable filter
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma, xp)
            
            # Apply separable convolution
            temp = self._convolve_1d_gpu(image.astype(xp.float32), kernel_1d, axis=1)
            result = self._convolve_1d_gpu(temp, kernel_1d, axis=0)
            
            return result.astype(xp.uint8)
    
    @gpu_accelerated
    def _box_filter_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], kernel_size: int) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated box filter (uniform filter)"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp == np:
            return cv2.boxFilter(image, -1, (kernel_size, kernel_size))
        else:
            # GPU box filter using integral image
            integral = xp.cumsum(xp.cumsum(image, axis=0), axis=1)
            
            # Pad integral image
            k = kernel_size // 2
            padded = xp.pad(integral, ((k+1, k), (k+1, k)), mode='edge')
            
            # Calculate box filter using integral image
            result = (padded[kernel_size:, kernel_size:] - 
                     padded[:-kernel_size, kernel_size:] - 
                     padded[kernel_size:, :-kernel_size] + 
                     padded[:-kernel_size, :-kernel_size])
            
            return result / (kernel_size * kernel_size)
    
    @gpu_accelerated
    def _morphology_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], operation: str, 
                       kernel: Union[np.ndarray, 'cp.ndarray'], iterations: int = 1) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU-accelerated morphological operations"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp == np:
            # CPU morphology
            if operation == 'dilate':
                return cv2.dilate(image, kernel, iterations=iterations)
            elif operation == 'erode':
                return cv2.erode(image, kernel, iterations=iterations)
        else:
            # GPU morphology
            result = image.copy()
            for _ in range(iterations):
                if operation == 'dilate':
                    result = self._dilate_gpu(result, kernel)
                elif operation == 'erode':
                    result = self._erode_gpu(result, kernel)
            return result
    
    @gpu_accelerated
    def _dilate_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], 
                   kernel: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU dilation using maximum filter"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp.__name__ == 'cupy':
            from cupyx.scipy import ndimage
            return ndimage.maximum_filter(image, footprint=kernel)
        else:
            return cv2.dilate(image, kernel)
    
    @gpu_accelerated
    def _erode_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], 
                  kernel: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """GPU erosion using minimum filter"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp.__name__ == 'cupy':
            from cupyx.scipy import ndimage
            return ndimage.minimum_filter(image, footprint=kernel)
        else:
            return cv2.erode(image, kernel)
    
    @gpu_accelerated
    def _inpaint_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], 
                    mask: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Simplified GPU inpainting using averaging"""
        xp = self.gpu_manager.get_array_module(image)
        
        # Simple inpainting by averaging neighboring pixels
        result = image.copy()
        
        # Create dilated mask for neighbor search
        kernel = xp.ones((3, 3), xp.uint8)
        dilated_mask = self._dilate_gpu(mask, kernel)
        
        # Find pixels to inpaint
        inpaint_pixels = mask > 0
        neighbor_pixels = (dilated_mask > 0) & (mask == 0)
        
        # For each channel
        for c in range(image.shape[2]):
            channel = image[:, :, c].astype(xp.float32)
            
            # Average neighboring pixels
            neighbor_sum = self._box_filter_gpu(channel * neighbor_pixels, 3)
            neighbor_count = self._box_filter_gpu(neighbor_pixels.astype(xp.float32), 3)
            
            # Replace defect pixels with average of neighbors
            avg_values = xp.divide(neighbor_sum, neighbor_count, 
                                  out=xp.zeros_like(neighbor_sum), 
                                  where=neighbor_count > 0)
            
            result[:, :, c] = xp.where(inpaint_pixels, avg_values, channel).astype(xp.uint8)
        
        return result
    
    def _gaussian_kernel_1d(self, size: int, sigma: float, xp) -> Any:
        """Create 1D Gaussian kernel"""
        x = xp.arange(size) - size // 2
        kernel = xp.exp(-(x ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()
    
    @gpu_accelerated
    def _convolve_1d_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], 
                        kernel: Union[np.ndarray, 'cp.ndarray'], axis: int) -> Union[np.ndarray, 'cp.ndarray']:
        """1D convolution along specified axis"""
        xp = self.gpu_manager.get_array_module(image)
        
        if xp.__name__ == 'cupy':
            from cupyx.scipy import ndimage
            return ndimage.convolve1d(image, kernel, axis=axis, mode='constant')
        else:
            from scipy import ndimage
            return ndimage.convolve1d(image, kernel, axis=axis, mode='constant')
    
    def _run_method_safe(self, method_name: str, image_path: str, 
                        image_shape: Tuple[int, int]) -> MethodResult:
        """Run a segmentation method safely with error handling"""
        start_time = time.time()
        
        try:
            # Import and run the method
            module_name = self.methods[method_name]['module']
            module = __import__(module_name, fromlist=['process'])
            
            # Run the method
            result = module.process(image_path)
            
            # Validate result
            if result and all(k in result for k in ['center', 'core_radius', 'cladding_radius']):
                execution_time = time.time() - start_time
                return MethodResult(
                    name=method_name,
                    center=result['center'],
                    core_radius=result['core_radius'],
                    cladding_radius=result['cladding_radius'],
                    confidence=result.get('confidence', 0.5),
                    execution_time=execution_time,
                    parameters=result.get('parameters', {}),
                    debug_info=result.get('debug_info')
                )
            else:
                raise ValueError("Invalid result format")
                
        except Exception as e:
            execution_time = time.time() - start_time
            return MethodResult(
                name=method_name,
                center=None,
                core_radius=None,
                cladding_radius=None,
                confidence=0.0,
                execution_time=execution_time,
                parameters={},
                error=str(e)
            )
    
    @gpu_accelerated
    def _apply_masks_to_image_gpu(self, image: Union[np.ndarray, 'cp.ndarray'], 
                                 masks: Dict[str, Union[np.ndarray, 'cp.ndarray']]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Apply masks to extract regions using GPU"""
        xp = self.gpu_manager.get_array_module(image)
        
        regions = {}
        for region_name, mask in masks.items():
            # Ensure mask is same type as image
            if type(mask) != type(image):
                mask = self.gpu_manager.array_to_gpu(mask) if xp != np else self.gpu_manager.array_to_cpu(mask)
            
            # Apply mask to each channel
            if len(image.shape) == 3:
                region = xp.zeros_like(image)
                for c in range(image.shape[2]):
                    region[:, :, c] = image[:, :, c] * mask
            else:
                region = image * mask
            
            regions[region_name] = region
        
        return regions
    
    def _update_learning(self, consensus: Dict, all_results: List[MethodResult]):
        """Update performance tracking"""
        self.dataset_stats['total_processed'] += 1
        
        if consensus:
            self.dataset_stats['consensus_achieved'] += 1
            
            # Update method accuracy
            for result in all_results:
                if result.name in self.dataset_stats['method_accuracy']:
                    self.dataset_stats['method_accuracy'][result.name]['total'] += 1
                    if not result.error and result.name in consensus['contributing_methods']:
                        self.dataset_stats['method_accuracy'][result.name]['success'] += 1


class ConsensusSystemGPU:
    """GPU-accelerated consensus generation for segmentation results"""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.logger = logging.getLogger('ConsensusSystemGPU')
    
    def generate_consensus(self, results: List[MethodResult], 
                         method_weights: Dict[str, float],
                         image_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Generate consensus from multiple segmentation results"""
        # Filter valid results
        valid_results = [r for r in results if not r.error and r.center and r.core_radius and r.cladding_radius]
        
        if len(valid_results) < 3:
            self.logger.warning(f"Insufficient valid results for consensus: {len(valid_results)}")
            return None
        
        # Extract parameters
        centers = np.array([r.center for r in valid_results])
        core_radii = np.array([r.core_radius for r in valid_results])
        cladding_radii = np.array([r.cladding_radius for r in valid_results])
        confidences = np.array([r.confidence for r in valid_results])
        weights = np.array([method_weights[r.name] for r in valid_results])
        
        # Calculate weighted consensus
        total_weight = np.sum(weights * confidences)
        consensus_center = np.sum(centers * (weights * confidences)[:, np.newaxis], axis=0) / total_weight
        consensus_core_radius = np.sum(core_radii * weights * confidences) / total_weight
        consensus_cladding_radius = np.sum(cladding_radii * weights * confidences) / total_weight
        
        # Generate masks
        masks = self._generate_masks_gpu(
            consensus_center, 
            consensus_core_radius, 
            consensus_cladding_radius, 
            image_shape
        )
        
        # Prepare consensus result
        consensus = {
            'center': tuple(consensus_center),
            'core_radius': float(consensus_core_radius),
            'cladding_radius': float(consensus_cladding_radius),
            'confidence': float(np.mean(confidences)),
            'masks': masks,
            'contributing_methods': [r.name for r in valid_results],
            'all_results': results
        }
        
        return consensus
    
    @gpu_accelerated
    def _generate_masks_gpu(self, center: np.ndarray, core_radius: float, 
                           cladding_radius: float, 
                           image_shape: Tuple[int, int]) -> Dict[str, Union[np.ndarray, 'cp.ndarray']]:
        """Generate region masks using GPU"""
        xp = self.gpu_manager.get_array_module(np.array([0]))
        
        h, w = image_shape
        
        # Create coordinate grids
        y, x = xp.ogrid[:h, :w]
        
        # Calculate distances from center
        cx, cy = center
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        
        # Create masks
        core_mask = (dist_sq <= core_radius ** 2).astype(xp.uint8)
        cladding_mask = ((dist_sq > core_radius ** 2) & 
                        (dist_sq <= cladding_radius ** 2)).astype(xp.uint8)
        ferrule_mask = (dist_sq > cladding_radius ** 2).astype(xp.uint8)
        
        return {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }


def process_image_with_separation(image_path: str, config: Optional[Dict] = None,
                                 force_cpu: bool = False) -> SeparationResult:
    """
    Process a single image through the separation pipeline
    
    Args:
        image_path: Path to the image
        config: Configuration dictionary
        force_cpu: Force CPU mode for testing
        
    Returns:
        SeparationResult with masks, regions, and metadata
    """
    separator = UnifiedSeparationGPU(config, force_cpu)
    return separator.process_image(image_path)


if __name__ == "__main__":
    # Test the GPU separator
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python separation_gpu.py <image_path> [--cpu]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    force_cpu = '--cpu' in sys.argv
    
    # Process image
    result = process_image_with_separation(image_path, force_cpu=force_cpu)
    
    print(f"Separation completed successfully!")
    print(f"Center: {result.metadata['center']}")
    print(f"Core radius: {result.metadata['core_radius']:.2f}")
    print(f"Cladding radius: {result.metadata['cladding_radius']:.2f}")
    print(f"Processing time: {result.metadata['processing_time']:.2f}s")
    print(f"Contributing methods: {', '.join(result.metadata['contributing_methods'])}")