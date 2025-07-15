#!/usr/bin/env python3
"""
GPU-Accelerated Defect Detection Module
Analyzes individual fiber regions (core, cladding, ferrule) for anomalies
"""

import json
import os
import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

from gpu_utils import GPUManager, gpu_accelerated, log_gpu_memory

# Configure logging
logger = logging.getLogger('DetectionGPU')


@dataclass
class DefectInfo:
    """Information about a detected defect"""
    location: Tuple[int, int]  # (x, y) coordinates
    size: float  # Area in pixels
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, NEGLIGIBLE
    confidence: float  # 0-1 confidence score
    region: str  # core, cladding, or ferrule
    features: Dict[str, float]  # Feature values that triggered detection
    anomaly_score: float  # Statistical anomaly score


@dataclass
class RegionAnalysisResult:
    """Result of analyzing a single region"""
    region_name: str  # core, cladding, or ferrule
    defects: List[DefectInfo]
    statistics: Dict[str, float]  # Region statistics
    quality_score: float  # 0-100 quality score
    processing_time: float


@dataclass
class DetectionResult:
    """Complete detection result for all regions"""
    core_result: RegionAnalysisResult
    cladding_result: RegionAnalysisResult
    ferrule_result: RegionAnalysisResult
    overall_quality: float  # 0-100 overall quality score
    total_defects: int
    critical_defects: int
    total_processing_time: float
    metadata: Dict[str, Any]


@dataclass
class OmniConfigGPU:
    """Configuration for GPU-accelerated OmniFiberAnalyzer"""
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


class OmniFiberAnalyzerGPU:
    """GPU-accelerated fiber optic defect analyzer"""
    
    def __init__(self, config: Optional[OmniConfigGPU] = None, force_cpu: bool = False):
        """Initialize the GPU-accelerated analyzer"""
        self.config = config or OmniConfigGPU()
        self.gpu_manager = GPUManager(force_cpu or not self.config.use_gpu)
        self.logger = logging.getLogger('OmniFiberAnalyzerGPU')
        
        # Knowledge base for statistical models
        self.knowledge_base = {
            'core': {'features': {}, 'statistics': {}},
            'cladding': {'features': {}, 'statistics': {}},
            'ferrule': {'features': {}, 'statistics': {}}
        }
        
        # Load existing knowledge base if available
        if self.config.knowledge_base_path:
            self.load_knowledge_base(self.config.knowledge_base_path)
        
        self.logger.info(f"Initialized OmniFiberAnalyzerGPU with GPU={self.gpu_manager.use_gpu}")
    
    def analyze_regions(self, regions: Dict[str, np.ndarray], 
                       original_image: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Analyze fiber regions for defects
        
        Args:
            regions: Dictionary with 'core', 'cladding', 'ferrule' image regions
            original_image: Optional original image for reference
            
        Returns:
            DetectionResult containing analysis for all regions
        """
        start_time = time.time()
        self.logger.info("Starting GPU-accelerated defect detection")
        
        # Analyze each region
        core_result = self._analyze_region(regions['core'], 'core')
        cladding_result = self._analyze_region(regions['cladding'], 'cladding')
        ferrule_result = self._analyze_region(regions['ferrule'], 'ferrule')
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(
            core_result, cladding_result, ferrule_result
        )
        
        # Count defects
        all_defects = (core_result.defects + cladding_result.defects + 
                      ferrule_result.defects)
        total_defects = len(all_defects)
        critical_defects = len([d for d in all_defects if d.severity == 'CRITICAL'])
        
        # Prepare result
        total_time = time.time() - start_time
        result = DetectionResult(
            core_result=core_result,
            cladding_result=cladding_result,
            ferrule_result=ferrule_result,
            overall_quality=overall_quality,
            total_defects=total_defects,
            critical_defects=critical_defects,
            total_processing_time=total_time,
            metadata={
                'gpu_used': self.gpu_manager.use_gpu,
                'timestamp': time.time()
            }
        )
        
        self.logger.info(f"Detection completed in {total_time:.2f}s - "
                        f"Found {total_defects} defects ({critical_defects} critical)")
        log_gpu_memory()
        
        return result
    
    @gpu_accelerated
    def _analyze_region(self, region_image: np.ndarray, region_name: str) -> RegionAnalysisResult:
        """Analyze a single region for defects using GPU acceleration"""
        start_time = time.time()
        self.logger.debug(f"Analyzing {region_name} region")
        
        # Skip if region is empty
        if np.sum(region_image) == 0:
            self.logger.debug(f"{region_name} region is empty, skipping")
            return RegionAnalysisResult(
                region_name=region_name,
                defects=[],
                statistics={},
                quality_score=100.0,
                processing_time=time.time() - start_time
            )
        
        # Transfer to GPU
        region_gpu = self.gpu_manager.array_to_gpu(region_image)
        
        # Extract features
        features = self._extract_features_gpu(region_gpu, region_name)
        
        # Compute statistics
        statistics = self._compute_statistics_gpu(region_gpu)
        
        # Detect anomalies
        anomalies = self._detect_anomalies_gpu(region_gpu, features, statistics, region_name)
        
        # Find defect locations
        defects = self._locate_defects_gpu(region_gpu, anomalies, region_name)
        
        # Calculate quality score
        quality_score = self._calculate_region_quality(defects, statistics)
        
        # Convert results back to CPU
        statistics_cpu = {k: float(v) if hasattr(v, 'item') else v 
                         for k, v in statistics.items()}
        
        processing_time = time.time() - start_time
        
        return RegionAnalysisResult(
            region_name=region_name,
            defects=defects,
            statistics=statistics_cpu,
            quality_score=quality_score,
            processing_time=processing_time
        )
    
    @gpu_accelerated
    def _extract_features_gpu(self, region: Union[np.ndarray, 'cp.ndarray'], 
                             region_name: str) -> Dict[str, float]:
        """Extract features from region using GPU"""
        xp = self.gpu_manager.get_array_module(region)
        features = {}
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = xp.dot(region[..., :3], xp.array([0.299, 0.587, 0.114])).astype(xp.uint8)
        else:
            gray = region
        
        # Skip if empty
        mask = gray > 0
        if not xp.any(mask):
            return features
        
        # Extract only from non-zero pixels
        valid_pixels = gray[mask]
        
        # Basic statistics
        features['mean_intensity'] = float(xp.mean(valid_pixels))
        features['std_intensity'] = float(xp.std(valid_pixels))
        features['min_intensity'] = float(xp.min(valid_pixels))
        features['max_intensity'] = float(xp.max(valid_pixels))
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        # Texture features
        texture_features = self._extract_texture_features_gpu(gray, mask)
        features.update(texture_features)
        
        # Gradient features
        gradient_features = self._extract_gradient_features_gpu(gray, mask)
        features.update(gradient_features)
        
        # Frequency features
        frequency_features = self._extract_frequency_features_gpu(gray, mask)
        features.update(frequency_features)
        
        return features
    
    @gpu_accelerated
    def _extract_texture_features_gpu(self, gray: Union[np.ndarray, 'cp.ndarray'],
                                     mask: Union[np.ndarray, 'cp.ndarray']) -> Dict[str, float]:
        """Extract texture features using GPU"""
        xp = self.gpu_manager.get_array_module(gray)
        features = {}
        
        # Only process masked region
        if not xp.any(mask):
            return features
        
        # Local Binary Pattern approximation
        lbp_sum = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # Shift image
                shifted = xp.zeros_like(gray)
                if dy >= 0 and dx >= 0:
                    shifted[dy:, dx:] = gray[:-dy if dy else None, :-dx if dx else None]
                elif dy >= 0 and dx < 0:
                    shifted[dy:, :dx] = gray[:-dy if dy else None, -dx:]
                elif dy < 0 and dx >= 0:
                    shifted[:dy, dx:] = gray[-dy:, :-dx if dx else None]
                else:
                    shifted[:dy, :dx] = gray[-dy:, -dx:]
                
                # Compare with center
                lbp_sum += xp.sum((shifted > gray) & mask)
        
        features['texture_lbp_sum'] = float(lbp_sum)
        
        # Contrast (using gradients)
        dx = xp.zeros_like(gray, dtype=xp.float32)
        dy = xp.zeros_like(gray, dtype=xp.float32)
        
        dx[:, 1:] = gray[:, 1:].astype(xp.float32) - gray[:, :-1].astype(xp.float32)
        dy[1:, :] = gray[1:, :].astype(xp.float32) - gray[:-1, :].astype(xp.float32)
        
        gradient_magnitude = xp.sqrt(dx**2 + dy**2)
        features['texture_contrast'] = float(xp.mean(gradient_magnitude[mask]))
        
        # Homogeneity (inverse of contrast)
        features['texture_homogeneity'] = 1.0 / (1.0 + features['texture_contrast'])
        
        return features
    
    @gpu_accelerated
    def _extract_gradient_features_gpu(self, gray: Union[np.ndarray, 'cp.ndarray'],
                                      mask: Union[np.ndarray, 'cp.ndarray']) -> Dict[str, float]:
        """Extract gradient-based features using GPU"""
        xp = self.gpu_manager.get_array_module(gray)
        features = {}
        
        if not xp.any(mask):
            return features
        
        # Sobel gradients
        # X gradient
        sobel_x = xp.zeros_like(gray, dtype=xp.float32)
        sobel_x[1:-1, 1:-1] = (
            -1 * gray[:-2, :-2] + 1 * gray[:-2, 2:] +
            -2 * gray[1:-1, :-2] + 2 * gray[1:-1, 2:] +
            -1 * gray[2:, :-2] + 1 * gray[2:, 2:]
        ) / 8.0
        
        # Y gradient
        sobel_y = xp.zeros_like(gray, dtype=xp.float32)
        sobel_y[1:-1, 1:-1] = (
            -1 * gray[:-2, :-2] - 2 * gray[:-2, 1:-1] - 1 * gray[:-2, 2:] +
            1 * gray[2:, :-2] + 2 * gray[2:, 1:-1] + 1 * gray[2:, 2:]
        ) / 8.0
        
        # Gradient magnitude and direction
        grad_mag = xp.sqrt(sobel_x**2 + sobel_y**2)
        grad_dir = xp.arctan2(sobel_y, sobel_x)
        
        # Features
        valid_grad = grad_mag[mask]
        features['gradient_mean'] = float(xp.mean(valid_grad))
        features['gradient_std'] = float(xp.std(valid_grad))
        features['gradient_max'] = float(xp.max(valid_grad))
        
        # Gradient histogram
        hist, _ = xp.histogram(valid_grad, bins=10, range=(0, 255))
        hist = hist.astype(xp.float32) / (xp.sum(hist) + 1e-10)
        features['gradient_entropy'] = float(-xp.sum(hist * xp.log(hist + 1e-10)))
        
        return features
    
    @gpu_accelerated
    def _extract_frequency_features_gpu(self, gray: Union[np.ndarray, 'cp.ndarray'],
                                       mask: Union[np.ndarray, 'cp.ndarray']) -> Dict[str, float]:
        """Extract frequency domain features using GPU"""
        xp = self.gpu_manager.get_array_module(gray)
        features = {}
        
        if not xp.any(mask):
            return features
        
        # Get bounding box of mask
        rows, cols = xp.where(mask)
        if len(rows) == 0:
            return features
        
        min_row, max_row = int(xp.min(rows)), int(xp.max(rows)) + 1
        min_col, max_col = int(xp.min(cols)), int(xp.max(cols)) + 1
        
        # Extract ROI
        roi = gray[min_row:max_row, min_col:max_col].astype(xp.float32)
        
        # Apply window to reduce edge effects
        h, w = roi.shape
        window_y = xp.hanning(h)
        window_x = xp.hanning(w)
        window = window_y[:, xp.newaxis] * window_x[xp.newaxis, :]
        roi_windowed = roi * window
        
        # FFT
        fft = xp.fft.fft2(roi_windowed)
        fft_shift = xp.fft.fftshift(fft)
        magnitude = xp.abs(fft_shift)
        
        # Power spectrum
        power = magnitude ** 2
        
        # Radial average
        cy, cx = h // 2, w // 2
        y, x = xp.ogrid[:h, :w]
        r = xp.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Features
        features['freq_total_power'] = float(xp.sum(power))
        features['freq_dc_power'] = float(power[cy, cx])
        
        # High frequency power (outer 50% of radius)
        max_r = min(cy, cx)
        high_freq_mask = r > (max_r * 0.5)
        features['freq_high_power_ratio'] = float(
            xp.sum(power[high_freq_mask]) / (xp.sum(power) + 1e-10)
        )
        
        return features
    
    @gpu_accelerated
    def _compute_statistics_gpu(self, region: Union[np.ndarray, 'cp.ndarray']) -> Dict[str, Any]:
        """Compute region statistics using GPU"""
        xp = self.gpu_manager.get_array_module(region)
        stats = {}
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = xp.dot(region[..., :3], xp.array([0.299, 0.587, 0.114])).astype(xp.uint8)
        else:
            gray = region
        
        # Get mask of non-zero pixels
        mask = gray > 0
        pixel_count = int(xp.sum(mask))
        
        stats['pixel_count'] = pixel_count
        stats['region_area'] = pixel_count
        
        if pixel_count > 0:
            valid_pixels = gray[mask]
            
            # Intensity statistics
            stats['mean_intensity'] = float(xp.mean(valid_pixels))
            stats['std_intensity'] = float(xp.std(valid_pixels))
            stats['median_intensity'] = float(xp.median(valid_pixels))
            stats['min_intensity'] = float(xp.min(valid_pixels))
            stats['max_intensity'] = float(xp.max(valid_pixels))
            
            # Percentiles
            stats['percentile_25'] = float(xp.percentile(valid_pixels, 25))
            stats['percentile_75'] = float(xp.percentile(valid_pixels, 75))
            stats['iqr'] = stats['percentile_75'] - stats['percentile_25']
            
            # Skewness and kurtosis approximation
            mean = stats['mean_intensity']
            std = stats['std_intensity']
            if std > 0:
                normalized = (valid_pixels - mean) / std
                stats['skewness'] = float(xp.mean(normalized ** 3))
                stats['kurtosis'] = float(xp.mean(normalized ** 4) - 3)
            else:
                stats['skewness'] = 0.0
                stats['kurtosis'] = 0.0
        
        return stats
    
    @gpu_accelerated
    def _detect_anomalies_gpu(self, region: Union[np.ndarray, 'cp.ndarray'],
                             features: Dict[str, float],
                             statistics: Dict[str, Any],
                             region_name: str) -> Union[np.ndarray, 'cp.ndarray']:
        """Detect anomalies in the region using GPU"""
        xp = self.gpu_manager.get_array_module(region)
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = xp.dot(region[..., :3], xp.array([0.299, 0.587, 0.114])).astype(xp.uint8)
        else:
            gray = region
        
        # Initialize anomaly map
        anomaly_map = xp.zeros_like(gray, dtype=xp.float32)
        
        # Get mask of valid pixels
        mask = gray > 0
        if not xp.any(mask):
            return anomaly_map
        
        # Statistical anomaly detection
        mean = statistics.get('mean_intensity', 128)
        std = statistics.get('std_intensity', 30)
        
        # Z-score based anomaly
        z_scores = xp.abs(gray.astype(xp.float32) - mean) / (std + 1e-6)
        anomaly_map = xp.where(mask, z_scores, 0)
        
        # Local anomaly detection
        local_anomalies = self._detect_local_anomalies_gpu(gray, mask)
        
        # Combine anomalies
        anomaly_map = xp.maximum(anomaly_map, local_anomalies)
        
        # Threshold based on configuration
        threshold = self.config.anomaly_threshold_multiplier
        anomaly_map = xp.where(anomaly_map > threshold, anomaly_map, 0)
        
        return anomaly_map
    
    @gpu_accelerated
    def _detect_local_anomalies_gpu(self, gray: Union[np.ndarray, 'cp.ndarray'],
                                   mask: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """Detect local anomalies using GPU"""
        xp = self.gpu_manager.get_array_module(gray)
        
        # Local statistics using box filter
        gray_float = gray.astype(xp.float32)
        
        # Compute local mean and variance
        kernel_size = 15
        k = kernel_size // 2
        
        # Pad image
        padded = xp.pad(gray_float, k, mode='constant', constant_values=0)
        padded_mask = xp.pad(mask.astype(xp.float32), k, mode='constant', constant_values=0)
        
        # Integral images for fast box filtering
        integral = xp.cumsum(xp.cumsum(padded, axis=0), axis=1)
        integral_sq = xp.cumsum(xp.cumsum(padded**2, axis=0), axis=1)
        integral_mask = xp.cumsum(xp.cumsum(padded_mask, axis=0), axis=1)
        
        # Calculate sums using integral images
        def box_sum(integral, y1, y2, x1, x2):
            return (integral[y2, x2] - integral[y1, x2] - 
                   integral[y2, x1] + integral[y1, x1])
        
        h, w = gray.shape
        local_mean = xp.zeros((h, w), dtype=xp.float32)
        local_var = xp.zeros((h, w), dtype=xp.float32)
        
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                
                y1, y2 = y, y + kernel_size
                x1, x2 = x, x + kernel_size
                
                count = box_sum(integral_mask, y1, y2, x1, x2)
                if count > 0:
                    sum_val = box_sum(integral, y1, y2, x1, x2)
                    sum_sq = box_sum(integral_sq, y1, y2, x1, x2)
                    
                    local_mean[y, x] = sum_val / count
                    local_var[y, x] = (sum_sq / count) - (sum_val / count) ** 2
        
        # Calculate local z-scores
        local_std = xp.sqrt(xp.maximum(local_var, 0))
        local_z = xp.where(
            (mask) & (local_std > 0),
            xp.abs(gray_float - local_mean) / (local_std + 1e-6),
            0
        )
        
        return local_z
    
    @gpu_accelerated
    def _locate_defects_gpu(self, region: Union[np.ndarray, 'cp.ndarray'],
                           anomaly_map: Union[np.ndarray, 'cp.ndarray'],
                           region_name: str) -> List[DefectInfo]:
        """Locate and characterize defects from anomaly map using GPU"""
        xp = self.gpu_manager.get_array_module(region)
        
        # Threshold anomaly map
        threshold = self.config.anomaly_threshold_multiplier
        binary_defects = (anomaly_map > threshold).astype(xp.uint8)
        
        # Transfer to CPU for connected components (not available in CuPy)
        binary_cpu = self.gpu_manager.array_to_cpu(binary_defects)
        anomaly_cpu = self.gpu_manager.array_to_cpu(anomaly_map)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_cpu)
        
        defects = []
        
        for label_id in range(1, num_labels):
            # Get defect mask
            defect_mask = (labels == label_id)
            
            # Calculate defect properties
            area = np.sum(defect_mask)
            
            # Check size constraints
            if area < self.config.min_defect_size or area > self.config.max_defect_size:
                continue
            
            # Get defect location (centroid)
            y_coords, x_coords = np.where(defect_mask)
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            
            # Calculate severity based on anomaly scores
            defect_scores = anomaly_cpu[defect_mask]
            max_score = float(np.max(defect_scores))
            mean_score = float(np.mean(defect_scores))
            
            # Determine severity
            severity = self._determine_severity(mean_score, max_score)
            
            # Calculate confidence
            confidence = min(1.0, mean_score / 10.0)  # Normalize to 0-1
            
            # Extract features for this defect
            defect_features = {
                'area': float(area),
                'max_anomaly_score': max_score,
                'mean_anomaly_score': mean_score,
                'eccentricity': self._calculate_eccentricity(defect_mask)
            }
            
            # Create DefectInfo
            defect = DefectInfo(
                location=(cx, cy),
                size=float(area),
                severity=severity,
                confidence=confidence,
                region=region_name,
                features=defect_features,
                anomaly_score=mean_score
            )
            
            defects.append(defect)
        
        return defects
    
    def _determine_severity(self, mean_score: float, max_score: float) -> str:
        """Determine defect severity based on anomaly scores"""
        # Use combination of mean and max scores
        combined_score = 0.7 * mean_score + 0.3 * max_score
        
        # Map to severity levels
        if combined_score > 8.0:
            return 'CRITICAL'
        elif combined_score > 6.0:
            return 'HIGH'
        elif combined_score > 4.0:
            return 'MEDIUM'
        elif combined_score > 2.5:
            return 'LOW'
        else:
            return 'NEGLIGIBLE'
    
    def _calculate_eccentricity(self, mask: np.ndarray) -> float:
        """Calculate eccentricity of a defect"""
        # Find contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Fit ellipse if enough points
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            
            if major > 0:
                eccentricity = np.sqrt(1 - (minor/major)**2)
                return float(eccentricity)
        
        return 0.0
    
    def _calculate_region_quality(self, defects: List[DefectInfo], 
                                 statistics: Dict[str, Any]) -> float:
        """Calculate quality score for a region"""
        if not defects:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Penalty based on defect count and severity
        severity_weights = {
            'CRITICAL': 20.0,
            'HIGH': 10.0,
            'MEDIUM': 5.0,
            'LOW': 2.0,
            'NEGLIGIBLE': 0.5
        }
        
        for defect in defects:
            penalty = severity_weights.get(defect.severity, 1.0)
            # Scale penalty by defect size relative to region
            region_area = statistics.get('region_area', 10000)
            size_factor = min(1.0, defect.size / region_area * 100)
            score -= penalty * size_factor * defect.confidence
        
        return max(0.0, score)
    
    def _calculate_overall_quality(self, core_result: RegionAnalysisResult,
                                  cladding_result: RegionAnalysisResult,
                                  ferrule_result: RegionAnalysisResult) -> float:
        """Calculate overall quality score"""
        # Weighted average based on region importance
        weights = {
            'core': 0.5,
            'cladding': 0.3,
            'ferrule': 0.2
        }
        
        overall = (
            weights['core'] * core_result.quality_score +
            weights['cladding'] * cladding_result.quality_score +
            weights['ferrule'] * ferrule_result.quality_score
        )
        
        return overall
    
    def save_results(self, result: DetectionResult, output_path: str):
        """Save detection results to file"""
        # Convert to dictionary
        result_dict = {
            'core': asdict(result.core_result),
            'cladding': asdict(result.cladding_result),
            'ferrule': asdict(result.ferrule_result),
            'overall_quality': result.overall_quality,
            'total_defects': result.total_defects,
            'critical_defects': result.critical_defects,
            'processing_time': result.total_processing_time,
            'metadata': result.metadata
        }
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Saved detection results to {output_path}")
    
    def load_knowledge_base(self, path: str):
        """Load knowledge base from file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.knowledge_base = json.load(f)
            self.logger.info(f"Loaded knowledge base from {path}")
    
    def save_knowledge_base(self, path: str):
        """Save knowledge base to file"""
        with open(path, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        self.logger.info(f"Saved knowledge base to {path}")


def analyze_fiber_regions(regions: Dict[str, np.ndarray], 
                         config: Optional[OmniConfigGPU] = None,
                         force_cpu: bool = False) -> DetectionResult:
    """
    Analyze fiber regions for defects
    
    Args:
        regions: Dictionary with 'core', 'cladding', 'ferrule' regions
        config: Optional configuration
        force_cpu: Force CPU mode for testing
        
    Returns:
        DetectionResult with analysis for all regions
    """
    analyzer = OmniFiberAnalyzerGPU(config, force_cpu)
    return analyzer.analyze_regions(regions)


if __name__ == "__main__":
    # Test the GPU detector
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python detection_gpu.py <core_image> <cladding_image> <ferrule_image> [--cpu]")
        sys.exit(1)
    
    force_cpu = '--cpu' in sys.argv
    
    # Load test regions
    regions = {
        'core': cv2.imread(sys.argv[1]),
        'cladding': cv2.imread(sys.argv[2]),
        'ferrule': cv2.imread(sys.argv[3])
    }
    
    # Analyze
    result = analyze_fiber_regions(regions, force_cpu=force_cpu)
    
    print(f"Detection completed!")
    print(f"Overall quality: {result.overall_quality:.1f}%")
    print(f"Total defects: {result.total_defects}")
    print(f"Critical defects: {result.critical_defects}")
    print(f"Processing time: {result.total_processing_time:.2f}s")