#!/usr/bin/env python3
"""
Real-Time Detection Adapter for OmniFiberAnalyzer

This module provides a specialized adapter for your OmniFiberAnalyzer
to enable real-time defect detection with specific reference images.

Key Features:
- Specific reference image loading and comparison
- Real-time frame processing optimization
- Simplified result format for real-time use
- Memory-efficient processing
- Configurable detection sensitivity
"""

import time
import threading
import logging
import numpy as np
import cv2
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

# Import your detection module
try:
    from detection import OmniFiberAnalyzer, OmniConfig
    DETECTION_AVAILABLE = True
except ImportError:
    print("WARNING: detection.py not found. Please ensure it's in the Python path.")
    DETECTION_AVAILABLE = False


@dataclass
class RealTimeConfig:
    """Configuration for real-time detection."""
    # Reference image settings
    reference_image_path: str = None
    
    # Detection sensitivity
    anomaly_threshold: float = 2.0
    ssim_threshold: float = 0.8
    confidence_threshold: float = 0.5
    
    # Processing optimization
    enable_fast_mode: bool = True
    resize_factor: float = 1.0  # Scale images for faster processing
    max_processing_time: float = 0.1  # Max time per frame in seconds
    
    # Result filtering
    min_defect_area: int = 25
    max_defect_area: int = 5000
    
    # Visualization
    enable_visualization: bool = True
    save_detections: bool = False
    output_dir: str = "realtime_results"


@dataclass
class DetectionResult:
    """Simplified result format for real-time detection."""
    timestamp: float
    is_anomalous: bool
    confidence: float
    ssim_score: float
    defect_count: int
    defect_regions: List[Dict]
    processing_time: float
    frame_id: int = 0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'is_anomalous': self.is_anomalous,
            'confidence': self.confidence,
            'ssim_score': self.ssim_score,
            'defect_count': self.defect_count,
            'defect_regions': self.defect_regions,
            'processing_time': self.processing_time,
            'frame_id': self.frame_id
        }


class RealTimeDetector:
    """
    Real-time detector adapter for OmniFiberAnalyzer with specific reference image.
    
    This class wraps your existing OmniFiberAnalyzer to provide:
    - Fast reference image comparison
    - Real-time optimized processing
    - Simplified result format
    - Memory-efficient operation
    """
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Detection components
        self.analyzer = None
        self.reference_image = None
        self.reference_gray = None
        self.reference_features = None
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_result = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the detector with reference image."""
        if not DETECTION_AVAILABLE:
            raise RuntimeError("Detection module not available")
        
        try:
            # Create OmniFiberAnalyzer config
            omni_config = OmniConfig(
                min_defect_size=self.config.min_defect_area,
                max_defect_size=self.config.max_defect_area,
                confidence_threshold=self.config.confidence_threshold,
                anomaly_threshold_multiplier=self.config.anomaly_threshold,
                enable_visualization=False  # Disable for speed
            )
            
            # Initialize analyzer
            self.analyzer = OmniFiberAnalyzer(omni_config)
            
            # Load reference image
            if self.config.reference_image_path:
                self._load_reference_image()
                self.logger.info("Real-time detector initialized successfully")
            else:
                self.logger.warning("No reference image specified")
        
        except Exception as e:
            self.logger.error(f"Detector initialization failed: {e}")
            raise
    
    def _load_reference_image(self):
        """Load and prepare the reference image."""
        ref_path = Path(self.config.reference_image_path)
        
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        
        self.logger.info(f"Loading reference image: {ref_path}")
        
        # Load reference image
        self.reference_image = cv2.imread(str(ref_path))
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
        
        # Apply resize factor if specified
        if self.config.resize_factor != 1.0:
            height, width = self.reference_image.shape[:2]
            new_height = int(height * self.config.resize_factor)
            new_width = int(width * self.config.resize_factor)
            self.reference_image = cv2.resize(
                self.reference_image, (new_width, new_height)
            )
        
        # Prepare grayscale version
        if len(self.reference_image.shape) == 3:
            self.reference_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        else:
            self.reference_gray = self.reference_image.copy()
        
        # Build minimal reference model using the specific image
        self._build_reference_model()
        
        self.logger.info(f"Reference image loaded: {self.reference_image.shape}")
    
    def _build_reference_model(self):
        """Build a reference model from the specific reference image."""
        try:
            # Extract features from reference image
            features, feature_names = self.analyzer.extract_ultra_comprehensive_features(
                self.reference_image
            )
            
            # Create feature vector
            feature_vector = np.array([features[fname] for fname in feature_names])
            
            # Build minimal statistical model
            self.analyzer.reference_model = {
                'features': [features],
                'feature_names': feature_names,
                'statistical_model': {
                    'mean': feature_vector,
                    'std': np.ones_like(feature_vector) * 0.1,  # Small std for tight comparison
                    'median': feature_vector,
                    'robust_mean': feature_vector,
                    'robust_cov': np.eye(len(feature_vector)) * 0.01,  # Small covariance
                    'robust_inv_cov': np.eye(len(feature_vector)) * 100,  # Inverse
                    'n_samples': 1,
                },
                'archetype_image': self.reference_gray,
                'learned_thresholds': {
                    'anomaly_mean': 0.5,
                    'anomaly_std': 0.2,
                    'anomaly_p90': 1.0,
                    'anomaly_p95': 1.5,
                    'anomaly_p99': 2.0,
                    'anomaly_threshold': self.config.anomaly_threshold,
                },
                'timestamp': time.strftime("%Y-%m-%d_%H:%M:%S")
            }
            
            self.logger.info("Reference model built successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to build reference model: {e}")
            raise
    
    def detect_defects(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Perform real-time defect detection on a frame.
        
        Args:
            frame: Input frame as numpy array
            frame_id: Optional frame identifier
            
        Returns:
            DetectionResult: Detection results
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Prepare frame
                processed_frame = self._prepare_frame(frame)
                
                if self.config.enable_fast_mode:
                    # Fast detection using SSIM and basic comparison
                    result = self._fast_detection(processed_frame, frame_id)
                else:
                    # Full detection using your existing analyzer
                    result = self._full_detection(processed_frame, frame_id)
                
                # Update statistics
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                self.frame_count += 1
                self.total_processing_time += processing_time
                self.last_result = result
                
                return result
        
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            # Return error result
            return DetectionResult(
                timestamp=time.time(),
                is_anomalous=False,
                confidence=0.0,
                ssim_score=0.0,
                defect_count=0,
                defect_regions=[],
                processing_time=time.time() - start_time,
                frame_id=frame_id
            )
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for detection."""
        # Resize if needed
        if self.config.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.config.resize_factor)
            new_width = int(width * self.config.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def _fast_detection(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        """Fast detection using SSIM and basic comparison."""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        # Ensure same size as reference
        if frame_gray.shape != self.reference_gray.shape:
            frame_gray = cv2.resize(frame_gray, 
                                   (self.reference_gray.shape[1], self.reference_gray.shape[0]))
        
        # Calculate SSIM
        ssim_score = self._calculate_ssim(frame_gray, self.reference_gray)
        
        # Simple difference-based defect detection
        diff = cv2.absdiff(frame_gray, self.reference_gray)
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours (defects)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_defect_area <= area <= self.config.max_defect_area:
                valid_contours.append(contour)
        
        # Create defect regions
        defect_regions = []
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            defect_regions.append({
                'id': i,
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': float(area),
                'confidence': max(0, 1.0 - ssim_score),  # Higher diff = higher confidence
                'type': 'anomaly'
            })
        
        # Determine if anomalous
        is_anomalous = (
            ssim_score < self.config.ssim_threshold or 
            len(defect_regions) > 0
        )
        
        confidence = max(0, 1.0 - ssim_score) if is_anomalous else ssim_score
        
        return DetectionResult(
            timestamp=time.time(),
            is_anomalous=is_anomalous,
            confidence=confidence,
            ssim_score=ssim_score,
            defect_count=len(defect_regions),
            defect_regions=defect_regions,
            processing_time=0,  # Will be set by caller
            frame_id=frame_id
        )
    
    def _full_detection(self, frame: np.ndarray, frame_id: int) -> DetectionResult:
        """Full detection using the complete analyzer."""
        try:
            # Save frame temporarily for analyzer
            temp_path = Path(self.config.output_dir) / f"temp_frame_{frame_id}.jpg"
            temp_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(temp_path), frame)
            
            # Run full analysis
            results = self.analyzer.detect_anomalies_comprehensive(str(temp_path))
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            if results:
                # Convert to simplified format
                verdict = results['verdict']
                structural = results['structural_analysis']
                local_analysis = results['local_analysis']
                
                # Extract defect regions
                defect_regions = []
                for i, region in enumerate(local_analysis['anomaly_regions']):
                    defect_regions.append({
                        'id': i,
                        'bbox': region['bbox'],
                        'area': region['area'],
                        'confidence': region['confidence'],
                        'type': 'anomaly'
                    })
                
                return DetectionResult(
                    timestamp=time.time(),
                    is_anomalous=verdict['is_anomalous'],
                    confidence=verdict['confidence'],
                    ssim_score=structural['ssim'],
                    defect_count=len(defect_regions),
                    defect_regions=defect_regions,
                    processing_time=0,
                    frame_id=frame_id
                )
            else:
                # Return default result
                return self._fast_detection(frame, frame_id)
        
        except Exception as e:
            self.logger.warning(f"Full detection failed, falling back to fast: {e}")
            return self._fast_detection(frame, frame_id)
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index between two images."""
        try:
            # Simple SSIM implementation
            mu1 = cv2.GaussianBlur(img1.astype(float), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2.astype(float), (11, 11), 1.5)
            
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1.astype(float)**2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2.astype(float)**2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1.astype(float) * img2.astype(float), (11, 11), 1.5) - mu1_mu2
            
            C1 = (0.01 * 255)**2
            C2 = (0.03 * 255)**2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return np.mean(ssim_map)
        
        except:
            # Fallback to simple correlation
            correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
            return np.max(correlation)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        avg_processing_time = (
            self.total_processing_time / self.frame_count 
            if self.frame_count > 0 else 0
        )
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time': avg_processing_time,
            'fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'reference_loaded': self.reference_image is not None,
            'last_result': self.last_result.to_dict() if self.last_result else None
        }
    
    def set_reference_image(self, image_path: str):
        """Change the reference image."""
        self.config.reference_image_path = image_path
        self._load_reference_image()
        self.logger.info(f"Reference image updated: {image_path}")
    
    def visualize_result(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Create visualization of detection result.
        
        Args:
            frame: Original frame
            result: Detection result
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw defect regions
        for region in result.defect_regions:
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Color based on confidence (red = high confidence defect)
            color = (0, 0, int(255 * confidence))
            thickness = 2 if confidence > 0.7 else 1
            
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Add confidence text
            cv2.putText(vis_frame, f"{confidence:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add status text
        status_color = (0, 0, 255) if result.is_anomalous else (0, 255, 0)
        status_text = f"DEFECT DETECTED" if result.is_anomalous else "OK"
        
        cv2.putText(vis_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add metrics
        cv2.putText(vis_frame, f"SSIM: {result.ssim_score:.3f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"Confidence: {result.confidence:.3f}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"Defects: {result.defect_count}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_frame


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    config = RealTimeConfig(
        reference_image_path="reference.jpg",  # Replace with your reference image
        anomaly_threshold=2.0,
        ssim_threshold=0.85,
        enable_fast_mode=True,
        resize_factor=0.5  # Half size for faster processing
    )
    
    try:
        # Create detector
        detector = RealTimeDetector(config)
        print("Real-time detector initialized successfully")
        
        # Test with a sample frame (replace with actual frame)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Perform detection
        result = detector.detect_defects(test_frame, frame_id=1)
        
        print(f"Detection result: {result.to_dict()}")
        print(f"Statistics: {detector.get_statistics()}")
    
    except Exception as e:
        print(f"Test failed: {e}")