#!/usr/bin/env python3
"""
Integrated Real-Time Defect Detection System

This is the main integrated system that combines:
- Enhanced Pylon Frame Grabber for camera capture
- Real-time defect detection using your detection.py
- Live visualization and monitoring
- Comprehensive logging and result saving

Usage:
    python realtime_defect_detection.py <reference_image_path>
    
Example:
    python realtime_defect_detection.py reference.jpg
"""

import time
import threading
import queue
import logging
import signal
import sys
import json
import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import cv2
import numpy as np

# Import detection components
try:
    from detection import OmniFiberAnalyzer, OmniConfig
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: detection.py not found: {e}")
    DETECTION_AVAILABLE = False

# Pylon SDK availability check
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("INFO: Pylon SDK found. Basler camera support enabled.")
    
    try:
        from genicam import GenericException
    except ImportError:
        class GenericException(Exception):
            pass
            
except ImportError:
    print("WARNING: Pylon SDK not found. Cannot use Basler camera.")
    print("Please install pypylon: pip install pypylon")


@dataclass
class DetectionConfig:
    """Configuration for real-time detection."""
    # Reference image settings
    reference_image_path: str = None
    
    # Detection sensitivity
    anomaly_threshold: float = 2.0
    ssim_threshold: float = 0.8
    confidence_threshold: float = 0.5
    
    # Processing optimization
    enable_fast_mode: bool = True
    resize_factor: float = 1.0
    max_processing_time: float = 0.1
    
    # Result filtering
    min_defect_area: int = 25
    max_defect_area: int = 5000
    
    # Visualization
    enable_visualization: bool = True
    save_results: bool = True
    output_dir: str = "realtime_output"
    
    # Camera settings
    exposure_time: int = 10000
    gain: int = 0
    buffer_size: int = 5
    grab_strategy: str = "LatestImageOnly"
    
    # Processing settings
    processing_fps: float = 10.0


@dataclass
class DetectionResult:
    """Detection result format."""
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


class EnhancedPylonGrabber(threading.Thread):
    """Enhanced Pylon frame grabber for real-time capture."""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(name="EnhancedPylonGrabber")
        self.daemon = True
        self.config = config
        
        # Camera components
        self.camera = None
        self.converter = None
        
        # Frame management
        self.latest_frame = None
        self.frame_metadata = {}
        self.lock = threading.RLock()
        
        # Buffer management
        self.frame_buffer = queue.Queue(maxsize=config.buffer_size)
        
        # Control flags
        self.is_running = threading.Event()
        self.is_initialized = threading.Event()
        
        # Performance monitoring
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Error handling
        self.error_count = 0
        self.last_error = None
        self.max_errors = 10
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if PYLON_AVAILABLE:
            self._setup_converter()
    
    def _setup_converter(self):
        """Initialize image format converter."""
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    def initialize_camera(self):
        """Initialize camera with configuration."""
        if not PYLON_AVAILABLE:
            self.logger.error("Pylon SDK not available")
            return False
        
        try:
            # Create and configure camera
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            
            self.camera.Open()
            self.logger.info(f"Camera initialized: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # Configure camera parameters
            if self.config.exposure_time is not None:
                self.camera.ExposureTime.SetValue(self.config.exposure_time)
                self.logger.info(f"Exposure time set to: {self.config.exposure_time} μs")
            
            if self.config.gain is not None:
                self.camera.Gain.SetValue(self.config.gain)
                self.logger.info(f"Gain set to: {self.config.gain}")
            
            # Set buffer count
            self.camera.MaxNumBuffer = self.config.buffer_size + 2
            
            # Configure grab strategy
            if self.config.grab_strategy == "LatestImageOnly":
                grab_strategy = pylon.GrabStrategy_LatestImageOnly
            else:
                grab_strategy = pylon.GrabStrategy_OneByOne
            
            self.camera.StartGrabbing(grab_strategy)
            self.is_initialized.set()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.last_error = str(e)
            return False
    
    def run(self):
        """Main grabbing loop."""
        self.logger.info("Enhanced frame grabber thread started")
        
        if not PYLON_AVAILABLE:
            self.logger.critical("Pylon SDK not available - cannot run")
            return
        
        # Wait for initialization
        if not self.is_initialized.wait(timeout=10):
            self.logger.error("Camera initialization timeout")
            return
        
        self.is_running.set()
        self.last_fps_time = time.time()
        
        try:
            while self.is_running.is_set() and self.camera.IsGrabbing():
                try:
                    grab_start_time = time.time()
                    
                    # Retrieve frame with timeout
                    grab_result = self.camera.RetrieveResult(
                        1000, pylon.TimeoutHandling_Return
                    )
                    
                    if grab_result and grab_result.GrabSucceeded():
                        self._process_frame(grab_result, grab_start_time)
                        self.error_count = 0
                    else:
                        self._handle_grab_failure(grab_result)
                    
                    if grab_result:
                        grab_result.Release()
                
                except GenericException as e:
                    self._handle_error(f"GenICam error: {e}")
                except Exception as e:
                    self._handle_error(f"Unexpected error: {e}")
                
                # Update FPS calculation
                self._update_fps()
        
        finally:
            self._cleanup()
    
    def _process_frame(self, grab_result, grab_start_time):
        """Process successfully grabbed frame."""
        try:
            # Convert frame format
            image = self.converter.Convert(grab_result)
            frame_array = image.GetArray().copy()
            
            # Create frame metadata
            processing_time = time.time() - grab_start_time
            metadata = {
                'timestamp': time.time(),
                'frame_number': grab_result.GetBlockID(),
                'processing_time': processing_time,
                'camera_timestamp': grab_result.GetTimeStamp(),
                'frame_size': frame_array.shape
            }
            
            # Thread-safe frame update
            with self.lock:
                self.latest_frame = frame_array
                self.frame_metadata = metadata
                
                # Add to buffer (non-blocking)
                try:
                    self.frame_buffer.put_nowait((frame_array.copy(), metadata.copy()))
                except queue.Full:
                    self.dropped_frames += 1
                
                self.frame_count += 1
        
        except Exception as e:
            self.logger.warning(f"Frame processing error: {e}")
            self.dropped_frames += 1
    
    def _handle_grab_failure(self, grab_result):
        """Handle failed frame grab."""
        if grab_result:
            error_msg = f"Grab failed: {grab_result.GetErrorDescription()}"
        else:
            error_msg = "Grab result is None (timeout)"
        
        self._handle_error(error_msg)
    
    def _handle_error(self, error_msg):
        """Centralized error handling."""
        self.error_count += 1
        self.last_error = error_msg
        
        if self.error_count <= 3:
            self.logger.warning(f"Error {self.error_count}: {error_msg}")
        elif self.error_count >= self.max_errors:
            self.logger.critical(f"Too many errors ({self.error_count}). Stopping grabber.")
            self.stop()
        
        time.sleep(0.01)
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            time_diff = current_time - self.last_fps_time
            self.current_fps = self.frame_count / time_diff if time_diff > 0 else 0
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _cleanup(self):
        """Clean up camera resources."""
        try:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                self.logger.info("Camera stopped grabbing")
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                self.logger.info("Camera closed")
        
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
        
        finally:
            self.is_running.clear()
            self.logger.info("Enhanced frame grabber thread finished")
    
    def read_latest_frame(self):
        """Get the most recent frame with metadata."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_metadata.copy()
            return None, None
    
    def get_statistics(self):
        """Get current performance statistics."""
        with self.lock:
            return {
                'fps': self.current_fps,
                'total_frames': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'error_count': self.error_count,
                'last_error': self.last_error,
                'buffer_size': self.frame_buffer.qsize(),
                'is_running': self.is_running.is_set(),
                'is_initialized': self.is_initialized.is_set()
            }
    
    def stop(self):
        """Stop the frame grabber gracefully."""
        self.logger.info("Stopping enhanced frame grabber...")
        self.is_running.clear()
    
    def wait_for_initialization(self, timeout=10):
        """Wait for camera initialization to complete."""
        return self.is_initialized.wait(timeout)
    
    def is_healthy(self):
        """Check if the grabber is running healthily."""
        return (
            self.is_running.is_set() and 
            self.error_count < self.max_errors and
            (time.time() - self.frame_metadata.get('timestamp', 0)) < 5.0
        )


class RealTimeDetector:
    """Real-time detector using your detection.py."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Detection components
        self.analyzer = None
        self.reference_image = None
        self.reference_gray = None
        
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
                enable_visualization=False
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
                    'std': np.ones_like(feature_vector) * 0.1,
                    'median': feature_vector,
                    'robust_mean': feature_vector,
                    'robust_cov': np.eye(len(feature_vector)) * 0.01,
                    'robust_inv_cov': np.eye(len(feature_vector)) * 100,
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
        """Perform real-time defect detection on a frame."""
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
                'confidence': max(0, 1.0 - ssim_score),
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
            is_anomalous=bool(is_anomalous),
            confidence=float(confidence),
            ssim_score=float(ssim_score),
            defect_count=int(len(defect_regions)),
            defect_regions=defect_regions,
            processing_time=float(0),
            frame_id=int(frame_id)
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
                    is_anomalous=bool(verdict['is_anomalous']),
                    confidence=float(verdict['confidence']),
                    ssim_score=float(structural['ssim']),
                    defect_count=int(len(defect_regions)),
                    defect_regions=defect_regions,
                    processing_time=float(0),
                    frame_id=int(frame_id)
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
    
    def visualize_result(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Create visualization of detection result."""
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


class RealTimeController:
    """Main controller for real-time defect detection system."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # System components
        self.frame_grabber = None
        self.detector = None
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=20)
        self.processing_thread = None
        self.result_thread = None
        self.visualization_thread = None
        
        # Control flags
        self.running = threading.Event()
        self.shutdown_requested = threading.Event()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'defects_detected': 0,
            'average_processing_time': 0,
            'start_time': None
        }
        
        # Alert callback
        self.defect_alert_callback = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Configure logging system."""
        log_file = self.output_dir / "realtime_detection.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger.info(f"Real-time controller initialized - Output: {self.output_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.info("Initializing real-time detection system...")
            
            # Initialize frame grabber
            self.logger.info("Initializing frame grabber...")
            self.frame_grabber = EnhancedPylonGrabber(self.config)
            
            # Initialize camera
            if not self.frame_grabber.initialize_camera():
                self.logger.error("Failed to initialize camera")
                return False
            
            # Initialize detector
            self.logger.info("Initializing detector...")
            self.detector = RealTimeDetector(self.config)
            
            self.logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the real-time detection system."""
        if not self.initialize():
            return False
        
        try:
            self.logger.info("Starting real-time detection system...")
            self.stats['start_time'] = time.time()
            
            # Start frame grabber
            self.frame_grabber.start()
            if not self.frame_grabber.wait_for_initialization(timeout=10):
                self.logger.error("Frame grabber initialization timeout")
                return False
            
            # Start processing threads
            self.running.set()
            
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="ProcessingThread",
                daemon=True
            )
            self.processing_thread.start()
            
            self.result_thread = threading.Thread(
                target=self._result_loop,
                name="ResultThread", 
                daemon=True
            )
            self.result_thread.start()
            
            if self.config.enable_visualization:
                self.visualization_thread = threading.Thread(
                    target=self._visualization_loop,
                    name="VisualizationThread",
                    daemon=True
                )
                self.visualization_thread.start()
            
            # Start main monitoring loop
            self._main_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.shutdown()
            return False
    
    def _main_loop(self):
        """Main system monitoring loop."""
        self.logger.info("Real-time detection system started successfully")
        self.logger.info("Press Ctrl+C to stop...")
        
        last_stats_time = time.time()
        
        try:
            while self.running.is_set() and not self.shutdown_requested.is_set():
                # Get latest frame from grabber
                frame, metadata = self.frame_grabber.read_latest_frame()
                
                if frame is not None:
                    self.stats['frames_captured'] += 1
                    
                    # Add frame to processing queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait((frame.copy(), metadata.copy()))
                    except queue.Full:
                        # Queue is full, skip this frame
                        pass
                
                # Print statistics every 5 seconds
                if time.time() - last_stats_time >= 5.0:
                    self._print_statistics()
                    last_stats_time = time.time()
                
                # Control processing rate
                time.sleep(1.0 / self.config.processing_fps)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        finally:
            self.shutdown()
    
    def _processing_loop(self):
        """Processing thread loop."""
        self.logger.info("Processing thread started")
        frame_id = 0
        
        while self.running.is_set():
            try:
                # Get frame from queue with timeout
                frame, metadata = self.frame_queue.get(timeout=1.0)
                frame_id += 1
                
                # Process frame
                start_time = time.time()
                result = self.detector.detect_defects(frame, frame_id)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['frames_processed'] += 1
                if result.is_anomalous:
                    self.stats['defects_detected'] += 1
                    
                    # Trigger alert callback if registered
                    if self.defect_alert_callback:
                        try:
                            self.defect_alert_callback(result, frame)
                        except Exception as e:
                            self.logger.warning(f"Alert callback error: {e}")
                
                # Update average processing time
                total_frames = self.stats['frames_processed']
                current_avg = self.stats['average_processing_time']
                self.stats['average_processing_time'] = (
                    (current_avg * (total_frames - 1) + processing_time) / total_frames
                )
                
                # Send to result queue
                try:
                    self.result_queue.put_nowait((result, frame.copy(), metadata))
                except queue.Full:
                    # Result queue is full, skip
                    pass
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
        
        self.logger.info("Processing thread finished")
    
    def _result_loop(self):
        """Result handling thread loop."""
        self.logger.info("Result thread started")
        
        while self.running.is_set():
            try:
                # Get result from queue
                result, frame, metadata = self.result_queue.get(timeout=1.0)
                
                # Save results if enabled
                if self.config.save_results:
                    self._save_result(result, frame, metadata)
                
                # Log significant detections
                if result.is_anomalous and result.confidence > 0.7:
                    self.logger.warning(
                        f"DEFECT DETECTED - Frame {result.frame_id}: "
                        f"Confidence={result.confidence:.3f}, "
                        f"Defects={result.defect_count}, "
                        f"SSIM={result.ssim_score:.3f}"
                    )
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Result handling error: {e}")
        
        self.logger.info("Result thread finished")
    
    def _visualization_loop(self):
        """Visualization thread loop."""
        self.logger.info("Visualization thread started")
        
        # Create window
        cv2.namedWindow("Real-Time Defect Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time Defect Detection", 1200, 800)
        
        while self.running.is_set():
            try:
                # Get result from queue (non-blocking)
                try:
                    result, frame, metadata = self.result_queue.get_nowait()
                except queue.Empty:
                    # Show last frame or continue
                    time.sleep(0.03)  # ~30 FPS display rate
                    continue
                
                # Create visualization
                vis_frame = self.detector.visualize_result(frame, result)
                
                # Add system statistics overlay
                self._add_stats_overlay(vis_frame)
                
                # Display frame
                cv2.imshow("Real-Time Defect Detection", vis_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.info("Visualization window closed")
                    self.shutdown_requested.set()
                    break
                
            except Exception as e:
                self.logger.error(f"Visualization error: {e}")
        
        cv2.destroyAllWindows()
        self.logger.info("Visualization thread finished")
    
    def _add_stats_overlay(self, frame: np.ndarray):
        """Add statistics overlay to visualization frame."""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 120), (400, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        stats_text = [
            f"Captured: {self.stats['frames_captured']}",
            f"Processed: {self.stats['frames_processed']}",
            f"Defects: {self.stats['defects_detected']}",
            f"Avg Time: {self.stats['average_processing_time']*1000:.1f}ms",
            f"FPS: {1.0/max(self.stats['average_processing_time'], 0.001):.1f}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (15, height - 100 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_result(self, result: DetectionResult, frame: np.ndarray, metadata: dict):
        """Save detection result and frame."""
        timestamp = int(result.timestamp)
        
        # Save result JSON
        result_file = self.output_dir / f"result_{timestamp}_{result.frame_id:06d}.json"
        with open(result_file, 'w') as f:
            json.dump({
                'result': result.to_dict(),
                'metadata': metadata
            }, f, indent=2)
        
        # Save frame if defect detected
        if result.is_anomalous and result.confidence > 0.5:
            frame_file = self.output_dir / f"defect_{timestamp}_{result.frame_id:06d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            
            # Save visualization
            vis_frame = self.detector.visualize_result(frame, result)
            vis_file = self.output_dir / f"defect_vis_{timestamp}_{result.frame_id:06d}.jpg"
            cv2.imwrite(str(vis_file), vis_frame)
    
    def _print_statistics(self):
        """Print current system statistics."""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        # Get additional statistics
        grabber_stats = self.frame_grabber.get_statistics() if self.frame_grabber else {}
        detector_stats = self.detector.get_statistics() if self.detector else {}
        
        self.logger.info(f"=== STATISTICS (Runtime: {runtime:.1f}s) ===")
        self.logger.info(f"Capture FPS: {grabber_stats.get('fps', 0):.1f}")
        self.logger.info(f"Processing FPS: {1.0/max(self.stats['average_processing_time'], 0.001):.1f}")
        self.logger.info(f"Frames Captured: {self.stats['frames_captured']}")
        self.logger.info(f"Frames Processed: {self.stats['frames_processed']}")
        self.logger.info(f"Defects Detected: {self.stats['defects_detected']}")
        self.logger.info(f"Detection Rate: {self.stats['defects_detected']/max(self.stats['frames_processed'], 1)*100:.1f}%")
        self.logger.info(f"Queue Sizes - Frame: {self.frame_queue.qsize()}, Result: {self.result_queue.qsize()}")
    
    def register_defect_alert(self, callback: Callable[[DetectionResult, np.ndarray], None]):
        """Register a callback function for defect alerts."""
        self.defect_alert_callback = callback
        self.logger.info("Defect alert callback registered")
    
    def shutdown(self):
        """Shutdown the system gracefully."""
        self.logger.info("Initiating system shutdown...")
        
        # Stop processing
        self.running.clear()
        
        # Stop frame grabber
        if self.frame_grabber:
            self.frame_grabber.stop()
        
        # Wait for threads to finish
        threads_to_wait = [
            self.processing_thread,
            self.result_thread,
            self.visualization_thread
        ]
        
        for thread in threads_to_wait:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Final statistics
        self._print_statistics()
        
        self.logger.info("System shutdown completed")


def defect_alert_handler(result: DetectionResult, frame: np.ndarray):
    """Example defect alert handler."""
    print(f"🚨 DEFECT ALERT! Frame {result.frame_id} - Confidence: {result.confidence:.3f}")
    # Add your custom alert logic here (email, alarm, etc.)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python realtime_defect_detection.py <reference_image_path>")
        print("Example: python realtime_defect_detection.py reference.jpg")
        sys.exit(1)
    
    reference_image = sys.argv[1]
    
    if not Path(reference_image).exists():
        print(f"Error: Reference image not found: {reference_image}")
        sys.exit(1)
    
    print("🎥 Real-Time Defect Detection System")
    print("====================================")
    print(f"Reference Image: {reference_image}")
    print("Press Ctrl+C to stop or 'q' in visualization window")
    print()
    
    # Create configuration
    config = DetectionConfig(
        reference_image_path=reference_image,
        anomaly_threshold=2.0,
        ssim_threshold=0.85,
        confidence_threshold=0.5,
        enable_fast_mode=True,
        resize_factor=1.0,
        min_defect_area=25,
        max_defect_area=5000,
        enable_visualization=True,
        save_results=True,
        output_dir="realtime_output",
        exposure_time=10000,
        gain=0,
        buffer_size=5,
        grab_strategy="LatestImageOnly",
        processing_fps=10.0
    )
    
    # Create controller
    controller = RealTimeController(config)
    
    # Register alert handler
    controller.register_defect_alert(defect_alert_handler)
    
    # Start system
    try:
        success = controller.start()
        if not success:
            print("❌ Failed to start real-time detection system")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 