#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PYLON Integration for Real-time Fiber Optic Quality Assurance
Integrates Basler cameras with CNN-based inspection system
"""

import os
import sys
import time
import json
import threading
import queue
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import argparse

# PYLON imports
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False
    print("Warning: PYLON not available. Install with: pip install pypylon")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Camera configuration settings"""
    exposure_time: float = 10000.0  # microseconds
    gain: float = 0.0
    pixel_format: str = "RGB8"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    trigger_mode: str = "Off"  # Off, On, OnRisingEdge
    auto_exposure: bool = True
    auto_gain: bool = True

@dataclass
class InspectionResult:
    """Result of fiber optic inspection"""
    timestamp: datetime
    image: np.ndarray
    zones: np.ndarray  # Zone segmentation
    defects: np.ndarray  # Defect detection
    quality_score: float  # Overall quality score
    quality_class: str  # pass, warning, fail
    defect_locations: List[Tuple[int, int, int]]  # x, y, defect_type
    processing_time: float
    confidence: float

class PylonCamera:
    """PYLON camera interface for fiber optic inspection"""
    
    def __init__(self, camera_config: CameraConfig = None):
        self.config = camera_config or CameraConfig()
        self.camera = None
        self.is_connected = False
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        if not PYLON_AVAILABLE:
            raise ImportError("PYLON not available. Install with: pip install pypylon")
    
    def connect(self, camera_serial: str = None) -> bool:
        """Connect to PYLON camera"""
        try:
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()
            
            # Get all attached devices and sort them by serial number
            devices = tl_factory.EnumerateDevices()
            if not devices:
                logger.error("No PYLON cameras found")
                return False
            
            # Select camera by serial number or use first available
            if camera_serial:
                device = next((d for d in devices if d.GetSerialNumber() == camera_serial), None)
                if not device:
                    logger.error(f"Camera with serial {camera_serial} not found")
                    return False
            else:
                device = devices[0]
                logger.info(f"Using camera: {device.GetFriendlyName()} (Serial: {device.GetSerialNumber()})")
            
            # Create camera object
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            self.camera.Open()
            
            # Configure camera settings
            self._configure_camera()
            
            self.is_connected = True
            logger.info("Camera connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False
    
    def _configure_camera(self):
        """Configure camera settings"""
        try:
            # Set pixel format
            if self.camera.PixelFormat.GetValue() != self.config.pixel_format:
                self.camera.PixelFormat.SetValue(self.config.pixel_format)
            
            # Set resolution
            self.camera.Width.SetValue(self.config.width)
            self.camera.Height.SetValue(self.config.height)
            
            # Set exposure and gain
            if not self.config.auto_exposure:
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureTime.SetValue(self.config.exposure_time)
            else:
                self.camera.ExposureAuto.SetValue("Continuous")
            
            if not self.config.auto_gain:
                self.camera.GainAuto.SetValue("Off")
                self.camera.Gain.SetValue(self.config.gain)
            else:
                self.camera.GainAuto.SetValue("Continuous")
            
            # Set trigger mode
            self.camera.TriggerMode.SetValue(self.config.trigger_mode)
            
            # Set acquisition frame rate
            if hasattr(self.camera, 'AcquisitionFrameRate'):
                self.camera.AcquisitionFrameRate.SetValue(self.config.fps)
            
            logger.info("Camera configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure camera: {e}")
    
    def start_streaming(self):
        """Start camera streaming"""
        if not self.is_connected:
            logger.error("Camera not connected")
            return False
        
        try:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_streaming = True
            
            # Start frame acquisition thread
            self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
            self.acquisition_thread.start()
            
            logger.info("Camera streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def _acquisition_loop(self):
        """Frame acquisition loop"""
        while not self.stop_event.is_set() and self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Convert to numpy array
                    image = grab_result.Array
                    
                    # Convert to RGB if needed
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Add to queue (remove old frames if queue is full)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(image)
                
                grab_result.Release()
                
            except Exception as e:
                logger.error(f"Error in acquisition loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest frame from camera"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.stop_event.set()
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.is_streaming = False
        logger.info("Camera streaming stopped")
    
    def disconnect(self):
        """Disconnect camera"""
        self.stop_streaming()
        if self.camera:
            self.camera.Close()
        self.is_connected = False
        logger.info("Camera disconnected")

class RealTimeInspector:
    """Real-time fiber optic inspector with PYLON integration"""
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.camera = None
        self.inspection_queue = queue.Queue(maxsize=5)
        self.results_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'avg_processing_time': 0.0,
            'quality_distribution': {'pass': 0, 'warning': 0, 'fail': 0},
            'defect_counts': [0, 0, 0, 0]  # 4 defect types
        }
    
    def load_model(self, model_path: str):
        """Load trained CNN model"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
            
            self.model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def setup_camera(self, camera_config: CameraConfig = None) -> bool:
        """Setup PYLON camera"""
        try:
            self.camera = PylonCamera(camera_config)
            return self.camera.connect()
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            return False
    
    def start_inspection(self):
        """Start real-time inspection"""
        if not self.camera:
            logger.error("Camera not setup")
            return False
        
        if not self.model:
            logger.error("Model not loaded")
            return False
        
        # Start camera streaming
        if not self.camera.start_streaming():
            return False
        
        # Start inspection thread
        self.inspection_thread = threading.Thread(target=self._inspection_loop, daemon=True)
        self.inspection_thread.start()
        
        logger.info("Real-time inspection started")
        return True
    
    def _inspection_loop(self):
        """Main inspection loop"""
        while not self.stop_event.is_set():
            try:
                # Get frame from camera
                frame = self.camera.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                # Process frame
                result = self._process_frame(frame)
                if result:
                    # Add to results queue
                    if self.results_queue.full():
                        try:
                            self.results_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.results_queue.put(result)
                    
                    # Update statistics
                    self._update_stats(result)
                
            except Exception as e:
                logger.error(f"Error in inspection loop: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Optional[InspectionResult]:
        """Process single frame"""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_frame = self._preprocess_image(frame)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(processed_frame.unsqueeze(0).to(self.device))
            
            # Post-process results
            zones = torch.softmax(outputs['zones'], dim=1).cpu().numpy()[0]
            defects = torch.sigmoid(outputs['defects']).cpu().numpy()[0]
            quality_logits = outputs['quality'].cpu().numpy()[0]
            
            # Get quality class
            quality_class_idx = np.argmax(quality_logits)
            quality_classes = ['pass', 'warning', 'fail']
            quality_class = quality_classes[quality_class_idx]
            quality_score = float(np.max(quality_logits))
            
            # Find defect locations
            defect_locations = self._find_defect_locations(defects)
            
            processing_time = time.time() - start_time
            
            return InspectionResult(
                timestamp=datetime.now(),
                image=frame,
                zones=zones,
                defects=defects,
                quality_score=quality_score,
                quality_class=quality_class,
                defect_locations=defect_locations,
                processing_time=processing_time,
                confidence=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to model input size
        image = cv2.resize(image, (512, 512))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor and permute dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def _find_defect_locations(self, defects: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find locations of defects in the image"""
        defect_locations = []
        
        for defect_type in range(defects.shape[0]):
            defect_map = defects[defect_type]
            
            # Find regions with high defect probability
            threshold = 0.5
            defect_regions = defect_map > threshold
            
            if np.any(defect_regions):
                # Find connected components
                from scipy import ndimage
                labeled, num_features = ndimage.label(defect_regions)
                
                for i in range(1, num_features + 1):
                    # Get centroid of defect region
                    coords = np.where(labeled == i)
                    if len(coords[0]) > 0:
                        y, x = int(np.mean(coords[0])), int(np.mean(coords[1]))
                        defect_locations.append((x, y, defect_type))
        
        return defect_locations
    
    def _update_stats(self, result: InspectionResult):
        """Update inspection statistics"""
        self.stats['total_frames'] += 1
        self.stats['processed_frames'] += 1
        
        # Update average processing time
        alpha = 0.9
        self.stats['avg_processing_time'] = (
            alpha * self.stats['avg_processing_time'] + 
            (1 - alpha) * result.processing_time
        )
        
        # Update quality distribution
        self.stats['quality_distribution'][result.quality_class] += 1
        
        # Update defect counts
        for x, y, defect_type in result.defect_locations:
            if defect_type < len(self.stats['defect_counts']):
                self.stats['defect_counts'][defect_type] += 1
    
    def get_latest_result(self) -> Optional[InspectionResult]:
        """Get latest inspection result"""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current inspection statistics"""
        return self.stats.copy()
    
    def stop_inspection(self):
        """Stop real-time inspection"""
        self.stop_event.set()
        if self.camera:
            self.camera.stop_streaming()
        logger.info("Real-time inspection stopped")

class PylonMonitor:
    """Real-time monitoring interface with PYLON integration"""
    
    def __init__(self, inspector: RealTimeInspector):
        self.inspector = inspector
        self.display_window = "Fiber Optic Inspection"
        self.is_running = False
        
        # Create display window
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, 1200, 800)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.is_running = True
        
        while self.is_running:
            try:
                # Get latest result
                result = self.inspector.get_latest_result()
                
                if result:
                    # Display results
                    self._display_result(result)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_result(result)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def _display_result(self, result: InspectionResult):
        """Display inspection result"""
        # Create display image
        display_img = result.image.copy()
        
        # Overlay zone segmentation
        zones_overlay = self._create_zones_overlay(result.zones, display_img.shape[:2])
        display_img = cv2.addWeighted(display_img, 0.7, zones_overlay, 0.3, 0)
        
        # Overlay defect markers
        for x, y, defect_type in result.defect_locations:
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][defect_type % 4]
            cv2.circle(display_img, (x, y), 10, color, 2)
            cv2.putText(display_img, f"D{defect_type}", (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add text overlay
        text_lines = [
            f"Quality: {result.quality_class.upper()} ({result.quality_score:.2f})",
            f"Processing Time: {result.processing_time*1000:.1f}ms",
            f"Defects Found: {len(result.defect_locations)}",
            f"Timestamp: {result.timestamp.strftime('%H:%M:%S')}"
        ]
        
        for i, line in enumerate(text_lines):
            cv2.putText(display_img, line, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display image
        cv2.imshow(self.display_window, display_img)
    
    def _create_zones_overlay(self, zones: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create zone segmentation overlay"""
        overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        
        # Zone colors: Core (red), Cladding (green), Ferrule (blue)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for i, zone in enumerate(zones):
            if i < len(colors):
                mask = (zone > 0.5).astype(np.uint8) * 255
                mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
                mask_colored = np.zeros_like(overlay)
                mask_colored[mask > 0] = colors[i]
                overlay = cv2.add(overlay, mask_colored)
        
        return overlay
    
    def _save_result(self, result: InspectionResult):
        """Save current result"""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"inspection_result_{timestamp}.jpg"
        
        # Save image with overlays
        display_img = result.image.copy()
        zones_overlay = self._create_zones_overlay(result.zones, display_img.shape[:2])
        display_img = cv2.addWeighted(display_img, 0.7, zones_overlay, 0.3, 0)
        
        for x, y, defect_type in result.defect_locations:
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][defect_type % 4]
            cv2.circle(display_img, (x, y), 10, color, 2)
        
        cv2.imwrite(filename, display_img)
        logger.info(f"Result saved as {filename}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False

def main():
    """Main function for PYLON integration"""
    parser = argparse.ArgumentParser(description='PYLON Fiber Optic Inspection')
    parser.add_argument('--model-path', type=str, default='checkpoints/fiber_analysis_model.pth',
                       help='Path to trained model')
    parser.add_argument('--camera-serial', type=str, default=None,
                       help='Camera serial number')
    parser.add_argument('--exposure', type=float, default=10000.0,
                       help='Exposure time (microseconds)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Camera frame rate')
    parser.add_argument('--width', type=int, default=1920,
                       help='Image width')
    parser.add_argument('--height', type=int, default=1080,
                       help='Image height')
    
    args = parser.parse_args()
    
    # Create camera configuration
    camera_config = CameraConfig(
        exposure_time=args.exposure,
        fps=args.fps,
        width=args.width,
        height=args.height
    )
    
    # Create inspector
    inspector = RealTimeInspector(model_path=args.model_path)
    
    # Setup camera
    if not inspector.setup_camera(camera_config):
        logger.error("Failed to setup camera")
        return
    
    # Start inspection
    if not inspector.start_inspection():
        logger.error("Failed to start inspection")
        return
    
    # Create and start monitor
    monitor = PylonMonitor(inspector)
    
    try:
        logger.info("Starting real-time monitoring. Press 'q' to quit, 's' to save result.")
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Stopping inspection...")
    finally:
        inspector.stop_inspection()
        monitor.stop_monitoring()

if __name__ == "__main__":
    main() 