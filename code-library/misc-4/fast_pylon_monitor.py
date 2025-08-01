#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-Performance PYLON Real-time Monitor for Fiber Optic Inspection
Optimized for maximum speed and real-time processing
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
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Performance optimization
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available for performance optimization")

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
class FastCameraConfig:
    """Optimized camera configuration for high-speed inspection"""
    exposure_time: float = 5000.0  # Reduced for faster capture
    gain: float = 0.0
    pixel_format: str = "RGB8"
    width: int = 1280  # Reduced for faster processing
    height: int = 720
    fps: float = 60.0  # Higher frame rate
    trigger_mode: str = "Off"
    auto_exposure: bool = False  # Manual for consistency
    auto_gain: bool = False
    buffer_size: int = 5  # Smaller buffer for lower latency

@dataclass
class FastInspectionResult:
    """Optimized inspection result"""
    timestamp: datetime
    image: np.ndarray
    quality_score: float
    quality_class: str
    processing_time: float
    fps: float
    defects_detected: int

class FastPylonCamera:
    """High-performance PYLON camera interface"""
    
    def __init__(self, config: FastCameraConfig):
        self.config = config
        self.camera = None
        self.is_connected = False
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=config.buffer_size)
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
    def connect(self, camera_serial: str = None) -> bool:
        """Connect to PYLON camera with optimized settings"""
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if not devices:
                logger.error("No PYLON cameras found")
                return False
            
            # Select camera
            if camera_serial:
                device = next((d for d in devices if d.GetSerialNumber() == camera_serial), None)
                if not device:
                    logger.error(f"Camera with serial {camera_serial} not found")
                    return False
            else:
                device = devices[0]
                logger.info(f"Using camera: {device.GetFriendlyName()}")
            
            # Create camera with optimized settings
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            self.camera.Open()
            
            # Optimize camera settings for speed
            self._optimize_camera_settings()
            
            self.is_connected = True
            logger.info("Fast camera connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            return False
    
    def _optimize_camera_settings(self):
        """Optimize camera settings for maximum speed"""
        try:
            # Set pixel format
            self.camera.PixelFormat.SetValue(self.config.pixel_format)
            
            # Set resolution
            self.camera.Width.SetValue(self.config.width)
            self.camera.Height.SetValue(self.config.height)
            
            # Optimize exposure and gain for speed
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureTime.SetValue(self.config.exposure_time)
            self.camera.GainAuto.SetValue("Off")
            self.camera.Gain.SetValue(self.config.gain)
            
            # Disable trigger for continuous streaming
            self.camera.TriggerMode.SetValue("Off")
            
            # Set high frame rate
            if hasattr(self.camera, 'AcquisitionFrameRate'):
                self.camera.AcquisitionFrameRate.SetValue(self.config.fps)
            
            # Optimize buffer settings
            if hasattr(self.camera, 'MaxNumBuffer'):
                self.camera.MaxNumBuffer.SetValue(self.config.buffer_size)
            
            # Disable unnecessary features for speed
            if hasattr(self.camera, 'AutoFunctionROIEnable'):
                self.camera.AutoFunctionROIEnable.SetValue(False)
            
            logger.info("Camera optimized for high-speed operation")
            
        except Exception as e:
            logger.error(f"Failed to optimize camera settings: {e}")
    
    def start_streaming(self):
        """Start high-speed camera streaming"""
        if not self.is_connected:
            return False
        
        try:
            # Use latest image only strategy for minimum latency
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_streaming = True
            
            # Start optimized acquisition thread
            self.acquisition_thread = threading.Thread(target=self._fast_acquisition_loop, daemon=True)
            self.acquisition_thread.start()
            
            logger.info("High-speed camera streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def _fast_acquisition_loop(self):
        """Optimized frame acquisition loop"""
        while not self.stop_event.is_set() and self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Fast image conversion
                    image = grab_result.Array
                    
                    # Optimize image format
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Update FPS calculation
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                    
                    # Fast queue management
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(image)
                
                grab_result.Release()
                
            except Exception as e:
                logger.error(f"Error in fast acquisition loop: {e}")
                time.sleep(0.01)  # Minimal sleep
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get latest frame with minimal timeout"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current camera FPS"""
        return self.current_fps
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.stop_event.set()
        if self.camera and self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.is_streaming = False
    
    def disconnect(self):
        """Disconnect camera"""
        self.stop_streaming()
        if self.camera:
            self.camera.Close()
        self.is_connected = False

class FastInspector:
    """High-performance real-time inspector"""
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.camera = None
        self.results_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.processing_times = []
        self.fps_history = []
        self.quality_scores = []
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load and optimize model for inference"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
            
            self.model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Optimize for inference
            if hasattr(torch, 'jit'):
                self.model = torch.jit.script(self.model)
            
            logger.info(f"Optimized model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def setup_camera(self, config: FastCameraConfig) -> bool:
        """Setup high-speed camera"""
        try:
            self.camera = FastPylonCamera(config)
            return self.camera.connect()
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            return False
    
    def start_inspection(self):
        """Start high-speed inspection"""
        if not self.camera or not self.model:
            return False
        
        if not self.camera.start_streaming():
            return False
        
        # Start inspection thread
        self.inspection_thread = threading.Thread(target=self._fast_inspection_loop, daemon=True)
        self.inspection_thread.start()
        
        logger.info("High-speed inspection started")
        return True
    
    def _fast_inspection_loop(self):
        """Optimized inspection loop"""
        while not self.stop_event.is_set():
            try:
                frame = self.camera.get_frame(timeout=0.01)
                if frame is None:
                    continue
                
                # Fast processing
                result = self._fast_process_frame(frame)
                if result:
                    if self.results_queue.full():
                        try:
                            self.results_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.results_queue.put(result)
                    
                    # Update performance metrics
                    self.processing_times.append(result.processing_time)
                    self.fps_history.append(result.fps)
                    self.quality_scores.append(result.quality_score)
                    
                    # Keep only recent history
                    if len(self.processing_times) > 100:
                        self.processing_times = self.processing_times[-100:]
                        self.fps_history = self.fps_history[-100:]
                        self.quality_scores = self.quality_scores[-100:]
                
            except Exception as e:
                logger.error(f"Error in fast inspection loop: {e}")
                time.sleep(0.01)
    
    def _fast_process_frame(self, frame: np.ndarray) -> Optional[FastInspectionResult]:
        """Fast frame processing"""
        start_time = time.time()
        
        try:
            # Fast preprocessing
            processed_frame = self._fast_preprocess(frame)
            
            # Fast inference
            with torch.no_grad():
                outputs = self.model(processed_frame.unsqueeze(0).to(self.device))
            
            # Fast post-processing
            quality_logits = outputs['quality'].cpu().numpy()[0]
            quality_class_idx = np.argmax(quality_logits)
            quality_classes = ['pass', 'warning', 'fail']
            quality_class = quality_classes[quality_class_idx]
            quality_score = float(np.max(quality_logits))
            
            # Count defects quickly
            defects = torch.sigmoid(outputs['defects']).cpu().numpy()[0]
            defects_detected = np.sum(defects > 0.5)
            
            processing_time = time.time() - start_time
            fps = self.camera.get_fps() if self.camera else 0.0
            
            return FastInspectionResult(
                timestamp=datetime.now(),
                image=frame,
                quality_score=quality_score,
                quality_class=quality_class,
                processing_time=processing_time,
                fps=fps,
                defects_detected=int(defects_detected)
            )
            
        except Exception as e:
            logger.error(f"Error in fast frame processing: {e}")
            return None
    
    def _fast_preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Fast image preprocessing"""
        # Fast resize
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Fast normalization
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Fast tensor conversion
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def get_latest_result(self) -> Optional[FastInspectionResult]:
        """Get latest result"""
        try:
            return self.results_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0.0,
            'avg_quality_score': np.mean(self.quality_scores) if self.quality_scores else 0.0,
            'total_frames_processed': len(self.processing_times)
        }
    
    def stop_inspection(self):
        """Stop inspection"""
        self.stop_event.set()
        if self.camera:
            self.camera.stop_streaming()

class FastMonitor:
    """High-performance real-time monitor"""
    
    def __init__(self, inspector: FastInspector):
        self.inspector = inspector
        self.display_window = "Fast Fiber Inspection"
        self.is_running = False
        
        # Performance display
        self.display_fps = 30  # Display refresh rate
        self.last_display_time = time.time()
        
        # Create optimized display window
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, 1280, 720)
    
    def start_monitoring(self):
        """Start high-performance monitoring"""
        self.is_running = True
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Control display refresh rate
                if current_time - self.last_display_time >= 1.0 / self.display_fps:
                    result = self.inspector.get_latest_result()
                    if result:
                        self._fast_display_result(result)
                    self.last_display_time = current_time
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_result(result)
                elif key == ord('p'):
                    self._print_performance_stats()
                
            except Exception as e:
                logger.error(f"Error in fast monitoring loop: {e}")
                time.sleep(0.01)
        
        cv2.destroyAllWindows()
    
    def _fast_display_result(self, result: FastInspectionResult):
        """Fast result display"""
        # Create display image
        display_img = result.image.copy()
        
        # Fast text overlay
        text_lines = [
            f"Quality: {result.quality_class.upper()} ({result.quality_score:.2f})",
            f"Processing: {result.processing_time*1000:.1f}ms",
            f"Camera FPS: {result.fps:.1f}",
            f"Defects: {result.defects_detected}",
            f"Time: {result.timestamp.strftime('%H:%M:%S')}"
        ]
        
        # Fast text rendering
        for i, line in enumerate(text_lines):
            cv2.putText(display_img, line, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Quality indicator
        color = (0, 255, 0) if result.quality_class == 'pass' else \
                (0, 255, 255) if result.quality_class == 'warning' else (0, 0, 255)
        
        cv2.circle(display_img, (display_img.shape[1] - 50, 50), 20, color, -1)
        
        # Display
        cv2.imshow(self.display_window, display_img)
    
    def _save_result(self, result: FastInspectionResult):
        """Save current result"""
        if result:
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"fast_inspection_{timestamp}.jpg"
            cv2.imwrite(filename, result.image)
            logger.info(f"Result saved as {filename}")
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        stats = self.inspector.get_performance_stats()
        if stats:
            logger.info("Performance Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value:.3f}")

def main():
    """Main function for high-performance PYLON monitoring"""
    parser = argparse.ArgumentParser(description='High-Performance PYLON Fiber Inspection')
    parser.add_argument('--model-path', type=str, default='checkpoints/fiber_analysis_model.pth',
                       help='Path to trained model')
    parser.add_argument('--camera-serial', type=str, default=None,
                       help='Camera serial number')
    parser.add_argument('--exposure', type=float, default=5000.0,
                       help='Exposure time (microseconds)')
    parser.add_argument('--fps', type=float, default=60.0,
                       help='Camera frame rate')
    parser.add_argument('--width', type=int, default=1280,
                       help='Image width')
    parser.add_argument('--height', type=int, default=720,
                       help='Image height')
    parser.add_argument('--display-fps', type=int, default=30,
                       help='Display refresh rate')
    
    args = parser.parse_args()
    
    # Create optimized camera configuration
    camera_config = FastCameraConfig(
        exposure_time=args.exposure,
        fps=args.fps,
        width=args.width,
        height=args.height
    )
    
    # Create fast inspector
    inspector = FastInspector(model_path=args.model_path)
    
    # Setup camera
    if not inspector.setup_camera(camera_config):
        logger.error("Failed to setup camera")
        return
    
    # Start inspection
    if not inspector.start_inspection():
        logger.error("Failed to start inspection")
        return
    
    # Create and start fast monitor
    monitor = FastMonitor(inspector)
    monitor.display_fps = args.display_fps
    
    try:
        logger.info("Starting high-performance monitoring.")
        logger.info("Controls: 'q'=quit, 's'=save, 'p'=performance stats")
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Stopping inspection...")
    finally:
        inspector.stop_inspection()

if __name__ == "__main__":
    main() 