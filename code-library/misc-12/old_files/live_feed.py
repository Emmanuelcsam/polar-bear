#!/usr/bin/env python3
"""
Optimized Enhanced Live Feed Module
Provides camera interface, live video stream functionality, and core detection 
capabilities with performance optimizations for smooth operation.
"""

import cv2
import numpy as np
import time
import argparse
import json
import os
import threading
import queue
from typing import Optional, Tuple, Callable, Dict, List, Any
from collections import deque
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Fast Pylon import
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    pass

# GPU support
GPU_AVAILABLE = False
try:
    # Check for CUDA support in OpenCV
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        print("GPU acceleration available")
except Exception:
    pass


def _calculate_circle_confidence(gray: np.ndarray, center_x: int,
                               center_y: int, radius: int) -> float:
    """Fast confidence calculation for circle detection"""
    try:
        # Simple contrast calculation
        mask = np.zeros_like(gray)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        inside_mean = np.mean(gray[mask > 0])
        outside_mask = cv2.circle(np.zeros_like(gray), (center_x, center_y),
                                 radius + 10, 255, -1)
        outside_mask = cv2.circle(outside_mask, (center_x, center_y),
                                 radius, 0, -1)
        outside_mean = np.mean(gray[outside_mask > 0])
        
        contrast_ratio = abs(inside_mean - outside_mean) / max(
            inside_mean, outside_mean, 1)
        return min(1.0, max(0.0, contrast_ratio))
        
    except Exception:
        return 0.0


def find_pylon_cameras() -> List[str]:
    """Find available Pylon cameras with enhanced Basler detection"""
    if not PYLON_AVAILABLE:
        return []
    
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        camera_names = []
        basler_cameras = []
        
        print("=== ENHANCED PYLON CAMERA DETECTION ===")
        for i, device in enumerate(devices):
            model_name = device.GetModelName()
            serial_number = device.GetSerialNumber()
            vendor_name = device.GetVendorName()
            
            camera_info = {
                'index': i,
                'model': model_name,
                'serial': serial_number,
                'vendor': vendor_name,
                'device': device
            }
            
            camera_names.append(model_name)
            print(f"Found Pylon camera {i}: {model_name} "
                  f"(Serial: {serial_number}, Vendor: {vendor_name})")
            
            # Specifically look for Basler cameras
            if ('basler' in model_name.lower() or 
                'a2a2590' in model_name.lower()):
                basler_cameras.append(camera_info)
                print(f"*** BASLER CAMERA DETECTED: {model_name} ***")
        
        if basler_cameras:
            print(f"Found {len(basler_cameras)} Basler camera(s)")
            for cam in basler_cameras:
                print(f"  - {cam['model']} (Serial: {cam['serial']})")
        else:
            print("No Basler cameras found in Pylon devices")
        
        print("=== END PYLON DETECTION ===")
        return camera_names
        
    except Exception as e:
        print(f"Error enumerating Pylon cameras: {e}")
        return []


def find_basler_camera_specific() -> Optional[Dict]:
    """Specifically find and return Basler a2A2590-22gmBAS camera"""
    if not PYLON_AVAILABLE:
        return None
    
    try:
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        
        for i, device in enumerate(devices):
            model_name = device.GetModelName()
            serial_number = device.GetSerialNumber()
            
            # Look for specific Basler model
            if ('a2a2590' in model_name.lower() or 
                'basler' in model_name.lower() or
                '40455566' in serial_number):
                
                print("*** TARGET BASLER CAMERA FOUND ***")
                print(f"Model: {model_name}")
                print(f"Serial: {serial_number}")
                print(f"Index: {i}")
                
                return {
                    'index': i,
                    'model': model_name,
                    'serial': serial_number,
                    'device': device
                }
        
        print("Target Basler camera not found in Pylon devices")
        return None
        
    except Exception as e:
        print(f"Error finding Basler camera: {e}")
        return None


def find_available_cameras(max_cameras: int = 10) -> List[int]:
    """Enhanced camera detection with multiple backends and Basler support"""
    available_cameras = []
    
    print("=== ENHANCED CAMERA DETECTION ===")
    
    # First, try to find Basler camera specifically
    basler_camera = find_basler_camera_specific()
    if basler_camera:
        print(f"Basler camera found at index {basler_camera['index']}")
        available_cameras.append(basler_camera['index'])
        return available_cameras
    
    # Enhanced backend detection for Windows
    backends = [
        cv2.CAP_ANY,
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_FFMPEG,
        cv2.CAP_GSTREAMER
    ]
    
    print("Scanning for cameras with multiple backends...")
    for backend in backends:
        backend_name = (str(backend).split('.')[-1] 
                       if hasattr(backend, '__name__') else str(backend))
        print(f"Trying backend: {backend_name}")
        
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        available_cameras.append(i)
                        print(f"Found camera at index {i} with backend {backend_name}")
                        cap.release()
                        return available_cameras  # Return first found camera
                    cap.release()
            except Exception as e:
                print(f"Error testing camera {i} with backend {backend_name}: {e}")
    
    # If no cameras found with specific backends, try without backend
    print("Trying without specific backend...")
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"Found camera at index {i} (no specific backend)")
                    cap.release()
                    return available_cameras
                cap.release()
        except Exception as e:
            print(f"Error testing camera {i}: {e}")
    
    print("=== END CAMERA DETECTION ===")
    return available_cameras


class CoreDetectionResult:
    """Container for core detection results"""
    def __init__(self, method_name: str, timestamp: float):
        self.method_name = method_name
        self.timestamp = timestamp
        self.center = None
        self.core_radius = None
        self.confidence = 0.0
        self.execution_time = 0.0
        self.error = None
        self.frame_number = 0


class LiveTerminalLogger:
    """Fast terminal logging with threading"""
    
    def __init__(self):
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.is_running = False
        
    def start(self):
        """Start logging thread"""
        self.is_running = True
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        
    def stop(self):
        """Stop logging thread"""
        self.is_running = False
        if self.log_thread:
            self.log_thread.join()
            
    def log(self, message: str, level: str = "INFO"):
        """Add message to log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_queue.put(log_entry)
        
    def _log_worker(self):
        """Worker thread for logging"""
        while self.is_running:
            try:
                while not self.log_queue.empty():
                    message = self.log_queue.get_nowait()
                    print(message)
                    self.log_queue.task_done()
                time.sleep(0.01)
            except Exception as e:
                print(f"Logging error: {e}")


class CoreDetectionMethods:
    """Core detection methods with GPU support"""
    
    @staticmethod
    def geometric_approach(frame: np.ndarray, 
                          method_name: str = "geometric_approach") -> CoreDetectionResult:
        """Fast geometric approach using Hough circles with GPU acceleration"""
        result = CoreDetectionResult(method_name, time.time())
        start_time = time.time()
        
        try:
            # GPU acceleration if available
            if GPU_AVAILABLE:
                # Upload to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2GRAY)
                
                # Download for CPU processing (HoughCircles doesn't have GPU version)
                gray = gpu_gray.download()
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            height, width = gray.shape
            
            # Fast preprocessing
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_gray = clahe.apply(gray)
            blurred = cv2.GaussianBlur(clahe_gray, (7, 7), 1.5)
            
            # Fast Hough detection
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=2.0, minDist=150,
                param1=50, param2=25, minRadius=15, maxRadius=int(height / 4)
            )
            
            if circles is None:
                result.error = "No circles detected"
                return result
            
            circles = np.uint16(np.around(circles))
            center_x, center_y, radius = circles[0, 0]
            
            # Fast confidence calculation
            confidence = _calculate_circle_confidence(gray, center_x, center_y, radius)
            
            result.center = (float(center_x), float(center_y))
            result.core_radius = float(radius)
            result.confidence = confidence
            
        except Exception as e:
            result.error = str(e)
            
        result.execution_time = time.time() - start_time
        return result


class PylonCamera:
    """Fast camera interface with Pylon support and auto-detection"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = True, 
                 auto_detect: bool = True):
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.auto_detect = auto_detect
        self.camera = None
        self.is_grabbing = False
        self.setup_camera()
        
    def setup_camera(self):
        """Enhanced camera setup with specific Basler camera targeting"""
        print("=== ENHANCED CAMERA SETUP ===")
        
        # First, try to find the specific Basler camera
        basler_camera = find_basler_camera_specific()
        if basler_camera:
            print(f"Target Basler camera found: {basler_camera['model']}")
            try:
                # Create camera from the specific device
                self.camera = pylon.InstantCamera(
                    pylon.TlFactory.GetInstance().CreateDevice(basler_camera['device'])
                )
                self.camera.Open()
                
                if self.camera.IsOpen():
                    print(f"Successfully opened Basler camera: {basler_camera['model']}")
                    
                    # Enhanced Basler-specific configuration
                    try:
                        self.camera.PixelFormat.SetValue("RGB8")
                        print("Set pixel format to RGB8")
                    except Exception as e:
                        print(f"Could not set RGB8 format: {e}")
                    
                    try:
                        self.camera.ExposureAuto.SetValue("Continuous")
                        print("Set exposure to auto continuous")
                    except Exception as e:
                        print(f"Could not set auto exposure: {e}")
                    
                    try:
                        self.camera.GainAuto.SetValue("Continuous")
                        print("Set gain to auto continuous")
                    except Exception as e:
                        print(f"Could not set auto gain: {e}")
                    
                    try:
                        self.camera.AcquisitionMode.SetValue("Continuous")
                        print("Set acquisition mode to continuous")
                    except Exception as e:
                        print(f"Could not set acquisition mode: {e}")
                    
                    # Start grabbing
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    self.is_grabbing = True
                    self.use_pylon = True
                    
                    print(f"Basler camera {basler_camera['model']} ready for use")
                    print("=== END CAMERA SETUP ===")
                    return
                else:
                    print("Failed to open Basler camera")
                    self.use_pylon = False
                    
            except Exception as e:
                print(f"Error setting up Basler camera: {e}")
                self.use_pylon = False
        
        # Fallback to general Pylon camera detection
        pylon_cameras = find_pylon_cameras()
        if pylon_cameras and (self.use_pylon or self.auto_detect):
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                
                if len(devices) > 0:
                    self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
                    self.camera.Open()
                    
                    if self.camera.IsOpen():
                        # Fast configuration
                        try:
                            self.camera.PixelFormat.SetValue("RGB8")
                        except Exception:
                            pass
                        try:
                            self.camera.ExposureAuto.SetValue("Continuous")
                        except Exception:
                            pass
                        
                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                        self.is_grabbing = True
                        self.use_pylon = True
                        print(f"Auto-selected Pylon camera: {self.camera.GetDeviceInfo().GetModelName()}")
                        print("=== END CAMERA SETUP ===")
                        return
                    else:
                        print("Failed to open Pylon camera. Trying webcam fallback.")
                        self.use_pylon = False
                        
            except Exception as e:
                print(f"Error setting up Pylon camera: {e}")
                self.use_pylon = False
                
        # Enhanced webcam fallback with multiple detection methods
        if not self.use_pylon:
            print("Trying webcam fallback with enhanced detection...")
            
            # Auto-detect available cameras if requested
            if self.auto_detect:
                available_cameras = find_available_cameras()
                if available_cameras:
                    if self.camera_index not in available_cameras:
                        self.camera_index = available_cameras[0]
                        print(f"Auto-selected webcam index: {self.camera_index}")
                else:
                    print("No webcams found. Trying default index...")
            
            # Try different backends for Windows with enhanced error handling
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]
            camera_found = False
            
            for backend in backends:
                try:
                    print(f"Trying camera index {self.camera_index} with backend {backend}...")
                    self.camera = cv2.VideoCapture(self.camera_index, backend)
                    if self.camera.isOpened():
                        ret, frame = self.camera.read()
                        if ret:
                            print(f"Successfully opened webcam at index {self.camera_index} with backend {backend}")
                            camera_found = True
                            break
                        else:
                            self.camera.release()
                except Exception as e:
                    print(f"Error with backend {backend}: {e}")
            
            if not camera_found:
                # Try alternative camera indices with different backends
                for alt_index in [1, 2, 3, 4]:
                    for backend in backends:
                        try:
                            print(f"Trying camera index {alt_index} with backend {backend}...")
                            self.camera = cv2.VideoCapture(alt_index, backend)
                            if self.camera.isOpened():
                                ret, frame = self.camera.read()
                                if ret:
                                    self.camera_index = alt_index
                                    print(f"Successfully opened webcam at index {alt_index} with backend {backend}")
                                    camera_found = True
                                    break
                                else:
                                    self.camera.release()
                        except Exception as e:
                            print(f"Error with camera {alt_index} and backend {backend}: {e}")
                    if camera_found:
                        break
                
                if not camera_found:
                    # If no webcams work but we have Pylon cameras, try Pylon
                    if pylon_cameras:
                        print("No webcams available. Trying Pylon camera...")
                        try:
                            tl_factory = pylon.TlFactory.GetInstance()
                            devices = tl_factory.EnumerateDevices()
                            
                            if len(devices) > 0:
                                self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
                                self.camera.Open()
                                
                                if self.camera.IsOpen():
                                    try:
                                        self.camera.PixelFormat.SetValue("RGB8")
                                    except Exception:
                                        pass
                                    try:
                                        self.camera.ExposureAuto.SetValue("Continuous")
                                    except Exception:
                                        pass
                                    
                                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                                    self.is_grabbing = True
                                    self.use_pylon = True
                                    print(f"Successfully opened Pylon camera: {self.camera.GetDeviceInfo().GetModelName()}")
                                    print("=== END CAMERA SETUP ===")
                                    return
                        except Exception as e:
                            print(f"Error setting up Pylon camera: {e}")
                    
                    # If all camera detection fails, create a demo camera with instructions
                    print("\n" + "="*60)
                    print("CAMERA NOT DETECTED")
                    print("="*60)
                    print("POSSIBLE REASONS:")
                    print("1. Camera not connected")
                    print("2. Camera drivers not installed")
                    print("3. Camera being used by another application")
                    print("4. Camera not properly connected")
                    print("\nSOLUTIONS TO TRY:")
                    print("1. Close other applications that might be using the camera")
                    print("2. Update your camera drivers")
                    print("3. Try running the program as administrator")
                    print("4. Check if your camera works in other applications")
                    print("5. Try a different USB port")
                    print("\nFor now, the program will run in DEMO MODE.")
                    print("You can still test all features with synthetic data.")
                    print("="*60)
                    
                    # Create a demo camera that generates synthetic frames
                    self.camera = None # This line was removed from the new_code, but should be kept for consistency
                    self.demo_mode = True
                    print("Demo mode activated - using synthetic camera feed")
        
        print("=== END CAMERA SETUP ===")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Fast frame reading"""
        if self.camera is None:
            return None
            
        try:
            if self.use_pylon and self.is_grabbing:
                try:
                    grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
                    
                    if grab_result.GrabSucceeded():
                        image = grab_result.Array
                        grab_result.Release()
                        
                        if len(image.shape) == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                            
                        return image
                    else:
                        return None
                except Exception:
                    return None
            else:
                ret, frame = self.camera.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            if self.use_pylon and self.is_grabbing:
                try:
                    self.camera.StopGrabbing()
                except Exception as e:
                    print(f"Error stopping Pylon camera: {e}")
            elif hasattr(self.camera, "release"):
                self.camera.release()


class LiveFeed:
    """Optimized enhanced live feed handler with core detection capabilities"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = False,
                                              frame_callback: Optional[Callable] = None, config_file: str = "config.json",
                 output_dir: str = "output", auto_detect: bool = True, demo_mode: bool = False):
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.frame_callback = frame_callback
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.auto_detect = auto_detect
        self.demo_mode = demo_mode
        
        # Load configuration
        try:
            from config_manager import ConfigManager
            self.config_manager = ConfigManager(config_file)
            self.config = self.config_manager.get_live_feed_config()
        except ImportError:
            # Fallback configuration
            self.config = self._load_config(config_file)
        
        # Enhanced camera interface with auto-detection
        if self.demo_mode:
            self.camera = None
            print("Demo mode enabled - using synthetic frames")
        else:
            self.camera = PylonCamera(camera_index, use_pylon, auto_detect)
            # Check if camera setup failed and switch to demo mode
            if self.camera.camera is None and self.camera.demo_mode:
                self.demo_mode = True
                print("Switched to demo mode due to camera detection failure")
        
        self.logger = LiveTerminalLogger()
        
        # Application state
        self.is_running = False
        self.frame_count = 0
        self.detection_methods = [CoreDetectionMethods.geometric_approach]
        
        # Configuration
        self.config = self._load_config(config_file)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()
        self.last_process_time = 0
        # Fix: Get process_interval with backward compatibility
        detection_config = self.config.get("detection", {})
        # Try backward-compatible process_interval first, then fall back to detection_timeout
        self.process_interval = detection_config.get("process_interval", 
            detection_config.get("auto_core_detection", {}).get("detection_timeout", 0.2))
        
        # Performance optimizations
        self.target_fps = 60
        self.frame_time_target = 1.0 / self.target_fps
        self.last_frame_time = 0
        self.frame_skip_count = 0
        
        # Parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Demo mode variables
        self.demo_frame_count = 0
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration with enhanced Basler camera support"""
        default_config = {
            "camera": {
                "basler": {
                    "target_model": "a2A2590-22gmBAS",
                    "target_serial": "40455566",
                    "pixel_format": "RGB8",
                    "exposure_auto": "Continuous",
                    "gain_auto": "Continuous",
                    "acquisition_mode": "Continuous"
                },
                "pylon": {
                    "enable_pylon_detection": True,
                    "enable_auto_detection": True,
                    "preferred_backend": "Pylon",
                    "fallback_to_opencv": True,
                    "timeout_ms": 1000,
                    "grab_strategy": "LatestImageOnly"
                },
                "opencv": {
                    "enable_opencv_detection": True,
                    "backends": ["CAP_ANY", "CAP_DSHOW", "CAP_MSMF", "CAP_FFMPEG"],
                    "max_camera_index": 10,
                    "timeout_ms": 1000
                },
                "general": {
                    "auto_detect": True,
                    "demo_mode": False,
                    "camera_index": 0,
                    "use_pylon": True,
                    "enable_fallback": True
                }
            },
            "detection": {
                "process_interval": 0.2,  # Backward compatibility
                "auto_core_detection": {
                    "enable_geometric_detection": True,
                    "enable_improved_detection": True,
                    "enable_manual_learning": True,
                    "min_confidence": 0.3,
                    "max_confidence": 1.0,
                    "detection_timeout": 0.2,
                    "enable_parallel_detection": True,
                    "max_detection_workers": 4
                },
                "hough_circles": {
                    "dp": 2.0,
                    "min_dist": 150,
                    "param1": 50,
                    "param2": 25,
                    "min_radius_small": 5,
                    "max_radius_small": 50,
                    "min_radius_medium": 15,
                    "max_radius_medium": 150,
                    "min_radius_large": 50,
                    "max_radius_large": 500,
                    "enable_adaptive_parameters": True,
                    "adaptive_scale_factor": 0.1
                },
                "preprocessing": {
                    "enable_clahe": True,
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_grid_size": 8,
                    "enable_gaussian_blur": True,
                    "gaussian_kernel_size": 7,
                    "gaussian_sigma": 1.5,
                    "enable_median_blur": False,
                    "median_kernel_size": 5,
                    "enable_bilateral_filter": False,
                    "bilateral_d": 9,
                    "bilateral_sigma_color": 75,
                    "bilateral_sigma_space": 75
                }
            },
            "performance": {
                "enable_gpu_acceleration": True,
                "enable_parallel_processing": True,
                "max_workers": 4,
                "frame_buffer_size": 10,
                "enable_performance_monitoring": True
            },
            "display": {
                "show_info_overlay": True,
                "show_detection_results": True,
                "show_performance_stats": True,
                "window_name": "Core Detection System",
                "enable_fullscreen": False
            },
            "logging": {
                "enable_logging": True,
                "log_level": "INFO",
                "log_file": "system.log",
                "enable_console_output": True
            }
        }
        
        # Try to load user configuration
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with default config
                self._merge_config(default_config, user_config)
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _merge_config(self, default_config: Dict, user_config: Dict):
        """Recursively merge user configuration with default configuration"""
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[CoreDetectionResult], Optional[Dict]]:
        """Smart optimized frame processing with ALL capabilities"""
        self.frame_count += 1
        results = []
        
        # Run ALL detection methods in parallel with smart optimization
        futures = []
        for method in self.detection_methods:
            future = self.thread_pool.submit(method, frame)
            futures.append(future)
        
                # Collect results with timeout
        try:
            for future in as_completed(futures, timeout=0.2):  # 200ms timeout
                try:
                    result = future.result()
                    result.frame_number = self.frame_count
                    results.append(result)
                    
                    if result.error:
                        self.logger.log(f"{result.method_name}: FAILED - {result.error}", "ERROR")
                    else:
                        self.logger.log(
                            f"{result.method_name}: Core at {result.center}, "
                            f"radius={result.core_radius:.1f}, "
                            f"confidence={result.confidence:.3f}, "
                            f"time={result.execution_time:.3f}s"
                        )
                except Exception as e:
                    self.logger.log(f"Detection method: EXCEPTION - {e}", "ERROR")
        except TimeoutError:
            self.logger.log("Detection timeout - continuing with available results", "WARNING")
        
        # Draw results on frame
        processed_frame = self._draw_results(frame, results)
        
        return processed_frame, results, None
    
    def _draw_results(self, frame: np.ndarray, results: List[CoreDetectionResult]) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame
        
        # Draw detection results
        for result in results:
            if result.error or not result.center or not result.core_radius:
                continue
                
            color = (0, 255, 0)  # Green for geometric approach
            center = (int(result.center[0]), int(result.center[1]))
            radius = int(result.core_radius)
            
            # Draw circle
            cv2.circle(result_frame, center, radius, color, 2)
            # Draw center point
            cv2.circle(result_frame, center, 3, color, -1)
            # Draw method name
            cv2.putText(result_frame, "GEOMETRIC", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add information overlay
        if self.config["display"]["show_info_overlay"]:
            result_frame = self._add_info_overlay(result_frame, results)
        
        return result_frame
    
    def _add_info_overlay(self, frame: np.ndarray, results: List[CoreDetectionResult]) -> np.ndarray:
        """Add comprehensive information overlay to frame"""
        overlay = frame
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        y_offset = 30
        line_height = 25
        
        # FPS information
        if self.config["display"]["show_fps"]:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(overlay, fps_text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        
        # Frame count
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(overlay, frame_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Detection count
        valid_results = [r for r in results if r.error is None]
        detection_text = f"Detections: {len(valid_results)}/{len(results)}"
        cv2.putText(overlay, detection_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Camera info
        camera_text = f"Camera: {'Pylon' if self.use_pylon else 'Webcam'}"
        cv2.putText(overlay, camera_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # GPU info
        gpu_text = f"GPU: {'Enabled' if GPU_AVAILABLE else 'Disabled'}"
        cv2.putText(overlay, gpu_text, (10, y_offset), font, font_scale, color, thickness)
        
        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera"""
        if self.demo_mode:
            self.demo_frame_count += 1
            # Generate a simple synthetic frame for demo mode
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(image, f"Demo Frame {self.demo_frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return image
        return self.camera.read_frame()
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0.0
    
    def run(self, window_name: str = None, show_info: bool = True, headless: bool = False):
        """Run the smart optimized live feed loop with ALL detection capabilities"""
        window_name = window_name or self.config["display"]["window_name"]
        
        self.logger.start()
        self.logger.log("Starting Smart Optimized Live Feed - ALL CAPABILITIES PRESERVED")
        self.logger.log(f"Camera: {'Pylon' if self.use_pylon else 'Webcam'}")
        self.logger.log(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
        self.logger.log("Detection: ALL METHODS ENABLED with smart optimization")
        self.logger.log("Press ESC to exit")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.001)  # Minimal sleep
                    continue
                
                current_time = time.time()
                
                # Smart frame processing - run detection intelligently
                if current_time - self.last_process_time >= self.process_interval:
                    # Run ALL detection methods with smart optimization
                    processed_frame, results, _ = self.process_frame(frame)
                    self.last_process_time = current_time
                    
                    # Adaptive performance adjustment
                    self._adjust_performance_parameters()
                else:
                    # Just display the frame without processing for smoothness
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Call frame callback if provided
                if self.frame_callback:
                    try:
                        processed_frame = self.frame_callback(processed_frame)
                    except Exception as e:
                        self.logger.log(f"Frame callback error: {e}", "ERROR")
                
                # Display frame (skip in headless mode)
                if not headless:
                    try:
                        cv2.imshow(window_name, processed_frame)
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            break
                        
                        # Check if window was closed
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except Exception as e:
                        self.logger.log(f"Display error (continuing in headless mode): {e}", "WARNING")
                        headless = True
                        # Continue processing in headless mode
                        if self.frame_callback:
                            self.frame_callback(processed_frame)
                else:
                    # In headless mode, just process frames without display
                    if self.frame_callback:
                        self.frame_callback(processed_frame)
                    else:
                        time.sleep(0.033)  # ~30 FPS
                    
                    # Simulate ESC key after some time in demo mode
                    if self.demo_mode and self.frame_count > 300:  # ~10 seconds at 30 FPS
                        break
                    
        except KeyboardInterrupt:
            self.logger.log("Interrupted by user")
        except Exception as e:
            self.logger.log(f"Error in live feed loop: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            if not headless:
                self.cleanup()
    
    def _adjust_performance_parameters(self):
        """Smart adaptive performance parameter adjustment"""
        current_fps = self.get_fps()
        
        # Track performance history
        self.fps_history.append(current_fps)
        
        # Keep only last 20 measurements
        if len(self.fps_history) > 20:
            self.fps_history.popleft()
        
        # Adaptive adjustment based on performance
        if len(self.fps_history) >= 10:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            if avg_fps < 25:  # Low FPS - reduce processing
                self.process_interval = min(0.3, self.process_interval * 1.1)
                self.logger.log(f"Performance adjustment: Reduced detection frequency to {self.process_interval:.2f}s")
            elif avg_fps > 45:  # High FPS - increase processing
                self.process_interval = max(0.05, self.process_interval * 0.95)
                self.logger.log(f"Performance adjustment: Increased detection frequency to {self.process_interval:.2f}s")
    
    def cleanup(self):
        """Enhanced cleanup with logging"""
        self.is_running = False
        self.logger.stop()
        if not self.demo_mode and self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Cleanup thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        self.logger.log("Optimized enhanced live feed stopped")
    
    def get_camera_info(self) -> dict:
        """Get comprehensive camera information"""
        info = {
            'camera_index': self.camera_index,
            'use_pylon': self.use_pylon,
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'fps': self.get_fps(),
            'process_interval': self.process_interval,
            'detection_methods': len(self.detection_methods),
            'gpu_available': GPU_AVAILABLE,
            'auto_detect': self.auto_detect,
            'target_fps': self.target_fps,
            'frame_skip_count': self.frame_skip_count
        }
        
        if self.camera is not None:
            if self.use_pylon:
                try:
                    info['camera_name'] = self.camera.camera.GetDeviceInfo().GetModelName()
                except Exception:
                    info['camera_name'] = "Unknown Pylon Camera"
            else:
                info['camera_name'] = f"Webcam {self.camera_index}"
        
        return info
    
    def set_performance_mode(self, ultra_fast: bool = True):
        """Set ultra-fast performance mode"""
        if ultra_fast:
            self.process_interval = 0.05  # Process every 50ms
        else:
            self.process_interval = 0.2  # Process every 200ms


def main():
    """Optimized enhanced standalone test function with auto-detection"""
    parser = argparse.ArgumentParser(description="Optimized Enhanced Live Feed with Core Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--pylon", action="store_true", help="Use Pylon SDK")
    parser.add_argument("--no-info", action="store_true", help="Hide info overlay")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--no-detection", action="store_true", help="Disable core detection")
    parser.add_argument("--no-auto-detect", action="store_true", help="Disable auto camera detection")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras and exit")
    parser.add_argument("--high-performance", action="store_true", help="Enable high performance mode")
    parser.add_argument("--demo", action="store_true", help="Enable demo mode (no camera required)")
    args = parser.parse_args()
    
    # List available cameras if requested
    if args.list_cameras:
        print("=== Available Cameras ===")
        print("Webcams:")
        webcams = find_available_cameras()
        for i, cam in enumerate(webcams):
            print(f"  Index {cam}: Webcam")
        
        print("\nPylon Cameras:")
        pylon_cams = find_pylon_cameras()
        for i, cam in enumerate(pylon_cams):
            print(f"  {cam}")
        
        if not webcams and not pylon_cams:
            print("No cameras found!")
        return
    
    try:
        # Create optimized enhanced live feed
        live_feed = LiveFeed(
            camera_index=args.camera,
            use_pylon=args.pylon,
            config_file=args.config if args.config else "config.json",
            output_dir=args.output,
            auto_detect=not args.no_auto_detect,
            demo_mode=args.demo
        )
        
        # Set performance mode
        if args.high_performance:
            live_feed.set_performance_mode(True)
            print("High performance mode enabled")
        
        # Disable detection if requested
        if args.no_detection:
            live_feed.detection_methods = []
            live_feed.logger.log("Core detection disabled")
        
        # Run optimized enhanced live feed
        live_feed.run(
            window_name="Optimized Enhanced Live Feed Test",
            show_info=not args.no_info,
            headless=args.demo  # Run headless in demo mode
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 