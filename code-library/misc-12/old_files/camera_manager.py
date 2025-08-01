#!/usr/bin/env python3
"""
Comprehensive Camera Manager
Handles camera detection, setup, and management with Windows compatibility
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Pylon import with proper error handling
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("✓ Pylon SDK available")
except ImportError:
    print("⚠ Pylon SDK not available - using OpenCV fallback")


class CameraManager:
    """Comprehensive camera management with Windows compatibility"""
    
    def __init__(self):
        self.camera = None
        self.camera_info = {}
        self.is_demo_mode = False
        self.backend_preferences = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,   # Media Foundation (Windows)
            cv2.CAP_ANY,    # Auto-detect
            cv2.CAP_FFMPEG  # FFmpeg
        ]
        
    def detect_cameras(self) -> List[Dict]:
        """Comprehensive camera detection for Windows"""
        print("=== COMPREHENSIVE CAMERA DETECTION ===")
        available_cameras = []
        
        # Try Pylon cameras first
        if PYLON_AVAILABLE:
            pylon_cameras = self._detect_pylon_cameras()
            available_cameras.extend(pylon_cameras)
        
        # Try OpenCV cameras
        opencv_cameras = self._detect_opencv_cameras()
        available_cameras.extend(opencv_cameras)
        
        print(f"Found {len(available_cameras)} camera(s)")
        return available_cameras
    
    def _detect_pylon_cameras(self) -> List[Dict]:
        """Detect Pylon cameras"""
        cameras = []
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            for i, device in enumerate(devices):
                camera_info = {
                    'index': i,
                    'type': 'pylon',
                    'model': device.GetModelName(),
                    'serial': device.GetSerialNumber(),
                    'vendor': device.GetVendorName(),
                    'device': device
                }
                cameras.append(camera_info)
                print(f"✓ Pylon camera {i}: {camera_info['model']}")
                
        except Exception as e:
            print(f"⚠ Pylon detection error: {e}")
        
        return cameras
    
    def _detect_opencv_cameras(self) -> List[Dict]:
        """Detect OpenCV cameras with multiple backends"""
        cameras = []
        max_index = 5  # Check first 5 indices
        
        for backend in self.backend_preferences:
            for index in range(max_index):
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        camera_info = {
                            'index': index,
                            'type': 'opencv',
                            'backend': backend,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'model': f"Camera {index}",
                            'serial': f"CV_{index}",
                            'vendor': 'OpenCV'
                        }
                        cameras.append(camera_info)
                        print(f"✓ OpenCV camera {index} (backend {backend}): {width}x{height} @ {fps}fps")
                        
                        cap.release()
                        break  # Found camera at this index, try next index
                        
                except Exception as e:
                    continue
        
        return cameras
    
    def setup_camera(self, camera_info: Optional[Dict] = None) -> bool:
        """Setup camera with fallback to demo mode"""
        if camera_info is None:
            cameras = self.detect_cameras()
            if not cameras:
                print("❌ No cameras detected - switching to demo mode")
                self.is_demo_mode = True
                return True
            
            camera_info = cameras[0]  # Use first available camera
        
        try:
            if camera_info['type'] == 'pylon':
                return self._setup_pylon_camera(camera_info)
            else:
                return self._setup_opencv_camera(camera_info)
                
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            print("🔄 Switching to demo mode")
            self.is_demo_mode = True
            return True
    
    def _setup_pylon_camera(self, camera_info: Dict) -> bool:
        """Setup Pylon camera"""
        try:
            device = camera_info['device']
            camera = pylon.InstantCamera()
            camera.Attach(device)
            camera.Open()
            
            # Configure camera settings
            camera.ExposureAuto.SetValue("Continuous")
            camera.GainAuto.SetValue("Continuous")
            camera.AcquisitionMode.SetValue("Continuous")
            
            self.camera = camera
            self.camera_info = camera_info
            print(f"✓ Pylon camera setup successful: {camera_info['model']}")
            return True
            
        except Exception as e:
            print(f"❌ Pylon camera setup failed: {e}")
            return False
    
    def _setup_opencv_camera(self, camera_info: Dict) -> bool:
        """Setup OpenCV camera"""
        try:
            cap = cv2.VideoCapture(camera_info['index'], camera_info['backend'])
            
            if not cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Configure camera settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.camera = cap
            self.camera_info = camera_info
            print(f"✓ OpenCV camera setup successful: {camera_info['model']}")
            return True
            
        except Exception as e:
            print(f"❌ OpenCV camera setup failed: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera or generate demo frame"""
        if self.is_demo_mode:
            return self._generate_demo_frame()
        
        try:
            if self.camera_info['type'] == 'pylon':
                return self._read_pylon_frame()
            else:
                return self._read_opencv_frame()
                
        except Exception as e:
            print(f"⚠ Frame read error: {e}")
            return None
    
    def _read_pylon_frame(self) -> Optional[np.ndarray]:
        """Read frame from Pylon camera"""
        try:
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = grab_result.Array
                grab_result.Release()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return None
        except Exception:
            return None
    
    def _read_opencv_frame(self) -> Optional[np.ndarray]:
        """Read frame from OpenCV camera"""
        try:
            ret, frame = self.camera.read()
            if ret:
                return frame
            return None
        except Exception:
            return None
    
    def _generate_demo_frame(self) -> np.ndarray:
        """Generate synthetic demo frame"""
        # Create a synthetic frame with a moving circle
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some background pattern
        for i in range(0, 640, 50):
            cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
        for i in range(0, 480, 50):
            cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)
        
        # Add a moving circle
        t = time.time()
        center_x = int(320 + 100 * np.sin(t * 0.5))
        center_y = int(240 + 80 * np.cos(t * 0.3))
        radius = 30
        
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Add text
        cv2.putText(frame, "DEMO MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def release(self):
        """Release camera resources"""
        if self.camera:
            if self.camera_info['type'] == 'pylon':
                self.camera.Close()
            else:
                self.camera.release()
            self.camera = None
    
    def get_camera_info(self) -> Dict:
        """Get current camera information"""
        return {
            'is_demo_mode': self.is_demo_mode,
            'camera_info': self.camera_info,
            'is_available': self.camera is not None
        } 