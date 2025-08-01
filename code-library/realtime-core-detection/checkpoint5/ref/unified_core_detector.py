#!/usr/bin/env python3
"""
Unified Core Detector - Merged Automatic and Manual Detection
Combines the best of both approaches with GPU acceleration and fallback support.
Optimized for maximum performance while maintaining all functionality.
"""

import cv2
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Fast Pylon import
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    pass


class ConfigManager:
    """Enhanced configuration manager with performance settings"""
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use optimized defaults"""
        default_config = {
            "camera": {
                "camera_index": 0,
                "use_pylon": True,
                "auto_exposure": True,
                "exposure_time": 10000,
                "gain": 0
            },
            "detection": {
                "min_confidence": 0.3,
                "process_interval": 0.1,
                "enable_parallel_detection": True,
                "max_detection_workers": 2,
                "use_gpu_acceleration": True,
                "enable_automatic_detection": True,
                "enable_manual_overlay": True,
                "detection_method": "hybrid"  # "automatic", "manual", "hybrid"
            },
            "circle_overlay": {
                "initial_center_x": 320,
                "initial_center_y": 240,
                "initial_radius": 50,
                "move_step": 8,
                "resize_step": 5,
                "color_red": 0,
                "color_green": 255,
                "color_blue": 0,
                "thickness": 2,
                "center_point_size": 3,
                "enable_boundary_restrictions": False,
                "max_x": 10000,
                "max_y": 10000,
                "min_x": -10000,
                "min_y": -10000
            },
            "display": {
                "window_name": "Unified Core Detector",
                "show_fps": True,
                "show_detections": True,
                "show_info": True,
                "show_circle_info": True,
                "show_performance_stats": True,
                "show_detection_method": True
            },
            "performance": {
                "enable_performance_tracking": True,
                "frame_time_history_size": 60,
                "target_fps": 120,
                "gpu_memory_optimization": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                self._merge_config(default_config, user_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        return default_config
    
    def _merge_config(self, default_config: Dict, user_config: Dict):
        """Recursively merge user configuration with defaults"""
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def get_config(self, section: str = None) -> Dict:
        """Get configuration section"""
        if section:
            return self.config.get(section, {})
        return self.config


class PylonCamera:
    """Enhanced camera interface with GPU optimization"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = True):
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.camera = None
        self.is_grabbing = False
        self.setup_camera()
        
    def setup_camera(self):
        """Fast camera setup with fallback"""
        if self.use_pylon:
            try:
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                
                if len(devices) == 0:
                    print("No Pylon cameras found. Using webcam fallback.")
                    self.use_pylon = False
                else:
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
                        print(f"Pylon camera initialized: {self.camera.GetDeviceInfo().GetModelName()}")
                    else:
                        print("Failed to open Pylon camera. Using webcam fallback.")
                        self.use_pylon = False
                        
            except Exception as e:
                print(f"Error setting up Pylon camera: {e}")
                self.use_pylon = False
                
        if not self.use_pylon:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open webcam at index {self.camera_index}")
            print(f"Using webcam at index {self.camera_index}")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Fast frame reading with error handling"""
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
        """Release camera resources with error handling"""
        if self.camera is not None:
            try:
                if self.use_pylon and self.is_grabbing:
                    self.camera.StopGrabbing()
                elif hasattr(self.camera, "release"):
                    self.camera.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")


class CoreDetectionResult:
    """Enhanced container for core detection results"""
    def __init__(self, method_name: str, timestamp: float):
        self.method_name = method_name
        self.timestamp = timestamp
        self.center = None
        self.core_radius = None
        self.confidence = 0.0
        self.execution_time = 0.0
        self.error = None
        self.frame_number = 0
        self.detection_type = "unknown"  # "automatic", "manual", "hybrid"


def calculate_circle_confidence(gray: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
    """Fast confidence calculation with GPU optimization"""
    try:
        # Use UMat for GPU acceleration if available
        if hasattr(cv2, 'UMat'):
            gray_umat = cv2.UMat(gray)
            mask = cv2.UMat(np.zeros_like(gray))
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            inside_mean = np.mean(gray_umat.get()[mask.get() > 0])
            outside_mask = cv2.circle(cv2.UMat(np.zeros_like(gray)), (center_x, center_y), radius + 10, 255, -1)
            outside_mask = cv2.circle(outside_mask, (center_x, center_y), radius, 0, -1)
            outside_mean = np.mean(gray_umat.get()[outside_mask.get() > 0])
        else:
            # Fallback to CPU
            mask = np.zeros_like(gray)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            inside_mean = np.mean(gray[mask > 0])
            outside_mask = cv2.circle(np.zeros_like(gray), (center_x, center_y), radius + 10, 255, -1)
            outside_mask = cv2.circle(outside_mask, (center_x, center_y), radius, 0, -1)
            outside_mean = np.mean(gray[outside_mask > 0])
        
        contrast_ratio = abs(inside_mean - outside_mean) / max(inside_mean, outside_mean, 1)
        return min(1.0, max(0.0, contrast_ratio))
        
    except Exception:
        return 0.0


def automatic_detection(frame: Union[np.ndarray, cv2.UMat], manual_center: Tuple[float, float] = None, 
                       manual_radius: float = None, method_name: str = "automatic_approach") -> CoreDetectionResult:
    """Automatic detection with GPU optimization - excludes manual overlay area"""
    result = CoreDetectionResult(method_name, time.time())
    result.detection_type = "automatic"
    start_time = time.time()
    
    try:
        # Convert to UMat for GPU acceleration if available
        if hasattr(cv2, 'UMat') and not isinstance(frame, cv2.UMat):
            frame_umat = cv2.UMat(frame)
        elif isinstance(frame, cv2.UMat):
            frame_umat = frame
        else:
            frame_umat = frame
        
        # GPU-optimized preprocessing
        if hasattr(cv2, 'UMat'):
            gray_umat = cv2.cvtColor(frame_umat, cv2.COLOR_RGB2GRAY)
            height, width = gray_umat.get().shape[:2]
            
            # Fast preprocessing on GPU
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_gray = clahe.apply(gray_umat)
            blurred = cv2.GaussianBlur(clahe_gray, (7, 7), 1.5)
        else:
            # CPU fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_gray = clahe.apply(gray)
            blurred = cv2.GaussianBlur(clahe_gray, (7, 7), 1.5)
        
        # Create mask to exclude manual overlay area
        if manual_center and manual_radius:
            mask = np.ones_like(blurred) * 255
            manual_x, manual_y = int(manual_center[0]), int(manual_center[1])
            manual_r = int(manual_radius)
            
            # Create exclusion zone around manual circle (with some buffer)
            exclusion_radius = manual_r + 20  # Add buffer to avoid edge detection
            
            if hasattr(cv2, 'UMat'):
                mask_umat = cv2.UMat(mask)
                cv2.circle(mask_umat, (manual_x, manual_y), exclusion_radius, 0, -1)
                masked_blurred = cv2.bitwise_and(blurred, blurred, mask=mask_umat)
            else:
                cv2.circle(mask, (manual_x, manual_y), exclusion_radius, 0, -1)
                masked_blurred = cv2.bitwise_and(blurred, blurred, mask=mask)
        else:
            # No manual overlay to exclude
            if hasattr(cv2, 'UMat'):
                masked_blurred = blurred
            else:
                masked_blurred = blurred
        
        # Enhanced edge detection for sharp intensity gradients
        if hasattr(cv2, 'UMat'):
            # Apply additional edge detection for sharp gradients
            edges = cv2.Canny(masked_blurred, 50, 150)
            # Combine with original for better detection
            enhanced = cv2.addWeighted(masked_blurred, 0.7, edges, 0.3, 0)
        else:
            edges = cv2.Canny(masked_blurred, 50, 150)
            enhanced = cv2.addWeighted(masked_blurred, 0.7, edges, 0.3, 0)
        
        # Optimized Hough detection with focus on sharp gradients
        circles = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
            param1=70, param2=20, minRadius=5,
            maxRadius=int(height // 2)
        )
        
        if circles is None:
            result.error = "No other objects detected"
            return result
        
        circles = np.uint16(np.around(circles))
        
        # Find the best circle (highest confidence) that's not the manual overlay
        best_circle = None
        best_confidence = 0.0
        
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            
            # Skip if too close to manual overlay
            if manual_center and manual_radius:
                manual_x, manual_y = int(manual_center[0]), int(manual_center[1])
                manual_r = int(manual_radius)
                distance = np.sqrt((center_x - manual_x)**2 + (center_y - manual_y)**2)
                if distance < (manual_r + radius + 30):  # Buffer zone
                    continue
            
            # Calculate confidence for this circle
            if hasattr(cv2, 'UMat'):
                confidence = calculate_circle_confidence(gray_umat.get(), center_x, center_y, radius)
            else:
                confidence = calculate_circle_confidence(gray, center_x, center_y, radius)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_circle = (center_x, center_y, radius)
        
        if best_circle is None:
            result.error = "No valid objects detected (excluding manual overlay)"
            return result
        
        center_x, center_y, radius = best_circle
        
        result.center = (float(center_x), float(center_y))
        result.core_radius = float(radius)
        result.confidence = best_confidence
        
    except Exception as e:
        result.error = str(e)
        
    result.execution_time = time.time() - start_time
    return result


def manual_detection(frame: Union[np.ndarray, cv2.UMat], manual_center: Tuple[float, float], 
                    manual_radius: float, method_name: str = "manual_approach") -> CoreDetectionResult:
    """Manual detection using provided circle parameters"""
    result = CoreDetectionResult(method_name, time.time())
    result.detection_type = "manual"
    start_time = time.time()
    
    try:
        result.center = manual_center
        result.core_radius = manual_radius
        result.confidence = 1.0  # Manual detection has full confidence
        
    except Exception as e:
        result.error = str(e)
        
    result.execution_time = time.time() - start_time
    return result


class InteractiveCircleOverlay:
    """Enhanced interactive circle overlay with GPU acceleration"""
    
    def __init__(self, config: Dict):
        circle_config = config["circle_overlay"]
        
        # Circle properties
        self.center = [circle_config["initial_center_x"], circle_config["initial_center_y"]]
        self.radius = circle_config["initial_radius"]
        self.color = (circle_config["color_blue"], circle_config["color_green"], circle_config["color_red"])
        self.thickness = circle_config["thickness"]
        self.center_point_size = circle_config["center_point_size"]
        self.is_locked = False
        
        # Movement settings
        self.move_step = circle_config["move_step"]
        self.resize_step = circle_config["resize_step"]
        
        # Boundary settings
        self.enable_boundary_restrictions = circle_config["enable_boundary_restrictions"]
        self.max_x = circle_config["max_x"]
        self.max_y = circle_config["max_y"]
        self.min_x = circle_config["min_x"]
        self.min_y = circle_config["min_y"]
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        self.last_frame_time = time.time()
        
        # Detection mode
        self.detection_mode = "hybrid"  # "automatic", "manual", "hybrid"
        
        # Instructions
        self.instructions = {
            "WASD": "Move circle (W=up, S=down, A=left, D=right)",
            "Q/E": "Resize circle (Q=smaller, E=larger)",
            "L": "Lock/Unlock circle position",
            "R": "Reset circle to center",
            "M": "Toggle detection mode (Auto/Manual/Hybrid)",
            "ESC": "Exit application"
        }
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for circle control"""
        if key == 27:  # ESC key
            return False
        
        # Handle special keys
        if key == ord('l') or key == ord('L'):
            self.is_locked = not self.is_locked
        elif key == ord('r') or key == ord('R'):
            # Reset to initial values
            self.center = [320, 240]  # Default center
            self.radius = 50  # Default radius
        elif key == ord('m') or key == ord('M'):
            # Toggle detection mode
            modes = ["automatic", "manual", "hybrid"]
            current_index = modes.index(self.detection_mode)
            self.detection_mode = modes[(current_index + 1) % len(modes)]
            print(f"Detection mode: {self.detection_mode.upper()}")
        else:
            # Movement keys
            self._apply_movement(key)
        
        return True
    
    def _apply_movement(self, key: int):
        """Apply movement based on key press with no restrictions"""
        if self.is_locked:
            return
        
        # NO BOUNDARY RESTRICTIONS for maximum freedom
        if key in [ord('w'), ord('W')]:
            self.center[1] -= self.move_step
        elif key in [ord('s'), ord('S')]:
            self.center[1] += self.move_step
        elif key in [ord('a'), ord('A')]:
            self.center[0] -= self.move_step
        elif key in [ord('d'), ord('D')]:
            self.center[0] += self.move_step
        elif key in [ord('q'), ord('Q')]:
            new_radius = self.radius - self.resize_step
            if new_radius >= 1:  # Only prevent negative radius
                self.radius = new_radius
        elif key in [ord('e'), ord('E')]:
            self.radius += self.resize_step
            # NO UPPER LIMIT - circle can grow as large as needed
    
    def draw_circle_on_frame(self, frame: Union[np.ndarray, cv2.UMat]) -> Union[np.ndarray, cv2.UMat]:
        """Draw circle overlay on frame with GPU optimization"""
        # Ensure we're working with UMat for GPU acceleration
        if hasattr(cv2, 'UMat') and not isinstance(frame, cv2.UMat):
            result_frame = cv2.UMat(frame)
        elif isinstance(frame, cv2.UMat):
            result_frame = frame
        else:
            result_frame = frame.copy()
        
        # Ensure circle is within frame bounds
        if hasattr(cv2, 'UMat') and isinstance(result_frame, cv2.UMat):
            height, width = result_frame.get().shape[:2]
        else:
            height, width = result_frame.shape[:2]
            
        center_x = max(self.radius, min(width - self.radius, int(self.center[0])))
        center_y = max(self.radius, min(height - self.radius, int(self.center[1])))
        center = (center_x, center_y)
        radius = int(self.radius)
        
        # Draw main circle
        cv2.circle(result_frame, center, radius, self.color, self.thickness)
        
        # Draw center point
        cv2.circle(result_frame, center, self.center_point_size, self.color, -1)
        
        # Draw lock indicator if enabled
        if self.is_locked:
            lock_color = (0, 0, 255)  # Red for locked
            cv2.circle(result_frame, center, radius + 5, lock_color, 1)
        
        # Draw detection mode indicator
        mode_color = (255, 255, 0)  # Yellow for mode indicator
        mode_text = f"MODE: {self.detection_mode.upper()}"
        cv2.putText(result_frame, mode_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Update performance tracking
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        return result_frame
    
    def get_circle_info(self) -> Dict:
        """Get circle information"""
        return {
            'center': tuple(self.center),
            'radius': self.radius,
            'color': self.color,
            'is_locked': self.is_locked,
            'move_step': self.move_step,
            'resize_step': self.resize_step,
            'detection_mode': self.detection_mode
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {'avg_frame_time': 0, 'fps': 0}
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'avg_frame_time': avg_frame_time,
            'fps': fps
        }


class UnifiedCoreDetector:
    """Unified application combining automatic and manual core detection"""
    
    def __init__(self, config_file: str = "config.json"):
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_config()
        
        # Initialize components
        camera_config = self.config["camera"]
        self.camera = PylonCamera(
            camera_index=camera_config["camera_index"],
            use_pylon=camera_config["use_pylon"]
        )
        
        self.circle_overlay = InteractiveCircleOverlay(self.config)
        
        # Application state
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_process_time = 0
        self.process_interval = self.config["detection"]["process_interval"]
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
        # Detection results
        self.last_detection_results = []
        
        # GPU availability check
        self.gpu_available = hasattr(cv2, 'UMat')
        if self.gpu_available:
            print("GPU acceleration enabled")
        else:
            print("GPU acceleration not available, using CPU")
        
    def process_frame(self, frame: Union[np.ndarray, cv2.UMat]) -> Tuple[Union[np.ndarray, cv2.UMat], List[CoreDetectionResult]]:
        """Process frame with hybrid detection"""
        self.frame_count += 1
        results = []
        
        # Convert to UMat for GPU acceleration if available
        if self.gpu_available and not isinstance(frame, cv2.UMat):
            frame_umat = cv2.UMat(frame)
        elif isinstance(frame, cv2.UMat):
            frame_umat = frame
        else:
            frame_umat = frame
        
        # Get manual overlay information for exclusion
        circle_info = self.circle_overlay.get_circle_info()
        manual_center = circle_info['center']
        manual_radius = circle_info['radius']
        
        # Run detection based on mode
        detection_mode = self.circle_overlay.detection_mode
        
        if detection_mode == "automatic":
            # Automatic detection only - exclude manual overlay
            result = automatic_detection(frame_umat, manual_center, manual_radius)
            result.frame_number = self.frame_count
            results.append(result)
            
        elif detection_mode == "manual":
            # Manual detection only
            result = manual_detection(
                frame_umat, 
                manual_center, 
                manual_radius
            )
            result.frame_number = self.frame_count
            results.append(result)
            
        else:  # hybrid mode
            # Both automatic and manual detection
            # Automatic detection excludes manual overlay area
            auto_result = automatic_detection(frame_umat, manual_center, manual_radius, "automatic_approach")
            auto_result.frame_number = self.frame_count
            results.append(auto_result)
            
            # Manual detection uses overlay parameters
            manual_result = manual_detection(
                frame_umat, 
                manual_center, 
                manual_radius,
                "manual_approach"
            )
            manual_result.frame_number = self.frame_count
            results.append(manual_result)
        
        # Store results
        self.last_detection_results = results
        
        return frame_umat, results
    
    def draw_results_on_frame(self, frame: Union[np.ndarray, cv2.UMat], 
                            results: List[CoreDetectionResult]) -> Union[np.ndarray, cv2.UMat]:
        """Draw detection results on frame"""
        result_frame = frame
        
        # Draw detection results
        for result in results:
            if result.error or not result.center or not result.core_radius:
                continue
                
            # Different colors for different detection types
            if result.detection_type == "automatic":
                color = (0, 255, 0)  # Green for automatic
                label = "AUTO"
            elif result.detection_type == "manual":
                color = (0, 255, 0)  # Green for manual
                label = "MANUAL"
            else:
                color = (0, 255, 255)  # Yellow for hybrid
                label = "HYBRID"
            
            center = (int(result.center[0]), int(result.center[1]))
            radius = int(result.core_radius)
            
            # Draw circle
            cv2.circle(result_frame, center, radius, color, 2)
            # Draw center point
            cv2.circle(result_frame, center, 3, color, -1)
            # Draw method name
            cv2.putText(result_frame, label, 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw confidence if available
            if result.confidence > 0:
                conf_text = f"{result.confidence:.2f}"
                cv2.putText(result_frame, conf_text, 
                           (center[0] - 20, center[1] + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_frame
    
    def add_info_overlay(self, frame: Union[np.ndarray, cv2.UMat]) -> Union[np.ndarray, cv2.UMat]:
        """Add information overlay to frame"""
        overlay = frame
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Get frame dimensions
        if hasattr(cv2, 'UMat') and isinstance(overlay, cv2.UMat):
            height, width = overlay.get().shape[:2]
        else:
            height, width = overlay.shape[:2]
        
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
        valid_results = [r for r in self.last_detection_results if r.error is None]
        detection_text = f"Detections: {len(valid_results)}/{len(self.last_detection_results)}"
        cv2.putText(overlay, detection_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Circle information
        if self.config["display"]["show_circle_info"]:
            circle_info = self.circle_overlay.get_circle_info()
            circle_text = f"Circle: ({circle_info['center'][0]:.0f}, {circle_info['center'][1]:.0f}) R:{circle_info['radius']:.0f}"
            cv2.putText(overlay, circle_text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            
            if circle_info['is_locked']:
                lock_text = "LOCKED"
                cv2.putText(overlay, lock_text, (10, y_offset), font, font_scale, (0, 0, 255), thickness)
                y_offset += line_height
        
        # Performance stats
        if self.config["display"]["show_performance_stats"]:
            perf_stats = self.circle_overlay.get_performance_stats()
            perf_text = f"Overlay FPS: {perf_stats['fps']:.1f}"
            cv2.putText(overlay, perf_text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        
        # GPU status
        gpu_text = f"GPU: {'ON' if self.gpu_available else 'OFF'}"
        cv2.putText(overlay, gpu_text, (10, y_offset), font, font_scale, color, thickness)
        
        return overlay
    
    def run(self):
        """Main application loop"""
        print("Starting Unified Core Detector - Merged Automatic and Manual Detection")
        print("=" * 70)
        print("Controls:")
        for control, description in self.circle_overlay.instructions.items():
            print(f"  {control}: {description}")
        print("Press Ctrl+C to stop")
        
        # Create window with error handling
        window_name = self.config["display"]["window_name"]
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"Warning: Could not create window: {e}")
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                current_time = time.time()
                
                # Process frame only at specified intervals
                if current_time - self.last_process_time >= self.process_interval:
                    processed_frame, results = self.process_frame(frame)
                    self.last_process_time = current_time
                else:
                    # Just use the frame without processing
                    if self.gpu_available:
                        processed_frame = cv2.UMat(frame)
                    else:
                        processed_frame = frame
                    results = self.last_detection_results
                
                # Draw detection results
                if self.config["display"]["show_detections"]:
                    processed_frame = self.draw_results_on_frame(processed_frame, results)
                
                # Draw circle overlay
                processed_frame = self.circle_overlay.draw_circle_on_frame(processed_frame)
                
                # Add information overlay
                if self.config["display"]["show_info"]:
                    processed_frame = self.add_info_overlay(processed_frame)
                
                # Convert UMat to regular array for display if needed
                if hasattr(cv2, 'UMat') and isinstance(processed_frame, cv2.UMat):
                    display_frame = processed_frame.get()
                else:
                    display_frame = processed_frame
                
                # Display frame
                try:
                    cv2.imshow(window_name, display_frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.circle_overlay.handle_keyboard_input(key):
                    break
                
                # Check if window is closed
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    # Continue if property not available
                    pass
                    
        except KeyboardInterrupt:
            print("Application interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources with error handling"""
        self.is_running = False
        
        try:
            self.camera.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")
        
        print("Application stopped")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Core Detector - Merged Automatic and Manual Detection"
    )
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run application
        app = UnifiedCoreDetector(config_file=args.config)
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 