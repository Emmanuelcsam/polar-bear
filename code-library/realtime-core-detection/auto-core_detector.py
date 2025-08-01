#!/usr/bin/env python3
"""
Unified Core Detector with Interactive Circle Overlay
Combines live core detection and interactive circle overlay in a single 
process. Fixes all OpenCV window errors and provides maximum functionality.
"""

import cv2
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
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
    """Simple configuration manager"""
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or use defaults"""
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
                "max_detection_workers": 2
            },
            "circle_overlay": {
                "initial_center_x": 320,
                "initial_center_y": 240,
                "initial_radius": 50,
                "move_step": 8,
                "resize_step": 5,
                "color_red": 255,
                "color_green": 0,
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
                "show_performance_stats": True
            },
            "performance": {
                "enable_performance_tracking": True,
                "frame_time_history_size": 30,
                "target_fps": 60
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
    """Fast camera interface with error handling"""
    
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


def calculate_circle_confidence(gray: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
    """Fast confidence calculation"""
    try:
        # Simple contrast calculation
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


def geometric_detection(frame: np.ndarray, method_name: str = "geometric_approach") -> CoreDetectionResult:
    """Fast geometric approach for core detection"""
    result = CoreDetectionResult(method_name, time.time())
    start_time = time.time()
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # Fast preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(clahe_gray, (7, 7), 1.5)
        
        # Fast Hough detection
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=2.0, minDist=150,
            param1=50, param2=25, minRadius=15, 
            maxRadius=int(height / 4)
        )
        
        if circles is None:
            result.error = "No circles detected"
            return result
        
        circles = np.uint16(np.around(circles))
        center_x, center_y, radius = circles[0, 0]
        
        # Fast confidence calculation
        confidence = calculate_circle_confidence(gray, center_x, center_y, radius)
        
        result.center = (float(center_x), float(center_y))
        result.core_radius = float(radius)
        result.confidence = confidence
        
    except Exception as e:
        result.error = str(e)
        
    result.execution_time = time.time() - start_time
    return result


class InteractiveCircleOverlay:
    """Interactive circle overlay with error handling"""
    
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
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Instructions
        self.instructions = {
            "WASD": "Move circle (W=up, S=down, A=left, D=right)",
            "Q/E": "Resize circle (Q=smaller, E=larger)",
            "L": "Lock/Unlock circle position",
            "R": "Reset circle to center",
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
            circle_config = self.config_manager.get_config("circle_overlay")
            self.center = [circle_config["initial_center_x"], circle_config["initial_center_y"]]
            self.radius = circle_config["initial_radius"]
        else:
            # Movement keys
            self._apply_movement(key)
        
        return True
    
    def _apply_movement(self, key: int):
        """Apply movement based on key press"""
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
    
    def draw_circle_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw circle overlay on frame"""
        result_frame = frame.copy()
        
        # Ensure circle is within frame bounds
        height, width = frame.shape[:2]
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
            'resize_step': self.resize_step
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
    """Unified application combining core detection and circle overlay"""
    
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
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[CoreDetectionResult]]:
        """Process frame with core detection"""
        self.frame_count += 1
        results = []
        
        # Run detection
        result = geometric_detection(frame)
        result.frame_number = self.frame_count
        results.append(result)
        
        # Store results
        self.last_detection_results = results
        
        return frame, results
    
    def draw_results_on_frame(self, frame: np.ndarray, results: List[CoreDetectionResult]) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
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
        
        return result_frame
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        
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
            perf_text = f"Circle FPS: {perf_stats['fps']:.1f}"
            cv2.putText(overlay, perf_text, (10, y_offset), font, font_scale, color, thickness)
        
        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("Starting Unified Core Detector with Interactive Circle Overlay")
        print("=" * 60)
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
                
                # Display frame
                try:
                    cv2.imshow(window_name, processed_frame)
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
        description="Unified Core Detector with Interactive Circle Overlay"
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