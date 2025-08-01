#!/usr/bin/env python3
"""
GUI Core Detector with Live Camera Display
Shows real-time camera feed with detection overlays
"""

import cv2
import numpy as np
import time
import json
from typing import Optional, Dict
from camera_manager import CameraManager


class GUICoreDetector:
    """GUI core detection with live camera display"""
    
    def __init__(self, config_file: str = "config.json"):
        self.camera_manager = CameraManager()
        self.config = self._load_config(config_file)
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            "display": {
                "window_name": "Live Core Detection",
                "window_width": 800,
                "window_height": 600,
                "show_fps": True,
                "show_info": True,
                "show_detections": True
            },
            "detection": {
                "min_radius": 10,
                "max_radius": 200,
                "confidence_threshold": 0.3
            },
            "circle_overlay": {
                "center_x": 320,
                "center_y": 240,
                "radius": 50,
                "color": (0, 255, 0),
                "thickness": 2,
                "enabled": True
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self._merge_config(default_config, user_config)
        except Exception as e:
            print(f"⚠ Config load error: {e}, using defaults")
        
        return default_config
    
    def _merge_config(self, default_config: Dict, user_config: Dict):
        """Merge user config with defaults"""
        for key, value in user_config.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    self._merge_config(default_config[key], value)
                else:
                    default_config[key] = value
            else:
                default_config[key] = value
    
    def setup(self) -> bool:
        """Setup camera and GUI"""
        print("=== SETTING UP GUI CORE DETECTOR ===")
        
        # Setup camera
        if not self.camera_manager.setup_camera():
            print("❌ Camera setup failed")
            return False
        
        # Create GUI window
        try:
            window_name = self.config["display"]["window_name"]
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 
                           self.config["display"]["window_width"],
                           self.config["display"]["window_height"])
            print("✓ GUI window created")
        except Exception as e:
            print(f"❌ GUI setup failed: {e}")
            return False
        
        print("✓ GUI core detector setup complete")
        return True
    
    def detect_core(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect core in frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
            
            # Detect circles
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=2.0, minDist=100,
                param1=50, param2=25,
                minRadius=self.config["detection"]["min_radius"],
                maxRadius=self.config["detection"]["max_radius"]
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center_x, center_y, radius = circle
                    confidence = self._calculate_confidence(gray, center_x, center_y, radius)
                    
                    if confidence >= self.config["detection"]["confidence_threshold"]:
                        return {
                            'center': (center_x, center_y),
                            'radius': radius,
                            'confidence': confidence
                        }
            
            return None
            
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            return None
    
    def _calculate_confidence(self, gray: np.ndarray, center_x: int, 
                            center_y: int, radius: int) -> float:
        """Calculate confidence for detected circle"""
        try:
            # Create mask for circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Calculate contrast
            inside_mean = np.mean(gray[mask > 0])
            outside_mask = cv2.circle(np.zeros_like(gray), (center_x, center_y),
                                    radius + 10, 255, -1)
            outside_mask = cv2.circle(outside_mask, (center_x, center_y),
                                    radius, 0, -1)
            outside_mean = np.mean(gray[outside_mask > 0])
            
            contrast = abs(inside_mean - outside_mean) / max(inside_mean, outside_mean, 1)
            return min(1.0, contrast)
            
        except Exception:
            return 0.0
    
    def draw_overlay(self, frame: np.ndarray, detection: Optional[Dict] = None) -> np.ndarray:
        """Draw overlays on frame"""
        result = frame.copy()
        
        # Draw manual circle overlay
        if self.config["circle_overlay"]["enabled"]:
            center_x = self.config["circle_overlay"]["center_x"]
            center_y = self.config["circle_overlay"]["center_y"]
            radius = self.config["circle_overlay"]["radius"]
            color = self.config["circle_overlay"]["color"]
            thickness = self.config["circle_overlay"]["thickness"]
            
            cv2.circle(result, (center_x, center_y), radius, color, thickness)
            cv2.circle(result, (center_x, center_y), 3, (255, 255, 255), -1)
        
        # Draw detection result
        if detection and self.config["display"]["show_detections"]:
            det_center = detection['center']
            det_radius = detection['radius']
            confidence = detection['confidence']
            
            # Draw detected circle
            cv2.circle(result, det_center, det_radius, (0, 0, 255), 3)
            cv2.circle(result, det_center, 5, (255, 255, 255), -1)
            
            # Draw confidence text
            cv2.putText(result, f"Conf: {confidence:.2f}", 
                       (det_center[0] + 10, det_center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw detection number
            cv2.putText(result, f"#{self.detection_count}", 
                       (det_center[0] + 10, det_center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def add_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame"""
        result = frame.copy()
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / max(elapsed_time, 1)
        
        # Add info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Time: {time.strftime('%H:%M:%S')}"
        ]
        
        camera_info = self.camera_manager.get_camera_info()
        if camera_info['is_demo_mode']:
            info_lines.append("DEMO MODE")
        else:
            info_lines.append("LIVE CAMERA")
        
        # Draw info background
        y_offset = 30
        for i, line in enumerate(info_lines):
            # Background rectangle
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result, (5, y_offset - 20), 
                         (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(result, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return result
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == 27:  # ESC
            return False
        
        # Circle movement
        if self.config["circle_overlay"]["enabled"]:
            center_x = self.config["circle_overlay"]["center_x"]
            center_y = self.config["circle_overlay"]["center_y"]
            radius = self.config["circle_overlay"]["radius"]
            
            if key == ord('w'):  # Up
                center_y = max(radius, center_y - 10)
            elif key == ord('s'):  # Down
                center_y = min(480 - radius, center_y + 10)
            elif key == ord('a'):  # Left
                center_x = max(radius, center_x - 10)
            elif key == ord('d'):  # Right
                center_x = min(640 - radius, center_x + 10)
            elif key == ord('q'):  # Decrease radius
                radius = max(5, radius - 5)
            elif key == ord('e'):  # Increase radius
                radius = min(200, radius + 5)
            
            self.config["circle_overlay"]["center_x"] = center_x
            self.config["circle_overlay"]["center_y"] = center_y
            self.config["circle_overlay"]["radius"] = radius
        
        return True
    
    def run(self):
        """Main run loop with GUI"""
        print("=== STARTING GUI CORE DETECTOR ===")
        print("Controls: WASD to move circle, Q/E to resize, ESC to exit")
        
        self.is_running = True
        self.start_time = time.time()
        
        while self.is_running:
            # Read frame
            frame = self.camera_manager.read_frame()
            if frame is None:
                continue
            
            # Detect core
            detection = self.detect_core(frame)
            
            # Process detection
            if detection:
                self.detection_count += 1
                if self.config["display"]["show_info"]:
                    timestamp = time.strftime("%H:%M:%S")
                    center = detection['center']
                    radius = detection['radius']
                    confidence = detection['confidence']
                    print(f"[{timestamp}] Detection #{self.detection_count}: "
                          f"Center=({center[0]}, {center[1]}), "
                          f"Radius={radius}, Confidence={confidence:.3f}")
            
            # Draw overlays
            result = self.draw_overlay(frame, detection)
            
            # Add info overlay
            if self.config["display"]["show_info"]:
                result = self.add_info_overlay(result)
            
            # Display frame
            cv2.imshow(self.config["display"]["window_name"], result)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard(key):
                break
            
            self.frame_count += 1
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("=== CLEANING UP ===")
        self.is_running = False
        self.camera_manager.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / max(elapsed, 1)
        print(f"Final Statistics:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Detections found: {self.detection_count}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Total time: {elapsed:.1f}s")
        print("✓ Cleanup complete")


def main():
    """Main function"""
    detector = GUICoreDetector()
    
    if detector.setup():
        try:
            detector.run()
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
        finally:
            detector.cleanup()
    else:
        print("❌ Setup failed")


if __name__ == "__main__":
    main() 