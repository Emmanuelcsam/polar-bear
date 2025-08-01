#!/usr/bin/env python3
"""
Headless Core Detector
Works without GUI support, focuses on camera detection and processing
"""

import cv2
import numpy as np
import time
import json
from typing import Optional, Dict
from camera_manager import CameraManager


class HeadlessCoreDetector:
    """Headless core detection with guaranteed functionality"""
    
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
            "detection": {
                "min_radius": 10,
                "max_radius": 200,
                "confidence_threshold": 0.3
            },
            "output": {
                "save_frames": False,
                "output_directory": "output",
                "log_detections": True
            },
            "processing": {
                "max_frames": 1000,
                "frame_interval": 0.1
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
        """Setup camera and detection system"""
        print("=== SETTING UP HEADLESS CORE DETECTOR ===")
        
        # Setup camera
        if not self.camera_manager.setup_camera():
            print("❌ Camera setup failed")
            return False
        
        # Create output directory if needed
        if self.config["output"]["save_frames"]:
            import os
            os.makedirs(self.config["output"]["output_directory"], exist_ok=True)
        
        print("✓ Headless core detector setup complete")
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
    
    def save_frame(self, frame: np.ndarray, detection: Optional[Dict] = None):
        """Save frame if enabled"""
        if not self.config["output"]["save_frames"]:
            return
        
        try:
            import os
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{self.frame_count:04d}_{timestamp}.jpg"
            filepath = os.path.join(self.config["output"]["output_directory"], filename)
            
            # Draw detection on frame if available
            if detection:
                center = detection['center']
                radius = detection['radius']
                confidence = detection['confidence']
                
                # Draw circle
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
                cv2.circle(frame, center, 3, (255, 255, 255), -1)
                
                # Add text
                cv2.putText(frame, f"Conf: {confidence:.2f}", 
                           (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(filepath, frame)
            print(f"✓ Saved frame: {filename}")
            
        except Exception as e:
            print(f"⚠ Save error: {e}")
    
    def log_detection(self, detection: Dict):
        """Log detection results"""
        if not self.config["output"]["log_detections"]:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        center = detection['center']
        radius = detection['radius']
        confidence = detection['confidence']
        
        print(f"[{timestamp}] Detection #{self.detection_count}: "
              f"Center=({center[0]}, {center[1]}), "
              f"Radius={radius}, Confidence={confidence:.3f}")
    
    def run(self):
        """Main run loop"""
        print("=== STARTING HEADLESS CORE DETECTOR ===")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        self.start_time = time.time()
        max_frames = self.config["processing"]["max_frames"]
        frame_interval = self.config["processing"]["frame_interval"]
        
        while self.is_running and self.frame_count < max_frames:
            # Read frame
            frame = self.camera_manager.read_frame()
            if frame is None:
                time.sleep(frame_interval)
                continue
            
            # Detect core
            detection = self.detect_core(frame)
            
            # Process results
            if detection:
                self.detection_count += 1
                self.log_detection(detection)
                self.save_frame(frame, detection)
            else:
                self.save_frame(frame)
            
            # Progress update
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / max(elapsed, 1)
                print(f"Processed {self.frame_count} frames, "
                      f"FPS: {fps:.1f}, Detections: {self.detection_count}")
            
            self.frame_count += 1
            time.sleep(frame_interval)
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("=== CLEANING UP ===")
        self.is_running = False
        self.camera_manager.release()
        
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
    detector = HeadlessCoreDetector()
    
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