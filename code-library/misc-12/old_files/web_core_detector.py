#!/usr/bin/env python3
"""
Web-based Core Detector
Displays camera feed in web browser with real-time detection
"""

import cv2
import numpy as np
import time
import json
import base64
import threading
from typing import Optional, Dict
from camera_manager import CameraManager
from flask import Flask, render_template, Response, jsonify
import io
from PIL import Image


class WebCoreDetector:
    """Web-based core detection with browser display"""
    
    def __init__(self, config_file: str = "config.json"):
        self.camera_manager = CameraManager()
        self.config = self._load_config(config_file)
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        self.current_frame = None
        self.current_detection = None
        self.app = Flask(__name__)
        self.setup_routes()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            "web": {
                "host": "localhost",
                "port": 5000,
                "debug": False
            },
            "detection": {
                "min_radius": 10,
                "max_radius": 200,
                "confidence_threshold": 0.3
            },
            "display": {
                "show_fps": True,
                "show_info": True,
                "show_detections": True
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
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/detection')
        def get_detection():
            if self.current_detection:
                return jsonify({
                    'detection': self.current_detection,
                    'frame_count': self.frame_count,
                    'detection_count': self.detection_count,
                    'fps': self.frame_count / max(time.time() - self.start_time, 1)
                })
            return jsonify({'detection': None})
    
    def generate_frames(self):
        """Generate video frames for web stream"""
        while self.is_running:
            if self.current_frame is not None:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', self.current_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
    
    def setup(self) -> bool:
        """Setup camera and web server"""
        print("=== SETTING UP WEB CORE DETECTOR ===")
        
        # Setup camera
        if not self.camera_manager.setup_camera():
            print("❌ Camera setup failed")
            return False
        
        print("✓ Web core detector setup complete")
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
        
        # Add info overlay
        if self.config["display"]["show_info"]:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / max(elapsed_time, 1)
            
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
            for line in info_lines:
                # Background rectangle
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result, (5, y_offset - 20), 
                             (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
                
                # Text
                cv2.putText(result, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        return result
    
    def process_frame(self):
        """Process frames in background"""
        while self.is_running:
            # Read frame
            frame = self.camera_manager.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect core
            detection = self.detect_core(frame)
            
            # Process detection
            if detection:
                self.detection_count += 1
                self.current_detection = detection
                if self.config["display"]["show_info"]:
                    timestamp = time.strftime("%H:%M:%S")
                    center = detection['center']
                    radius = detection['radius']
                    confidence = detection['confidence']
                    print(f"[{timestamp}] Detection #{self.detection_count}: "
                          f"Center=({center[0]}, {center[1]}), "
                          f"Radius={radius}, Confidence={confidence:.3f}")
            else:
                self.current_detection = None
            
            # Draw overlays
            result = self.draw_overlay(frame, detection)
            self.current_frame = result
            
            self.frame_count += 1
            time.sleep(0.1)  # 10 FPS
    
    def run(self):
        """Main run loop with web server"""
        print("=== STARTING WEB CORE DETECTOR ===")
        print(f"🌐 Web interface: http://{self.config['web']['host']}:{self.config['web']['port']}")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start frame processing in background
        frame_thread = threading.Thread(target=self.process_frame)
        frame_thread.daemon = True
        frame_thread.start()
        
        # Start web server
        try:
            self.app.run(
                host=self.config['web']['host'],
                port=self.config['web']['port'],
                debug=self.config['web']['debug'],
                use_reloader=False
            )
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        finally:
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
    detector = WebCoreDetector()
    
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