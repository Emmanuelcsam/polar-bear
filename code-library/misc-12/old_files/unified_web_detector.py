#!/usr/bin/env python3
"""
Unified Web-based Core Detection System
Combines manual circle overlay, automatic detection, and visual display via web browser
"""

import cv2
import numpy as np
import time
import json
import threading
from typing import Optional, Dict, Tuple
from collections import deque
from camera_manager import CameraManager
from circle_overlay import UltraFastCircleOverlay
from flask import Flask, render_template, Response, jsonify
import io
from PIL import Image


class UnifiedWebDetector:
    """Unified web-based core detection with manual overlay and automatic detection"""
    
    def __init__(self, config_file: str = "config.json"):
        # Initialize components
        self.camera_manager = CameraManager()
        self.circle_overlay = UltraFastCircleOverlay(config_file)
        self.config = self._load_config(config_file)
        
        # Detection state
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0
        self.auto_detection_enabled = True
        self.manual_override = False
        
        # Display settings
        self.show_manual = True
        self.show_automatic = True
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Current frame and detection
        self.current_frame = None
        self.current_detections = {}
        
        # Flask app
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Instructions
        self.instructions = {
            "WASD": "Move manual circle",
            "Q/E": "Resize manual circle", 
            "L": "Lock/unlock manual circle",
            "M": "Toggle manual override",
            "A": "Toggle automatic detection",
            "R": "Reset to center",
            "ESC": "Exit"
        }
    
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
                "show_manual": True,
                "show_automatic": True
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
            return render_template('unified_index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/api/detections')
        def get_detections():
            return jsonify({
                'detections': self.current_detections,
                'frame_count': self.frame_count,
                'detection_count': self.detection_count,
                'fps': self.frame_count / max(time.time() - self.start_time, 1),
                'manual_override': self.manual_override,
                'auto_detection_enabled': self.auto_detection_enabled
            })
    
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
        print("=== SETTING UP UNIFIED WEB DETECTOR ===")
        
        # Setup camera
        if not self.camera_manager.setup_camera():
            print("❌ Camera setup failed")
            return False
        
        print("✓ Unified web detector setup complete")
        return True
    
    def detect_core_automatic(self, frame: np.ndarray) -> Optional[Dict]:
        """Automatic core detection using Hough circles"""
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
                            'confidence': confidence,
                            'type': 'automatic'
                        }
            
            return None
            
        except Exception as e:
            print(f"⚠ Automatic detection error: {e}")
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
    
    def get_manual_detection(self) -> Dict:
        """Get manual detection from circle overlay"""
        return {
            'center': tuple(self.circle_overlay.center),
            'radius': self.circle_overlay.radius,
            'confidence': 1.0 if self.circle_overlay.is_locked else 0.5,
            'type': 'manual'
        }
    
    def draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw all detections on frame"""
        result = frame.copy()
        
        # Draw manual detection
        if 'manual' in detections and self.show_manual:
            manual = detections['manual']
            color = (0, 0, 255) if manual['confidence'] > 0.9 else (0, 255, 255)  # Red if locked, yellow if not
            center = (int(manual['center'][0]), int(manual['center'][1]))
            radius = int(manual['radius'])
            
            cv2.circle(result, center, radius, color, 2)
            cv2.circle(result, center, 3, color, -1)
            
            # Add lock indicator
            if manual['confidence'] > 0.9:
                cv2.circle(result, center, radius + 5, (0, 0, 255), 1)
                cv2.putText(result, "MANUAL (LOCKED)", 
                           (center[0] - 50, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(result, "MANUAL", 
                           (center[0] - 30, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw automatic detection
        if 'automatic' in detections and self.show_automatic and self.auto_detection_enabled:
            auto = detections['automatic']
            if auto['confidence'] > 0.1:  # Only show if confident
                color = (0, 255, 0)  # Green
                center = (int(auto['center'][0]), int(auto['center'][1]))
                radius = int(auto['radius'])
                
                cv2.circle(result, center, radius, color, 2)
                cv2.circle(result, center, 3, color, -1)
                cv2.putText(result, f"AUTO ({auto['confidence']:.2f})", 
                           (center[0] - 40, center[1] + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def add_info_overlay(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Add information overlay to frame"""
        result = frame.copy()
        
        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(result, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        y_offset = 30
        line_height = 25
        
        # System status
        status_text = f"Manual Override: {'ON' if self.manual_override else 'OFF'}"
        cv2.putText(frame, status_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Automatic detection status
        auto_text = f"Auto Detection: {'ON' if self.auto_detection_enabled else 'OFF'}"
        cv2.putText(frame, auto_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Detection counts
        detection_count = len([d for d in detections.values() if d['confidence'] > 0.1])
        detection_text = f"Active Detections: {detection_count}"
        cv2.putText(frame, detection_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Camera info
        camera_info = self.camera_manager.get_camera_info()
        if camera_info['is_demo_mode']:
            camera_text = "DEMO MODE"
        else:
            camera_text = "LIVE CAMERA"
        cv2.putText(frame, camera_text, (10, y_offset), font, font_scale, color, thickness)
        
        self.last_frame_time = current_time
        
        return frame
    
    def process_frame(self):
        """Process frames in background"""
        while self.is_running:
            # Read frame
            frame = self.camera_manager.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Get detections
            detections = {}
            
            # Manual detection (circle overlay)
            if self.show_manual:
                detections['manual'] = self.get_manual_detection()
            
            # Automatic detection
            if self.show_automatic and self.auto_detection_enabled:
                auto_detection = self.detect_core_automatic(frame)
                if auto_detection:
                    detections['automatic'] = auto_detection
                    self.detection_count += 1
                    
                    # Log detection
                    timestamp = time.strftime("%H:%M:%S")
                    center = auto_detection['center']
                    radius = auto_detection['radius']
                    confidence = auto_detection['confidence']
                    print(f"[{timestamp}] Auto Detection #{self.detection_count}: "
                          f"Center=({center[0]}, {center[1]}), "
                          f"Radius={radius}, Confidence={confidence:.3f}")
            
            # Handle continuous input for circle overlay
            self.circle_overlay.handle_continuous_input(frame.shape[:2])
            
            # Draw detections
            result = self.draw_detections(frame, detections)
            
            # Add info overlay
            result = self.add_info_overlay(result, detections)
            
            # Store current frame and detections
            self.current_frame = result
            self.current_detections = detections
            
            self.frame_count += 1
            time.sleep(0.1)  # 10 FPS
    
    def run(self):
        """Main run loop with web server"""
        print("=== STARTING UNIFIED WEB DETECTOR ===")
        print(f"🌐 Web interface: http://{self.config['web']['host']}:{self.config['web']['port']}")
        print("Controls:")
        for control, description in self.instructions.items():
            print(f"  {control}: {description}")
        print("\nPress Ctrl+C to stop")
        
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
        self.circle_overlay.cleanup()
        
        # Final statistics
        elapsed = time.time() - self.start_time
        fps = self.frame_count / max(elapsed, 1)
        print(f"Final Statistics:")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Auto detections found: {self.detection_count}")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Total time: {elapsed:.1f}s")
        print("✓ Cleanup complete")


def main():
    """Main function"""
    detector = UnifiedWebDetector()
    
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