#!/usr/bin/env python3
"""
Integrated Learning System for Core Detection
Combines manual circle overlay with automatic geometric detection
and implements PyTorch-based learning for automatic alignment.
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Tuple, Dict, List
from collections import deque

# Import our modules
from auto_core_detection import EnhancedGeometricCoreDetector, DetectionResult
from circle_overlay import UltraFastCircleOverlay
from config_manager import ConfigManager


class IntegratedLearningSystem:
    """Integrated learning system for core detection with manual-to-automatic alignment"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = False, 
                                              config_file: str = "config.json"):
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_pytorch_config()
        
        # Initialize components
        self.detector = EnhancedGeometricCoreDetector(config_file=config_file)
        self.circle_overlay = UltraFastCircleOverlay(config_file)
        
        # Camera setup
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.camera = None
        self.setup_camera()
        
        # Learning state
        self.is_learning = False
        self.learning_history = deque(maxlen=1000)
        self.auto_detection_enabled = True
        self.manual_override = False
        
        # Display settings
        self.window_name = "Integrated Learning System"
        self.show_manual = True
        self.show_automatic = True
        self.show_improved = True
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Instructions
        self.instructions = {
            "WASD": "Move manual circle",
            "Q/E": "Resize manual circle",
            "L": "Lock/unlock manual circle",
            "M": "Toggle manual override",
            "A": "Toggle automatic detection",
            "I": "Toggle improved detection",
            "T": "Train from manual detection",
            "R": "Reset to center",
            "ESC": "Exit"
        }
    
    def setup_camera(self):
        """Setup camera interface"""
        try:
            # Try to use Pylon if available
            if self.use_pylon:
                try:
                    from pypylon import pylon
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
                            print(f"Using Pylon camera: {self.camera.GetDeviceInfo().GetModelName()}")
                            return
                except ImportError:
                    print("Pylon not available, using webcam")
                except Exception as e:
                    print(f"Error setting up Pylon camera: {e}")
            
            # Fallback to webcam
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                # Try alternative indices
                for alt_index in [1, 2, 3]:
                    self.camera = cv2.VideoCapture(alt_index)
                    if self.camera.isOpened():
                        self.camera_index = alt_index
                        break
                
                if not self.camera.isOpened():
                    print("\n" + "="*60)
                    print("CAMERA NOT DETECTED")
                    print("="*60)
                    print("Your camera was not detected by OpenCV.")
                    print("This could be due to:")
                    print("1. Camera drivers not installed or outdated")
                    print("2. Camera being used by another application")
                    print("3. Permission issues")
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
                    self.camera = None
                    self.demo_mode = True
                    print("Demo mode activated - using synthetic camera feed")
                    return
            
            print(f"Using webcam at index {self.camera_index}")
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            raise
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read frame from camera"""
        if self.camera is None:
            return None
        
        # Handle demo mode
        if hasattr(self, 'demo_mode') and self.demo_mode:
            if not hasattr(self, 'demo_frame_count'):
                self.demo_frame_count = 0
            self.demo_frame_count += 1
            
            # Generate a simple synthetic frame for demo mode
            height, width = 480, 640
            image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(image, f"Demo Frame {self.demo_frame_count}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return image
        
        try:
            if hasattr(self.camera, 'RetrieveResult'):  # Pylon camera
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
            else:  # Webcam
                ret, frame = self.camera.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process frame with all detection methods"""
        results = {}
        
        # Manual detection (circle overlay)
        if self.show_manual:
            manual_result = DetectionResult(
                center=tuple(self.circle_overlay.center),
                radius=self.circle_overlay.radius,
                confidence=1.0 if self.circle_overlay.is_locked else 0.5,
                method="manual",
                timestamp=time.time()
            )
            results['manual'] = manual_result
        
        # Automatic geometric detection
        if self.show_automatic and self.auto_detection_enabled:
            geometric_result = self.detector.geometric_detection(frame)
            results['automatic'] = geometric_result
        
        # Improved detection (learned)
        if self.show_improved and self.auto_detection_enabled:
            improved_result = self.detector.improved_detection(frame)
            results['improved'] = improved_result
        
        # Draw results on frame
        processed_frame = self.draw_results(frame, results)
        
        return processed_frame, results
    
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw all detection results on frame"""
        result_frame = frame.copy()
        
        # Draw manual circle overlay
        if 'manual' in results:
            manual = results['manual']
            color = (0, 0, 255) if manual.confidence > 0.9 else (0, 255, 255)  # Red if locked, yellow if not
            center = (int(manual.center[0]), int(manual.center[1]))
            radius = int(manual.radius)
            
            cv2.circle(result_frame, center, radius, color, 2)
            cv2.circle(result_frame, center, 3, color, -1)
            
            # Add lock indicator
            if manual.confidence > 0.9:
                cv2.circle(result_frame, center, radius + 5, (0, 0, 255), 1)
                cv2.putText(result_frame, "LOCKED", 
                           (center[0] - 30, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(result_frame, "MANUAL", 
                           (center[0] - 30, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw automatic detection
        if 'automatic' in results:
            auto = results['automatic']
            if auto.confidence > 0.1:  # Only show if confident
                color = (0, 255, 0)  # Green
                center = (int(auto.center[0]), int(auto.center[1]))
                radius = int(auto.radius)
                
                cv2.circle(result_frame, center, radius, color, 2)
                cv2.circle(result_frame, center, 3, color, -1)
                cv2.putText(result_frame, f"AUTO ({auto.confidence:.2f})", 
                           (center[0] - 30, center[1] + radius + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw improved detection
        if 'improved' in results:
            improved = results['improved']
            if improved.confidence > 0.1:  # Only show if confident
                color = (255, 0, 255)  # Magenta
                center = (int(improved.center[0]), int(improved.center[1]))
                radius = int(improved.radius)
                
                cv2.circle(result_frame, center, radius, color, 2)
                cv2.circle(result_frame, center, 3, color, -1)
                cv2.putText(result_frame, f"IMPROVED ({improved.confidence:.2f})", 
                           (center[0] - 40, center[1] + radius + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add information overlay
        result_frame = self.add_info_overlay(result_frame, results)
        
        return result_frame
    
    def add_info_overlay(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        
        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        y_offset = 30
        line_height = 25
        
        # System status
        status_text = f"Learning: {'ON' if self.is_learning else 'OFF'}"
        cv2.putText(frame, status_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Manual override status
        override_text = f"Manual Override: {'ON' if self.manual_override else 'OFF'}"
        cv2.putText(frame, override_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Detection counts
        detection_count = len([r for r in results.values() if r.confidence > 0.1])
        detection_text = f"Active Detections: {detection_count}"
        cv2.putText(frame, detection_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Training samples
        training_text = f"Training Samples: {len(self.detector.training_data)}"
        cv2.putText(frame, training_text, (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, y_offset), font, font_scale, color, thickness)
        
        self.last_frame_time = current_time
        
        return frame
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == 27:  # ESC
            return False
        
        # Circle overlay controls
        if not self.circle_overlay.handle_keyboard_input(key, (640, 480)):
            return False
        
        # System controls
        if key == ord('m') or key == ord('M'):
            self.manual_override = not self.manual_override
            print(f"Manual override: {'ON' if self.manual_override else 'OFF'}")
        
        elif key == ord('a') or key == ord('A'):
            self.auto_detection_enabled = not self.auto_detection_enabled
            print(f"Automatic detection: {'ON' if self.auto_detection_enabled else 'OFF'}")
        
        elif key == ord('i') or key == ord('I'):
            self.show_improved = not self.show_improved
            print(f"Improved detection: {'ON' if self.show_improved else 'OFF'}")
        
        elif key == ord('t') or key == ord('T'):
            self.train_from_manual()
        
        return True
    
    def train_from_manual(self):
        """Train the model from manual detection"""
        if not self.circle_overlay.is_locked:
            print("Please lock the manual circle first (press L)")
            return
        
        # Get current frame
        frame = self.read_frame()
        if frame is None:
            print("No frame available for training")
            return
        
        # Train from manual detection
        manual_center = tuple(self.circle_overlay.center)
        manual_radius = self.circle_overlay.radius
        
        self.detector.learn_from_manual_detection(frame, manual_center, manual_radius)
        
        # Store in learning history
        learning_entry = {
            'timestamp': time.time(),
            'manual_center': manual_center,
            'manual_radius': manual_radius,
            'frame_shape': frame.shape
        }
        self.learning_history.append(learning_entry)
        
        print(f"Trained from manual detection: center={manual_center}, radius={manual_radius}")
    
    def run(self):
        """Run the integrated learning system"""
        print("Integrated Learning System for Core Detection")
        print("Controls:")
        for control, description in self.instructions.items():
            print(f"  {control}: {description}")
        print("\nPress ESC to exit")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Handle continuous input for circle overlay
                self.circle_overlay.handle_continuous_input(frame.shape[:2])
                
                # Display frame
                cv2.imshow(self.window_name, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in learning system: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.camera is not None:
            if hasattr(self.camera, 'StopGrabbing'):
                try:
                    self.camera.StopGrabbing()
                except Exception:
                    pass
            elif hasattr(self.camera, 'release'):
                self.camera.release()
        
        self.circle_overlay.cleanup()
        cv2.destroyAllWindows()
        
        # Save model and data
        self.detector.save_model()
        self.detector.save_training_data()
        
        print("Learning system stopped")
    
    def export_learning_data(self, output_path: str = "learning_data.pkl"):
        """Export all learning data for analysis"""
        try:
            export_data = {
                'detector_data': {
                    'training_data': self.detector.training_data,
                    'detection_history': list(self.detector.detection_history),
                    'model_path': self.detector.model_path
                },
                'learning_history': list(self.learning_history),
                'system_config': {
                    'camera_index': self.camera_index,
                    'use_pylon': self.use_pylon,
                    'auto_detection_enabled': self.auto_detection_enabled,
                    'manual_override': self.manual_override
                }
            }
            
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            print(f"Learning data exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting learning data: {e}")


def main():
    """Main function for the integrated learning system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Learning System for Core Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--pylon", action="store_true", help="Use Pylon SDK")
    parser.add_argument("--export", type=str, help="Export learning data to file")
    
    args = parser.parse_args()
    
    try:
        # Create learning system
        learning_system = IntegratedLearningSystem(
            camera_index=args.camera,
            use_pylon=args.pylon
        )
        
        # Export data if requested
        if args.export:
            learning_system.export_learning_data(args.export)
            return
        
        # Run the system
        learning_system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 