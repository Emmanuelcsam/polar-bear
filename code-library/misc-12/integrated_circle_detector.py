#!/usr/bin/env python3
"""
Integrated Circle Overlay with Core Detector
Combines the interactive circle overlay with the existing core detection system.
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional
from interactive_circle_overlay import InteractiveCircleOverlay


class IntegratedCircleDetector:
    """Combines circle overlay with core detection"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = False):
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.camera = None
        self.circle_overlay = InteractiveCircleOverlay()
        self.is_running = False
        
        # Initialize camera
        self._setup_camera()
        
    def _setup_camera(self):
        """Setup camera (webcam or Pylon)"""
        if self.use_pylon:
            try:
                from pypylon import pylon
                tl_factory = pylon.TlFactory.GetInstance()
                devices = tl_factory.EnumerateDevices()
                
                if len(devices) == 0:
                    print("No Pylon cameras found. Using webcam fallback.")
                    self.use_pylon = False
                else:
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
                        print(f"Pylon camera initialized: {self.camera.GetDeviceInfo().GetModelName()}")
                    else:
                        print("Failed to open Pylon camera. Using webcam fallback.")
                        self.use_pylon = False
                        
            except ImportError:
                print("Pylon SDK not available. Using webcam.")
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
        """Read frame from camera"""
        if self.camera is None:
            return None
            
        try:
            if self.use_pylon:
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
    
    def detect_core_in_circle_region(self, frame: np.ndarray) -> dict:
        """Detect core within the circle region"""
        circle_info = self.circle_overlay.get_circle_info()
        center = circle_info['center']
        radius = circle_info['radius']
        
        # Create mask for circle region
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Simple core detection in masked region
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)
        
        # Find contours in the masked region
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the core)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding circle
            (x, y), radius_core = cv2.minEnclosingCircle(largest_contour)
            
            return {
                'center': (int(x), int(y)),
                'radius': int(radius_core),
                'area': cv2.contourArea(largest_contour),
                'detected': True
            }
        else:
            return {
                'center': None,
                'radius': None,
                'area': 0,
                'detected': False
            }
    
    def draw_detection_results(self, frame: np.ndarray, detection_result: dict) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        if detection_result['detected']:
            # Draw detected core
            center = detection_result['center']
            radius = detection_result['radius']
            
            # Green circle for detected core
            cv2.circle(result_frame, center, radius, (0, 255, 0), 2)
            cv2.circle(result_frame, center, 3, (0, 255, 0), -1)
            
            # Add detection info
            cv2.putText(result_frame, f"Core: {radius}px", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add area info
            area_text = f"Area: {detection_result['area']:.0f}"
            cv2.putText(result_frame, area_text,
                       (center[0] - 30, center[1] - radius - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # No core detected
            cv2.putText(result_frame, "No core detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_frame
    
    def add_info_overlay(self, frame: np.ndarray, detection_result: dict) -> np.ndarray:
        """Add information overlay"""
        overlay = frame.copy()
        
        # Create semi-transparent background
        height, width = frame.shape[:2]
        info_bg = np.zeros((150, width, 3), dtype=np.uint8)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        y_offset = 25
        line_height = 25
        
        # Title
        cv2.putText(info_bg, "Integrated Circle Core Detector", 
                   (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Detection status
        if detection_result['detected']:
            status_text = "Core DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "No core detected"
            status_color = (0, 0, 255)
        
        cv2.putText(info_bg, status_text, 
                   (10, y_offset), font, font_scale, status_color, thickness)
        y_offset += line_height
        
        # Circle info
        circle_info = self.circle_overlay.get_circle_info()
        circle_text = f"Circle: ({circle_info['center'][0]}, {circle_info['center'][1]}) R={circle_info['radius']}"
        cv2.putText(info_bg, circle_text, 
                   (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        # Lock status
        lock_text = "LOCKED" if circle_info['is_locked'] else "UNLOCKED"
        lock_color = (0, 0, 255) if circle_info['is_locked'] else (0, 255, 0)
        cv2.putText(info_bg, f"Circle: {lock_text}", 
                   (10, y_offset), font, font_scale, lock_color, thickness)
        
        # Overlay on frame
        frame[height-150:height, :] = cv2.addWeighted(
            frame[height-150:height, :], 0.3, info_bg, 0.7, 0
        )
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("Integrated Circle Core Detector Started")
        print("Controls:")
        for control, description in self.circle_overlay.instructions.items():
            print(f"  {control}: {description}")
        print("\nPress any key to start...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Detect core in circle region
                detection_result = self.detect_core_in_circle_region(frame)
                
                # Draw circle overlay
                frame_with_circle = self.circle_overlay.draw_circle(frame)
                
                # Draw detection results
                frame_with_detection = self.draw_detection_results(
                    frame_with_circle, detection_result
                )
                
                # Add information overlay
                frame_with_info = self.add_info_overlay(
                    frame_with_detection, detection_result
                )
                
                # Convert to BGR for display
                display_frame = cv2.cvtColor(frame_with_info, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow("Integrated Circle Core Detector", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    should_continue = self.circle_overlay.handle_keyboard_input(
                        key, frame.shape
                    )
                    if not should_continue:
                        break
                
                # Check if window was closed
                if cv2.getWindowProperty("Integrated Circle Core Detector", 
                                       cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.camera is not None:
            if self.use_pylon:
                try:
                    self.camera.StopGrabbing()
                except Exception as e:
                    print(f"Error stopping Pylon camera: {e}")
            elif hasattr(self.camera, "release"):
                self.camera.release()
        
        cv2.destroyAllWindows()
        print("Application stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Integrated Circle Core Detector"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--pylon", action="store_true", help="Use Pylon SDK if available"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run application
        app = IntegratedCircleDetector(
            camera_index=args.camera,
            use_pylon=args.pylon
        )
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 