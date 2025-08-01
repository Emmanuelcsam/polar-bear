#!/usr/bin/env python3
"""
Interactive Circle Overlay for Live Video Stream
Adds a blue circle overlay that can be moved, resized, and locked using 
keyboard controls.
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional


class InteractiveCircleOverlay:
    """Interactive circle overlay with keyboard controls"""
    
    def __init__(self, initial_center: Tuple[int, int] = (320, 240), 
                 initial_radius: int = 50, move_step: int = 10, 
                 resize_step: int = 5):
        self.center = list(initial_center)
        self.radius = initial_radius
        self.move_step = move_step
        self.resize_step = resize_step
        self.is_locked = False
        self.color = (255, 0, 0)  # Blue color (BGR format)
        self.thickness = 2
        
        # Control instructions
        self.instructions = {
            'WASD': 'Move circle (W=up, S=down, A=left, D=right)',
            'Q/E': 'Resize circle (Q=smaller, E=larger)',
            'L': 'Lock/Unlock circle position',
            'R': 'Reset circle to center',
            'ESC': 'Exit application'
        }
        
    def draw_circle(self, frame: np.ndarray) -> np.ndarray:
        """Draw the circle overlay on the frame"""
        result_frame = frame.copy()
        
        # Draw the circle
        cv2.circle(result_frame, tuple(self.center), self.radius,
                   self.color, self.thickness)
        
        # Draw center point
        cv2.circle(result_frame, tuple(self.center), 3, self.color, -1)
        
        # Draw lock indicator
        if self.is_locked:
            lock_text = "LOCKED"
            lock_color = (0, 0, 255)  # Red
        else:
            lock_text = "UNLOCKED"
            lock_color = (0, 255, 0)  # Green
            
        cv2.putText(result_frame, lock_text,
                   (self.center[0] - 30, self.center[1] - self.radius - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, lock_color, 1)
        
        return result_frame
    
    def add_instructions_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add instruction overlay to the frame"""
        # Create semi-transparent background for instructions
        height, width = frame.shape[:2]
        instruction_bg = np.zeros((200, width, 3), dtype=np.uint8)
        
        # Add instructions text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        y_offset = 25
        line_height = 20
        
        cv2.putText(instruction_bg, "Interactive Circle Controls:",
                   (10, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
        for control, description in self.instructions.items():
            cv2.putText(instruction_bg, f"{control}: {description}",
                       (10, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        
        # Add circle info
        circle_info = (f"Circle: Center({self.center[0]}, {self.center[1]}), "
                      f"Radius: {self.radius}")
        cv2.putText(instruction_bg, circle_info,
                   (10, y_offset), font, font_scale, color, thickness)
        
        # Overlay instructions on frame
        frame[height-200:height, :] = cv2.addWeighted(
            frame[height-200:height, :], 0.3, instruction_bg, 0.7, 0
        )
        
        return frame
    
    def handle_keyboard_input(self, key: int, frame_shape: Tuple[int, int]) -> bool:
        """Handle keyboard input and return True if should continue, False if exit"""
        height, width = frame_shape[:2]
        
        if key == 27:  # ESC key
            return False
            
        if key == ord('l') or key == ord('L'):  # Lock/Unlock
            self.is_locked = not self.is_locked
            print(f"Circle {'LOCKED' if self.is_locked else 'UNLOCKED'}")
            return True
            
        if self.is_locked:
            return True  # Ignore movement keys when locked
            
        if key == ord('w') or key == ord('W'):  # Move up
            self.center[1] = max(self.radius, self.center[1] - self.move_step)
        elif key == ord('s') or key == ord('S'):  # Move down
            self.center[1] = min(height - self.radius, self.center[1] + self.move_step)
        elif key == ord('a') or key == ord('A'):  # Move left
            self.center[0] = max(self.radius, self.center[0] - self.move_step)
        elif key == ord('d') or key == ord('D'):  # Move right
            self.center[0] = min(width - self.radius, self.center[0] + self.move_step)
        elif key == ord('q') or key == ord('Q'):  # Decrease radius
            self.radius = max(5, self.radius - self.resize_step)
        elif key == ord('e') or key == ord('E'):  # Increase radius
            self.radius = min(min(width, height) // 2, self.radius + self.resize_step)
        elif key == ord('r') or key == ord('R'):  # Reset to center
            self.center = [width // 2, height // 2]
            self.radius = 50
            print("Circle reset to center")
            
        return True
    
    def get_circle_info(self) -> dict:
        """Get current circle information"""
        return {
            'center': tuple(self.center),
            'radius': self.radius,
            'is_locked': self.is_locked,
            'color': self.color
        }


class InteractiveVideoStream:
    """Main application for interactive circle overlay"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = False):
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.camera = None
        self.circle_overlay = None
        self.is_running = False
        
        # Initialize camera
        self._setup_camera()
        
        # Initialize circle overlay
        self.circle_overlay = InteractiveCircleOverlay()
        
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
    
    def run(self):
        """Main application loop"""
        print("Interactive Circle Overlay Started")
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
                
                # Draw circle overlay
                frame_with_circle = self.circle_overlay.draw_circle(frame)
                
                # Add instructions overlay
                frame_with_instructions = self.circle_overlay.add_instructions_overlay(frame_with_circle)
                
                # Convert to BGR for display
                display_frame = cv2.cvtColor(frame_with_instructions, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow("Interactive Circle Overlay", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    should_continue = self.circle_overlay.handle_keyboard_input(key, frame.shape)
                    if not should_continue:
                        break
                
                # Check if window was closed
                if cv2.getWindowProperty("Interactive Circle Overlay", cv2.WND_PROP_VISIBLE) < 1:
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
        description="Interactive Circle Overlay for Live Video Stream"
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
        app = InteractiveVideoStream(
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