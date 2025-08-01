#!/usr/bin/env python3
"""
Parallel Circle Overlay for Live Core Detector
Runs as a separate process to overlay an interactive circle on top of the live core 
detector. Provides maximum freedom of movement and resizing without any parameter limits.
"""

import cv2
import numpy as np
import time
import threading
from typing import Tuple, Dict
from collections import deque

# Import configuration system
try:
    from config_manager import ConfigManager
except ImportError:
    # Fallback configuration if config module not available
    class ConfigManager:
        def __init__(self, config_file="config.json"):
            self.config = {
                "circle_overlay": {
                    "movement": {
                        "move_step": 8,
                        "resize_step": 5,
                        "enable_continuous_movement": False,
                        "enable_parallel_worker": False,
                        "key_repeat_rate": 0.008,
                    },
                    "circle": {
                        "initial_center_x": 320,
                        "initial_center_y": 240,
                        "initial_radius": 50,
                        "min_radius": 1,
                        "max_radius": 999999,
                        "color_red": 255,
                        "color_green": 0,
                        "color_blue": 0,
                        "thickness": 2,
                        "center_point_size": 3,
                    },
                    "keyboard": {
                        "enable_key_repeat": False,
                        "enable_simultaneous_keys": False,
                    },
                    "performance": {
                        "enable_performance_tracking": True,
                        "frame_time_history_size": 30,
                    }
                }
            }
        
        def get_circle_overlay_config(self):
            return self.config.get("circle_overlay", {})


class ParallelCircleOverlay:
    """Parallel circle overlay that runs as a separate process"""
    
    def __init__(self, window_name: str = "Circle Overlay",
                 overlay_on_window: str = "Live Core Detector"):
        # Load configuration
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_circle_overlay_config()
        
        # Window settings
        self.window_name = window_name
        self.overlay_on_window = overlay_on_window
        
        # Initialize circle properties from config
        circle_config = self.config["circle"]
        self.center = [circle_config["initial_center_x"], 
                      circle_config["initial_center_y"]]
        self.radius = circle_config["initial_radius"]
        self.color = (circle_config["color_blue"], 
                     circle_config["color_green"], 
                     circle_config["color_red"])  # BGR format
        self.thickness = circle_config["thickness"]
        self.center_point_size = circle_config["center_point_size"]
        self.is_locked = False
        
        # Movement settings - NO LIMITS for maximum freedom
        movement_config = self.config["movement"]
        self.move_step = movement_config["move_step"]
        self.resize_step = movement_config["resize_step"]
        self.enable_continuous_movement = movement_config["enable_continuous_movement"]
        self.enable_parallel_worker = movement_config["enable_parallel_worker"]
        self.key_repeat_rate = movement_config["key_repeat_rate"]
        
        # Keyboard settings from config
        keyboard_config = self.config["keyboard"]
        self.enable_key_repeat = keyboard_config["enable_key_repeat"]
        self.enable_simultaneous_keys = keyboard_config["enable_simultaneous_keys"]
        
        # Performance settings from config
        performance_config = self.config["performance"]
        self.enable_performance_tracking = performance_config["enable_performance_tracking"]
        self.frame_time_history_size = performance_config["frame_time_history_size"]
        
        # Manual control - disable automatic movement
        self.pressed_keys = {}
        self.last_key_time = 0
        
        # Performance tracking
        self.frame_times = deque(maxlen=self.frame_time_history_size)
        self.last_frame_time = time.time()
        
        # Parallel processing (disabled by default)
        self.movement_thread = None
        self.is_movement_running = False
        self.movement_queue = deque(maxlen=100)
        
        # Window management
        self.is_running = False
        self.window_created = False
        self.overlay_window = None
        
        # Instructions
        self.instructions = {
            "WASD": "Move circle (W=up, S=down, A=left, D=right)",
            "Q/E": "Resize circle (Q=smaller, E=larger)",
            "L": "Lock/Unlock circle position",
            "R": "Reset circle to center",
            "ESC": "Exit overlay"
        }
        
        # Start parallel worker only if enabled
        if self.enable_parallel_worker:
            self._start_movement_thread()
    
    def _start_movement_thread(self):
        """Start parallel movement processing thread (only if enabled)"""
        if not self.enable_parallel_worker:
            return
            
        self.is_movement_running = True
        self.movement_thread = threading.Thread(target=self._movement_worker, 
                                              daemon=True)
        self.movement_thread.start()
    
    def _movement_worker(self):
        """Parallel movement processing worker (only if enabled)"""
        if not self.enable_parallel_worker:
            return
            
        while self.is_movement_running:
            try:
                # Process movement commands
                if self.movement_queue:
                    movement = self.movement_queue.popleft()
                    self._apply_movement(movement)
                
                # Manual key repeat (only if enabled)
                if self.enable_continuous_movement:
                    current_time = time.time()
                    if current_time - self.last_key_time >= self.key_repeat_rate:
                        self._process_continuous_movement()
                        self.last_key_time = current_time
                
                time.sleep(0.001)  # 1ms sleep for maximum responsiveness
                
            except Exception as e:
                print(f"Movement worker error: {e}")
    
    def _apply_movement(self, movement: str):
        """Apply movement command with NO BOUNDARY LIMITS for maximum freedom"""
        if self.is_locked:
            return
        
        # NO BOUNDARY RESTRICTIONS - complete freedom of movement
        if movement == "up":
            self.center[1] -= self.move_step
        elif movement == "down":
            self.center[1] += self.move_step
        elif movement == "left":
            self.center[0] -= self.move_step
        elif movement == "right":
            self.center[0] += self.move_step
        elif movement == "smaller":
            new_radius = self.radius - self.resize_step
            if new_radius >= 1:  # Only prevent negative radius
                self.radius = new_radius
        elif movement == "larger":
            self.radius += self.resize_step
            # NO UPPER LIMIT - circle can grow as large as needed
    
    def _process_continuous_movement(self):
        """Process continuous key presses (only if enabled)"""
        if self.is_locked or not self.enable_continuous_movement:
            return
        
        # Check for pressed keys and apply movement
        if self.pressed_keys.get(ord('w')) or self.pressed_keys.get(ord('W')):
            self._apply_movement("up")
        if self.pressed_keys.get(ord('s')) or self.pressed_keys.get(ord('S')):
            self._apply_movement("down")
        if self.pressed_keys.get(ord('a')) or self.pressed_keys.get(ord('A')):
            self._apply_movement("left")
        if self.pressed_keys.get(ord('d')) or self.pressed_keys.get(ord('D')):
            self._apply_movement("right")
        if self.pressed_keys.get(ord('q')) or self.pressed_keys.get(ord('Q')):
            self._apply_movement("smaller")
        if self.pressed_keys.get(ord('e')) or self.pressed_keys.get(ord('E')):
            self._apply_movement("larger")
    
    def update_pressed_keys(self, key: int):
        """Update pressed keys state for manual control"""
        if key == 255:  # No key pressed
            if not self.enable_simultaneous_keys:
                self.pressed_keys.clear()
            return
        
        # Manual key handling - no automatic repeat
        if not self.enable_key_repeat:
            self.pressed_keys.clear()
        
        # Set the pressed key
        self.pressed_keys[key] = True
    
    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input with manual control"""
        if key == 27:  # ESC key
            return False
        
        # Handle special keys
        if key == ord('l') or key == ord('L'):
            self.is_locked = not self.is_locked
        elif key == ord('r') or key == ord('R'):
            # Reset to config values
            circle_config = self.config["circle"]
            self.center = [circle_config["initial_center_x"], 
                          circle_config["initial_center_y"]]
            self.radius = circle_config["initial_radius"]
        else:
            # Manual movement - add to queue only if parallel worker enabled
            movement = self._get_movement_from_key(key)
            if movement and self.enable_parallel_worker:
                self.movement_queue.append(movement)
            elif movement:
                # Direct movement for manual control
                self._apply_movement(movement)
        
        return True
    
    def _get_movement_from_key(self, key: int) -> str:
        """Get movement command from key press"""
        if key in [ord('w'), ord('W')]:
            return "up"
        elif key in [ord('s'), ord('S')]:
            return "down"
        elif key in [ord('a'), ord('A')]:
            return "left"
        elif key in [ord('d'), ord('D')]:
            return "right"
        elif key in [ord('q'), ord('Q')]:
            return "smaller"
        elif key in [ord('e'), ord('E')]:
            return "larger"
        return ""
    
    def handle_continuous_input(self):
        """Handle continuous input for manual control"""
        # Only process if continuous movement is enabled
        if self.enable_continuous_movement:
            self._process_continuous_movement()
    
    def create_overlay_window(self, width: int = 800, height: int = 600):
        """Create overlay window that can be positioned over the main detector window"""
        if self.window_created:
            return
        
        # Create named window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Set window properties for overlay - handle version compatibility
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except AttributeError:
            print("Warning: WND_PROP_TOPMOST not available in this OpenCV version")
        
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TRANSPARENT, 1)
        except AttributeError:
            print("Warning: WND_PROP_TRANSPARENT not available in this OpenCV version")
        
        # Position window over the main detector window
        try:
            # Get main window position and size
            try:
                main_window_rect = cv2.getWindowImageRect(self.overlay_on_window)
                if main_window_rect:
                    x, y, w, h = main_window_rect
                    cv2.moveWindow(self.window_name, x, y)
                    cv2.resizeWindow(self.window_name, w, h)
                else:
                    # Default position if main window not found
                    cv2.moveWindow(self.window_name, 100, 100)
                    cv2.resizeWindow(self.window_name, width, height)
            except AttributeError:
                print("Warning: getWindowImageRect not available in this OpenCV version")
                # Default position
                cv2.moveWindow(self.window_name, 100, 100)
                cv2.resizeWindow(self.window_name, width, height)
        except Exception as e:
            # Fallback positioning
            print(f"Warning: Could not position overlay window: {e}")
            cv2.moveWindow(self.window_name, 100, 100)
            cv2.resizeWindow(self.window_name, width, height)
        
        self.window_created = True
    
    def draw_circle_overlay(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Draw circle on transparent overlay frame"""
        # Create transparent frame
        overlay_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw circle with config-based styling
        center = (int(self.center[0]), int(self.center[1]))
        radius = int(self.radius)
        
        # Ensure circle is within frame bounds
        center_x = max(radius, min(width - radius, center[0]))
        center_y = max(radius, min(height - radius, center[1]))
        center = (center_x, center_y)
        
        # Draw main circle
        cv2.circle(overlay_frame, center, radius, self.color, self.thickness)
        
        # Draw center point
        cv2.circle(overlay_frame, center, self.center_point_size, self.color, -1)
        
        # Draw lock indicator if enabled
        if self.is_locked:
            lock_color = (0, 0, 255)  # Red for locked
            cv2.circle(overlay_frame, center, radius + 5, lock_color, 1)
        
        # Update performance tracking if enabled
        if self.enable_performance_tracking:
            current_time = time.time()
            self.frame_times.append(current_time - self.last_frame_time)
            self.last_frame_time = current_time
        
        return overlay_frame
    
    def create_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for circle region"""
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        center = (int(self.center[0]), int(self.center[1]))
        radius = int(self.radius)
        
        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
    def get_circle_info(self) -> Dict:
        """Get circle information"""
        return {
            'center': tuple(self.center),
            'radius': self.radius,
            'color': self.color,
            'is_locked': self.is_locked,
            'move_step': self.move_step,
            'resize_step': self.resize_step,
            'window_name': self.window_name
        }
    
    def set_performance_mode(self, ultra_fast: bool = True):
        """Set performance mode based on configuration"""
        if ultra_fast:
            # Use config values for ultra-fast mode
            self.move_step = self.config["movement"]["move_step"]
            self.resize_step = self.config["movement"]["resize_step"]
            self.key_repeat_rate = self.config["movement"]["key_repeat_rate"]
        else:
            # Use slower values
            self.move_step = max(1, self.move_step // 2)
            self.resize_step = max(1, self.resize_step // 2)
            self.key_repeat_rate = self.key_repeat_rate * 2
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {'avg_frame_time': 0, 'fps': 0}
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'avg_frame_time': avg_frame_time,
            'fps': fps,
            'queue_size': len(self.movement_queue),
            'is_movement_running': self.is_movement_running,
            'enable_continuous_movement': self.enable_continuous_movement,
            'enable_parallel_worker': self.enable_parallel_worker
        }
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config_manager = ConfigManager(self.config_manager.config_file)
        self.config = self.config_manager.get_circle_overlay_config()
        
        # Update circle properties
        circle_config = self.config["circle"]
        self.color = (circle_config["color_blue"], 
                     circle_config["color_green"], 
                     circle_config["color_red"])
        self.thickness = circle_config["thickness"]
        self.center_point_size = circle_config["center_point_size"]
        
        # Update movement settings
        movement_config = self.config["movement"]
        self.move_step = movement_config["move_step"]
        self.resize_step = movement_config["resize_step"]
        self.enable_continuous_movement = movement_config["enable_continuous_movement"]
        self.enable_parallel_worker = movement_config["enable_parallel_worker"]
        self.key_repeat_rate = movement_config["key_repeat_rate"]
        
        # Update keyboard settings
        keyboard_config = self.config["keyboard"]
        self.enable_key_repeat = keyboard_config["enable_key_repeat"]
        self.enable_simultaneous_keys = keyboard_config["enable_simultaneous_keys"]
        
        print("Configuration reloaded successfully!")
    
    def run_overlay(self, width: int = 800, height: int = 600):
        """Run the overlay as a separate process"""
        print(f"Starting Circle Overlay Process")
        print(f"Window: {self.window_name}")
        print(f"Overlay on: {self.overlay_on_window}")
        print("Controls:")
        for control, description in self.instructions.items():
            print(f"  {control}: {description}")
        
        # Create overlay window
        self.create_overlay_window(width, height)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Create overlay frame
                overlay_frame = self.draw_circle_overlay(width, height)
                
                # Display overlay frame
                cv2.imshow(self.window_name, overlay_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break
                
                # Handle continuous input
                self.handle_continuous_input()
                
                # Check if window is closed
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except AttributeError:
                    # Fallback check - just continue if property not available
                    pass
                    
        except KeyboardInterrupt:
            print("Circle overlay interrupted by user")
        except Exception as e:
            print(f"Error in overlay loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        self.is_movement_running = False
        
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=1.0)
        
        if self.window_created:
            cv2.destroyWindow(self.window_name)
        
        print("Circle overlay stopped")


# Backward compatibility
CircleOverlay = ParallelCircleOverlay


def run_circle_overlay_process(window_name: str = "Circle Overlay", 
                              overlay_on_window: str = "Live Core Detector",
                              width: int = 800, height: int = 600):
    """Run circle overlay as a separate process"""
    overlay = ParallelCircleOverlay(window_name, overlay_on_window)
    overlay.run_overlay(width, height)


def main():
    """Main function to run circle overlay as parallel process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel Circle Overlay")
    parser.add_argument("--window-name", default="Circle Overlay", 
                       help="Name of the overlay window")
    parser.add_argument("--overlay-on", default="Live Core Detector", 
                       help="Name of the window to overlay on")
    parser.add_argument("--width", type=int, default=800, 
                       help="Overlay window width")
    parser.add_argument("--height", type=int, default=600, 
                       help="Overlay window height")
    
    args = parser.parse_args()
    
    try:
        run_circle_overlay_process(
            window_name=args.window_name,
            overlay_on_window=args.overlay_on,
            width=args.width,
            height=args.height
        )
    except KeyboardInterrupt:
        print("\nCircle overlay interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 