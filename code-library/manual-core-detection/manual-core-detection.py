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
                "process_interval": 0.2,
                "enable_parallel_detection": True,
                "max_detection_workers": 2
            },
            "circle_overlay": {
                "initial_center_x": 320,
                "initial_center_y": 240,
                "initial_radius": 50,
                "move_step": 10,
                "resize_step": 5,
                "color_red": 255,
                "color_green": 0,
                "color_blue": 0,
                "thickness": 2,
                "center_point_size": 3
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
                "frame_time_history_size": 60,
                "target_fps": 120
            }
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                self._merge_config(default_config, user_config);
            except Exception as e:
                print(f"Error loading config file: {e}")

        return default_config

    def _merge_config(self, default_config: Dict, user_config: Dict):
        """Recursively merge user configuration with defaults"""
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_config(default_config[key], value);
            else:
                default_config[key] = value

    def get_config(self, section: str = None) -> Dict:
        """Get configuration section"""
        if section:
            return self.config.get(section, {});
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
                tl_factory = pylon.TlFactory.GetInstance();
                devices = tl_factory.EnumerateDevices();

                if len(devices) == 0:
                    print("No Pylon cameras found. Using webcam fallback.");
                    self.use_pylon = False
                else:
                    self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice());
                    self.camera.Open();

                    if self.camera.IsOpen():
                        # Fast configuration
                        try:
                            self.camera.PixelFormat.SetValue("RGB8");
                        except Exception:
                            pass
                        try:
                            self.camera.ExposureAuto.SetValue("Continuous");
                        except Exception:
                            pass

                        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly);
                        self.is_grabbing = True
                        print(f"Pylon camera initialized: {self.camera.GetDeviceInfo().GetModelName()}");
                    else:
                        print("Failed to open Pylon camera. Using webcam fallback.");
                        self.use_pylon = False

            except Exception as e:
                print(f"Error setting up Pylon camera: {e}");
                self.use_pylon = False

        if not self.use_pylon:
            self.camera = cv2.VideoCapture(self.camera_index);
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open webcam at index {self.camera_index}");
            print(f"Using webcam at index {self.camera_index}")

    def read_frame(self) -> Optional[np.ndarray]:
        """Fast frame reading with error handling"""
        if self.camera is None:
            return None

        try:
            if self.use_pylon and self.is_grabbing:
                try:
                    grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return);

                    if grab_result.GrabSucceeded():
                        image = grab_result.Array;
                        grab_result.Release();

                        if len(image.shape) == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB);

                        return image
                    else:
                        return None
                except Exception:
                    return None
            else:
                ret, frame = self.camera.read();
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                return None

        except Exception as e:
            print(f"Error reading frame: {e}");
            return None

    def release(self):
        """Release camera resources with error handling"""
        if self.camera is not None:
            try:
                if self.use_pylon and self.is_grabbing:
                    self.camera.StopGrabbing();
                elif hasattr(self.camera, "release"):
                    self.camera.release();
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


def geometric_detection(frame_umat: cv2.UMat, method_name: str = "geometric_approach") -> CoreDetectionResult:
    """Fast geometric approach for core detection using UMat for GPU acceleration"""
    result = CoreDetectionResult(method_name, time.time());
    start_time = time.time();

    try:
        gray_umat = cv2.cvtColor(frame_umat, cv2.COLOR_RGB2GRAY);
        height, width = gray_umat.get().shape[:2];

        # Fast preprocessing on the GPU
        blurred_umat = cv2.GaussianBlur(gray_umat, (9, 9), 2);

        # Optimized Hough detection on the GPU
        circles = cv2.HoughCircles(
            blurred_umat, cv2.HOUGH_GRADIENT, dp=1.5, minDist=width / 4,
            param1=100, param2=30, minRadius=10,
            maxRadius=int(height / 3)
        );

        if circles is None:
            result.error = "No circles detected";
            return result

        circles = np.uint16(np.around(circles));
        center_x, center_y, radius = circles[0, 0];

        # Simplified confidence for speed
        result.confidence = 0.9;  # Placeholder confidence

        result.center = (float(center_x), float(center_y));
        result.core_radius = float(radius);

    except Exception as e:
        result.error = str(e);

    result.execution_time = time.time() - start_time;
    return result


class InteractiveCircleOverlay:
    """Interactive circle overlay with no boundaries"""

    def __init__(self, config: Dict):
        circle_config = config["circle_overlay"];
        self.config_manager = ConfigManager();

        # Circle properties
        self.center = [circle_config["initial_center_x"], circle_config["initial_center_y"]];
        self.radius = circle_config["initial_radius"];
        self.color = (circle_config["color_blue"], circle_config["color_green"], circle_config["color_red"]);
        self.thickness = circle_config["thickness"];
        self.center_point_size = circle_config["center_point_size"];
        self.is_locked = False

        # Movement settings
        self.move_step = circle_config["move_step"];
        self.resize_step = circle_config["resize_step"];

        # Performance tracking
        self.frame_times = deque(maxlen=60);
        self.last_frame_time = time.time();

        # Instructions
        self.instructions = {
            "WASD": "Move circle (W=up, S=down, A=left, D=right)",
            "Q/E": "Resize circle (Q=smaller, E=larger)",
            "L": "Lock/Unlock circle position",
            "R": "Reset circle to center",
            "ESC": "Exit application"
        };

    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for circle control"""
        if key == 27:  # ESC key
            return False

        if key == ord('l') or key == ord('L'):
            self.is_locked = not self.is_locked;
        elif key == ord('r') or key == ord('R'):
            circle_config = self.config_manager.get_config("circle_overlay");
            self.center = [circle_config["initial_center_x"], circle_config["initial_center_y"]];
            self.radius = circle_config["initial_radius"];
        else:
            self._apply_movement(key);

        return True

    def _apply_movement(self, key: int):
        """Apply movement based on key press with no restrictions"""
        if self.is_locked:
            return

        if key in [ord('w'), ord('W')]:
            self.center[1] -= self.move_step;
        elif key in [ord('s'), ord('S')]:
            self.center[1] += self.move_step;
        elif key in [ord('a'), ord('A')]:
            self.center[0] -= self.move_step;
        elif key in [ord('d'), ord('D')]:
            self.center[0] += self.move_step;
        elif key in [ord('q'), ord('Q')]:
            self.radius = max(1, self.radius - self.resize_step);
        elif key in [ord('e'), ord('E')]:
            self.radius += self.resize_step

    def draw_circle_on_frame(self, frame_umat: cv2.UMat) -> cv2.UMat:
        """Draw circle overlay on a UMat frame"""
        center = (int(self.center[0]), int(self.center[1]));
        radius = int(self.radius);

        if radius > 0:
            cv2.circle(frame_umat, center, radius, self.color, self.thickness);
            cv2.circle(frame_umat, center, self.center_point_size, self.color, -1);

        if self.is_locked:
            lock_color = (0, 0, 255);  # Red for locked
            cv2.circle(frame_umat, center, radius + 5, lock_color, 1);

        current_time = time.time();
        self.frame_times.append(current_time - self.last_frame_time);
        self.last_frame_time = current_time;

        return frame_umat

    def get_circle_info(self) -> Dict:
        """Get circle information"""
        return {
            'center': tuple(self.center),
            'radius': self.radius,
            'color': self.color,
            'is_locked': self.is_locked,
        };

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.frame_times:
            return {'avg_frame_time': 0, 'fps': 0};

        avg_frame_time = sum(self.frame_times) / len(self.frame_times);
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0;

        return {
            'avg_frame_time': avg_frame_time,
            'fps': fps
        }


class UnifiedCoreDetector:
    """Unified application combining core detection and circle overlay"""

    def __init__(self, config_file: str = "config.json"):
        self.config_manager = ConfigManager(config_file);
        self.config = self.config_manager.get_config();

        camera_config = self.config["camera"];
        self.camera = PylonCamera(
            camera_index=camera_config["camera_index"],
            use_pylon=camera_config["use_pylon"]
        );

        self.circle_overlay = InteractiveCircleOverlay(self.config);

        self.is_running = False;
        self.frame_count = 0;
        self.start_time = time.time();
        self.last_process_time = 0;
        self.process_interval = self.config["detection"]["process_interval"];

        self.last_detection_results = [];

    def process_frame(self, frame_umat: cv2.UMat):
        """Process frame with core detection and update results"""
        self.frame_count += 1;
        result = geometric_detection(frame_umat);
        result.frame_number = self.frame_count;
        self.last_detection_results = [result];

    def draw_results_on_frame(self, frame_umat: cv2.UMat, results: List[CoreDetectionResult]) -> cv2.UMat:
        """Draw detection results on a UMat frame"""
        if not self.config["display"]["show_detections"]:
            return frame_umat

        for result in results:
            if result.error or not result.center or not result.core_radius:
                continue

            color = (0, 255, 0);  # Green for geometric approach
            center = (int(result.center[0]), int(result.center[1]));
            radius = int(result.core_radius);

            cv2.circle(frame_umat, center, radius, color, 2);
            cv2.circle(frame_umat, center, 3, color, -1);
            cv2.putText(frame_umat, "GEOMETRIC",
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        return frame_umat

    def add_info_overlay(self, frame_umat: cv2.UMat) -> cv2.UMat:
        """Add information overlay to a UMat frame"""
        if not self.config["display"]["show_info"]:
            return frame_umat

        font = cv2.FONT_HERSHEY_SIMPLEX;
        font_scale = 0.6;
        color = (255, 255, 255);
        thickness = 1;
        y_offset = 30;
        line_height = 25;

        # Overall FPS
        if self.config["display"]["show_fps"]:
            elapsed_time = time.time() - self.start_time;
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0;
            cv2.putText(frame_umat, f"FPS: {fps:.1f}", (10, y_offset), font, font_scale, color, thickness);
            y_offset += line_height;

        # Circle Info
        if self.config["display"]["show_circle_info"]:
            info = self.circle_overlay.get_circle_info();
            text = f"Circle: ({info['center'][0]:.0f}, {info['center'][1]:.0f}) R:{info['radius']:.0f}";
            cv2.putText(frame_umat, text, (10, y_offset), font, font_scale, color, thickness);
            y_offset += line_height;
            if info['is_locked']:
                cv2.putText(frame_umat, "LOCKED", (10, y_offset), font, font_scale, (0, 0, 255), thickness);
                y_offset += line_height;

        # Performance Stats
        if self.config["display"]["show_performance_stats"]:
            stats = self.circle_overlay.get_performance_stats();
            cv2.putText(frame_umat, f"Overlay FPS: {stats['fps']:.1f}", (10, y_offset), font, font_scale, color, thickness);

        return frame_umat

    def run(self):
        """Main application loop"""
        print("Starting Unified Core Detector with Interactive Circle Overlay");
        print("=" * 60);
        print("Controls:");
        for control, description in self.circle_overlay.instructions.items():
            print(f"  {control}: {description}");
        print("Press Ctrl+C to stop");

        window_name = self.config["display"]["window_name"];
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL);
        self.is_running = True;
        self.start_time = time.time();

        try:
            while self.is_running:
                frame = self.camera.read_frame();
                if frame is None:
                    time.sleep(0.01);
                    continue
                
                self.frame_count +=1;

                frame_umat = cv2.UMat(frame);

                current_time = time.time();
                if current_time - self.last_process_time >= self.process_interval:
                    self.process_frame(frame_umat);
                    self.last_process_time = current_time;

                display_umat = self.draw_results_on_frame(frame_umat, self.last_detection_results);
                display_umat = self.circle_overlay.draw_circle_on_frame(display_umat);
                display_umat = self.add_info_overlay(display_umat);

                display_frame = display_umat.get();
                cv2.imshow(window_name, display_frame);

                key = cv2.waitKey(1) & 0xFF;
                if not self.circle_overlay.handle_keyboard_input(key):
                    self.is_running = False;

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.is_running = False;

        except KeyboardInterrupt:
            print("Application interrupted by user");
        except Exception as e:
            print(f"Error in main loop: {e}");
            import traceback;
            traceback.print_exc();
        finally:
            self.cleanup();

    def cleanup(self):
        """Cleanup resources with error handling"""
        self.is_running = False;
        self.camera.release();
        cv2.destroyAllWindows();
        print("Application stopped")


def main():
    """Main function"""
    import argparse;

    parser = argparse.ArgumentParser(
        description="Unified Core Detector with Interactive Circle Overlay"
    );
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to configuration file (default: config.json)"
    );

    args = parser.parse_args();

    try:
        app = UnifiedCoreDetector(config_file=args.config);
        app.run();

    except KeyboardInterrupt:
        print("\nApplication interrupted by user");
    except Exception as e:
        print(f"Error: {e}");
        import traceback;
        traceback.print_exc();


if __name__ == "__main__":
    main()