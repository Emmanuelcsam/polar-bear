#!/usr/bin/env python3
"""
Fast Unified Live Core Detector
Optimized for quick startup and real-time core detection.
"""

import argparse
import json
import os
import time
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fast Pylon import
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
except ImportError:
    pass


def _calculate_circle_confidence(gray: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
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


class PylonCamera:
    """Fast camera interface"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = True):
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.camera = None
        self.is_grabbing = False
        self.setup_camera()
        
    def setup_camera(self):
        """Fast camera setup"""
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
        """Fast frame reading"""
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
        """Release camera resources"""
        if self.camera is not None:
            if self.use_pylon and self.is_grabbing:
                try:
                    self.camera.StopGrabbing()
                except Exception as e:
                    print(f"Error stopping Pylon camera: {e}")
            elif hasattr(self.camera, "release"):
                self.camera.release()


class CoreDetectionMethods:
    """Fast core detection methods"""
    
    @staticmethod
    def geometric_approach(frame: np.ndarray, method_name: str = "geometric_approach") -> CoreDetectionResult:
        """Fast geometric approach"""
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
                param1=50, param2=25, minRadius=15, maxRadius=int(height / 4)
            )
            
            if circles is None:
                result.error = "No circles detected"
                return result
            
            circles = np.uint16(np.around(circles))
            center_x, center_y, radius = circles[0, 0]
            
            # Fast confidence calculation
            confidence = _calculate_circle_confidence(gray, center_x, center_y, radius)
            
            result.center = (float(center_x), float(center_y))
            result.core_radius = float(radius)
            result.confidence = confidence
            
        except Exception as e:
            result.error = str(e)
            
        result.execution_time = time.time() - start_time
        return result


class LiveTerminalLogger:
    """Fast terminal logging"""
    
    def __init__(self):
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.is_running = False
        
    def start(self):
        """Start logging thread"""
        self.is_running = True
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()
        
    def stop(self):
        """Stop logging thread"""
        self.is_running = False
        if self.log_thread:
            self.log_thread.join()
            
    def log(self, message: str, level: str = "INFO"):
        """Add message to log queue"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_queue.put(log_entry)
        
    def _log_worker(self):
        """Worker thread for logging"""
        while self.is_running:
            try:
                while not self.log_queue.empty():
                    message = self.log_queue.get_nowait()
                    print(message)
                    self.log_queue.task_done()
                time.sleep(0.01)
            except Exception as e:
                print(f"Logging error: {e}")


class UnifiedLiveCoreDetector:
    """Fast main application"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = True, 
                 config_file: str = None, output_dir: str = "output"):
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Fast initialization
        self.camera = PylonCamera(camera_index, use_pylon)
        self.logger = LiveTerminalLogger()
        
        # Application state
        self.is_running = False
        self.frame_count = 0
        self.detection_methods = [CoreDetectionMethods.geometric_approach]
        
        # Fast configuration
        self.config = self._load_config(config_file)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()
        self.last_process_time = 0
        self.process_interval = 0.1  # Process every 100ms
        
        # Parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
    def _load_config(self, config_file: str) -> Dict:
        """Fast configuration loading"""
        default_config = {
            "display": {
                "window_name": "Live Core Detection",
                "show_fps": True,
                "show_detections": True,
                "show_info": True,
            },
            "camera": {
                "auto_exposure": True,
                "exposure_time": 10000,
                "gain": 0,
            },
            "detection": {
                "min_confidence": 0.3,
                "max_detection_distance": 100,
                "learning_enabled": True,
                "process_interval": 0.1,
            },
            "output": {
                "save_frames": False,
                "record_video": False,
                "video_fps": 30,
            },
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
            except Exception as e:
                print(f"Error loading config file: {e}")
                
        return default_config
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[CoreDetectionResult], Optional[Dict]]:
        """Fast frame processing"""
        self.frame_count += 1
        results = []
        
        # Run detection methods in parallel
        futures = []
        for method in self.detection_methods:
            future = self.thread_pool.submit(method, frame)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                result.frame_number = self.frame_count
                results.append(result)
                
                if result.error:
                    self.logger.log(f"{result.method_name}: FAILED - {result.error}", "ERROR")
                else:
                    self.logger.log(
                        f"{result.method_name}: Core at {result.center}, "
                        f"radius={result.core_radius:.1f}, "
                        f"confidence={result.confidence:.3f}, "
                        f"time={result.execution_time:.3f}s"
                    )
            except Exception as e:
                self.logger.log(f"Detection method: EXCEPTION - {e}", "ERROR")
        
        # Draw results on frame
        processed_frame = self._draw_results(frame, results)
        
        return processed_frame, results, None
    
    def _draw_results(self, frame: np.ndarray, results: List[CoreDetectionResult]) -> np.ndarray:
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        # Draw geometric approach result
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
        
        # Add information overlay
        if self.config["display"]["show_info"]:
            result_frame = self._add_info_overlay(result_frame, results)
        
        return result_frame
    
    def _add_info_overlay(self, frame: np.ndarray, results: List[CoreDetectionResult]) -> np.ndarray:
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
        valid_results = [r for r in results if r.error is None]
        detection_text = f"Detections: {len(valid_results)}/{len(results)}"
        cv2.putText(overlay, detection_text, (10, y_offset), font, font_scale, color, thickness)
        
        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def run(self):
        """Fast main application loop"""
        self.logger.start()
        self.logger.log("Starting Fast Unified Live Core Detector")
        self.logger.log(f"Camera: {'Pylon' if self.use_pylon else 'Webcam'}")
        self.logger.log("Detection method: Geometric Approach Only")
        self.logger.log("Press Ctrl+C to stop")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Read frame
                frame = self.camera.read_frame()
                if frame is None:
                    time.sleep(0.01)  # Fast polling
                    continue
                
                current_time = time.time()
                
                # Process frame only at specified intervals
                if current_time - self.last_process_time >= self.process_interval:
                    processed_frame, results, _ = self.process_frame(frame)
                    self.last_process_time = current_time
                else:
                    # Just display the frame without processing
                    processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Display frame
                cv2.imshow(self.config["display"]["window_name"], processed_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(
                    self.config["display"]["window_name"], cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            self.logger.log("Interrupted by user")
        except Exception as e:
            self.logger.log(f"Error in main loop: {e}", "ERROR")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Fast cleanup"""
        self.is_running = False
        self.logger.stop()
        self.camera.release()
        cv2.destroyAllWindows()
        
        # Cleanup thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        self.logger.log("Application stopped")


def main():
    """Fast main function"""
    parser = argparse.ArgumentParser(
        description="Fast Unified Live Core Detector"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-pylon", action="store_true", help="Disable Pylon SDK and use webcam only"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run application
        app = UnifiedLiveCoreDetector(
            camera_index=args.camera,
            use_pylon=not args.no_pylon,
            config_file=args.config,
            output_dir=args.output
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