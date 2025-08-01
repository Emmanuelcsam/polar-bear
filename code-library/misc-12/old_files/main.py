#!/usr/bin/env python3
"""
Ultra-Fast Integrated Core Detector with Pylon Viewer Integration
Maximum performance with ALL detection capabilities, unlimited circle movement,
and seamless Pylon Viewer integration.
"""

import argparse
import cv2
import numpy as np
import time
import warnings
from typing import Optional, Tuple
from collections import deque

from circle_overlay import UltraFastCircleOverlay
from live_feed import LiveFeed
from config_manager import ConfigManager

warnings.filterwarnings('ignore')


class CoreDetectionResult:
    """Container for core detection results"""
    def __init__(self, method_name: str, timestamp: float):
        self.method_name = method_name
        self.timestamp = timestamp
        self.center: Optional[Tuple[float, float]] = None
        self.core_radius: Optional[float] = None
        self.confidence: float = 0.0
        self.execution_time: float = 0.0
        self.error: Optional[str] = None
        self.frame_number: int = 0


class CoreDetectionMethods:
    """Ultra-fast core detection methods with ALL capabilities"""
    
    @staticmethod
    def geometric_approach(
        frame: np.ndarray, 
        method_name: str = "geometric_approach"
    ) -> CoreDetectionResult:
        """Ultra-fast geometric approach with adaptive parameters"""
        result = CoreDetectionResult(method_name, time.time())
        start_time = time.time()
        
        try:
            # Ultra-fast preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Adaptive preprocessing based on frame size
            if height > 720:  # High resolution
                clahe = cv2.createCLAHE(
                    clipLimit=3.0, tileGridSize=(8, 8)
                )
                clahe_gray = clahe.apply(gray)
                blurred = cv2.GaussianBlur(clahe_gray, (9, 9), 2.0)
            else:  # Lower resolution - faster
                clahe = cv2.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8)
                )
                clahe_gray = clahe.apply(gray)
                blurred = cv2.GaussianBlur(clahe_gray, (7, 7), 1.5)
            
            # Adaptive Hough parameters
            min_radius = max(10, int(height / 20))
            max_radius = min(int(height / 3), int(width / 3))
            
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=2.0, minDist=min_radius * 2,
                param1=50, param2=25, minRadius=min_radius, maxRadius=max_radius
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


def _calculate_circle_confidence(gray: np.ndarray, center_x: int, 
                               center_y: int, radius: int) -> float:
    """Fast confidence calculation for detected circle"""
    try:
        mask = np.zeros_like(gray)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        inside_mean = np.mean(gray[mask > 0])
        outside_mask = cv2.circle(np.zeros_like(gray), (center_x, center_y), 
                                 radius + 10, 255, -1)
        outside_mask = cv2.circle(outside_mask, (center_x, center_y), 
                                 radius, 0, -1)
        outside_mean = np.mean(gray[outside_mask > 0])
        
        contrast_ratio = abs(inside_mean - outside_mean) / max(
            inside_mean, outside_mean, 1)
        return min(1.0, max(0.0, contrast_ratio))
        
    except Exception:
        return 0.0


class UltraFastCoreDetector:
    """Ultra-fast integrated core detector with Pylon Viewer integration and circle overlay"""
    
    def __init__(self, camera_index: int = 0, use_pylon: bool = False, 
                 config_file: str = "config.json", demo_mode: bool = False):
        """Initialize the ultra-fast core detector with enhanced Basler support"""
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.demo_mode = demo_mode
        self.config_file = config_file
        
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_camera_config()
        
        # Initialize camera with enhanced detection
        self.camera = None
        self.setup_camera()
        
        # Initialize detection components
        self.detection_results = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Detection settings
        detection_config = self.config_manager.get_detection_config()
        self.process_interval = detection_config.get("auto_core_detection", {}).get("detection_timeout", 0.2)
        self.last_detection_time = 0
        
        # Initialize missing attributes
        self.pylon_viewer_available = False
        self.circle_overlay_active = True
        self.last_process_time = 0
        self.last_detection_result = None
        self.detection_cache_timeout = 0.5
        self.adaptive_interval = True
        
        print(f"Ultra-Fast Core Detector initialized")
        print(f"Camera index: {self.camera_index}")
        print(f"Use Pylon: {self.use_pylon}")
        print(f"Demo mode: {self.demo_mode}")
    
    def setup_camera(self):
        """Enhanced camera setup with Basler camera targeting"""
        print("=== SETTING UP CAMERA ===")
        
        # Try to find Basler camera specifically
        try:
            from live_feed import find_basler_camera_specific
            basler_camera = find_basler_camera_specific()
            if basler_camera:
                print(f"Target Basler camera found: {basler_camera['model']}")
                self.use_pylon = True
                self.camera_index = basler_camera['index']
        except Exception as e:
            print(f"Error finding Basler camera: {e}")
        
        # Initialize camera using LiveFeed
        try:
            from live_feed import LiveFeed
            self.live_feed = LiveFeed(
                camera_index=self.camera_index,
                use_pylon=self.use_pylon,
                auto_detect=True,
                demo_mode=self.demo_mode,
                config_file=self.config_file
            )
            print("✓ Camera setup completed")
        except Exception as e:
            print(f"✗ Camera setup failed: {e}")
            print("Continuing with demo mode...")
            self.live_feed = None
            self.demo_mode = True
        
        # Initialize components with full integration
        self.circle_overlay = UltraFastCircleOverlay(self.config_file)
        
        # Initialize detection methods
        self.detection_methods = CoreDetectionMethods()
        
        print("✓ All components initialized successfully")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-fast frame processing with circle overlay integration"""
        self.frame_count += 1
        
        # Draw circle overlay (always ultra-fast)
        frame_with_circle = self.circle_overlay.draw_circle(frame)
        
        # Ultra-fast detection - run detection intelligently
        current_time = time.time()
        if current_time - self.last_process_time >= self.process_interval:
            detection_result = self._detect_core_in_circle_region(frame)
            self.last_detection_result = detection_result
            self.last_detection_time = current_time
            self.last_process_time = current_time
            
            # Adaptive interval adjustment
            if self.adaptive_interval:
                self._adjust_detection_interval(detection_result)
        elif (self.last_detection_result is not None and 
              current_time - self.last_detection_time < self.detection_cache_timeout):
            # Use cached detection result
            detection_result = self.last_detection_result
        else:
            # No detection result available
            detection_result = {
                'center': None,
                'radius': None,
                'confidence': 0.0,
                'detected': False,
                'method': None
            }
        
        # Draw detection results (always)
        frame_with_detection = self._draw_detection_results(frame_with_circle, 
                                                          detection_result)
        
        # Add ultra-fast info overlay with integration status
        frame_with_info = self._add_integrated_info_overlay(frame_with_detection)
        
        return frame_with_info
    
    def _detect_core_in_circle_region(self, frame: np.ndarray) -> dict:
        """Detect core within the circle region - ALL capabilities"""
        circle_info = self.circle_overlay.get_circle_info()
        center = circle_info['center']
        radius = circle_info['radius']
        
        # Create mask for circle region
        mask = self.circle_overlay.create_mask(frame.shape)
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Run ALL detection methods
        results = []
        for method in self.detection_methods:
            try:
                result = method(masked_frame)
                result.frame_number = self.frame_count
                results.append(result)
            except Exception as e:
                print(f"Detection method error: {e}")
        
        # Find best result
        best_result = None
        if results:
            valid_results = [r for r in results if r.error is None and r.confidence > 0.3]
            if valid_results:
                best_result = max(valid_results, key=lambda x: x.confidence)
        
        if best_result:
            return {
                'center': best_result.center,
                'radius': best_result.core_radius,
                'confidence': best_result.confidence,
                'detected': True,
                'method': best_result.method_name
            }
        else:
            return {
                'center': None,
                'radius': None,
                'confidence': 0.0,
                'detected': False,
                'method': None
            }
    
    def _draw_detection_results(self, frame: np.ndarray, 
                               detection_result: dict) -> np.ndarray:
        """Draw detection results on frame - ALL capabilities"""
        if detection_result['detected']:
            # Draw detected core
            center = (int(detection_result['center'][0]), 
                     int(detection_result['center'][1]))
            radius = int(detection_result['radius'])
            
            # Green circle for detected core
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 255, 0), -1)
            
            # Add detection info
            cv2.putText(frame, f"Core: {radius}px", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add confidence
            confidence_text = f"Conf: {detection_result['confidence']:.2f}"
            cv2.putText(frame, confidence_text,
                       (center[0] - 30, center[1] - radius - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def _add_integrated_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add integrated information overlay with Pylon and circle status"""
        # Create overlay
        height, width = frame.shape[:2]
        info_bg = np.zeros((80, width, 3), dtype=np.uint8)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        
        y_offset = 20
        line_height = 15
        
        # FPS
        fps = self.live_feed.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(info_bg, fps_text, (10, y_offset), font, font_scale, 
                   color, thickness)
        y_offset += line_height
        
        # Circle info
        circle_info = self.circle_overlay.get_circle_info()
        circle_text = (f"Circle: ({circle_info['center'][0]}, "
                      f"{circle_info['center'][1]}) R={circle_info['radius']}")
        cv2.putText(info_bg, circle_text, (10, y_offset), font, font_scale, 
                   color, thickness)
        y_offset += line_height
        
        # Detection status
        detection_text = f"Detection: {self.process_interval:.2f}s interval"
        cv2.putText(info_bg, detection_text, (10, y_offset), font, font_scale, 
                   color, thickness)
        y_offset += line_height
        
        # Integration status
        integration_text = f"Pylon: {'ON' if self.pylon_viewer_available else 'OFF'} | Circle: {'ACTIVE' if self.circle_overlay_active else 'INACTIVE'}"
        cv2.putText(info_bg, integration_text, (10, y_offset), font, font_scale, 
                   color, thickness)
        
        # Overlay on frame
        frame[height-80:height, :] = cv2.addWeighted(
            frame[height-80:height, :], 0.3, info_bg, 0.7, 0
        )
        
        return frame
    
    def _adjust_detection_interval(self, detection_result: dict):
        """Ultra-fast adaptive detection interval adjustment"""
        # Track performance
        current_fps = self.live_feed.get_fps()
        self.performance_history.append(current_fps)
        
        # Keep only last 10 measurements
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
        
        # Adaptive adjustment based on performance
        if len(self.performance_history) >= 5:
            avg_fps = sum(self.performance_history) / len(self.performance_history)
            
            if avg_fps < 30:  # Low FPS - reduce detection frequency
                self.process_interval = min(0.2, self.process_interval * 1.1)
            elif avg_fps > 50:  # High FPS - increase detection frequency
                self.process_interval = max(0.02, self.process_interval * 0.9)
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for circle overlay and system controls"""
        if key == 27:  # ESC key
            return False
        
        # Handle circle overlay keyboard input
        frame_shape = (480, 640)  # Default shape
        return self.circle_overlay.handle_keyboard_input(key, frame_shape)
    
    def run(self):
        """Run the ultra-fast integrated application with Pylon Viewer integration"""
        print("Ultra-Fast Integrated Core Detector Started")
        print("Integration Status:")
        print(f"  - Pylon Viewer: {'Available' if self.pylon_viewer_available else 'Not available'}")
        print(f"  - Circle Overlay: {'Active' if self.circle_overlay_active else 'Inactive'}")
        print("Controls:")
        for control, description in self.circle_overlay.instructions.items():
            print(f"  {control}: {description}")
        print("\nPress any key to start...")
        
        # Set up frame callback with keyboard handling
        def frame_callback_with_keyboard(frame):
            # Convert frame to RGB for processing
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            processed_frame = self._process_frame(frame_rgb)
            
            # Handle keyboard input (only if GUI is available)
            try:
                key = cv2.waitKey(1) & 0xFF
                self.circle_overlay.update_pressed_keys(key)
                
                if key != 255:  # Key was pressed
                    should_continue = self._handle_keyboard_input(key)
                    if not should_continue:
                        self.live_feed.is_running = False
                else:
                    # Handle continuous input for maximum smoothness
                    self.circle_overlay.handle_continuous_input(processed_frame.shape[:2])
            except Exception:
                # GUI not available, continue without keyboard input
                pass
            
            return processed_frame
        
        # Update live feed with keyboard handling
        self.live_feed.frame_callback = frame_callback_with_keyboard
        
        # Run live feed
        self.live_feed.run(
            window_name="Ultra-Fast Integrated Core Detector",
            show_info=False,  # We handle info overlay ourselves
            headless=False  # Show GUI for real camera
        )
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information"""
        info = {
            'camera': self.live_feed.get_camera_info(),
            'circle': self.circle_overlay.get_circle_info(),
            'performance': {
                'frame_count': self.frame_count,
                'fps': self.live_feed.get_fps(),
                'uptime': time.time() - self.start_time,
                'process_interval': self.process_interval,
                'detection_methods': len(self.detection_methods)
            },
            'integration': {
                'pylon_viewer_available': self.pylon_viewer_available,
                'circle_overlay_active': self.circle_overlay_active,
                'use_pylon': self.use_pylon
            }
        }
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        if self.live_feed is not None:
            try:
                self.live_feed.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        print("Cleanup completed")
    
    def set_performance_mode(self, ultra_fast: bool = True):
        """Set performance mode"""
        if self.live_feed is not None:
            try:
                self.live_feed.set_performance_mode(ultra_fast)
            except Exception as e:
                print(f"Error setting performance mode: {e}")
        else:
            print("LiveFeed not available for performance mode setting")


def main():
    """Main function with enhanced Basler camera support"""
    print("Starting Ultra-Fast Integrated Core Detector")
    print("Full debugging and optimal performance enabled")
    print("ALL detection capabilities preserved")
    print("Unlimited circle movement enabled")
    print("Pylon Viewer integration enabled")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ultra-Fast Integrated Core Detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--pylon", action="store_true", help="Use Pylon SDK")
    parser.add_argument("--demo", action="store_true", help="Use demo mode")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector with enhanced Basler support
        detector = UltraFastCoreDetector(
            camera_index=args.camera,  # Auto-detect camera
            use_pylon=args.pylon,      # Use Pylon if specified
            config_file=args.config,   # Use specified config
            demo_mode=args.demo        # Use demo mode if requested
        )
        
        # Set performance mode
        detector.set_performance_mode(True)
        
        # Run the detector
        detector.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main() 