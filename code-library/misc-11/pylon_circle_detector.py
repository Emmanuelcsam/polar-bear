import argparse
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Pylon imports
try:
    from pypylon import pylon

    PYLON_AVAILABLE = True
except ImportError:
    print("Warning: Pylon SDK not found. Install with: pip install pypylon")
    print("Falling back to webcam mode.")
    PYLON_AVAILABLE = False


class CircleDetector:
    """Advanced circle detection with multiple algorithms and real-time processing."""

    def __init__(
        self,
        hough_params: Dict = None,
        contour_params: Dict = None,
        use_gpu: bool = False,
    ):
        """
        Initialize circle detector with configurable parameters.

        Args:
            hough_params: Parameters for Hough circle detection
            contour_params: Parameters for contour-based detection
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.hough_params = hough_params or {
            "dp": 1,  # Inverse ratio of accumulator resolution
            "min_dist": 20,  # Minimum distance between circles
            "param1": 50,  # Upper threshold for edge detection
            "param2": 30,  # Threshold for center detection
            "min_radius": 5,  # Minimum circle radius (very small)
            "max_radius": 1000,  # Maximum circle radius (very large)
        }

        self.contour_params = contour_params or {
            "min_area": 50,  # Minimum contour area (very small)
            "max_area": 500000,  # Maximum contour area (very large)
            "circularity_threshold": 0.5,  # Lower circularity threshold for more shapes
        }

        self.use_gpu = use_gpu
        self.detection_history = deque(maxlen=30)  # Store last 30 detections
        self.fps_history = deque(maxlen=30)

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0

        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None

    def detect_circles_hough(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles using Hough Circle Transform.

        Args:
            image: Input image (grayscale)

        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_params["dp"],
            minDist=self.hough_params["min_dist"],
            param1=self.hough_params["param1"],
            param2=self.hough_params["param2"],
            minRadius=self.hough_params["min_radius"],
            maxRadius=self.hough_params["max_radius"],
        )

        if circles is not None:
            # Convert to list of tuples
            circles_array = np.round(circles[0, :]).astype("int")
            return [tuple(circle) for circle in circles_array]
        return []

    def detect_circles_contour(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles using contour analysis.

        Args:
            image: Input image (grayscale)

        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        circles = []

        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            if (
                area >= self.contour_params["min_area"]
                and area <= self.contour_params["max_area"]
            ):
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    if circularity >= self.contour_params["circularity_threshold"]:
                        # Fit circle to contour
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        circles.append((int(x), int(y), int(radius)))

        return circles

    def detect_circles_combined(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Combine Hough and contour detection for better results.

        Args:
            image: Input image (grayscale)

        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        hough_circles = self.detect_circles_hough(image)
        contour_circles = self.detect_circles_contour(image)

        # Combine and remove duplicates
        # Convert numpy arrays to lists if needed
        if isinstance(hough_circles, np.ndarray):
            hough_circles = hough_circles.tolist()
        elif not isinstance(hough_circles, list):
            hough_circles = list(hough_circles)

        all_circles = hough_circles + contour_circles
        unique_circles = self._remove_duplicate_circles(all_circles)

        return unique_circles

    def _remove_duplicate_circles(
        self, circles: List[Tuple[int, int, int]], threshold: int = 20
    ) -> List[Tuple[int, int, int]]:
        """
        Remove duplicate circles based on center distance.

        Args:
            circles: List of (x, y, radius) tuples
            threshold: Distance threshold for considering circles as duplicates

        Returns:
            List of unique circles
        """
        if not circles:
            return []

        unique_circles = [circles[0]]

        for circle in circles[1:]:
            is_duplicate = False
            for unique_circle in unique_circles:
                distance = np.sqrt(
                    (circle[0] - unique_circle[0]) ** 2
                    + (circle[1] - unique_circle[1]) ** 2
                )
                if distance < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_circles.append(circle)

        return unique_circles

    def draw_circles(
        self,
        image: np.ndarray,
        circles: List[Tuple[int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detected circles on the image.

        Args:
            image: Input image
            circles: List of (x, y, radius) tuples
            color: BGR color tuple
            thickness: Line thickness

        Returns:
            Image with drawn circles
        """
        result = image.copy()

        for x, y, radius in circles:
            # Draw circle
            cv2.circle(result, (x, y), radius, color, thickness)
            # Draw center point
            cv2.circle(result, (x, y), 2, color, -1)

        return result

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > 0:
            self.current_fps = self.frame_count / elapsed_time
            self.fps_history.append(self.current_fps)

    def get_average_fps(self) -> float:
        """Get average FPS over the last 30 frames."""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0


class PylonCamera:
    """Camera interface supporting both Pylon and webcam."""

    def __init__(
        self, camera_index: int = 0, use_pylon: bool = True, camera_config: Dict = None
    ):
        """
        Initialize camera interface.

        Args:
            camera_index: Camera index for webcam
            use_pylon: Whether to use Pylon SDK
            camera_config: Camera configuration settings
        """
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.camera_config = camera_config or {}
        self.camera = None
        self.is_grabbing = False
        self.setup_camera()

    def setup_camera(self):
        """Setup camera (Pylon or webcam)."""
        if self.use_pylon:
            try:
                # Get the transport layer factory
                tl_factory = pylon.TlFactory.GetInstance()

                # Get all attached devices
                devices = tl_factory.EnumerateDevices()

                if len(devices) == 0:
                    print("No Pylon cameras found. Using webcam fallback.")
                    self.use_pylon = False
                else:
                    # Use the first available camera
                    self.camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())

                    # Open camera
                    self.camera.Open()

                    if self.camera.IsOpen():
                        # Configure camera settings
                        self._configure_pylon_camera()
                        print(
                            f"Pylon camera initialized: {self.camera.GetDeviceInfo().GetModelName()}"
                        )
                    else:
                        print("Failed to open Pylon camera. Using webcam fallback.")
                        self.use_pylon = False

            except Exception as e:
                print(f"Error setting up Pylon camera: {e}")
                self.use_pylon = False

        if not self.use_pylon:
            # Fallback to webcam
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(
                    f"Failed to open webcam at index {self.camera_index}"
                )
            print(f"Using webcam at index {self.camera_index}")

    def _configure_pylon_camera(self):
        """Configure Pylon camera settings."""
        try:
            # Set pixel format to RGB8, fallback to Mono8 if not supported
            try:
                self.camera.PixelFormat.SetValue("RGB8")
                print("  Set pixel format to RGB8")
            except Exception:
                print(
                    f"  Camera doesn't support RGB8, using current format: {self.camera.PixelFormat.GetValue()}"
                )

            # Configure exposure settings
            auto_exposure = self.camera_config.get("auto_exposure", True)
            exposure_time = self.camera_config.get("exposure_time", 10000)
            gain = self.camera_config.get("gain", 0)

            if auto_exposure:
                try:
                    # Enable auto exposure
                    self.camera.ExposureAuto.SetValue("Continuous")
                    print("  Auto exposure enabled")
                except Exception:
                    print("  Could not enable auto exposure")
            else:
                try:
                    # Disable auto exposure and set manual values
                    self.camera.ExposureAuto.SetValue("Off")
                    self.camera.ExposureTime.SetValue(exposure_time)
                    print(f"  Manual exposure set to {exposure_time} microseconds")
                except Exception:
                    print("  Could not set manual exposure")

            # Set gain
            try:
                self.camera.Gain.SetValue(gain)
                print(f"  Gain set to {gain}")
            except Exception:
                print("  Could not set gain")

            # Enable continuous acquisition
            try:
                self.camera.AcquisitionMode.SetValue("Continuous")
            except Exception:
                print("  Could not set acquisition mode")

            # Set trigger mode to software
            try:
                self.camera.TriggerMode.SetValue("Off")
            except Exception:
                print("  Could not set trigger mode")

            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_grabbing = True

        except Exception as e:
            print(f"Error configuring Pylon camera: {e}")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.

        Returns:
            Frame as numpy array or None if failed
        """
        if self.camera is None:
            return None

        try:
            if self.use_pylon and self.is_grabbing:
                # Pylon camera
                try:
                    # Try to retrieve result with timeout
                    grab_result = self.camera.RetrieveResult(
                        500, pylon.TimeoutHandling_Return
                    )

                    if grab_result.GrabSucceeded():
                        # Convert to numpy array
                        image = grab_result.Array
                        grab_result.Release()

                        # Convert monochrome to RGB if needed
                        if len(image.shape) == 2:  # Monochrome
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                        return image
                    else:
                        # If grab failed, return None (will be handled by retry logic)
                        return None
                except Exception as e:
                    # Don't print error for every failed attempt to avoid spam
                    return None
            else:
                # Webcam
                ret, frame = self.camera.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None

        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            if self.use_pylon and self.is_grabbing:
                try:
                    self.camera.StopGrabbing()
                except Exception as e:
                    print(f"Error stopping Pylon camera: {e}")
            elif hasattr(self.camera, "release"):
                self.camera.release()


class CircleDetectionApp:
    """Main application for real-time circle detection."""

    def __init__(
        self,
        camera_index: int = 0,
        use_pylon: bool = True,
        use_gpu: bool = False,
        config_file: str = None,
    ):
        """
        Initialize the application.

        Args:
            camera_index: Camera index
            use_pylon: Whether to use Pylon SDK
            use_gpu: Whether to use GPU acceleration
            config_file: Path to configuration file
        """
        self.camera_index = camera_index
        self.use_pylon = use_pylon
        self.use_gpu = use_gpu
        self.config_file = config_file

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize camera
        self.camera = PylonCamera(camera_index, use_pylon, self.config["camera"])

        # Initialize detector
        self.detector = CircleDetector(
            hough_params=self.config["hough_params"],
            contour_params=self.config["contour_params"],
            use_gpu=use_gpu,
        )

        # Application state
        self.is_running = False
        self.detection_method = "combined"  # hough, contour, combined
        self.show_controls = False

        # Create output directory
        os.makedirs(self.config["output"]["directory"], exist_ok=True)

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "hough_params": {
                "dp": 1,
                "min_dist": 20,
                "param1": 50,
                "param2": 30,
                "min_radius": 5,  # Very small circles
                "max_radius": 1000,  # Very large circles
            },
            "contour_params": {
                "min_area": 50,  # Very small areas
                "max_area": 500000,  # Very large areas
                "circularity_threshold": 0.5,  # More permissive
            },
            "display": {
                "window_name": "Circle Detection",
                "show_fps": True,
                "show_circles": True,
                "show_info": True,
            },
            "camera": {
                "auto_exposure": True,  # Enable/disable auto exposure
                "exposure_time": 10000,  # Manual exposure time in microseconds
                "gain": 0,  # Manual gain value
            },
            "output": {
                "directory": "output",
                "save_frames": True,
                "record_video": False,
                "video_fps": 30,
            },
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                for key in user_config:
                    if key in default_config:
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")

        return default_config

    def create_control_window(self):
        """Create control window with sliders for parameter adjustment."""
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 400, 600)

        # Hough parameters
        cv2.createTrackbar(
            "Hough DP",
            "Controls",
            self.detector.hough_params["dp"],
            5,
            self._on_hough_dp_change,
        )
        cv2.createTrackbar(
            "Min Dist",
            "Controls",
            self.detector.hough_params["min_dist"],
            100,
            self._on_hough_min_dist_change,
        )
        cv2.createTrackbar(
            "Param1",
            "Controls",
            self.detector.hough_params["param1"],
            200,
            self._on_hough_param1_change,
        )
        cv2.createTrackbar(
            "Param2",
            "Controls",
            self.detector.hough_params["param2"],
            100,
            self._on_hough_param2_change,
        )
        cv2.createTrackbar(
            "Min Radius",
            "Controls",
            self.detector.hough_params["min_radius"],
            50,
            self._on_hough_min_radius_change,
        )
        cv2.createTrackbar(
            "Max Radius",
            "Controls",
            self.detector.hough_params["max_radius"],
            2000,
            self._on_hough_max_radius_change,
        )

        # Contour parameters
        cv2.createTrackbar(
            "Min Area",
            "Controls",
            self.detector.contour_params["min_area"],
            1000,
            self._on_contour_min_area_change,
        )
        cv2.createTrackbar(
            "Max Area",
            "Controls",
            self.detector.contour_params["max_area"],
            1000000,
            self._on_contour_max_area_change,
        )
        cv2.createTrackbar(
            "Circularity",
            "Controls",
            int(self.detector.contour_params["circularity_threshold"] * 100),
            100,
            self._on_circularity_threshold_change,
        )

        # Camera parameters
        cv2.createTrackbar(
            "Auto Exposure",
            "Controls",
            1 if self.config["camera"]["auto_exposure"] else 0,
            1,
            self._on_auto_exposure_change,
        )

    def _on_hough_dp_change(self, value):
        self.detector.hough_params["dp"] = max(1, value)

    def _on_hough_min_dist_change(self, value):
        self.detector.hough_params["min_dist"] = value

    def _on_hough_param1_change(self, value):
        self.detector.hough_params["param1"] = value

    def _on_hough_param2_change(self, value):
        self.detector.hough_params["param2"] = value

    def _on_hough_min_radius_change(self, value):
        self.detector.hough_params["min_radius"] = value

    def _on_hough_max_radius_change(self, value):
        self.detector.hough_params["max_radius"] = value

    def _on_contour_min_area_change(self, value):
        self.detector.contour_params["min_area"] = value

    def _on_contour_max_area_change(self, value):
        self.detector.contour_params["max_area"] = value

    def _on_circularity_threshold_change(self, value):
        self.detector.contour_params["circularity_threshold"] = value / 100.0

    def _on_auto_exposure_change(self, value):
        self.config["camera"]["auto_exposure"] = bool(value)
        self.camera.camera_config["auto_exposure"] = bool(value)
        if self.camera.camera is not None:
            try:
                if self.config["camera"]["auto_exposure"]:
                    self.camera.camera.ExposureAuto.SetValue("Continuous")
                else:
                    self.camera.camera.ExposureAuto.SetValue("Off")
            except Exception as e:
                print(f"Error setting auto exposure: {e}")

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Process a single frame for circle detection.

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed_frame, detected_circles)
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles based on selected method
        if self.detection_method == "hough":
            circles = self.detector.detect_circles_hough(blurred)
        elif self.detection_method == "contour":
            circles = self.detector.detect_circles_contour(blurred)
        else:  # combined
            circles = self.detector.detect_circles_combined(blurred)

        # Draw circles on the frame
        if self.config["display"]["show_circles"]:
            frame = self.detector.draw_circles(frame, circles)

        # Add information overlay
        if self.config["display"]["show_info"]:
            frame = self._add_info_overlay(frame, circles)

        return frame, circles

    def _add_info_overlay(
        self, frame: np.ndarray, circles: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """
        Add information overlay to the frame.

        Args:
            frame: Input frame
            circles: Detected circles

        Returns:
            Frame with information overlay
        """
        overlay = frame.copy()

        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2

        # FPS information
        if self.config["display"]["show_fps"]:
            fps_text = f"FPS: {self.detector.current_fps:.1f}"
            cv2.putText(overlay, fps_text, (10, 30), font, font_scale, color, thickness)

        # Detection method
        method_text = f"Method: {self.detection_method.upper()}"
        cv2.putText(overlay, method_text, (10, 60), font, font_scale, color, thickness)

        # Circle count
        circle_text = f"Circles: {len(circles)}"
        cv2.putText(overlay, circle_text, (10, 90), font, font_scale, color, thickness)

        # Recording status
        if self.detector.is_recording:
            record_text = "RECORDING"
            cv2.putText(
                overlay,
                record_text,
                (10, 120),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
            )

        # Add semi-transparent overlay
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def start_recording(self):
        """Start video recording."""
        if not self.detector.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                self.config["output"]["directory"],
                f"circle_detection_{timestamp}.avi",
            )

            # Get frame dimensions from camera
            test_frame = self.camera.read_frame()
            if test_frame is not None:
                height, width = test_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.detector.video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    self.config["output"]["video_fps"],
                    (width, height),
                )
                self.detector.is_recording = True
                self.detector.recording_path = video_path
                print(f"Started recording: {video_path}")

    def stop_recording(self):
        """Stop video recording."""
        if self.detector.is_recording and self.detector.video_writer is not None:
            self.detector.video_writer.release()
            self.detector.is_recording = False
            print(f"Stopped recording: {self.detector.recording_path}")

    def save_frame(self, frame: np.ndarray, circles: List[Tuple[int, int, int]]):
        """Save current frame with detected circles."""
        if self.config["output"]["save_frames"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(
                self.config["output"]["directory"],
                f"frame_{timestamp}.jpg",
            )

            # Save frame with circles
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Save circle information
            info_path = frame_path.replace(".jpg", "_circles.txt")
            with open(info_path, "w") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Detection Method: {self.detection_method}\n")
                f.write(f"Number of Circles: {len(circles)}\n")
                f.write("Circle Details:\n")
                for i, (x, y, radius) in enumerate(circles):
                    f.write(f"  Circle {i+1}: Center=({x}, {y}), Radius={radius}\n")

            print(f"Saved frame: {frame_path}")

    def run(self):
        """Main application loop."""
        print("Starting Circle Detection Application")
        print("Application is running continuously...")
        print("Press Ctrl+C to stop")

        # Create control window
        self.create_control_window()

        self.is_running = True
        frame_count = 0

        while self.is_running:
            # Read frame
            frame = self.camera.read_frame()
            if frame is None:
                # Just wait a bit and continue, don't count failures
                time.sleep(0.1)
                continue

            frame_count += 1

            # Process frame
            processed_frame, circles = self.process_frame(frame)

            # Update FPS
            self.detector.update_fps()

            # Write to video if recording
            if self.detector.is_recording and self.detector.video_writer is not None:
                self.detector.video_writer.write(
                    cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                )

            # Display frame
            cv2.imshow(
                self.config["display"]["window_name"],
                cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR),
            )

            # Check for window close or escape key to exit
            key = cv2.waitKey(1) & 0xFF
            if (
                key == 27
                or cv2.getWindowProperty(
                    self.config["display"]["window_name"], cv2.WND_PROP_VISIBLE
                )
                < 1
            ):
                break

        # Cleanup
        self.stop_recording()
        self.camera.release()
        cv2.destroyAllWindows()
        print("Application stopped")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Real-time Circle Detection with Pylon Camera"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-pylon", action="store_true", help="Disable Pylon SDK and use webcam only"
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU acceleration (if available)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for recordings and captures",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    try:
        # Create and run application
        app = CircleDetectionApp(
            camera_index=args.camera,
            use_pylon=not args.no_pylon,
            use_gpu=args.gpu,
            config_file=args.config,
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
