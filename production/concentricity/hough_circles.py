"""
OpenCV Hough Circles Detection Module.
Provides circle detection functionality for real-time video processing.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional


class HoughCirclesDetector:
    """
    OpenCV Hough Circles detector with configurable parameters.
    """

    def __init__(self,
                 dp=1.0,
                 min_dist=50,
                 param1=100,
                 param2=50,
                 min_radius=5,
                 max_radius=200,
                 blur_kernel_size=9,
                 blur_sigma=2.0):
        """
        Initialize the Hough circles detector.

        Args:
            dp (float): Inverse ratio of the accumulator resolution to the image resolution (0.1-5.0)
            min_dist (int): Minimum distance between detected circle centers (1-1000)
            param1 (int): First method-specific parameter (upper threshold for edge detection) (1-500)
            param2 (int): Second method-specific parameter (accumulator threshold for center detection) (1-300)
            min_radius (int): Minimum circle radius (0-500)
            max_radius (int): Maximum circle radius (1-2000)
            blur_kernel_size (int): Gaussian blur kernel size (must be odd, 1-51)
            blur_sigma (float): Gaussian blur sigma value (0.1-10.0)
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma

        # Statistics
        self.circles_detected = 0
        self.frames_processed = 0

    def detect_circles(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect circles in the given frame using HoughCircles.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[np.ndarray], np.ndarray]:
                - Detected circles array (x, y, radius) or None if no circles found
                - Processed frame with circles drawn
        """
        if frame is None:
            return None, frame

        try:
            # Convert to grayscale for circle detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise with configurable parameters
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.dp,
                minDist=self.min_dist,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )

            # Create output frame
            output_frame = frame.copy()

            # Draw detected circles
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                self.circles_detected = len(circles)

                for (x, y, r) in circles:
                    # Draw the circle outline
                    cv2.circle(output_frame, (x, y), r, (0, 255, 0), 2)
                    # Draw the circle center
                    cv2.circle(output_frame, (x, y), 2, (0, 0, 255), 3)

                # Add circles count text
                text = f"Circles: {len(circles)}"
                cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                self.circles_detected = 0
                # Add "No circles" text
                cv2.putText(output_frame, "Circles: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 255), 2, cv2.LINE_AA)

            self.frames_processed += 1
            return circles, output_frame

        except Exception as e:
            logging.error(f"Error detecting circles: {e}")
            return None, frame

    def update_parameters(self, dp=None, min_dist=None, param1=None, param2=None,
                         min_radius=None, max_radius=None, blur_kernel_size=None, blur_sigma=None):
        """
        Update detection parameters dynamically with extensive range support.

        Args:
            dp (float, optional): Inverse ratio of the accumulator resolution (0.1-5.0)
            min_dist (int, optional): Minimum distance between circle centers (1-1000)
            param1 (int, optional): Upper threshold for edge detection (1-500)
            param2 (int, optional): Accumulator threshold for center detection (1-300)
            min_radius (int, optional): Minimum circle radius (0-500)
            max_radius (int, optional): Maximum circle radius (1-2000)
            blur_kernel_size (int, optional): Gaussian blur kernel size (1-51, must be odd)
            blur_sigma (float, optional): Gaussian blur sigma value (0.1-10.0)
        """
        if dp is not None:
            self.dp = max(0.1, min(5.0, dp))
        if min_dist is not None:
            self.min_dist = max(1, min(1000, min_dist))
        if param1 is not None:
            self.param1 = max(1, min(500, param1))
        if param2 is not None:
            self.param2 = max(1, min(300, param2))
        if min_radius is not None:
            self.min_radius = max(0, min(500, min_radius))
        if max_radius is not None:
            self.max_radius = max(1, min(2000, max_radius))
        if blur_kernel_size is not None:
            # Ensure kernel size is odd and within bounds
            kernel_size = max(1, min(51, blur_kernel_size))
            self.blur_kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if blur_sigma is not None:
            self.blur_sigma = max(0.1, min(10.0, blur_sigma))

        logging.info(f"Updated Hough circles parameters: dp={self.dp:.1f}, "
                    f"min_dist={self.min_dist}, param1={self.param1}, "
                    f"param2={self.param2}, min_radius={self.min_radius}, "
                    f"max_radius={self.max_radius}, blur_kernel={self.blur_kernel_size}, "
                    f"blur_sigma={self.blur_sigma:.1f}")

    def get_statistics(self) -> dict:
        """
        Get detection statistics.

        Returns:
            dict: Statistics including circles detected and frames processed
        """
        return {
            'circles_detected': self.circles_detected,
            'frames_processed': self.frames_processed,
            'detection_rate': self.circles_detected / max(1, self.frames_processed)
        }

    def reset_statistics(self):
        """Reset detection statistics."""
        self.circles_detected = 0
        self.frames_processed = 0


class HoughCirclesProcessor:
    """
    High-level processor that applies Hough circles detection to video streams.
    """

    def __init__(self, detector: HoughCirclesDetector = None):
        """
        Initialize the processor.

        Args:
            detector (HoughCirclesDetector, optional): Circle detector instance
        """
        self.detector = detector or HoughCirclesDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with circle detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with circles highlighted
        """
        if not self.processing_enabled or frame is None:
            return frame

        circles, processed_frame = self.detector.detect_circles(frame)
        return processed_frame

    def toggle_processing(self) -> bool:
        """
        Toggle circle detection processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Hough circles processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled
