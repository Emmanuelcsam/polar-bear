"""
OpenCV Hough Lines Detection Module.
Provides line detection functionality for real-time video processing (scratch detection).
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional


class HoughLinesDetector:
    """
    OpenCV Hough Lines detector with configurable parameters for scratch detection.
    """

    def __init__(self,
                 rho=1,
                 theta_degrees=1,
                 threshold=50,
                 min_line_length=30,
                 max_line_gap=5,
                 blur_kernel_size=5,
                 blur_sigma=1.0,
                 canny_low=50,
                 canny_high=150,
                 use_probabilistic=True):
        """
        Initialize the Hough lines detector.

        Args:
            rho (int): Distance resolution of the accumulator in pixels (1-10)
            theta_degrees (float): Angle resolution of the accumulator in degrees (0.1-5.0)
            threshold (int): Accumulator threshold for line detection (10-300)
            min_line_length (int): Minimum line length for probabilistic Hough (5-200)
            max_line_gap (int): Maximum allowed gap between line segments (1-50)
            blur_kernel_size (int): Gaussian blur kernel size (must be odd, 1-15)
            blur_sigma (float): Gaussian blur sigma value (0.1-5.0)
            canny_low (int): Lower threshold for Canny edge detection (10-200)
            canny_high (int): Upper threshold for Canny edge detection (50-400)
            use_probabilistic (bool): Use probabilistic Hough transform (better for line segments)
        """
        self.rho = rho
        self.theta_degrees = theta_degrees
        self.theta = np.pi * theta_degrees / 180.0  # Convert to radians
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.use_probabilistic = use_probabilistic

        # Statistics
        self.lines_detected = 0
        self.frames_processed = 0

    def detect_lines(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect lines in the given frame using HoughLines or HoughLinesP.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[np.ndarray], np.ndarray]:
                - Detected lines array or None if no lines found
                - Processed frame with lines drawn
        """
        if frame is None:
            return None, frame

        try:
            # Convert to grayscale for line detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

            # Detect lines using Hough transform
            if self.use_probabilistic:
                # Probabilistic Hough Line Transform (better for line segments)
                lines = cv2.HoughLinesP(
                    edges,
                    rho=self.rho,
                    theta=self.theta,
                    threshold=self.threshold,
                    minLineLength=self.min_line_length,
                    maxLineGap=self.max_line_gap
                )
            else:
                # Standard Hough Line Transform
                lines = cv2.HoughLines(
                    edges,
                    rho=self.rho,
                    theta=self.theta,
                    threshold=self.threshold
                )

            # Create output frame
            output_frame = frame.copy()

            # Draw detected lines
            if lines is not None:
                self.lines_detected = len(lines)

                if self.use_probabilistic:
                    # Draw line segments from probabilistic Hough
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw endpoints
                        cv2.circle(output_frame, (x1, y1), 3, (255, 0, 0), -1)
                        cv2.circle(output_frame, (x2, y2), 3, (0, 0, 255), -1)
                else:
                    # Draw infinite lines from standard Hough
                    h, w = gray.shape
                    for line in lines:
                        rho, theta = line[0]
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                        x0 = cos_theta * rho
                        y0 = sin_theta * rho
                        x1 = int(x0 + w * (-sin_theta))
                        y1 = int(y0 + w * cos_theta)
                        x2 = int(x0 - w * (-sin_theta))
                        y2 = int(y0 - w * cos_theta)
                        cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add lines count text
                text = f"Lines: {len(lines)}"
                cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                self.lines_detected = 0
                # Add "No lines" text
                cv2.putText(output_frame, "Lines: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 255), 2, cv2.LINE_AA)

            # Add detection method info
            method_text = "Method: Probabilistic" if self.use_probabilistic else "Method: Standard"
            cv2.putText(output_frame, method_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 0), 1, cv2.LINE_AA)

            self.frames_processed += 1
            return lines, output_frame

        except Exception as e:
            logging.error(f"Error detecting lines: {e}")
            return None, frame

    def update_parameters(self, rho=None, theta_degrees=None, threshold=None,
                         min_line_length=None, max_line_gap=None,
                         blur_kernel_size=None, blur_sigma=None,
                         canny_low=None, canny_high=None, use_probabilistic=None):
        """
        Update detection parameters dynamically.

        Args:
            rho (int, optional): Distance resolution (1-10)
            theta_degrees (float, optional): Angle resolution in degrees (0.1-5.0)
            threshold (int, optional): Accumulator threshold (10-300)
            min_line_length (int, optional): Minimum line length (5-200)
            max_line_gap (int, optional): Maximum line gap (1-50)
            blur_kernel_size (int, optional): Gaussian blur kernel size (1-15, must be odd)
            blur_sigma (float, optional): Gaussian blur sigma (0.1-5.0)
            canny_low (int, optional): Canny lower threshold (10-200)
            canny_high (int, optional): Canny upper threshold (50-400)
            use_probabilistic (bool, optional): Use probabilistic Hough transform
        """
        if rho is not None:
            self.rho = max(1, min(10, rho))
        if theta_degrees is not None:
            self.theta_degrees = max(0.1, min(5.0, theta_degrees))
            self.theta = np.pi * self.theta_degrees / 180.0
        if threshold is not None:
            self.threshold = max(10, min(300, threshold))
        if min_line_length is not None:
            self.min_line_length = max(5, min(200, min_line_length))
        if max_line_gap is not None:
            self.max_line_gap = max(1, min(50, max_line_gap))
        if blur_kernel_size is not None:
            # Ensure kernel size is odd and within bounds
            kernel_size = max(1, min(15, blur_kernel_size))
            self.blur_kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if blur_sigma is not None:
            self.blur_sigma = max(0.1, min(5.0, blur_sigma))
        if canny_low is not None:
            self.canny_low = max(10, min(200, canny_low))
        if canny_high is not None:
            self.canny_high = max(50, min(400, canny_high))
        if use_probabilistic is not None:
            self.use_probabilistic = use_probabilistic

        logging.info(f"Updated Hough lines parameters: rho={self.rho}, "
                    f"theta={self.theta_degrees:.1f}°, threshold={self.threshold}, "
                    f"min_length={self.min_line_length}, max_gap={self.max_line_gap}, "
                    f"blur_kernel={self.blur_kernel_size}, blur_sigma={self.blur_sigma:.1f}, "
                    f"canny_low={self.canny_low}, canny_high={self.canny_high}, "
                    f"probabilistic={self.use_probabilistic}")

    def get_statistics(self) -> dict:
        """
        Get detection statistics.

        Returns:
            dict: Statistics including lines detected and frames processed
        """
        return {
            'lines_detected': self.lines_detected,
            'frames_processed': self.frames_processed,
            'detection_rate': self.lines_detected / max(1, self.frames_processed)
        }

    def reset_statistics(self):
        """Reset detection statistics."""
        self.lines_detected = 0
        self.frames_processed = 0


class HoughLinesProcessor:
    """
    High-level processor that applies Hough lines detection to video streams.
    """

    def __init__(self, detector: HoughLinesDetector = None):
        """
        Initialize the processor.

        Args:
            detector (HoughLinesDetector, optional): Lines detector instance
        """
        self.detector = detector or HoughLinesDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with line detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with lines highlighted
        """
        if not self.processing_enabled or frame is None:
            return frame

        lines, processed_frame = self.detector.detect_lines(frame)
        return processed_frame

    def toggle_processing(self) -> bool:
        """
        Toggle line detection processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Hough lines processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled
