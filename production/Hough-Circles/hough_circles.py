#!/usr/bin/env python3
"""
OpenCV Hough Circles Detection Module.

This module provides robust circle detection functionality for real-time video processing
using OpenCV's HoughCircles algorithm. It includes configurable parameters for fine-tuning
detection sensitivity and performance.

Usage Example:
    # Basic usage with default parameters
    from hough_circles import HoughCirclesDetector, HoughCirclesProcessor
    
    # Initialize detector with custom parameters
    detector = HoughCirclesDetector(
        dp=1.0,              # Accumulator resolution ratio
        min_dist=50,         # Minimum distance between circle centers
        param1=100,          # Edge detection threshold
        param2=50,           # Center detection threshold
        min_radius=5,        # Minimum circle radius
        max_radius=200,      # Maximum circle radius
        blur_kernel_size=9,  # Gaussian blur kernel size
        blur_sigma=2.0       # Gaussian blur sigma
    )
    
    # Process a frame
    import cv2
    frame = cv2.imread('image.bmp')
    circles, processed_frame = detector.detect_circles(frame)
    
    # Use with video stream
    processor = HoughCirclesProcessor(detector)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if ret:
            processed = processor.process_frame(frame)
            cv2.imshow('Circles', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

Author: Vision System Development Team
Version: 1.0.0
Date: 2024
License: MIT
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional


class HoughCirclesDetector:
    """
    OpenCV Hough Circles detector with configurable parameters.
    
    This class implements the Hough Circle Transform algorithm for detecting
    circular shapes in images. It provides extensive parameter control for
    fine-tuning detection sensitivity and accuracy.
    
    Attributes:
        dp (float): Inverse ratio of accumulator resolution to image resolution
        min_dist (int): Minimum distance between detected circle centers
        param1 (int): Upper threshold for edge detection (Canny)
        param2 (int): Accumulator threshold for center detection
        min_radius (int): Minimum circle radius to detect
        max_radius (int): Maximum circle radius to detect
        blur_kernel_size (int): Size of Gaussian blur kernel (must be odd)
        blur_sigma (float): Standard deviation for Gaussian blur
        circles_detected (int): Number of circles in last processed frame
        frames_processed (int): Total frames processed since initialization
    
    Note:
        The detector automatically handles edge cases and provides
        robust error handling for various input conditions.
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
        Initialize the Hough circles detector with specified parameters.

        Args:
            dp (float, optional): Inverse ratio of the accumulator resolution to the 
                image resolution. Smaller values give higher resolution but are slower.
                Range: 0.1-5.0, Default: 1.0
            min_dist (int, optional): Minimum distance between detected circle centers.
                Prevents multiple detections of the same circle.
                Range: 1-1000 pixels, Default: 50
            param1 (int, optional): Upper threshold for the Canny edge detector.
                Higher values reduce noise but may miss weak edges.
                Range: 1-500, Default: 100
            param2 (int, optional): Accumulator threshold for circle center detection.
                Lower values detect more circles (including false positives).
                Range: 1-300, Default: 50
            min_radius (int, optional): Minimum circle radius to detect in pixels.
                Range: 0-500, Default: 5
            max_radius (int, optional): Maximum circle radius to detect in pixels.
                Range: 1-2000, Default: 200
            blur_kernel_size (int, optional): Size of Gaussian blur kernel.
                Must be odd. Larger values provide more smoothing.
                Range: 1-51 (odd numbers only), Default: 9
            blur_sigma (float, optional): Standard deviation for Gaussian blur.
                Higher values increase blur effect.
                Range: 0.1-10.0, Default: 2.0
        
        Raises:
            ValueError: If parameters are outside valid ranges.
        
        Example:
            >>> detector = HoughCirclesDetector(dp=1.5, min_dist=100, param1=150)
        """
        # Validate and set parameters with bounds checking
        self.dp = self._validate_parameter(dp, 0.1, 5.0, "dp")
        self.min_dist = self._validate_parameter(min_dist, 1, 1000, "min_dist", is_int=True)
        self.param1 = self._validate_parameter(param1, 1, 500, "param1", is_int=True)
        self.param2 = self._validate_parameter(param2, 1, 300, "param2", is_int=True)
        self.min_radius = self._validate_parameter(min_radius, 0, 500, "min_radius", is_int=True)
        self.max_radius = self._validate_parameter(max_radius, 1, 2000, "max_radius", is_int=True)
        
        # Ensure kernel size is odd
        blur_kernel_size = self._validate_parameter(blur_kernel_size, 1, 51, "blur_kernel_size", is_int=True)
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = self._validate_parameter(blur_sigma, 0.1, 10.0, "blur_sigma")

        # Statistics tracking
        self.circles_detected = 0
        self.frames_processed = 0
        
        # Log initialization
        logging.info(f"HoughCirclesDetector initialized with parameters: "
                    f"dp={self.dp:.1f}, min_dist={self.min_dist}, "
                    f"param1={self.param1}, param2={self.param2}, "
                    f"min_radius={self.min_radius}, max_radius={self.max_radius}")

    def _validate_parameter(self, value, min_val, max_val, name, is_int=False):
        """
        Validate and clamp a parameter value within specified bounds.
        
        Args:
            value: Parameter value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value  
            name (str): Parameter name for error messages
            is_int (bool): Whether to convert to integer
        
        Returns:
            Validated and clamped value
        
        Raises:
            ValueError: If value cannot be converted to appropriate type
        """
        try:
            if is_int:
                value = int(value)
            else:
                value = float(value)
            
            if value < min_val or value > max_val:
                logging.warning(f"Parameter '{name}' value {value} outside range [{min_val}, {max_val}]. Clamping.")
                value = max(min_val, min(max_val, value))
            
            return value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for parameter '{name}': {value}. Error: {e}")

    def detect_circles(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect circles in the given frame using HoughCircles algorithm.

        This method performs the following steps:
        1. Converts the frame to grayscale
        2. Applies Gaussian blur to reduce noise
        3. Runs HoughCircles detection
        4. Draws detected circles on the frame
        5. Updates statistics

        Args:
            frame (np.ndarray): Input frame in BGR format (height x width x 3)

        Returns:
            Tuple[Optional[np.ndarray], np.ndarray]:
                - circles: Array of detected circles with shape (n, 3) where each row 
                  contains [x, y, radius], or None if no circles found
                - output_frame: Processed frame with circles drawn (green outlines, 
                  red centers) and detection count overlay
        
        Raises:
            None: Exceptions are caught and logged, returning (None, original_frame)
        
        Example:
            >>> frame = cv2.imread('image.bmp')
            >>> circles, result = detector.detect_circles(frame)
            >>> if circles is not None:
            ...     print(f"Found {len(circles)} circles")
        """
        # Input validation
        if frame is None:
            logging.warning("Received None frame for circle detection")
            return None, frame
        
        if len(frame.shape) != 3:
            logging.error(f"Invalid frame shape: {frame.shape}. Expected 3D array (H,W,C)")
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

        except cv2.error as e:
            logging.error(f"OpenCV error during circle detection: {e}")
            return None, frame
        except Exception as e:
            logging.error(f"Unexpected error detecting circles: {e}", exc_info=True)
            return None, frame

    def update_parameters(self, dp=None, min_dist=None, param1=None, param2=None,
                         min_radius=None, max_radius=None, blur_kernel_size=None, blur_sigma=None):
        """
        Update detection parameters dynamically during runtime.
        
        This method allows real-time adjustment of detection parameters without
        recreating the detector instance. Only specified parameters are updated;
        others retain their current values.

        Args:
            dp (float, optional): Inverse ratio of accumulator resolution (0.1-5.0)
            min_dist (int, optional): Minimum distance between centers (1-1000 pixels)
            param1 (int, optional): Upper threshold for edge detection (1-500)
            param2 (int, optional): Accumulator threshold for centers (1-300)
            min_radius (int, optional): Minimum circle radius (0-500 pixels)
            max_radius (int, optional): Maximum circle radius (1-2000 pixels)
            blur_kernel_size (int, optional): Gaussian blur kernel (1-51, odd only)
            blur_sigma (float, optional): Gaussian blur sigma (0.1-10.0)
        
        Returns:
            None
        
        Note:
            Parameters are automatically clamped to valid ranges if out of bounds.
            Changes take effect on the next call to detect_circles().
        
        Example:
            >>> detector.update_parameters(param1=150, param2=30)  # More sensitive
            >>> detector.update_parameters(min_radius=50, max_radius=100)  # Specific size range
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
        Get comprehensive detection statistics.

        Returns:
            dict: Dictionary containing:
                - circles_detected (int): Circles found in last frame
                - frames_processed (int): Total frames analyzed
                - detection_rate (float): Average circles per frame
                - current_parameters (dict): Current detector settings
        
        Example:
            >>> stats = detector.get_statistics()
            >>> print(f"Detection rate: {stats['detection_rate']:.2f} circles/frame")
        """
        return {
            'circles_detected': self.circles_detected,
            'frames_processed': self.frames_processed,
            'detection_rate': self.circles_detected / max(1, self.frames_processed),
            'current_parameters': {
                'dp': self.dp,
                'min_dist': self.min_dist,
                'param1': self.param1,
                'param2': self.param2,
                'min_radius': self.min_radius,
                'max_radius': self.max_radius,
                'blur_kernel_size': self.blur_kernel_size,
                'blur_sigma': self.blur_sigma
            }
        }

    def reset_statistics(self):
        """
        Reset detection statistics to initial state.
        
        This clears the frame counter and circle detection count.
        Useful when starting a new detection session.
        
        Returns:
            None
        """
        self.circles_detected = 0
        self.frames_processed = 0
        logging.info("Detection statistics reset")


class HoughCirclesProcessor:
    """
    High-level processor for applying Hough circles detection to video streams.
    
    This class provides a simplified interface for integrating circle detection
    into video processing pipelines. It manages the detector instance and 
    provides convenient methods for frame processing and control.
    
    Attributes:
        detector (HoughCirclesDetector): The underlying circle detector
        processing_enabled (bool): Whether detection is currently active
    
    Example:
        >>> processor = HoughCirclesProcessor()
        >>> # Process video frames in a loop
        >>> while True:
        ...     frame = get_next_frame()  # Your frame source
        ...     result = processor.process_frame(frame)
        ...     display(result)
    """

    def __init__(self, detector: HoughCirclesDetector = None):
        """
        Initialize the processor with a detector instance.

        Args:
            detector (HoughCirclesDetector, optional): Custom detector instance.
                If None, creates a detector with default parameters.
        
        Example:
            >>> # Use default detector
            >>> processor = HoughCirclesProcessor()
            >>> 
            >>> # Use custom detector
            >>> custom_detector = HoughCirclesDetector(dp=2.0, min_dist=100)
            >>> processor = HoughCirclesProcessor(custom_detector)
        """
        self.detector = detector or HoughCirclesDetector()
        self.processing_enabled = True
        logging.info("HoughCirclesProcessor initialized")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with circle detection.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            np.ndarray: Processed frame with detected circles highlighted.
                Returns original frame if processing is disabled or frame is invalid.
        
        Note:
            Processing can be toggled on/off using toggle_processing() method.
        """
        if not self.processing_enabled:
            return frame
        
        if frame is None:
            logging.warning("Attempted to process None frame")
            return frame

        try:
            circles, processed_frame = self.detector.detect_circles(frame)
            return processed_frame
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def toggle_processing(self) -> bool:
        """
        Toggle circle detection processing on/off.

        Returns:
            bool: New processing state (True if enabled, False if disabled)
        
        Example:
            >>> if processor.toggle_processing():
            ...     print("Detection enabled")
            ... else:
            ...     print("Detection disabled")
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Hough circles processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is currently enabled.

        Returns:
            bool: True if processing is enabled, False otherwise
        """
        return self.processing_enabled
    
    def get_detector(self) -> HoughCirclesDetector:
        """
        Get the underlying detector instance.
        
        Returns:
            HoughCirclesDetector: The detector being used by this processor
        
        Example:
            >>> detector = processor.get_detector()
            >>> detector.update_parameters(param1=200)
        """
        return self.detector
    
    def set_detector(self, detector: HoughCirclesDetector):
        """
        Set a new detector instance.
        
        Args:
            detector (HoughCirclesDetector): New detector to use
        
        Returns:
            None
        
        Example:
            >>> new_detector = HoughCirclesDetector(dp=2.0)
            >>> processor.set_detector(new_detector)
        """
        if not isinstance(detector, HoughCirclesDetector):
            raise TypeError("Detector must be an instance of HoughCirclesDetector")
        self.detector = detector
        logging.info("Detector instance updated")
