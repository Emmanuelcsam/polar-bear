"""
OpenCV Blob Detection Module.
Provides blob detection functionality for real-time video processing.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional


class BlobDetector:
    """
    OpenCV-based blob detector with configurable parameters.
    """

    def __init__(self,
                 min_blob_area=50,
                 max_blob_area=5000,
                 min_blob_circularity=0.3,
                 blur_kernel_size=5,
                 blur_sigma=1.0,
                 threshold_value=127,
                 max_value=255,
                 threshold_type=cv2.THRESH_BINARY):
        """
        Initialize the blob detector.

        Args:
            min_blob_area (int): Minimum blob area in pixels (10-1000)
            max_blob_area (int): Maximum blob area in pixels (100-50000)
            min_blob_circularity (float): Minimum circularity threshold (0.1-1.0)
            blur_kernel_size (int): Gaussian blur kernel size (must be odd, 1-51)
            blur_sigma (float): Gaussian blur sigma value (0.1-10.0)
            threshold_value (int): Binary threshold value (1-255)
            max_value (int): Maximum value for thresholding (1-255)
            threshold_type: OpenCV threshold type
        """
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.min_blob_circularity = min_blob_circularity
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma
        self.threshold_value = threshold_value
        self.max_value = max_value
        self.threshold_type = threshold_type

        # Statistics
        self.blobs_detected = 0
        self.frames_processed = 0

    def detect_blobs(self, frame: np.ndarray) -> Tuple[Optional[List[dict]], np.ndarray]:
        """
        Detect blobs in the given frame using contour analysis.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[List[dict]], np.ndarray]:
                - List of detected blob dictionaries or None if no blobs found
                - Processed frame with blobs drawn
        """
        if frame is None:
            return None, frame

        try:
            # Convert to grayscale for blob detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

            # Apply binary thresholding
            _, thresh = cv2.threshold(blurred, self.threshold_value, self.max_value, self.threshold_type)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create output frame
            output_frame = frame.copy()

            # Analyze contours for blob-like shapes
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area
                if self.min_blob_area < area < self.max_blob_area:
                    perimeter = cv2.arcLength(contour, True)

                    # Skip degenerate contours
                    if perimeter == 0:
                        continue

                    # Calculate circularity
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)

                    # Filter by circularity (blob-like shape)
                    if circularity > self.min_blob_circularity:
                        x, y, w, h = cv2.boundingRect(contour)

                        # Calculate center and equivalent radius
                        center_x = x + w // 2
                        center_y = y + h // 2
                        equivalent_radius = int(np.sqrt(area / np.pi))

                        detection = {
                            "type": "Blob",
                            "location": (x, y, w, h),
                            "center": (center_x, center_y),
                            "area": area,
                            "circularity": circularity,
                            "radius": equivalent_radius,
                            "confidence": min(1.0, area / self.max_blob_area)
                        }
                        detections.append(detection)

                        # Draw the blob
                        # Draw bounding rectangle
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Draw center point
                        cv2.circle(output_frame, (center_x, center_y), 3, (0, 0, 255), -1)

                        # Draw equivalent circle
                        cv2.circle(output_frame, (center_x, center_y), equivalent_radius, (0, 255, 0), 2)

                        # Add text labels
                        label = f"Blob A:{int(area)} C:{circularity:.2f}"
                        cv2.putText(output_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # Update statistics
            self.blobs_detected = len(detections)
            self.frames_processed += 1

            # Add summary text
            summary_text = f"Blobs: {len(detections)}"
            cv2.putText(output_frame, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 255), 2, cv2.LINE_AA)

            return detections if detections else None, output_frame

        except Exception as e:
            logging.error(f"Error detecting blobs: {e}")
            return None, frame

    def update_parameters(self, min_blob_area=None, max_blob_area=None, min_blob_circularity=None,
                         blur_kernel_size=None, blur_sigma=None, threshold_value=None,
                         max_value=None, threshold_type=None):
        """
        Update detection parameters dynamically.

        Args:
            min_blob_area (int, optional): Minimum blob area (10-1000)
            max_blob_area (int, optional): Maximum blob area (100-50000)
            min_blob_circularity (float, optional): Minimum circularity (0.1-1.0)
            blur_kernel_size (int, optional): Gaussian blur kernel size (1-51, must be odd)
            blur_sigma (float, optional): Gaussian blur sigma (0.1-10.0)
            threshold_value (int, optional): Binary threshold value (1-255)
            max_value (int, optional): Maximum threshold value (1-255)
            threshold_type (optional): OpenCV threshold type
        """
        if min_blob_area is not None:
            self.min_blob_area = max(10, min(1000, min_blob_area))
        if max_blob_area is not None:
            self.max_blob_area = max(100, min(50000, max_blob_area))
        if min_blob_circularity is not None:
            self.min_blob_circularity = max(0.1, min(1.0, min_blob_circularity))
        if blur_kernel_size is not None:
            # Ensure kernel size is odd and within bounds
            kernel_size = max(1, min(51, blur_kernel_size))
            self.blur_kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if blur_sigma is not None:
            self.blur_sigma = max(0.1, min(10.0, blur_sigma))
        if threshold_value is not None:
            self.threshold_value = max(1, min(255, threshold_value))
        if max_value is not None:
            self.max_value = max(1, min(255, max_value))
        if threshold_type is not None:
            self.threshold_type = threshold_type

        logging.info(f"Updated blob detection parameters: min_area={self.min_blob_area}, "
                    f"max_area={self.max_blob_area}, min_circularity={self.min_blob_circularity:.2f}, "
                    f"blur_kernel={self.blur_kernel_size}, blur_sigma={self.blur_sigma:.1f}, "
                    f"threshold={self.threshold_value}")

    def get_statistics(self) -> dict:
        """
        Get detection statistics.

        Returns:
            dict: Statistics including blobs detected and frames processed
        """
        return {
            'blobs_detected': self.blobs_detected,
            'frames_processed': self.frames_processed,
            'detection_rate': self.blobs_detected / max(1, self.frames_processed)
        }

    def reset_statistics(self):
        """Reset detection statistics."""
        self.blobs_detected = 0
        self.frames_processed = 0


class BlobDetectorProcessor:
    """
    High-level processor that applies blob detection to video streams.
    """

    def __init__(self, detector: BlobDetector = None):
        """
        Initialize the processor.

        Args:
            detector (BlobDetector, optional): Blob detector instance
        """
        self.detector = detector or BlobDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with blob detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with blobs highlighted
        """
        if not self.processing_enabled or frame is None:
            return frame

        detections, processed_frame = self.detector.detect_blobs(frame)
        return processed_frame

    def toggle_processing(self) -> bool:
        """
        Toggle blob detection processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Blob detection processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled
