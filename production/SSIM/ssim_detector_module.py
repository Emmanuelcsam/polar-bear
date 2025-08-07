"""
SSIM (Structural Similarity Index) Detection Module.
Provides SSIM-based difference detection functionality for real-time video processing.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List

# Check if scikit-image is available
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Using fallback difference method.")


class SSIMDetector:
    """
    SSIM-based detector with configurable parameters for difference detection.
    """

    def __init__(self,
                 ssim_threshold=0.95,
                 min_defect_area=50,
                 max_defect_area=5000,
                 blur_kernel_size=5,
                 blur_sigma=1.0,
                 use_manual_ssim=False,
                 ssim_window_size=11,
                 diff_threshold=30):
        """
        Initialize the SSIM detector.

        Args:
            ssim_threshold (float): SSIM threshold above which images are considered too similar (0.1-1.0)
            min_defect_area (int): Minimum defect area in pixels (10-1000)
            max_defect_area (int): Maximum defect area in pixels (100-50000)
            blur_kernel_size (int): Gaussian blur kernel size (must be odd, 1-31)
            blur_sigma (float): Gaussian blur sigma value (0.1-10.0)
            use_manual_ssim (bool): Use manual SSIM implementation instead of scikit-image
            ssim_window_size (int): Window size for manual SSIM calculation (3-31, must be odd)
            diff_threshold (int): Threshold for simple difference method (1-255)
        """
        self.ssim_threshold = ssim_threshold
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = blur_sigma
        self.use_manual_ssim = use_manual_ssim
        self.ssim_window_size = ssim_window_size if ssim_window_size % 2 == 1 else ssim_window_size + 1
        self.diff_threshold = diff_threshold

        # Reference image for comparison
        self.reference_image = None

        # Statistics
        self.defects_detected = 0
        self.frames_processed = 0
        self.current_ssim_score = 0.0

    def set_reference_image(self, ref_image: np.ndarray):
        """
        Set the reference image for SSIM comparison.

        Args:
            ref_image (np.ndarray): Reference image in BGR format
        """
        if ref_image is None:
            logging.error("Reference image is None")
            return

        # Convert to grayscale if needed
        if len(ref_image.shape) == 3:
            self.reference_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        else:
            self.reference_image = ref_image.copy()

        # Apply blur if needed
        if self.blur_kernel_size > 1:
            self.reference_image = cv2.GaussianBlur(
                self.reference_image,
                (self.blur_kernel_size, self.blur_kernel_size),
                self.blur_sigma
            )

        logging.info(f"Reference image set: {self.reference_image.shape}")

    def detect_differences(self, frame: np.ndarray) -> Tuple[Optional[List[dict]], np.ndarray]:
        """
        Detect differences between the frame and reference image using SSIM.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[List[dict]], np.ndarray]:
                - List of detected difference regions or None if no differences found
                - Processed frame with differences highlighted
        """
        if frame is None:
            logging.warning("Input frame is None")
            return None, np.zeros((480, 640, 3), dtype=np.uint8)

        if self.reference_image is None:
            # If no reference set, use the first frame as reference
            self.set_reference_image(frame)
            return None, frame

        try:
            self.frames_processed += 1

            # Convert frame to grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()

            # Apply blur if needed
            if self.blur_kernel_size > 1:
                gray_frame = cv2.GaussianBlur(
                    gray_frame,
                    (self.blur_kernel_size, self.blur_kernel_size),
                    self.blur_sigma
                )

            # Compute SSIM difference
            diff_mask, ssim_score = self._compute_ssim_difference(
                self.reference_image, gray_frame
            )
            self.current_ssim_score = ssim_score

            # Create output frame
            output_frame = frame.copy()

            if diff_mask is not None:
                # Find difference regions
                regions = self._find_difference_regions(diff_mask)

                if regions:
                    self.defects_detected += len(regions)

                    # Draw detected regions
                    for region in regions:
                        x, y, w, h = region['bbox']
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # Draw center point
                        cx, cy = region['centroid']
                        cv2.circle(output_frame, (cx, cy), 3, (0, 255, 255), -1)

                        # Add area text
                        cv2.putText(output_frame, f"Area: {region['area']}",
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    return regions, output_frame

            return None, output_frame

        except Exception as e:
            logging.error(f"Error in SSIM detection: {e}")
            return None, frame

    def _compute_ssim_difference(self, ref_img: np.ndarray,
                                live_img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute SSIM difference between reference and live images.
        """
        # Ensure both images have the same size
        if ref_img.shape != live_img.shape:
            live_img = cv2.resize(live_img, (ref_img.shape[1], ref_img.shape[0]))

        if SKIMAGE_AVAILABLE and not self.use_manual_ssim:
            # Use scikit-image SSIM implementation
            try:
                (score, diff) = ssim(ref_img, live_img, full=True)
                diff = (diff * 255).astype("uint8")

                logging.info(f"SSIM score: {score:.6f}, threshold: {self.ssim_threshold}")

                if score > self.ssim_threshold:
                    # Images are too similar - no significant defects
                    return None, score

                # Threshold the difference image to get a binary mask
                _, thresh = cv2.threshold(diff, 0, 255,
                                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                return thresh, score
            except Exception as e:
                logging.warning(f"Scikit-image SSIM failed, using manual implementation: {e}")
                self.use_manual_ssim = True

        if self.use_manual_ssim:
            # Use manual SSIM implementation
            score, ssim_map = self._compute_ssim_manual(ref_img, live_img)

            logging.info(f"Manual SSIM score: {score:.6f}, threshold: {self.ssim_threshold}")

            if score > self.ssim_threshold:
                return None, score

            # Convert SSIM map to difference mask
            diff_map = 1.0 - ssim_map
            diff_map = (diff_map * 255).astype(np.uint8)

            # Threshold to get binary mask
            _, thresh = cv2.threshold(diff_map, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return thresh, score
        else:
            # Fallback to simple absolute difference
            diff = cv2.absdiff(ref_img, live_img)
            score = 1.0 - (np.mean(diff) / 255.0)

            logging.info(f"Simple diff score: {score:.6f}, threshold: {self.ssim_threshold}")

            # For large images, even small differences can be significant
            # Check if there are enough different pixels regardless of SSIM score
            non_zero_pixels = np.count_nonzero(diff)
            total_pixels = diff.shape[0] * diff.shape[1]
            diff_percentage = non_zero_pixels / total_pixels

            logging.info(f"Difference pixels: {non_zero_pixels}/{total_pixels} ({diff_percentage:.4f})")

            # If more than 0.1% of pixels are different, consider it significant
            if score > self.ssim_threshold and diff_percentage < 0.001:
                return None, score

            # Apply threshold to get binary mask
            _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

            # Additional morphological operations to clean up small noise
            if thresh is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            return thresh, score

    def _compute_ssim_manual(self, ref_img: np.ndarray,
                            live_img: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Manual SSIM implementation using OpenCV.
        """
        # SSIM constants
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        # Create Gaussian window
        kernel = cv2.getGaussianKernel(self.ssim_window_size, 1.5)
        window = np.outer(kernel, kernel.transpose())

        # Convert to float
        img1 = ref_img.astype(float)
        img2 = live_img.astype(float)

        # Compute local means
        mu1 = cv2.filter2D(img1, -1, window)
        mu2 = cv2.filter2D(img2, -1, window)

        # Compute local statistics
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = cv2.filter2D(img1**2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

        # SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast = (2 * np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2) / (sigma1_sq + sigma2_sq + C2)
        structure = (sigma12 + C2/2) / (np.sqrt(np.abs(sigma1_sq * sigma2_sq)) + C2/2)

        # Combine components
        ssim_map = luminance * contrast * structure
        ssim_index = np.mean(ssim_map)

        return float(ssim_index), ssim_map.astype(np.float32)

    def _find_difference_regions(self, diff_mask: np.ndarray) -> List[dict]:
        """
        Find and analyze regions of difference in a binary mask.
        """
        regions = []

        if diff_mask is None:
            return regions

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            diff_mask, connectivity=8)

        # Process each component (skip background at index 0)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if self.min_defect_area <= area <= self.max_defect_area:
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': int(area),
                    'centroid': (int(centroids[i][0]), int(centroids[i][1]))
                })

        return regions

    def update_parameters(self, ssim_threshold=None, min_defect_area=None, max_defect_area=None,
                         blur_kernel_size=None, blur_sigma=None, use_manual_ssim=None,
                         ssim_window_size=None, diff_threshold=None):
        """
        Update detection parameters dynamically.
        """
        if ssim_threshold is not None:
            self.ssim_threshold = max(0.1, min(1.0, ssim_threshold))
        if min_defect_area is not None:
            self.min_defect_area = max(10, min(1000, min_defect_area))
        if max_defect_area is not None:
            self.max_defect_area = max(100, min(50000, max_defect_area))
        if blur_kernel_size is not None:
            self.blur_kernel_size = max(1, min(31, blur_kernel_size))
            if self.blur_kernel_size % 2 == 0:
                self.blur_kernel_size += 1
        if blur_sigma is not None:
            self.blur_sigma = max(0.1, min(10.0, blur_sigma))
        if use_manual_ssim is not None:
            self.use_manual_ssim = use_manual_ssim
        if ssim_window_size is not None:
            self.ssim_window_size = max(3, min(31, ssim_window_size))
            if self.ssim_window_size % 2 == 0:
                self.ssim_window_size += 1
        if diff_threshold is not None:
            self.diff_threshold = max(1, min(255, diff_threshold))

        logging.info(f"Updated SSIM parameters: threshold={self.ssim_threshold:.2f}, "
                    f"min_area={self.min_defect_area}, max_area={self.max_defect_area}, "
                    f"blur_kernel={self.blur_kernel_size}, blur_sigma={self.blur_sigma:.1f}, "
                    f"manual_ssim={self.use_manual_ssim}, window_size={self.ssim_window_size}")

    def get_statistics(self) -> dict:
        """
        Get detection statistics.
        """
        return {
            'defects_detected': self.defects_detected,
            'frames_processed': self.frames_processed,
            'detection_rate': self.defects_detected / max(1, self.frames_processed),
            'current_ssim_score': self.current_ssim_score
        }

    def reset_statistics(self):
        """Reset detection statistics."""
        self.defects_detected = 0
        self.frames_processed = 0
        self.current_ssim_score = 0.0


class SSIMDetectorProcessor:
    """
    High-level processor that applies SSIM detection to video streams.
    """

    def __init__(self, detector: SSIMDetector = None):
        """
        Initialize the processor.

        Args:
            detector (SSIMDetector, optional): SSIM detector instance
        """
        self.detector = detector or SSIMDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with SSIM detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with differences highlighted
        """
        if not self.processing_enabled or frame is None:
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        detections, processed_frame = self.detector.detect_differences(frame)
        return processed_frame

    def set_reference_image(self, ref_image: np.ndarray):
        """Set reference image for SSIM comparison."""
        self.detector.set_reference_image(ref_image)

    def toggle_processing(self) -> bool:
        """
        Toggle SSIM detection processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"SSIM detection processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled
