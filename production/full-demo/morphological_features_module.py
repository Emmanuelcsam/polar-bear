#!/usr/bin/env python3
"""
Morphological Features Detection Module.
Provides morphological analysis functionality for real-time video processing.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Import the morphological features functions
from dev.morphological_features import (
    extract_morphological_features,
    detect_morphological_defects,
    extract_shape_complexity,
    extract_skeleton_features,
    apply_morphological_filter,
    detect_connected_components
)


class MorphologicalDetector:
    """
    Morphological features detector with configurable parameters.
    """

    def __init__(self,
                 analysis_types=['features', 'complexity', 'skeleton', 'defects', 'components'],
                 kernel_sizes=[3, 5, 7],
                 min_component_area=50,
                 defect_threshold=30,
                 filter_operation='gradient',
                 filter_kernel_size=5,
                 blur_kernel_size=5,
                 blur_sigma=1.0):
        """
        Initialize the morphological detector.

        Args:
            analysis_types (list): Types of analysis to perform
            kernel_sizes (list): Kernel sizes for multi-scale analysis (1-21)
            min_component_area (int): Minimum area for connected components (10-1000)
            defect_threshold (int): Threshold for defect detection (1-255)
            filter_operation (str): Morphological filter to apply
            filter_kernel_size (int): Size for morphological filter (1-21)
            blur_kernel_size (int): Gaussian blur kernel size (1-31, odd)
            blur_sigma (float): Gaussian blur sigma value (0.1-10.0)
        """
        self.analysis_types = analysis_types
        self.kernel_sizes = [max(1, min(21, k)) for k in kernel_sizes]
        self.min_component_area = max(10, min(1000, min_component_area))
        self.defect_threshold = max(1, min(255, defect_threshold))
        self.filter_operation = filter_operation
        self.filter_kernel_size = max(1, min(21, filter_kernel_size))
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.blur_sigma = max(0.1, min(10.0, blur_sigma))

        # Statistics
        self.features_extracted = 0
        self.defects_detected = 0
        self.components_found = 0
        self.frames_processed = 0

        # Available filter operations
        self.filter_operations = ['opening', 'closing', 'gradient', 'tophat', 'blackhat']

    def analyze_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Analyze morphological features in the given frame.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[Dict], np.ndarray]:
                - Dictionary of analysis results or None if error
                - Processed frame with visualizations
        """
        if frame is None:
            return None, np.zeros((100, 100, 3), dtype=np.uint8)

        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            if self.blur_kernel_size > 1:
                gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma)

            # Create output frame
            output_frame = frame.copy()

            # Analysis results
            results = {}

            # Extract morphological features
            if 'features' in self.analysis_types:
                features = extract_morphological_features(gray)
                results['features'] = features
                self.features_extracted = len(features)

            # Extract shape complexity
            if 'complexity' in self.analysis_types:
                complexity = extract_shape_complexity(gray)
                results['complexity'] = complexity

            # Extract skeleton features
            if 'skeleton' in self.analysis_types:
                skeleton_features = extract_skeleton_features(gray)
                results['skeleton'] = skeleton_features

            # Detect morphological defects
            if 'defects' in self.analysis_types:
                defect_maps = detect_morphological_defects(gray, self.kernel_sizes)
                results['defects'] = defect_maps

                # Count defects above threshold
                total_defects = 0
                for defect_map in defect_maps.values():
                    total_defects += np.sum(defect_map > self.defect_threshold)
                self.defects_detected = total_defects

            # Detect connected components
            if 'components' in self.analysis_types:
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                components = detect_connected_components(binary, self.min_component_area)
                results['components'] = components
                self.components_found = len(components)

            # Apply morphological filter for visualization
            filtered = apply_morphological_filter(gray, self.filter_operation, self.filter_kernel_size)

            # Overlay filtered result on original frame
            filtered_colored = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
            output_frame = cv2.addWeighted(output_frame, 0.7, filtered_colored, 0.3, 0)

            # Draw components if detected
            if 'components' in results:
                for comp in results['components'][:10]:  # Limit to first 10
                    x, y, w, h = comp['bbox']
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"A:{comp['area']}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Add summary text
            summary_lines = []
            if 'features' in results:
                summary_lines.append(f"Features: {len(results['features'])}")
            if 'components' in results:
                summary_lines.append(f"Components: {len(results['components'])}")
            if 'defects' in results:
                summary_lines.append(f"Defects: {self.defects_detected}")

            for i, line in enumerate(summary_lines):
                cv2.putText(output_frame, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Update statistics
            self.frames_processed += 1

            return results, output_frame

        except Exception as e:
            logging.error(f"Error in morphological analysis: {e}")
            return None, frame.copy()

    def update_parameters(self, analysis_types=None, kernel_sizes=None, min_component_area=None,
                         defect_threshold=None, filter_operation=None, filter_kernel_size=None,
                         blur_kernel_size=None, blur_sigma=None):
        """
        Update analysis parameters dynamically.

        Args:
            analysis_types (list, optional): Types of analysis to perform
            kernel_sizes (list, optional): Kernel sizes for multi-scale analysis (1-21)
            min_component_area (int, optional): Minimum component area (10-1000)
            defect_threshold (int, optional): Defect detection threshold (1-255)
            filter_operation (str, optional): Morphological filter operation
            filter_kernel_size (int, optional): Filter kernel size (1-21)
            blur_kernel_size (int, optional): Blur kernel size (1-31, odd)
            blur_sigma (float, optional): Blur sigma (0.1-10.0)
        """
        if analysis_types is not None:
            self.analysis_types = analysis_types
        if kernel_sizes is not None:
            self.kernel_sizes = [max(1, min(21, k)) for k in kernel_sizes]
        if min_component_area is not None:
            self.min_component_area = max(10, min(1000, min_component_area))
        if defect_threshold is not None:
            self.defect_threshold = max(1, min(255, defect_threshold))
        if filter_operation is not None and filter_operation in self.filter_operations:
            self.filter_operation = filter_operation
        if filter_kernel_size is not None:
            self.filter_kernel_size = max(1, min(21, filter_kernel_size))
        if blur_kernel_size is not None:
            value = max(1, min(31, blur_kernel_size))
            self.blur_kernel_size = value if value % 2 == 1 else value + 1
        if blur_sigma is not None:
            self.blur_sigma = max(0.1, min(10.0, blur_sigma))

        logging.info(f"Updated morphological parameters: types={self.analysis_types}, "
                    f"kernels={self.kernel_sizes}, min_area={self.min_component_area}, "
                    f"defect_thresh={self.defect_threshold}, filter={self.filter_operation}")

    def get_statistics(self) -> dict:
        """
        Get analysis statistics.

        Returns:
            dict: Statistics including features extracted and frames processed
        """
        return {
            'features_extracted': self.features_extracted,
            'defects_detected': self.defects_detected,
            'components_found': self.components_found,
            'frames_processed': self.frames_processed,
            'analysis_rate': self.frames_processed / max(1, self.frames_processed) if self.frames_processed > 0 else 0
        }

    def reset_statistics(self):
        """Reset analysis statistics."""
        self.features_extracted = 0
        self.defects_detected = 0
        self.components_found = 0
        self.frames_processed = 0


class MorphologicalProcessor:
    """
    High-level processor that applies morphological analysis to video streams.
    """

    def __init__(self, detector: MorphologicalDetector = None):
        """
        Initialize the processor.

        Args:
            detector (MorphologicalDetector, optional): Morphological detector instance
        """
        self.detector = detector or MorphologicalDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with morphological analysis.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with morphological visualizations
        """
        if not self.processing_enabled or frame is None:
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

        results, processed_frame = self.detector.analyze_frame(frame)
        return processed_frame

    def toggle_processing(self) -> bool:
        """
        Toggle morphological processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Morphological processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled

    def get_detector(self) -> MorphologicalDetector:
        """Get the underlying detector."""
        return self.detector

    def get_latest_results(self) -> Optional[Dict]:
        """Get the most recent analysis results."""
        if hasattr(self.detector, 'latest_results'):
            return self.detector.latest_results
        return None
