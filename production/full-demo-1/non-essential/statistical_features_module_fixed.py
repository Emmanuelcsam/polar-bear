"""
Fixed Statistical Features Processing Module.
Provides stable real-time statistical feature extraction for video processing.
Simplified to avoid threading and synchronization issues.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
import os
import sys

# Add path to dev/modular_scripts
current_dir = os.path.dirname(__file__)
modular_scripts_path = os.path.join(current_dir, 'dev', 'modular_scripts')
if os.path.exists(modular_scripts_path):
    sys.path.insert(0, modular_scripts_path)

# Try to import statistical features functions
try:
    from statistical_features import (
        extract_all_statistical_features,
        compare_feature_vectors,
        extract_basic_statistics,
        extract_histogram_features,
        extract_texture_statistics,
        extract_moment_features
    )
    STATISTICAL_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import statistical_features module: {e}")
    STATISTICAL_FEATURES_AVAILABLE = False

    # Provide fallback implementations
    def extract_basic_statistics(gray):
        """Extract basic statistical features."""
        return {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'min': float(np.min(gray)),
            'max': float(np.max(gray)),
            'median': float(np.median(gray)),
            'var': float(np.var(gray)),
            'entropy': float(-np.sum(np.histogram(gray, bins=256, density=True)[0] *
                            np.log(np.histogram(gray, bins=256, density=True)[0] + 1e-10)))
        }

    def extract_histogram_features(gray, bins=32):
        """Extract histogram-based features."""
        hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
        hist = hist.astype(float) / hist.sum()  # Normalize

        return {
            'hist_mean': float(np.sum(hist * np.arange(bins))),
            'hist_std': float(np.sqrt(np.sum(hist * (np.arange(bins) - np.sum(hist * np.arange(bins)))**2))),
            'hist_skew': float(np.sum(hist * ((np.arange(bins) - np.sum(hist * np.arange(bins))) /
                              np.sqrt(np.sum(hist * (np.arange(bins) - np.sum(hist * np.arange(bins)))**2)))**3)),
            'hist_mode': float(np.argmax(hist))
        }

    def extract_texture_statistics(gray, window_size=5):
        """Extract texture statistics."""
        # Simple texture measures using local standard deviation
        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_std = np.sqrt(np.maximum(0, local_sq_mean - local_mean**2))

        return {
            'texture_contrast': float(np.mean(local_std)),
            'texture_homogeneity': float(1.0 / (1.0 + np.var(local_std))),
            'texture_energy': float(np.sum(local_std**2) / local_std.size),
            'texture_correlation': float(np.corrcoef(gray.flatten(), local_std.flatten())[0, 1] if local_std.size > 1 else 0.0)
        }

    def extract_moment_features(gray):
        """Extract moment-based features."""
        moments = cv2.moments(gray)

        if moments['m00'] != 0:
            centroid_x = moments['m10'] / moments['m00']
            centroid_y = moments['m01'] / moments['m00']
        else:
            centroid_x = gray.shape[1] / 2
            centroid_y = gray.shape[0] / 2

        # Normalize centroids to [0, 1]
        centroid_x_norm = centroid_x / gray.shape[1]
        centroid_y_norm = centroid_y / gray.shape[0]

        return {
            'moment_00': float(moments['m00']),
            'moment_10': float(moments['m10']),
            'moment_01': float(moments['m01']),
            'centroid_x': centroid_x_norm,
            'centroid_y': centroid_y_norm,
            'hu_moment_1': float(cv2.HuMoments(moments)[0][0]) if moments['m00'] > 0 else 0.0
        }

    def extract_all_statistical_features(gray, histogram_bins=32, texture_window_size=5):
        """Extract all statistical features."""
        features = {}
        features.update(extract_basic_statistics(gray))
        features.update(extract_histogram_features(gray, histogram_bins))
        features.update(extract_texture_statistics(gray, texture_window_size))
        features.update(extract_moment_features(gray))
        return features

    def compare_feature_vectors(f1, f2):
        """Compare two feature vectors."""
        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return {'similarity': 0.0}

        # Compute normalized differences
        diffs = [abs(f1[key] - f2[key]) / (max(abs(f1[key]), abs(f2[key]), 1e-10)) for key in common_keys]
        similarity = 1.0 - np.mean(diffs)

        return {
            'similarity': float(max(0.0, similarity)),
            'num_compared_features': len(common_keys)
        }


class StatisticalFeaturesDetector:
    """
    Simplified real-time statistical feature detector without complex threading.
    """

    def __init__(self,
                 enable_basic_stats=True,
                 enable_histogram_features=True,
                 enable_texture_stats=True,
                 enable_moment_features=True,
                 histogram_bins=32,
                 texture_window_size=5,
                 feature_update_interval=0.1):
        """
        Initialize the statistical features detector.
        """
        self.enable_basic_stats = enable_basic_stats
        self.enable_histogram_features = enable_histogram_features
        self.enable_texture_stats = enable_texture_stats
        self.enable_moment_features = enable_moment_features
        self.histogram_bins = max(8, min(256, histogram_bins))
        self.texture_window_size = max(3, min(15, texture_window_size))
        self.feature_update_interval = max(0.01, feature_update_interval)  # Faster updates

        # Statistics
        self.frames_processed = 0
        self.features_extracted = 0
        self.last_feature_update = 0
        self.current_features = {}
        self.previous_features = {}
        self.processing_times = []

    def extract_features(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
        """
        Extract statistical features from the given frame.
        Simplified synchronous processing.
        """
        if frame is None:
            return None, frame

        try:
            current_time = time.time()

            # Check if enough time has passed for feature update
            if current_time - self.last_feature_update < self.feature_update_interval:
                # Return cached features and visualization
                if self.current_features:
                    output_frame = self._visualize_features(frame, self.current_features)
                    return self.current_features, output_frame
                return None, frame

            start_processing = time.time()

            # Convert to grayscale for feature extraction
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Extract features based on enabled options (simplified, no parallel processing)
            features = {}

            if self.enable_basic_stats:
                try:
                    basic_features = extract_basic_statistics(gray)
                    features.update(basic_features)
                except Exception as e:
                    logging.debug(f"Error extracting basic stats: {e}")

            if self.enable_histogram_features:
                try:
                    hist_features = extract_histogram_features(gray, self.histogram_bins)
                    features.update(hist_features)
                except Exception as e:
                    logging.debug(f"Error extracting histogram features: {e}")

            if self.enable_texture_stats:
                try:
                    texture_features = extract_texture_statistics(gray, self.texture_window_size)
                    features.update(texture_features)
                except Exception as e:
                    logging.debug(f"Error extracting texture stats: {e}")

            if self.enable_moment_features:
                try:
                    moment_features = extract_moment_features(gray)
                    features.update(moment_features)
                except Exception as e:
                    logging.debug(f"Error extracting moment features: {e}")

            # Update timing and statistics
            processing_time = time.time() - start_processing
            self.processing_times.append(processing_time)

            # Keep only last 50 times for average
            if len(self.processing_times) > 50:
                self.processing_times = self.processing_times[-50:]

            # Store previous features for comparison
            self.previous_features = self.current_features.copy() if self.current_features else {}
            self.current_features = features

            # Update timing and statistics
            self.last_feature_update = current_time
            self.frames_processed += 1
            self.features_extracted = len(features)

            # Create output frame with feature visualization
            output_frame = self._visualize_features(frame, features)

            return features, output_frame

        except Exception as e:
            logging.error(f"Error extracting statistical features: {e}")
            return None, frame

    def _visualize_features(self, frame: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """
        Create visualization of extracted features on the frame.
        """
        try:
            output_frame = frame.copy()
            h, w = output_frame.shape[:2]

            # Add semi-transparent overlay
            overlay = np.zeros_like(output_frame)
            cv2.rectangle(overlay, (0, 0), (min(350, w), min(200, h)), (50, 50, 50), -1)
            cv2.addWeighted(output_frame, 0.8, overlay, 0.2, 0, output_frame)

            # Display key features
            y_offset = 25
            line_height = 20

            # Title
            cv2.putText(output_frame, "Statistical Features", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height + 5

            # Show only the most important features to avoid clutter
            important_features = [
                ('mean', 'Mean', (0, 255, 255)),
                ('std', 'Std Dev', (0, 255, 255)),
                ('entropy', 'Entropy', (0, 255, 255)),
                ('hist_mode', 'Hist Mode', (255, 255, 0)),
                ('texture_contrast', 'Texture', (255, 0, 255)),
                ('centroid_x', 'Centroid X', (0, 255, 0)),
                ('centroid_y', 'Centroid Y', (0, 255, 0))
            ]

            for feature_key, label, color in important_features:
                if feature_key in features and y_offset < h - 30:
                    value = features[feature_key]
                    if feature_key.startswith('centroid'):
                        text = f"{label}: {value:.3f}"
                    elif feature_key == 'entropy':
                        text = f"{label}: {value:.2f}"
                    else:
                        text = f"{label}: {value:.1f}"

                    cv2.putText(output_frame, text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += line_height

            # Draw centroid if available
            if 'centroid_x' in features and 'centroid_y' in features:
                cx = int(features['centroid_x'] * w)
                cy = int(features['centroid_y'] * h)
                cx = max(5, min(w-5, cx))
                cy = max(5, min(h-5, cy))
                cv2.circle(output_frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.circle(output_frame, (cx, cy), 8, (0, 255, 0), 1)

            # Performance info
            if self.processing_times:
                avg_time = np.mean(self.processing_times[-10:])  # Last 10 frames average
                fps_est = 1.0 / max(avg_time, 0.001)
                perf_text = f"Features: {len(features)} | Est FPS: {fps_est:.1f}"
            else:
                perf_text = f"Features: {len(features)} | Frames: {self.frames_processed}"

            cv2.putText(output_frame, perf_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            return output_frame

        except Exception as e:
            logging.error(f"Error in feature visualization: {e}")
            return frame

    def update_parameters(self, enable_basic_stats=None, enable_histogram_features=None,
                         enable_texture_stats=None, enable_moment_features=None,
                         histogram_bins=None, texture_window_size=None, feature_update_interval=None):
        """Update detection parameters dynamically."""
        if enable_basic_stats is not None:
            self.enable_basic_stats = enable_basic_stats
        if enable_histogram_features is not None:
            self.enable_histogram_features = enable_histogram_features
        if enable_texture_stats is not None:
            self.enable_texture_stats = enable_texture_stats
        if enable_moment_features is not None:
            self.enable_moment_features = enable_moment_features
        if histogram_bins is not None:
            self.histogram_bins = max(8, min(256, histogram_bins))
        if texture_window_size is not None:
            self.texture_window_size = max(3, min(15, texture_window_size))
        if feature_update_interval is not None:
            self.feature_update_interval = max(0.01, min(1.0, feature_update_interval))

        logging.info(f"Updated parameters: basic={self.enable_basic_stats}, "
                    f"hist={self.enable_histogram_features}, texture={self.enable_texture_stats}, "
                    f"moments={self.enable_moment_features}")

    def get_statistics(self) -> dict:
        """Get processing statistics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0

        return {
            'frames_processed': self.frames_processed,
            'features_extracted': self.features_extracted,
            'current_feature_count': len(self.current_features),
            'processing_rate': 1.0 / max(avg_processing_time, 0.001) if avg_processing_time > 0 else 0.0,
            'avg_processing_time': avg_processing_time
        }

    def compare_with_previous(self) -> Optional[Dict[str, float]]:
        """Compare current features with previous frame features."""
        if not self.previous_features or not self.current_features:
            return None
        return compare_feature_vectors(self.previous_features, self.current_features)

    def reset_statistics(self):
        """Reset processing statistics."""
        self.frames_processed = 0
        self.features_extracted = 0
        self.current_features = {}
        self.previous_features = {}
        self.processing_times = []


class StatisticalFeaturesProcessor:
    """
    Simplified high-level processor for statistical feature extraction.
    """

    def __init__(self, detector: StatisticalFeaturesDetector = None):
        """Initialize the processor."""
        if detector is not None:
            self.detector = detector
        else:
            self.detector = StatisticalFeaturesDetector()
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with statistical feature extraction."""
        if not self.processing_enabled or frame is None:
            return frame

        try:
            features, processed_frame = self.detector.extract_features(frame)
            return processed_frame
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def toggle_processing(self) -> bool:
        """Toggle statistical feature processing on/off."""
        self.processing_enabled = not self.processing_enabled
        logging.info(f"Statistical features processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """Check if processing is enabled."""
        return self.processing_enabled

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = self.detector.get_statistics()
        stats.update({
            'parallel_processing': False,  # We're not using parallel processing in this simplified version
            'frames_dropped': 0,  # No frame dropping in simplified version
        })
        return stats
