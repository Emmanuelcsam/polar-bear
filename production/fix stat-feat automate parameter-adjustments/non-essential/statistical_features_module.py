"""
Statistical Features Processing Module.
Provides real-time statistical feature extraction for video processing with parallel processing.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

# Import the statistical features functions
import sys
import os

# Add path to dev/modular_scripts
current_dir = os.path.dirname(__file__)
modular_scripts_path = os.path.join(current_dir, 'dev', 'modular_scripts')
if os.path.exists(modular_scripts_path):
    sys.path.insert(0, modular_scripts_path)

try:
    from statistical_features import (
        extract_all_statistical_features,
        compare_feature_vectors,
        extract_basic_statistics,
        extract_histogram_features,
        extract_texture_statistics,
        extract_moment_features
    )
except ImportError as e:
    logging.warning(f"Could not import statistical_features module: {e}")
    # Provide fallback implementations
    def extract_all_statistical_features(gray):
        return {'mean': float(np.mean(gray)), 'std': float(np.std(gray))}

    def compare_feature_vectors(f1, f2):
        return {'similarity': 0.5}

    def extract_basic_statistics(gray):
        return {'mean': float(np.mean(gray)), 'std': float(np.std(gray))}

    def extract_histogram_features(gray, bins=32):
        return {'hist_mean': float(np.mean(gray))}

    def extract_texture_statistics(gray, window_size=5):
        return {'texture_mean': float(np.mean(gray))}

    def extract_moment_features(gray):
        return {'moment_00': 1.0}


def parallel_extract_features(gray_frame, feature_functions, max_workers=None):
    """
    Extract multiple statistical features in parallel using ThreadPoolExecutor.

    Args:
        gray_frame: Grayscale image frame
        feature_functions: List of tuples (function_name, function, args)
        max_workers: Maximum number of worker threads (default: CPU count)

    Returns:
        Dict containing all extracted features
    """
    if max_workers is None:
        max_workers = min(4, (mp.cpu_count() or 1) + 1)

    results = {}

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all feature extraction tasks
            future_to_name = {}
            for func_name, func, args in feature_functions:
                future = executor.submit(func, gray_frame, *args)
                future_to_name[future] = func_name

            # Collect results
            for future in future_to_name:
                try:
                    func_name = future_to_name[future]
                    result = future.result(timeout=1.0)  # 1 second timeout
                    if isinstance(result, dict):
                        results.update(result)
                    else:
                        results[func_name] = result
                except Exception as e:
                    logging.warning(f"Feature extraction failed for {func_name}: {e}")
                    results[future_to_name[future]] = {}

    except Exception as e:
        logging.error(f"Parallel feature extraction failed: {e}")
        # Fallback to sequential processing
        for func_name, func, args in feature_functions:
            try:
                result = func(gray_frame, *args)
                if isinstance(result, dict):
                    results.update(result)
                else:
                    results[func_name] = result
            except Exception as ex:
                logging.warning(f"Sequential fallback failed for {func_name}: {ex}")
                results[func_name] = {}

    return results


class FrameBuffer:
    """
    Thread-safe frame buffer for managing frame processing queue.
    """

    def __init__(self, maxsize=10):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()

    def put_frame(self, frame, timeout=0.1):
        """Add a frame to the buffer, dropping old frames if full."""
        try:
            # If buffer is full, remove old frames
            while self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break

            self.buffer.put(frame, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get_frame(self, timeout=0.1):
        """Get the next frame from buffer."""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        """Clear all frames from buffer."""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break


class ParallelStatisticalProcessor:
    """
    High-performance statistical features processor using parallel processing.
    """

    def __init__(self, max_workers=4, buffer_size=10):
        self.max_workers = max_workers
        self.frame_buffer = FrameBuffer(maxsize=buffer_size)
        self.processing_enabled = True
        self.stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'processing_time_avg': 0.0,
            'current_feature_count': 0
        }

        # Processing thread
        self.processing_thread = None
        self.stop_processing = threading.Event()

        # Features cache
        self.last_features = {}

    def start_processing(self):
        """Start the parallel processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

    def stop_processing(self):
        """Stop the parallel processing thread."""
        self.stop_processing.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

    def process_frame_async(self, frame):
        """Queue a frame for asynchronous processing."""
        if not self.processing_enabled:
            return frame

        # Add frame to buffer (non-blocking)
        success = self.frame_buffer.put_frame(frame.copy())
        if not success:
            self.stats['frames_dropped'] += 1

        return frame  # Return original frame immediately

    def get_latest_features(self):
        """Get the latest processed features."""
        return self.last_features.copy()

    def _processing_loop(self):
        """Main processing loop that runs in background thread."""
        processing_times = []

        while not self.stop_processing.is_set():
            try:
                # Get frame from buffer
                frame = self.frame_buffer.get_frame(timeout=0.1)
                if frame is None:
                    continue

                start_time = time.time()

                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame

                # Define feature extraction functions for parallel processing
                feature_functions = [
                    ('basic_stats', extract_basic_statistics, ()),
                    ('histogram', extract_histogram_features, (32,)),
                    ('texture', extract_texture_statistics, (5,)),
                    ('moments', extract_moment_features, ())
                ]

                # Extract features in parallel
                features = parallel_extract_features(gray, feature_functions, self.max_workers)

                # Update cache
                self.last_features = features
                self.stats['current_feature_count'] = len(features)
                self.stats['frames_processed'] += 1

                # Update timing statistics
                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Keep only last 100 times for average
                if len(processing_times) > 100:
                    processing_times = processing_times[-100:]

                self.stats['processing_time_avg'] = np.mean(processing_times)

            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                time.sleep(0.01)

    def get_statistics(self):
        """Get processing statistics."""
        return self.stats.copy()

    def enable_processing(self):
        """Enable feature processing."""
        self.processing_enabled = True

    def disable_processing(self):
        """Disable feature processing."""
        self.processing_enabled = False

    def is_processing_enabled(self):
        """Check if processing is enabled."""
        return self.processing_enabled


class StatisticalFeaturesDetector:
    """
    Real-time statistical feature detector with configurable parameters and parallel processing.
    """

    def __init__(self,
                 enable_basic_stats=True,
                 enable_histogram_features=True,
                 enable_texture_stats=True,
                 enable_moment_features=True,
                 histogram_bins=32,
                 texture_window_size=5,
                 feature_update_interval=1.0,
                 use_parallel_processing=True,
                 max_workers=4):
        """
        Initialize the statistical features detector.

        Args:
            enable_basic_stats (bool): Enable basic statistical features
            enable_histogram_features (bool): Enable histogram-based features
            enable_texture_stats (bool): Enable texture statistics
            enable_moment_features (bool): Enable moment features
            histogram_bins (int): Number of histogram bins (8-256)
            texture_window_size (int): Size of texture analysis window (3-15)
            feature_update_interval (float): Minimum time between feature updates (seconds)
            use_parallel_processing (bool): Enable parallel processing for better performance
            max_workers (int): Maximum number of parallel workers
        """
        self.enable_basic_stats = enable_basic_stats
        self.enable_histogram_features = enable_histogram_features
        self.enable_texture_stats = enable_texture_stats
        self.enable_moment_features = enable_moment_features
        self.histogram_bins = max(8, min(256, histogram_bins))
        self.texture_window_size = max(3, min(15, texture_window_size))
        self.feature_update_interval = max(0.1, feature_update_interval)
        self.use_parallel_processing = use_parallel_processing

        # Statistics
        self.frames_processed = 0
        self.features_extracted = 0
        self.last_feature_update = 0
        self.current_features = {}
        self.previous_features = {}

        # Initialize parallel processor if enabled
        if self.use_parallel_processing:
            self.parallel_processor = ParallelStatisticalProcessor(max_workers=max_workers)
            self.parallel_processor.start_processing()
        else:
            self.parallel_processor = None

    def extract_features(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, float]], np.ndarray]:
        """
        Extract statistical features from the given frame.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            Tuple[Optional[Dict[str, float]], np.ndarray]:
                - Dictionary of extracted features or None if not updated
                - Processed frame with feature visualization
        """
        if frame is None:
            return None, frame

        try:
            current_time = time.time()

            # If using parallel processing, queue frame and get latest features
            if self.use_parallel_processing and self.parallel_processor:
                # Queue frame for async processing
                self.parallel_processor.process_frame_async(frame)

                # Check if enough time has passed for feature update
                if current_time - self.last_feature_update < self.feature_update_interval:
                    return self.current_features if self.current_features else None, frame

                # Get latest features from parallel processor
                features = self.parallel_processor.get_latest_features()
                if features:
                    self.previous_features = self.current_features.copy() if self.current_features else {}
                    self.current_features = features
                    self.last_feature_update = current_time
                    self.frames_processed += 1
                    self.features_extracted = len(features)

                # Create output frame with feature visualization
                output_frame = self._visualize_features(frame, self.current_features)
                return self.current_features if self.current_features else None, output_frame

            # Sequential processing (fallback)
            else:
                # Check if enough time has passed for feature update
                if current_time - self.last_feature_update < self.feature_update_interval:
                    # Return previous features if not enough time has passed
                    return self.current_features if self.current_features else None, frame

                # Convert to grayscale for feature extraction
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Extract features based on enabled options
                features = {}

                if self.enable_basic_stats:
                    logging.debug("Extracting basic statistical features")
                    features.update(extract_basic_statistics(gray))

                if self.enable_histogram_features:
                    logging.debug("Extracting histogram features")
                    features.update(extract_histogram_features(gray, self.histogram_bins))

                if self.enable_texture_stats:
                    logging.debug("Extracting texture statistics")
                    features.update(extract_texture_statistics(gray, self.texture_window_size))

                if self.enable_moment_features:
                    logging.debug("Extracting moment features")
                    features.update(extract_moment_features(gray))

                # Store previous features for comparison
                self.previous_features = self.current_features.copy() if self.current_features else {}
                self.current_features = features

                # Update timing and statistics
                self.last_feature_update = current_time
                self.frames_processed += 1
                self.features_extracted = len(features)

                # Create output frame with feature visualization
                output_frame = self._visualize_features(frame, features)

                logging.debug(f"Extracted {len(features)} features from frame {self.frames_processed}")
                return features, output_frame

        except Exception as e:
            logging.error(f"Error extracting statistical features: {e}")
            return None, frame

    def _visualize_features(self, frame: np.ndarray, features: Dict[str, float]) -> np.ndarray:
        """
        Create visualization of extracted features on the frame.

        Args:
            frame (np.ndarray): Input frame
            features (Dict[str, float]): Extracted features

        Returns:
            np.ndarray: Frame with feature visualization
        """
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]

        # Create a semi-transparent overlay for text
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)

        # Display key features
        y_offset = 30
        line_height = 25

        # Title
        cv2.putText(output_frame, "Statistical Features", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10

        # Basic statistics
        if 'mean' in features:
            cv2.putText(output_frame, f"Mean: {features['mean']:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height

        if 'std' in features:
            cv2.putText(output_frame, f"Std: {features['std']:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height

        if 'entropy' in features:
            cv2.putText(output_frame, f"Entropy: {features['entropy']:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height

        # Histogram features
        if 'hist_mode' in features:
            cv2.putText(output_frame, f"Hist Mode: {features['hist_mode']:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += line_height

        # Texture features
        if 'texture_contrast' in features:
            cv2.putText(output_frame, f"Texture Contrast: {features['texture_contrast']:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_offset += line_height

        # Centroid location
        if 'centroid_x' in features and 'centroid_y' in features:
            cx = int(features['centroid_x'] * w)
            cy = int(features['centroid_y'] * h)
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(output_frame, f"Centroid: ({cx}, {cy})", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height

        # Feature count and processing info
        cv2.putText(output_frame, f"Features: {len(features)} | Frames: {self.frames_processed}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_frame

    def compare_with_previous(self) -> Optional[Dict[str, float]]:
        """
        Compare current features with previous frame features.

        Returns:
            Optional[Dict[str, float]]: Comparison metrics or None if no previous features
        """
        if not self.previous_features or not self.current_features:
            return None

        return compare_feature_vectors(self.previous_features, self.current_features)

    def update_parameters(self, enable_basic_stats=None, enable_histogram_features=None,
                         enable_texture_stats=None, enable_moment_features=None,
                         histogram_bins=None, texture_window_size=None, feature_update_interval=None):
        """
        Update detection parameters dynamically.

        Args:
            enable_basic_stats (bool, optional): Enable basic statistical features
            enable_histogram_features (bool, optional): Enable histogram-based features
            enable_texture_stats (bool, optional): Enable texture statistics
            enable_moment_features (bool, optional): Enable moment features
            histogram_bins (int, optional): Number of histogram bins (8-256)
            texture_window_size (int, optional): Size of texture analysis window (3-15)
            feature_update_interval (float, optional): Minimum time between feature updates (0.1-10.0)
        """
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
            self.feature_update_interval = max(0.1, min(10.0, feature_update_interval))

        logging.info(f"Updated statistical features parameters: "
                    f"basic_stats={self.enable_basic_stats}, "
                    f"histogram_features={self.enable_histogram_features}, "
                    f"texture_stats={self.enable_texture_stats}, "
                    f"moment_features={self.enable_moment_features}, "
                    f"histogram_bins={self.histogram_bins}, "
                    f"texture_window_size={self.texture_window_size}, "
                    f"update_interval={self.feature_update_interval:.1f}")

    def get_statistics(self) -> dict:
        """
        Get processing statistics.

        Returns:
            dict: Statistics including frames processed and features extracted
        """
        return {
            'frames_processed': self.frames_processed,
            'features_extracted': self.features_extracted,
            'current_feature_count': len(self.current_features),
            'processing_rate': self.frames_processed / max(1, time.time() - self.last_feature_update + self.feature_update_interval)
        }

    def reset_statistics(self):
        """Reset processing statistics."""
        self.frames_processed = 0
        self.features_extracted = 0
        self.current_features = {}
        self.previous_features = {}


class StatisticalFeaturesProcessor:
    """
    High-level processor that applies statistical feature extraction to video streams with parallel processing.
    """

    def __init__(self, detector: StatisticalFeaturesDetector = None, use_parallel_processing=True, max_workers=4):
        """
        Initialize the processor.

        Args:
            detector (StatisticalFeaturesDetector, optional): Statistical features detector instance
            use_parallel_processing (bool): Enable parallel processing for better performance
            max_workers (int): Maximum number of parallel workers
        """
        if detector is not None:
            self.detector = detector
        else:
            # Create detector with parallel processing enabled by default
            self.detector = StatisticalFeaturesDetector(
                use_parallel_processing=use_parallel_processing,
                max_workers=max_workers
            )
        self.processing_enabled = True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with statistical feature extraction.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with feature visualization
        """
        if not self.processing_enabled or frame is None:
            return frame

        features, processed_frame = self.detector.extract_features(frame)
        return processed_frame

    def toggle_processing(self) -> bool:
        """
        Toggle statistical feature processing on/off.

        Returns:
            bool: New processing state
        """
        self.processing_enabled = not self.processing_enabled

        # Also toggle the parallel processor if available
        if hasattr(self.detector, 'parallel_processor') and self.detector.parallel_processor:
            if self.processing_enabled:
                self.detector.parallel_processor.enable_processing()
            else:
                self.detector.parallel_processor.disable_processing()

        logging.info(f"Statistical features processing {'enabled' if self.processing_enabled else 'disabled'}")
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """
        Check if processing is enabled.

        Returns:
            bool: True if processing is enabled
        """
        return self.processing_enabled

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics from the parallel processor.

        Returns:
            dict: Performance statistics
        """
        stats = self.detector.get_statistics()

        # Add parallel processing stats if available
        if hasattr(self.detector, 'parallel_processor') and self.detector.parallel_processor:
            parallel_stats = self.detector.parallel_processor.get_statistics()
            stats.update({
                'parallel_processing': True,
                'frames_dropped': parallel_stats.get('frames_dropped', 0),
                'avg_processing_time': parallel_stats.get('processing_time_avg', 0.0)
            })
        else:
            stats.update({
                'parallel_processing': False,
                'frames_dropped': 0,
                'avg_processing_time': 0.0
            })

        return stats
