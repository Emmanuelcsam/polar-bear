#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time
from scipy.stats import sigmoid

# Configure logging system to display timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level to INFO (won't show DEBUG messages)
    format='%(asctime)s - [%(levelname)s] - %(message)s'  # Define log message format with timestamp
)


@dataclass  # Decorator to automatically generate __init__ and other methods
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - matches expected structure from app.py"""
    # Path to saved knowledge base file containing reference model data (None means use default)
    knowledge_base_path: Optional[str] = None
    # Minimum pixel area for a region to be considered a defect (filters noise)
    min_defect_size: int = 10
    # Maximum pixel area for a defect (larger areas might be image artifacts)
    max_defect_size: int = 5000
    # Dictionary mapping severity levels to confidence thresholds for classification
    severity_thresholds: Optional[Dict[str, float]] = None
    # Minimum confidence score (0-1) to report a detected anomaly
    confidence_threshold: float = 0.3
    # Multiplier for standard deviation to set anomaly detection threshold
    anomaly_threshold_multiplier: float = 2.5
    # Whether to generate and save visualization images
    enable_visualization: bool = True
    # Fixed size for resizing sub-regions in manual mode
    sub_region_fixed_size: int = 64

    def __post_init__(self):
        # Initialize default severity thresholds if none provided
        if self.severity_thresholds is None:
            # Create mapping from severity levels to minimum confidence scores
            self.severity_thresholds = {
                'CRITICAL': 0.9,  # 90%+ confidence = critical defect
                'HIGH': 0.7,      # 70-89% = high severity
                'MEDIUM': 0.5,    # 50-69% = medium severity
                'LOW': 0.3,       # 30-49% = low severity
                'NEGLIGIBLE': 0.1 # 10-29% = negligible
            }


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        # Convert numpy integer types to Python int for JSON compatibility
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert numpy float types to Python float for JSON compatibility
        if isinstance(obj, np.floating):
            return float(obj)
        # Convert numpy arrays to Python lists for JSON compatibility
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Fall back to default JSON encoder for other types
        return super(NumpyEncoder, self).default(obj)


class OmniFiberAnalyzer:
    """The ultimate fiber optic anomaly detection system - pipeline compatible version with defect library and manual mode."""

    def __init__(self, config: OmniConfig):
        # Store configuration object containing all analysis parameters
        self.config = config
        # Set knowledge base path, defaulting to "fiber_anomaly_kb.json" if not specified
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        # Set defect library path
        self.defect_base_path = self.knowledge_base_path.replace("anomaly", "defect") if self.knowledge_base_path else "fiber_defect_kb.json"
        # Initialize empty reference model structure for storing learned patterns
        self.reference_model = {
            'features': [],              # List of feature dictionaries from reference images
            'statistical_model': None,   # Statistical parameters (mean, std, covariance)
            'archetype_image': None,     # Median image representing typical fiber
            'feature_names': [],         # List of feature names in consistent order
            'comparison_results': {},    # Cached comparison results
            'learned_thresholds': {},    # Learned anomaly detection thresholds
            'timestamp': None           # When model was created/updated
        }
        # Initialize defect model for library of defect features
        self.defect_model = {
            'features': [],              # List of feature dictionaries from defect regions
            'statistical_model': None,   # Statistical parameters for defects
            'feature_names': [],         # List of feature names
            'learned_thresholds': {},    # Learned thresholds for defects
            'timestamp': None
        }
        # Initialize metadata storage for current image being processed
        self.current_metadata = None
        # Create logger instance for this class
        self.logger = logging.getLogger(__name__)
        # Attempt to load existing knowledge base from disk
        self.load_knowledge_base()
        self.load_defect_base()

    def load_defect_base(self):
        """Load previously saved defect library from JSON."""
        if os.path.exists(self.defect_base_path):
            try:
                with open(self.defect_base_path, 'r') as f:
                    loaded_data = json.load(f)

                # Convert statistical model lists back to numpy arrays
                if loaded_data.get('statistical_model'):
                    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                        if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                            loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)

                self.defect_model = loaded_data
                self.logger.info(f"Loaded defect library from {self.defect_base_path}")
            except Exception as e:
                self.logger.warning(f"Could not load defect library: {e}")

    def save_defect_base(self):
        """Save current defect library to JSON."""
        try:
            save_data = self.defect_model.copy()

            if save_data.get('statistical_model'):
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                        save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()

            save_data['timestamp'] = self._get_timestamp()

            with open(self.defect_base_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            self.logger.info(f"Defect library saved to {self.defect_base_path}")
        except Exception as e:
            self.logger.error(f"Error saving defect library: {e}")

    def manual_annotation(self, image_path):
        """Manual mode for selecting defect and non-defect regions to learn from."""
        self.logger.info(f"Entering manual annotation for {image_path}")

        image = self.load_image(image_path)
        if image is None:
            self.logger.error("Failed to load image for manual annotation")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 2 else image.copy()
        clone = image.copy()

        mask_defect = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_normal = np.zeros(image.shape[:2], dtype=np.uint8)

        drawing = False
        mode = 'd'  # 'd' defect (red), 'n' normal (green)
        brush_size = 5

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                _draw_circle(clone, mask_defect if mode == 'd' else mask_normal, mode, x, y, brush_size)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    _draw_circle(clone, mask_defect if mode == 'd' else mask_normal, mode, x, y, brush_size)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False

        def _draw_circle(img, mask, mode, x, y, radius):
            color = (0, 0, 255) if mode == 'd' else (0, 255, 0)
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(mask, (x, y), radius, 255, -1)

        cv2.namedWindow('Annotation')
        cv2.setMouseCallback('Annotation', mouse_callback)

        while True:
            cv2.imshow('Annotation', clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                mode = 'd'
            elif key == ord('n'):
                mode = 'n'
            elif key == ord(' + '):
                brush_size += 1
            elif key == ord('-'):
                brush_size = max(1, brush_size - 1)
            elif key == ord('c'):
                clone = image.copy()
                mask_defect.fill(0)
                mask_normal.fill(0)
            elif key == ord('s'):
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        # Process defect mask
        contours, _ = cv2.findContours(mask_defect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.config.min_defect_size:
                x, y, w, h = cv2.boundingRect(contour)
                sub_gray = gray[y:y+h, x:x+w]
                sub_resized = cv2.resize(sub_gray, (self.config.sub_region_fixed_size, self.config.sub_region_fixed_size))
                features, _ = self.extract_ultra_comprehensive_features(sub_resized)
                self.defect_model['features'].append(features)

        # Process normal mask
        contours, _ = cv2.findContours(mask_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.config.min_defect_size:
                x, y, w, h = cv2.boundingRect(contour)
                sub_gray = gray[y:y+h, x:x+w]
                sub_resized = cv2.resize(sub_gray, (self.config.sub_region_fixed_size, self.config.sub_region_fixed_size))
                features, _ = self.extract_ultra_comprehensive_features(sub_resized)
                self.reference_model['features'].append(features)

        # Update models
        self._update_statistical_model(self.reference_model)
        self._update_statistical_model(self.defect_model)

        # Save
        self.save_knowledge_base()
        self.save_defect_base()

        self.logger.info("Manual annotation completed and models updated")

    def _update_statistical_model(self, model):
        """Update statistical model from features."""
        all_features = model['features']
        if not all_features:
            return

        feature_names = sorted(all_features[0].keys())
        feature_matrix = np.array([[f.get(name, 0) for name in feature_names] for f in all_features])

        mean_vector = np.mean(feature_matrix, axis=0)
        std_vector = np.std(feature_matrix, axis=0)
        median_vector = np.median(feature_matrix, axis=0)
        robust_mean, robust_cov, robust_inv_cov = self._compute_robust_statistics(feature_matrix)

        model['feature_names'] = feature_names
        model['statistical_model'] = {
            'mean': mean_vector,
            'std': std_vector,
            'median': median_vector,
            'robust_mean': robust_mean,
            'robust_cov': robust_cov,
            'robust_inv_cov': robust_inv_cov,
            'n_samples': len(all_features)
        }
        model['timestamp'] = self._get_timestamp()

        # For reference model, update thresholds
        if model is self.reference_model:
            # Compute pairwise for thresholds
            comparison_scores = []
            for i in range(len(all_features)):
                for j in range(i + 1, len(all_features)):
                    comp = self.compute_exhaustive_comparison(all_features[i], all_features[j])
                    score = (comp['euclidean_distance'] * 0.2 + comp['manhattan_distance'] * 0.1 + comp['cosine_distance'] * 0.2 + (1 - abs(comp['pearson_correlation'])) * 0.1 + min(comp['kl_divergence'], 10.0) * 0.1 + comp['js_divergence'] * 0.1 + min(comp['chi_square'], 10.0) * 0.1 + min(comp['wasserstein_distance'], 10.0) * 0.1)
                    comparison_scores.append(score)
            if comparison_scores:
                scores_array = np.array(comparison_scores)
                valid_scores = scores_array[np.isfinite(scores_array)]
                if len(valid_scores) > 0:
                    mean_score = np.mean(valid_scores)
                    std_score = np.std(valid_scores)
                    model['learned_thresholds'] = {
                        'anomaly_mean': mean_score,
                        'anomaly_std': std_score,
                        'anomaly_p90': np.percentile(valid_scores, 90),
                        'anomaly_p95': np.percentile(valid_scores, 95),
                        'anomaly_p99': np.percentile(valid_scores, 99),
                        'anomaly_threshold': mean_score + self.config.anomaly_threshold_multiplier * std_score
                    }
                else:
                    model['learned_thresholds'] = self._get_default_thresholds()
            else:
                model['learned_thresholds'] = self._get_default_thresholds()

    # Rest of the class remains the same, with modifications in detect_anomalies_comprehensive for using defect_model

    def _find_anomaly_regions(self, anomaly_map, original_shape):
        """Find distinct anomaly regions from the anomaly map, using defect library for refinement."""
        positive_values = anomaly_map[anomaly_map > 0]
        if positive_values.size == 0:
            return []

        threshold = np.percentile(positive_values, 80)
        binary_map = (anomaly_map > threshold).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

        regions = []
        h_scale = original_shape[0] / anomaly_map.shape[0]
        w_scale = original_shape[1] / anomaly_map.shape[1]

        has_defect_model = self.defect_model.get('statistical_model') is not None

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            x_orig = int(x * w_scale)
            y_orig = int(y * h_scale)
            w_orig = int(w * w_scale)
            h_orig = int(h * h_scale)

            region_mask = (labels == i)
            region_values = anomaly_map[region_mask]
            confidence = float(np.mean(region_values))

            if area > 20:
                region = {
                    'bbox': (x_orig, y_orig, w_orig, h_orig),
                    'area': int(area * h_scale * w_scale),
                    'confidence': confidence,
                    'centroid': (int(centroids[i][0] * w_scale), int(centroids[i][1] * h_scale)),
                    'max_intensity': float(np.max(region_values)),
                }

                if has_defect_model:
                    # Extract sub-region features for classification
                    sub_gray = self.test_gray[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]  # Assume self.test_gray set in detect
                    if sub_gray.size == 0:
                        continue
                    sub_resized = cv2.resize(sub_gray, (self.config.sub_region_fixed_size, self.config.sub_region_fixed_size))
                    sub_features, _ = self.extract_ultra_comprehensive_features(sub_resized)
                    sub_vector = np.array([sub_features.get(fname, 0) for fname in self.reference_model['feature_names']])

                    # Compute distances
                    ref_mean = self.reference_model['statistical_model']['robust_mean']
                    ref_inv_cov = self.reference_model['statistical_model']['robust_inv_cov']
                    diff_normal = sub_vector - ref_mean
                    dist_normal = np.sqrt(diff_normal.T @ ref_inv_cov @ diff_normal)

                    def_mean = self.defect_model['statistical_model']['robust_mean']
                    def_inv_cov = self.defect_model['statistical_model']['robust_inv_cov']
                    diff_defect = sub_vector - def_mean
                    dist_defect = np.sqrt(diff_defect.T @ def_inv_cov @ diff_defect)

                    # Adjust confidence based on distances
                    relative_conf = sigmoid(dist_normal - dist_defect)
                    region['confidence'] = relative_conf if relative_conf > self.config.confidence_threshold else 0

                    if region['confidence'] == 0:
                        continue  # Discard if not defect-like

                regions.append(region)

        regions.sort(key=lambda x: x['confidence'], reverse=True)
        return regions

    # The rest of the script remains the same as the original, with manual mode integrated in main if needed.

def main():
    """Main execution function for standalone testing."""
    print("\n" + "="*80)
    print("OMNIFIBER ANALYZER - DETECTION MODULE (v1.6 with Defect Library and Manual Mode)".center(80))
    print("="*80)
    print("\nThis module is designed to be called from app.py in the pipeline.")
    print("For standalone testing, you can analyze individual images or use manual mode.\n")

    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)

    while True:
        print("\nOptions:")
        print("1. Analyze image")
        print("2. Build reference model from directory")
        print("3. Manual annotation mode")
        print("4. Quit")
        choice = input("Enter choice (1/2/3/4): ").strip()

        if choice == '4':
            break
        elif choice == '3':
            image_path = input("\nEnter path to image for manual annotation: ").strip().strip('"\'')
            if os.path.isfile(image_path):
                analyzer.manual_annotation(image_path)
            else:
                print("File not found.")
        elif choice == '1':
            test_path = input("\nEnter path to test image: ").strip().strip('"\'')
            if os.path.isfile(test_path):
                output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                analyzer.analyze_end_face(test_path, output_dir)
                print(f"\nResults saved to: {output_dir}/")
            else:
                print("File not found.")
        elif choice == '2':
            ref_dir = input("\nEnter path to reference directory: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                analyzer.build_comprehensive_reference_model(ref_dir)
            else:
                print("Directory not found.")

    print("\nThank you for using the OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
