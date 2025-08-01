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

# Configure logging system to display timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


@dataclass
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - enhanced with cascade classifier support"""
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    # New parameters for cascade classifiers
    enable_cascade_detection: bool = True
    cascade_classifiers_path: str = "cascade_classifiers"
    scale_factor: float = 1.05  # From tutorial - 5% decrease per scale
    min_neighbors: int = 5      # From tutorial - neighbor threshold
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class OmniFiberAnalyzer:
    """Enhanced fiber optic anomaly detection system with cascade classifier support."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'timestamp': None
        }
        self.current_metadata = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize cascade classifiers for specific defect types
        self.cascade_classifiers = {}
        self._load_cascade_classifiers()
        
        # Load knowledge base
        self.load_knowledge_base()
    
    def _load_cascade_classifiers(self):
        """Load Haar cascade classifiers for specific defect types."""
        if not self.config.enable_cascade_detection:
            return
            
        # Define cascade classifier files for different defect types
        cascade_files = {
            'scratch': 'fiber_scratch_cascade.xml',
            'dig': 'fiber_dig_cascade.xml',
            'contamination': 'fiber_contamination_cascade.xml',
            'crack': 'fiber_crack_cascade.xml',
        }
        
        # Load each cascade classifier
        for defect_type, filename in cascade_files.items():
            cascade_path = os.path.join(self.config.cascade_classifiers_path, filename)
            if os.path.exists(cascade_path):
                try:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.cascade_classifiers[defect_type] = cascade
                        self.logger.info(f"Loaded cascade classifier for {defect_type}")
                except Exception as e:
                    self.logger.warning(f"Failed to load cascade for {defect_type}: {e}")
            else:
                self.logger.debug(f"Cascade file not found: {cascade_path}")
    
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Main analysis method - enhanced with cascade detection"""
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Building from single image...")
            self._build_minimal_reference(image_path)
        
        # Run comprehensive anomaly detection analysis
        results = self.detect_anomalies_comprehensive(image_path)
        
        if results:
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
            self.logger.info(f"Saved detection report to {report_path}")
            
            if self.config.enable_visualization:
                viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
                self.visualize_comprehensive_results(results, str(viz_path))
                
                # Enhanced: Save interactive visualization if cascade detections exist
                if results.get('cascade_detections'):
                    interactive_path = output_path / f"{Path(image_path).stem}_interactive.png"
                    self._save_interactive_visualization(results, str(interactive_path))
                
                mask_path = output_path / f"{Path(image_path).stem}_defect_mask.npy"
                defect_mask = self._create_defect_mask(results)
                np.save(mask_path, defect_mask)
            
            text_report_path = output_path / f"{Path(image_path).stem}_detailed.txt"
            self.generate_detailed_report(results, str(text_report_path))
            
            return pipeline_report
            
        else:
            self.logger.error(f"Analysis failed for {image_path}")
            empty_report = {
                'image_path': image_path,
                'timestamp': self._get_timestamp(),
                'success': False,
                'error': 'Analysis failed',
                'defects': []
            }
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(empty_report, f, indent=2)
            
            return empty_report
    
    def load_image(self, path):
        """Enhanced image loading with proper parameter usage from tutorial."""
        self.current_metadata = None
        
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            # Use cv2.imread with explicit color flag as shown in tutorial
            img = cv2.imread(path, cv2.IMREAD_COLOR)  # Explicitly read as color
            if img is None:
                self.logger.error(f"Could not read image: {path}")
                return None
            
            # Store metadata including shape info
            self.current_metadata = {
                'filename': os.path.basename(path),
                'shape': img.shape,  # As shown in tutorial
                'channels': img.shape[2] if len(img.shape) == 3 else 1
            }
            return img
    
    def _apply_cascade_detection(self, gray_image):
        """Apply Haar cascade classifiers to detect specific defect patterns."""
        cascade_detections = {}
        
        for defect_type, cascade in self.cascade_classifiers.items():
            # Use detectMultiScale as shown in tutorial
            detections = cascade.detectMultiScale(
                gray_image,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=(self.config.min_defect_size, self.config.min_defect_size),
                maxSize=(self.config.max_defect_size, self.config.max_defect_size)
            )
            
            if len(detections) > 0:
                cascade_detections[defect_type] = []
                for (x, y, w, h) in detections:
                    cascade_detections[defect_type].append({
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': w * h,
                        'confidence': 0.8  # Cascade detections have high confidence
                    })
                self.logger.info(f"Cascade detected {len(detections)} {defect_type} defects")
        
        return cascade_detections
    
    def detect_anomalies_comprehensive(self, test_path):
        """Enhanced anomaly detection with cascade classifiers."""
        self.logger.info(f"Analyzing: {test_path}")
        
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None
        
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        self.logger.info(f"Loaded image: {self.current_metadata}")
        
        # Convert to grayscale using BGR2GRAY as emphasized in tutorial
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Apply Gaussian blur as shown in tutorial for noise reduction
        test_gray = cv2.GaussianBlur(test_gray, (3, 3), 0)
        
        # Extract features from test image
        self.logger.info("Extracting features from test image...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # Apply cascade detection if enabled
        cascade_detections = {}
        if self.config.enable_cascade_detection and self.cascade_classifiers:
            self.logger.info("Applying cascade classifiers...")
            cascade_detections = self._apply_cascade_detection(test_gray)
        
        # Continue with existing analysis...
        self.logger.info("Performing global anomaly analysis...")
        
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Ensure numpy arrays
        for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
            if key in stat_model and isinstance(stat_model[key], list):
                stat_model[key] = np.array(stat_model[key], dtype=np.float64)
        
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        # Compute Mahalanobis distance
        diff = test_vector - stat_model['robust_mean']
        try:
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except:
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)
        
        # Compute Z-scores
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        
        top_indices = np.argsort(z_scores)[::-1][:10]
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # Individual comparisons
        self.logger.info(f"Comparing against {len(self.reference_model['features'])} reference samples...")
        
        individual_scores = []
        for i, ref_features in enumerate(self.reference_model['features']):
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            
            euclidean_term = min(comp['euclidean_distance'], 1000.0) * 0.2
            manhattan_term = min(comp['manhattan_distance'], 10000.0) * 0.1
            cosine_term = comp['cosine_distance'] * 0.2
            correlation_term = (1 - abs(comp['pearson_correlation'])) * 0.1
            kl_term = min(comp['kl_divergence'], 10.0) * 0.1
            js_term = comp['js_divergence'] * 0.1
            chi_term = min(comp['chi_square'], 10.0) * 0.1
            wasserstein_term = min(comp['wasserstein_distance'], 10.0) * 0.1
            
            score = (euclidean_term + manhattan_term + cosine_term + 
                    correlation_term + kl_term + js_term + 
                    chi_term + wasserstein_term)
            
            score = min(score, 100.0)
            individual_scores.append(score)
        
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # Structural analysis
        self.logger.info("Performing structural analysis...")
        
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        
        # Enhanced: Use proper resizing as shown in tutorial
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, 
                                         (archetype.shape[1], archetype.shape[0]),
                                         interpolation=cv2.INTER_AREA)  # Better for downsampling
        else:
            test_gray_resized = test_gray
        
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # Local anomaly detection
        self.logger.info("Detecting local anomalies...")
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # Specific defect detection
        self.logger.info("Detecting specific defects...")
        specific_defects = self._detect_specific_defects(test_gray)
        
        # Merge cascade detections with specific defects
        if cascade_detections:
            self._merge_cascade_detections(specific_defects, cascade_detections)
        
        # Determine overall status
        thresholds = self.reference_model['learned_thresholds']
        
        # Enhanced criteria including cascade detections
        cascade_defect_count = sum(len(dets) for dets in cascade_detections.values())
        
        is_anomalous = (
            mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or
            comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6) or
            structural_comp['ssim'] < 0.7 or
            len(anomaly_regions) > 3 or
            any(region['confidence'] > 0.8 for region in anomaly_regions) or
            cascade_defect_count > 0  # Cascade detection indicates anomaly
        )
        
        confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),
            comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),
            1 - structural_comp['ssim'],
            len(anomaly_regions) / 10,
            cascade_defect_count / 5  # Include cascade detections in confidence
        ))
        
        self.logger.info("Analysis Complete!")
        
        return {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            
            'global_analysis': {
                'mahalanobis_distance': float(mahalanobis_dist),
                'deviant_features': deviant_features,
                'comparison_stats': comparison_stats,
            },
            
            'structural_analysis': structural_comp,
            
            'local_analysis': {
                'anomaly_map': anomaly_map,
                'anomaly_regions': anomaly_regions,
            },
            
            'specific_defects': specific_defects,
            'cascade_detections': cascade_detections,  # New field
            
            'verdict': {
                'is_anomalous': is_anomalous,
                'confidence': float(confidence),
                'criteria_triggered': {
                    'mahalanobis': mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6),
                    'comparison': comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6),
                    'structural': structural_comp['ssim'] < 0.7,
                    'local': len(anomaly_regions) > 3,
                    'cascade': cascade_defect_count > 0,  # New criterion
                }
            }
        }
    
    def _merge_cascade_detections(self, specific_defects, cascade_detections):
        """Merge cascade detections into specific defects."""
        # Map cascade types to defect categories
        cascade_to_defect_map = {
            'scratch': 'scratches',
            'dig': 'digs',
            'contamination': 'blobs',
            'crack': 'scratches'  # Treat cracks as scratches
        }
        
        for cascade_type, detections in cascade_detections.items():
            defect_category = cascade_to_defect_map.get(cascade_type)
            if defect_category and defect_category in specific_defects:
                # Add cascade detections to existing category
                for det in detections:
                    if cascade_type in ['scratch', 'crack']:
                        # Convert bbox to line for scratches
                        x, y, w, h = det['bbox']
                        specific_defects[defect_category].append({
                            'line': (x, y, x + w, y + h),
                            'length': float(np.sqrt(w**2 + h**2)),
                            'angle': float(np.arctan2(h, w) * 180 / np.pi),
                            'cascade_detected': True
                        })
                    else:
                        # Add as area-based defect
                        specific_defects[defect_category].append({
                            'center': det['center'],
                            'area': float(det['area']),
                            'bbox': det['bbox'],
                            'cascade_detected': True,
                            'contour': None  # No contour from cascade
                        })
    
    def _save_interactive_visualization(self, results, output_path):
        """Save an interactive visualization showing cascade detection results."""
        test_img = results['test_image'].copy()
        
        # Color scheme for different detection methods
        colors = {
            'cascade': (255, 0, 255),    # Magenta for cascade
            'traditional': (0, 255, 0),   # Green for traditional
            'anomaly': (0, 0, 255)        # Red for anomaly regions
        }
        
        # Draw cascade detections with distinct visualization
        if 'cascade_detections' in results:
            for defect_type, detections in results['cascade_detections'].items():
                for det in detections:
                    x, y, w, h = det['bbox']
                    # Draw dashed rectangle for cascade detections
                    self._draw_dashed_rectangle(test_img, (x, y), (x+w, y+h), 
                                              colors['cascade'], 2)
                    # Add label
                    cv2.putText(test_img, f"CASCADE: {defect_type}", 
                              (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, colors['cascade'], 1)
        
        # Draw traditional detections
        defects = results['specific_defects']
        
        # Draw scratches
        for scratch in defects['scratches']:
            if not scratch.get('cascade_detected', False):
                x1, y1, x2, y2 = scratch['line']
                cv2.line(test_img, (x1, y1), (x2, y2), colors['traditional'], 2)
        
        # Draw anomaly regions
        for region in results['local_analysis']['anomaly_regions']:
            x, y, w, h = region['bbox']
            cv2.rectangle(test_img, (x, y), (x+w, y+h), colors['anomaly'], 2)
            cv2.putText(test_img, f"ANOM: {region['confidence']:.2f}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, colors['anomaly'], 1)
        
        # Add legend
        legend_y = 30
        for method, color in colors.items():
            cv2.rectangle(test_img, (10, legend_y-15), (30, legend_y-5), color, -1)
            cv2.putText(test_img, method.capitalize(), (40, legend_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            legend_y += 25
        
        # Add summary text
        cascade_count = sum(len(d) for d in results.get('cascade_detections', {}).values())
        summary = f"Cascade: {cascade_count} | Traditional: {len(defects['scratches']) + len(defects['digs']) + len(defects['blobs'])}"
        cv2.putText(test_img, summary, (10, test_img.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, test_img)
        self.logger.info(f"Interactive visualization saved to: {output_path}")
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=2, dash_length=5):
        """Draw a dashed rectangle (like cascade detection visualization)."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def analyze_video_stream(self, video_path: str, output_dir: str, display_live: bool = False):
        """Process video stream for continuous fiber inspection (from tutorial concept)."""
        self.logger.info(f"Starting video analysis: {video_path}")
        
        # Open video capture (0 for webcam as shown in tutorial)
        if video_path == "0" or video_path == 0:
            cap = cv2.VideoCapture(0)
            self.logger.info("Using webcam for live inspection")
        else:
            cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        anomaly_frames = []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        
        self.logger.info("Press 'q' to quit video analysis")
        
        # Process video frame by frame as shown in tutorial
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply cascade detection if available
            cascade_results = {}
            if self.cascade_classifiers:
                cascade_results = self._apply_cascade_detection(gray)
            
            # Draw detections on frame
            display_frame = frame.copy()
            anomaly_detected = False
            
            # Process cascade detections
            for defect_type, detections in cascade_results.items():
                for det in detections:
                    x, y, w, h = det['bbox']
                    # Draw rectangle as shown in tutorial
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                                (0, 0, 255), 2)  # Blue color (BGR format)
                    cv2.putText(display_frame, defect_type, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    anomaly_detected = True
            
            # Initialize video writer with first frame dimensions
            if out is None:
                h, w = display_frame.shape[:2]
                out = cv2.VideoWriter(
                    str(output_path / 'analyzed_video.avi'),
                    fourcc, 20.0, (w, h)
                )
            
            # Write frame to output video
            out.write(display_frame)
            
            # Track anomaly frames
            if anomaly_detected:
                anomaly_frames.append({
                    'frame_number': frame_count,
                    'detections': cascade_results,
                    'timestamp': frame_count / 30.0  # Assuming 30 fps
                })
            
            # Display live if requested
            if display_live:
                cv2.imshow('Fiber Inspection Video', display_frame)
                
                # Check for 'q' key to quit (as shown in tutorial)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User pressed 'q' - stopping video analysis")
                    break
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count} frames, found {len(anomaly_frames)} anomalies")
        
        # Release everything
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Save video analysis summary
        summary = {
            'video_path': video_path,
            'total_frames': frame_count,
            'anomaly_frames': len(anomaly_frames),
            'anomaly_details': anomaly_frames[:100],  # First 100 anomalies
            'fps': 30,  # Assumed
            'duration_seconds': frame_count / 30.0
        }
        
        summary_path = output_path / 'video_analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Video analysis complete: {frame_count} frames, {len(anomaly_frames)} anomalies")
        return summary

    # Keep all existing methods from original code...
    # (All the statistical functions, feature extraction, comparison methods, etc. remain the same)
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        return time.strftime("%Y-%m-%d_%H:%M:%S")
    
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    loaded_data = json.load(f)
                
                if loaded_data.get('archetype_image'):
                    loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
                
                if loaded_data.get('statistical_model'):
                    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                        if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                            loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)
                
                self.reference_model = loaded_data
                self.logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                self.logger.warning(f"Could not load knowledge base: {e}")
    
    def save_knowledge_base(self):
        """Save current knowledge base to JSON."""
        try:
            save_data = self.reference_model.copy()
            
            if isinstance(save_data.get('archetype_image'), np.ndarray):
                save_data['archetype_image'] = save_data['archetype_image'].tolist()
            
            if save_data.get('statistical_model'):
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                        save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()
            
            if 'comparison_scores' in save_data:
                del save_data['comparison_scores']
            
            save_data['timestamp'] = self._get_timestamp()
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            self.logger.info(f"Knowledge base saved to {self.knowledge_base_path}")
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {e}")

    # ... (Include all other existing methods from the original script)
    # I'm not repeating them all here for brevity, but they should all be included

def main():
    """Enhanced main function with video processing option."""
    print("\n" + "="*80)
    print("OMNIFIBER ANALYZER - ENHANCED DETECTION MODULE (v2.0)".center(80))
    print("="*80)
    print("\nEnhanced with Cascade Classifiers and Video Processing")
    print("Options:")
    print("1. Analyze single image")
    print("2. Process video stream")
    print("3. Build reference model")
    print("4. Live webcam inspection")
    print("\n")
    
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    
    while True:
        choice = input("\nSelect option (1-4) or 'quit' to exit: ").strip()
        
        if choice.lower() == 'quit':
            break
        
        elif choice == '1':
            # Single image analysis
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            
            if not os.path.isfile(test_path):
                print(f"✗ File not found: {test_path}")
                continue
            
            output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            print(f"\nAnalyzing {test_path}...")
            analyzer.analyze_end_face(test_path, output_dir)
            
            print(f"\nResults saved to: {output_dir}/")
            print("  - JSON report: *_report.json")
            print("  - Visualization: *_analysis.png")
            print("  - Interactive view: *_interactive.png")
            print("  - Detailed text: *_detailed.txt")
        
        elif choice == '2':
            # Video file processing
            video_path = input("Enter path to video file: ").strip().strip('"\'')
            
            if not os.path.isfile(video_path):
                print(f"✗ File not found: {video_path}")
                continue
            
            output_dir = f"video_output_{Path(video_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
            display = input("Display live preview? (y/n): ").lower() == 'y'
            
            print(f"\nProcessing video {video_path}...")
            print("Press 'q' in video window to stop early")
            analyzer.analyze_video_stream(video_path, output_dir, display_live=display)
            
            print(f"\nResults saved to: {output_dir}/")
        
        elif choice == '3':
            # Build reference model
            ref_dir = input("Enter directory with reference images: ").strip().strip('"\'')
            
            if not os.path.isdir(ref_dir):
                print(f"✗ Directory not found: {ref_dir}")
                continue
            
            print(f"\nBuilding reference model from {ref_dir}...")
            success = analyzer.build_comprehensive_reference_model(ref_dir)
            
            if success:
                print("✓ Reference model built successfully!")
            else:
                print("✗ Failed to build reference model")
        
        elif choice == '4':
            # Live webcam inspection
            output_dir = f"webcam_output_{time.strftime('%Y%m%d_%H%M%S')}"
            
            print("\nStarting live webcam inspection...")
            print("Press 'q' in video window to stop")
            analyzer.analyze_video_stream(0, output_dir, display_live=True)
            
            print(f"\nResults saved to: {output_dir}/")
        
        else:
            print("Invalid option. Please select 1-4 or 'quit'")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
