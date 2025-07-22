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
import threading
from queue import Queue

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

@dataclass
class OmniConfig:
    """Enhanced configuration for OmniFiberAnalyzer with real-time capabilities"""
    # Core detection parameters
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    
    # Real-time processing parameters
    enable_realtime: bool = False
    target_fps: int = 30
    buffer_size: int = 5
    
    # Visualization parameters
    enable_visualization: bool = True
    show_intermediate_steps: bool = False
    visualization_scale: float = 1.0
    
    # Performance parameters
    enable_gpu_acceleration: bool = False
    max_threads: int = 4
    enable_caching: bool = True
    
    # Severity thresholds
    severity_thresholds: Optional[Dict[str, float]] = None
    
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
    """Enhanced JSON encoder with better error handling"""
    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except (TypeError, ValueError) as e:
            logging.warning(f"Could not serialize object {type(obj)}: {e}")
            return str(obj)
        return super().default(obj)

class RealTimeProcessor:
    """Real-time video processing module inspired by MediaPipe approach"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.result_queue = Queue(maxsize=config.buffer_size)
        self.processing = False
        self.cap = None
        
    def start_camera(self, camera_id: int = 0):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
    def process_frame(self, frame, analyzer):
        """Process a single frame"""
        try:
            # Save frame temporarily
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Analyze frame
            results = analyzer.detect_anomalies_comprehensive(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return results
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return None
    
    def run_realtime(self, analyzer):
        """Run real-time processing loop"""
        if not self.cap:
            raise RuntimeError("Camera not initialized")
        
        self.processing = True
        fps_tracker = FPSTracker()
        
        while self.processing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.process_frame(frame, analyzer)
            
            if results and self.config.enable_visualization:
                # Draw results on frame
                annotated_frame = self._draw_results(frame, results)
                
                # Add FPS counter
                fps = fps_tracker.get_fps()
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Fiber Optic Analysis", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def _draw_results(self, frame, results):
        """Draw analysis results on frame"""
        if not results:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw anomaly regions
        for region in results.get('local_analysis', {}).get('anomaly_regions', []):
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 0, 255)  # Red for high confidence
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange for medium
            else:
                color = (0, 255, 255)  # Yellow for low
            
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated_frame, f"{confidence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw verdict
        verdict = results.get('verdict', {})
        status = "ANOMALOUS" if verdict.get('is_anomalous', False) else "NORMAL"
        status_color = (0, 0, 255) if verdict.get('is_anomalous', False) else (0, 255, 0)
        
        cv2.putText(annotated_frame, f"Status: {status}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return annotated_frame
    
    def stop(self):
        """Stop processing"""
        self.processing = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

class FPSTracker:
    """FPS tracking utility inspired by tutorial"""
    
    def __init__(self):
        self.prev_time = 0
        self.current_time = 0
        
    def get_fps(self):
        """Calculate and return current FPS"""
        self.current_time = time.time()
        if self.prev_time == 0:
            self.prev_time = self.current_time
            return 0
        
        fps = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time
        return fps

class EnhancedOmniFiberAnalyzer:
    """Enhanced fiber optic anomaly detection system with real-time capabilities"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.real_time_processor = RealTimeProcessor(config) if config.enable_realtime else None
        self.fps_tracker = FPSTracker()
        
        # Cache for performance
        self._feature_cache = {} if config.enable_caching else None
        
        # Initialize reference model
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'timestamp': None,
            'version': '2.0'  # Version tracking
        }
        
        self.current_metadata = None
        self.load_knowledge_base()
    
    def analyze_end_face(self, image_path: str, output_dir: str, real_time: bool = False):
        """Enhanced analysis method with real-time option"""
        start_time = time.time()
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if reference model exists
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Building from single image...")
            self._build_minimal_reference(image_path)
        
        # Run analysis
        results = self.detect_anomalies_comprehensive(image_path)
        
        if results:
            # Add performance metrics
            processing_time = time.time() - start_time
            results['performance_metrics'] = {
                'processing_time': processing_time,
                'fps': 1 / processing_time if processing_time > 0 else 0,
                'timestamp': self._get_timestamp()
            }
            
            # Convert to pipeline format
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            # Save results
            self._save_results(pipeline_report, output_path, image_path)
            
            # Real-time visualization
            if real_time and self.config.enable_visualization:
                self._show_realtime_results(results)
            
            return pipeline_report
        else:
            return self._create_error_report(image_path, output_path)
    
    def start_realtime_analysis(self, camera_id: int = 0):
        """Start real-time analysis from camera"""
        if not self.real_time_processor:
            raise RuntimeError("Real-time processing not enabled in config")
        
        try:
            self.real_time_processor.start_camera(camera_id)
            self.real_time_processor.run_realtime(self)
        except Exception as e:
            self.logger.error(f"Real-time analysis failed: {e}")
            raise
    
    def _save_results(self, pipeline_report, output_path, image_path):
        """Save analysis results"""
        # JSON report
        report_path = output_path / f"{Path(image_path).stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"Saved detection report to {report_path}")
        
        # Visualization
        if self.config.enable_visualization:
            viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
            self.visualize_comprehensive_results(pipeline_report, str(viz_path))
    
    def _show_realtime_results(self, results):
        """Show real-time visualization"""
        # This would integrate with the real-time processor
        pass
    
    def _create_error_report(self, image_path, output_path):
        """Create error report"""
        error_report = {
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'success': False,
            'error': 'Analysis failed',
            'defects': []
        }
        
        report_path = output_path / f"{Path(image_path).stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        return error_report
    
    # Include all the original methods from the previous script
    # (load_image, extract_features, detect_anomalies_comprehensive, etc.)
    # with enhanced error handling and performance optimizations
    
    def load_image(self, path):
        """Enhanced image loading with better error handling"""
        self.current_metadata = None
        
        try:
            if path.lower().endswith('.json'):
                return self._load_from_json(path)
            else:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read image: {path}")
                
                self.current_metadata = {
                    'filename': os.path.basename(path),
                    'shape': img.shape,
                    'dtype': str(img.dtype)
                }
                return img
        except Exception as e:
            self.logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def extract_ultra_comprehensive_features(self, image):
        """Enhanced feature extraction with caching"""
        if self._feature_cache is not None:
            # Simple cache key based on image hash
            image_hash = hash(image.tobytes())
            if image_hash in self._feature_cache:
                return self._feature_cache[image_hash]
        
        features = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        self.logger.info("  Extracting features...")
        
        # Feature extraction methods with enhanced error handling
        feature_extractors = [
            ("Stats", self._extract_statistical_features),
            ("Norms", self._extract_matrix_norms),
            ("LBP", self._extract_lbp_features),
            ("GLCM", self._extract_glcm_features),
            ("FFT", self._extract_fourier_features),
            ("MultiScale", self._extract_multiscale_features),
            ("Morph", self._extract_morphological_features),
            ("Shape", self._extract_shape_features),
            ("SVD", self._extract_svd_features),
            ("Entropy", self._extract_entropy_features),
            ("Gradient", self._extract_gradient_features),
            ("Topology", self._extract_topological_proxy_features),
        ]
        
        for name, extractor in feature_extractors:
            try:
                extracted_features = extractor(gray)
                features.update(extracted_features)
                if self.config.show_intermediate_steps:
                    self.logger.info(f"    Extracted {len(extracted_features)} {name} features")
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for {name}: {e}")
        
        # Sanitize features
        sanitized_features = {}
        for key, value in features.items():
            sanitized_features[key] = self._sanitize_feature_value(value)
        
        # Cache results
        if self._feature_cache is not None:
            self._feature_cache[image_hash] = (sanitized_features, sorted(sanitized_features.keys()))
        
        feature_names = sorted(sanitized_features.keys())
        return sanitized_features, feature_names
    
    def detect_anomalies_comprehensive(self, test_path):
        """Enhanced anomaly detection with performance tracking"""
        start_time = time.time()
        self.logger.info(f"Analyzing: {test_path}")
        
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None
        
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        self.logger.info(f"Loaded image: {self.current_metadata}")
        
        # Convert to grayscale
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Extract features
        self.logger.info("Extracting features from test image...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # Perform analysis steps with timing
        analysis_times = {}
        
        # Global analysis
        global_start = time.time()
        global_analysis = self._perform_global_analysis(test_features)
        analysis_times['global_analysis'] = time.time() - global_start
        
        # Structural analysis
        struct_start = time.time()
        structural_analysis = self._perform_structural_analysis(test_gray)
        analysis_times['structural_analysis'] = time.time() - struct_start
        
        # Local analysis
        local_start = time.time()
        local_analysis = self._perform_local_analysis(test_gray)
        analysis_times['local_analysis'] = time.time() - local_start
        
        # Specific defects
        defect_start = time.time()
        specific_defects = self._detect_specific_defects(test_gray)
        analysis_times['specific_defects'] = time.time() - defect_start
        
        # Make verdict
        verdict = self._make_verdict(global_analysis, structural_analysis, local_analysis)
        
        total_time = time.time() - start_time
        
        self.logger.info("Analysis Complete!")
        
        return {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            'global_analysis': global_analysis,
            'structural_analysis': structural_analysis,
            'local_analysis': local_analysis,
            'specific_defects': specific_defects,
            'verdict': verdict,
            'analysis_times': analysis_times,
            'total_processing_time': total_time
        }
    
    def _perform_global_analysis(self, test_features):
        """Perform global statistical analysis"""
        # Implementation similar to original but with better error handling
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Create feature vector
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        # Compute Mahalanobis distance with error handling
        try:
            diff = test_vector - stat_model['robust_mean']
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except Exception as e:
            self.logger.warning(f"Mahalanobis calculation failed: {e}")
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)
        
        return {
            'mahalanobis_distance': float(mahalanobis_dist),
            'test_vector': test_vector,
            'feature_deviations': self._compute_feature_deviations(test_vector, stat_model, feature_names)
        }
    
    def _perform_structural_analysis(self, test_gray):
        """Perform structural similarity analysis"""
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        return self.compute_image_structural_comparison(test_gray_resized, archetype)
    
    def _perform_local_analysis(self, test_gray):
        """Perform local anomaly analysis"""
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        return {
            'anomaly_map': anomaly_map,
            'anomaly_regions': anomaly_regions
        }
    
    def _make_verdict(self, global_analysis, structural_analysis, local_analysis):
        """Make final verdict based on all analyses"""
        thresholds = self.reference_model['learned_thresholds']
        
        # Multiple criteria for anomaly detection
        is_anomalous = (
            global_analysis['mahalanobis_distance'] > max(thresholds.get('anomaly_threshold', 1.0), 1e-6) or
            structural_analysis['ssim'] < 0.7 or
            len(local_analysis['anomaly_regions']) > 3 or
            any(region['confidence'] > 0.8 for region in local_analysis['anomaly_regions'])
        )
        
        # Calculate confidence
        confidence = min(1.0, max(
            global_analysis['mahalanobis_distance'] / max(thresholds.get('anomaly_threshold', 1.0), 1e-6),
            1 - structural_analysis['ssim'],
            len(local_analysis['anomaly_regions']) / 10
        ))
        
        return {
            'is_anomalous': is_anomalous,
            'confidence': float(confidence),
            'criteria_triggered': {
                'mahalanobis': global_analysis['mahalanobis_distance'] > max(thresholds.get('anomaly_threshold', 1.0), 1e-6),
                'structural': structural_analysis['ssim'] < 0.7,
                'local': len(local_analysis['anomaly_regions']) > 3,
            }
        }
    
    # Include all other methods from the original script with enhancements...
    # (Due to length constraints, I'm showing the key enhanced methods)
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return time.strftime("%Y-%m-%d_%H:%M:%S")

def main():
    """Enhanced main function with multiple modes"""
    print("\n" + "="*80)
    print("ENHANCED OMNIFIBER ANALYZER - DETECTION MODULE (v2.0)".center(80))
    print("="*80)
    print("\nAvailable modes:")
    print("1. Single image analysis")
    print("2. Real-time camera analysis")
    print("3. Batch processing")
    
    # Create configuration
    config = OmniConfig(
        enable_realtime=True,
        enable_visualization=True,
        show_intermediate_steps=True
    )
    
    # Initialize analyzer
    analyzer = EnhancedOmniFiberAnalyzer(config)
    
    while True:
        mode = input("\nSelect mode (1-3) or 'quit': ").strip()
        
        if mode.lower() == 'quit':
            break
        elif mode == '1':
            # Single image mode
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            if os.path.isfile(test_path):
                output_dir = f"enhanced_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                analyzer.analyze_end_face(test_path, output_dir)
            else:
                print(f"✗ File not found: {test_path}")
        elif mode == '2':
            # Real-time mode
            try:
                camera_id = int(input("Enter camera ID (0 for default): ") or "0")
                print("Starting real-time analysis. Press 'q' to quit.")
                analyzer.start_realtime_analysis(camera_id)
            except ValueError:
                print("Invalid camera ID")
            except Exception as e:
                print(f"Real-time analysis failed: {e}")
        elif mode == '3':
            # Batch mode
            dir_path = input("Enter directory path: ").strip().strip('"\'')
            if os.path.isdir(dir_path):
                for file_path in Path(dir_path).glob("*"):
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                        output_dir = f"batch_output_{time.strftime('%Y%m%d_%H%M%S')}"
                        analyzer.analyze_end_face(str(file_path), output_dir)
            else:
                print(f"✗ Directory not found: {dir_path}")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")

if __name__ == "__main__":
    main()
