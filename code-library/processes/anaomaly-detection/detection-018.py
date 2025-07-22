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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')

# Configure logging system to display timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


@dataclass
class OmniConfig:
    """Enhanced configuration for OmniFiberAnalyzer with multiple anomaly detection methods"""
    # Path to saved knowledge base file containing reference model data
    knowledge_base_path: Optional[str] = None
    # Minimum pixel area for a region to be considered a defect
    min_defect_size: int = 10
    # Maximum pixel area for a defect
    max_defect_size: int = 5000
    # Dictionary mapping severity levels to confidence thresholds
    severity_thresholds: Optional[Dict[str, float]] = None
    # Minimum confidence score (0-1) to report a detected anomaly
    confidence_threshold: float = 0.3
    # Multiplier for standard deviation to set anomaly detection threshold
    anomaly_threshold_multiplier: float = 2.5
    # Whether to generate and save visualization images
    enable_visualization: bool = True
    
    # New parameters for enhanced anomaly detection
    # Expected contamination rate (proportion of anomalies in training data)
    contamination_rate: float = 0.1
    # Robust Z-score threshold (replace traditional Z-score)
    robust_zscore_threshold: float = 3.5
    # Number of neighbors for LOF
    lof_n_neighbors: int = 20
    # Isolation Forest parameters
    isolation_forest_estimators: int = 100
    # Enable ensemble voting (combine multiple methods)
    enable_ensemble: bool = True
    # Minimum number of methods that must agree for anomaly detection
    ensemble_min_votes: int = 2
    
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


class EnhancedOmniFiberAnalyzer:
    """Enhanced fiber optic anomaly detection system with multiple ML methods."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "enhanced_fiber_anomaly_kb.json"
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'ml_models': {},  # Store trained ML models
            'timestamp': None
        }
        self.current_metadata = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models
        self.isolation_forest = None
        self.lof_model = None
        
        self.load_knowledge_base()
        
    def compute_robust_zscore(self, values: np.ndarray, test_value: float) -> float:
        """
        Compute robust Z-score using Median Absolute Deviation (MAD)
        Based on the tutorial's robust Z-score method
        """
        # Compute median and MAD
        median_val = np.median(values)
        mad = median_abs_deviation(values)
        
        # Handle case where MAD is zero or very small
        if mad < 1e-10:
            self.logger.warning("MAD is very small, using fallback robust scaling")
            # Use interquartile range as fallback
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            if iqr < 1e-10:
                return 0.0  # All values are the same
            mad = iqr / 1.349  # Convert IQR to MAD equivalent
        
        # Compute robust Z-score with scaling factor
        robust_zscore = 0.6745 * (test_value - median_val) / mad
        return abs(robust_zscore)
    
    def detect_anomalies_robust_zscore(self, test_features: Dict, reference_features: List[Dict]) -> Dict:
        """Detect anomalies using robust Z-score method"""
        if not reference_features:
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'robust_zscore'}
        
        # Get feature names
        feature_names = sorted(test_features.keys())
        anomaly_scores = []
        deviant_features = []
        
        # Compute robust Z-score for each feature
        for fname in feature_names:
            # Extract reference values for this feature
            ref_values = np.array([ref.get(fname, 0) for ref in reference_features])
            test_value = test_features.get(fname, 0)
            
            # Compute robust Z-score
            robust_z = self.compute_robust_zscore(ref_values, test_value)
            anomaly_scores.append(robust_z)
            
            # Track highly deviant features
            if robust_z > self.config.robust_zscore_threshold:
                deviant_features.append({
                    'feature': fname,
                    'robust_zscore': robust_z,
                    'test_value': test_value,
                    'reference_median': np.median(ref_values)
                })
        
        # Overall anomaly decision
        max_zscore = max(anomaly_scores) if anomaly_scores else 0
        is_anomalous = max_zscore > self.config.robust_zscore_threshold
        confidence = min(1.0, max_zscore / self.config.robust_zscore_threshold) if self.config.robust_zscore_threshold > 0 else 0
        
        return {
            'is_anomalous': is_anomalous,
            'confidence': confidence,
            'max_robust_zscore': max_zscore,
            'deviant_features': sorted(deviant_features, key=lambda x: x['robust_zscore'], reverse=True)[:10],
            'method': 'robust_zscore'
        }
    
    def train_isolation_forest(self, reference_features: List[Dict]) -> bool:
        """Train Isolation Forest model on reference features"""
        if len(reference_features) < 5:
            self.logger.warning("Not enough reference samples for Isolation Forest training")
            return False
        
        try:
            # Convert features to matrix
            feature_names = sorted(reference_features[0].keys())
            feature_matrix = np.zeros((len(reference_features), len(feature_names)))
            
            for i, features in enumerate(reference_features):
                for j, fname in enumerate(feature_names):
                    feature_matrix[i, j] = features.get(fname, 0)
            
            # Handle NaN and infinite values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Initialize and train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.config.contamination_rate,
                n_estimators=self.config.isolation_forest_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            self.isolation_forest.fit(feature_matrix)
            self.logger.info("Isolation Forest trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to train Isolation Forest: {e}")
            return False
    
    def detect_anomalies_isolation_forest(self, test_features: Dict) -> Dict:
        """Detect anomalies using Isolation Forest"""
        if self.isolation_forest is None:
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'isolation_forest', 'error': 'Model not trained'}
        
        try:
            # Convert test features to vector
            feature_names = sorted(test_features.keys())
            test_vector = np.array([test_features.get(fname, 0) for fname in feature_names]).reshape(1, -1)
            
            # Handle NaN and infinite values
            test_vector = np.nan_to_num(test_vector, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Predict anomaly (-1 = anomaly, 1 = normal)
            prediction = self.isolation_forest.predict(test_vector)[0]
            
            # Get anomaly score (more negative = more anomalous)
            anomaly_score = self.isolation_forest.decision_function(test_vector)[0]
            
            # Convert to confidence (0-1 scale)
            confidence = max(0.0, min(1.0, -anomaly_score))
            
            return {
                'is_anomalous': prediction == -1,
                'confidence': confidence,
                'anomaly_score': anomaly_score,
                'method': 'isolation_forest'
            }
            
        except Exception as e:
            self.logger.error(f"Isolation Forest prediction failed: {e}")
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'isolation_forest', 'error': str(e)}
    
    def train_local_outlier_factor(self, reference_features: List[Dict]) -> bool:
        """Train Local Outlier Factor model on reference features"""
        if len(reference_features) < self.config.lof_n_neighbors + 1:
            self.logger.warning(f"Not enough reference samples for LOF (need at least {self.config.lof_n_neighbors + 1})")
            return False
        
        try:
            # Convert features to matrix
            feature_names = sorted(reference_features[0].keys())
            feature_matrix = np.zeros((len(reference_features), len(feature_names)))
            
            for i, features in enumerate(reference_features):
                for j, fname in enumerate(feature_names):
                    feature_matrix[i, j] = features.get(fname, 0)
            
            # Handle NaN and infinite values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Initialize LOF for novelty detection
            self.lof_model = LocalOutlierFactor(
                n_neighbors=min(self.config.lof_n_neighbors, len(reference_features) - 1),
                contamination=self.config.contamination_rate,
                novelty=True,  # Enable novelty detection for new samples
                n_jobs=-1
            )
            
            self.lof_model.fit(feature_matrix)
            self.logger.info("Local Outlier Factor trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to train LOF: {e}")
            return False
    
    def detect_anomalies_lof(self, test_features: Dict) -> Dict:
        """Detect anomalies using Local Outlier Factor"""
        if self.lof_model is None:
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'lof', 'error': 'Model not trained'}
        
        try:
            # Convert test features to vector
            feature_names = sorted(test_features.keys())
            test_vector = np.array([test_features.get(fname, 0) for fname in feature_names]).reshape(1, -1)
            
            # Handle NaN and infinite values
            test_vector = np.nan_to_num(test_vector, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Predict anomaly (-1 = anomaly, 1 = normal)
            prediction = self.lof_model.predict(test_vector)[0]
            
            # Get anomaly score (more negative = more anomalous)
            anomaly_score = self.lof_model.decision_function(test_vector)[0]
            
            # Convert to confidence (0-1 scale)
            confidence = max(0.0, min(1.0, -anomaly_score))
            
            return {
                'is_anomalous': prediction == -1,
                'confidence': confidence,
                'anomaly_score': anomaly_score,
                'method': 'lof'
            }
            
        except Exception as e:
            self.logger.error(f"LOF prediction failed: {e}")
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'lof', 'error': str(e)}
    
    def ensemble_anomaly_detection(self, test_features: Dict, reference_features: List[Dict]) -> Dict:
        """
        Ensemble anomaly detection combining multiple methods
        Similar to the tutorial's approach of comparing different methods
        """
        methods_results = []
        
        # Method 1: Robust Z-score
        robust_result = self.detect_anomalies_robust_zscore(test_features, reference_features)
        methods_results.append(robust_result)
        
        # Method 2: Isolation Forest
        isolation_result = self.detect_anomalies_isolation_forest(test_features)
        methods_results.append(isolation_result)
        
        # Method 3: Local Outlier Factor
        lof_result = self.detect_anomalies_lof(test_features)
        methods_results.append(lof_result)
        
        # Method 4: Original statistical method (Mahalanobis)
        original_result = self._detect_anomalies_statistical(test_features)
        methods_results.append(original_result)
        
        # Ensemble voting
        anomaly_votes = sum(1 for result in methods_results if result.get('is_anomalous', False))
        confidence_scores = [result.get('confidence', 0.0) for result in methods_results]
        
        # Ensemble decision
        ensemble_anomalous = anomaly_votes >= self.config.ensemble_min_votes
        ensemble_confidence = np.mean(confidence_scores)
        
        # Find best performing method (highest confidence)
        best_method = max(methods_results, key=lambda x: x.get('confidence', 0))
        
        return {
            'ensemble_result': {
                'is_anomalous': ensemble_anomalous,
                'confidence': ensemble_confidence,
                'votes': anomaly_votes,
                'total_methods': len(methods_results),
                'best_method': best_method.get('method', 'unknown')
            },
            'individual_results': {
                'robust_zscore': robust_result,
                'isolation_forest': isolation_result,
                'local_outlier_factor': lof_result,
                'statistical_mahalanobis': original_result
            },
            'method_comparison': {
                'agreement_rate': anomaly_votes / len(methods_results),
                'confidence_range': [min(confidence_scores), max(confidence_scores)],
                'most_confident': best_method.get('method', 'unknown')
            }
        }
    
    def _detect_anomalies_statistical(self, test_features: Dict) -> Dict:
        """Original statistical anomaly detection for comparison"""
        if not self.reference_model.get('statistical_model'):
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'statistical'}
        
        try:
            stat_model = self.reference_model['statistical_model']
            feature_names = self.reference_model['feature_names']
            
            # Create feature vector
            test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
            
            # Compute Mahalanobis distance
            diff = test_vector - stat_model['robust_mean']
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
            
            # Get threshold
            threshold = self.reference_model['learned_thresholds'].get('anomaly_threshold', 2.5)
            
            return {
                'is_anomalous': mahalanobis_dist > threshold,
                'confidence': min(1.0, mahalanobis_dist / threshold) if threshold > 0 else 0,
                'mahalanobis_distance': mahalanobis_dist,
                'method': 'statistical'
            }
            
        except Exception as e:
            self.logger.error(f"Statistical detection failed: {e}")
            return {'is_anomalous': False, 'confidence': 0.0, 'method': 'statistical', 'error': str(e)}
    
    def build_enhanced_reference_model(self, ref_dir: str) -> bool:
        """Build enhanced reference model with ML training"""
        self.logger.info(f"Building Enhanced Reference Model from: {ref_dir}")
        
        # First build the original statistical model
        success = self.build_comprehensive_reference_model(ref_dir)
        if not success:
            return False
        
        # Train ML models on the reference features
        reference_features = self.reference_model.get('features', [])
        
        if len(reference_features) < 2:
            self.logger.error("Not enough reference features for ML training")
            return False
        
        # Train Isolation Forest
        self.logger.info("Training Isolation Forest...")
        iso_success = self.train_isolation_forest(reference_features)
        
        # Train Local Outlier Factor
        self.logger.info("Training Local Outlier Factor...")
        lof_success = self.train_local_outlier_factor(reference_features)
        
        # Store model training status
        self.reference_model['ml_models'] = {
            'isolation_forest_trained': iso_success,
            'lof_trained': lof_success,
            'training_timestamp': self._get_timestamp()
        }
        
        # Save enhanced model
        self.save_knowledge_base()
        
        self.logger.info("Enhanced Reference Model Built Successfully!")
        self.logger.info(f"  - Isolation Forest: {'✓' if iso_success else '✗'}")
        self.logger.info(f"  - Local Outlier Factor: {'✓' if lof_success else '✗'}")
        
        return True
    
    def detect_anomalies_enhanced(self, test_path: str) -> Dict:
        """Enhanced anomaly detection using ensemble of methods"""
        self.logger.info(f"Enhanced Analysis: {test_path}")
        
        # Load and process test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Extract features
        self.logger.info("Extracting features...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # Get reference features for comparison
        reference_features = self.reference_model.get('features', [])
        
        if not reference_features:
            self.logger.warning("No reference features available")
            return None
        
        # Run ensemble anomaly detection
        self.logger.info("Running ensemble anomaly detection...")
        ensemble_results = self.ensemble_anomaly_detection(test_features, reference_features)
        
        # Add traditional analysis for compatibility
        traditional_results = self.detect_anomalies_comprehensive(test_path)
        
        # Combine results
        enhanced_results = {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            
            # Enhanced anomaly detection results
            'enhanced_detection': ensemble_results,
            
            # Original comprehensive results for compatibility
            'traditional_analysis': traditional_results,
            
            # Final verdict based on ensemble
            'final_verdict': {
                'is_anomalous': ensemble_results['ensemble_result']['is_anomalous'],
                'confidence': ensemble_results['ensemble_result']['confidence'],
                'detection_method': 'ensemble',
                'contributing_methods': [
                    result['method'] for result in ensemble_results['individual_results'].values()
                    if result.get('is_anomalous', False)
                ]
            }
        }
        
        return enhanced_results
    
    def analyze_end_face_enhanced(self, image_path: str, output_dir: str) -> Dict:
        """Enhanced analysis method for pipeline compatibility"""
        self.logger.info(f"Enhanced analyzing fiber end face: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if models are trained
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Building minimal reference...")
            self._build_minimal_reference(image_path)
        
        # Run enhanced analysis
        results = self.detect_anomalies_enhanced(image_path)
        
        if results:
            # Convert to pipeline format
            pipeline_report = self._convert_enhanced_to_pipeline_format(results, image_path)
            
            # Save JSON report
            report_path = output_path / f"{Path(image_path).stem}_enhanced_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
            
            # Generate enhanced visualization
            if self.config.enable_visualization:
                viz_path = output_path / f"{Path(image_path).stem}_enhanced_analysis.png"
                self.visualize_enhanced_results(results, str(viz_path))
            
            # Generate detailed comparison report
            comparison_path = output_path / f"{Path(image_path).stem}_method_comparison.txt"
            self.generate_method_comparison_report(results, str(comparison_path))
            
            return pipeline_report
        else:
            self.logger.error(f"Enhanced analysis failed for {image_path}")
            return self._create_error_report(image_path)
    
    def _convert_enhanced_to_pipeline_format(self, results: Dict, image_path: str) -> Dict:
        """Convert enhanced results to pipeline format"""
        enhanced = results['enhanced_detection']
        ensemble = enhanced['ensemble_result']
        
        # Get traditional defects for compatibility
        traditional = results.get('traditional_analysis', {})
        defects = []
        
        if traditional:
            # Use original defect conversion
            defects = self._extract_defects_from_traditional(traditional)
        
        # Calculate quality score based on ensemble
        quality_score = float(100 * (1 - ensemble['confidence']))
        
        report = {
            'source_image': image_path,
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'analysis_complete': True,
            'success': True,
            'overall_quality_score': quality_score,
            'defects': defects,
            
            # Enhanced detection results
            'enhanced_detection': {
                'ensemble_verdict': ensemble['is_anomalous'],
                'ensemble_confidence': float(ensemble['confidence']),
                'methods_agreement': float(ensemble['votes'] / ensemble['total_methods']),
                'best_method': ensemble['best_method'],
                'individual_methods': {
                    method: {
                        'anomalous': result.get('is_anomalous', False),
                        'confidence': float(result.get('confidence', 0.0)),
                        'details': {k: v for k, v in result.items() 
                                  if k not in ['is_anomalous', 'confidence']}
                    }
                    for method, result in enhanced['individual_results'].items()
                }
            },
            
            'summary': {
                'total_defects': len(defects),
                'is_anomalous': ensemble['is_anomalous'],
                'anomaly_confidence': float(ensemble['confidence']),
                'quality_score': quality_score,
                'detection_method': 'enhanced_ensemble'
            },
            
            'analysis_metadata': {
                'analyzer': 'enhanced_omni_fiber_analyzer',
                'version': '2.0',
                'methods_used': list(enhanced['individual_results'].keys()),
                'ensemble_min_votes': self.config.ensemble_min_votes,
                'contamination_rate': self.config.contamination_rate
            }
        }
        
        return report
    
    def _extract_defects_from_traditional(self, traditional_results: Dict) -> List[Dict]:
        """Extract defects from traditional analysis for compatibility"""
        defects = []
        defect_id = 1
        
        # Extract from local analysis if available
        local_analysis = traditional_results.get('local_analysis', {})
        for region in local_analysis.get('anomaly_regions', []):
            x, y, w, h = region['bbox']
            cx, cy = region['centroid']
            
            defect = {
                'defect_id': f"ENH_ANOM_{defect_id:04d}",
                'defect_type': 'ENHANCED_ANOMALY',
                'location_xy': [cx, cy],
                'bbox': [x, y, w, h],
                'area_px': region['area'],
                'confidence': float(region['confidence']),
                'severity': self._confidence_to_severity(region['confidence']),
                'contributing_algorithms': ['enhanced_ensemble']
            }
            defects.append(defect)
            defect_id += 1
        
        return defects
    
    def visualize_enhanced_results(self, results: Dict, output_path: str):
        """Create enhanced visualization with method comparison"""
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Get images
        test_img = results['test_image']
        if len(test_img.shape) == 3:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        
        enhanced = results['enhanced_detection']
        
        # Panel 1: Original Test Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(test_img_rgb)
        ax1.set_title('Test Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Ensemble Result
        ax2 = fig.add_subplot(gs[0, 1])
        ensemble = enhanced['ensemble_result']
        result_color = 'red' if ensemble['is_anomalous'] else 'green'
        
        ax2.imshow(test_img_rgb)
        if ensemble['is_anomalous']:
            # Add red border for anomalies
            for spine in ax2.spines.values():
                spine.set_color('red')
                spine.set_linewidth(5)
        
        ax2.set_title(f'Ensemble: {"ANOMALOUS" if ensemble["is_anomalous"] else "NORMAL"}\n'
                     f'Confidence: {ensemble["confidence"]:.1%} '
                     f'({ensemble["votes"]}/{ensemble["total_methods"]} votes)',
                     fontsize=14, fontweight='bold', color=result_color)
        ax2.axis('off')
        
        # Panel 3: Method Agreement Chart
        ax3 = fig.add_subplot(gs[0, 2])
        methods = list(enhanced['individual_results'].keys())
        method_labels = [m.replace('_', '\n') for m in methods]
        confidences = [enhanced['individual_results'][m].get('confidence', 0) for m in methods]
        anomalous = [enhanced['individual_results'][m].get('is_anomalous', False) for m in methods]
        
        colors = ['red' if anom else 'green' for anom in anomalous]
        bars = ax3.bar(method_labels, confidences, color=colors, alpha=0.7)
        ax3.set_ylabel('Confidence Score')
        ax3.set_title('Method Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Panel 4-6: Individual Method Results
        method_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
        method_names = ['robust_zscore', 'isolation_forest', 'local_outlier_factor']
        method_titles = ['Robust Z-Score', 'Isolation Forest', 'Local Outlier Factor']
        
        for ax, method, title in zip(method_axes, method_names, method_titles):
            result = enhanced['individual_results'].get(method, {})
            is_anom = result.get('is_anomalous', False)
            conf = result.get('confidence', 0)
            
            ax.imshow(test_img_rgb)
            if is_anom:
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(3)
            
            ax.set_title(f'{title}\n{"ANOMALY" if is_anom else "NORMAL"} ({conf:.1%})',
                        fontsize=12, fontweight='bold',
                        color='red' if is_anom else 'green')
            ax.axis('off')
        
        # Panel 7: Detailed Robust Z-Score Analysis
        ax7 = fig.add_subplot(gs[2, :])
        robust_result = enhanced['individual_results'].get('robust_zscore', {})
        deviant_features = robust_result.get('deviant_features', [])
        
        if deviant_features:
            feature_names = [f['feature'].replace('_', '\n') for f in deviant_features[:10]]
            z_scores = [f['robust_zscore'] for f in deviant_features[:10]]
            
            colors = ['red' if z > 3.5 else 'orange' if z > 2.5 else 'yellow' for z in z_scores]
            bars = ax7.barh(feature_names, z_scores, color=colors)
            ax7.set_xlabel('Robust Z-Score')
            ax7.set_title('Most Deviant Features (Robust Z-Score Analysis)', fontsize=14, fontweight='bold')
            ax7.axvline(x=3.5, color='red', linestyle='--', alpha=0.5, label='Anomaly threshold')
            ax7.legend()
            
            for bar, z in zip(bars, z_scores):
                width = bar.get_width()
                ax7.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{z:.1f}', va='center', fontsize=10)
        else:
            ax7.text(0.5, 0.5, 'No deviant features detected', ha='center', va='center',
                    transform=ax7.transAxes, fontsize=14)
            ax7.set_title('Robust Z-Score Analysis', fontsize=14, fontweight='bold')
        
        # Panel 8: Summary Statistics
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create summary text
        summary_text = f"""ENHANCED ANOMALY DETECTION SUMMARY

Ensemble Decision: {'ANOMALOUS' if ensemble['is_anomalous'] else 'NORMAL'}
Overall Confidence: {ensemble['confidence']:.1%}
Method Agreement: {ensemble['votes']}/{ensemble['total_methods']} methods agree
Best Method: {ensemble['best_method']}

Individual Method Results:
"""
        
        for method, result in enhanced['individual_results'].items():
            method_name = method.replace('_', ' ').title()
            is_anom = result.get('is_anomalous', False)
            conf = result.get('confidence', 0)
            summary_text += f"• {method_name}: {'ANOMALY' if is_anom else 'NORMAL'} ({conf:.1%})\n"
        
        # Method comparison stats
        comparison = enhanced['method_comparison']
        summary_text += f"""
Method Comparison:
• Agreement Rate: {comparison['agreement_rate']:.1%}
• Confidence Range: {comparison['confidence_range'][0]:.2f} - {comparison['confidence_range'][1]:.2f}
• Most Confident: {comparison['most_confident']}

Configuration:
• Contamination Rate: {self.config.contamination_rate:.1%}
• Robust Z-Score Threshold: {self.config.robust_zscore_threshold}
• Ensemble Min Votes: {self.config.ensemble_min_votes}
• LOF Neighbors: {self.config.lof_n_neighbors}"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Main title
        source_name = results['metadata'].get('filename', 'Unknown')
        fig.suptitle(f'Enhanced Multi-Method Anomaly Analysis\nTest: {source_name}',
                    fontsize=18, fontweight='bold')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Enhanced visualization saved to: {output_path}")
    
    def generate_method_comparison_report(self, results: Dict, output_path: str):
        """Generate detailed method comparison report"""
        enhanced = results['enhanced_detection']
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED MULTI-METHOD ANOMALY DETECTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # File information
            f.write("FILE INFORMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Test File: {results['metadata'].get('filename', 'Unknown')}\n")
            f.write(f"Analysis Date: {self._get_timestamp()}\n\n")
            
            # Ensemble results
            ensemble = enhanced['ensemble_result']
            f.write("ENSEMBLE RESULTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Final Decision: {'ANOMALOUS' if ensemble['is_anomalous'] else 'NORMAL'}\n")
            f.write(f"Ensemble Confidence: {ensemble['confidence']:.1%}\n")
            f.write(f"Method Agreement: {ensemble['votes']}/{ensemble['total_methods']} methods\n")
            f.write(f"Best Performing Method: {ensemble['best_method']}\n\n")
            
            # Individual method results
            f.write("INDIVIDUAL METHOD RESULTS\n")
            f.write("-"*40 + "\n")
            
            for method, result in enhanced['individual_results'].items():
                method_name = method.replace('_', ' ').title()
                f.write(f"\n{method_name}:\n")
                f.write(f"  Decision: {'ANOMALY' if result.get('is_anomalous', False) else 'NORMAL'}\n")
                f.write(f"  Confidence: {result.get('confidence', 0):.1%}\n")
                
                # Method-specific details
                if method == 'robust_zscore':
                    f.write(f"  Max Robust Z-Score: {result.get('max_robust_zscore', 0):.2f}\n")
                    f.write(f"  Threshold: {self.config.robust_zscore_threshold}\n")
                elif method == 'isolation_forest':
                    f.write(f"  Anomaly Score: {result.get('anomaly_score', 0):.4f}\n")
                elif method == 'local_outlier_factor':
                    f.write(f"  LOF Score: {result.get('anomaly_score', 0):.4f}\n")
                elif method == 'statistical_mahalanobis':
                    f.write(f"  Mahalanobis Distance: {result.get('mahalanobis_distance', 0):.4f}\n")
            
            # Method comparison statistics
            comparison = enhanced['method_comparison']
            f.write(f"\nMETHOD COMPARISON STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Agreement Rate: {comparison['agreement_rate']:.1%}\n")
            f.write(f"Confidence Range: {comparison['confidence_range'][0]:.3f} - {comparison['confidence_range'][1]:.3f}\n")
            f.write(f"Most Confident Method: {comparison['most_confident']}\n\n")
            
            # Configuration used
            f.write("ANALYSIS CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Contamination Rate: {self.config.contamination_rate:.1%}\n")
            f.write(f"Robust Z-Score Threshold: {self.config.robust_zscore_threshold}\n")
            f.write(f"Ensemble Min Votes: {self.config.ensemble_min_votes}\n")
            f.write(f"LOF Neighbors: {self.config.lof_n_neighbors}\n")
            f.write(f"Isolation Forest Estimators: {self.config.isolation_forest_estimators}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF ENHANCED REPORT\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Method comparison report saved to: {output_path}")
    
    # Include all original methods for compatibility
    def load_image(self, path):
        """Load image from JSON or standard image file."""
        self.current_metadata = None
        
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not read image: {path}")
                return None
            self.current_metadata = {'filename': os.path.basename(path)}
            return img
    
    def _load_from_json(self, json_path):
        """Load matrix from JSON file with bounds checking."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            channels = data['image_dimensions'].get('channels', 3)
            
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            oob_count = 0
            
            for pixel in data['pixels']:
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                if 0 <= x < width and 0 <= y < height:
                    bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                    if isinstance(bgr, (int, float)):
                        bgr = [bgr] * 3
                    matrix[y, x] = bgr[:3]
                else:
                    oob_count += 1
            
            if oob_count > 0:
                self.logger.warning(f"Skipped {oob_count} out-of-bounds pixels")
            
            self.current_metadata = {
                'filename': data.get('filename', os.path.basename(json_path)),
                'width': width,
                'height': height,
                'channels': channels,
                'json_path': json_path
            }
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading JSON {json_path}: {e}")
            return None
    
    def extract_ultra_comprehensive_features(self, image):
        """Extract comprehensive features (same as original)"""
        features = {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Extract all feature types (simplified for space - use original implementation)
        features.update(self._extract_statistical_features(gray))
        features.update(self._extract_matrix_norms(gray))
        # Add other feature extraction methods as needed...
        
        sanitized_features = {}
        for key, value in features.items():
            sanitized_features[key] = self._sanitize_feature_value(value)
        
        feature_names = sorted(sanitized_features.keys())
        return sanitized_features, feature_names
    
    def _extract_statistical_features(self, gray):
        """Extract basic statistical features"""
        flat = gray.flatten()
        percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
        
        return {
            'stat_mean': float(np.mean(gray)),
            'stat_std': float(np.std(gray)),
            'stat_variance': float(np.var(gray)),
            'stat_min': float(np.min(gray)),
            'stat_max': float(np.max(gray)),
            'stat_range': float(np.max(gray) - np.min(gray)),
            'stat_median': float(np.median(gray)),
            'stat_p10': float(percentiles[0]),
            'stat_p25': float(percentiles[1]),
            'stat_p50': float(percentiles[2]),
            'stat_p75': float(percentiles[3]),
            'stat_p90': float(percentiles[4]),
        }
    
    def _extract_matrix_norms(self, gray):
        """Extract matrix norms"""
        return {
            'norm_frobenius': float(np.linalg.norm(gray, 'fro')),
            'norm_l1': float(np.sum(np.abs(gray))),
            'norm_l2': float(np.sqrt(np.sum(gray**2))),
            'norm_linf': float(np.max(np.abs(gray))),
        }
    
    def _sanitize_feature_value(self, value):
        """Ensure feature value is finite and valid."""
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) > 0 else 0.0
        
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    
    def build_comprehensive_reference_model(self, ref_dir):
        """Build reference model (simplified version)"""
        # Implementation would be similar to original but adapted
        # This is a placeholder - use original implementation
        pass
    
    def detect_anomalies_comprehensive(self, test_path):
        """Original comprehensive detection (simplified)"""
        # Implementation would be the original method
        # This is a placeholder - use original implementation
        return {}
    
    def _confidence_to_severity(self, confidence: float) -> str:
        """Convert confidence score to severity level"""
        for severity, threshold in sorted(self.config.severity_thresholds.items(), 
                                        key=lambda x: x[1], reverse=True):
            if confidence >= threshold:
                return severity
        return 'NEGLIGIBLE'
    
    def _build_minimal_reference(self, image_path: str):
        """Build minimal reference from single image"""
        # Use original implementation
        pass
    
    def load_knowledge_base(self):
        """Load knowledge base"""
        # Use original implementation
        pass
    
    def save_knowledge_base(self):
        """Save knowledge base"""
        # Use original implementation
        pass
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return time.strftime("%Y-%m-%d_%H:%M:%S")
    
    def _create_error_report(self, image_path: str) -> Dict:
        """Create error report"""
        return {
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'success': False,
            'error': 'Enhanced analysis failed',
            'defects': []
        }


def main():
    """Main execution with enhanced capabilities"""
    print("\n" + "="*80)
    print("ENHANCED OMNI-FIBER ANALYZER - MULTI-METHOD DETECTION (v2.0)".center(80))
    print("="*80)
    print("\nFeatures:")
    print("• Robust Z-Score (MAD-based)")
    print("• Isolation Forest")
    print("• Local Outlier Factor") 
    print("• Ensemble Voting")
    print("• Method Comparison\n")
    
    # Create enhanced configuration
    config = OmniConfig(
        contamination_rate=0.05,  # Expect 5% anomalies
        robust_zscore_threshold=3.5,
        lof_n_neighbors=20,
        isolation_forest_estimators=100,
        enable_ensemble=True,
        ensemble_min_votes=2
    )
    
    # Initialize enhanced analyzer
    analyzer = EnhancedOmniFiberAnalyzer(config)
    
    # Interactive loop
    while True:
        print("\nOptions:")
        print("1. Analyze single image")
        print("2. Build enhanced reference model")
        print("3. Quit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '3':
            break
        elif choice == '1':
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            
            if not os.path.isfile(test_path):
                print(f"✗ File not found: {test_path}")
                continue
            
            output_dir = f"enhanced_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
            
            print(f"\nRunning enhanced analysis on {test_path}...")
            analyzer.analyze_end_face_enhanced(test_path, output_dir)
            
            print(f"\nEnhanced results saved to: {output_dir}/")
            print("  - Enhanced JSON report: *_enhanced_report.json")
            print("  - Enhanced visualization: *_enhanced_analysis.png")
            print("  - Method comparison: *_method_comparison.txt")
            
        elif choice == '2':
            ref_dir = input("Enter path to reference images directory: ").strip().strip('"\'')
            
            if not os.path.isdir(ref_dir):
                print(f"✗ Directory not found: {ref_dir}")
                continue
            
            print(f"\nBuilding enhanced reference model from {ref_dir}...")
            success = analyzer.build_enhanced_reference_model(ref_dir)
            
            if success:
                print("✓ Enhanced reference model built successfully!")
            else:
                print("✗ Failed to build enhanced reference model")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
