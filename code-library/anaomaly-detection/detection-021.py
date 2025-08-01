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

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16, ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not installed. CNN features will be disabled.")

# Configure logging system to display timestamps, log levels, and messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


@dataclass
class OmniConfig:
    """Enhanced configuration with CNN options"""
    # Original configuration parameters
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    
    # New CNN-related parameters
    use_cnn_features: bool = True
    cnn_model_type: str = 'vgg16'  # Options: 'vgg16', 'resnet50', 'custom'
    cnn_layer_name: str = 'block5_conv3'  # Which layer to extract features from
    cnn_feature_weight: float = 0.3  # Weight of CNN features in anomaly score
    enable_activation_maps: bool = True
    cnn_batch_size: int = 1
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


class CNNFeatureExtractor:
    """Extract deep features using pre-trained CNN models"""
    
    def __init__(self, model_type='vgg16', layer_name='block5_conv3'):
        self.model_type = model_type
        self.layer_name = layer_name
        self.logger = logging.getLogger(__name__)
        
        if not HAS_TENSORFLOW:
            self.model = None
            self.feature_model = None
            return
            
        # Load pre-trained model
        self.logger.info(f"Loading {model_type} model...")
        
        if model_type == 'vgg16':
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.preprocess = vgg_preprocess
        elif model_type == 'resnet50':
            self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.preprocess = resnet_preprocess
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create feature extraction model
        self.feature_model = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(layer_name).output
        )
        
        # Freeze the model
        self.feature_model.trainable = False
        
        self.logger.info(f"CNN feature extractor initialized with {model_type}")
    
    def extract_features(self, img):
        """Extract CNN features from an image"""
        if self.feature_model is None:
            return None
            
        # Ensure image is RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model's expected input size
        img_resized = cv2.resize(img, (224, 224))
        
        # Prepare for model
        x = np.expand_dims(img_resized, axis=0)
        x = self.preprocess(x)
        
        # Extract features
        features = self.feature_model.predict(x, verbose=0)
        
        return features[0]  # Remove batch dimension
    
    def get_activation_maps(self, img, num_filters=16):
        """Get activation maps from the CNN layer"""
        if self.feature_model is None:
            return None
            
        features = self.extract_features(img)
        if features is None:
            return None
            
        # Select top activation maps based on average activation
        avg_activations = np.mean(features, axis=(0, 1))
        top_indices = np.argsort(avg_activations)[-num_filters:]
        
        return features[:, :, top_indices]
    
    def compute_feature_vector(self, img):
        """Compute a flattened feature vector from CNN activations"""
        features = self.extract_features(img)
        if features is None:
            return {}
            
        # Global Average Pooling
        gap_features = np.mean(features, axis=(0, 1))
        
        # Global Max Pooling
        gmp_features = np.max(features, axis=(0, 1))
        
        # Spatial statistics
        spatial_mean = np.mean(features, axis=2)
        spatial_std = np.std(features, axis=2)
        
        # Create feature dictionary
        cnn_features = {}
        
        # Add pooled features
        for i, (gap, gmp) in enumerate(zip(gap_features, gmp_features)):
            cnn_features[f'cnn_gap_{i}'] = float(gap)
            cnn_features[f'cnn_gmp_{i}'] = float(gmp)
        
        # Add spatial statistics
        cnn_features['cnn_spatial_mean'] = float(np.mean(spatial_mean))
        cnn_features['cnn_spatial_std'] = float(np.mean(spatial_std))
        cnn_features['cnn_spatial_max'] = float(np.max(spatial_mean))
        cnn_features['cnn_spatial_min'] = float(np.min(spatial_mean))
        
        # Add activation statistics
        cnn_features['cnn_activation_sparsity'] = float(np.mean(features == 0))
        cnn_features['cnn_activation_energy'] = float(np.sum(features**2))
        cnn_features['cnn_activation_entropy'] = float(self._compute_activation_entropy(features))
        
        return cnn_features
    
    def _compute_activation_entropy(self, features):
        """Compute entropy of activation maps"""
        # Normalize features to [0, 1]
        features_norm = (features - features.min()) / (features.max() - features.min() + 1e-10)
        
        # Compute histogram
        hist, _ = np.histogram(features_norm.flatten(), bins=50, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        
        # Compute entropy
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def compute_cnn_similarity(self, features1, features2):
        """Compute similarity between two CNN feature maps"""
        if features1 is None or features2 is None:
            return 0.0
            
        # Ensure same shape
        if features1.shape != features2.shape:
            return 0.0
            
        # Cosine similarity in feature space
        features1_flat = features1.flatten()
        features2_flat = features2.flatten()
        
        dot_product = np.dot(features1_flat, features2_flat)
        norm1 = np.linalg.norm(features1_flat)
        norm2 = np.linalg.norm(features2_flat)
        
        cosine_sim = dot_product / (norm1 * norm2 + 1e-10)
        
        # Structural similarity of activation patterns
        # Normalize features
        f1_norm = (features1 - features1.mean()) / (features1.std() + 1e-10)
        f2_norm = (features2 - features2.mean()) / (features2.std() + 1e-10)
        
        # Compute correlation
        correlation = np.mean(f1_norm * f2_norm)
        
        # Combined similarity
        similarity = 0.7 * cosine_sim + 0.3 * correlation
        
        return float(similarity)


class HybridAnomalyDetector:
    """Combines traditional and CNN-based anomaly detection"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize CNN feature extractor if enabled
        if config.use_cnn_features and HAS_TENSORFLOW:
            self.cnn_extractor = CNNFeatureExtractor(
                model_type=config.cnn_model_type,
                layer_name=config.cnn_layer_name
            )
        else:
            self.cnn_extractor = None
    
    def extract_hybrid_features(self, image, traditional_extractor):
        """Extract both traditional and CNN features"""
        # Get traditional features
        trad_features, feature_names = traditional_extractor.extract_ultra_comprehensive_features(image)
        
        # Get CNN features if available
        if self.cnn_extractor is not None:
            cnn_features = self.cnn_extractor.compute_feature_vector(image)
            
            # Merge features
            all_features = {**trad_features, **cnn_features}
            all_feature_names = feature_names + sorted(cnn_features.keys())
        else:
            all_features = trad_features
            all_feature_names = feature_names
        
        return all_features, all_feature_names
    
    def compute_cnn_anomaly_score(self, test_image, reference_images):
        """Compute anomaly score using CNN features"""
        if self.cnn_extractor is None:
            return 0.0
            
        # Extract test features
        test_features = self.cnn_extractor.extract_features(test_image)
        if test_features is None:
            return 0.0
            
        # Compare with reference features
        similarities = []
        for ref_image in reference_images:
            ref_features = self.cnn_extractor.extract_features(ref_image)
            if ref_features is not None:
                sim = self.cnn_extractor.compute_cnn_similarity(test_features, ref_features)
                similarities.append(sim)
        
        if not similarities:
            return 0.0
            
        # Anomaly score is inverse of maximum similarity
        max_similarity = max(similarities)
        return 1.0 - max_similarity
    
    def visualize_cnn_analysis(self, image, output_path):
        """Visualize CNN activation maps and analysis"""
        if self.cnn_extractor is None:
            return
            
        activation_maps = self.cnn_extractor.get_activation_maps(image, num_filters=16)
        if activation_maps is None:
            return
            
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(16, activation_maps.shape[2])):
            ax = axes[i]
            
            # Get activation map
            act_map = activation_maps[:, :, i]
            
            # Resize to original image size
            act_map_resized = cv2.resize(act_map, (image.shape[1], image.shape[0]))
            
            # Normalize
            act_map_norm = (act_map_resized - act_map_resized.min()) / (act_map_resized.max() - act_map_resized.min() + 1e-10)
            
            # Show original image with activation overlay
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.3)
            im = ax.imshow(act_map_norm, cmap='jet', alpha=0.7)
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        plt.suptitle('CNN Activation Maps - Top 16 Filters', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"CNN activation visualization saved to {output_path}")


class OmniFiberAnalyzer:
    """Enhanced fiber optic anomaly detection with CNN integration"""
    
    def __init__(self, config: OmniConfig):
        # Store configuration object containing all analysis parameters
        self.config = config
        # Set knowledge base path, defaulting to "fiber_anomaly_kb.json" if not specified
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        # Initialize empty reference model structure for storing learned patterns
        self.reference_model = {
            'features': [],              # List of feature dictionaries from reference images
            'statistical_model': None,   # Statistical parameters (mean, std, covariance)
            'archetype_image': None,     # Median image representing typical fiber
            'feature_names': [],         # List of feature names in consistent order
            'comparison_results': {},    # Cached comparison results
            'learned_thresholds': {},    # Learned anomaly detection thresholds
            'timestamp': None,          # When model was created/updated
            'cnn_reference_features': [] # CNN features for reference images
        }
        # Initialize metadata storage for current image being processed
        self.current_metadata = None
        # Create logger instance for this class
        self.logger = logging.getLogger(__name__)
        
        # Initialize hybrid detector
        self.hybrid_detector = HybridAnomalyDetector(config)
        
        # Load existing knowledge base from disk
        self.load_knowledge_base()
    
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Enhanced analysis method with CNN integration"""
        # Log start of analysis for debugging
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        # Create Path object for easier directory manipulation
        output_path = Path(output_dir)
        # Create output directory and any missing parent directories
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if reference model exists (needed for comparison)
        if not self.reference_model.get('statistical_model'):
            # Warn that no reference exists and create minimal one
            self.logger.warning("No reference model available. Building from single image...")
            # Build minimal reference using current image as sole reference
            self._build_minimal_reference(image_path)
        
        # Run comprehensive anomaly detection analysis
        results = self.detect_anomalies_comprehensive(image_path)
        
        # Check if analysis succeeded
        if results:
            # Convert internal results format to pipeline-expected format
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            # Construct path for JSON report file
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            # Open file for writing
            with open(report_path, 'w') as f:
                # Write JSON with indentation and custom numpy encoder
                json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
            # Log successful save
            self.logger.info(f"Saved detection report to {report_path}")
            
            # Generate visualizations if enabled in config
            if self.config.enable_visualization:
                # Original visualization
                viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
                self.visualize_comprehensive_results(results, str(viz_path))
                
                # CNN activation visualization if enabled
                if self.config.use_cnn_features and self.config.enable_activation_maps:
                    cnn_viz_path = output_path / f"{Path(image_path).stem}_cnn_activations.png"
                    self.hybrid_detector.visualize_cnn_analysis(results['test_image'], str(cnn_viz_path))
                
                # Defect mask
                mask_path = output_path / f"{Path(image_path).stem}_defect_mask.npy"
                defect_mask = self._create_defect_mask(results)
                np.save(mask_path, defect_mask)
            
            # Construct path for detailed text report
            text_report_path = output_path / f"{Path(image_path).stem}_detailed.txt"
            # Generate human-readable text report
            self.generate_detailed_report(results, str(text_report_path))
            
            # Return the pipeline report
            return pipeline_report
            
        else:
            # Log analysis failure
            self.logger.error(f"Analysis failed for {image_path}")
            # Create minimal error report structure
            empty_report = {
                'image_path': image_path,
                'timestamp': self._get_timestamp(),
                'success': False,
                'error': 'Analysis failed',
                'defects': []
            }
            # Save error report
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(empty_report, f, indent=2)
            
            # Return the empty report
            return empty_report
    
    def detect_anomalies_comprehensive(self, test_path):
        """Enhanced anomaly detection with CNN features"""
        # Log start of analysis
        self.logger.info(f"Analyzing: {test_path}")
        
        # Check if reference model exists
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None
        
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        # Log loaded image metadata
        self.logger.info(f"Loaded image: {self.current_metadata}")
        
        # Convert to grayscale for analysis
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Extract hybrid features (traditional + CNN)
        self.logger.info("Extracting hybrid features from test image...")
        test_features, _ = self.hybrid_detector.extract_hybrid_features(test_image, self)
        
        # --- Global Analysis ---
        self.logger.info("Performing global anomaly analysis...")
        
        # Get reference statistics
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        # Ensure numpy arrays (in case loaded from JSON)
        for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
            if key in stat_model and isinstance(stat_model[key], list):
                stat_model[key] = np.array(stat_model[key], dtype=np.float64)
        
        # Create feature vector in consistent order
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
        
        # Compute Z-scores for each feature
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        
        # Find most deviant features
        top_indices = np.argsort(z_scores)[::-1][:10]
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # --- CNN-based Anomaly Score ---
        cnn_anomaly_score = 0.0
        if self.config.use_cnn_features and len(self.reference_model.get('reference_images', [])) > 0:
            self.logger.info("Computing CNN-based anomaly score...")
            reference_images = self.reference_model['reference_images']
            cnn_anomaly_score = self.hybrid_detector.compute_cnn_anomaly_score(test_image, reference_images)
        
        # --- Individual Comparisons ---
        self.logger.info(f"Comparing against {len(self.reference_model['features'])} reference samples...")
        
        # Compare test against each reference sample
        individual_scores = []
        for i, ref_features in enumerate(self.reference_model['features']):
            # Compute comprehensive comparison
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            
            # Compute weighted anomaly score
            score = self._compute_weighted_anomaly_score(comp)
            
            individual_scores.append(score)
        
        # Compute statistics of individual comparisons
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # --- Structural Analysis ---
        self.logger.info("Performing structural analysis...")
        
        # Get reference archetype image
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        
        # Resize test image to match archetype if needed
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        # Compute structural similarity
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # --- Local Anomaly Detection ---
        self.logger.info("Detecting local anomalies...")
        
        # Compute local anomaly map using sliding window
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        
        # Enhance anomaly map with CNN if available
        if self.config.use_cnn_features and self.cnn_extractor is not None:
            anomaly_map = self._enhance_anomaly_map_with_cnn(anomaly_map, test_image, archetype)
        
        # Find distinct anomaly regions
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # --- Specific Defect Detection ---
        self.logger.info("Detecting specific defects...")
        specific_defects = self._detect_specific_defects(test_gray)
        
        # --- Determine Overall Status ---
        thresholds = self.reference_model['learned_thresholds']
        
        # Combine traditional and CNN scores
        if self.config.use_cnn_features:
            combined_score = (1 - self.config.cnn_feature_weight) * comparison_stats['max'] + \
                           self.config.cnn_feature_weight * cnn_anomaly_score
        else:
            combined_score = comparison_stats['max']
        
        # Multiple criteria for anomaly detection
        is_anomalous = (
            mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or
            combined_score > max(thresholds['anomaly_p95'], 1e-6) or
            structural_comp['ssim'] < 0.7 or
            len(anomaly_regions) > 3 or
            any(region['confidence'] > 0.8 for region in anomaly_regions) or
            cnn_anomaly_score > 0.7
        )
        
        # Overall confidence score
        confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),
            combined_score / max(thresholds['anomaly_p95'], 1e-6),
            1 - structural_comp['ssim'],
            len(anomaly_regions) / 10,
            cnn_anomaly_score
        ))
        
        self.logger.info("Analysis Complete!")
        
        # Return comprehensive results dictionary
        return {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            
            'global_analysis': {
                'mahalanobis_distance': float(mahalanobis_dist),
                'deviant_features': deviant_features,
                'comparison_stats': comparison_stats,
                'cnn_anomaly_score': float(cnn_anomaly_score),
            },
            
            'structural_analysis': structural_comp,
            
            'local_analysis': {
                'anomaly_map': anomaly_map,
                'anomaly_regions': anomaly_regions,
            },
            
            'specific_defects': specific_defects,
            
            'verdict': {
                'is_anomalous': is_anomalous,
                'confidence': float(confidence),
                'criteria_triggered': {
                    'mahalanobis': mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6),
                    'comparison': combined_score > max(thresholds['anomaly_p95'], 1e-6),
                    'structural': structural_comp['ssim'] < 0.7,
                    'local': len(anomaly_regions) > 3,
                    'cnn': cnn_anomaly_score > 0.7 if self.config.use_cnn_features else False,
                }
            }
        }
    
    def _enhance_anomaly_map_with_cnn(self, anomaly_map, test_image, reference_image):
        """Enhance anomaly map using CNN activation differences"""
        if self.hybrid_detector.cnn_extractor is None:
            return anomaly_map
            
        # Get activation maps for both images
        test_activations = self.hybrid_detector.cnn_extractor.get_activation_maps(test_image)
        ref_activations = self.hybrid_detector.cnn_extractor.get_activation_maps(reference_image)
        
        if test_activations is None or ref_activations is None:
            return anomaly_map
            
        # Compute activation difference
        activation_diff = np.mean(np.abs(test_activations - ref_activations), axis=2)
        
        # Resize to match anomaly map
        activation_diff_resized = cv2.resize(activation_diff, 
                                           (anomaly_map.shape[1], anomaly_map.shape[0]))
        
        # Normalize
        activation_diff_norm = (activation_diff_resized - activation_diff_resized.min()) / \
                              (activation_diff_resized.max() - activation_diff_resized.min() + 1e-10)
        
        # Combine with original anomaly map
        enhanced_map = 0.7 * anomaly_map + 0.3 * activation_diff_norm
        
        return enhanced_map
    
    def _compute_weighted_anomaly_score(self, comp):
        """Compute weighted anomaly score from comparison metrics"""
        # Compute weighted anomaly score with bounds
        euclidean_term = min(comp['euclidean_distance'], 1000.0) * 0.2
        manhattan_term = min(comp['manhattan_distance'], 10000.0) * 0.1
        cosine_term = comp['cosine_distance'] * 0.2
        correlation_term = (1 - abs(comp['pearson_correlation'])) * 0.1
        kl_term = min(comp['kl_divergence'], 10.0) * 0.1
        js_term = comp['js_divergence'] * 0.1
        chi_term = min(comp['chi_square'], 10.0) * 0.1
        wasserstein_term = min(comp['wasserstein_distance'], 10.0) * 0.1
        
        # Sum weighted terms
        score = (euclidean_term + manhattan_term + cosine_term + 
                correlation_term + kl_term + js_term + 
                chi_term + wasserstein_term)
        
        # Cap the final score
        return min(score, 100.0)
    
    def build_comprehensive_reference_model(self, ref_dir):
        """Enhanced reference model building with CNN features"""
        # Log start of model building
        self.logger.info(f"Building Comprehensive Reference Model from: {ref_dir}")
        
        # Define supported file extensions
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        all_files = []
        
        # List all files in directory
        try:
            for filename in os.listdir(ref_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    all_files.append(os.path.join(ref_dir, filename))
        except Exception as e:
            self.logger.error(f"Error reading directory: {e}")
            return False
        
        # Sort files for consistent processing order
        all_files.sort()
        
        if not all_files:
            self.logger.error(f"No valid files found in {ref_dir}")
            return False
        
        self.logger.info(f"Found {len(all_files)} files to process")
        
        # Initialize storage lists
        all_features = []
        all_images = []
        reference_images = []  # Store color images for CNN
        feature_names = []
        cnn_reference_features = []
        
        # Process each file
        self.logger.info("Processing files:")
        for i, file_path in enumerate(all_files, 1):
            self.logger.info(f"[{i}/{len(all_files)}] {os.path.basename(file_path)}")
            
            # Load image
            image = self.load_image(file_path)
            if image is None:
                self.logger.warning(f"  Failed to load")
                continue
            
            # Store color image for CNN
            reference_images.append(image)
            
            # Convert to grayscale for consistent storage
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Extract hybrid features
            features, f_names = self.hybrid_detector.extract_hybrid_features(image, self)
            
            # Store feature names from first image
            if not feature_names:
                feature_names = f_names
            
            # Extract CNN features if available
            if self.hybrid_detector.cnn_extractor is not None:
                cnn_feat = self.hybrid_detector.cnn_extractor.extract_features(image)
                if cnn_feat is not None:
                    cnn_reference_features.append(cnn_feat)
            
            # Add to collections
            all_features.append(features)
            all_images.append(gray)
            
            self.logger.info(f"  Processed: {len(features)} features extracted")
        
        if not all_features:
            self.logger.error("No features could be extracted from any file")
            return False
        
        if len(all_features) < 2:
            self.logger.error(f"At least 2 reference files are required, but only {len(all_features)} were successfully processed.")
            return False
        
        self.logger.info("Building Statistical Model...")
        
        # Convert features to matrix
        feature_matrix = np.zeros((len(all_features), len(feature_names)))
        for i, features in enumerate(all_features):
            for j, fname in enumerate(feature_names):
                feature_matrix[i, j] = features.get(fname, 0)
        
        # Compute basic statistics
        mean_vector = np.mean(feature_matrix, axis=0)
        std_vector = np.std(feature_matrix, axis=0)
        median_vector = np.median(feature_matrix, axis=0)
        
        # Compute robust statistics
        self.logger.info("Computing robust statistics...")
        robust_mean, robust_cov, robust_inv_cov = self._compute_robust_statistics(feature_matrix)
        
        # Create archetype image
        self.logger.info("Creating archetype image...")
        target_shape = all_images[0].shape
        aligned_images = []
        for img in all_images:
            if img.shape != target_shape:
                img = cv2.resize(img, (target_shape[1], target_shape[0]))
            aligned_images.append(img)
        
        archetype_image = np.median(aligned_images, axis=0).astype(np.uint8)
        
        # Learn anomaly thresholds
        self.logger.info("Learning anomaly thresholds...")
        thresholds = self._learn_anomaly_thresholds(all_features, cnn_reference_features)
        
        # Store complete reference model
        self.reference_model = {
            'features': all_features,
            'feature_names': feature_names,
            'statistical_model': {
                'mean': mean_vector,
                'std': std_vector,
                'median': median_vector,
                'robust_mean': robust_mean,
                'robust_cov': robust_cov,
                'robust_inv_cov': robust_inv_cov,
                'n_samples': len(all_features),
            },
            'archetype_image': archetype_image,
            'reference_images': reference_images[:5],  # Store up to 5 reference images
            'cnn_reference_features': cnn_reference_features[:5],  # Store CNN features
            'learned_thresholds': thresholds,
            'timestamp': self._get_timestamp(),
        }
        
        # Save model to disk
        self.save_knowledge_base()
        
        # Log success summary
        self.logger.info("Reference Model Built Successfully!")
        self.logger.info(f"  - Samples: {len(all_features)}")
        self.logger.info(f"  - Features: {len(feature_names)}")
        self.logger.info(f"  - Anomaly threshold: {thresholds['anomaly_threshold']:.4f}")
        if cnn_reference_features:
            self.logger.info(f"  - CNN features extracted: {len(cnn_reference_features)}")
        
        return True
    
    def _learn_anomaly_thresholds(self, all_features, cnn_features):
        """Learn anomaly thresholds from pairwise comparisons"""
        self.logger.info("Computing pairwise comparisons for threshold learning...")
        
        n_comparisons = len(all_features) * (len(all_features) - 1) // 2
        self.logger.info(f"Total comparisons to compute: {n_comparisons}")
        
        comparison_scores = []
        cnn_scores = []
        comparison_count = 0
        
        # Compare all pairs of reference samples
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                # Traditional comparison
                comp = self.compute_exhaustive_comparison(all_features[i], all_features[j])
                score = self._compute_weighted_anomaly_score(comp)
                comparison_scores.append(score)
                
                # CNN comparison if available
                if i < len(cnn_features) and j < len(cnn_features):
                    cnn_sim = self.hybrid_detector.cnn_extractor.compute_cnn_similarity(
                        cnn_features[i], cnn_features[j]
                    )
                    cnn_scores.append(1 - cnn_sim)  # Convert similarity to distance
                
                comparison_count += 1
                
                if comparison_count % 100 == 0:
                    self.logger.info(f"  Progress: {comparison_count}/{n_comparisons} ({comparison_count/n_comparisons*100:.1f}%)")
        
        # Learn thresholds from scores
        scores_array = np.array(comparison_scores)
        
        if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
            valid_scores = scores_array[~np.isnan(scores_array)]
            valid_scores = valid_scores[np.isfinite(valid_scores)]
            
            if len(valid_scores) > 0:
                valid_scores = np.clip(valid_scores, 0, np.percentile(valid_scores, 99.9))
                
                mean_score = float(np.mean(valid_scores))
                std_score = float(np.std(valid_scores))
                
                thresholds = {
                    'anomaly_mean': mean_score,
                    'anomaly_std': std_score,
                    'anomaly_p90': float(np.percentile(valid_scores, 90)),
                    'anomaly_p95': float(np.percentile(valid_scores, 95)),
                    'anomaly_p99': float(np.percentile(valid_scores, 99)),
                    'anomaly_threshold': float(min(mean_score + self.config.anomaly_threshold_multiplier * std_score,
                                                   np.percentile(valid_scores, 99.5),
                                                   10.0)),
                }
                
                # Add CNN thresholds if available
                if cnn_scores:
                    cnn_array = np.array(cnn_scores)
                    thresholds['cnn_mean'] = float(np.mean(cnn_array))
                    thresholds['cnn_std'] = float(np.std(cnn_array))
                    thresholds['cnn_threshold'] = float(np.percentile(cnn_array, 95))
            else:
                thresholds = self._get_default_thresholds()
        else:
            thresholds = self._get_default_thresholds()
        
        return thresholds
    
    # Include all the original methods from the base class here...
    # (I'll include key methods and you can add the rest from your original script)
    
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
    
    # Add remaining methods from original script...
    # (Due to space constraints, I'm showing the structure. You would include all methods from the original script)
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        return time.strftime("%Y-%m-%d_%H:%M:%S")
    
    # Include all statistical functions, feature extraction methods, comparison methods, etc.
    # from your original script here...


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


def main():
    """Main execution function for standalone testing."""
    print("\n" + "="*80)
    print("OMNIFIBER ANALYZER - ENHANCED DETECTION MODULE WITH CNN (v2.0)".center(80))
    print("="*80)
    print("\nThis enhanced module now includes CNN-based feature extraction!")
    print("For standalone testing, you can analyze individual images.\n")
    
    # Check TensorFlow availability
    if HAS_TENSORFLOW:
        print("✓ TensorFlow is available - CNN features enabled")
        print(f"  TensorFlow version: {tf.__version__}")
    else:
        print("✗ TensorFlow not found - CNN features disabled")
        print("  Install with: pip install tensorflow")
    print()
    
    # Create default configuration
    config = OmniConfig()
    
    # Initialize analyzer with configuration
    analyzer = OmniFiberAnalyzer(config)
    
    # Interactive testing loop
    while True:
        test_path = input("\nEnter path to test image (or 'quit' to exit): ").strip()
        test_path = test_path.strip('"\'')
        
        if test_path.lower() == 'quit':
            break
            
        if not os.path.isfile(test_path):
            print(f"✗ File not found: {test_path}")
            continue
            
        # Create unique output directory with timestamp
        output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Run analysis
        print(f"\nAnalyzing {test_path}...")
        if config.use_cnn_features and HAS_TENSORFLOW:
            print("  Using hybrid approach (Traditional + CNN features)")
        else:
            print("  Using traditional features only")
            
        analyzer.analyze_end_face(test_path, output_dir)
        
        # Report output files
        print(f"\nResults saved to: {output_dir}/")
        print("  - JSON report: *_report.json")
        print("  - Visualization: *_analysis.png")
        if config.use_cnn_features and config.enable_activation_maps:
            print("  - CNN activations: *_cnn_activations.png")
        print("  - Detailed text: *_detailed.txt")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
