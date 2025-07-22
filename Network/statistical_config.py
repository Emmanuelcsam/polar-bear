#!/usr/bin/env python3
"""
Statistical Configuration for Fiber Optics Neural Network
Configuration settings for statistical integration components
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class StatisticalConfig:
    """
    Configuration for statistical integration components
    Based on findings from comprehensive statistical analysis
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing StatisticalConfig")
        
        # Global dataset statistics from global_statistics.json
        self.dataset_statistics = {
            'total_images': 65606,
            'num_classes': 40,
            'imbalance_ratio': 34571.0,
            'samples_per_class_mean': 1640.15,
            'samples_per_class_std': 5448.92,
            'dominant_class': 'dirty-image',
            'dominant_class_samples': 34571
        }
        
        # Statistical feature extraction settings
        self.statistical_features = {
            'enabled': True,
            'feature_dim': 88,  # 88-dimensional feature vector
            'extract_glcm': True,
            'glcm_distances': [1, 2, 3],
            'glcm_angles': [0, 45, 90, 135],
            'extract_lbp': True,
            'lbp_radii': [1, 2, 3, 5],
            'extract_morphological': True,
            'morph_kernel_sizes': [3, 5, 7, 11],
            'extract_statistical': True,
            'extract_topological': True,
            'use_pca': True,
            'pca_components': 12,  # 95% variance explained
            # Advanced distribution metrics from advanced-separation-statistics.json
            'extract_distribution_metrics': True,
            'distribution_metrics': ['skewness', 'kurtosis', 'jarque_bera', 'sem', 'mad', 'trimmed_mean', 'gini', 'cv'],
            'pc1_variance_explained': 0.453,  # Overall image degradation
            'pc2_variance_explained': 0.181   # Fine-grained textural anomalies
        }
        
        # Master similarity equation settings
        self.similarity_settings = {
            'enabled': True,
            'learnable_weights': True,
            'initial_weights': {
                'center_x': 0.362261,
                'center_y': 0.202874,
                'core_radius': 0.164540,
                'cladding_radius': 0.270316,
                'core_cladding_ratio': 0.000001,
                'num_valid_results': 0.000008
            },
            'similarity_threshold': 0.7  # From goal.txt
        }
        
        # Zone parameter prediction settings
        self.zone_prediction = {
            'enabled': True,
            'use_regression_models': True,
            'learnable_refinement': True,
            'core_radius_range': [5.0, 500.0],
            'cladding_radius_range': [14.0, 800.0],
            'ratio_range': [0.05, 1.0],
            'physical_constraints': True,
            # Linear regression models from statistics
            'core_radius_model': {
                'intercept': 181.278193,
                'coefficients': {
                    'accuracy_adaptive_intensity': -25.380857,
                    'accuracy_computational_separation': 77.136231,
                    'accuracy_geometric_approach': -11.234236,
                    'accuracy_guess_approach': 16.665568,
                    'accuracy_hough_separation': -129.820486,
                    'accuracy_threshold_separation': 10.694743,
                    'accuracy_unified_core_cladding_detector': -39.121880
                },
                'r_squared': 0.5745
            },
            'cladding_radius_model': {
                'intercept': 313.894572,
                'coefficients': {
                    'accuracy_adaptive_intensity': -5.826730,
                    'accuracy_computational_separation': 102.567323,
                    'accuracy_geometric_approach': -70.932718,
                    'accuracy_guess_approach': -14.220535,
                    'accuracy_hough_separation': -75.585671,
                    'accuracy_threshold_separation': -5.723188,
                    'accuracy_unified_core_cladding_detector': -70.009830
                },
                'r_squared': 0.4290
            },
            'core_cladding_ratio_model': {
                'intercept': 0.600998,
                'coefficients': {
                    'accuracy_adaptive_intensity': -0.051924,
                    'accuracy_computational_separation': 0.079397,
                    'accuracy_geometric_approach': 0.128683,
                    'accuracy_guess_approach': 0.026918,
                    'accuracy_hough_separation': -0.383285,
                    'accuracy_threshold_separation': 0.042756,
                    'accuracy_unified_core_cladding_detector': 0.032219
                },
                'r_squared': 0.5231
            }
        }
        
        # Consensus module settings
        self.consensus_settings = {
            'enabled': True,
            'num_methods': 7,
            'min_agreement_ratio': 0.3,
            'iou_threshold': 0.6,
            'circularity_threshold': 0.85,
            'use_weighted_voting': True,
            'update_method_scores': True,
            'learning_rate': 0.1  # For exponential moving average
        }
        
        # Anomaly detection settings
        self.anomaly_detection = {
            'enabled': True,
            'use_mahalanobis': True,
            'threshold_multiplier': 2.5,
            'min_defect_size': 10,
            'max_defect_size': 5000,
            'detect_specific_defects': True,
            'defect_types': ['scratch', 'dig', 'blob'],
            'severity_thresholds': {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            },
            # Anomaly prediction models from statistics
            'confidence_model': {
                'intercept': 110.5,
                'coefficients': {
                    'SSIM_Index': -25.0,
                    'Mahalanobis_Distance': 15.2,
                    'Total_Scratches': -0.002,
                    'Total_Anomaly_Regions': 0.05
                },
                'r_squared': 0.88
            },
            'ssim_scratch_model': {
                'intercept': 0.92,
                'coefficient': -0.00015,  # SSIM = -0.00015 * Total_Scratches + 0.92
                'r_squared': 0.72
            }
        }
        
        # Correlation-guided attention settings
        self.correlation_attention = {
            'enabled': True,
            'num_features': 19,  # Core statistical features
            'use_learned_correlations': True,
            'temperature': 1.0,
            'key_correlations': [
                ('center_x', 'center_y', 0.9088),
                ('core_radius', 'cladding_radius', 0.7964),
                ('core_radius', 'core_cladding_ratio', 0.6972),
                ('core_radius', 'accuracy_hough', -0.6945),
                # Additional strong correlations from analysis
                ('center_x', 'center_distance_from_origin', 0.988),
                ('center_y', 'center_distance_from_origin', 0.962),
                ('core_radius', 'core_area', 0.933),
                ('cladding_radius', 'cladding_area', 0.958),
                ('cladding_area', 'cladding_core_area_diff', 0.887),
                ('Total_Scratches', 'SSIM_Index', -0.85)
            ]
        }
        
        # Loss function settings
        self.loss_settings = {
            'loss_type': 'composite',  # 'composite', 'iou', 'similarity', etc.
            'loss_weights': {
                'segmentation': 1.0,
                'iou': 0.5,
                'circularity': 0.1,
                'similarity': 0.3,
                'correlation': 0.1,
                'anomaly': 0.5,
                'zone_regression': 0.3,
                'method_accuracy': 0.2,
                'consensus': 0.2,
                'defect_specific': 0.3
            },
            'use_focal_loss': True,
            'focal_gamma': 2.0,
            'focal_alpha': 0.25,
            # Class imbalance handling from global_statistics.json
            'use_class_weights': True,
            'class_imbalance_ratio': 34571.0,
            'num_classes': 40,
            'balance_strategy': 'effective_number',  # or 'inverse_frequency'
            'beta': 0.99  # for effective number of samples
        }
        
        # Training settings specific to statistical components
        self.training_settings = {
            'pretrain_statistical': True,  # Pretrain statistical components
            'pretrain_epochs': 10,
            'freeze_base_network': False,
            'statistical_lr_multiplier': 2.0,  # Higher LR for statistical components
            'update_reference_statistics': True,
            'reference_update_frequency': 100,  # Update every N batches
            'use_knowledge_distillation': True,
            'distillation_temperature': 3.0,
            'distillation_alpha': 0.7,
            # From neural_network_config.json
            'batch_size': 128,
            'initial_lr': 0.001,
            'scheduler': 'cosine_annealing',
            'min_lr': 1e-5,
            'epochs': 50,
            'early_stopping_patience': 10,
            'optimizer': 'AdamW',
            'use_mixup': True,
            'mixup_alpha': 0.1
        }
        
        # Method-specific settings (from separation.py analysis)
        self.method_settings = {
            'adaptive_intensity': {'initial_score': 0.7, 'vulnerable': True, 'avg_accuracy': 0.2325, 'std': 0.2724},
            'computational_separation': {'initial_score': 0.8, 'vulnerable': False, 'avg_accuracy': 0.2687, 'std': 0.3134},
            'geometric_approach': {'initial_score': 0.9, 'vulnerable': False, 'avg_accuracy': 0.7485, 'std': 0.3222},
            'guess_approach': {'initial_score': 0.6, 'vulnerable': True, 'avg_accuracy': 0.6086, 'std': 0.3289},
            'hough_separation': {'initial_score': 0.5, 'vulnerable': False, 'avg_accuracy': 0.5812, 'std': 0.3557},
            'threshold_separation': {'initial_score': 0.4, 'vulnerable': True, 'avg_accuracy': 0.4539, 'std': 0.4094},
            'unified_core_cladding_detector': {'initial_score': 0.85, 'vulnerable': False, 'avg_accuracy': 0.3277, 'std': 0.1911}
        }
        
        # Data augmentation settings for statistical robustness
        self.augmentation_settings = {
            'statistical_augmentation': True,
            'add_synthetic_defects': True,
            'defect_probability': 0.3,
            'synthetic_scratch_params': {
                'min_length': 20,
                'max_length': 100,
                'min_width': 1,
                'max_width': 3
            },
            'synthetic_dig_params': {
                'min_radius': 3,
                'max_radius': 15,
                'intensity_range': [0.3, 0.7]
            },
            'synthetic_blob_params': {
                'min_area': 50,
                'max_area': 500,
                'irregularity': 0.3
            },
            # Advanced augmentation from neural_network_config.json
            'normalization': 'imagenet_stats',
            'resize_strategy': 'resize_then_crop',
            'basic_augmentations': ['horizontal_flip', 'vertical_flip', 'rotation_15', 'brightness_contrast'],
            'advanced_augmentations': ['elastic_transform', 'grid_distortion', 'optical_distortion'],
            'color_augmentations': ['hue_saturation', 'rgb_shift', 'channel_shuffle'],
            'noise_augmentations': ['gaussian_noise', 'blur', 'jpeg_compression'],
            'use_randaugment': True
        }
        
        # Evaluation metrics settings
        self.evaluation_settings = {
            'metrics': [
                'iou',
                'dice',
                'accuracy',
                'precision',
                'recall',
                'f1',
                'mahalanobis_distance',
                'circularity',
                'method_agreement',
                'anomaly_detection_rate'
            ],
            'save_visualizations': True,
            'visualization_frequency': 50,  # Every N batches
            'log_statistical_features': True,
            'track_method_performance': True
        }
        
        # Integration settings
        self.integration_settings = {
            'integration_mode': 'full',  # 'full', 'partial', 'ablation'
            'use_statistical_features': True,
            'use_similarity_matching': True,
            'use_zone_prediction': True,
            'use_consensus': True,
            'use_anomaly_detection': True,
            'use_correlation_attention': True,
            'statistical_feature_weight': 0.3  # Weight for statistical features in fusion
        }
        
        # Reference model settings
        self.reference_settings = {
            'num_reference_embeddings': 100,
            'embedding_dim': 256,
            'update_embeddings': True,
            'embedding_momentum': 0.99,
            'use_archetype_images': True,
            'archetype_update_frequency': 1000
        }
        
        # Hardware optimization settings
        self.optimization_settings = {
            'use_mixed_precision': True,
            'gradient_checkpointing': True,
            'statistical_component_device': 'cuda:0',  # Can be different from main network
            'parallel_feature_extraction': True,
            'cache_statistical_features': True,
            'cache_size': 1000
        }
        
        # Network architecture recommendations from neural_network_config.json
        self.architecture_recommendations = {
            'use_vision_transformer': True,  # Recommended in statistics
            'vit_config': {
                'patch_size': 16,
                'embed_dim': 768
            },
            'cnn_architecture': {
                'type': 'ConvNet',
                'layers': [
                    {'conv': 64, 'kernel': 3, 'pool': 2},
                    {'conv': 128, 'kernel': 3, 'pool': 2},
                    {'conv': 256, 'kernel': 3, 'pool': 2},
                    {'conv': 512, 'kernel': 3, 'pool': 2},
                    {'fc': [1024, 512, 40]}
                ],
                'dropout_rates': [0.2, 0.3, 0.4]
            },
            'resnet_config': {
                'variant': 'ResNet34',
                'pretrained': False
            }
        }
        
        # Ensemble settings from statistics
        self.ensemble_settings = {
            'use_ensemble': True,
            'ensemble_size': 5,
            'strategies': [
                'different_architectures',
                'different_initializations',
                'cross_validation_folds'
            ],
            'voting_strategy': 'weighted',  # weighted by individual model performance
            'ensemble_learning_rate': 0.001
        }
        
        # Logging and debugging
        self.debug_settings = {
            'log_level': 'INFO',
            'save_intermediate_features': False,
            'visualize_correlations': True,
            'plot_method_scores': True,
            'save_consensus_masks': True,
            'track_feature_statistics': True
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'dataset_statistics': self.dataset_statistics,
            'statistical_features': self.statistical_features,
            'similarity_settings': self.similarity_settings,
            'zone_prediction': self.zone_prediction,
            'consensus_settings': self.consensus_settings,
            'anomaly_detection': self.anomaly_detection,
            'correlation_attention': self.correlation_attention,
            'loss_settings': self.loss_settings,
            'training_settings': self.training_settings,
            'method_settings': self.method_settings,
            'augmentation_settings': self.augmentation_settings,
            'evaluation_settings': self.evaluation_settings,
            'integration_settings': self.integration_settings,
            'reference_settings': self.reference_settings,
            'optimization_settings': self.optimization_settings,
            'architecture_recommendations': self.architecture_recommendations,
            'ensemble_settings': self.ensemble_settings,
            'debug_settings': self.debug_settings
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_enabled_components(self) -> List[str]:
        """Get list of enabled statistical components"""
        enabled = []
        
        if self.statistical_features.get('enabled', False):
            enabled.append('statistical_features')
        if self.similarity_settings.get('enabled', False):
            enabled.append('similarity_matching')
        if self.zone_prediction.get('enabled', False):
            enabled.append('zone_prediction')
        if self.consensus_settings.get('enabled', False):
            enabled.append('consensus')
        if self.anomaly_detection.get('enabled', False):
            enabled.append('anomaly_detection')
        if self.correlation_attention.get('enabled', False):
            enabled.append('correlation_attention')
            
        return enabled
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Check feature dimensions
        if self.statistical_features['feature_dim'] != 88:
            print(f"Warning: Expected 88 features, got {self.statistical_features['feature_dim']}")
        
        # Check similarity threshold
        if self.similarity_settings['similarity_threshold'] < 0.7:
            print(f"Warning: Similarity threshold {self.similarity_settings['similarity_threshold']} < 0.7 (goal.txt requirement)")
        
        # Check method count
        if self.consensus_settings['num_methods'] != 7:
            print(f"Warning: Expected 7 methods, got {self.consensus_settings['num_methods']}")
        
        # Check loss weights sum
        total_weight = sum(self.loss_settings['loss_weights'].values())
        if abs(total_weight - 5.5) > 0.1:  # Expected sum based on default weights
            print(f"Info: Total loss weight is {total_weight}")
        
        return True
    
    def print_summary(self):
        """Print configuration summary"""
        print(f"\n{'='*60}")
        print("Statistical Configuration Summary")
        print(f"{'='*60}")
        
        print(f"\nEnabled Components: {', '.join(self.get_enabled_components())}")
        
        print(f"\nFeature Extraction:")
        print(f"  - Feature dimension: {self.statistical_features['feature_dim']}")
        print(f"  - GLCM extraction: {self.statistical_features['extract_glcm']}")
        print(f"  - LBP extraction: {self.statistical_features['extract_lbp']}")
        print(f"  - PCA components: {self.statistical_features['pca_components']}")
        
        print(f"\nSimilarity Settings:")
        print(f"  - Threshold: {self.similarity_settings['similarity_threshold']}")
        print(f"  - Learnable weights: {self.similarity_settings['learnable_weights']}")
        
        print(f"\nConsensus Settings:")
        print(f"  - Number of methods: {self.consensus_settings['num_methods']}")
        print(f"  - IoU threshold: {self.consensus_settings['iou_threshold']}")
        print(f"  - Circularity threshold: {self.consensus_settings['circularity_threshold']}")
        
        print(f"\nAnomaly Detection:")
        print(f"  - Mahalanobis threshold: {self.anomaly_detection['threshold_multiplier']}")
        print(f"  - Defect types: {', '.join(self.anomaly_detection['defect_types'])}")
        
        print(f"\nLoss Settings:")
        print(f"  - Loss type: {self.loss_settings['loss_type']}")
        print(f"  - Main weights: seg={self.loss_settings['loss_weights']['segmentation']}, "
              f"iou={self.loss_settings['loss_weights']['iou']}, "
              f"anomaly={self.loss_settings['loss_weights']['anomaly']}")
        
        print(f"{'='*60}\n")


# Create default configuration instance
def get_statistical_config() -> StatisticalConfig:
    """Get default statistical configuration"""
    return StatisticalConfig()


# Function to merge with existing configuration
def merge_with_base_config(base_config: Dict[str, Any], 
                          statistical_config: Optional[StatisticalConfig] = None) -> Dict[str, Any]:
    """
    Merge statistical configuration with base configuration
    
    Args:
        base_config: Existing configuration dictionary
        statistical_config: Statistical configuration (uses default if None)
        
    Returns:
        Merged configuration dictionary
    """
    if statistical_config is None:
        statistical_config = get_statistical_config()
    
    # Convert statistical config to dict
    stat_dict = statistical_config.to_dict()
    
    # Create merged config
    merged_config = base_config.copy()
    
    # Add statistical sections
    merged_config['statistical'] = stat_dict
    
    # Update relevant base config sections
    if 'model' in merged_config:
        merged_config['model']['use_statistical_features'] = True
        merged_config['model']['statistical_feature_dim'] = 88
    
    if 'training' in merged_config:
        merged_config['training']['use_statistical_loss'] = True
        merged_config['training']['statistical_pretrain_epochs'] = statistical_config.training_settings['pretrain_epochs']
    
    if 'loss' not in merged_config:
        merged_config['loss'] = {}
    merged_config['loss'].update(statistical_config.loss_settings)
    
    return merged_config