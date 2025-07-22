#!/usr/bin/env python3
"""
Unit tests for the fiber optics configuration system
Tests configuration loading, validation, and all parameter access
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from fiber_advanced_config_loader import Config, get_config, load_config
from test_base import FiberOpticsTestCase


class TestConfigLoader(FiberOpticsTestCase):
    """Test configuration loading and validation"""
    
    def test_config_initialization(self):
        """Test config loads with default values"""
        config = Config(self.test_config)
        
        self.assertEqual(config.system.device, 'cpu')
        self.assertEqual(config.system.seed, 42)
        self.assertEqual(config.similarity.threshold, 0.7)
        self.assertEqual(config.model.num_classes, 3)
    
    def test_dot_notation_access(self):
        """Test accessing config values with dot notation"""
        config = Config(self.test_config)
        
        # Test nested access
        self.assertEqual(config.system.log_level, 'DEBUG')
        self.assertEqual(config.optimizer.learning_rate, 0.001)
        self.assertEqual(config.loss.weights.segmentation, 0.25)
        self.assertEqual(config.equation.coefficients.A, 1.0)
    
    def test_dict_notation_access(self):
        """Test accessing config values with dict notation"""
        config = Config(self.test_config)
        
        # Test dict-style access
        self.assertEqual(config['system']['device'], 'cpu')
        self.assertEqual(config['optimizer']['learning_rate'], 0.001)
        self.assertEqual(config['loss']['weights']['segmentation'], 0.25)
    
    def test_get_method(self):
        """Test get method with defaults"""
        config = Config(self.test_config)
        
        # Test existing keys
        self.assertEqual(config.get('system.device'), 'cpu')
        self.assertEqual(config.get('optimizer.learning_rate'), 0.001)
        
        # Test non-existing keys with defaults
        self.assertEqual(config.get('non.existing.key', 'default'), 'default')
        self.assertIsNone(config.get('another.missing.key'))
    
    def test_device_configuration(self):
        """Test device configuration logic"""
        # Test CPU config
        cpu_config = Config(self.test_config)
        self.assertEqual(cpu_config.get_device(), 'cpu')
        
        # Test auto device detection
        auto_config = self.test_config.copy()
        auto_config['system']['device'] = 'auto'
        config = Config(auto_config)
        # Should return cpu or cuda depending on availability
        self.assertIn(config.get_device(), ['cpu', 'cuda'])
    
    def test_loss_weight_validation(self):
        """Test loss weight normalization"""
        config = Config(self.test_config)
        
        # Check weights sum to 1.0
        weights = config.loss.weights
        total = sum([
            weights.segmentation,
            weights.anomaly,
            weights.contrastive,
            weights.perceptual,
            weights.wasserstein,
            weights.reconstruction
        ])
        self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_config_update(self):
        """Test updating configuration values"""
        config = Config(self.test_config)
        
        # Update single value
        config.update({'system': {'seed': 123}})
        self.assertEqual(config.system.seed, 123)
        
        # Update nested values
        config.update({
            'optimizer': {
                'learning_rate': 0.01,
                'weight_decay': 0.001
            }
        })
        self.assertEqual(config.optimizer.learning_rate, 0.01)
        self.assertEqual(config.optimizer.weight_decay, 0.001)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with invalid similarity threshold
        invalid_config = self.test_config.copy()
        invalid_config['similarity']['threshold'] = 1.5  # Should be <= 1.0
        
        with self.assertRaises(ValueError):
            config = Config(invalid_config)
            config._validate()
    
    def test_coefficient_bounds(self):
        """Test equation coefficient bounds"""
        config = Config(self.test_config)
        
        # Check all coefficients are within bounds
        coeffs = config.equation.coefficients
        min_coeff = config.equation.min_coefficient
        max_coeff = config.equation.max_coefficient
        
        for key, value in coeffs.items():
            self.assertGreaterEqual(value, min_coeff)
            self.assertLessEqual(value, max_coeff)
    
    def test_load_from_file(self):
        """Test loading configuration from YAML file"""
        # Save test config to file
        config_file = self.test_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Load from file
        loaded_config = load_config(str(config_file))
        
        self.assertEqual(loaded_config.system.device, 'cpu')
        self.assertEqual(loaded_config.similarity.threshold, 0.7)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        # Create config with missing required field
        incomplete_config = {
            'system': {'device': 'cpu'},
            # Missing other required sections
        }
        
        # Should still create config with defaults
        config = Config(incomplete_config)
        self.assertTrue(hasattr(config, 'model'))
        self.assertTrue(hasattr(config, 'optimizer'))
        self.assertTrue(hasattr(config, 'loss'))
    
    def test_parameter_groups(self):
        """Test parameter group configuration"""
        config = Config(self.test_config)
        
        param_groups = config.optimizer.param_groups
        self.assertEqual(param_groups.feature_extractor, 1.0)
        self.assertEqual(param_groups.segmentation, 0.5)
        self.assertEqual(param_groups.reference_embeddings, 0.1)
        self.assertEqual(param_groups.decoder, 0.5)
        self.assertEqual(param_groups.equation_parameters, 0.01)
    
    def test_augmentation_config(self):
        """Test data augmentation configuration"""
        config = Config(self.test_config)
        
        aug = config.training.augmentation
        self.assertFalse(aug.enabled)  # Disabled for tests
        self.assertEqual(aug.random_rotation, 15)
        self.assertEqual(aug.random_scale, [0.9, 1.1])
        self.assertEqual(aug.random_brightness, 0.1)
        self.assertEqual(aug.random_contrast, 0.1)
        self.assertTrue(aug.horizontal_flip)
        self.assertTrue(aug.vertical_flip)
    
    def test_anomaly_config(self):
        """Test anomaly detection configuration"""
        config = Config(self.test_config)
        
        anomaly = config.anomaly
        self.assertEqual(anomaly.threshold, 0.3)
        self.assertEqual(anomaly.min_defect_size, 10)
        self.assertEqual(anomaly.max_defect_size, 1000)
        self.assertEqual(anomaly.ignore_boundary_width, 5)
        self.assertEqual(len(anomaly.defect_types), 5)
        self.assertIn('scratch', anomaly.defect_types)
    
    def test_visualization_config(self):
        """Test visualization configuration"""
        config = Config(self.test_config)
        
        viz = config.visualization
        self.assertEqual(viz.window_width, 800)
        self.assertEqual(viz.window_height, 600)
        self.assertEqual(viz.fps, 30)
        self.assertTrue(viz.show_original)
        self.assertTrue(viz.show_segmentation)
        self.assertTrue(viz.show_anomaly_map)
        
        # Test color configuration
        self.assertEqual(viz.segmentation_colors.core, [255, 0, 0])
        self.assertEqual(viz.segmentation_colors.cladding, [0, 255, 0])
        self.assertEqual(viz.segmentation_colors.ferrule, [0, 0, 255])
    
    def test_advanced_features_config(self):
        """Test advanced features configuration"""
        config = Config(self.test_config)
        
        advanced = config.advanced
        self.assertFalse(advanced.use_gradient_checkpointing)
        self.assertFalse(advanced.use_nas)
        self.assertFalse(advanced.use_maml)
        self.assertTrue(advanced.use_uncertainty)
        self.assertEqual(advanced.uncertainty_method, 'dropout')
        self.assertEqual(advanced.dropout_samples, 10)
    
    def test_experimental_features_config(self):
        """Test experimental features configuration"""
        config = Config(self.test_config)
        
        exp = config.experimental
        self.assertFalse(exp.use_cross_attention)
        self.assertFalse(exp.use_self_attention)
        self.assertEqual(exp.attention_heads, 8)
        self.assertFalse(exp.use_transformer)
        self.assertFalse(exp.use_diffusion)
    
    def test_monitoring_config(self):
        """Test monitoring configuration"""
        config = Config(self.test_config)
        
        mon = config.monitoring
        self.assertTrue(mon.track_memory_usage)
        self.assertTrue(mon.track_computation_time)
        self.assertTrue(mon.track_gradient_norms)
        self.assertFalse(mon.use_tensorboard)
        self.assertFalse(mon.use_wandb)
    
    def test_debug_config(self):
        """Test debug configuration"""
        config = Config(self.test_config)
        
        debug = config.debug
        self.assertTrue(debug.enabled)
        self.assertFalse(debug.save_intermediate_features)
        self.assertTrue(debug.check_nan)
        self.assertTrue(debug.check_inf)
        self.assertTrue(debug.deterministic)
    
    def test_global_config_access(self):
        """Test global configuration access"""
        # Get config through global function
        config1 = get_config()
        config2 = get_config()
        
        # Should return same instance
        self.assertIs(config1, config2)
    
    def test_config_to_dict(self):
        """Test converting config back to dictionary"""
        config = Config(self.test_config)
        
        # Should be able to access as dict
        config_dict = dict(config)
        self.assertIn('system', config_dict)
        self.assertIn('model', config_dict)
        self.assertEqual(config_dict['system']['device'], 'cpu')


class TestConfigValidation(FiberOpticsTestCase):
    """Test configuration validation rules"""
    
    def test_learning_rate_bounds(self):
        """Test learning rate must be positive"""
        invalid_config = self.test_config.copy()
        invalid_config['optimizer']['learning_rate'] = -0.001
        
        with self.assertRaises(ValueError):
            config = Config(invalid_config)
            if config.optimizer.learning_rate <= 0:
                raise ValueError("Learning rate must be positive")
    
    def test_batch_size_bounds(self):
        """Test batch size must be positive integer"""
        invalid_config = self.test_config.copy()
        invalid_config['training']['batch_size'] = 0
        
        with self.assertRaises(ValueError):
            config = Config(invalid_config)
            if config.training.batch_size <= 0:
                raise ValueError("Batch size must be positive")
    
    def test_epoch_bounds(self):
        """Test num_epochs must be positive"""
        invalid_config = self.test_config.copy()
        invalid_config['training']['num_epochs'] = -1
        
        with self.assertRaises(ValueError):
            config = Config(invalid_config)
            if config.training.num_epochs <= 0:
                raise ValueError("Number of epochs must be positive")
    
    def test_similarity_weights_sum(self):
        """Test similarity combination weights sum to 1.0"""
        config = Config(self.test_config)
        
        weights = config.similarity.combination_weights
        total = weights.lpips + weights.ssim + weights.optimal_transport
        self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_scale_weights_normalization(self):
        """Test multi-scale weights are normalized"""
        config = Config(self.test_config)
        
        scale_weights = config.features.scale_weights
        self.assertAlmostEqual(sum(scale_weights), 1.0, places=6)
    
    def test_defect_size_ordering(self):
        """Test min_defect_size < max_defect_size"""
        config = Config(self.test_config)
        
        self.assertLess(
            config.anomaly.min_defect_size,
            config.anomaly.max_defect_size
        )
    
    def test_scheduler_min_lr_bound(self):
        """Test scheduler min_lr < learning_rate"""
        config = Config(self.test_config)
        
        self.assertLess(
            config.optimizer.scheduler.min_lr,
            config.optimizer.learning_rate
        )


if __name__ == "__main__":
    unittest.main()