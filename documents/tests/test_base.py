#!/usr/bin/env python3
"""
Base test utilities and fixtures for Fiber Optics Neural Network tests
Provides common testing infrastructure and mock data generators
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from typing import Dict, Any, Tuple
import json
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))


class FiberOpticsTestCase(unittest.TestCase):
    """Base test case with common utilities"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.test_data_dir = cls.test_dir / "test_data"
        cls.test_logs_dir = cls.test_dir / "logs"
        cls.test_results_dir = cls.test_dir / "results"
        
        # Create directories
        cls.test_data_dir.mkdir(parents=True)
        cls.test_logs_dir.mkdir(parents=True)
        cls.test_results_dir.mkdir(parents=True)
        
        # Create test configuration
        cls.test_config = cls.create_test_config()
        cls.test_config_path = cls.test_dir / "test_config.yaml"
        with open(cls.test_config_path, 'w') as f:
            yaml.dump(cls.test_config, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_test_config(cls) -> Dict[str, Any]:
        """Create minimal test configuration"""
        return {
            'system': {
                'device': 'cpu',
                'gpu_id': 0,
                'num_workers': 1,
                'seed': 42,
                'log_level': 'DEBUG',
                'log_to_file': True,
                'log_file_path': str(cls.test_logs_dir / 'test.log'),
                'logs_path': str(cls.test_logs_dir),
                'verbose_logging': True,
                'data_path': str(cls.test_data_dir),
                'tensorized_data_path': str(cls.test_data_dir / 'tensorized'),
                'reference_data_path': str(cls.test_data_dir / 'references'),
                'checkpoints_path': str(cls.test_dir / 'checkpoints'),
                'results_path': str(cls.test_results_dir)
            },
            'model': {
                'architecture': 'basic',
                'input_channels': 3,
                'base_channels': 16,
                'num_blocks': [1, 1, 1, 1],
                'use_se_blocks': False,
                'se_reduction': 16,
                'use_deformable_conv': False,
                'use_cbam': False,
                'use_efficient_channel_attention': False,
                'use_adaptive_computation': False,
                'adaptive_threshold': 0.95,
                'max_computation_steps': 10,
                'num_classes': 3,
                'num_reference_embeddings': 100,
                'embedding_dim': 64
            },
            'equation': {
                'coefficients': {
                    'A': 1.0,
                    'B': 1.0,
                    'C': 1.0,
                    'D': 1.0,
                    'E': 1.0
                },
                'min_coefficient': -2.0,
                'max_coefficient': 2.0,
                'use_evolution': False,
                'evolution_interval': 100,
                'population_size': 20,
                'evolution_sigma': 0.1,
                'evolution_learning_rate': 0.01
            },
            'optimizer': {
                'type': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'betas': [0.9, 0.999],
                'eps': 1.0e-8,
                'sam_rho': 0.05,
                'sam_adaptive': True,
                'lookahead_k': 5,
                'lookahead_alpha': 0.5,
                'lookahead_pullback': 'none',
                'scheduler': {
                    'type': 'reduce_on_plateau',
                    'patience': 5,
                    'factor': 0.5,
                    'min_lr': 1.0e-6
                },
                'param_groups': {
                    'feature_extractor': 1.0,
                    'segmentation': 0.5,
                    'reference_embeddings': 0.1,
                    'decoder': 0.5,
                    'equation_parameters': 0.01
                }
            },
            'loss': {
                'weights': {
                    'segmentation': 0.25,
                    'anomaly': 0.20,
                    'contrastive': 0.15,
                    'perceptual': 0.15,
                    'wasserstein': 0.10,
                    'reconstruction': 0.15
                },
                'focal_loss': {
                    'segmentation_alpha': 0.25,
                    'segmentation_gamma': 2.0,
                    'anomaly_alpha': 0.75,
                    'anomaly_gamma': 3.0
                },
                'contrastive_loss': {
                    'temperature': 0.07,
                    'normalize': True
                },
                'wasserstein_loss': {
                    'p': 1,
                    'epsilon': 0.1,
                    'max_iter': 100
                },
                'perceptual_loss': {
                    'network': 'vgg16',
                    'layers': [3, 8, 15, 22, 29],
                    'use_spatial': False
                }
            },
            'similarity': {
                'threshold': 0.7,
                'lpips': {
                    'network': 'vgg16',
                    'use_dropout': True,
                    'spatial': False
                },
                'optimal_transport': {
                    'epsilon': 0.1,
                    'max_iter': 100,
                    'metric': 'euclidean'
                },
                'ssim': {
                    'window_size': 11,
                    'use_edges': True,
                    'multi_scale': True
                },
                'combination_weights': {
                    'lpips': 0.4,
                    'ssim': 0.3,
                    'optimal_transport': 0.3
                }
            },
            'training': {
                'num_epochs': 2,
                'batch_size': 2,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'augmentation': {
                    'enabled': False,
                    'random_rotation': 15,
                    'random_scale': [0.9, 1.1],
                    'random_brightness': 0.1,
                    'random_contrast': 0.1,
                    'horizontal_flip': True,
                    'vertical_flip': True
                },
                'gradient_clip_value': 1.0,
                'gradient_clip_norm': 5.0,
                'use_amp': False,
                'amp_opt_level': 'O1',
                'use_distillation': False,
                'distillation_alpha': 0.7,
                'distillation_temperature': 4.0,
                'teacher_model_path': None
            },
            'anomaly': {
                'threshold': 0.3,
                'min_defect_size': 10,
                'max_defect_size': 1000,
                'ignore_boundary_width': 5,
                'region_transition_tolerance': 0.1,
                'defect_types': ['scratch', 'contamination', 'chip', 'crack', 'other'],
                'confidence_threshold': 0.5
            },
            'features': {
                'scales': [1.0, 0.75, 0.5, 0.25],
                'scale_weights': [0.4, 0.3, 0.2, 0.1],
                'gradient_kernel_size': 3,
                'gradient_normalization': True,
                'use_position_encoding': True,
                'position_encoding_dim': 64,
                'trend_window_size': 7,
                'trend_polynomial_degree': 2
            },
            'realtime': {
                'target_fps': 30,
                'use_pruning': False,
                'pruning_ratio': 0.3,
                'use_quantization': False,
                'quantization_backend': 'qnnpack',
                'max_batch_size': 8,
                'dynamic_batching': True,
                'enable_caching': True,
                'cache_size': 100
            },
            'visualization': {
                'window_width': 800,
                'window_height': 600,
                'fps': 30,
                'show_original': True,
                'show_segmentation': True,
                'show_anomaly_map': True,
                'show_reference_match': True,
                'show_statistics': True,
                'show_coefficients': True,
                'segmentation_colors': {
                    'core': [255, 0, 0],
                    'cladding': [0, 255, 0],
                    'ferrule': [0, 0, 255]
                },
                'anomaly_colormap': 'hot',
                'enable_parameter_adjustment': True,
                'parameter_adjustment_step': 0.01,
                'save_visualizations': True,
                'visualization_format': 'png',
                'video_codec': 'h264'
            },
            'advanced': {
                'use_gradient_checkpointing': False,
                'use_nas': False,
                'nas_search_space': 'darts',
                'use_maml': False,
                'maml_inner_steps': 5,
                'maml_inner_lr': 0.01,
                'use_self_supervised': False,
                'ssl_method': 'simclr',
                'use_uncertainty': True,
                'uncertainty_method': 'dropout',
                'dropout_samples': 10,
                'use_continual_learning': False,
                'replay_buffer_size': 1000,
                'use_neural_ode': False,
                'ode_solver': 'dopri5',
                'ode_rtol': 1e-3,
                'ode_atol': 1e-4
            },
            'experimental': {
                'use_cross_attention': False,
                'use_self_attention': False,
                'attention_heads': 8,
                'use_graph_features': False,
                'graph_hidden_dim': 128,
                'graph_num_layers': 3,
                'use_transformer': False,
                'transformer_depth': 12,
                'transformer_heads': 12,
                'transformer_dim': 768,
                'use_diffusion': False,
                'diffusion_steps': 1000,
                'diffusion_beta_schedule': 'linear'
            },
            'monitoring': {
                'track_memory_usage': True,
                'track_computation_time': True,
                'track_gradient_norms': True,
                'use_tensorboard': False,
                'tensorboard_dir': 'runs',
                'use_wandb': False,
                'wandb_project': 'fiber-optics',
                'wandb_entity': None,
                'enable_profiling': False,
                'profile_batches': [10, 20]
            },
            'debug': {
                'enabled': True,
                'save_intermediate_features': False,
                'save_gradient_flow': False,
                'save_attention_maps': False,
                'check_nan': True,
                'check_inf': True,
                'deterministic': True,
                'benchmark': False
            }
        }
    
    def create_mock_image(self, size: Tuple[int, int] = (256, 256), 
                         channels: int = 3) -> torch.Tensor:
        """Create a mock fiber optic image tensor"""
        # Create synthetic fiber optic image with core, cladding, and ferrule
        h, w = size
        image = torch.zeros(channels, h, w)
        
        # Create circular regions
        center_x, center_y = w // 2, h // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        
        # Core (inner circle)
        core_radius = min(h, w) // 8
        core_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < core_radius ** 2
        
        # Cladding (annulus)
        cladding_radius = min(h, w) // 4
        cladding_mask = (((x - center_x) ** 2 + (y - center_y) ** 2) < cladding_radius ** 2) & ~core_mask
        
        # Ferrule (outer region)
        ferrule_mask = ~(core_mask | cladding_mask)
        
        # Assign intensities
        for c in range(channels):
            image[c][core_mask] = 0.9 + torch.randn(core_mask.sum()) * 0.05
            image[c][cladding_mask] = 0.6 + torch.randn(cladding_mask.sum()) * 0.05
            image[c][ferrule_mask] = 0.3 + torch.randn(ferrule_mask.sum()) * 0.05
        
        # Add some defects
        num_defects = torch.randint(1, 5, (1,)).item()
        for _ in range(num_defects):
            defect_x = torch.randint(10, w - 10, (1,)).item()
            defect_y = torch.randint(10, h - 10, (1,)).item()
            defect_radius = torch.randint(5, 15, (1,)).item()
            
            defect_mask = ((x - defect_x) ** 2 + (y - defect_y) ** 2) < defect_radius ** 2
            for c in range(channels):
                image[c][defect_mask] *= 0.5  # Darken defect areas
        
        return image.clamp(0, 1)
    
    def create_mock_batch(self, batch_size: int = 4, 
                         size: Tuple[int, int] = (256, 256)) -> Dict[str, torch.Tensor]:
        """Create a batch of mock data"""
        images = torch.stack([self.create_mock_image(size) for _ in range(batch_size)])
        
        # Create mock labels
        labels = {
            'segmentation': torch.randint(0, 3, (batch_size, *size)),
            'anomaly': torch.rand(batch_size, *size) > 0.9,
            'reference_id': [f'ref_{i:03d}' for i in range(batch_size)]
        }
        
        return {
            'image': images,
            'labels': labels,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'batch_size': batch_size,
                'image_size': size
            }
        }
    
    def create_mock_model_output(self, batch_size: int = 4, 
                                size: Tuple[int, int] = (256, 256)) -> Dict[str, torch.Tensor]:
        """Create mock model output"""
        return {
            'segmentation_logits': torch.randn(batch_size, 3, *size),
            'anomaly_map': torch.rand(batch_size, 1, *size),
            'features': torch.randn(batch_size, 256),
            'reconstruction': torch.rand(batch_size, 3, *size),
            'reference_similarity': torch.rand(batch_size, 100),
            'final_similarity': torch.rand(batch_size),
            'meets_threshold': torch.rand(batch_size) > 0.3
        }
    
    def save_test_image(self, image: torch.Tensor, filename: str):
        """Save test image to file"""
        import torchvision
        path = self.test_data_dir / filename
        torchvision.utils.save_image(image, path)
        return path
    
    def assertTensorEqual(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                         msg: str = None, rtol: float = 1e-5, atol: float = 1e-8):
        """Assert two tensors are equal within tolerance"""
        if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
            diff = torch.abs(tensor1 - tensor2)
            max_diff = diff.max().item()
            if msg:
                self.fail(f"{msg}\nMax difference: {max_diff}")
            else:
                self.fail(f"Tensors not equal. Max difference: {max_diff}")
    
    def assertTensorShape(self, tensor: torch.Tensor, expected_shape: Tuple, 
                         msg: str = None):
        """Assert tensor has expected shape"""
        if tensor.shape != expected_shape:
            if msg:
                self.fail(f"{msg}\nExpected shape: {expected_shape}, got: {tensor.shape}")
            else:
                self.fail(f"Expected shape: {expected_shape}, got: {tensor.shape}")
    
    def assertNoNaN(self, tensor: torch.Tensor, msg: str = None):
        """Assert tensor contains no NaN values"""
        if torch.isnan(tensor).any():
            if msg:
                self.fail(f"{msg}\nTensor contains NaN values")
            else:
                self.fail("Tensor contains NaN values")
    
    def assertNoInf(self, tensor: torch.Tensor, msg: str = None):
        """Assert tensor contains no Inf values"""
        if torch.isinf(tensor).any():
            if msg:
                self.fail(f"{msg}\nTensor contains Inf values")
            else:
                self.fail("Tensor contains Inf values")
    
    def assertValidProbability(self, tensor: torch.Tensor, msg: str = None):
        """Assert tensor contains valid probability values [0, 1]"""
        if (tensor < 0).any() or (tensor > 1).any():
            if msg:
                self.fail(f"{msg}\nTensor contains values outside [0, 1]")
            else:
                self.fail("Tensor contains values outside [0, 1]")


if __name__ == "__main__":
    # Test the base test case
    class TestBaseTestCase(FiberOpticsTestCase):
        def test_mock_image_creation(self):
            """Test mock image creation"""
            image = self.create_mock_image()
            self.assertTensorShape(image, (3, 256, 256))
            self.assertNoNaN(image)
            self.assertValidProbability(image)
        
        def test_mock_batch_creation(self):
            """Test mock batch creation"""
            batch = self.create_mock_batch(batch_size=8)
            self.assertTensorShape(batch['image'], (8, 3, 256, 256))
            self.assertEqual(len(batch['labels']['reference_id']), 8)
        
        def test_config_creation(self):
            """Test configuration creation"""
            self.assertIsInstance(self.test_config, dict)
            self.assertIn('system', self.test_config)
            self.assertIn('model', self.test_config)
            self.assertEqual(self.test_config['similarity']['threshold'], 0.7)
    
    unittest.main()