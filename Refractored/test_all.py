# test_all.py
# Comprehensive unit tests for the fiber optic analysis system

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import cv2

# Import all modules to test
from config import ConfigManager, setup_logging, setup_distributed, ensure_directories, Box
from model import CBAM, FiberOpticNet, create_model, load_model
from dataset import FiberOpticsDataset, create_dataloaders
from losses import FocalLoss, CombinedLoss, get_loss_function
from optimizer import ModelOptimizer, create_optimizer
from trainer import Trainer, create_trainer
from evaluator import Evaluator, create_evaluator
from utils import (
    setup_seed, get_device_info, create_segmentation_overlay,
    create_anomaly_heatmap, preprocess_image_for_inference,
    save_predictions_to_json, plot_training_history, plot_confusion_matrix,
    calculate_class_weights, log_model_summary, MetricsTracker
)
from app import FiberOpticAnalysisApp


class TestConfig(unittest.TestCase):
    """Test cases for config.py"""
    
    def setUp(self):
        """Create a temporary config file for testing"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test config
        test_config = {
            'system': {
                'checkpoints_path': './test_checkpoints',
                'mode': 'train',
                'seed': 42
            },
            'data': {
                'path': './test_data',
                'class_names': ['class1', 'class2']
            },
            'model': {
                'backbone': 'resnet18',
                'num_classes': 2
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_box_class(self):
        """Test the Box class fallback"""
        box = Box({'key1': 'value1', 'nested': {'key2': 'value2'}})
        self.assertEqual(box.key1, 'value1')
        box.key3 = 'value3'
        self.assertEqual(box['key3'], 'value3')
    
    def test_config_manager_init(self):
        """Test ConfigManager initialization"""
        cm = ConfigManager(str(self.config_path))
        self.assertIsNotNone(cm.config)
        self.assertEqual(cm.config.system.mode, 'train')
    
    def test_config_manager_get(self):
        """Test ConfigManager get method"""
        cm = ConfigManager(str(self.config_path))
        self.assertEqual(cm.get('system.mode'), 'train')
        self.assertEqual(cm.get('nonexistent.key', 'default'), 'default')
    
    def test_config_manager_update(self):
        """Test ConfigManager update method"""
        cm = ConfigManager(str(self.config_path))
        cm.update('system.mode', 'eval')
        self.assertEqual(cm.config.system.mode, 'eval')
    
    def test_config_manager_save(self):
        """Test ConfigManager save method"""
        cm = ConfigManager(str(self.config_path))
        cm.update('system.mode', 'optimize')
        
        new_path = Path(self.temp_dir) / "saved_config.yaml"
        cm.save_config(str(new_path))
        
        with open(new_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config['system']['mode'], 'optimize')
    
    def test_setup_logging(self):
        """Test setup_logging function"""
        logger = setup_logging(logging.DEBUG)
        self.assertIsInstance(logger, logging.Logger)
    
    @patch.dict(os.environ, {'RANK': '0', 'WORLD_SIZE': '2', 'LOCAL_RANK': '0'})
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.set_device')
    @patch('torch.distributed.init_process_group')
    def test_setup_distributed(self, mock_init_pg, mock_set_device, mock_cuda):
        """Test setup_distributed function"""
        rank, world_size, local_rank, is_distributed = setup_distributed()
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 2)
        self.assertEqual(local_rank, 0)
        self.assertTrue(is_distributed)
    
    def test_setup_distributed_no_env(self):
        """Test setup_distributed without environment variables"""
        # Clear environment variables if they exist
        for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']:
            os.environ.pop(key, None)
        
        rank, world_size, local_rank, is_distributed = setup_distributed()
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        self.assertEqual(local_rank, 0)
        self.assertFalse(is_distributed)
    
    def test_ensure_directories(self):
        """Test ensure_directories function"""
        config = Box({
            'system': {'checkpoints_path': str(Path(self.temp_dir) / 'checkpoints')},
            'data': {'path': str(self.temp_dir)}
        })
        
        checkpoint_dir = ensure_directories(config)
        self.assertTrue(checkpoint_dir.exists())
    
    def test_ensure_directories_missing_data(self):
        """Test ensure_directories with missing data directory"""
        config = Box({
            'system': {'checkpoints_path': str(Path(self.temp_dir) / 'checkpoints')},
            'data': {'path': str(Path(self.temp_dir) / 'nonexistent')}
        })
        
        with self.assertRaises(FileNotFoundError):
            ensure_directories(config)


class TestModel(unittest.TestCase):
    """Test cases for model.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Box({
            'model': {
                'backbone': 'resnet18',
                'image_size': 224,
                'num_classes': 4,
                'embedding_dim': 128
            },
            'equation': {
                'coefficients': {'A': 0.5, 'B': 0.3, 'C': 0.2}
            }
        })
        self.device = 'cpu'
    
    def test_cbam_module(self):
        """Test CBAM attention module"""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        output = cbam(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_fiber_optic_net_init(self):
        """Test FiberOpticNet initialization"""
        model = FiberOpticNet(self.config.model)
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.attention)
        self.assertIsNotNone(model.region_classifier)
        self.assertIsNotNone(model.anomaly_head)
        self.assertIsNotNone(model.embedding_head)
    
    def test_fiber_optic_net_forward(self):
        """Test FiberOpticNet forward pass"""
        model = FiberOpticNet(self.config.model)
        model.eval()
        
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(image)
        
        # Check output shapes
        self.assertEqual(outputs['region_logits'].shape, (batch_size, 4))
        self.assertEqual(outputs['region_probs'].shape, (batch_size, 4))
        self.assertEqual(outputs['anomaly_map'].shape[0], batch_size)
        self.assertEqual(outputs['embedding'].shape, (batch_size, 128))
        self.assertEqual(outputs['classification_confidence'].shape, (batch_size,))
        self.assertEqual(outputs['anomaly_score'].shape, (batch_size,))
        self.assertEqual(outputs['final_similarity_score'].shape, (batch_size,))
    
    def test_fiber_optic_net_with_reference(self):
        """Test FiberOpticNet forward pass with reference image"""
        model = FiberOpticNet(self.config.model)
        model.eval()
        
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)
        ref_image = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(image, ref_image=ref_image)
        
        self.assertIsNotNone(outputs['ref_embedding'])
        self.assertEqual(outputs['ref_embedding'].shape, (batch_size, 128))
        self.assertGreater(outputs['embedding_similarity'].sum().item(), 0)
    
    def test_create_model(self):
        """Test create_model factory function"""
        config = type('Config', (), {'model': self.config.model})()
        model = create_model(config)
        self.assertIsInstance(model, FiberOpticNet)
    
    def test_load_model(self):
        """Test load_model function"""
        # Create a temporary checkpoint
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
        
        # Create and save a model
        config = type('Config', (), {'model': self.config.model})()
        model = create_model(config)
        torch.save({'state_dict': model.state_dict()}, checkpoint_path)
        
        # Load the model
        loaded_model = load_model(config, str(checkpoint_path), device='cpu')
        self.assertIsInstance(loaded_model, FiberOpticNet)
        
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_load_model_nonexistent(self):
        """Test load_model with nonexistent checkpoint"""
        config = type('Config', (), {'model': self.config.model})()
        model = load_model(config, 'nonexistent.pth', device='cpu')
        self.assertIsInstance(model, FiberOpticNet)


class TestDataset(unittest.TestCase):
    """Test cases for dataset.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "data"
        self.data_path.mkdir()
        
        # Create dummy data structure
        for class_name in ['class1', 'class2']:
            class_dir = self.data_path / class_name
            class_dir.mkdir()
            
            # Create dummy image file
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(class_dir / f"{class_name}_001.png"), dummy_image)
            
            # Create dummy tensor file
            dummy_tensor = torch.zeros(3, 224, 224)
            torch.save(dummy_tensor, class_dir / f"{class_name}_001.pt")
        
        self.config = Box({
            'data': {
                'path': str(self.data_path),
                'num_workers': 0,
                'image_size': 224,
                'class_names': ['class1', 'class2']
            },
            'training': {
                'batch_size': 2
            }
        })
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_init(self):
        """Test FiberOpticsDataset initialization"""
        dataset = FiberOpticsDataset(self.config, mode='train')
        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset.class_to_idx), 2)
    
    def test_dataset_getitem(self):
        """Test FiberOpticsDataset __getitem__"""
        dataset = FiberOpticsDataset(self.config, mode='train')
        sample = dataset[0]
        
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertIn('reference', sample)
        self.assertIn('file_path', sample)
        
        self.assertEqual(sample['image'].shape, (3, 224, 224))
        self.assertIsInstance(sample['label'], torch.Tensor)
    
    def test_dataset_load_image_file(self):
        """Test _load_image_file method"""
        dataset = FiberOpticsDataset(self.config, mode='train')
        image_path = list(self.data_path.glob("**/*.png"))[0]
        image = dataset._load_image_file(image_path)
        self.assertEqual(image.shape, (224, 224, 3))
    
    def test_dataset_load_tensor_file(self):
        """Test _load_tensor_file method"""
        dataset = FiberOpticsDataset(self.config, mode='train')
        tensor_path = list(self.data_path.glob("**/*.pt"))[0]
        image = dataset._load_tensor_file(tensor_path)
        self.assertEqual(image.shape, (224, 224, 3))
    
    def test_create_dataloaders(self):
        """Test create_dataloaders function"""
        train_loader, val_loader, train_sampler = create_dataloaders(self.config)
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNone(train_sampler)  # Not distributed
        
        # Test batch
        batch = next(iter(train_loader))
        self.assertEqual(batch['image'].shape[0], 2)  # batch_size


class TestLosses(unittest.TestCase):
    """Test cases for losses.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Box({
            'loss': {
                'type': 'focal',
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'weights': {
                    'classification': 1.0,
                    'anomaly': 0.7,
                    'similarity': 0.5
                }
            },
            'data': {
                'class_map': {'defects': 3}
            }
        })
        self.device = 'cpu'
    
    def test_focal_loss(self):
        """Test FocalLoss"""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        
        loss = loss_fn(inputs, targets)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_combined_loss(self):
        """Test CombinedLoss"""
        loss_fn = CombinedLoss(self.config)
        
        # Create mock outputs
        batch_size = 4
        outputs = {
            'region_logits': torch.randn(batch_size, 4),
            'anomaly_map': torch.randn(batch_size, 1, 32, 32),
            'embedding': torch.randn(batch_size, 128),
            'ref_embedding': torch.randn(batch_size, 128)
        }
        
        batch = {
            'label': torch.tensor([0, 1, 2, 3])
        }
        
        losses = loss_fn(outputs, batch, self.device)
        
        self.assertIn('total', losses)
        self.assertIn('classification', losses)
        self.assertIn('anomaly', losses)
        self.assertIn('similarity', losses)
        
        self.assertGreater(losses['total'].item(), 0)
    
    def test_get_loss_function(self):
        """Test get_loss_function factory"""
        loss_fn = get_loss_function(self.config)
        self.assertIsInstance(loss_fn, CombinedLoss)


class TestOptimizer(unittest.TestCase):
    """Test cases for optimizer.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Box({
            'model': {
                'backbone': 'resnet18',
                'image_size': 224,
                'num_classes': 4,
                'embedding_dim': 128
            },
            'student_model': {
                'backbone': 'mobilenet_v3_small',
                'image_size': 224,
                'num_classes': 4,
                'embedding_dim': 64
            },
            'optimizer': {
                'learning_rate': 0.001
            },
            'training': {
                'use_amp': False,
                'batch_size': 2
            },
            'optimization': {
                'temperature': 4.0,
                'alpha': 0.7,
                'distillation_epochs': 1,
                'prune_after_training': True,
                'pruning_ratio': 0.3,
                'distill_after_training': True
            }
        })
        self.device = 'cpu'
    
    def test_model_optimizer_init(self):
        """Test ModelOptimizer initialization"""
        optimizer = ModelOptimizer(self.config, self.device)
        self.assertEqual(optimizer.device, self.device)
    
    def test_prune_model(self):
        """Test model pruning"""
        optimizer = ModelOptimizer(self.config, self.device)
        model = create_model(type('Config', (), {'model': self.config.model})())
        
        # Count parameters before pruning
        total_params_before = sum(p.numel() for p in model.parameters())
        
        # Prune model
        pruned_model = optimizer.prune_model(model, ratio=0.3)
        
        # Model should still work
        with torch.no_grad():
            output = pruned_model(torch.randn(1, 3, 224, 224))
        self.assertIsNotNone(output)
    
    def test_calculate_model_size(self):
        """Test calculate_model_size method"""
        optimizer = ModelOptimizer(self.config, self.device)
        model = create_model(type('Config', (), {'model': self.config.model})())
        
        size_info = optimizer.calculate_model_size(model)
        
        self.assertIn('total_parameters', size_info)
        self.assertIn('trainable_parameters', size_info)
        self.assertIn('memory_mb', size_info)
        self.assertGreater(size_info['total_parameters'], 0)
    
    def test_create_optimizer(self):
        """Test create_optimizer factory function"""
        optimizer = create_optimizer(self.config, self.device)
        self.assertIsInstance(optimizer, ModelOptimizer)


class TestTrainer(unittest.TestCase):
    """Test cases for trainer.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Box({
            'model': {
                'backbone': 'resnet18',
                'image_size': 224,
                'num_classes': 4,
                'embedding_dim': 128
            },
            'training': {
                'use_amp': False,
                'num_epochs': 1,
                'log_interval': 1,
                'batch_size': 2
            },
            'optimizer': {
                'learning_rate': 0.001,
                'weight_decay': 0.0001
            },
            'loss': {
                'type': 'crossentropy',
                'weights': {
                    'classification': 1.0,
                    'anomaly': 0.7,
                    'similarity': 0.5
                }
            },
            'data': {
                'class_map': {'defects': 3}
            },
            'equation': {
                'coefficients': {'A': 0.5, 'B': 0.3, 'C': 0.2}
            },
            'system': {
                'checkpoints_path': tempfile.mkdtemp()
            }
        })
        self.device = 'cpu'
    
    def tearDown(self):
        """Clean up temporary files"""
        if hasattr(self.config.system, 'checkpoints_path'):
            shutil.rmtree(self.config.system.checkpoints_path, ignore_errors=True)
    
    def test_trainer_init(self):
        """Test Trainer initialization"""
        trainer = Trainer(self.config, rank=0, world_size=1, local_rank=0, is_distributed=False)
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.loss_fn)
        self.assertIsNotNone(trainer.evaluator)
    
    def test_create_trainer(self):
        """Test create_trainer factory function"""
        trainer = create_trainer(self.config, rank=0, world_size=1, local_rank=0, is_distributed=False)
        self.assertIsInstance(trainer, Trainer)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        trainer = Trainer(self.config, rank=0, world_size=1, local_rank=0, is_distributed=False)
        
        checkpoint_dir = Path(self.config.system.checkpoints_path)
        val_metrics = {'accuracy': 0.9, 'avg_similarity': 0.8}
        
        trainer._save_checkpoint(0, 0.1, val_metrics, checkpoint_dir)
        
        # Check if checkpoint was saved
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        self.assertGreater(len(checkpoint_files), 0)
    
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        trainer = Trainer(self.config, rank=0, world_size=1, local_rank=0, is_distributed=False)
        
        # Save a checkpoint first
        checkpoint_dir = Path(self.config.system.checkpoints_path)
        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        
        checkpoint = {
            'epoch': 5,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        success = trainer.load_checkpoint(str(checkpoint_path))
        self.assertTrue(success)
        self.assertEqual(trainer.current_epoch, 5)


class TestEvaluator(unittest.TestCase):
    """Test cases for evaluator.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Box({
            'data': {
                'class_names': ['class1', 'class2', 'class3']
            },
            'training': {
                'use_amp': False
            },
            'equation': {
                'coefficients': {'A': 0.5, 'B': 0.3, 'C': 0.2}
            },
            'similarity': {
                'threshold': 0.7
            }
        })
        self.device = 'cpu'
    
    def test_evaluator_init(self):
        """Test Evaluator initialization"""
        evaluator = Evaluator(self.config, self.device)
        self.assertEqual(evaluator.device, self.device)
    
    def test_evaluate_single_sample(self):
        """Test evaluate_single_sample method"""
        evaluator = Evaluator(self.config, self.device)
        
        # Create a mock model
        model = Mock()
        model.eval = Mock()
        model.return_value = {
            'region_logits': torch.randn(1, 3),
            'region_probs': torch.softmax(torch.randn(1, 3), dim=1),
            'final_similarity_score': torch.tensor([0.8]),
            'anomaly_score': torch.tensor([0.2]),
            'anomaly_map': torch.randn(1, 1, 32, 32),
            'embedding': torch.randn(1, 128),
            'classification_confidence': torch.tensor([0.9])
        }
        
        # Test with single image
        image_tensor = torch.randn(3, 224, 224)
        result = evaluator.evaluate_single_sample(model, image_tensor)
        
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('similarity_score', result)
        self.assertIn('passes_threshold', result)
        self.assertIn('status', result)
    
    def test_create_evaluator(self):
        """Test create_evaluator factory function"""
        evaluator = create_evaluator(self.config, self.device)
        self.assertIsInstance(evaluator, Evaluator)


class TestUtils(unittest.TestCase):
    """Test cases for utils.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_seed(self):
        """Test setup_seed function"""
        setup_seed(42)
        # Generate some random numbers
        rand1 = torch.rand(10)
        np_rand1 = np.random.rand(10)
        
        # Reset seed and generate again
        setup_seed(42)
        rand2 = torch.rand(10)
        np_rand2 = np.random.rand(10)
        
        # Should be identical
        self.assertTrue(torch.allclose(rand1, rand2))
        self.assertTrue(np.allclose(np_rand1, np_rand2))
    
    def test_get_device_info(self):
        """Test get_device_info function"""
        info = get_device_info()
        
        self.assertIn('cuda_available', info)
        self.assertIn('cuda_device_count', info)
        self.assertIn('current_device', info)
        self.assertIn('device_names', info)
    
    def test_preprocess_image_for_inference(self):
        """Test preprocess_image_for_inference function"""
        # Create dummy image
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor = preprocess_image_for_inference(image, target_size=224)
        
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertIsInstance(tensor, torch.Tensor)
    
    def test_save_predictions_to_json(self):
        """Test save_predictions_to_json function"""
        predictions = {
            'class': 1,
            'score': 0.95,
            'probabilities': np.array([0.05, 0.95]),
            'nested': {
                'array': np.array([[1, 2], [3, 4]])
            }
        }
        
        output_path = Path(self.temp_dir) / "predictions.json"
        save_predictions_to_json(predictions, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        import json
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['class'], 1)
        self.assertEqual(loaded['score'], 0.95)
        self.assertIsInstance(loaded['probabilities'], list)
    
    def test_calculate_class_weights(self):
        """Test calculate_class_weights function"""
        # Create mock dataset
        dataset = Mock()
        dataset.__len__ = Mock(return_value=10)
        
        # Mock samples with different labels
        def mock_getitem(self, idx):
            labels = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
            return {'label': torch.tensor(labels[idx])}
        
        dataset.__getitem__ = mock_getitem
        
        weights = calculate_class_weights(dataset, num_classes=4)
        self.assertEqual(len(weights), 4)
        # Class 3 has only 1 sample, class 0 has 3, class 1 has 2, class 2 has 4
        # So weights should be inversely proportional: class 3 > class 1 > class 0 > class 2
        self.assertGreater(weights[3], weights[0])  # Class 3 (1 sample) > class 0 (3 samples)
        self.assertGreater(weights[0], weights[2])  # Class 0 (3 samples) > class 2 (4 samples)
    
    def test_metrics_tracker(self):
        """Test MetricsTracker class"""
        tracker = MetricsTracker()
        
        # Update metrics
        tracker.update(loss=0.5, accuracy=0.8)
        tracker.update(loss=0.3, accuracy=0.85)
        
        # Test get latest
        self.assertEqual(tracker.get_latest('loss'), 0.3)
        self.assertEqual(tracker.get_latest('accuracy'), 0.85)
        
        # Test get best
        self.assertEqual(tracker.get_best('loss', mode='min'), 0.3)
        self.assertEqual(tracker.get_best('accuracy', mode='max'), 0.85)
        
        # Test save/load
        save_path = Path(self.temp_dir) / "metrics.json"
        tracker.save(save_path)
        
        new_tracker = MetricsTracker()
        new_tracker.load(save_path)
        self.assertEqual(new_tracker.get_latest('loss'), 0.3)


class TestApp(unittest.TestCase):
    """Test cases for app.py"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # Create test config
        test_config = {
            'system': {
                'checkpoints_path': str(self.temp_dir)
            },
            'webapp': {
                'default_checkpoint': 'test.pth',
                'host': '127.0.0.1',
                'port': 7860,
                'share': False
            },
            'data': {
                'image_size': 224,
                'class_names': ['class1', 'class2']
            },
            'model': {
                'backbone': 'resnet18',
                'image_size': 224,
                'num_classes': 2,
                'embedding_dim': 128
            },
            'equation': {
                'coefficients': {'A': 0.5, 'B': 0.3, 'C': 0.2}
            },
            'similarity': {
                'threshold': 0.7
            },
            'visualization': {
                'segmentation_colors': [[255, 0, 0], [0, 255, 0]]
            },
            'training': {
                'use_amp': False
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    @patch('app.GRADIO_AVAILABLE', False)
    def test_app_init_no_gradio(self):
        """Test FiberOpticAnalysisApp initialization without Gradio"""
        app = FiberOpticAnalysisApp(str(self.config_path))
        self.assertIsNone(app.interface)
    
    def test_analyze_image_no_image(self):
        """Test analyze_image with no image"""
        app = FiberOpticAnalysisApp(str(self.config_path))
        result = app.analyze_image(None)
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No image provided')


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_structure()
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_structure(self):
        """Create a complete test structure"""
        # Create config
        config_path = Path(self.temp_dir) / "config.yaml"
        config = {
            'system': {
                'checkpoints_path': str(Path(self.temp_dir) / 'checkpoints'),
                'mode': 'train',
                'seed': 42,
                'verbose': False,
                'config_path': str(config_path)
            },
            'data': {
                'path': str(Path(self.temp_dir) / 'data'),
                'num_workers': 0,
                'image_size': 64,  # Small for testing
                'class_names': ['class1', 'class2'],
                'class_map': {'defects': 1}
            },
            'model': {
                'backbone': 'resnet18',
                'image_size': 64,
                'num_classes': 2,
                'embedding_dim': 32
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 2,
                'log_interval': 1,
                'use_amp': False
            },
            'optimizer': {
                'learning_rate': 0.001,
                'weight_decay': 0.0001
            },
            'loss': {
                'type': 'crossentropy',
                'weights': {
                    'classification': 1.0,
                    'anomaly': 0.7,
                    'similarity': 0.5
                }
            },
            'equation': {
                'coefficients': {'A': 0.5, 'B': 0.3, 'C': 0.2}
            },
            'similarity': {
                'threshold': 0.7
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create data directory
        data_path = Path(self.temp_dir) / 'data'
        for class_name in ['class1', 'class2']:
            class_dir = data_path / class_name
            class_dir.mkdir(parents=True)
            
            # Create dummy images
            for i in range(4):
                dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f"{class_name}_{i:03d}.png"), dummy_image)
    
    def test_full_pipeline(self):
        """Test the full training pipeline"""
        config_path = Path(self.temp_dir) / "config.yaml"
        
        # Load config
        config_manager = ConfigManager(str(config_path))
        config = config_manager.config
        
        # Create data loaders
        train_loader, val_loader, _ = create_dataloaders(config)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Create model
        model = create_model(type('Config', (), {'model': config.model})())
        self.assertIsNotNone(model)
        
        # Create loss function
        loss_fn = get_loss_function(config)
        self.assertIsNotNone(loss_fn)
        
        # Test forward pass
        batch = next(iter(train_loader))
        outputs = model(batch['image'], batch['reference'])
        
        # Calculate loss
        losses = loss_fn(outputs, batch, 'cpu')
        self.assertGreater(losses['total'].item(), 0)
        
        # Create evaluator and test evaluation
        evaluator = create_evaluator(config, 'cpu')
        metrics = evaluator.evaluate(model, val_loader)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('avg_similarity', metrics)
        self.assertIsInstance(metrics['accuracy'], float)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)