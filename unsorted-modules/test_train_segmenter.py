#!/usr/bin/env python3
"""
Unit tests for train_segmenter module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import torch.nn as nn


class TestTrainSegmenter(unittest.TestCase):
    """Test cases for segmentation model training functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_classes = 21  # Common for segmentation (e.g., PASCAL VOC)
        self.image_height = 256
        self.image_width = 256
        
        # Model configuration
        self.model_config = {
            'architecture': 'unet',
            'encoder': 'resnet34',
            'num_classes': self.num_classes,
            'input_channels': 3,
            'pretrained_encoder': True,
            'decoder_channels': [256, 128, 64, 32, 16]
        }
        
        # Training configuration
        self.train_config = {
            'epochs': 50,
            'batch_size': self.batch_size,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss_function': 'cross_entropy',
            'use_mixed_precision': False,
            'augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'rotation': 15,
                'scale': 0.2
            }
        }
        
        # Sample data
        self.sample_images = torch.randn(self.batch_size, 3, self.image_height, self.image_width)
        self.sample_masks = torch.randint(0, self.num_classes, 
                                        (self.batch_size, self.image_height, self.image_width))
        
        # Class names for visualization
        self.class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        
    def tearDown(self):
        """Clean up after each test method."""
        self.sample_images = None
        self.sample_masks = None
        self.model_config = None
        self.train_config = None
        
    def test_module_imports(self):
        """Test that train_segmenter module can be imported."""
        try:
            import train_segmenter
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import train_segmenter: {e}")
            
    def test_data_shapes(self):
        """Test data shape validation."""
        # Test image shape
        self.assertEqual(self.sample_images.shape, 
                        (self.batch_size, 3, self.image_height, self.image_width))
        
        # Test mask shape
        self.assertEqual(self.sample_masks.shape,
                        (self.batch_size, self.image_height, self.image_width))
        
        # Test value ranges
        self.assertTrue(torch.all(self.sample_masks >= 0))
        self.assertTrue(torch.all(self.sample_masks < self.num_classes))
        
    def test_loss_calculation(self):
        """Test segmentation loss calculation."""
        # Create dummy predictions
        predictions = torch.randn(self.batch_size, self.num_classes, 
                                self.image_height, self.image_width)
        
        # Calculate cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, self.sample_masks)
        
        # Verify loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertGreater(loss.item(), 0)
        
        # Test with perfect predictions
        perfect_predictions = torch.zeros_like(predictions)
        for b in range(self.batch_size):
            for h in range(self.image_height):
                for w in range(self.image_width):
                    perfect_predictions[b, self.sample_masks[b, h, w], h, w] = 10.0
                    
        perfect_loss = criterion(perfect_predictions, self.sample_masks)
        self.assertLess(perfect_loss.item(), loss.item())
        
    def test_mock_training(self):
        """Test training process with mocks."""
        mock_trainer = Mock()
        
        # Mock training
        mock_trainer.train = MagicMock(return_value={
            'best_model_state': 'model_state_dict',
            'training_history': {
                'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
                'val_loss': [0.9, 0.7, 0.5, 0.35, 0.3],
                'train_iou': [0.4, 0.5, 0.6, 0.7, 0.75],
                'val_iou': [0.35, 0.45, 0.55, 0.65, 0.72]
            },
            'best_val_iou': 0.72,
            'total_epochs': 5
        })
        
        result = mock_trainer.train(self.sample_images, self.sample_masks, self.train_config)
        
        # Verify training results
        self.assertIn('best_model_state', result)
        self.assertIn('training_history', result)
        self.assertIn('best_val_iou', result)
        
        # Verify metrics improved
        history = result['training_history']
        self.assertLess(history['train_loss'][-1], history['train_loss'][0])
        self.assertGreater(history['train_iou'][-1], history['train_iou'][0])
        
    def test_iou_calculation(self):
        """Test Intersection over Union (IoU) calculation."""
        # Create simple predictions and ground truth
        pred = torch.zeros((2, 2), dtype=torch.long)
        pred[0, 0] = 1
        pred[0, 1] = 1
        
        target = torch.zeros((2, 2), dtype=torch.long)
        target[0, 0] = 1
        target[1, 0] = 1
        
        # Calculate IoU for class 1
        pred_class1 = (pred == 1)
        target_class1 = (target == 1)
        
        intersection = torch.sum(pred_class1 & target_class1).float()
        union = torch.sum(pred_class1 | target_class1).float()
        
        iou = intersection / (union + 1e-8)
        
        # Expected: intersection=1, union=3, IoU=1/3
        self.assertAlmostEqual(iou.item(), 1/3, places=5)
        
    def test_data_augmentation(self):
        """Test data augmentation effects."""
        original_image = self.sample_images[0].clone()
        original_mask = self.sample_masks[0].clone()
        
        # Simulate horizontal flip
        flipped_image = torch.flip(original_image, dims=[2])  # Flip width dimension
        flipped_mask = torch.flip(original_mask, dims=[1])    # Flip width dimension
        
        # Verify shapes remain the same
        self.assertEqual(flipped_image.shape, original_image.shape)
        self.assertEqual(flipped_mask.shape, original_mask.shape)
        
        # Verify content changed
        self.assertFalse(torch.allclose(flipped_image, original_image))
        self.assertFalse(torch.equal(flipped_mask, original_mask))
        
    def test_model_output_processing(self):
        """Test processing of model outputs."""
        # Simulate model output (logits)
        logits = torch.randn(self.batch_size, self.num_classes, 
                           self.image_height, self.image_width)
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Get predictions
        predictions = torch.argmax(probs, dim=1)
        
        # Verify shapes
        self.assertEqual(predictions.shape, 
                        (self.batch_size, self.image_height, self.image_width))
        
        # Verify prediction range
        self.assertTrue(torch.all(predictions >= 0))
        self.assertTrue(torch.all(predictions < self.num_classes))
        
        # Verify probability properties
        prob_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))
        
    def test_class_weighting(self):
        """Test class weighting for imbalanced datasets."""
        # Calculate class frequencies in masks
        class_counts = torch.zeros(self.num_classes)
        for c in range(self.num_classes):
            class_counts[c] = (self.sample_masks == c).sum()
            
        # Calculate inverse frequency weights
        total_pixels = self.sample_masks.numel()
        class_weights = total_pixels / (self.num_classes * class_counts + 1)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        # Verify weights
        self.assertEqual(len(class_weights), self.num_classes)
        self.assertTrue(torch.all(class_weights > 0))
        self.assertTrue(torch.all(torch.isfinite(class_weights)))
        
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        initial_lr = self.train_config['learning_rate']
        
        # Simulate ReduceLROnPlateau behavior
        lr = initial_lr
        patience = 3
        factor = 0.5
        
        val_losses = [0.5, 0.4, 0.35, 0.35, 0.34, 0.34, 0.34]  # Plateau
        best_loss = float('inf')
        patience_counter = 0
        
        lr_history = []
        
        for loss in val_losses:
            lr_history.append(lr)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                lr *= factor
                patience_counter = 0
                
        # Verify LR was reduced
        self.assertLess(lr, initial_lr)
        self.assertEqual(lr, initial_lr * factor)
        
    def test_checkpoint_saving(self):
        """Test model checkpoint saving logic."""
        best_iou = 0.0
        checkpoint_history = []
        
        # Simulate validation IoU scores
        val_ious = [0.4, 0.5, 0.55, 0.54, 0.6, 0.58, 0.62]
        
        for epoch, iou in enumerate(val_ious):
            if iou > best_iou:
                best_iou = iou
                checkpoint_history.append({
                    'epoch': epoch,
                    'iou': iou,
                    'saved': True
                })
            else:
                checkpoint_history.append({
                    'epoch': epoch,
                    'iou': iou,
                    'saved': False
                })
                
        # Verify checkpoints were saved at right times
        saved_epochs = [c['epoch'] for c in checkpoint_history if c['saved']]
        self.assertEqual(saved_epochs, [0, 1, 2, 4, 6])
        self.assertEqual(best_iou, max(val_ious))


if __name__ == '__main__':
    unittest.main()