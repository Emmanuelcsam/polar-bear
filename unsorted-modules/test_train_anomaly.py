#!/usr/bin/env python3
"""
Unit tests for train_anomaly module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch


class TestTrainAnomaly(unittest.TestCase):
    """Test cases for anomaly detection training functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cpu')
        self.sequence_length = 100
        self.feature_dim = 32
        self.batch_size = 8
        self.latent_dim = 16
        
        # Training configuration
        self.train_config = {
            'model_type': 'autoencoder',
            'epochs': 100,
            'batch_size': self.batch_size,
            'learning_rate': 0.001,
            'reconstruction_threshold': 0.05,
            'contamination_ratio': 0.1,
            'validation_split': 0.2
        }
        
        # Generate normal data (low variance)
        self.normal_data = torch.randn(1000, self.sequence_length, self.feature_dim) * 0.5
        
        # Generate anomalous data (high variance, different patterns)
        self.anomaly_data = torch.randn(100, self.sequence_length, self.feature_dim) * 2.0
        self.anomaly_data += torch.sin(torch.linspace(0, 10, self.sequence_length)).unsqueeze(0).unsqueeze(2)
        
        # Combined dataset
        self.all_data = torch.cat([self.normal_data, self.anomaly_data], dim=0)
        self.labels = torch.cat([
            torch.zeros(len(self.normal_data)),
            torch.ones(len(self.anomaly_data))
        ])
        
    def tearDown(self):
        """Clean up after each test method."""
        self.normal_data = None
        self.anomaly_data = None
        self.all_data = None
        self.labels = None
        
    def test_module_imports(self):
        """Test that train_anomaly module can be imported."""
        try:
            import train_anomaly
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import train_anomaly: {e}")
            
    def test_data_properties(self):
        """Test data properties and distributions."""
        # Test shapes
        self.assertEqual(self.normal_data.shape[1:], (self.sequence_length, self.feature_dim))
        self.assertEqual(self.anomaly_data.shape[1:], (self.sequence_length, self.feature_dim))
        
        # Test variance differences
        normal_var = torch.var(self.normal_data).item()
        anomaly_var = torch.var(self.anomaly_data).item()
        self.assertGreater(anomaly_var, normal_var)
        
        # Test label distribution
        anomaly_ratio = torch.sum(self.labels).item() / len(self.labels)
        self.assertAlmostEqual(anomaly_ratio, 
                              len(self.anomaly_data) / len(self.all_data), 
                              places=3)
        
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        # Simulate autoencoder outputs
        original = torch.randn(self.batch_size, self.sequence_length, self.feature_dim)
        
        # Perfect reconstruction
        perfect_recon = original.clone()
        perfect_error = torch.mean((original - perfect_recon) ** 2)
        self.assertAlmostEqual(perfect_error.item(), 0.0, places=6)
        
        # Noisy reconstruction
        noisy_recon = original + torch.randn_like(original) * 0.1
        noisy_error = torch.mean((original - noisy_recon) ** 2)
        self.assertGreater(noisy_error.item(), 0)
        
        # Poor reconstruction
        poor_recon = torch.randn_like(original)
        poor_error = torch.mean((original - poor_recon) ** 2)
        self.assertGreater(poor_error.item(), noisy_error.item())
        
    def test_mock_anomaly_training(self):
        """Test anomaly detection training with mocks."""
        mock_trainer = Mock()
        
        # Mock training process
        mock_trainer.train = MagicMock(return_value={
            'model_state': 'trained',
            'final_loss': 0.023,
            'threshold': 0.045,
            'training_history': {
                'loss': [0.5, 0.3, 0.1, 0.05, 0.023],
                'val_loss': [0.6, 0.35, 0.15, 0.08, 0.04]
            },
            'performance': {
                'precision': 0.92,
                'recall': 0.88,
                'f1_score': 0.90,
                'auc_roc': 0.94
            }
        })
        
        result = mock_trainer.train(self.normal_data, self.train_config)
        
        # Verify training results
        self.assertEqual(result['model_state'], 'trained')
        self.assertLess(result['final_loss'], 0.1)
        self.assertGreater(result['threshold'], 0)
        
        # Verify performance metrics
        perf = result['performance']
        for metric in ['precision', 'recall', 'f1_score', 'auc_roc']:
            self.assertIn(metric, perf)
            self.assertGreaterEqual(perf[metric], 0)
            self.assertLessEqual(perf[metric], 1)
            
    def test_anomaly_detection(self):
        """Test anomaly detection on new data."""
        # Simulate trained model predictions
        threshold = 0.05
        
        # Calculate reconstruction errors
        normal_errors = torch.rand(100) * 0.04  # Below threshold
        anomaly_errors = torch.rand(20) * 0.5 + 0.1  # Above threshold
        
        # Detect anomalies
        normal_predictions = normal_errors > threshold
        anomaly_predictions = anomaly_errors > threshold
        
        # Calculate detection rates
        false_positive_rate = torch.sum(normal_predictions).item() / len(normal_predictions)
        true_positive_rate = torch.sum(anomaly_predictions).item() / len(anomaly_predictions)
        
        self.assertLess(false_positive_rate, 0.1)  # Low false positives
        self.assertGreater(true_positive_rate, 0.9)  # High true positives
        
    def test_threshold_selection(self):
        """Test threshold selection strategies."""
        # Simulate reconstruction errors on validation set
        val_errors = torch.cat([
            torch.rand(900) * 0.05,  # Normal samples
            torch.rand(100) * 0.5 + 0.1  # Anomalies
        ])
        
        # Method 1: Percentile-based
        percentile_threshold = torch.quantile(val_errors, 0.95).item()
        
        # Method 2: Mean + k*std (assuming normal distribution)
        mean_error = torch.mean(val_errors[:900])  # Only normal samples
        std_error = torch.std(val_errors[:900])
        statistical_threshold = (mean_error + 3 * std_error).item()
        
        # Both thresholds should separate normal from anomalous
        self.assertGreater(percentile_threshold, 0)
        self.assertGreater(statistical_threshold, 0)
        self.assertLess(percentile_threshold, 0.5)  # Below most anomalies
        
    def test_model_validation(self):
        """Test model validation during training."""
        # Simulate validation loop
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 5
        
        # Simulate improving then plateauing validation loss
        simulated_losses = [0.5, 0.3, 0.2, 0.15, 0.14, 0.14, 0.15, 0.14, 0.14, 0.15]
        
        for epoch, val_loss in enumerate(simulated_losses):
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                break
                
        # Verify early stopping worked
        self.assertLess(len(val_losses), len(simulated_losses))
        self.assertEqual(best_val_loss, min(val_losses))
        
    def test_data_preprocessing(self):
        """Test data preprocessing for anomaly detection."""
        # Standardization
        data = torch.randn(100, self.sequence_length, self.feature_dim)
        
        # Calculate statistics
        mean = data.mean(dim=(0, 1), keepdim=True)
        std = data.std(dim=(0, 1), keepdim=True)
        
        # Standardize
        standardized = (data - mean) / (std + 1e-8)
        
        # Verify standardization
        new_mean = standardized.mean(dim=(0, 1))
        new_std = standardized.std(dim=(0, 1))
        
        self.assertTrue(torch.allclose(new_mean, torch.zeros_like(new_mean), atol=1e-6))
        self.assertTrue(torch.allclose(new_std, torch.ones_like(new_std), atol=1e-1))


if __name__ == '__main__':
    unittest.main()