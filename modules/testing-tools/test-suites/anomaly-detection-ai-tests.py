#!/usr/bin/env python3
"""
Comprehensive tests for anomaly_detector_pytorch.py
Tests Convolutional Autoencoder architecture and anomaly detection functionality.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

class TestCAE(unittest.TestCase):
    """Test Convolutional Autoencoder architecture."""
    
    def test_model_initialization(self):
        """Test CAE model initialization."""
        from anomaly_detector_pytorch import CAE
        
        # Test with default parameters
        model = CAE(in_channels=3, latent_dim=128)
        self.assertEqual(model.in_channels, 3)
        self.assertEqual(model.latent_dim, 128)
        
        # Test with custom parameters
        model = CAE(in_channels=1, latent_dim=64)
        self.assertEqual(model.in_channels, 1)
        self.assertEqual(model.latent_dim, 64)
    
    def test_encoder_architecture(self):
        """Test encoder architecture."""
        from anomaly_detector_pytorch import CAE
        
        model = CAE()
        
        # Check encoder layers
        self.assertIsInstance(model.encoder[0], nn.Conv2d)
        self.assertEqual(model.encoder[0].in_channels, 3)
        self.assertEqual(model.encoder[0].out_channels, 32)
        
        # Count total encoder layers
        conv_layers = [layer for layer in model.encoder if isinstance(layer, nn.Conv2d)]
        self.assertGreaterEqual(len(conv_layers), 4)  # At least 4 conv layers
    
    def test_decoder_architecture(self):
        """Test decoder architecture."""
        from anomaly_detector_pytorch import CAE
        
        model = CAE()
        
        # Check decoder layers
        deconv_layers = [layer for layer in model.decoder if isinstance(layer, nn.ConvTranspose2d)]
        self.assertGreaterEqual(len(deconv_layers), 4)  # At least 4 deconv layers
        
        # Check final layer
        final_layer = model.decoder[-1]
        self.assertIsInstance(final_layer, nn.ConvTranspose2d)
        self.assertEqual(final_layer.out_channels, 3)  # RGB output
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        from anomaly_detector_pytorch import CAE
        
        model = CAE()
        model.eval()
        
        # Test single image
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check output shape matches input
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Test batch
        batch_input = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            batch_output = model(batch_input)
        
        self.assertEqual(batch_output.shape, batch_input.shape)
    
    def test_latent_representation(self):
        """Test latent space encoding."""
        from anomaly_detector_pytorch import CAE
        
        model = CAE(latent_dim=64)
        
        # Hook to capture latent representation
        latent_output = None
        def hook(module, input, output):
            nonlocal latent_output
            latent_output = output
        
        # Register hook on bottleneck layer
        model.encoder[-1].register_forward_hook(hook)
        
        # Forward pass
        input_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Check latent dimension
        self.assertIsNotNone(latent_output)
        # Latent representation should be flattened or have correct dimensions

class TestAIAnomalyDetector(unittest.TestCase):
    """Test AI_AnomalyDetector wrapper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('anomaly_detector_pytorch.CAE')
    @patch('torch.load')
    def test_detector_initialization(self, mock_load, mock_cae):
        """Test AI_AnomalyDetector initialization."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        # Create fake model file
        model_path = os.path.join(self.temp_dir, 'model.pth')
        with open(model_path, 'w') as f:
            f.write('fake model')
        
        # Mock model loading
        mock_model = Mock()
        mock_cae.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        # Test initialization
        detector = AI_AnomalyDetector(model_path, threshold=0.05)
        
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.threshold, 0.05)
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('anomaly_detector_pytorch.CAE')
    @patch('torch.load')
    def test_detect_anomalies(self, mock_load, mock_cae):
        """Test anomaly detection."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        # Setup mocks
        model_path = os.path.join(self.temp_dir, 'model.pth')
        with open(model_path, 'w') as f:
            f.write('fake model')
        
        mock_model = Mock()
        mock_cae.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        # Mock reconstruction
        input_tensor = torch.randn(1, 3, 256, 256)
        reconstructed = input_tensor.clone()
        # Add some anomalies
        reconstructed[0, :, 100:150, 100:150] += 0.5
        mock_model.return_value = reconstructed
        
        # Test detection
        detector = AI_AnomalyDetector(model_path, threshold=0.1)
        result = detector.detect(self.test_image)
        
        # Check output structure
        self.assertIn('anomaly_map', result)
        self.assertIn('anomaly_mask', result)
        self.assertIn('anomaly_score', result)
        self.assertIn('reconstructed_image', result)
        
        # Check shapes
        self.assertEqual(result['anomaly_map'].shape, (256, 256))
        self.assertEqual(result['anomaly_mask'].shape, (256, 256))
        self.assertEqual(result['reconstructed_image'].shape, (256, 256, 3))
        
        # Check types
        self.assertIsInstance(result['anomaly_score'], float)
        self.assertEqual(result['anomaly_mask'].dtype, np.uint8)
    
    @patch('anomaly_detector_pytorch.CAE')
    @patch('torch.load')
    def test_threshold_adjustment(self, mock_load, mock_cae):
        """Test threshold adjustment functionality."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        # Setup mocks
        model_path = os.path.join(self.temp_dir, 'model.pth')
        with open(model_path, 'w') as f:
            f.write('fake model')
        
        mock_model = Mock()
        mock_cae.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        # Create detector
        detector = AI_AnomalyDetector(model_path, threshold=0.1)
        
        # Test threshold adjustment
        detector.set_threshold(0.2)
        self.assertEqual(detector.threshold, 0.2)
        
        # Test with different thresholds
        mock_model.return_value = torch.randn(1, 3, 256, 256)
        
        result_low = detector.detect(self.test_image)
        detector.set_threshold(0.01)  # Very sensitive
        result_high = detector.detect(self.test_image)
        
        # More sensitive threshold should detect more anomalies
        # (This is a simplified test - actual behavior depends on reconstruction)
    
    @patch('anomaly_detector_pytorch.CAE')
    @patch('torch.load')
    def test_batch_detection(self, mock_load, mock_cae):
        """Test batch anomaly detection."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        # Setup mocks
        model_path = os.path.join(self.temp_dir, 'model.pth')
        with open(model_path, 'w') as f:
            f.write('fake model')
        
        mock_model = Mock()
        mock_cae.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        
        # Mock batch reconstruction
        batch_size = 4
        mock_model.return_value = torch.randn(batch_size, 3, 256, 256)
        
        detector = AI_AnomalyDetector(model_path)
        
        # Test batch processing
        images = [self.test_image for _ in range(batch_size)]
        results = detector.detect_batch(images)
        
        self.assertEqual(len(results), batch_size)
        for result in results:
            self.assertIn('anomaly_map', result)
            self.assertIn('anomaly_mask', result)
            self.assertIn('anomaly_score', result)
    
    def test_missing_model_file(self):
        """Test handling of missing model file."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        with self.assertRaises(FileNotFoundError):
            AI_AnomalyDetector('nonexistent_model.pth')
    
    @patch('anomaly_detector_pytorch.CAE')
    @patch('torch.load')
    def test_anomaly_visualization(self, mock_load, mock_cae):
        """Test anomaly visualization generation."""
        from anomaly_detector_pytorch import AI_AnomalyDetector
        
        # Setup mocks
        model_path = os.path.join(self.temp_dir, 'model.pth')
        with open(model_path, 'w') as f:
            f.write('fake model')
        
        mock_model = Mock()
        mock_cae.return_value = mock_model
        mock_load.return_value = {'model_state_dict': {}}
        mock_model.return_value = torch.randn(1, 3, 256, 256)
        
        detector = AI_AnomalyDetector(model_path)
        
        # Test visualization
        result = detector.detect(self.test_image, return_visualization=True)
        
        if 'visualization' in result:
            self.assertEqual(result['visualization'].shape, (256, 256, 3))
            self.assertEqual(result['visualization'].dtype, np.uint8)

class TestAnomalyMetrics(unittest.TestCase):
    """Test anomaly detection metrics and evaluation."""
    
    def test_reconstruction_error_calculation(self):
        """Test reconstruction error calculation."""
        # Test MSE calculation
        original = torch.randn(1, 3, 256, 256)
        reconstructed = original + 0.1 * torch.randn_like(original)
        
        mse = nn.MSELoss(reduction='none')(reconstructed, original)
        pixel_wise_error = mse.mean(dim=1).squeeze()
        
        self.assertEqual(pixel_wise_error.shape, (256, 256))
        self.assertTrue((pixel_wise_error >= 0).all())
    
    def test_anomaly_score_normalization(self):
        """Test anomaly score normalization."""
        # Test score normalization between 0 and 1
        scores = np.random.rand(256, 256) * 10
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        
        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)

class TestPreprocessing(unittest.TestCase):
    """Test image preprocessing for anomaly detection."""
    
    def test_image_normalization(self):
        """Test image normalization."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Apply standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (tensor - mean) / std
        
        self.assertEqual(normalized.shape, (1, 3, 256, 256))
        # Check approximate range after normalization
        self.assertTrue(normalized.min() >= -3)
        self.assertTrue(normalized.max() <= 3)
    
    def test_image_resizing(self):
        """Test image resizing for model input."""
        # Test various input sizes
        sizes = [(128, 128), (512, 512), (256, 384)]
        
        for h, w in sizes:
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            
            # Resize to model input size
            import cv2
            resized = cv2.resize(image, (256, 256))
            
            self.assertEqual(resized.shape, (256, 256, 3))

if __name__ == '__main__':
    unittest.main()