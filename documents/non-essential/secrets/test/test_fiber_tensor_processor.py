#!/usr/bin/env python3
"""
Test suite for TensorProcessor from fiber_tensor_processor.py
"""

import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_tensor_processor import TensorProcessor


class TestTensorProcessor:
    """Test cases for TensorProcessor class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.IMAGE_SIZE = (224, 224)
        mock_cfg.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        mock_cfg.NORMALIZE_STD = [0.229, 0.224, 0.225]
        mock_cfg.GRADIENT_WEIGHT_FACTOR = 1.0
        mock_cfg.POSITION_WEIGHT_FACTOR = 1.0
        mock_cfg.EDGE_THRESHOLD = 50
        mock_cfg.get_device.return_value = torch.device('cpu')
        return mock_cfg
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger"""
        mock_log = Mock()
        mock_log.log_class_init = Mock()
        mock_log.log_tensor_operation = Mock()
        mock_log.info = Mock()
        mock_log.debug = Mock()
        mock_log.error = Mock()
        return mock_log
    
    @pytest.fixture
    @patch('fiber_tensor_processor.get_config')
    @patch('fiber_tensor_processor.get_logger')
    def processor(self, mock_get_logger, mock_get_config, mock_config, mock_logger):
        """Create TensorProcessor instance with mocks"""
        mock_get_config.return_value = mock_config
        mock_get_logger.return_value = mock_logger
        
        processor = TensorProcessor()
        return processor
    
    def test_initialization(self, processor, mock_logger):
        """Test processor initialization"""
        assert processor is not None
        mock_logger.log_class_init.assert_called_with("TensorProcessor")
        assert hasattr(processor, 'config')
        assert hasattr(processor, 'logger')
        assert hasattr(processor, 'device')
        assert hasattr(processor, 'transform')
    
    @patch('cv2.imread')
    def test_load_image_success(self, mock_imread, processor):
        """Test successful image loading"""
        # Mock image data
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        
        result = processor.load_image("test.jpg")
        
        mock_imread.assert_called_with("test.jpg")
        assert result.shape == mock_image.shape
        processor.logger.log_tensor_operation.assert_called_with(
            "load_image", "test.jpg", mock_image.shape
        )
    
    @patch('cv2.imread')
    def test_load_image_failure(self, mock_imread, processor):
        """Test image loading failure"""
        mock_imread.return_value = None
        
        with pytest.raises(ValueError, match="Failed to load image"):
            processor.load_image("invalid.jpg")
        
        processor.logger.error.assert_called()
    
    def test_image_to_tensor(self, processor):
        """Test image to tensor conversion"""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        tensor = processor.image_to_tensor(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Based on config IMAGE_SIZE
        assert tensor.dtype == torch.float32
        processor.logger.log_tensor_operation.assert_called()
    
    @patch('torch.load')
    def test_load_tensor_success(self, mock_load, processor):
        """Test successful tensor loading"""
        mock_tensor = torch.randn(1, 3, 224, 224)
        mock_load.return_value = mock_tensor
        
        result = processor.load_tensor("test.pt")
        
        mock_load.assert_called_with("test.pt", map_location=processor.device)
        assert torch.equal(result, mock_tensor)
        processor.logger.log_tensor_operation.assert_called()
    
    @patch('torch.load')
    def test_load_tensor_failure(self, mock_load, processor):
        """Test tensor loading failure"""
        mock_load.side_effect = Exception("Load failed")
        
        with pytest.raises(ValueError, match="Failed to load tensor"):
            processor.load_tensor("invalid.pt")
        
        processor.logger.error.assert_called()
    
    @patch('torch.save')
    def test_save_tensor(self, mock_save, processor):
        """Test tensor saving"""
        tensor = torch.randn(1, 3, 224, 224)
        
        processor.save_tensor(tensor, "output.pt")
        
        mock_save.assert_called_with(tensor, "output.pt")
        processor.logger.log_tensor_operation.assert_called_with(
            "save_tensor", tensor.shape, "output.pt"
        )
    
    def test_normalize_tensor(self, processor):
        """Test tensor normalization"""
        # Create tensor with known values
        tensor = torch.ones(1, 3, 224, 224) * 0.5
        
        normalized = processor.normalize_tensor(tensor)
        
        # Check shape preserved
        assert normalized.shape == tensor.shape
        
        # Check normalization applied
        expected_c0 = (0.5 - 0.485) / 0.229
        assert torch.allclose(normalized[0, 0, 0, 0], torch.tensor(expected_c0), atol=1e-6)
    
    def test_denormalize_tensor(self, processor):
        """Test tensor denormalization"""
        # Create normalized tensor
        normalized = torch.randn(1, 3, 224, 224)
        
        denormalized = processor.denormalize_tensor(normalized)
        
        # Check shape preserved
        assert denormalized.shape == normalized.shape
        
        # Verify inverse operation
        renormalized = processor.normalize_tensor(denormalized)
        assert torch.allclose(normalized, renormalized, atol=1e-5)
    
    def test_calculate_gradient_weights(self, processor):
        """Test gradient weight calculation"""
        tensor = torch.randn(1, 3, 224, 224)
        
        weights = processor.calculate_gradient_weights(tensor)
        
        assert weights.shape == (1, 1, 224, 224)
        assert weights.min() >= 0
        assert weights.max() <= 1
        processor.logger.log_tensor_operation.assert_called_with(
            "gradient_weights", tensor.shape, weights.shape
        )
    
    def test_calculate_position_weights(self, processor):
        """Test position weight calculation"""
        shape = (224, 224)
        
        weights = processor.calculate_position_weights(shape)
        
        assert weights.shape == (1, 1, 224, 224)
        assert weights.min() >= 0
        assert weights.max() <= 1
        
        # Check center has higher weight
        center_weight = weights[0, 0, 112, 112].item()
        corner_weight = weights[0, 0, 0, 0].item()
        assert center_weight > corner_weight
    
    def test_extract_edges(self, processor):
        """Test edge extraction"""
        tensor = torch.randn(1, 3, 224, 224)
        
        edges = processor.extract_edges(tensor)
        
        assert edges.shape == (1, 1, 224, 224)
        assert edges.min() >= 0
        assert edges.max() <= 1
        processor.logger.log_tensor_operation.assert_called_with(
            "extract_edges", tensor.shape, edges.shape
        )
    
    def test_apply_equation_weights(self, processor, mock_config):
        """Test equation weight application"""
        tensor = torch.randn(1, 3, 224, 224)
        
        # Setup config coefficients
        mock_config.get_coefficient = Mock(side_effect=lambda x: {
            'A': 1.0, 'B': 0.5, 'D': 0.3
        }.get(x, 0.0))
        
        weighted = processor.apply_equation_weights(tensor)
        
        assert weighted.shape == tensor.shape
        processor.logger.debug.assert_called()
    
    def test_tensorize_image(self, processor):
        """Test complete image tensorization"""
        with patch.object(processor, 'load_image') as mock_load:
            with patch.object(processor, 'image_to_tensor') as mock_to_tensor:
                with patch.object(processor, 'apply_equation_weights') as mock_weights:
                    # Setup mocks
                    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    mock_tensor = torch.randn(1, 3, 224, 224)
                    mock_weighted = torch.randn(1, 3, 224, 224)
                    
                    mock_load.return_value = mock_image
                    mock_to_tensor.return_value = mock_tensor
                    mock_weights.return_value = mock_weighted
                    
                    # Test with weights
                    result = processor.tensorize_image("test.jpg", apply_weights=True)
                    assert torch.equal(result, mock_weighted)
                    mock_weights.assert_called_once()
                    
                    # Test without weights
                    result = processor.tensorize_image("test.jpg", apply_weights=False)
                    assert torch.equal(result, mock_tensor)
    
    def test_batch_tensorize(self, processor):
        """Test batch tensorization"""
        with patch.object(processor, 'tensorize_image') as mock_tensorize:
            # Setup mock tensors
            tensors = [torch.randn(1, 3, 224, 224) for _ in range(3)]
            mock_tensorize.side_effect = tensors
            
            image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
            batch = processor.batch_tensorize(image_paths)
            
            assert batch.shape == (3, 3, 224, 224)
            assert mock_tensorize.call_count == 3
            processor.logger.info.assert_called_with(
                f"Batch tensorized {len(image_paths)} images"
            )
    
    def test_tensor_to_image(self, processor):
        """Test tensor to image conversion"""
        tensor = torch.randn(1, 3, 224, 224)
        
        image = processor.tensor_to_image(tensor)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8
        assert image.min() >= 0
        assert image.max() <= 255
    
    def test_resize_tensor(self, processor):
        """Test tensor resizing"""
        tensor = torch.randn(2, 3, 224, 224)
        new_size = (112, 112)
        
        resized = processor.resize_tensor(tensor, new_size)
        
        assert resized.shape == (2, 3, 112, 112)
        processor.logger.log_tensor_operation.assert_called_with(
            "resize", tensor.shape, resized.shape
        )
    
    def test_augment_tensor(self, processor):
        """Test tensor augmentation"""
        tensor = torch.randn(1, 3, 224, 224)
        
        # Test different augmentations
        augmentations = ['flip', 'rotate', 'noise', 'brightness', 'invalid']
        
        for aug in augmentations[:-1]:
            augmented = processor.augment_tensor(tensor.clone(), aug)
            assert augmented.shape == tensor.shape
            # Ensure tensor was modified (except for some edge cases)
            if aug != 'rotate':  # Rotate with 0 degrees might not change
                assert not torch.equal(augmented, tensor)
        
        # Test invalid augmentation
        result = processor.augment_tensor(tensor.clone(), 'invalid')
        assert torch.equal(result, tensor)  # Should return original
    
    def test_get_tensor_statistics(self, processor):
        """Test tensor statistics calculation"""
        tensor = torch.randn(2, 3, 224, 224)
        
        stats = processor.get_tensor_statistics(tensor)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'shape' in stats
        
        assert stats['shape'] == list(tensor.shape)
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)


class TestTensorProcessorIntegration:
    """Integration tests for TensorProcessor"""
    
    @pytest.fixture
    def processor(self):
        """Create real processor instance"""
        return TensorProcessor()
    
    def test_full_processing_pipeline(self, processor, tmp_path):
        """Test complete processing pipeline"""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)
        
        # Process image
        tensor = processor.tensorize_image(str(image_path), apply_weights=True)
        
        # Verify tensor properties
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        
        # Save and reload tensor
        tensor_path = tmp_path / "test.pt"
        processor.save_tensor(tensor, str(tensor_path))
        loaded_tensor = processor.load_tensor(str(tensor_path))
        
        assert torch.equal(tensor, loaded_tensor)
        
        # Convert back to image
        recovered_image = processor.tensor_to_image(loaded_tensor)
        assert recovered_image.shape == (224, 224, 3)
    
    def test_batch_processing(self, processor, tmp_path):
        """Test batch processing capabilities"""
        # Create multiple test images
        image_paths = []
        for i in range(5):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            path = tmp_path / f"test_{i}.jpg"
            cv2.imwrite(str(path), image)
            image_paths.append(str(path))
        
        # Batch process
        batch = processor.batch_tensorize(image_paths)
        
        assert batch.shape == (5, 3, 224, 224)
        
        # Test individual processing matches batch
        individual_tensors = []
        for path in image_paths:
            tensor = processor.tensorize_image(path)
            individual_tensors.append(tensor)
        
        stacked = torch.cat(individual_tensors, dim=0)
        assert torch.allclose(batch, stacked, atol=1e-5)
    
    def test_augmentation_pipeline(self, processor):
        """Test augmentation effects"""
        # Create base tensor
        base_tensor = torch.ones(1, 3, 224, 224) * 0.5
        
        # Apply various augmentations
        flipped = processor.augment_tensor(base_tensor.clone(), 'flip')
        rotated = processor.augment_tensor(base_tensor.clone(), 'rotate')
        noisy = processor.augment_tensor(base_tensor.clone(), 'noise')
        bright = processor.augment_tensor(base_tensor.clone(), 'brightness')
        
        # Verify augmentations created different tensors
        tensors = [base_tensor, flipped, rotated, noisy, bright]
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                if i == 0 or j == 2:  # Base vs others, or rotated (might be 0 degrees)
                    continue
                assert not torch.equal(tensors[i], tensors[j])
    
    def test_edge_extraction_quality(self, processor):
        """Test edge extraction on synthetic image"""
        # Create image with clear edges (checkerboard pattern)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 32):
            for j in range(0, 224, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    image[i:i+32, j:j+32] = 255
        
        # Convert to tensor
        tensor = processor.image_to_tensor(image)
        
        # Extract edges
        edges = processor.extract_edges(tensor)
        
        # Edges should be high at boundaries
        edge_image = edges[0, 0].numpy()
        
        # Check some edge locations
        assert edge_image[31, 16] > 0.1  # Horizontal edge
        assert edge_image[16, 31] > 0.1  # Vertical edge
        assert edge_image[16, 16] < 0.1  # Center of square (no edge)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])