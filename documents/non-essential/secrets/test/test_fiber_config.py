#!/usr/bin/env python3
"""
Test suite for FiberOpticsConfig from fiber_config.py
"""

import pytest
import torch
import json
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_config import FiberOpticsConfig, get_config


class TestFiberOpticsConfig:
    """Test cases for FiberOpticsConfig class"""
    
    @pytest.fixture
    def config(self):
        """Create config instance"""
        with patch('fiber_config.Path.mkdir'):
            config = FiberOpticsConfig()
        return config
    
    def test_initialization(self, config):
        """Test config initialization"""
        assert config is not None
        assert hasattr(config, 'PROJECT_ROOT')
        assert hasattr(config, 'DATA_PATH')
        assert hasattr(config, 'TENSORIZED_DATA_PATH')
        assert hasattr(config, 'REFERENCE_PATH')
        assert hasattr(config, 'RESULTS_PATH')
        assert hasattr(config, 'CHECKPOINTS_PATH')
        assert hasattr(config, 'LOGS_PATH')
    
    def test_paths(self, config):
        """Test path configurations"""
        assert config.TENSORIZED_DATA_PATH == config.DATA_PATH / "tensorized-data"
        assert config.REFERENCE_PATH == config.DATA_PATH / "reference-images"
        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.RESULTS_PATH, Path)
    
    @patch('torch.cuda.is_available')
    def test_device_configuration_cuda(self, mock_cuda, config):
        """Test device configuration with CUDA available"""
        mock_cuda.return_value = True
        new_config = FiberOpticsConfig()
        
        assert new_config.DEVICE == torch.device("cuda")
        assert new_config.NUM_WORKERS == 4
        assert new_config.PIN_MEMORY == True
    
    @patch('torch.cuda.is_available')
    def test_device_configuration_cpu(self, mock_cuda, config):
        """Test device configuration with CPU only"""
        mock_cuda.return_value = False
        new_config = FiberOpticsConfig()
        
        assert new_config.DEVICE == torch.device("cpu")
        assert new_config.NUM_WORKERS == 0
        assert new_config.PIN_MEMORY == False
    
    def test_equation_coefficients(self, config):
        """Test equation coefficients initialization"""
        assert 'A' in config.EQUATION_COEFFICIENTS
        assert 'B' in config.EQUATION_COEFFICIENTS
        assert 'C' in config.EQUATION_COEFFICIENTS
        assert 'D' in config.EQUATION_COEFFICIENTS
        assert 'E' in config.EQUATION_COEFFICIENTS
        
        assert config.EQUATION_COEFFICIENTS['A'] == 1.0
        assert config.EQUATION_COEFFICIENTS['B'] == 1.0
        assert config.EQUATION_COEFFICIENTS['C'] == 0.0  # Commented out as requested
        assert config.EQUATION_COEFFICIENTS['D'] == 1.0
        assert config.EQUATION_COEFFICIENTS['E'] == 1.0
    
    def test_gradient_weights(self, config):
        """Test gradient weight configurations"""
        assert config.GRADIENT_WEIGHT_FACTOR == 1.0
        assert config.POSITION_WEIGHT_FACTOR == 1.0
    
    def test_threshold_parameters(self, config):
        """Test threshold parameters"""
        assert hasattr(config, 'SIMILARITY_THRESHOLD')
        assert config.SIMILARITY_THRESHOLD == 0.8
        assert hasattr(config, 'QUALITY_THRESHOLD')
        assert config.QUALITY_THRESHOLD == 0.6
        assert hasattr(config, 'ANOMALY_THRESHOLD')
        assert config.ANOMALY_THRESHOLD == 0.3
    
    def test_training_parameters(self, config):
        """Test training parameters"""
        assert config.BATCH_SIZE == 32
        assert config.LEARNING_RATE == 0.001
        assert config.WEIGHT_DECAY == 1e-5
        assert config.GRADIENT_CLIP == 1.0
        assert config.WARMUP_EPOCHS == 5
    
    def test_feature_extraction_parameters(self, config):
        """Test feature extraction parameters"""
        assert config.FEATURE_DIM == 512
        assert config.NUM_FEATURES == 2048
        assert config.FEATURE_SCALES == [1, 2, 4, 8]
        assert config.EDGE_THRESHOLD == 50
    
    def test_image_processing_parameters(self, config):
        """Test image processing parameters"""
        assert config.IMAGE_SIZE == (224, 224)
        assert config.NORMALIZE_MEAN == [0.485, 0.456, 0.406]
        assert config.NORMALIZE_STD == [0.229, 0.224, 0.225]
        assert config.AUGMENTATION_PROB == 0.5
    
    def test_anomaly_detection_parameters(self, config):
        """Test anomaly detection parameters"""
        assert config.ANOMALY_KERNEL_SIZE == 5
        assert config.ANOMALY_SIGMA == 1.0
        assert config.MIN_ANOMALY_SIZE == 10
    
    def test_real_time_parameters(self, config):
        """Test real-time processing parameters"""
        assert config.FRAME_BUFFER_SIZE == 30
        assert config.FPS_TARGET == 30
        assert config.LATENCY_TARGET == 0.033  # ~30ms
    
    def test_get_device(self, config):
        """Test get_device method"""
        device = config.get_device()
        assert isinstance(device, torch.device)
        assert device == config.DEVICE
    
    def test_update_equation_coefficient(self, config):
        """Test updating equation coefficients"""
        # Update existing coefficient
        config.update_equation_coefficient('A', 2.5)
        assert config.EQUATION_COEFFICIENTS['A'] == 2.5
        
        # Update new coefficient
        config.update_equation_coefficient('F', 0.5)
        assert config.EQUATION_COEFFICIENTS['F'] == 0.5
    
    def test_get_coefficient(self, config):
        """Test getting equation coefficients"""
        assert config.get_coefficient('A') == 1.0
        assert config.get_coefficient('B') == 1.0
        
        # Test non-existent coefficient
        assert config.get_coefficient('Z') == 0.0
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_save_config(self, mock_json_dump, mock_open, config):
        """Test saving configuration"""
        config.save_config('test_config.json')
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Check saved data structure
        saved_data = mock_json_dump.call_args[0][0]
        assert 'equation_coefficients' in saved_data
        assert 'thresholds' in saved_data
        assert 'training' in saved_data
        assert 'image_processing' in saved_data
    
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_load_config(self, mock_json_load, mock_open, config):
        """Test loading configuration"""
        mock_config_data = {
            'equation_coefficients': {'A': 2.0, 'B': 1.5},
            'thresholds': {'similarity': 0.9},
            'training': {'batch_size': 64}
        }
        mock_json_load.return_value = mock_config_data
        
        config.load_config('test_config.json')
        
        mock_open.assert_called_once()
        assert config.EQUATION_COEFFICIENTS['A'] == 2.0
        assert config.EQUATION_COEFFICIENTS['B'] == 1.5
        assert config.SIMILARITY_THRESHOLD == 0.9
        assert config.BATCH_SIZE == 64
    
    def test_str_representation(self, config):
        """Test string representation"""
        str_repr = str(config)
        assert "FiberOpticsConfig" in str_repr
        assert "Device:" in str_repr
        assert "Coefficients:" in str_repr
        assert "Paths:" in str_repr


class TestGetConfigFunction:
    """Test cases for get_config singleton function"""
    
    def test_singleton_pattern(self):
        """Test that get_config returns same instance"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_config_attributes(self):
        """Test that singleton config has all attributes"""
        config = get_config()
        
        assert hasattr(config, 'PROJECT_ROOT')
        assert hasattr(config, 'DEVICE')
        assert hasattr(config, 'EQUATION_COEFFICIENTS')
        assert hasattr(config, 'get_device')
        assert hasattr(config, 'update_equation_coefficient')


class TestConfigIntegration:
    """Integration tests for config usage"""
    
    @pytest.fixture
    def fresh_config(self):
        """Create fresh config by clearing singleton"""
        import fiber_config
        fiber_config._config = None
        config = get_config()
        yield config
        fiber_config._config = None
    
    def test_config_persistence(self, fresh_config):
        """Test that config changes persist"""
        config1 = get_config()
        config1.update_equation_coefficient('A', 3.0)
        
        config2 = get_config()
        assert config2.get_coefficient('A') == 3.0
    
    @patch('fiber_config.Path.exists')
    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_auto_load_config(self, mock_json_load, mock_open, mock_exists):
        """Test automatic config loading if file exists"""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            'equation_coefficients': {'A': 5.0}
        }
        
        import fiber_config
        fiber_config._config = None
        
        config = get_config()
        # Would check if auto-load is implemented
    
    def test_config_validation(self, fresh_config):
        """Test config validation"""
        # Test invalid coefficient update
        with pytest.raises(ValueError):
            fresh_config.update_equation_coefficient('A', 'invalid')
        
        # Test negative learning rate
        with pytest.raises(ValueError):
            fresh_config.LEARNING_RATE = -0.001
            fresh_config.validate()
    
    def test_config_export_import(self, fresh_config, tmp_path):
        """Test full config export and import cycle"""
        # Modify config
        fresh_config.update_equation_coefficient('A', 2.5)
        fresh_config.BATCH_SIZE = 64
        fresh_config.LEARNING_RATE = 0.002
        
        # Export
        config_file = tmp_path / "test_config.json"
        fresh_config.save_config(str(config_file))
        
        # Create new config and import
        import fiber_config
        fiber_config._config = None
        new_config = get_config()
        new_config.load_config(str(config_file))
        
        # Verify
        assert new_config.get_coefficient('A') == 2.5
        assert new_config.BATCH_SIZE == 64
        assert new_config.LEARNING_RATE == 0.002


if __name__ == '__main__':
    pytest.main([__file__, '-v'])