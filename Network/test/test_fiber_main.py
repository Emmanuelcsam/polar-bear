#!/usr/bin/env python3
"""
Test suite for FiberOpticsSystem from fiber_main.py
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_main import FiberOpticsSystem, print_usage, main


class TestFiberOpticsSystem:
    """Test cases for FiberOpticsSystem class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.get_device.return_value = torch.device('cpu')
        mock_cfg.TENSORIZED_DATA_PATH = Path('/tmp/tensorized-data')
        mock_cfg.RESULTS_PATH = Path('/tmp/results')
        mock_cfg.CHECKPOINTS_PATH = Path('/tmp/checkpoints')
        mock_cfg.update_equation_coefficient = Mock()
        return mock_cfg
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger"""
        mock_log = Mock()
        mock_log.log_class_init = Mock()
        mock_log.log_process_start = Mock()
        mock_log.log_process_end = Mock()
        mock_log.info = Mock()
        mock_log.warning = Mock()
        mock_log.error = Mock()
        return mock_log
    
    @pytest.fixture
    @patch('fiber_main.get_config')
    @patch('fiber_main.get_logger')
    @patch('fiber_main.TensorProcessor')
    @patch('fiber_main.IntegratedAnalysisPipeline')
    @patch('fiber_main.FiberOpticsTrainer')
    @patch('fiber_main.FiberOpticsDataLoader')
    @patch('fiber_main.ReferenceDataLoader')
    def system(self, mock_ref_loader, mock_data_loader, mock_trainer, 
               mock_pipeline, mock_tensor_proc, mock_get_logger, mock_get_config,
               mock_config, mock_logger):
        """Create FiberOpticsSystem instance with mocks"""
        mock_get_config.return_value = mock_config
        mock_get_logger.return_value = mock_logger
        
        # Setup pipeline mock
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.network = Mock()
        mock_pipeline_instance.analyze_image = Mock()
        mock_pipeline_instance.export_results = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Setup trainer mock
        mock_trainer_instance = Mock()
        mock_trainer_instance.train = Mock()
        mock_trainer_instance.history = {}
        mock_trainer_instance.load_checkpoint = Mock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Setup data loader mock
        mock_data_loader_instance = Mock()
        mock_data_loader_instance.get_data_loaders = Mock()
        mock_data_loader_instance.get_streaming_loader = Mock()
        mock_data_loader.return_value = mock_data_loader_instance
        
        system = FiberOpticsSystem()
        system.integrated_pipeline = mock_pipeline_instance
        system.trainer = mock_trainer_instance
        system.data_loader = mock_data_loader_instance
        
        return system
    
    def test_initialization(self, system):
        """Test system initialization"""
        assert system is not None
        assert system.is_trained == False
        assert system.training_history == {}
        system.logger.log_class_init.assert_called_with("FiberOpticsSystem")
        system.logger.log_process_start.assert_called_with("System Initialization")
        system.logger.log_process_end.assert_called_with("System Initialization")
    
    def test_train_model(self, system):
        """Test model training"""
        # Setup mock data loaders
        train_loader = Mock()
        val_loader = Mock()
        system.data_loader.get_data_loaders.return_value = (train_loader, val_loader)
        
        # Train without checkpoint
        system.train_model(num_epochs=10)
        
        system.data_loader.get_data_loaders.assert_called_with(
            train_ratio=0.8,
            use_weighted_sampling=True
        )
        system.trainer.train.assert_called_with(
            num_epochs=10,
            train_loader=train_loader,
            val_loader=val_loader
        )
        assert system.is_trained == True
        
        # Train with checkpoint
        system.train_model(num_epochs=5, load_checkpoint='checkpoint.pth')
        system.trainer.load_checkpoint.assert_called_with('checkpoint.pth')
    
    def test_analyze_single_image(self, system):
        """Test single image analysis"""
        # Setup mock results
        mock_results = {
            'summary': {
                'final_similarity_score': 0.95,
                'meets_threshold': True,
                'primary_region': 'center',
                'anomaly_score': 0.02
            }
        }
        system.integrated_pipeline.analyze_image.return_value = mock_results
        
        # Analyze image
        results = system.analyze_single_image('test_image.jpg')
        
        system.integrated_pipeline.analyze_image.assert_called_with('test_image.jpg')
        system.integrated_pipeline.export_results.assert_called()
        assert results == mock_results
        
        # Check logging
        system.logger.info.assert_any_call('Final similarity: 0.9500')
        system.logger.info.assert_any_call('Meets threshold: True')
        system.logger.info.assert_any_call('Primary region: center')
        system.logger.info.assert_any_call('Anomaly score: 0.0200')
    
    def test_batch_process(self, system):
        """Test batch processing"""
        # Create mock image files
        with patch('fiber_main.Path') as mock_path:
            mock_folder = Mock()
            mock_files = [Mock(name=f'image{i}.jpg', stem=f'image{i}') for i in range(3)]
            for f in mock_files:
                f.__str__ = lambda self=f: self.name
            
            mock_folder.glob.return_value = mock_files
            mock_path.return_value = mock_folder
            
            # Setup analyze results
            mock_results = {
                'summary': {
                    'final_similarity_score': 0.9,
                    'meets_threshold': True,
                    'primary_region': 'center',
                    'anomaly_score': 0.01
                }
            }
            system.integrated_pipeline.analyze_image.return_value = mock_results
            
            # Process batch
            results = system.batch_process('/test/folder', max_images=2)
            
            assert len(results) == 2
            assert all('error' not in r for r in results)
            
            # Check summary logging
            system.logger.info.assert_any_call('Found 2 images to process')
            system.logger.info.assert_any_call('  Total processed: 2')
            system.logger.info.assert_any_call('  Successful: 2')
    
    def test_batch_process_with_error(self, system):
        """Test batch processing with errors"""
        with patch('fiber_main.Path') as mock_path:
            mock_folder = Mock()
            mock_files = [Mock(name='good.jpg', stem='good'), 
                         Mock(name='bad.jpg', stem='bad')]
            for f in mock_files:
                f.__str__ = lambda self=f: self.name
            
            mock_folder.glob.return_value = mock_files
            mock_path.return_value = mock_folder
            
            # Make second image fail
            system.integrated_pipeline.analyze_image.side_effect = [
                {'summary': {'final_similarity_score': 0.9, 'meets_threshold': True,
                           'primary_region': 'center', 'anomaly_score': 0.01}},
                Exception('Processing failed')
            ]
            
            results = system.batch_process('/test/folder')
            
            assert len(results) == 2
            assert 'error' not in results[0]
            assert 'error' in results[1]
            assert results[1]['error'] == 'Processing failed'
    
    def test_realtime_process(self, system):
        """Test real-time processing"""
        # Setup mock streaming loader
        mock_batch = {
            'image': torch.randn(1, 3, 224, 224)
        }
        stream_loader = [mock_batch] * 3  # Simulate 3 frames
        system.data_loader.get_streaming_loader.return_value = stream_loader
        
        # Setup network output
        mock_output = {
            'final_similarity': torch.tensor([0.95]),
            'meets_threshold': torch.tensor([1.0]),
            'anomaly_map': torch.tensor([[0.01]])
        }
        system.integrated_pipeline.network.return_value = mock_output
        
        # Test with keyboard interrupt
        with patch('time.sleep') as mock_sleep:
            mock_sleep.side_effect = [None, None, KeyboardInterrupt]
            system.realtime_process()
        
        system.data_loader.get_streaming_loader.assert_called_with(batch_size=1)
        assert system.integrated_pipeline.network.call_count >= 2
        system.logger.info.assert_any_call('Real-time processing stopped by user')
    
    def test_update_parameters(self, system):
        """Test parameter updating"""
        system.update_parameters('A', 1.5)
        
        system.config.update_equation_coefficient.assert_called_with('A', 1.5)
        system.logger.info.assert_any_call('Updating coefficient A to 1.5')
        system.logger.info.assert_any_call('Model will use updated coefficients on next analysis')
    
    def test_evaluate_performance_not_trained(self, system):
        """Test performance evaluation when model not trained"""
        system.is_trained = False
        system.evaluate_performance()
        
        system.logger.warning.assert_called_with('Model not trained yet')
    
    def test_evaluate_performance_trained(self, system):
        """Test performance evaluation when model is trained"""
        system.is_trained = True
        system.training_history = {
            'val_similarity': [0.7, 0.8, 0.85, 0.9]
        }
        
        # Setup validation loader
        val_loader = [
            {'image': torch.randn(4, 3, 224, 224)},
            {'image': torch.randn(4, 3, 224, 224)}
        ]
        system.data_loader.get_data_loaders.return_value = (None, val_loader)
        
        # Setup network outputs
        mock_outputs = {
            'final_similarity': torch.tensor([0.9, 0.91, 0.89, 0.92]),
            'meets_threshold': torch.tensor([1.0, 1.0, 0.0, 1.0]),
            'anomaly_map': torch.ones(4, 1, 224, 224) * 0.01
        }
        system.integrated_pipeline.network.return_value = mock_outputs
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0, 0.1, 0.1, 0.2]  # Simulate timing
            system.evaluate_performance()
        
        # Check metrics logging
        system.logger.info.assert_any_call('\nPerformance Metrics:')
        # Average similarity should be around 0.905
        # Threshold ratio should be 75%
        # Check improvement calculation
        assert any('Training improvement:' in str(call) for call in system.logger.info.call_args_list)


class TestMainFunction:
    """Test cases for main entry point and CLI"""
    
    def test_print_usage(self, capsys):
        """Test usage printing"""
        print_usage()
        captured = capsys.readouterr()
        
        assert "Fiber Optics Neural Network System" in captured.out
        assert "python main.py train [epochs]" in captured.out
        assert "python main.py analyze <image>" in captured.out
        assert "python main.py batch <folder>" in captured.out
        assert "python main.py realtime" in captured.out
        assert "python main.py evaluate" in captured.out
        assert "python main.py update <coef> <value>" in captured.out
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_no_args(self, mock_system_class, capsys):
        """Test main with no arguments"""
        with patch('sys.argv', ['fiber_main.py']):
            main()
        
        captured = capsys.readouterr()
        assert "FIBER OPTICS NEURAL NETWORK SYSTEM" in captured.out
        assert "Usage:" in captured.out
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_train_command(self, mock_system_class):
        """Test main with train command"""
        mock_system = Mock()
        mock_system_class.return_value = mock_system
        
        with patch('sys.argv', ['fiber_main.py', 'train', '50', 'checkpoint.pth']):
            main()
        
        mock_system.train_model.assert_called_with(num_epochs=50, load_checkpoint='checkpoint.pth')
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_analyze_command(self, mock_system_class, capsys):
        """Test main with analyze command"""
        mock_system = Mock()
        mock_results = {
            'summary': {
                'final_similarity_score': 0.95,
                'meets_threshold': True,
                'primary_region': 'center',
                'anomaly_score': 0.02
            }
        }
        mock_system.analyze_single_image.return_value = mock_results
        mock_system_class.return_value = mock_system
        
        with patch('sys.argv', ['fiber_main.py', 'analyze', 'test.jpg']):
            main()
        
        mock_system.analyze_single_image.assert_called_with('test.jpg')
        captured = capsys.readouterr()
        assert "Analysis Results for test.jpg:" in captured.out
        assert "Final Similarity: 0.9500" in captured.out
        assert "Meets Threshold: Yes" in captured.out
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_batch_command(self, mock_system_class):
        """Test main with batch command"""
        mock_system = Mock()
        mock_system_class.return_value = mock_system
        
        with patch('sys.argv', ['fiber_main.py', 'batch', '/test/folder', '10']):
            main()
        
        mock_system.batch_process.assert_called_with('/test/folder', 10)
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_realtime_command(self, mock_system_class):
        """Test main with realtime command"""
        mock_system = Mock()
        mock_system_class.return_value = mock_system
        
        with patch('sys.argv', ['fiber_main.py', 'realtime']):
            main()
        
        mock_system.realtime_process.assert_called()
    
    @patch('fiber_main.FiberOpticsSystem')
    @patch('fiber_main.Path')
    def test_main_evaluate_command(self, mock_path, mock_system_class):
        """Test main with evaluate command"""
        mock_system = Mock()
        mock_system.config.CHECKPOINTS_PATH = Path('/tmp/checkpoints')
        mock_system_class.return_value = mock_system
        
        # Mock checkpoint path existence
        mock_best_model = Mock()
        mock_best_model.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_best_model
        
        with patch('sys.argv', ['fiber_main.py', 'evaluate']):
            main()
        
        mock_system.trainer.load_checkpoint.assert_called()
        mock_system.evaluate_performance.assert_called()
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_update_command(self, mock_system_class):
        """Test main with update command"""
        mock_system = Mock()
        mock_system_class.return_value = mock_system
        
        with patch('sys.argv', ['fiber_main.py', 'update', 'a', '1.5']):
            main()
        
        mock_system.update_parameters.assert_called_with('A', 1.5)
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_unknown_command(self, mock_system_class, capsys):
        """Test main with unknown command"""
        mock_system_class.return_value = Mock()
        
        with patch('sys.argv', ['fiber_main.py', 'unknown']):
            main()
        
        captured = capsys.readouterr()
        assert "Unknown command: unknown" in captured.out
        assert "Usage:" in captured.out
    
    @patch('fiber_main.FiberOpticsSystem')
    def test_main_missing_args(self, mock_system_class, capsys):
        """Test main with missing required arguments"""
        mock_system_class.return_value = Mock()
        
        # Test analyze without image path
        with patch('sys.argv', ['fiber_main.py', 'analyze']):
            main()
        
        captured = capsys.readouterr()
        assert "Error: Please provide image path" in captured.out
        
        # Test update without args
        with patch('sys.argv', ['fiber_main.py', 'update']):
            main()
        
        captured = capsys.readouterr()
        assert "Error: Please provide coefficient and value" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])