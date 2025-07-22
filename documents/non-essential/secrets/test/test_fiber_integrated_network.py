#!/usr/bin/env python3
"""
Test suite for IntegratedAnalysisPipeline and FiberOpticsIntegratedNetwork
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_integrated_network import (
    FiberOpticsIntegratedNetwork, 
    IntegratedAnalysisPipeline,
    EquationAdjuster,
    RegionClassifier
)


class TestEquationAdjuster:
    """Test cases for EquationAdjuster module"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.get_coefficient = Mock(side_effect=lambda x: {
            'A': 1.0, 'B': 0.5, 'D': 0.8, 'E': 0.3
        }.get(x, 0.0))
        return mock_cfg
    
    @pytest.fixture
    @patch('fiber_integrated_network.get_config')
    def adjuster(self, mock_get_config, mock_config):
        """Create EquationAdjuster instance"""
        mock_get_config.return_value = mock_config
        adjuster = EquationAdjuster()
        return adjuster
    
    def test_initialization(self, adjuster):
        """Test adjuster initialization"""
        assert adjuster is not None
        assert hasattr(adjuster, 'config')
        assert hasattr(adjuster, 'coefficient_layers')
    
    def test_forward(self, adjuster):
        """Test forward pass"""
        features = {
            'gradient': torch.randn(2, 512),
            'position': torch.randn(2, 512),
            'correlation': torch.randn(2, 512),
            'anomaly': torch.randn(2, 512)
        }
        
        output = adjuster(features)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 512)
    
    def test_coefficient_application(self, adjuster, mock_config):
        """Test that coefficients are properly applied"""
        # Create features with known values
        features = {
            'gradient': torch.ones(1, 512),
            'position': torch.ones(1, 512),
            'correlation': torch.ones(1, 512),
            'anomaly': torch.ones(1, 512)
        }
        
        output = adjuster(features)
        
        # Output should be weighted sum
        # A*gradient + B*position + D*correlation + E*anomaly
        # = 1.0*1 + 0.5*1 + 0.8*1 + 0.3*1 = 2.6
        expected_sum = 1.0 + 0.5 + 0.8 + 0.3
        assert torch.allclose(output.mean(), torch.tensor(expected_sum), atol=0.1)


class TestRegionClassifier:
    """Test cases for RegionClassifier module"""
    
    @pytest.fixture
    def classifier(self):
        """Create RegionClassifier instance"""
        return RegionClassifier(num_regions=5)
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert classifier.num_regions == 5
        assert hasattr(classifier, 'conv1')
        assert hasattr(classifier, 'fc')
    
    def test_forward(self, classifier):
        """Test forward pass"""
        features = torch.randn(2, 512, 14, 14)
        
        output = classifier(features)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 5)
        
        # Check probabilities sum to 1
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)


class TestFiberOpticsIntegratedNetwork:
    """Test cases for FiberOpticsIntegratedNetwork"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.SIMILARITY_THRESHOLD = 0.8
        mock_cfg.FEATURE_DIM = 512
        mock_cfg.get_device.return_value = torch.device('cpu')
        mock_cfg.get_coefficient = Mock(return_value=1.0)
        return mock_cfg
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components"""
        mocks = {
            'feature_extractor': Mock(),
            'trend_analyzer': Mock(),
            'similarity_calculator': Mock(),
            'anomaly_detector': Mock()
        }
        
        # Setup return values
        mocks['feature_extractor'].return_value = {
            'features': torch.randn(2, 512, 14, 14),
            'edges': torch.randn(2, 1, 224, 224),
            'gradients': torch.randn(2, 1, 224, 224)
        }
        
        mocks['trend_analyzer'].return_value = {
            'trend_features': torch.randn(2, 512),
            'trend_direction': torch.randn(2, 3)
        }
        
        mocks['similarity_calculator'].return_value = {
            'similarity_scores': torch.tensor([0.85, 0.75]),
            'region_similarities': torch.randn(2, 5)
        }
        
        mocks['anomaly_detector'].return_value = {
            'anomaly_map': torch.randn(2, 1, 224, 224),
            'anomaly_scores': torch.tensor([0.1, 0.3])
        }
        
        return mocks
    
    @pytest.fixture
    @patch('fiber_integrated_network.get_config')
    @patch('fiber_integrated_network.get_logger')
    @patch('fiber_integrated_network.MultiScaleFeatureExtractor')
    @patch('fiber_integrated_network.TrendAnalyzer')
    @patch('fiber_integrated_network.SimilarityCalculator')
    @patch('fiber_integrated_network.AnomalyDetector')
    def network(self, mock_anomaly_class, mock_similarity_class, mock_trend_class,
                mock_feature_class, mock_get_logger, mock_get_config, 
                mock_config, mock_components):
        """Create network instance with mocks"""
        mock_get_config.return_value = mock_config
        mock_get_logger.return_value = Mock()
        
        # Setup component classes
        mock_feature_class.return_value = mock_components['feature_extractor']
        mock_trend_class.return_value = mock_components['trend_analyzer']
        mock_similarity_class.return_value = mock_components['similarity_calculator']
        mock_anomaly_class.return_value = mock_components['anomaly_detector']
        
        network = FiberOpticsIntegratedNetwork()
        return network
    
    def test_initialization(self, network):
        """Test network initialization"""
        assert network is not None
        assert hasattr(network, 'feature_extractor')
        assert hasattr(network, 'trend_analyzer')
        assert hasattr(network, 'similarity_calculator')
        assert hasattr(network, 'anomaly_detector')
        assert hasattr(network, 'equation_adjuster')
        assert hasattr(network, 'region_classifier')
        assert hasattr(network, 'integration_layer')
    
    def test_forward(self, network):
        """Test forward pass"""
        x = torch.randn(2, 3, 224, 224)
        
        output = network(x)
        
        assert isinstance(output, dict)
        assert 'features' in output
        assert 'edges' in output
        assert 'similarity_scores' in output
        assert 'anomaly_map' in output
        assert 'final_similarity' in output
        assert 'meets_threshold' in output
        assert 'regions' in output
    
    def test_output_shapes(self, network):
        """Test output tensor shapes"""
        x = torch.randn(2, 3, 224, 224)
        output = network(x)
        
        assert output['final_similarity'].shape == (2,)
        assert output['meets_threshold'].shape == (2,)
        assert output['regions'].shape == (2, 5)  # 5 regions
        assert output['anomaly_map'].shape == (2, 1, 224, 224)
    
    def test_threshold_application(self, network, mock_config):
        """Test similarity threshold application"""
        x = torch.randn(2, 3, 224, 224)
        
        # Set known similarity scores
        network.similarity_calculator.return_value = {
            'similarity_scores': torch.tensor([0.9, 0.7]),  # Above and below threshold
            'region_similarities': torch.randn(2, 5)
        }
        
        output = network(x)
        
        assert output['meets_threshold'][0] == 1.0  # 0.9 > 0.8
        assert output['meets_threshold'][1] == 0.0  # 0.7 < 0.8
    
    def test_training_mode(self, network):
        """Test network in training mode"""
        network.train()
        assert network.training
        
        x = torch.randn(2, 3, 224, 224)
        output = network(x)
        
        # Should still produce all outputs
        assert 'final_similarity' in output
    
    def test_eval_mode(self, network):
        """Test network in evaluation mode"""
        network.eval()
        assert not network.training
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = network(x)
        
        # Should produce all outputs without gradients
        assert 'final_similarity' in output
        assert not output['final_similarity'].requires_grad


class TestIntegratedAnalysisPipeline:
    """Test cases for IntegratedAnalysisPipeline"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.get_device.return_value = torch.device('cpu')
        mock_cfg.RESULTS_PATH = Path('/tmp/results')
        return mock_cfg
    
    @pytest.fixture
    def mock_network(self):
        """Create mock network"""
        mock_net = Mock(spec=FiberOpticsIntegratedNetwork)
        mock_net.eval = Mock()
        mock_net.to = Mock(return_value=mock_net)
        
        # Mock network output
        mock_net.return_value = {
            'features': torch.randn(1, 512, 14, 14),
            'edges': torch.randn(1, 1, 224, 224),
            'similarity_scores': torch.tensor([0.85]),
            'anomaly_map': torch.randn(1, 1, 224, 224),
            'final_similarity': torch.tensor([0.85]),
            'meets_threshold': torch.tensor([1.0]),
            'regions': torch.randn(1, 5)
        }
        
        return mock_net
    
    @pytest.fixture
    @patch('fiber_integrated_network.get_config')
    @patch('fiber_integrated_network.get_logger')
    @patch('fiber_integrated_network.TensorProcessor')
    @patch('fiber_integrated_network.FiberOpticsIntegratedNetwork')
    def pipeline(self, mock_network_class, mock_processor_class, 
                 mock_get_logger, mock_get_config, mock_config, mock_network):
        """Create pipeline instance with mocks"""
        mock_get_config.return_value = mock_config
        mock_get_logger.return_value = Mock()
        
        # Setup processor mock
        mock_processor = Mock()
        mock_processor.load_image.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_processor.image_to_tensor.return_value = torch.randn(1, 3, 224, 224)
        mock_processor_class.return_value = mock_processor
        
        # Setup network mock
        mock_network_class.return_value = mock_network
        
        pipeline = IntegratedAnalysisPipeline()
        return pipeline
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline is not None
        assert hasattr(pipeline, 'network')
        assert hasattr(pipeline, 'tensor_processor')
        assert hasattr(pipeline, 'device')
    
    def test_analyze_image(self, pipeline):
        """Test single image analysis"""
        results = pipeline.analyze_image("test.jpg")
        
        assert isinstance(results, dict)
        assert 'raw_outputs' in results
        assert 'summary' in results
        assert 'metadata' in results
    
    def test_summary_generation(self, pipeline):
        """Test summary generation"""
        results = pipeline.analyze_image("test.jpg")
        summary = results['summary']
        
        assert 'final_similarity_score' in summary
        assert 'meets_threshold' in summary
        assert 'primary_region' in summary
        assert 'anomaly_score' in summary
        assert 'edge_strength' in summary
    
    def test_metadata_inclusion(self, pipeline):
        """Test metadata in results"""
        results = pipeline.analyze_image("test.jpg")
        metadata = results['metadata']
        
        assert 'image_path' in metadata
        assert 'analysis_timestamp' in metadata
        assert 'processing_time' in metadata
        assert metadata['image_path'] == "test.jpg"
    
    @patch('builtins.open', create=True)
    def test_export_results(self, mock_open, pipeline):
        """Test results export"""
        results = {
            'summary': {'final_similarity_score': 0.85},
            'metadata': {'image_path': 'test.jpg'}
        }
        
        pipeline.export_results(results, "output.txt")
        
        mock_open.assert_called_with("output.txt", 'w')
    
    def test_batch_analyze(self, pipeline):
        """Test batch analysis"""
        image_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]
        
        results = pipeline.batch_analyze(image_paths)
        
        assert len(results) == 3
        assert all('summary' in r for r in results)
    
    def test_error_handling(self, pipeline):
        """Test error handling in analysis"""
        # Make load_image fail
        pipeline.tensor_processor.load_image.side_effect = Exception("Load failed")
        
        results = pipeline.analyze_image("bad.jpg")
        
        assert 'error' in results
        assert results['error'] == "Load failed"
    
    @patch('fiber_integrated_network.Path')
    def test_load_checkpoint(self, mock_path, pipeline):
        """Test checkpoint loading"""
        mock_checkpoint_path = Mock()
        mock_checkpoint_path.exists.return_value = True
        mock_path.return_value = mock_checkpoint_path
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = {'model_state_dict': {}}
            
            success = pipeline.load_checkpoint("checkpoint.pth")
            
            assert success
            mock_load.assert_called_once()
            pipeline.network.load_state_dict.assert_called_once()


class TestIntegration:
    """Integration tests for the complete network"""
    
    @pytest.fixture
    def real_network(self):
        """Create real network instance (no mocks)"""
        network = FiberOpticsIntegratedNetwork()
        network.eval()
        return network
    
    def test_end_to_end_forward(self, real_network):
        """Test end-to-end forward pass"""
        x = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = real_network(x)
        
        # Verify all outputs are present and have correct shapes
        assert output['features'].shape[0] == 1
        assert output['edges'].shape == (1, 1, 224, 224)
        assert output['final_similarity'].shape == (1,)
        assert output['meets_threshold'].shape == (1,)
        assert 0 <= output['final_similarity'][0] <= 1
    
    def test_gradient_flow(self, real_network):
        """Test gradient flow through network"""
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        output = real_network(x)
        loss = output['final_similarity'].sum()
        
        # Check gradients can be computed
        loss.backward()
        assert x.grad is not None
    
    def test_batch_consistency(self, real_network):
        """Test batch processing consistency"""
        # Process individually
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            out1 = real_network(x1)
            out2 = real_network(x2)
        
        # Process as batch
        batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            batch_out = real_network(batch)
        
        # Results should match
        assert torch.allclose(out1['final_similarity'], batch_out['final_similarity'][0:1], atol=1e-5)
        assert torch.allclose(out2['final_similarity'], batch_out['final_similarity'][1:2], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])