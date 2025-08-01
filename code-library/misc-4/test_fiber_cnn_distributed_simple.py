#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Unit Tests for Distributed Fiber CNN Training Script
Tests core functions without external dependencies
"""

import unittest
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import logging
from unittest.mock import Mock, patch, MagicMock

# Suppress logging during tests
logging.getLogger().setLevel(logging.ERROR)

class TestFiberCNNDistributedSimple(unittest.TestCase):
    """Simplified test suite for fiber_cnn_distributed.py core functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create mock data directories
        self.data_dir = Path(self.temp_dir) / "dataset"
        self.reference_dir = Path(self.temp_dir) / "reference"
        self.output_dir = Path(self.temp_dir) / "checkpoints"
        
        self.data_dir.mkdir(exist_ok=True)
        self.reference_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create some mock .pt files in reference directory
        for i in range(5):
            mock_tensor = torch.randn(3, 64, 64)
            torch.save(mock_tensor, self.reference_dir / f"mock_ref_{i}.pt")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Cleanup any remaining distributed processes
        if dist.is_initialized():
            dist.destroy_process_group()
    
    @patch('torch.distributed.init_process_group')
    @patch('torch.cuda.set_device')
    def test_setup_distributed(self, mock_set_device, mock_init_process_group):
        """Test setup_distributed function"""
        # Import here to avoid dependency issues
        try:
            from fiber_cnn_distributed import setup_distributed
        except ImportError:
            self.skipTest("fiber_cnn_distributed not available")
        
        rank = 0
        world_size = 2
        
        setup_distributed(rank, world_size)
        
        # Check that environment variables are set
        self.assertEqual(os.environ.get('MASTER_ADDR'), 'localhost')
        self.assertEqual(os.environ.get('MASTER_PORT'), '29500')
        
        # Check that distributed initialization was called
        mock_init_process_group.assert_called_once_with("nccl", rank=rank, world_size=world_size)
        mock_set_device.assert_called_once_with(rank)
    
    @patch('torch.distributed.destroy_process_group')
    def test_cleanup_distributed(self, mock_destroy_process_group):
        """Test cleanup_distributed function"""
        try:
            from fiber_cnn_distributed import cleanup_distributed
        except ImportError:
            self.skipTest("fiber_cnn_distributed not available")
        
        cleanup_distributed()
        mock_destroy_process_group.assert_called_once()
    
    def test_attention_gate_implementation(self):
        """Test AttentionGate component implementation"""
        try:
            from fiber_cnn_pure import AttentionGate
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        F_g, F_l, F_int = 64, 32, 16
        attention_gate = AttentionGate(F_g, F_l, F_int)
        
        # Create test inputs
        g = torch.randn(2, F_g, 16, 16)  # Global features
        x = torch.randn(2, F_l, 32, 32)  # Local features
        
        output = attention_gate(g, x)
        
        # Check output shape
        expected_shape = (2, F_l, 32, 32)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output values are in reasonable range
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_mb_conv_block_implementation(self):
        """Test MBConvBlock component implementation"""
        try:
            from fiber_cnn_pure import MBConvBlock
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        in_channels, out_channels = 64, 128
        mb_conv = MBConvBlock(in_channels, out_channels)
        
        # Create test input
        x = torch.randn(2, in_channels, 32, 32)
        
        output = mb_conv(x)
        
        # Check output shape
        expected_shape = (2, out_channels, 32, 32)
        self.assertEqual(output.shape, expected_shape)
    
    def test_fiber_encoder_implementation(self):
        """Test FiberEncoder component implementation"""
        try:
            from fiber_cnn_pure import FiberEncoder
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        encoder = FiberEncoder(in_channels=3)
        
        # Create test input
        x = torch.randn(2, 3, 512, 512)
        
        features = encoder(x)
        
        # Check that we get the expected number of feature levels
        self.assertEqual(len(features), 5)
        
        # Check feature shapes (should be decreasing spatial dimensions)
        expected_channels = [64, 128, 256, 512, 1024]
        for i, (feature, expected_ch) in enumerate(zip(features, expected_channels)):
            self.assertEqual(feature.shape[1], expected_ch)
    
    def test_fiber_decoder_implementation(self):
        """Test FiberDecoder component implementation"""
        try:
            from fiber_cnn_pure import FiberDecoder
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        encoder_channels = [64, 128, 256, 512, 1024]
        decoder = FiberDecoder(encoder_channels)
        
        # Create test features (simulating encoder output)
        features = [
            torch.randn(2, 64, 256, 256),
            torch.randn(2, 128, 128, 128),
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 512, 32, 32),
            torch.randn(2, 1024, 16, 16)
        ]
        
        output = decoder(features)
        
        # Check output shape (should be same as input spatial dimensions)
        expected_shape = (2, 64, 256, 256)
        self.assertEqual(output.shape, expected_shape)
    
    def test_combined_loss_implementation(self):
        """Test CombinedLoss component implementation"""
        try:
            from fiber_cnn_pure import CombinedLoss
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        loss_fn = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)
        
        # Create test predictions and targets
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 2, (2, 3, 64, 64)).float()
        
        loss = loss_fn(pred, target)
        
        # Check that loss is a scalar tensor
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() > 0)
    
    def test_fiber_analysis_net_implementation(self):
        """Test FiberAnalysisNet model implementation"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        
        # Create test input
        x = torch.randn(2, 3, 512, 512)
        
        outputs = model(x)
        
        # Check output structure
        self.assertIn('zones', outputs)
        self.assertIn('defects', outputs)
        self.assertIn('quality', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['zones'].shape, (2, 3, 512, 512))
        self.assertEqual(outputs['defects'].shape, (2, 4, 512, 512))
        self.assertEqual(outputs['quality'].shape, (2, 3))  # 3 quality classes
    
    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Model should have millions of parameters but not too many
        self.assertGreater(total_params, 1e6)  # At least 1M parameters
        self.assertLess(total_params, 100e6)   # Less than 100M parameters
    
    def test_loss_functions(self):
        """Test all loss functions work correctly"""
        try:
            from fiber_cnn_pure import CombinedLoss
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        # Test CombinedLoss
        combined_loss = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)
        
        pred = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 2, (2, 3, 64, 64)).float()
        
        loss = combined_loss(pred, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss.item() > 0)
        
        # Test CrossEntropyLoss
        ce_loss = nn.CrossEntropyLoss()
        
        pred_ce = torch.randn(2, 3)  # 2 samples, 3 classes
        target_ce = torch.randint(0, 3, (2,))
        
        loss_ce = ce_loss(pred_ce, target_ce)
        self.assertTrue(torch.isfinite(loss_ce))
        self.assertTrue(loss_ce.item() > 0)
    
    def test_optimizer_and_scheduler(self):
        """Test optimizer and scheduler setup"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        
        # Test AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Test CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # Check initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        self.assertEqual(initial_lr, 1e-3)
        
        # Test scheduler step
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        self.assertLess(new_lr, initial_lr)  # LR should decrease
    
    def test_mixed_precision_training(self):
        """Test mixed precision training components"""
        if torch.cuda.is_available():
            # Test GradScaler
            scaler = torch.cuda.amp.GradScaler()
            
            # Create simple model and data
            model = nn.Linear(10, 1)
            x = torch.randn(2, 10)
            y = torch.randn(2, 1)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Test mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = nn.MSELoss()(output, y)
            
            # Test scaler scaling and stepping
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Check that no errors occurred
            self.assertTrue(torch.isfinite(loss))
    
    def test_distributed_sampler(self):
        """Test DistributedSampler functionality"""
        from torch.utils.data import DistributedSampler
        
        # Create mock dataset
        dataset = list(range(100))
        
        # Test sampler creation
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=True
        )
        
        # Test sampler iteration
        indices = list(sampler)
        
        # Check that we get indices
        self.assertGreater(len(indices), 0)
        self.assertLessEqual(len(indices), len(dataset))
        
        # Check that indices are unique
        self.assertEqual(len(indices), len(set(indices)))
    
    def test_model_save_and_load(self):
        """Test model saving and loading"""
        try:
            from fiber_cnn_pure import FiberAnalysisNet
        except ImportError:
            self.skipTest("fiber_cnn_pure not available")
        
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        
        # Save model
        save_path = Path(self.temp_dir) / "test_model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Check file exists
        self.assertTrue(save_path.exists())
        
        # Load model
        new_model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that models produce same output
        x = torch.randn(2, 3, 512, 512)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        # Check that outputs are the same
        for key in output1.keys():
            torch.testing.assert_close(output1[key], output2[key])
    
    def test_torch_basic_functionality(self):
        """Test basic PyTorch functionality"""
        # Test tensor operations
        x = torch.randn(2, 3, 64, 64)
        y = torch.randn(2, 3, 64, 64)
        
        # Test addition
        z = x + y
        self.assertEqual(z.shape, x.shape)
        
        # Test convolution
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        output = conv(x)
        self.assertEqual(output.shape, (2, 64, 64, 64))
        
        # Test batch normalization
        bn = nn.BatchNorm2d(64)
        output = bn(output)
        self.assertEqual(output.shape, (2, 64, 64, 64))
    
    def test_device_handling(self):
        """Test device handling"""
        # Test CPU device
        device_cpu = torch.device('cpu')
        x_cpu = torch.randn(2, 3, 64, 64).to(device_cpu)
        self.assertEqual(x_cpu.device, device_cpu)
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            device_cuda = torch.device('cuda')
            x_cuda = torch.randn(2, 3, 64, 64).to(device_cuda)
            self.assertEqual(x_cuda.device, device_cuda)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(2, 3, requires_grad=True)
        
        # Compute loss
        loss = torch.mean((x - y) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        self.assertTrue(x.grad is not None)
        self.assertTrue(y.grad is not None)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(y.grad.shape, y.shape)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 