#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Installation Script for Fiber Optic Quality Assurance CNN
Verify that all components are working correctly
"""

import os
import sys
import torch
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported"""
    logger.info("Testing package imports...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        logger.info("✓ PyTorch imported successfully")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
        logger.info(f"  OpenCV version: {cv2.__version__}")
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        logger.info("✓ Albumentations imported successfully")
    except ImportError as e:
        logger.error(f"✗ Albumentations import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
        logger.info(f"  NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        logger.info("✓ Matplotlib imported successfully")
    except ImportError as e:
        logger.error(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_model_architecture():
    """Test that the model architecture can be created"""
    logger.info("Testing model architecture...")
    
    try:
        from fiber_cnn_pure import (
            AttentionGate, MBConvBlock, FiberEncoder, FiberDecoder,
            CombinedLoss, FiberAnalysisNet
        )
        logger.info("✓ Model components imported successfully")
        
        # Test model creation
        model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
        logger.info("✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy input
        batch_size = 2
        image_size = 512
        dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Check output shapes
        expected_shapes = {
            'zones': (batch_size, 3, image_size, image_size),
            'defects': (batch_size, 4, image_size, image_size),
            'quality': (batch_size, 3)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = outputs[key].shape
            if actual_shape == expected_shape:
                logger.info(f"✓ {key} output shape correct: {actual_shape}")
            else:
                logger.error(f"✗ {key} output shape incorrect: expected {expected_shape}, got {actual_shape}")
                return False
        
        logger.info("✓ Model forward pass successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model architecture test failed: {e}")
        return False

def test_loss_functions():
    """Test that loss functions work correctly"""
    logger.info("Testing loss functions...")
    
    try:
        from fiber_cnn_pure import CombinedLoss
        
        # Create loss function
        criterion = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)
        
        # Create dummy predictions and targets
        batch_size = 2
        height, width = 64, 64
        
        predictions = torch.randn(batch_size, 3, height, width)
        targets = torch.randint(0, 2, (batch_size, 3, height, width)).float()
        
        # Test loss computation
        loss = criterion(predictions, targets)
        
        if isinstance(loss, torch.Tensor) and loss.item() > 0:
            logger.info("✓ Loss function working correctly")
            logger.info(f"  Loss value: {loss.item():.4f}")
            return True
        else:
            logger.error("✗ Loss function returned invalid value")
            return False
            
    except Exception as e:
        logger.error(f"✗ Loss function test failed: {e}")
        return False

def test_dataset():
    """Test that dataset can be created (if data is available)"""
    logger.info("Testing dataset creation...")
    
    try:
        from fiber_cnn_pure import FiberDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Check if dataset directory exists
        if not os.path.exists('dataset'):
            logger.warning("Dataset directory not found, skipping dataset test")
            return True
        
        # Create simple transform
        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Try to create dataset
        dataset = FiberDataset('dataset', 'reference', transform, mode='train')
        logger.info(f"✓ Dataset created successfully with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            expected_keys = ['image', 'zones', 'defects', 'quality', 'path']
            
            for key in expected_keys:
                if key in sample:
                    logger.info(f"✓ Sample contains {key}")
                else:
                    logger.error(f"✗ Sample missing {key}")
                    return False
            
            logger.info("✓ Dataset sample retrieval successful")
            return True
        else:
            logger.warning("Dataset is empty")
            return True
            
    except Exception as e:
        logger.error(f"✗ Dataset test failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and memory"""
    logger.info("Testing GPU availability...")
    
    if torch.cuda.is_available():
        logger.info("✓ CUDA is available")
        
        # Get GPU info
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU memory allocation
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000).to(device)
            memory_allocated = torch.cuda.memory_allocated(device) / 1e6
            logger.info(f"✓ GPU memory allocation successful ({memory_allocated:.1f} MB)")
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            logger.error(f"✗ GPU memory allocation failed: {e}")
            return False
    else:
        logger.warning("CUDA not available, using CPU")
        return True

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("FIBER OPTIC QUALITY ASSURANCE - INSTALLATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Architecture", test_model_architecture),
        ("Loss Functions", test_loss_functions),
        ("Dataset Creation", test_dataset),
        ("GPU Availability", test_gpu_availability)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Installation is successful.")
        logger.info("\nNext steps:")
        logger.info("1. Train the model: python fiber_cnn_pure.py")
        logger.info("2. Run inference: python inference.py --model-path checkpoints/fiber_analysis_model.pth --image-path your_image.jpg")
        logger.info("3. Deploy on HPC: sbatch run_pure_cnn.slurm")
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install missing packages: pip install -r requirements.txt")
        logger.info("2. Check CUDA installation if using GPU")
        logger.info("3. Verify dataset directory structure")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 