#!/usr/bin/env python3
"""
Test runner for Fiber Optics Neural Network
Simple script to test the data loading and processing pipeline
"""

import sys
import torch
from pathlib import Path
from datetime import datetime

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import required modules
from core.config_loader import get_config
from core.logger import get_logger
from data.data_loader import FiberOpticsDataLoader
from data.tensor_processor import TensorProcessor
from data.feature_extractor import FeatureExtractionPipeline

def main():
    """Main test function"""
    print(f"[{datetime.now()}] Starting Fiber Optics Network Test")
    
    # Initialize logger
    logger = get_logger("TestRunner")
    logger.log_process_start("Network Test")
    
    try:
        # 1. Test Config Loading
        print("\n1. Loading configuration...")
        config = get_config()
        print(f"   - Config loaded successfully")
        print(f"   - Device: {config.get_device()}")
        
        # 2. Test Data Loader
        print("\n2. Testing data loader...")
        data_loader = FiberOpticsDataLoader()
        print(f"   - Found {len(data_loader.data_paths)} data paths")
        
        # Try to create data loaders
        try:
            train_loader, val_loader = data_loader.get_data_loaders(
                train_ratio=0.8,
                batch_size=4,
                use_weighted_sampling=True,
                use_augmentation=False  # Disable augmentation for initial test
            )
            print(f"   - Train loader: {len(train_loader)} batches")
            print(f"   - Val loader: {len(val_loader)} batches")
        except Exception as e:
            print(f"   - Error creating data loaders: {e}")
            print("   - This is expected if no data files are found")
        
        # 3. Test Tensor Processor
        print("\n3. Testing tensor processor...")
        tensor_processor = TensorProcessor()
        
        # Create a test tensor
        test_tensor = torch.randn(3, 256, 256).to(tensor_processor.device)
        print(f"   - Created test tensor: shape={test_tensor.shape}")
        
        # Test gradient calculation
        gradient_info = tensor_processor.calculate_gradient_intensity(test_tensor)
        print(f"   - Gradient intensity: {gradient_info['average_gradient'].item():.4f}")
        
        # Test position calculation
        position_info = tensor_processor.calculate_pixel_positions(test_tensor.shape)
        print(f"   - Average radial position: {position_info['average_radial'].item():.4f}")
        
        # 4. Test Feature Extractor
        print("\n4. Testing feature extractor...")
        feature_pipeline = FeatureExtractionPipeline()
        
        # Extract features from test tensor
        features = feature_pipeline.extract_features(test_tensor.unsqueeze(0))
        print(f"   - Extracted {len(features['multi_scale_features'])} scale features")
        print(f"   - Anomaly map shape: {features['anomaly_map'].shape}")
        print(f"   - Quality map mean: {features['quality_map'].mean().item():.4f}")
        
        # 5. Try loading a batch if data is available
        if 'train_loader' in locals() and len(train_loader) > 0:
            print("\n5. Testing data batch loading...")
            for batch_idx, batch in enumerate(train_loader):
                print(f"   - Batch {batch_idx}:")
                print(f"     - Image shape: {batch['image'].shape}")
                print(f"     - Labels: {batch['label']}")
                print(f"     - Has anomaly: {batch['has_anomaly']}")
                
                # Process the batch through feature extractor
                batch_features = feature_pipeline.extract_features(batch['image'])
                print(f"     - Processed features successfully")
                
                break  # Only test one batch
        
        print(f"\n[{datetime.now()}] Test completed successfully!")
        logger.log_process_end("Network Test")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        logger.log_error("Test failed", e)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())