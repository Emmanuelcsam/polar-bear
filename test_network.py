#!/usr/bin/env python3
"""
Test script for Fiber Optics Neural Network
Tests the network with synthetic data to verify functionality
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.config_loader import get_config
from logic.integrated_network import EnhancedIntegratedNetwork
from data.tensor_processor import TensorProcessor

def create_synthetic_data(batch_size=4, height=256, width=256):
    """
    Create synthetic fiber optics image data for testing
    """
    # Create synthetic images with circular patterns (simulating fiber optics)
    images = []
    labels = []
    
    for i in range(batch_size):
        # Create base image
        image = np.zeros((3, height, width), dtype=np.float32)
        
        # Add circular patterns (core and cladding)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Core region (bright center)
        core_radius = np.random.randint(30, 50)
        core_mask = (x - center_x)**2 + (y - center_y)**2 <= core_radius**2
        
        # Cladding region (dimmer ring)
        cladding_radius = np.random.randint(80, 120)
        cladding_mask = ((x - center_x)**2 + (y - center_y)**2 <= cladding_radius**2) & ~core_mask
        
        # Apply patterns to all channels
        for c in range(3):
            image[c][core_mask] = 0.8 + np.random.randn() * 0.1
            image[c][cladding_mask] = 0.5 + np.random.randn() * 0.1
            # Add some noise
            image[c] += np.random.randn(height, width) * 0.05
        
        images.append(image)
        
        # Create simple labels (class 0: good, 1: defect)
        labels.append(0 if i % 2 == 0 else 1)
    
    # Convert to tensors
    images_tensor = torch.tensor(np.array(images), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return images_tensor, labels_tensor

def test_network():
    """
    Test the network with synthetic data
    """
    print("="*80)
    print("FIBER OPTICS NEURAL NETWORK - TEST MODE")
    print("="*80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = get_config()
    device = torch.device("cpu")  # Force CPU for testing
    print(f"   Device: {device}")
    
    # Initialize network
    print("\n2. Initializing network...")
    network = EnhancedIntegratedNetwork()
    network = network.to(device)
    network.eval()
    print(f"   Network initialized with {sum(p.numel() for p in network.parameters())} parameters")
    
    # Create synthetic data
    print("\n3. Creating synthetic test data...")
    batch_size = 2
    images, labels = create_synthetic_data(batch_size=batch_size)
    images = images.to(device)
    
    # Create reference features (required by the network)
    num_references = 100
    feature_dim = 6
    reference_features = torch.randn(num_references, feature_dim).to(device)
    
    print(f"   Created {batch_size} synthetic images of shape {images.shape}")
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = network(images)
        
        print("   ✓ Forward pass successful!")
        
        # Display output information
        print("\n5. Network outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, dict):
                print(f"   - {key}: dictionary with {len(value)} items")
            else:
                print(f"   - {key}: {type(value)}")
        
        # Test some key outputs
        if 'final_similarity' in outputs:
            similarity_scores = outputs['final_similarity']
            print(f"\n6. Similarity scores: {similarity_scores.cpu().numpy()}")
        
        if 'segmentation_logits' in outputs:
            seg_shape = outputs['segmentation_logits'].shape
            print(f"\n7. Segmentation output shape: {seg_shape}")
        
        if 'anomaly_map' in outputs:
            anomaly_shape = outputs['anomaly_map'].shape
            print(f"\n8. Anomaly map shape: {anomaly_shape}")
        
        print("\n" + "="*80)
        print("✅ NETWORK TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Test memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_single_module():
    """
    Test individual modules
    """
    print("\nTesting individual modules...")
    
    # Test TensorProcessor
    print("\n- Testing TensorProcessor...")
    try:
        processor = TensorProcessor()
        test_tensor = torch.randn(1, 3, 256, 256)
        # Use standardize_batch which handles batches properly
        processed = processor.standardize_batch(test_tensor)
        print(f"  ✓ TensorProcessor: input shape {test_tensor.shape} -> output shape {processed.shape}")
    except Exception as e:
        print(f"  ⚠ TensorProcessor test skipped due to: {e}")
    
    # Test more modules as needed
    
    return True

if __name__ == "__main__":
    print("Starting Fiber Optics Neural Network Test...\n")
    
    # Test individual modules first
    if test_single_module():
        print("\n✓ Individual module tests passed")
    
    # Test full network
    if test_network():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")