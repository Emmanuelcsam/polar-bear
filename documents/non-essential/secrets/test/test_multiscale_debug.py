#!/usr/bin/env python3
"""Debug MultiScaleFeatureExtractor to find the hang"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_tensor_processor import TensorProcessor
from fiber_feature_extractor import MultiScaleFeatureExtractor

# Monkey patch the forward method to add debug prints
original_forward = MultiScaleFeatureExtractor.forward

def debug_forward(self, x, gradient_info, position_info):
    print("DEBUG: MultiScaleFeatureExtractor.forward called")
    print(f"  Input shape: {x.shape}")
    
    # Call log_function_entry
    print("  Calling log_function_entry...")
    self.logger.log_function_entry("forward", input_shape=x.shape)
    print("  ✓ log_function_entry completed")
    
    # Calculate gradient and position weights
    print("  Calculating gradient weights...")
    avg_gradient = gradient_info['average_gradient'].unsqueeze(0)
    gradient_weights = self.gradient_weight_net(avg_gradient)
    print(f"  ✓ gradient_weights shape: {gradient_weights.shape}")
    
    print("  Calculating position weights...")
    avg_positions = torch.stack([
        position_info['average_x'],
        position_info['average_y'],
        position_info['average_radial']
    ]).unsqueeze(0)
    position_weights = self.position_weight_net(avg_positions)
    print(f"  ✓ position_weights shape: {position_weights.shape}")
    
    # Combined weights
    scale_weights = gradient_weights * position_weights * self.config.GRADIENT_WEIGHT_FACTOR
    print(f"  ✓ scale_weights shape: {scale_weights.shape}")
    
    # Extract features at each scale
    print(f"  Starting feature extraction for {len(self.scale_extractors)} scales...")
    all_features = []
    all_region_logits = []
    all_anomaly_logits = []
    all_quality_scores = []
    
    current = x
    
    for i, (extractor, weight) in enumerate(zip(self.scale_extractors, scale_weights[0])):
        print(f"  Scale {i}: Starting...")
        # Extract features with weight modulation
        scale_results = extractor(current)
        print(f"    ✓ Feature extraction done")
        
        # Apply scale weight
        weighted_features = scale_results['features'] * weight.item()
        print(f"    ✓ Weight applied: {weight.item():.4f}")
        
        # Store results
        all_features.append(weighted_features)
        all_region_logits.append(scale_results['region_logits'])
        all_anomaly_logits.append(scale_results['anomaly_logits'])
        all_quality_scores.append(scale_results['quality_scores'])
        
        # Use features as input for next scale
        current = weighted_features
        print(f"    ✓ Scale {i} completed")
    
    print("  All scales completed!")
    return {
        'features': all_features,
        'region_logits': all_region_logits,
        'anomaly_logits': all_anomaly_logits,
        'quality_scores': all_quality_scores,
        'scale_weights': scale_weights
    }

MultiScaleFeatureExtractor.forward = debug_forward

# Now test it
print("Creating MultiScaleFeatureExtractor...")
mfe = MultiScaleFeatureExtractor()

print("\nCreating test input...")
x = torch.randn(1, 3, 224, 224)
tp = TensorProcessor()
grad_info = tp.calculate_gradient_intensity(x)
pos_info = tp.calculate_pixel_positions(x.shape)

print("\nCalling forward...")
try:
    result = mfe(x, grad_info, pos_info)
    print(f"\n✓ Success! Result keys: {list(result.keys())}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()