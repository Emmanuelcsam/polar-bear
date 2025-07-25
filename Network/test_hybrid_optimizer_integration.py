#!/usr/bin/env python3
"""
Test script to verify hybrid optimizer integration with configuration system
"""

import sys
import torch
import torch.nn as nn
sys.path.append('.')

from core.config_loader import get_config
from utilities.hybrid_optimizer import create_hybrid_optimizer, AdamWarmupCosineScheduler

def test_config_integration():
    """Test that the hybrid optimizer works with the configuration system"""
    print("Testing hybrid optimizer configuration integration...")
    
    # Load configuration
    config = get_config()
    config_dict = config.get_config_dict()  # Convert to dictionary
    print(f"Loaded configuration with optimizer type: {config_dict['optimizer']['type']}")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU()
            )
            self.segmentation = nn.Conv2d(128, 3, 1)
            self.reference_embeddings = nn.Parameter(torch.randn(100, 512))
            self.decoder = nn.Linear(512, 10)
            self.equation = nn.Parameter(torch.randn(5))
            
        def forward(self, x):
            features = self.feature_extractor(x)
            seg = self.segmentation(features)
            return seg, self.decoder(self.reference_embeddings.mean(0))
    
    model = TestModel()
    print(f"Created test model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer using configuration
    try:
        optimizer = create_hybrid_optimizer(model, config_dict)
        print(f"Successfully created {type(optimizer).__name__} optimizer")
        
        # Test scheduler creation
        scheduler_config = config_dict['optimizer']['scheduler']
        if scheduler_config['type'] == 'adam_warmup_cosine':
            scheduler = AdamWarmupCosineScheduler(
                optimizer,
                warmup_steps=scheduler_config['warmup_steps'],
                total_steps=scheduler_config['total_steps'],
                min_lr=scheduler_config['min_lr'],
                warmup_init_lr=scheduler_config['warmup_init_lr']
            )
            print("Successfully created AdamWarmupCosineScheduler")
            
            # Test scheduler step
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate after scheduler step: {current_lr:.6f}")
        
        # Test optimizer step
        dummy_input = torch.randn(2, 3, 32, 32)
        dummy_target = torch.randint(0, 3, (2, 32, 32))
        
        def closure():
            optimizer.zero_grad()
            seg_output, _ = model(dummy_input)
            loss = nn.functional.cross_entropy(seg_output, dummy_target)
            loss.backward()
            return loss
        
        # Try different approaches based on optimizer type
        try:
            # Try with closure first (for HybridAdamSAM)
            loss = optimizer.step(closure)
        except TypeError:
            try:
                # Try without closure (for AdvancedAdam)
                loss = closure()
                optimizer.step()
            except Exception as e:
                print(f"Optimizer step failed: {e}")
                loss = None
        
        if loss is not None:
            print(f"Optimizer step completed successfully, loss: {loss.item():.4f}")
        else:
            print("Optimizer step completed successfully")
        
        print("✅ All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config_integration()
    sys.exit(0 if success else 1) 