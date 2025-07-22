#!/usr/bin/env python3
"""
Test script to verify Adam optimizer integration with the fiber optics system
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Test imports
try:
    from fiber_hybrid_optimizer import (
        AdvancedAdam, HybridAdamSAM, create_hybrid_optimizer,
        AdamWarmupCosineScheduler
    )
    print(f"[{datetime.now()}] ✓ Successfully imported hybrid optimizer modules")
except ImportError as e:
    print(f"[{datetime.now()}] ✗ Failed to import hybrid optimizer: {e}")
    exit(1)

try:
    from fiber_advanced_config_loader import get_config
    print(f"[{datetime.now()}] ✓ Successfully imported config loader")
except ImportError as e:
    print(f"[{datetime.now()}] ✗ Failed to import config loader: {e}")
    exit(1)


def test_advanced_adam():
    """Test AdvancedAdam optimizer variants"""
    print(f"\n[{datetime.now()}] Testing AdvancedAdam optimizer...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Test AdamW
    print(f"[{datetime.now()}] Testing AdamW variant...")
    optimizer = AdvancedAdam(
        model.parameters(), 
        lr=0.001, 
        decoupled_wd=True,
        weight_decay=0.01
    )
    assert len(optimizer.param_groups) > 0
    print(f"[{datetime.now()}] ✓ AdamW initialized successfully")
    
    # Test NAdam
    print(f"[{datetime.now()}] Testing NAdam variant...")
    optimizer = AdvancedAdam(
        model.parameters(), 
        lr=0.001, 
        nadam=True
    )
    assert len(optimizer.param_groups) > 0
    print(f"[{datetime.now()}] ✓ NAdam initialized successfully")
    
    # Test AMSGrad
    print(f"[{datetime.now()}] Testing AMSGrad variant...")
    optimizer = AdvancedAdam(
        model.parameters(), 
        lr=0.001, 
        amsgrad=True
    )
    assert len(optimizer.param_groups) > 0
    print(f"[{datetime.now()}] ✓ AMSGrad initialized successfully")
    
    # Test RectifiedAdam
    print(f"[{datetime.now()}] Testing RectifiedAdam variant...")
    optimizer = AdvancedAdam(
        model.parameters(), 
        lr=0.001, 
        rectified=True
    )
    assert len(optimizer.param_groups) > 0
    print(f"[{datetime.now()}] ✓ RectifiedAdam initialized successfully")
    
    # Test optimization step
    print(f"[{datetime.now()}] Testing optimization step...")
    x = torch.randn(5, 10)
    target = torch.randn(5, 10)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"[{datetime.now()}] ✓ Optimization step completed successfully")


def test_hybrid_adam_sam():
    """Test HybridAdamSAM optimizer"""
    print(f"\n[{datetime.now()}] Testing HybridAdamSAM optimizer...")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create optimizer
    optimizer = HybridAdamSAM(
        model.parameters(),
        lr=0.001,
        sam_rho=0.05,
        adam_variant="adamw"
    )
    
    assert len(optimizer.param_groups) > 0
    print(f"[{datetime.now()}] ✓ HybridAdamSAM initialized successfully")
    
    # Test optimization step with closure
    print(f"[{datetime.now()}] Testing HybridAdamSAM optimization step...")
    
    x = torch.randn(5, 10)
    target = torch.randn(5, 10)
    
    def closure():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        return loss
    
    # Optimizer step
    loss = closure()
    optimizer.step(closure)
    
    print(f"[{datetime.now()}] ✓ HybridAdamSAM optimization step completed")


def test_scheduler():
    """Test AdamWarmupCosineScheduler"""
    print(f"\n[{datetime.now()}] Testing AdamWarmupCosineScheduler...")
    
    # Create model and optimizer
    model = nn.Linear(10, 10)
    optimizer = AdvancedAdam(model.parameters(), lr=0.001)
    
    # Create scheduler
    scheduler = AdamWarmupCosineScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        min_lr=1e-6
    )
    
    # Test learning rate scheduling
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Step through warmup
    for _ in range(50):
        scheduler.step()
    
    warmup_lr = optimizer.param_groups[0]['lr']
    assert warmup_lr != initial_lr, "Learning rate should change during warmup"
    
    print(f"[{datetime.now()}] ✓ Scheduler working correctly")
    print(f"[{datetime.now()}]   Initial LR: {initial_lr:.6f}")
    print(f"[{datetime.now()}]   After 50 steps: {warmup_lr:.6f}")


def test_config_integration():
    """Test optimizer creation with config"""
    print(f"\n[{datetime.now()}] Testing config integration...")
    
    try:
        config = get_config()
        print(f"[{datetime.now()}] ✓ Config loaded successfully")
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Create optimizer using config
        optimizer = create_hybrid_optimizer(model, config.config)
        
        assert optimizer is not None
        print(f"[{datetime.now()}] ✓ Optimizer created from config successfully")
        
    except Exception as e:
        print(f"[{datetime.now()}] ⚠ Config integration test failed: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ADAM OPTIMIZER INTEGRATION TEST")
    print("=" * 60)
    
    # Run tests
    test_advanced_adam()
    test_hybrid_adam_sam()
    test_scheduler()
    test_config_integration()
    
    print(f"\n[{datetime.now()}] All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()