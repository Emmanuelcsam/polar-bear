#!/usr/bin/env python3
"""
Advanced Optimization Techniques for Fiber Optics Neural Network
Implements SAM (Sharpness-Aware Minimization) and Lookahead optimizers
Based on research: Foret et al., 2021 and Zhang et al., 2019
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from collections import defaultdict
import copy

# FIX: To make this file runnable standalone for testing, added dummy logger.
class DummyLogger:
    def info(self, msg): print(msg)
    def log_class_init(self, *args, **kwargs): pass
    def log_function_entry(self, *args, **kwargs): pass
    def log_function_exit(self, *args, **kwargs): pass

# Import logger with fallback
try:
    from core.config_loader import get_config
    from core.logger import get_logger
except ImportError:
    # Fallback for standalone testing
    def get_logger(name): 
        return DummyLogger()

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    It seeks parameters in neighborhoods having uniformly low loss (flat minima),
    which is known to improve model generalization. It does this by performing a
    two-step update: first, it finds a "high-loss" point in the neighborhood of
    the current weights, and second, it updates the weights using the gradient
    from that high-loss point.
    """
    
    def __init__(self, params, base_optimizer: type[Optimizer], rho: float = 0.05, adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        # This wrapper takes the class of the base optimizer (e.g., torch.optim.AdamW) and its arguments.
        self.base_optimizer = base_optimizer(params, **kwargs)
        
        # Inherit parameter groups from the base optimizer.
        super(SAM, self).__init__(self.base_optimizer.param_groups, dict(rho=rho, adaptive=adaptive))
        
        # For logging
        self.logger = get_logger("SAM_Optimizer")
        self.logger.log_class_init("SAM")
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First SAM step: calculates the gradient, computes the perturbation `e_w`,
        and moves the weights to `w + e_w`. This is the "ascent" step to find the
        point of highest loss within the rho-ball.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Calculate perturbation `e_w`.
                # The adaptive version scales the perturbation by the parameter norms.
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale
                
                # Climb to the point of highest loss in the neighborhood.
                p.add_(e_w)
                
                # Store the perturbation for the second step.
                self.state[p]['e_w'] = e_w
        
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second SAM step: moves the weights back to `w` and then performs
        the actual update using the gradient computed at `w + e_w`. This is the
        "descent" step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                # Move weights back to the original position before the update.
                p.sub_(self.state[p]['e_w'])
        
        # The base optimizer performs its update step using the gradient from the perturbed point.
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad(set_to_none=True)
            
    def step(self, closure: Callable):
        """
        Performs the full SAM optimization step.
        Requires a closure that re-evaluates the model and returns the loss.
        """
        # First forward/backward pass to get gradient at `w`.
        loss = closure()
        
        self.first_step(zero_grad=True)
        
        # Second forward/backward pass to get gradient at `w + e_w`.
        closure()
        
        self.second_step()
        return loss

    def _grad_norm(self):
        """Computes the norm of all gradients combined."""
        # FIX: Ensure all norms are on the same device before stacking.
        # This prevents errors in distributed or multi-GPU settings.
        device = self.param_groups[0]['params'][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ])
        )

    # Delegate state dict and load state dict to the base optimizer
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        # Ensure param groups are in sync
        self.param_groups = self.base_optimizer.param_groups


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    It maintains two sets of weights (fast and slow) and updates the slow weights
    by interpolating towards the fast weights every `k` steps. This improves stability and
    can lead to faster convergence.
    """
    
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0: raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k: raise ValueError(f"Invalid lookahead steps: {k}")
        
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.step_counter = 0
        
        # For logging
        self.logger = get_logger("Lookahead_Optimizer")
        self.logger.log_class_init("Lookahead")
        
        # Initialize slow weights as a copy of the initial model weights.
        self.slow_weights = [
            [p.clone().detach() for p in group['params']]
            for group in self.param_groups
        ]

    def step(self, closure: Optional[Callable] = None):
        """Performs one step of the inner optimizer and, if it's a `k`-th step, updates slow weights."""
        # The inner optimizer (e.g., AdamW or SAM) performs its step.
        # The closure is passed down to the inner optimizer if it needs it (like SAM).
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        
        if self.step_counter % self.k == 0:
            self._update_slow_weights()
            
        return loss

    @torch.no_grad()
    def _update_slow_weights(self):
        """Updates the slow weights and syncs the fast weights to them."""
        for slow_group, fast_group in zip(self.slow_weights, self.param_groups):
            for p_slow, p_fast in zip(slow_group, fast_group['params']):
                # Update slow weights: slow_w = slow_w + alpha * (fast_w - slow_w)
                p_slow.data.add_(p_fast.data - p_slow.data, alpha=self.alpha)
                # Sync fast weights to the new slow weights.
                p_fast.data.copy_(p_slow.data)

    def state_dict(self):
        fast_state = self.optimizer.state_dict()
        slow_state = {
            'step_counter': self.step_counter,
            'slow_weights': [
                [p.cpu() for p in group] for group in self.slow_weights
            ]
        }
        return {'fast_state': fast_state, 'slow_state': slow_state}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['fast_state'])
        self.step_counter = state_dict['slow_state']['step_counter']
        
        # Move slow weights to the correct device
        device = self.param_groups[0]['params'][0].device
        self.slow_weights = [
            [p.to(device) for p in group]
            for group in state_dict['slow_state']['slow_weights']
        ]

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)


def create_advanced_optimizer(model: nn.Module, config: Dict) -> Optimizer:
    """
    Factory function to create an advanced optimizer that correctly composes
    AdamW, SAM, and Lookahead.
    
    Args:
        model: The neural network model.
        config: A configuration dictionary with optimizer settings.
        
    Returns:
        A composed Lookahead(SAM(AdamW(...))) optimizer instance.
    """
    logger = get_logger("OptimizerFactory")
    logger.log_function_entry("create_advanced_optimizer")
    
    # Extract hyperparameters from config.
    lr = config.get('LEARNING_RATE', 0.001)
    rho = config.get('SAM_RHO', 0.05)
    k = config.get('LOOKAHEAD_K', 5)
    alpha = config.get('LOOKAHEAD_ALPHA', 0.5)
    weight_decay = config.get('WEIGHT_DECAY', 0.01)
    betas = tuple(config.get('BETAS', (0.9, 0.999)))
    
    # Create parameter groups with different learning rates for different model components
    param_groups = [
        # Feature extraction layers - normal learning rate
        {'params': [p for n, p in model.named_parameters() 
                   if 'feature_extractor' in n], 'lr': lr},
        
        # Segmentation layers - slightly lower
        {'params': [p for n, p in model.named_parameters() 
                   if 'segmentation' in n], 'lr': lr * 0.5},
        
        # Reference embeddings - much slower
        {'params': [p for n, p in model.named_parameters() 
                   if 'reference_embeddings' in n], 'lr': lr * 0.1},
        
        # Decoder layers
        {'params': [p for n, p in model.named_parameters() 
                   if 'decoder' in n], 'lr': lr * 0.5},
        
        # Equation parameters - very slow updates
        {'params': [p for n, p in model.named_parameters() 
                   if 'equation' in n], 'lr': lr * 0.01},
        
        # Other parameters
        {'params': [p for n, p in model.named_parameters() 
                   if not any(x in n for x in ['feature_extractor', 'segmentation', 
                                               'reference_embeddings', 'decoder', 'equation'])], 
         'lr': lr}
    ]
    
    # Filter out empty parameter groups
    param_groups = [g for g in param_groups if len(list(g['params'])) > 0]
    
    # FIX: The original file had a confusing `SAMWithLookahead` class.
    # The correct approach is to wrap the optimizers.
    # 1. Base optimizer is AdamW.
    # 2. SAM wraps the AdamW class.
    # 3. Lookahead wraps the SAM instance.
    
    # 1. Define the base optimizer class and its arguments.
    base_optimizer_class = optim.AdamW
    base_optimizer_args = {'lr': lr, 'weight_decay': weight_decay, 'betas': betas}

    # 2. Create the SAM optimizer, which will internally create AdamW instances.
    sam_optimizer = SAM(param_groups, base_optimizer_class, rho=rho, adaptive=True, **base_optimizer_args)

    # 3. Wrap the SAM optimizer with Lookahead.
    final_optimizer = Lookahead(sam_optimizer, k=k, alpha=alpha)
    
    logger.info(f"Created composed Lookahead(SAM(AdamW)) optimizer with {len(param_groups)} parameter groups")
    logger.log_function_exit("create_advanced_optimizer")
    
    return final_optimizer


# Test the optimizers
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing advanced optimizers")
    
    test_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 10)
    
    # Test the composed optimizer
    print("\nTesting composed Lookahead(SAM(AdamW)) optimizer...")
    optimizer = create_advanced_optimizer(test_model, {})
    
    def closure():
        # The closure for SAM should NOT zero the grad. SAM handles it.
        output = test_model(dummy_input)
        loss = nn.MSELoss()(output, dummy_target)
        loss.backward()
        return loss
        
    # The step call for the composed optimizer will trigger SAM's two-step process
    # and Lookahead's update mechanism.
    optimizer.zero_grad()
    initial_loss = closure()
    print(f"Initial loss: {initial_loss.item():.4f}")
    
    # The `step` method of the wrapped SAM optimizer requires a closure.
    # Lookahead passes this closure down to SAM.
    optimizer.step(closure)
    
    optimizer.zero_grad()
    final_loss = closure()
    print(f"Loss after one step: {final_loss.item():.4f}")
    assert final_loss < initial_loss, "Loss did not decrease after one step."
    
    print(f"[{datetime.now()}] Advanced optimizers test completed")