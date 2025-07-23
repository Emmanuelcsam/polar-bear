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

from core.config_loader import get_config
from core.logger import get_logger

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer
    "Recent research (Foret et al., 2021) shows SAM improves generalization by finding flatter minima"
    
    Mathematical formulation:
    Instead of: θ_new = θ - α∇L(θ)
    SAM uses: θ_new = θ - α∇L(θ + ε∇L(θ)/||∇L(θ)||)
    
    This helps find solutions that generalize better to unseen fiber optic patterns
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        """
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.Adam)
            rho: Neighborhood size (default: 0.05)
            adaptive: Whether to adapt rho based on gradient norm
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # Fixed base_optimizer creation to pass self.param_groups instead of raw params
        # Original code passed params directly, but Optimizer requires param_groups; this ensures consistency
        defaults = dict(rho=rho)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho
        self.adaptive = adaptive
        
        # For logging
        self.logger = get_logger("SAM_Optimizer")
        self.logger.log_class_init("SAM")
        
        # Track perturbation statistics
        self.perturbation_stats = {
            'mean_norm': 0.0,
            'max_norm': 0.0,
            'updates': 0
        }
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Compute gradient at perturbed point
        This finds the worst-case perturbation within the rho-ball
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Store original gradient
                self.state[p]['old_grad'] = p.grad.data.clone()
                
                # Adaptive rho based on layer-wise gradient statistics
                if self.adaptive:
                    layer_norm = p.grad.data.norm()
                    scale = min(scale, 0.1 / (layer_norm + 1e-12))
                
                # Apply perturbation: θ + ε
                e_w = scale * p.grad.data
                # Store e_w instead of recomputing in second_step
                # Original code recomputed rho * old_grad / _grad_norm() in second_step, but after perturbation and zero_grad, _grad_norm() would be zero or wrong; fixed by storing e_w directly
                self.state[p]['e_w'] = e_w.clone()
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
                
                # Track statistics
                self.perturbation_stats['mean_norm'] += e_w.norm().item()
                self.perturbation_stats['max_norm'] = max(
                    self.perturbation_stats['max_norm'], 
                    e_w.norm().item()
                )
        
        self.perturbation_stats['updates'] += 1
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Update parameters using gradient at perturbed point
        This performs the actual parameter update
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'old_grad' not in self.state[p]:
                    continue
                    
                # Restore original parameters using stored e_w
                # Original code incorrectly recomputed using _grad_norm() after perturbation, leading to incorrect restoration; fixed by using stored e_w
                p.sub_(self.state[p]['e_w'])
                
        # Update with base optimizer using gradient at perturbed point
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
            
        # Log statistics periodically
        if self.perturbation_stats['updates'] % 100 == 0:
            mean_norm = self.perturbation_stats['mean_norm'] / 100
            self.logger.info(f"SAM perturbation stats - Mean norm: {mean_norm:.6f}, "
                           f"Max norm: {self.perturbation_stats['max_norm']:.6f}")
            self.perturbation_stats['mean_norm'] = 0.0
            self.perturbation_stats['max_norm'] = 0.0

    def step(self, closure=None):
        """
        Performs a single optimization step with SAM
        Requires closure for gradient computation twice: at current point and perturbed point
        """
        assert closure is not None, "SAM requires closure for gradient computation"
        
        # Enable gradient computation for first forward-backward pass
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass at perturbed point
        closure()
        
        # Actual parameter update
        self.second_step()

    def _grad_norm(self):
        """Compute gradient norm across all parameters"""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.data.norm().to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ])
        )
        return norm


class Lookahead(Optimizer):
    """
    Lookahead Optimizer (Zhang et al., 2019)
    Maintains slow and fast weights for better convergence
    
    Mathematical formulation:
    Fast weights: θ_f,t+1 = θ_f,t - α∇L(θ_f,t)
    Slow weights: θ_s,t+1 = θ_s,t + β(θ_f,t+k - θ_s,t)
    
    This helps navigate complex loss landscapes in fiber optic classification
    """
    
    def __init__(self, base_optimizer, k=5, alpha=0.5, pullback_momentum="none"):
        """
        Args:
            base_optimizer: Base optimizer instance
            k: Number of fast weight updates before slow weight update (default: 5)
            alpha: Slow weights step size (default: 0.5)
            pullback_momentum: Type of momentum for slow weights ("none" or "pullback")
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update step size: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
            
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.pullback_momentum = pullback_momentum
        
        # For logging
        self.logger = get_logger("Lookahead_Optimizer")
        self.logger.log_class_init("Lookahead")
        
        # Initialize slow weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                            for group in self.base_optimizer.param_groups]
        
        # For pullback momentum
        if pullback_momentum == "pullback":
            self.pullback_momentum_buffer = [
                [torch.zeros_like(p) for p in group['params']]
                for group in self.base_optimizer.param_groups
            ]
            
        # Track update statistics
        self.update_stats = {
            'fast_updates': 0,
            'slow_updates': 0,
            'weight_diff_norm': 0.0
        }
        
        # Inherit defaults from base optimizer
        defaults = dict(k=k, alpha=alpha)
        super(Lookahead, self).__init__(self.base_optimizer.param_groups, defaults)

    def step(self, closure=None):
        """
        Performs k fast weight updates followed by one slow weight update
        """
        # Fast weight update using base optimizer
        loss = self.base_optimizer.step(closure)
        self.update_stats['fast_updates'] += 1
        self.step_count += 1
        
        # Slow weight update every k steps
        if self.step_count % self.k == 0:
            self._update_slow_weights()
            self.update_stats['slow_updates'] += 1
            
            # Log statistics
            if self.update_stats['slow_updates'] % 10 == 0:
                avg_diff = self.update_stats['weight_diff_norm'] / 10
                self.logger.info(f"Lookahead stats - Fast updates: {self.update_stats['fast_updates']}, "
                               f"Slow updates: {self.update_stats['slow_updates']}, "
                               f"Avg weight diff norm: {avg_diff:.6f}")
                self.update_stats['weight_diff_norm'] = 0.0
                
        return loss

    def _update_slow_weights(self):
        """Update slow weights: θ_s = θ_s + α(θ_f - θ_s)"""
        for idx, (group, slow_group) in enumerate(zip(
            self.base_optimizer.param_groups, self.slow_weights
        )):
            for p_idx, (p, p_slow) in enumerate(zip(group['params'], slow_group)):
                if p.grad is None:
                    continue
                    
                # Track weight difference
                diff = p.data - p_slow.data
                self.update_stats['weight_diff_norm'] += diff.norm().item()
                
                # Update slow weights with momentum if specified
                if self.pullback_momentum == "pullback":
                    buf = self.pullback_momentum_buffer[idx][p_idx]
                    buf.mul_(self.alpha).add_(diff)
                    # Fixed update to correctly implement pullback: slow += buf, then fast = slow
                    # Original code incorrectly updated p.data directly without proper momentum application
                    p_slow.data.add_(buf)
                    p.data.copy_(p_slow.data)
                else:
                    # Standard update: θ_s = (1 - α) * θ_s + α * θ_f, then θ_f = θ_s
                    # Original code had reversed mul/add, leading to incorrect weighted average
                    p_slow.data.mul_(1 - self.alpha).add_(p.data, alpha=self.alpha)
                    p.data.copy_(p_slow.data)
                
                # Update slow weights reference (already done via p_slow.data update)

    def state_dict(self):
        """Returns the state of the optimizer"""
        fast_state = self.base_optimizer.state_dict()
        slow_state = {
            'step_count': self.step_count,
            'slow_weights': self.slow_weights,
            'pullback_momentum_buffer': getattr(self, 'pullback_momentum_buffer', None)
        }
        return {'fast_state': fast_state, 'slow_state': slow_state}

    def load_state_dict(self, state_dict):
        """Loads the optimizer state"""
        self.base_optimizer.load_state_dict(state_dict['fast_state'])
        self.step_count = state_dict['slow_state']['step_count']
        self.slow_weights = state_dict['slow_state']['slow_weights']
        if 'pullback_momentum_buffer' in state_dict['slow_state']:
            self.pullback_momentum_buffer = state_dict['slow_state']['pullback_momentum_buffer']


class SAMWithLookahead:
    """
    Combined SAM + Lookahead optimizer for optimal performance
    "Replace standard gradient descent with SAM + Lookahead"
    
    This combination provides:
    - SAM: Finds flatter minima for better generalization
    - Lookahead: Stabilizes training and improves convergence
    """
    
    def __init__(self, params, lr=0.001, rho=0.05, k=5, alpha=0.5, 
                 weight_decay=0.01, betas=(0.9, 0.999), adaptive=True):
        """
        Initialize combined optimizer
        """
        print(f"[{datetime.now()}] Initializing SAMWithLookahead optimizer")
        
        self.logger = get_logger("SAMWithLookahead")
        self.logger.log_class_init("SAMWithLookahead")
        
        # Create base AdamW optimizer
        base_optimizer_fn = lambda p: optim.AdamW(
            p, lr=lr, betas=betas, weight_decay=weight_decay
        )
        
        # Wrap with SAM
        self.sam_optimizer = SAM(
            params, base_optimizer_fn, rho=rho, adaptive=adaptive
        )
        
        # Wrap with Lookahead
        self.optimizer = Lookahead(self.sam_optimizer.base_optimizer, k=k, alpha=alpha)
        
        # Store parameters for easy access
        self.defaults = dict(
            lr=lr, rho=rho, k=k, alpha=alpha, 
            weight_decay=weight_decay, betas=betas
        )
        
        self.logger.info("SAMWithLookahead optimizer initialized")
        print(f"[{datetime.now()}] SAMWithLookahead initialized successfully")

    def step(self, closure):
        """
        Perform optimization step
        Requires closure for SAM gradient computation
        """
        # SAM first step
        self.sam_optimizer.first_step(zero_grad=True)
        
        # Compute gradient at perturbed point
        closure()
        
        # SAM second step (includes base optimizer step)
        self.sam_optimizer.second_step(zero_grad=True)
        
        # Lookahead handles slow weight updates internally
        self.optimizer.step_count += 1
        if self.optimizer.step_count % self.optimizer.k == 0:
            self.optimizer._update_slow_weights()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        """Get optimizer state"""
        return {
            'sam_state': self.sam_optimizer.state_dict(),
            'lookahead_state': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.sam_optimizer.load_state_dict(state_dict['sam_state'])
        self.optimizer.load_state_dict(state_dict['lookahead_state'])


def create_advanced_optimizer(model: nn.Module, config: Dict) -> SAMWithLookahead:
    """
    Factory function to create the advanced optimizer with proper configuration
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        
    Returns:
        SAMWithLookahead optimizer instance
    """
    logger = get_logger("OptimizerFactory")
    logger.log_function_entry("create_advanced_optimizer")
    
    # Extract hyperparameters from config
    lr = config.get('LEARNING_RATE', 0.001)
    rho = config.get('SAM_RHO', 0.05)
    k = config.get('LOOKAHEAD_K', 5)
    alpha = config.get('LOOKAHEAD_ALPHA', 0.5)
    weight_decay = config.get('WEIGHT_DECAY', 0.01)
    
    # Create parameter groups with different learning rates
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
    
    # Create optimizer
    optimizer = SAMWithLookahead(
        param_groups,
        lr=lr,
        rho=rho,
        k=k,
        alpha=alpha,
        weight_decay=weight_decay,
        adaptive=True
    )
    
    logger.info(f"Created SAMWithLookahead optimizer with {len(param_groups)} parameter groups")
    logger.log_function_exit("create_advanced_optimizer")
    
    return optimizer


# Test the optimizers
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing advanced optimizers")
    
    # Create a simple test model
    test_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Test SAM optimizer
    print("\nTesting SAM optimizer...")
    sam_opt = SAM(test_model.parameters(), optim.Adam, rho=0.05, lr=0.001)
    
    # Test Lookahead optimizer
    print("\nTesting Lookahead optimizer...")
    base_opt = optim.Adam(test_model.parameters(), lr=0.001)
    lookahead_opt = Lookahead(base_opt, k=5, alpha=0.5)
    
    # Test combined optimizer
    print("\nTesting SAMWithLookahead optimizer...")
    combined_opt = SAMWithLookahead(test_model.parameters())
    
    print(f"[{datetime.now()}] Advanced optimizers test completed")
    print(f"[{datetime.now()}] Next script: fiber_advanced_losses.py")