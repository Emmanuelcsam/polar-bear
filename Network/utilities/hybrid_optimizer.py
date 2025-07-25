#!/usr/bin/env python3
"""
Advanced Adam Optimizer Implementations for Fiber Optics Neural Network
Implements Adam, AdamW, NAdam, AMSGrad, and RectifiedAdam variants
Also provides hybrid combinations with SAM and Lookahead

Based on research:
- Adam: Kingma & Ba, 2014 - "Adam: A Method for Stochastic Optimization"
- AdamW: Loshchilov & Hutter, 2017 - "Decoupled Weight Decay Regularization"
- NAdam: Dozat, 2016 - "Incorporating Nesterov Momentum into Adam"
- AMSGrad: Reddi et al., 2018 - "On the Convergence of Adam and Beyond"
- RectifiedAdam: Liu et al., 2019 - "On the Variance of the Adaptive Learning Rate and Beyond"
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime
from collections import defaultdict

from core.logger import get_logger
from utilities.optimizers import SAM, Lookahead


class AdvancedAdam(Optimizer):
    """
    Advanced Adam optimizer with multiple variants and improvements
    
    Supports:
    - Standard Adam
    - AdamW (decoupled weight decay)
    - NAdam (Nesterov momentum)
    - AMSGrad (maximum of past squared gradients)
    - RectifiedAdam (variance rectification)
    - Gradient centralization
    - Per-parameter adaptive learning rates
    """
    
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 nadam: bool = False,
                 rectified: bool = False,
                 decoupled_wd: bool = True,
                 gradient_centralization: bool = False,
                 lookahead: bool = False,
                 lookahead_k: int = 5,
                 lookahead_alpha: float = 0.5,
                 warmup_steps: int = 0,
                 gradient_clipping: Optional[float] = None):
        """
        Initialize Advanced Adam optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use AMSGrad variant
            nadam: Whether to use NAdam variant
            rectified: Whether to use RectifiedAdam variant
            decoupled_wd: Whether to use decoupled weight decay (AdamW)
            gradient_centralization: Whether to centralize gradients
            lookahead: Whether to use Lookahead wrapper
            lookahead_k: Lookahead steps
            lookahead_alpha: Lookahead interpolation factor
            warmup_steps: Number of warmup steps
            gradient_clipping: Max gradient norm for clipping
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        # Initialize logger
        self.logger = get_logger("AdvancedAdam")
        self.logger.log_class_init("AdvancedAdam")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, nadam=nadam, rectified=rectified,
            decoupled_wd=decoupled_wd, gradient_centralization=gradient_centralization
        )
        super(AdvancedAdam, self).__init__(params, defaults)
        
        # Additional settings
        self.warmup_steps = warmup_steps
        self.gradient_clipping = gradient_clipping
        self.lookahead = lookahead
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        
        # Initialize lookahead if requested
        if self.lookahead:
            self._init_lookahead()
            
        # Track statistics
        self.stats = {
            'step': 0,
            'gradient_norm': 0.0,
            'update_norm': 0.0,
            'effective_lr': lr
        }
        
        self.logger.info(f"AdvancedAdam initialized with variants: "
                        f"AMSGrad={amsgrad}, NAdam={nadam}, Rectified={rectified}, "
                        f"DecoupledWD={decoupled_wd}, GradCentralization={gradient_centralization}, "
                        f"Lookahead={lookahead}")
    
    def _init_lookahead(self):
        """Initialize lookahead slow weights"""
        self.slow_state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.slow_state[p]
                param_state['slow_buffer'] = p.data.clone()
    
    def _centralize_gradient(self, gradient: torch.Tensor, gc_conv_only: bool = True):
        """
        Gradient centralization to improve training stability
        "Research shows gradient centralization can accelerate training and improve generalization"
        """
        # Only centralize gradients for convolutional layers (more than 3 dimensions)
        if gc_conv_only and len(gradient.shape) <= 3:
            return gradient
            
        # FIX: Ensure tensor is not empty before calling mean to avoid NaN
        if gradient.numel() > 1:
            if len(gradient.shape) > 3:  # For Conv layers
                gradient.add_(-gradient.mean(dim=tuple(range(1, len(gradient.shape))), keepdim=True))
            else:  # For other layers
                gradient.add_(-gradient.mean())
        return gradient
    
    def _get_effective_lr(self, step: int, lr: float) -> float:
        """
        Get effective learning rate with warmup
        Linear warmup followed by base learning rate
        """
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return lr * (step + 1) / self.warmup_steps
        return lr
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Gradient clipping if specified
        if self.gradient_clipping is not None:
            # FIX: Filter out params without gradients before clipping
            params_to_clip = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, self.gradient_clipping)
        
        # Track global step
        self.stats['step'] += 1
        
        # Compute gradient norm for monitoring
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
        self.stats['gradient_norm'] = math.sqrt(grad_norm)
        
        # Update parameters
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                params_with_grad.append(p)
                
                # Apply gradient centralization if enabled
                if group['gradient_centralization']:
                    grad = self._centralize_gradient(p.grad.data)
                else:
                    grad = p.grad.data
                    
                grads.append(grad)
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Perform optimization step
            self._adam_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                nadam=group['nadam'],
                rectified=group['rectified'],
                beta1=beta1,
                beta2=beta2,
                lr=self._get_effective_lr(self.stats['step'], group['lr']),
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                decoupled_wd=group['decoupled_wd']
            )
        
        # Lookahead step if enabled
        if self.lookahead and self.stats['step'] % self.lookahead_k == 0:
            self._lookahead_step()
        
        # Log statistics periodically
        if self.stats['step'] % 100 == 0:
            self.logger.info(f"Step {self.stats['step']} - "
                           f"Gradient norm: {self.stats['gradient_norm']:.6f}, "
                           f"Effective LR: {self.stats['effective_lr']:.6f}")
        
        return loss
    
    def _adam_step(
                   self,
                   params: List[torch.Tensor],
                   grads: List[torch.Tensor],
                   exp_avgs: List[torch.Tensor],
                   exp_avg_sqs: List[torch.Tensor],
                   max_exp_avg_sqs: List[torch.Tensor],
                   state_steps: List[int],
                   *,
                   amsgrad: bool,
                   nadam: bool,
                   rectified: bool,
                   beta1: float,
                   beta2: float,
                   lr: float,
                   weight_decay: float,
                   eps: float,
                   decoupled_wd: bool):
        """
        Functional API for Adam step
        Supports all variants: Adam, AdamW, NAdam, AMSGrad, RectifiedAdam
        """
        # Reset update norm for this step
        self.stats['update_norm'] = 0.0

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # Decoupled weight decay (AdamW)
            if weight_decay != 0 and decoupled_wd:
                param.data.mul_(1 - lr * weight_decay)

            # Update biased first and second moment estimates
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            step_size = lr / bias_correction1
            
            # Rectified Adam (RAdam) logic
            if rectified:
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / (1 - beta2 ** step)
                
                if rho_t > 5.0:  # Use adaptive momentum
                    r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    update = exp_avg / (denom * math.sqrt(bias_correction2)) * r_t
                else:  # Use un-adapted momentum
                    update = exp_avg
            else:  # Standard Adam / NAdam / AdamW
                update = exp_avg / (denom / math.sqrt(bias_correction2))

            # Nesterov-style update (NAdam)
            if nadam:
                update = (beta1 * update) + ((1 - beta1) / bias_correction1) * (grad / denom)

            param.data.add_(update, alpha=-step_size)

            # Standard L2 weight decay (not decoupled)
            if weight_decay != 0 and not decoupled_wd:
                param.data.add_(param.data, alpha=-weight_decay * lr)

            # FIX: Calculate update norm correctly and accumulate
            self.stats['update_norm'] += (step_size * update.norm()).item() ** 2

        self.stats['update_norm'] = math.sqrt(self.stats['update_norm'])
        self.stats['effective_lr'] = lr
    
    def _lookahead_step(self):
        """Perform lookahead step: interpolate between fast and slow weights"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_state = self.slow_state[p]
                slow_buf = param_state['slow_buffer']
                
                # Interpolate: slow = slow + alpha * (fast - slow)
                slow_buf.add_(p.data - slow_buf, alpha=self.lookahead_alpha)
                p.data.copy_(slow_buf)
    
    def state_dict(self):
        """Returns the state of the optimizer"""
        state_dict = super().state_dict()
        state_dict['stats'] = self.stats
        if self.lookahead:
            state_dict['slow_state'] = self.slow_state
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Loads the optimizer state"""
        self.stats = state_dict.pop('stats', self.stats)
        if self.lookahead and 'slow_state' in state_dict:
            self.slow_state = state_dict.pop('slow_state')
        super().load_state_dict(state_dict)


class HybridAdamSAM(Optimizer):
    """
    Hybrid optimizer combining Advanced Adam with SAM (Sharpness-Aware Minimization)
    
    This provides the benefits of:
    - Adam's adaptive learning rates
    - SAM's flat minima seeking
    - Multiple Adam variants (AdamW, NAdam, AMSGrad, etc.)
    """
    
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 sam_rho: float = 0.05,
                 sam_adaptive: bool = True,
                 adam_variant: str = "adamw",  # adam, adamw, nadam, amsgrad, rectified
                 gradient_centralization: bool = False,
                 warmup_steps: int = 0):
        """
        Initialize Hybrid Adam-SAM optimizer
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam beta parameters
            eps: Adam epsilon
            weight_decay: Weight decay coefficient
            sam_rho: SAM neighborhood size
            sam_adaptive: Whether to use adaptive SAM
            adam_variant: Which Adam variant to use
            gradient_centralization: Whether to centralize gradients
            warmup_steps: Number of warmup steps
        """
        print(f"[{datetime.now()}] Initializing HybridAdamSAM optimizer")
        
        self.logger = get_logger("HybridAdamSAM")
        self.logger.log_class_init("HybridAdamSAM")
        
        # Configure Adam variant
        adam_config = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'gradient_centralization': gradient_centralization,
            'warmup_steps': warmup_steps,
            'decoupled_wd': False,
            'nadam': False,
            'amsgrad': False,
            'rectified': False
        }
        
        # Set variant-specific flags
        if adam_variant == "adamw":
            adam_config['decoupled_wd'] = True
        elif adam_variant == "nadam":
            adam_config['nadam'] = True
        elif adam_variant == "amsgrad":
            adam_config['amsgrad'] = True
        elif adam_variant == "rectified":
            adam_config['rectified'] = True
        
        # Create base Adam optimizer
        self.base_optimizer = AdvancedAdam(params, **adam_config)
        
        # SAM parameters
        self.rho = sam_rho
        self.adaptive = sam_adaptive
        
        # Initialize with SAM defaults
        self.param_groups = self.base_optimizer.param_groups
        defaults = dict(rho=sam_rho, adaptive=sam_adaptive)
        super(HybridAdamSAM, self).__init__(self.param_groups, defaults)
        
        # Track SAM statistics
        self.sam_stats = {
            'perturbation_norm': 0.0,
            'gradient_similarity': 0.0
        }
        
        self.logger.info(f"HybridAdamSAM initialized with {adam_variant} variant")
        print(f"[{datetime.now()}] HybridAdamSAM initialized successfully")
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """SAM first step: perturb weights to find a point of higher loss"""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Store original gradient for stats and parameter restoration
                self.state[p]['old_grad'] = p.grad.data.clone()
                
                e_w = p.grad.data * scale
                if self.adaptive:
                    # Adaptive scaling per layer
                    param_norm = torch.norm(p.data)
                    grad_norm_layer = torch.norm(p.grad.data)
                    # FIX: Use a robust scaling factor
                    e_w = e_w * param_norm / (grad_norm_layer + 1e-12)
                
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        
        if zero_grad:
            self.zero_grad(set_to_none=True)  # Use set_to_none for performance

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """SAM second step: restore original weights and perform update with new gradient"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                # FIX: Restore original parameters by subtracting the stored perturbation
                # The original code had a fallback that was incorrect. This is more robust.
                p.sub_(self.state[p]['e_w'])
        
        # The base optimizer (AdvancedAdam) now uses the gradient from the perturbed point
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Callable):
        """Performs a single optimization step, requiring a closure to re-evaluate loss"""
        assert closure is not None, "HybridAdamSAM requires a closure"
        
        # The closure should compute the loss and call .backward()
        # This first call gets the gradient at the original weights
        loss = closure()
        
        self.first_step(zero_grad=True)
        
        # This second call gets the gradient at the perturbed weights
        closure()
        
        self.second_step()
        
        return loss

    def _grad_norm(self):
        """Compute the norm of the gradient across all parameters"""
        # FIX: Use a shared device to avoid device mismatch errors
        shared_device = self.param_groups[0]['params'][0].device
        # This combines all parameter gradients into a single vector to compute the total norm
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
            ])
        )
        return norm
    
    def state_dict(self):
        """Get optimizer state"""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'sam_stats': self.sam_stats
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.sam_stats = state_dict.get('sam_stats', self.sam_stats)


def create_hybrid_optimizer(model: nn.Module, config: Dict) -> Union[AdvancedAdam, HybridAdamSAM]:
    """
    Factory function to create hybrid optimizer based on configuration
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        
    Returns:
        Optimizer instance (AdvancedAdam or HybridAdamSAM)
    """
    logger = get_logger("HybridOptimizerFactory")
    logger.log_function_entry("create_hybrid_optimizer")
    
    # Extract optimizer configuration
    opt_config = config.get('optimizer', {})
    optimizer_type = opt_config.get('type', 'hybrid_adam_sam')
    
    # Common parameters
    lr = opt_config.get('learning_rate', 0.001)
    weight_decay = opt_config.get('weight_decay', 0.01)
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    eps = opt_config.get('eps', 1e-8)
    
    # Create parameter groups with different learning rates
    param_groups = []
    param_group_config = opt_config.get('param_groups', {})
    
    # Group parameters by component
    component_params = defaultdict(list)
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            component_params['feature_extractor'].append(param)
        elif 'segmentation' in name:
            component_params['segmentation'].append(param)
        elif 'reference_embeddings' in name:
            component_params['reference_embeddings'].append(param)
        elif 'decoder' in name:
            component_params['decoder'].append(param)
        elif 'equation' in name:
            component_params['equation'].append(param)
        else:
            component_params['other'].append(param)
    
    # Create parameter groups with component-specific learning rates
    for component, params in component_params.items():
        if params:
            lr_mult = param_group_config.get(component, 1.0)
            param_groups.append({
                'params': params,
                'lr': lr * lr_mult,
                'name': component
            })
    
    # Create optimizer based on type
    if optimizer_type == 'advanced_adam':
        # Pure Advanced Adam without SAM
        optimizer = AdvancedAdam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=opt_config.get('use_amsgrad', False),
            nadam=opt_config.get('use_nadam', False),
            rectified=opt_config.get('use_rectified', False),
            decoupled_wd=opt_config.get('use_adamw', True),
            gradient_centralization=opt_config.get('gradient_centralization', False),
            lookahead=opt_config.get('use_lookahead', True),
            lookahead_k=opt_config.get('lookahead_k', 5),
            lookahead_alpha=opt_config.get('lookahead_alpha', 0.5),
            warmup_steps=opt_config.get('warmup_steps', 0),
            gradient_clipping=opt_config.get('gradient_clipping', None)
        )
    else:
        # Hybrid Adam-SAM optimizer
        adam_variant = 'adamw'  # Default
        if opt_config.get('use_nadam', False):
            adam_variant = 'nadam'
        elif opt_config.get('use_amsgrad', False):
            adam_variant = 'amsgrad'
        elif opt_config.get('use_rectified', False):
            adam_variant = 'rectified'
        
        optimizer = HybridAdamSAM(
            model.parameters(),  # HybridAdamSAM handles param grouping internally via base_optimizer
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sam_rho=opt_config.get('sam_rho', 0.05),
            sam_adaptive=opt_config.get('sam_adaptive', True),
            adam_variant=adam_variant,
            gradient_centralization=opt_config.get('gradient_centralization', False),
            warmup_steps=opt_config.get('warmup_steps', 0)
        )
    
    logger.info(f"Created {optimizer_type} optimizer with {len(param_groups)} parameter groups")
    logger.log_function_exit("create_hybrid_optimizer")
    
    return optimizer


# Learning rate schedulers specifically designed for Adam variants
class AdamWarmupCosineScheduler:
    """
    Cosine learning rate scheduler with linear warmup
    Specifically designed for Adam optimizers
    """
    
    def __init__(self,
                 optimizer: Union[AdvancedAdam, HybridAdamSAM],
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 0.0,
                 warmup_init_lr: float = 0.0):
        """
        Initialize scheduler
        
        Args:
            optimizer: Adam optimizer instance
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            warmup_init_lr: Initial learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_init_lr = warmup_init_lr
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.current_step = 0
        
        # Logger
        self.logger = get_logger("AdamWarmupCosineScheduler")
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.warmup_init_lr + (self.base_lrs[i] - self.warmup_init_lr) * progress
                group['lr'] = lr
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                group['lr'] = lr
        
        # Log learning rate changes
        if self.current_step % 100 == 0:
            avg_lr = sum(g['lr'] for g in self.optimizer.param_groups) / len(self.optimizer.param_groups)
            self.logger.info(f"Step {self.current_step} - Average LR: {avg_lr:.6f}")
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


# Test the optimizers
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing hybrid optimizers")
    print(f"[{datetime.now()}] Previous script: fiber_advanced_optimizers.py")
    
    # Create a simple test model with proper initialization
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 8 * 8, 10)
            self.pool = nn.MaxPool2d(2, 2)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    test_model = TestModel()
    dummy_input = torch.rand(4, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (4,))

    def closure():
        opt.zero_grad()
        output = test_model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_target)
        loss.backward()
        return loss

    print("\nTesting Hybrid Adam-SAM...")
    opt = HybridAdamSAM(test_model.parameters(), adam_variant="adamw")
    loss_before = closure().item()
    opt.step(closure)
    loss_after = closure().item()
    print(f"HybridAdamSAM: Loss before={loss_before:.4f}, Loss after={loss_after:.4f}")
    assert loss_after < loss_before, "HybridAdamSAM failed to reduce loss"

    print("\nTesting Adam warmup cosine scheduler...")
    base_opt = AdvancedAdam(test_model.parameters(), lr=0.01)
    scheduler = AdamWarmupCosineScheduler(base_opt, warmup_steps=100, total_steps=1000)
    lrs = []
    for _ in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    print(f"Scheduler LR range: {min(lrs):.6f} to {max(lrs):.6f}")
    assert lrs[0] > 0 and lrs[-1] == 0.0, "Scheduler did not behave as expected"
    
    print(f"\n[{datetime.now()}] Hybrid optimizers test completed")
    print(f"[{datetime.now()}] Next script: fiber_enhanced_trainer.py")