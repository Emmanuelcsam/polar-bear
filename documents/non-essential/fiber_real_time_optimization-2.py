#!/usr/bin/env python3
"""
Real-time Optimization Techniques for Fiber Optics Neural Network
Implements Knowledge Distillation and Adaptive Computation Time
For faster inference while maintaining accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import time
from collections import OrderedDict

from fiber_config import get_config
from fiber_logger import get_logger
from fiber_advanced_architectures import FiberOpticsBackbone, SEBlock


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for training smaller, faster models
    "Train a smaller, faster model"
    
    Uses soft targets from teacher model to train student
    Balances between matching teacher outputs and true labels
    """
    
    def __init__(self, 
                 alpha: float = 0.7,
                 temperature: float = 4.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weight for distillation loss (1-alpha for hard target loss)
            temperature: Temperature for softening probabilities
            reduction: Loss reduction method
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing KnowledgeDistillationLoss")
        
        self.logger = get_logger("KnowledgeDistillation")
        self.logger.log_class_init("KnowledgeDistillationLoss", 
                                 alpha=alpha, temperature=temperature)
        
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        
        # For tracking distillation effectiveness
        self.stats = {
            'teacher_accuracy': 0,
            'student_accuracy': 0,
            'agreement_rate': 0,
            'updates': 0
        }
        
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Student model outputs [B, C]
            teacher_logits: Teacher model outputs [B, C]
            target: Optional true labels [B]
            
        Returns:
            Combined distillation loss
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        distillation_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction=self.reduction
        ) * (self.temperature ** 2)  # Scale by T^2 as per paper
        
        # Hard target loss if labels provided
        if target is not None:
            hard_loss = F.cross_entropy(student_logits, target, reduction=self.reduction)
            
            # Combined loss
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
            
            # Track statistics
            with torch.no_grad():
                teacher_pred = teacher_logits.argmax(dim=1)
                student_pred = student_logits.argmax(dim=1)
                
                self.stats['teacher_accuracy'] += (teacher_pred == target).float().mean().item()
                self.stats['student_accuracy'] += (student_pred == target).float().mean().item()
                self.stats['agreement_rate'] += (teacher_pred == student_pred).float().mean().item()
                self.stats['updates'] += 1
                
                # Log periodically
                if self.stats['updates'] % 100 == 0:
                    self.logger.info(
                        f"Distillation stats - Teacher acc: {self.stats['teacher_accuracy']/100:.2%}, "
                        f"Student acc: {self.stats['student_accuracy']/100:.2%}, "
                        f"Agreement: {self.stats['agreement_rate']/100:.2%}"
                    )
                    self.stats = {k: 0 for k in self.stats}
        else:
            total_loss = distillation_loss
        
        return total_loss


class AdaptiveComputationModule(nn.Module):
    """
    Adaptive Computation Time (ACT) module
    "Skip unnecessary computations"
    
    Dynamically decides how many layers to process based on input complexity
    Saves computation for simple cases while using full capacity for complex ones
    """
    
    def __init__(self, 
                 hidden_size: int,
                 threshold: float = 0.95,
                 max_steps: int = 10,
                 epsilon: float = 0.01):
        """
        Args:
            hidden_size: Size of hidden representations
            threshold: Cumulative halt threshold
            max_steps: Maximum computation steps
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing AdaptiveComputationModule")
        
        self.logger = get_logger("AdaptiveComputation")
        self.logger.log_class_init("AdaptiveComputationModule", threshold=threshold)
        
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.max_steps = max_steps
        self.epsilon = epsilon
        
        # Halting mechanism
        self.halting_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Computation tracking
        self.computation_stats = {
            'avg_steps': 0,
            'min_steps': float('inf'),
            'max_steps': 0,
            'total_samples': 0
        }
        
    def forward(self, 
                x: torch.Tensor,
                compute_fn: callable) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply adaptive computation
        
        Args:
            x: Input tensor [B, ...]
            compute_fn: Function to apply at each step
            
        Returns:
            Output tensor and computation info
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize
        output = torch.zeros_like(x)
        cumulative_halt = torch.zeros(batch_size, 1, device=device)
        remainders = torch.zeros(batch_size, 1, device=device)
        n_updates = torch.zeros(batch_size, 1, device=device)
        
        # Flags for which samples are still computing
        still_running = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        halting_probs = []
        intermediates = []
        
        for step in range(self.max_steps):
            # Only process samples that haven't halted
            if not still_running.any():
                break
            
            # Apply computation
            x = compute_fn(x)
            intermediates.append(x)
            
            # Compute halting probability
            halt_prob = self.halting_layer(x.view(batch_size, -1).mean(dim=1, keepdim=True))
            halting_probs.append(halt_prob)
            
            # Update still running samples
            still_running_mask = still_running.unsqueeze(1).float()
            
            # Check if we should halt
            new_cumulative = cumulative_halt + halt_prob * still_running_mask
            
            # Samples that will halt this step
            halt_now = (new_cumulative >= self.threshold) & still_running
            
            # Update remainders for halting samples
            remainders = torch.where(
                halt_now.unsqueeze(1),
                1 - cumulative_halt,
                remainders
            )
            
            # Update cumulative halt
            cumulative_halt = torch.where(
                still_running.unsqueeze(1),
                torch.min(new_cumulative, torch.ones_like(new_cumulative)),
                cumulative_halt
            )
            
            # Update output as weighted sum
            update_weight = torch.where(
                halt_now.unsqueeze(1),
                remainders,
                halt_prob * still_running_mask
            )
            
            # Expand update_weight to match x dimensions
            while update_weight.dim() < x.dim():
                update_weight = update_weight.unsqueeze(-1)
            
            output = output + update_weight * x
            n_updates = n_updates + still_running_mask
            
            # Update still running
            still_running = still_running & ~halt_now.squeeze(1)
        
        # Handle any remaining samples (those that didn't halt)
        if still_running.any():
            remainder_weight = 1 - cumulative_halt
            while remainder_weight.dim() < x.dim():
                remainder_weight = remainder_weight.unsqueeze(-1)
            
            still_running_mask = still_running.unsqueeze(1).float()
            while still_running_mask.dim() < x.dim():
                still_running_mask = still_running_mask.unsqueeze(-1)
            
            output = output + remainder_weight * x * still_running_mask
        
        # Update statistics
        with torch.no_grad():
            steps_taken = n_updates.squeeze(1)
            self.computation_stats['avg_steps'] += steps_taken.mean().item()
            self.computation_stats['min_steps'] = min(
                self.computation_stats['min_steps'], 
                steps_taken.min().item()
            )
            self.computation_stats['max_steps'] = max(
                self.computation_stats['max_steps'],
                steps_taken.max().item()
            )
            self.computation_stats['total_samples'] += batch_size
        
        # Computation info
        info = {
            'halting_probabilities': torch.stack(halting_probs, dim=1),
            'n_steps': n_updates,
            'remainders': remainders,
            'intermediates': intermediates
        }
        
        return output, info
    
    def get_computation_stats(self) -> Dict[str, float]:
        """Get computation statistics"""
        if self.computation_stats['total_samples'] == 0:
            return {'avg_steps': 0, 'min_steps': 0, 'max_steps': 0}
        
        return {
            'avg_steps': self.computation_stats['avg_steps'] / self.computation_stats['total_samples'],
            'min_steps': self.computation_stats['min_steps'],
            'max_steps': self.computation_stats['max_steps']
        }


class EfficientFiberOpticsNetwork(nn.Module):
    """
    Efficient version of the fiber optics network for real-time processing
    Uses knowledge distillation and adaptive computation
    """
    
    def __init__(self, 
                 num_classes: int = 3,
                 base_channels: int = 32,  # Smaller than original
                 use_adaptive: bool = True):
        """
        Args:
            num_classes: Number of output classes
            base_channels: Base number of channels (smaller for efficiency)
            use_adaptive: Whether to use adaptive computation
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing EfficientFiberOpticsNetwork")
        
        self.logger = get_logger("EfficientNetwork")
        self.logger.log_class_init("EfficientFiberOpticsNetwork")
        
        self.use_adaptive = use_adaptive
        
        # Efficient backbone (smaller)
        self.backbone = self._create_efficient_backbone(base_channels)
        
        # Adaptive computation modules
        if use_adaptive:
            self.adaptive_modules = nn.ModuleList([
                AdaptiveComputationModule(base_channels * (2**i))
                for i in range(4)
            ])
        
        # Lightweight heads
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, num_classes, 1)
        )
        
        self.anomaly_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 2, 1),
            nn.Sigmoid()
        )
        
        # Reference embedding (smaller)
        self.reference_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 128)  # Smaller embedding
        )
        
        self.logger.info("EfficientFiberOpticsNetwork initialized")
    
    def _create_efficient_backbone(self, base_channels: int) -> nn.ModuleList:
        """Create efficient backbone with depthwise separable convolutions"""
        layers = nn.ModuleList()
        
        # Initial layer
        layers.append(nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True)
        ))
        
        # Efficient blocks using depthwise separable convolutions
        in_channels = base_channels
        for i in range(4):
            out_channels = base_channels * (2 ** i)
            
            # Depthwise separable block
            block = nn.Sequential(
                # Depthwise
                nn.Conv2d(in_channels, in_channels, 3, 2 if i > 0 else 1, 1, 
                         groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                
                # Pointwise
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                
                # SE attention (lightweight)
                SEBlock(out_channels, reduction=8)
            )
            
            layers.append(block)
            in_channels = out_channels
        
        return layers
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional adaptive computation
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary of outputs
        """
        batch_size = x.shape[0]
        
        # Initial processing
        features = [self.backbone[0](x)]
        
        # Process through backbone with adaptive computation
        for i, layer in enumerate(self.backbone[1:]):
            if self.use_adaptive and i < len(self.adaptive_modules):
                # Adaptive computation
                feat, info = self.adaptive_modules[i](
                    features[-1],
                    lambda f: layer(f)
                )
                features.append(feat)
            else:
                # Regular computation
                features.append(layer(features[-1]))
        
        # Final features
        final_features = features[-1]
        
        # Segmentation
        seg_logits = self.segmentation_head(final_features)
        
        # Upsample to original size
        seg_logits = F.interpolate(seg_logits, size=x.shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        # Anomaly detection
        anomaly_score = self.anomaly_head(final_features)
        
        # Reference embedding
        reference_embedding = self.reference_encoder(final_features)
        
        # Compute inference time
        outputs = {
            'segmentation': seg_logits,
            'anomaly_score': anomaly_score,
            'reference_embedding': reference_embedding,
            'features': features
        }
        
        # Add adaptive computation stats if used
        if self.use_adaptive:
            computation_stats = []
            for module in self.adaptive_modules:
                computation_stats.append(module.get_computation_stats())
            outputs['computation_stats'] = computation_stats
        
        return outputs


class ModelCompressor:
    """
    Utility class for model compression techniques
    Including pruning, quantization, and knowledge distillation
    """
    
    def __init__(self, logger=None):
        """Initialize model compressor"""
        self.logger = logger or get_logger("ModelCompressor")
        
    def prune_model(self, 
                    model: nn.Module, 
                    prune_ratio: float = 0.3) -> nn.Module:
        """
        Prune model weights based on magnitude
        
        Args:
            model: Model to prune
            prune_ratio: Fraction of weights to prune
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Pruning model with ratio {prune_ratio}")
        
        import torch.nn.utils.prune as prune
        
        # Get all Conv2d and Linear layers
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                modules_to_prune.append((module, 'weight'))
        
        # Apply global magnitude pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio,
        )
        
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        # Count remaining parameters
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        
        self.logger.info(f"Pruning complete. Parameters: {nonzero_params}/{total_params} "
                        f"({nonzero_params/total_params:.1%} remaining)")
        
        return model
    
    def quantize_model(self, 
                      model: nn.Module, 
                      backend: str = 'qnnpack') -> nn.Module:
        """
        Quantize model to int8 for faster inference
        
        Args:
            model: Model to quantize
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        self.logger.info("Quantizing model")
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Prepare for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Fuse modules (Conv + BN + ReLU)
        model = torch.quantization.fuse_modules(model, 
            [['conv', 'bn', 'relu']], inplace=True)
        
        # Prepare model
        torch.quantization.prepare(model, inplace=True)
        
        # Would need calibration data here in practice
        # model(calibration_data)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        self.logger.info("Quantization complete")
        
        return model


def create_efficient_model(teacher_model: Optional[nn.Module] = None,
                          config: Optional[Dict] = None) -> EfficientFiberOpticsNetwork:
    """
    Create efficient model for real-time processing
    
    Args:
        teacher_model: Optional teacher model for initialization
        config: Configuration dictionary
        
    Returns:
        Efficient model instance
    """
    logger = get_logger("EfficientModelFactory")
    logger.log_function_entry("create_efficient_model")
    
    # Create efficient model
    model = EfficientFiberOpticsNetwork(
        num_classes=3,
        base_channels=32,
        use_adaptive=True
    )
    
    # Initialize from teacher if provided
    if teacher_model is not None:
        logger.info("Initializing from teacher model")
        # Copy relevant weights with adaptation
        # This would involve matching layers and adapting dimensions
    
    logger.log_function_exit("create_efficient_model")
    
    return model


# Test the modules
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing real-time optimization modules")
    
    # Test Knowledge Distillation
    print("\nTesting Knowledge Distillation...")
    kd_loss = KnowledgeDistillationLoss()
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    target = torch.randint(0, 10, (4,))
    loss = kd_loss(student_logits, teacher_logits, target)
    print(f"KD Loss: {loss.item():.4f}")
    
    # Test Adaptive Computation
    print("\nTesting Adaptive Computation...")
    act = AdaptiveComputationModule(hidden_size=64)
    x = torch.randn(4, 64)
    
    def compute_fn(x):
        return x + torch.randn_like(x) * 0.1
    
    output, info = act(x, compute_fn)
    print(f"Output shape: {output.shape}")
    print(f"Computation stats: {act.get_computation_stats()}")
    
    # Test Efficient Network
    print("\nTesting Efficient Network...")
    efficient_net = EfficientFiberOpticsNetwork()
    x = torch.randn(2, 3, 256, 256)
    
    # Time inference
    start = time.time()
    outputs = efficient_net(x)
    end = time.time()
    
    print(f"Inference time: {(end - start)*1000:.1f}ms")
    print(f"Segmentation shape: {outputs['segmentation'].shape}")
    print(f"Anomaly score shape: {outputs['anomaly_score'].shape}")
    
    print(f"[{datetime.now()}] Real-time optimization test completed")
    print(f"[{datetime.now()}] Next script: fiber_advanced_config.py")