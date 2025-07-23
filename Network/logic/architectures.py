#!/usr/bin/env python3
"""
Advanced Architecture Components for Fiber Optics Neural Network
Implements SE blocks, Deformable Convolutions, and other advanced modules
Based on research: Hu et al., 2018; Dai et al., 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import math
from torchvision.ops import deform_conv2d

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config_loader import get_config
from core.logger import get_logger


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Network (Hu et al., 2018)
    Adaptive feature recalibration for fiber optics analysis
    
    Learns channel-wise attention weights to emphasize important features
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio (default: 16)
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing SEBlock for {channels} channels")
        
        self.logger = get_logger("SEBlock")
        self.logger.log_class_init("SEBlock", channels=channels, reduction=reduction)
        
        # Squeeze: global pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: channel attention
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.excitation:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Calibrated features [B, C, H, W]
        """
        b, c, _, _ = x.shape
        
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.squeeze(x).view(b, c)
        
        # Excitation: [B, C] -> [B, C] (attention weights)
        y = self.excitation(y).view(b, c, 1, 1)
        
        # Scale features by attention weights
        return x * y.expand_as(x)


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution (Dai et al., 2017)
    Adaptive to fiber optic geometry - learns to adjust receptive field
    
    Instead of fixed grid sampling, learns offsets for each position
    Better for irregular fiber optic patterns and defects
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, 
                 num_deformable_groups: int = 1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding added to input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: Whether to add bias
            num_deformable_groups: Number of deformable groups
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing DeformableConv2d")
        
        self.logger = get_logger("DeformableConv2d")
        self.logger.log_class_init("DeformableConv2d", 
                                 in_channels=in_channels, 
                                 out_channels=out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_deformable_groups = num_deformable_groups
        
        # Regular convolution weight
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Offset learning network
        # For each position, learn 2D offset for each kernel position
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * num_deformable_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Modulation learning network (optional importance weights)
        self.modulator_conv = nn.Conv2d(
            in_channels,
            num_deformable_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize offset to zero (no deformation initially)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # Initialize modulator to one (all positions equally important)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 1.)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply deformable convolution
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H_out, W_out]
        """
        # Learn offsets for each position
        offset = self.offset_conv(x)
        
        # Learn importance weights (modulation)
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        # Apply deformable convolution
        # Note: PyTorch's deform_conv2d expects offset shape [B, 2*K*K, H, W]
        # where K is kernel size
        output = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=modulator
        )
        
        return output


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018)
    Combines channel attention (SE-like) and spatial attention
    
    Better than SE alone for spatially-aware tasks like defect localization
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: Number of channels
            reduction: Channel reduction ratio
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing CBAM")
        
        self.logger = get_logger("CBAM")
        self.logger.log_class_init("CBAM", channels=channels)
        
        # Channel attention module
        self.channel_attention = self._build_channel_attention(channels, reduction)
        
        # Spatial attention module
        self.spatial_attention = self._build_spatial_attention(kernel_size)
        
    def _build_channel_attention(self, channels: int, reduction: int) -> nn.Module:
        """Build channel attention module"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def _build_spatial_attention(self, kernel_size: int) -> nn.Module:
        """Build spatial attention module"""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Channel attention
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # Spatial attention
        # Concatenate mean and max across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        sa_weight = self.spatial_attention(concat)
        x = x * sa_weight
        
        return x


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (Wang et al., 2020)
    Lightweight alternative to SE block without dimensionality reduction
    
    Better computational efficiency for real-time processing
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        """
        Args:
            channels: Number of channels
            gamma: Kernel size parameter
            b: Adaptive parameter
        """
        super().__init__()
        
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ECA attention"""
        b, c, _, _ = x.shape
        
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution across channels
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class ResidualSEBlock(nn.Module):
    """
    Residual block with SE attention
    Combines residual connections with channel attention
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, reduction: Optional[int] = 16):  # Changed type hint to Optional[int] to allow None
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for downsampling
            reduction: SE reduction ratio (can be None to disable SE)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if reduction is not None:  # Added conditional to handle reduction=None; previously, passing None to SEBlock would raise TypeError since SEBlock expects int
            self.se = SEBlock(out_channels, reduction)
        else:
            self.se = nn.Identity()  # Use Identity if no SE to avoid errors in forward pass; this ensures compatibility when use_se=False in FiberOpticsBackbone
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention (or identity if disabled)
        out = self.se(out)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out


class DeformableResidualBlock(nn.Module):
    """
    Residual block with deformable convolutions
    Adaptive receptive field for irregular fiber patterns
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for downsampling
        """
        super().__init__()
        
        self.conv1 = DeformableConv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = DeformableConv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class FiberOpticsBackbone(nn.Module):
    """
    Advanced backbone architecture combining all improvements
    Uses SE blocks, deformable convolutions, and multi-scale processing
    """
    
    def __init__(self, in_channels: int = 3, 
                 base_channels: int = 64,
                 num_blocks: List[int] = [2, 2, 2, 2],
                 use_deformable: bool = True,
                 use_se: bool = True):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            base_channels: Base number of channels
            num_blocks: Number of blocks at each scale
            use_deformable: Whether to use deformable convolutions
            use_se: Whether to use SE attention
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing FiberOpticsBackbone")
        
        self.logger = get_logger("FiberOpticsBackbone")
        self.logger.log_class_init("FiberOpticsBackbone")
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        
        for i, num_block in enumerate(num_blocks):
            out_ch = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2
            
            blocks = []
            for j in range(num_block):
                if use_deformable and j == num_block - 1:  # Last block in stage
                    block = DeformableResidualBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        stride if j == 0 else 1
                    )
                else:
                    block = ResidualSEBlock(
                        in_ch if j == 0 else out_ch,
                        out_ch,
                        stride if j == 0 else 1,
                        reduction=16 if use_se else None  # Use None to disable SE
                    )
                blocks.append(block)
            
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        
        # Global attention
        self.global_attention = CBAM(in_ch)
        
        # Store output channels for each stage
        self.out_channels = []
        for i in range(len(num_blocks)):
            self.out_channels.append(base_channels * (2 ** i))
        
        self.logger.info(f"FiberOpticsBackbone initialized with {len(self.stages)} stages")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning features at multiple scales
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Process through stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # Apply global attention to last feature map
        features[-1] = self.global_attention(features[-1])
        
        return features


# Test the modules
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing advanced architecture components")
    
    # Test SE block
    print("\nTesting SE Block...")
    se = SEBlock(64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    out = se(x)
    print(f"SE output shape: {out.shape}")
    
    # Test Deformable Conv
    print("\nTesting Deformable Conv2d...")
    deform_conv = DeformableConv2d(64, 128, kernel_size=3)
    x = torch.randn(2, 64, 32, 32)
    out = deform_conv(x)
    print(f"Deformable Conv output shape: {out.shape}")
    
    # Test CBAM
    print("\nTesting CBAM...")
    cbam = CBAM(64)
    x = torch.randn(2, 64, 32, 32)
    out = cbam(x)
    print(f"CBAM output shape: {out.shape}")
    
    # Test full backbone
    print("\nTesting FiberOpticsBackbone...")
    backbone = FiberOpticsBackbone()
    x = torch.randn(2, 3, 256, 256)
    features = backbone(x)
    print(f"Backbone output scales: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape}")
    
    print(f"[{datetime.now()}] Advanced architecture components test completed")
    print(f"[{datetime.now()}] Next script: fiber_hybrid_optimizer.py")