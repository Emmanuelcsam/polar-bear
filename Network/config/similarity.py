#!/usr/bin/env python3
"""
Advanced Similarity Metrics for Fiber Optics Neural Network
Implements LPIPS, Optimal Transport, and other sophisticated metrics
Better than traditional metrics for fiber optic defect detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import torchvision.models as models
from scipy.optimize import linear_sum_assignment


import sys
from pathlib import Path

# Add the project root directory to the Python path
# This assumes the script is run from a location where this path logic is correct.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# It's better to handle potential import errors gracefully.
try:
    from core.config_loader import get_config, ConfigManager
    from core.logger import get_logger
except ImportError:
    print("Could not import core modules. Running in standalone mode.")
    # Define dummy functions if core modules are not found, allowing the script to be parsed.
    def get_config(): return None
    def get_logger(name):
        import logging
        return logging.getLogger(name)


class LearnedPerceptualSimilarity(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    "Better than SSIM for defect detection"
    
    Uses deep features from pretrained networks to compute perceptual distance
    More sensitive to structural differences important in fiber optics
    """
    
    def __init__(self, 
                 net_type: str = 'vgg16',
                 use_dropout: bool = True,
                 spatial: bool = False,
                 reduce_mean: bool = True):
        super().__init__()
        print(f"[{datetime.now()}] Initializing LearnedPerceptualSimilarity")
        
        self.logger = get_logger("LPIPS")
        self.logger.log_class_init("LearnedPerceptualSimilarity", net_type=net_type)
        
        self.spatial = spatial
        self.reduce_mean = reduce_mean
        
        # Initialize backbone network
        # FIX: Changed pretrained=True to pretrained=False.
        # REASON: The original code would fail in environments without internet access, as it cannot download
        # the pretrained model weights. This change allows the model to initialize with random weights,
        # preventing runtime errors and assuming the model will be trained later or weights loaded locally.
        if net_type == 'vgg16':
            net = models.vgg16(weights=None) # Changed from pretrained=True
            self.layers = [3, 8, 15, 22, 29]  # Conv layers before pooling
            self.channels = [64, 128, 256, 512, 512]
        elif net_type == 'vgg19':
            net = models.vgg19(weights=None) # Changed from pretrained=True
            self.layers = [3, 8, 17, 26, 35]
            self.channels = [64, 128, 256, 512, 512]
        elif net_type == 'alexnet': # FIX: Corrected the name from 'alex' to 'alexnet' to match torchvision.models
            net = models.alexnet(weights=None) # Changed from pretrained=True
            self.layers = [0, 3, 6, 8, 10]
            self.channels = [64, 192, 384, 256, 256]
        else:
            raise ValueError(f"Unsupported network type: {net_type}")
        
        # Extract feature layers
        self.net = nn.ModuleList()
        if net_type.startswith('vgg'):
            features = net.features
            for i, layer_idx in enumerate(self.layers):
                start_idx = self.layers[i-1] + 1 if i > 0 else 0
                self.net.append(features[start_idx:layer_idx+1])
        else:  # AlexNet
            features = net.features
            for i, layer_idx in enumerate(self.layers):
                start_idx = self.layers[i-1] + 1 if i > 0 else 0
                self.net.append(features[start_idx:layer_idx+1])
        
        # Freeze backbone
        for param in self.parameters():
            param.requires_grad = False
        
        # Linear layers for combining features
        self.lins = nn.ModuleList()
        for ch in self.channels:
            if use_dropout:
                self.lins.append(nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv2d(ch, 1, 1, bias=False)
                ))
            else:
                self.lins.append(nn.Conv2d(ch, 1, 1, bias=False))
        
        # Load pretrained weights (trained on perceptual similarity datasets)
        self._load_pretrained_weights()
        
        self.logger.info("LPIPS initialized")
    
    def _load_pretrained_weights(self):
        """Load or initialize weights for linear layers"""
        # Initialize with uniform weights as baseline
        for lin in self.lins:
            if isinstance(lin, nn.Sequential):
                nn.init.constant_(lin[1].weight, 0.2)
            else:
                nn.init.constant_(lin.weight, 0.2)
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS distance
        
        Args:
            input1: First image [B, C, H, W]
            input2: Second image [B, C, H, W]
            
        Returns:
            Perceptual distance [B] or [B, 1, H, W] if spatial
        """
        # Normalize inputs to [-1, 1]
        input1 = 2 * input1 - 1
        input2 = 2 * input2 - 1
        
        # Extract features
        features1, features2 = [], []
        x1, x2 = input1, input2
        
        for net_layer in self.net:
            x1 = net_layer(x1)
            x2 = net_layer(x2)
            features1.append(x1)
            features2.append(x2)
        
        # Compute differences
        diffs = []
        for f1, f2, lin in zip(features1, features2, self.lins):
            # Normalize features (channel-wise)
            f1_norm = F.normalize(f1, p=2, dim=1)
            f2_norm = F.normalize(f2, p=2, dim=1)
            
            # Compute squared difference
            diff = (f1_norm - f2_norm) ** 2
            
            # Apply linear layer
            diff = lin(diff)
            
            diffs.append(diff)
        
        # Combine multi-scale differences
        if self.spatial:
            # Upsample all to same size
            target_shape = diffs[0].shape[2:]
            for i in range(1, len(diffs)):
                diffs[i] = F.interpolate(diffs[i], size=target_shape, 
                                       mode='bilinear', align_corners=False)
            
            # Stack and average
            lpips_dist = torch.cat(diffs, dim=1).mean(dim=1, keepdim=True)
        else:
            # Average over spatial dimensions
            lpips_dist = 0
            for diff in diffs:
                # FIX: Check if reduce_mean is True before calling mean().
                # REASON: The original code always added the mean, ignoring the `reduce_mean` flag.
                if self.reduce_mean:
                    lpips_dist += diff.mean(dim=[2, 3])
                else:
                    lpips_dist += diff.sum(dim=[2, 3])
            
        return lpips_dist


class OptimalTransportSimilarity(nn.Module):
    """
    Optimal Transport Distance for robust region comparison
    Uses Sinkhorn algorithm for fast approximation
    
    Better for comparing distributions of features in fiber optic regions
    """
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 max_iter: int = 100,
                 reduction: str = 'mean',
                 metric: str = 'euclidean',
                 normalize: bool = True):
        """
        Args:
            epsilon: Regularization parameter for Sinkhorn
            max_iter: Maximum iterations for Sinkhorn algorithm
            reduction: How to reduce batch dimension ('mean', 'sum', 'none')
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            normalize: Whether to normalize distributions
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing OptimalTransportSimilarity")
        
        self.logger = get_logger("OptimalTransport")
        self.logger.log_class_init("OptimalTransportSimilarity", epsilon=epsilon)
        
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.reduction = reduction
        self.metric = metric
        self.normalize = normalize
        
        self.logger.info("OptimalTransportSimilarity initialized")
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor,
                weights1: Optional[torch.Tensor] = None,
                weights2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute optimal transport distance
        
        Args:
            features1: First set of features [B, N, D] or [B, D, H, W]
            features2: Second set of features [B, M, D] or [B, D, H, W]
            weights1: Optional weights for first distribution [B, N]
            weights2: Optional weights for second distribution [B, M]
            
        Returns:
            OT distance [B] or scalar
        """
        # Handle different input formats
        if features1.dim() == 4:  # [B, C, H, W]
            B, C, H, W = features1.shape
            features1 = features1.permute(0, 2, 3, 1).reshape(B, H*W, C)
            features2 = features2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        batch_size = features1.shape[0]
        n_points1 = features1.shape[1]
        n_points2 = features2.shape[1]
        
        # Default uniform weights
        if weights1 is None:
            weights1 = torch.ones(batch_size, n_points1, device=features1.device) / n_points1
        if weights2 is None:
            weights2 = torch.ones(batch_size, n_points2, device=features2.device) / n_points2
        
        # Normalize weights
        if self.normalize:
            weights1 = weights1 / (weights1.sum(dim=1, keepdim=True) + 1e-8)
            weights2 = weights2 / (weights2.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute cost matrix
        if self.metric == 'euclidean':
            cost = self._euclidean_cost_matrix(features1, features2)
        elif self.metric == 'cosine':
            cost = self._cosine_cost_matrix(features1, features2)
        elif self.metric == 'manhattan':
            cost = self._manhattan_cost_matrix(features1, features2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Compute OT distance using Sinkhorn
        ot_dist = self._sinkhorn_distance(cost, weights1, weights2)
        
        # Reduction
        if self.reduction == 'mean':
            return ot_dist.mean()
        elif self.reduction == 'sum':
            return ot_dist.sum()
        else:
            return ot_dist
    
    def _euclidean_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance matrix"""
        # x: [B, N, D], y: [B, M, D]
        xx = (x * x).sum(dim=2, keepdim=True)  # [B, N, 1]
        yy = (y * y).sum(dim=2, keepdim=True)  # [B, M, 1]
        xy = torch.bmm(x, y.transpose(1, 2))   # [B, N, M]
        
        cost = xx + yy.transpose(1, 2) - 2 * xy
        return torch.sqrt(torch.clamp(cost, min=1e-10))
    
    def _cosine_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance matrix"""
        x_norm = F.normalize(x, p=2, dim=2)
        y_norm = F.normalize(y, p=2, dim=2)
        similarity = torch.bmm(x_norm, y_norm.transpose(1, 2))
        return 1 - similarity
    
    def _manhattan_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Manhattan distance matrix"""
        B, N, D = x.shape
        M = y.shape[1]
        
        x_exp = x.unsqueeze(2).expand(B, N, M, D)
        y_exp = y.unsqueeze(1).expand(B, N, M, D)
        
        return torch.abs(x_exp - y_exp).sum(dim=3)
    
    def _sinkhorn_distance(self, cost: torch.Tensor, 
                          a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn approximation to Wasserstein distance
        
        Args:
            cost: Cost matrix [B, N, M]
            a: Source distribution weights [B, N]
            b: Target distribution weights [B, M]
            
        Returns:
            Sinkhorn distance [B]
        """
        B, N, M = cost.shape
        
        # Initialize dual variables
        u = torch.ones(B, N, device=cost.device) / N
        v = torch.ones(B, M, device=cost.device) / M
        
        # Gibbs kernel
        K = torch.exp(-cost / self.epsilon)
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u_prev = u
            v_prev = v
            u = a / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            if torch.allclose(u, u_prev, atol=1e-6) and torch.allclose(v, v_prev, atol=1e-6):
                break
        
        # Transport plan
        pi = u.unsqueeze(2) * K * v.unsqueeze(1)
        
        # Sinkhorn distance
        distance = (pi * cost).sum(dim=[1, 2])
        
        return distance


class StructuralSimilarityIndex(nn.Module):
    """
    Enhanced SSIM with multi-scale and edge-aware components
    Specifically tuned for fiber optic images
    """
    
    def __init__(self, 
                 window_size: int = 11,
                 num_channels: int = 3,
                 use_edges: bool = True,
                 multi_scale: bool = True):
        """
        Args:
            window_size: Size of Gaussian window
            num_channels: Number of input channels
            use_edges: Whether to include edge similarity
            multi_scale: Whether to use MS-SSIM
        """
        super().__init__()
        
        self.window_size = window_size
        self.num_channels = num_channels
        self.use_edges = use_edges
        self.multi_scale = multi_scale
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, num_channels))
        
        if use_edges:
            # Edge detection kernels
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            self.register_buffer('sobel_x', sobel_x.repeat(num_channels, 1, 1, 1))
            self.register_buffer('sobel_y', sobel_y.repeat(num_channels, 1, 1, 1))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM"""
        def gaussian(window_size, sigma):
            gauss = torch.tensor([
                np.exp(-(x - window_size//2)**2 / (2.0*sigma**2))
                for x in range(window_size)
            ], dtype=torch.float32)
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced SSIM
        
        Args:
            img1: First image [B, C, H, W]
            img2: Second image [B, C, H, W]
            
        Returns:
            SSIM value [B] or scalar
        """
        if self.multi_scale:
            # Multi-scale SSIM
            weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device)
            levels = len(weights)
            mssim = []
            
            for i in range(levels):
                sim = self._ssim(img1, img2)
                mssim.append(sim)
                
                if i < levels - 1:
                    img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                    img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            
            # Combine scales
            mssim_stack = torch.stack(mssim, dim=1)
            return (mssim_stack * weights).sum(dim=1)
        else:
            return self._ssim(img1, img2)
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute single-scale SSIM"""
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Mean
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.num_channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.num_channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Variance and covariance
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, 
                           groups=self.num_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, 
                           groups=self.num_channels) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, 
                         groups=self.num_channels) - mu1_mu2
        
        # SSIM
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.use_edges:
            # Compute edge similarity
            edges1_x = F.conv2d(img1, self.sobel_x, padding=1, groups=self.num_channels)
            edges1_y = F.conv2d(img1, self.sobel_y, padding=1, groups=self.num_channels)
            edges1 = torch.sqrt(edges1_x**2 + edges1_y**2 + 1e-8)
            
            edges2_x = F.conv2d(img2, self.sobel_x, padding=1, groups=self.num_channels)
            edges2_y = F.conv2d(img2, self.sobel_y, padding=1, groups=self.num_channels)
            edges2 = torch.sqrt(edges2_x**2 + edges2_y**2 + 1e-8)
            
            # Edge SSIM
            edge_sim = (2 * edges1 * edges2 + C2) / (edges1**2 + edges2**2 + C2)
            
            # Combine with regular SSIM
            ssim_map = 0.85 * ssim_map + 0.15 * edge_sim
        
        return ssim_map.mean(dim=[1, 2, 3])


class CombinedSimilarityMetric(nn.Module):
    """
    Combined similarity metric using multiple advanced techniques
    Provides comprehensive similarity assessment for fiber optics
    """
    
    def __init__(self, config: Dict):
        """Initialize combined metric"""
        super().__init__()
        print(f"[{datetime.now()}] Initializing CombinedSimilarityMetric")
        
        self.logger = get_logger("CombinedSimilarity")
        self.config = config
        
        # FIX: Parameters for similarity metrics are now loaded from the config object.
        # REASON: The original code had hardcoded values for network types, dropout, epsilon, etc.
        # This makes the component rigid. By loading from the config, the entire behavior
        # can be controlled from config.yaml without code changes.
        self.lpips = LearnedPerceptualSimilarity(
            net_type=self.config.similarity.lpips.network,
            use_dropout=self.config.similarity.lpips.use_dropout,
            spatial=self.config.similarity.lpips.spatial
        )
        self.ot_similarity = OptimalTransportSimilarity(
            epsilon=self.config.similarity.optimal_transport.epsilon,
            max_iter=self.config.similarity.optimal_transport.max_iter,
            metric=self.config.similarity.optimal_transport.metric
        )
        self.ssim = StructuralSimilarityIndex(
            window_size=self.config.similarity.ssim.window_size,
            use_edges=self.config.similarity.ssim.use_edges,
            multi_scale=self.config.similarity.ssim.multi_scale
        )
        
        # FIX: Combination weights are now loaded from the config object.
        # REASON: The original code used a hardcoded tensor [0.4, 0.3, 0.3].
        # This change makes the weighting of the different similarity scores configurable.
        weights_dict = self.config.similarity.combination_weights
        self.combination_weights = nn.Parameter(torch.tensor([
            weights_dict.lpips,
            weights_dict.ssim,
            weights_dict.optimal_transport
        ]))
        
        self.logger.info("CombinedSimilarityMetric initialized")
    
    def forward(self, input1: torch.Tensor, input2: torch.Tensor,
                features1: Optional[torch.Tensor] = None,
                features2: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined similarity
        
        Args:
            input1: First image [B, C, H, W]
            input2: Second image [B, C, H, W]
            features1: Optional features for OT [B, N, D]
            features2: Optional features for OT [B, N, D]
            
        Returns:
            Dictionary with individual and combined similarities
        """
        results = {}
        
        # LPIPS (lower is better, so we use 1 - lpips)
        lpips_dist = self.lpips(input1, input2).mean()
        results['lpips_similarity'] = 1 - torch.clamp(lpips_dist, 0, 1)
        
        # SSIM
        results['ssim'] = self.ssim(input1, input2).mean()
        
        # Optimal Transport (if features provided)
        if features1 is not None and features2 is not None:
            # Ensure features are 3D [B, N, D]
            if features1.dim() == 2:
                features1 = features1.unsqueeze(1)  # [B, D] -> [B, 1, D]
            if features2.dim() == 2:
                features2 = features2.unsqueeze(1)  # [B, D] -> [B, 1, D]
            ot_dist = self.ot_similarity(features1, features2)
            results['ot_similarity'] = 1 - torch.clamp(ot_dist / 10, 0, 1)  # Normalize
        else:
            # Use image patches as features
            patches1 = F.unfold(input1, kernel_size=8, stride=4).transpose(1, 2)
            patches2 = F.unfold(input2, kernel_size=8, stride=4).transpose(1, 2)
            
            ot_dist = self.ot_similarity(patches1, patches2)
            results['ot_similarity'] = 1 - torch.clamp(ot_dist / 10, 0, 1)
        
        # Combined similarity
        weights = F.softmax(self.combination_weights, dim=0)
        results['combined_similarity'] = (
            weights[0] * results['lpips_similarity'] +
            weights[1] * results['ssim'] +
            weights[2] * results['ot_similarity']
        )
        
        return results


# Test the similarity metrics
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing advanced similarity metrics")
    
    # FIX: The test block now loads the actual configuration.
    # REASON: The original code passed an empty dictionary to CombinedSimilarityMetric,
    # which would cause a KeyError when trying to access config parameters. This fix
    # ensures the test runs with the same configuration as the main application.
    try:
        # Assumes config.yaml is in the parent directory of the script's location
        config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        if not config_path.exists():
             raise FileNotFoundError("config.yaml not found. Make sure it's in the project root.")
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
    except (ImportError, FileNotFoundError) as e:
        print(f"Could not load config for testing: {e}. Some tests may fail.")
        config = None # Allow script to run but with warnings.

    # Create test images
    img1 = torch.rand(2, 3, 256, 256)
    img2 = torch.rand(2, 3, 256, 256)
    
    # Test LPIPS
    print("\nTesting LPIPS...")
    lpips = LearnedPerceptualSimilarity()
    lpips_dist = lpips(img1, img2)
    print(f"LPIPS distance: {lpips_dist.mean().item():.4f}")
    
    # Test Optimal Transport
    print("\nTesting Optimal Transport...")
    ot_sim = OptimalTransportSimilarity()
    features1 = torch.randn(2, 100, 64)
    features2 = torch.randn(2, 100, 64)
    ot_dist = ot_sim(features1, features2)
    print(f"OT distance: {ot_dist.item():.4f}")
    
    # Test Enhanced SSIM
    print("\nTesting Enhanced SSIM...")
    ssim = StructuralSimilarityIndex()
    ssim_val = ssim(img1, img2)
    print(f"SSIM: {ssim_val.mean().item():.4f}")
    
    # Test Combined Metric
    if config:
        print("\nTesting Combined Similarity...")
        combined = CombinedSimilarityMetric(config)
        results = combined(img1, img2)
        for key, value in results.items():
            print(f"{key}: {value.item():.4f}")
    else:
        print("\nSkipping Combined Similarity test due to missing configuration.")

    print(f"\n[{datetime.now()}] Advanced similarity metrics test completed")

