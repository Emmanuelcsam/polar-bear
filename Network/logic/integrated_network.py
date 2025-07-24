#!/usr/bin/env python3
"""
Enhanced Integrated Neural Network for Fiber Optics Analysis
Combines all advanced optimization techniques and computational improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import time
from einops import rearrange, repeat
from torch.cuda.amp import autocast

# Import all our advanced modules
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.config_loader import get_config
from core.logger import get_logger
from data.tensor_processor import TensorProcessor
from data.feature_extractor import MultiScaleFeatureExtractor, TrendAnalyzer
from data.reference_comparator import SimilarityCalculator
from logic.anomaly_detector import AnomalyDetector
from logic.architectures import (
    SEBlock, DeformableConv2d, CBAM, FiberOpticsBackbone,
    ResidualSEBlock, DeformableResidualBlock
)
from utilities.losses import CombinedAdvancedLoss
# from config.similarity import CombinedSimilarityMetric  # Moved to avoid circular import
# from fiber_real_time_optimization import AdaptiveComputationModule
from core.config_loader import get_config as get_advanced_config


class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on important regions
    Research: Oktay et al., 2018 - Attention U-Net
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature channels in gating signal
            F_l: Number of feature channels in input features
            F_int: Number of intermediate channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention gating
        
        Args:
            g: Gating signal [B, F_g, H, W]
            x: Input features [B, F_l, H, W]
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MixedPrecisionLayer(nn.Module):
    """
    Layer wrapper for mixed precision training
    Automatically handles FP16/FP32 conversion for better performance
    """
    
    def __init__(self, layer: nn.Module, use_amp: bool = True):
        """
        Args:
            layer: Layer to wrap
            use_amp: Whether to use automatic mixed precision
        """
        super().__init__()
        self.layer = layer
        self.use_amp = use_amp
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional mixed precision"""
        if self.use_amp and x.is_cuda:
            with autocast():
                return self.layer(x)
        else:
            return self.layer(x)


class StochasticDepth(nn.Module):
    """
    Stochastic Depth for regularization
    Research: Huang et al., 2016
    Randomly drops entire residual blocks during training
    """
    
    def __init__(self, drop_prob: float = 0.1):
        """
        Args:
            drop_prob: Probability of dropping the block
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth"""
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        # Create random tensor
        random_tensor = keep_prob
        random_tensor += torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        
        return x * binary_tensor / keep_prob


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer for domain adaptation
    Research: Ganin et al., 2016 - Domain-Adversarial Training
    """
    
    def __init__(self, lambda_val: float = 1.0):
        """
        Args:
            lambda_val: Gradient reversal weight
        """
        super().__init__()
        self.lambda_val = lambda_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity function)"""
        return GradientReversalFunction.apply(x, self.lambda_val)

# Added custom autograd Function to properly implement gradient reversal
# Original code incorrectly overrode nn.Module's backward method, which isn't how PyTorch custom gradients work
# This fixes the gradient reversal by using a proper autograd.Function
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class NeuralArchitectureCell(nn.Module):
    """
    Differentiable Architecture Search (DARTS) cell
    Research: Liu et al., 2019
    Learns optimal architecture during training
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Define candidate operations
        self.ops = nn.ModuleList([
            # Regular convolution
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            # Depthwise separable convolution
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1)
            ),
            # Dilated convolution
            nn.Conv2d(in_channels, out_channels, 3, 1, 2, dilation=2),
            # Max pooling followed by conv
            nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_channels, out_channels, 1)
            ),
            # Skip connection
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        ])
        
        # Architecture parameters (to be learned)
        self.alpha = nn.Parameter(torch.randn(len(self.ops)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted sum of operations
        """
        # Compute weights using softmax
        weights = F.softmax(self.alpha, dim=0)
        
        # Weighted sum of all operations
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        
        return output


class EnhancedIntegratedNetwork(nn.Module):
    """
    Complete enhanced neural network incorporating all advanced techniques
    Combines all research-based improvements for optimal performance
    """
    
    def __init__(self):
        """Initialize enhanced network with all improvements"""
        super().__init__()
        print(f"[{datetime.now()}] Initializing EnhancedIntegratedNetwork")
        
        self.config = get_advanced_config()
        self.logger = get_logger("EnhancedIntegratedNetwork")
        self.logger.log_class_init("EnhancedIntegratedNetwork")
        
        # Device setup
        self.device = self.config.get_device()
        
        # Advanced backbone with all architectural improvements
        self.backbone = FiberOpticsBackbone(
            in_channels=3,
            base_channels=self.config.model.base_channels,
            num_blocks=self.config.model.num_blocks,
            use_deformable=self.config.model.use_deformable_conv,
            use_se=self.config.model.use_se_blocks
        )
        
        # Multi-scale feature extraction with improvements
        self.feature_extractor = MultiScaleFeatureExtractor()
        
        # Trend analyzer
        self.trend_analyzer = TrendAnalyzer()
        
        # Neural Architecture Search cells
        if self.config.advanced.use_nas:
            self.nas_cells = nn.ModuleList([
                NeuralArchitectureCell(
                    self.config.model.base_channels * (2**i),
                    self.config.model.base_channels * (2**i)
                )
                for i in range(4)
            ])
        
        # Adaptive computation modules (placeholder for now)
        if self.config.model.use_adaptive_computation:
            # Create a single adaptive module that handles variable channel sizes
            self.adaptive_module = nn.Sequential(
                nn.Conv2d(1, 1, 1, groups=1),  # Channel-wise 1x1 conv
                nn.ReLU()
            )
        
        # Enhanced segmentation network with attention
        # After attention gates, we only get backbone channels: [128, 256, 512, 1024]
        # Total: 128 + 256 + 512 + 1024 = 1920
        seg_channels = sum(self.backbone.out_channels)  # 1920
        self.segmentation_net = nn.ModuleList([
            MixedPrecisionLayer(
                nn.Conv2d(seg_channels, 256, 1),
                use_amp=self.config.training.use_amp
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),  # Attention module
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            StochasticDepth(0.1),  # Regularization
            nn.Conv2d(128, self.config.model.num_classes, 1)
        ])
        
        # Reference comparison with enhanced similarity
        self.reference_encoder = nn.Sequential(
            nn.Conv2d(seg_channels, 512, 1),
            nn.ReLU(inplace=True),
            SEBlock(512, reduction=8),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.config.model.embedding_dim)
        )
        
        # Learnable reference embeddings with initialization
        self.reference_embeddings = nn.Parameter(
            torch.randn(self.config.model.num_reference_embeddings, 
                       self.config.model.embedding_dim) * 0.01
        )
        
        # Advanced anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Enhanced decoder with skip connections
        self.decoder = self._build_enhanced_decoder(seg_channels)
        
        # Equation parameter network with constraints
        self.equation_adjuster = nn.Sequential(
            nn.Linear(5, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Sigmoid()  # Ensures positive coefficients
        )
        
        # Combined similarity metric
        # Import postponed to avoid circular import issues
        self.similarity_metric = None  # Will be initialized when needed
        
        # Loss function
        self.loss_fn = CombinedAdvancedLoss(self.config)
        
        # Statistical components for comprehensive analysis
        self.statistical_feature_dim = 88
        self.statistical_extractor = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels[-1], 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.statistical_feature_dim)
        )
        
        # Zone parameter predictors
        self.zone_predictor = nn.Sequential(
            nn.Linear(self.statistical_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # core_radius, cladding_radius, ratio
        )
        
        # Consensus mechanism
        self.consensus_voter = nn.Sequential(
            nn.Conv2d(7, 16, 1),  # 7 method predictions
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)   # Final consensus for 3 regions
        )
        
        # Method accuracy tracker
        self.register_buffer('method_scores', torch.ones(7))
        self.method_score_ema = 0.9
        
        # Attention gates for feature fusion - Fixed to use backbone output channels
        self.attention_gates = nn.ModuleList([
            AttentionGate(
                F_g=self.backbone.out_channels[i],
                F_l=self.backbone.out_channels[i],
                F_int=self.backbone.out_channels[i] // 2
            )
            for i in range(len(self.backbone.out_channels))
        ])
        
        # Domain adaptation layer (if using)
        if self.config.experimental.use_graph_features:
            self.domain_classifier = nn.Sequential(
                GradientReversal(lambda_val=0.1),
                nn.Linear(self.config.model.embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # Source/target domain
            )
        
        # Initialize weights properly
        # self._init_weights()  # Commented out due to uninitialized parameter issues
        
        self.logger.info("EnhancedIntegratedNetwork initialized with all improvements")
        print(f"[{datetime.now()}] EnhancedIntegratedNetwork ready")
    
    def _build_enhanced_decoder(self, in_channels: int) -> nn.Module:
        """Build enhanced decoder with skip connections"""
        decoder_channels = [512, 256, 128, 64, 32, 3]
        decoder = nn.ModuleList()
        
        in_ch = in_channels
        for i, out_ch in enumerate(decoder_channels):
            if i < len(decoder_channels) - 1:
                # Regular decoder block
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    SEBlock(out_ch, reduction=8) if i < 3 else nn.Identity()
                )
            else:
                # Final layer
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.Sigmoid()
                )
            
            decoder.append(block)
            in_ch = out_ch
        
        return decoder
    
    def _init_weights(self):
        """Initialize weights using best practices"""
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Added check for initialized weight before applying init
                # Original code had try-except pass, but some modules (e.g., pretrained VGG) may already be initialized or have incompatible shapes; this ensures safe initialization
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    try:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    except ValueError:
                        # Skip if shape incompatible (e.g., 1x1 conv with no fan_out)
                        pass
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
                    try:
                        nn.init.xavier_uniform_(m.weight)
                    except ValueError:
                        pass  # Skip if shape incompatible
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_fn)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with all improvements and optimizations
        """
        batch_size = x.shape[0]
        computation_start = time.time()
        
        # Enable automatic mixed precision if configured
        use_amp = self.config.optimization.mixed_precision_training if hasattr(self.config, 'optimization') else True
        device = x.device
        
        # Calculate gradient and position information
        tensor_processor = TensorProcessor()
        gradient_info = tensor_processor.calculate_gradient_intensity(x)
        position_info = tensor_processor.calculate_pixel_positions(x.shape)
        
        # Extract features through advanced backbone
        backbone_features = self.backbone(x)
        
        # Apply NAS cells if enabled
        if self.config.advanced.use_nas:
            for i, (feat, nas_cell) in enumerate(zip(backbone_features, self.nas_cells)):
                backbone_features[i] = nas_cell(feat)
        
        # Multi-scale feature extraction with simultaneous analysis
        extraction_results = self.feature_extractor(x, gradient_info, position_info)
        
        # Combine backbone and extracted features
        combined_features = self._combine_features(backbone_features, extraction_results['features'])
        
        # Apply adaptive computation if enabled
        if self.config.model.use_adaptive_computation:
            combined_features = self._apply_adaptive_computation(combined_features)
        
        # Segmentation with attention
        seg_features = combined_features
        for layer in self.segmentation_net:
            if isinstance(layer, nn.Module):
                seg_features = layer(seg_features)
        segmentation_logits = seg_features
        
        # Upsample segmentation logits to original size for trend analysis
        segmentation_logits_upsampled = F.interpolate(
            segmentation_logits, 
            size=(x.shape[2], x.shape[3]),  # Original image size
            mode='bilinear', 
            align_corners=False
        )
        
        region_probs = F.softmax(segmentation_logits_upsampled, dim=1)
        
        # Trend analysis
        trend_results = self.trend_analyzer(
            gradient_info['gradient_map'],
            position_info['radial_positions'],
            region_probs
        )
        
        # Reference comparison with enhanced embedding
        feature_embedding = self.reference_encoder(combined_features)
        
        # Advanced similarity computation
        similarity_results = self._compute_advanced_similarity(
            feature_embedding, x, combined_features
        )
        
        # Enhanced reconstruction with skip connections
        reconstructed = self._reconstruct_with_skip(combined_features, backbone_features)
        
        # Advanced anomaly detection
        anomaly_results = self._detect_anomalies_advanced(
            x, reconstructed, gradient_info, region_probs
        )
        
        # Compute final similarity with learned coefficients
        final_similarity = self._compute_final_similarity(
            similarity_results, trend_results, anomaly_results, 
            region_probs, reconstructed, x
        )
        
        # Computation time
        computation_time = time.time() - computation_start
        
        return {
            # Core outputs
            'segmentation': segmentation_logits,  # Keep original resolution for output
            'region_probs': F.softmax(segmentation_logits, dim=1),  # Downsampled probs
            'anomaly_map': anomaly_results['final_anomaly_map'],
            'reconstruction': reconstructed,
            'final_similarity': final_similarity['score'],
            
            # Detailed results
            'similarity_results': similarity_results,
            'trend_results': trend_results,
            'anomaly_results': anomaly_results,
            'coefficients': final_similarity['coefficients'],
            
            # Features for analysis
            'backbone_features': backbone_features,
            'multi_scale_features': extraction_results['features'],
            'combined_features': combined_features,
            'feature_embedding': feature_embedding,
            
            # Metadata
            'computation_time': computation_time,
            'device': str(self.device),
            
            # Thresholds
            'meets_threshold': final_similarity['score'] > self.config.similarity.threshold,
            'best_reference_indices': similarity_results['best_indices']
        }
    
    def _combine_features(self, backbone_features: List[torch.Tensor], 
                         extracted_features: List[torch.Tensor]) -> torch.Tensor:
        """Combine features from backbone and extractor"""
        # Resize all to same size
        target_size = backbone_features[-1].shape[-2:]
        
        combined = []
        # Only combine features that have matching indices
        num_features = min(len(backbone_features), len(extracted_features), len(self.attention_gates))
        
        for i in range(num_features):
            bb_feat = backbone_features[i]
            ex_feat = extracted_features[i]
            
            # Resize if needed
            if bb_feat.shape[-2:] != target_size:
                bb_feat = F.interpolate(bb_feat, size=target_size, mode='bilinear', align_corners=False)
            if ex_feat.shape[-2:] != target_size:
                ex_feat = F.interpolate(ex_feat, size=target_size, mode='bilinear', align_corners=False)
            
            # Match channels if they don't match
            if bb_feat.shape[1] != ex_feat.shape[1]:
                # Use 1x1 conv to match channels
                channel_matcher = nn.Conv2d(ex_feat.shape[1], bb_feat.shape[1], 1).to(ex_feat.device)
                ex_feat = channel_matcher(ex_feat)
            
            # Apply attention gate
            attn_gate = self.attention_gates[i]
            gated_feat = attn_gate(bb_feat, ex_feat)
            combined.append(gated_feat)
        
        # Add any remaining backbone features without attention
        for i in range(num_features, len(backbone_features)):
            feat = backbone_features[i]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            combined.append(feat)
        
        # Concatenate all features
        return torch.cat(combined, dim=1)
    
    def _apply_adaptive_computation(self, features: torch.Tensor) -> torch.Tensor:
        """Apply adaptive computation to features"""
        # This is a simplified version - full implementation would process
        # through multiple layers adaptively
        if hasattr(self, 'adaptive_module'):
            # Apply channel-wise adaptive processing
            # Process each channel independently
            b, c, h, w = features.shape
            processed = []
            for i in range(c):
                channel_feat = features[:, i:i+1, :, :]
                processed_channel = self.adaptive_module(channel_feat)
                processed.append(processed_channel)
            return torch.cat(processed, dim=1)
        return features
    
    def _compute_advanced_similarity(self, embedding: torch.Tensor, 
                                   image: torch.Tensor,
                                   features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute similarity using advanced metrics"""
        # Reference similarity
        ref_similarities = F.cosine_similarity(
            embedding.unsqueeze(1),
            self.reference_embeddings.unsqueeze(0),
            dim=2
        )
        
        best_similarities, best_indices = ref_similarities.max(dim=1)
        
        # Get best reference features (would need actual reference images in practice)
        # Original code used placeholder randn_like; fixed to use actual reference embeddings for similarity (assuming references are pre-computed embeddings)
        # This avoids invalid randn_like which doesn't represent real references
        best_ref_idx = best_indices[0].item()  # Assume batch=1 for simplicity; extend for larger batches
        reference_embedding = self.reference_embeddings[best_ref_idx].unsqueeze(0)
        
        # Compute advanced similarities
        if self.similarity_metric is None:
            # Lazy import to avoid circular dependencies
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config.similarity import CombinedSimilarityMetric
            self.similarity_metric = CombinedSimilarityMetric(self.config)
        advanced_sim = self.similarity_metric(image, image, embedding, reference_embedding)  # Use self as placeholder for ref_image
        
        return {
            'reference_similarities': ref_similarities,
            'best_similarities': best_similarities,
            'best_indices': best_indices,
            'advanced_similarities': advanced_sim
        }
    
    def _reconstruct_with_skip(self, features: torch.Tensor, 
                              skip_features: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct with skip connections"""
        x = features
        
        # Decode with skip connections
        skip_idx = len(skip_features) - 1
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            
            # Add skip connection if available
            if skip_idx >= 0 and i < len(self.decoder) - 1:
                skip_feat = skip_features[skip_idx]
                # Resize skip feature to match
                if skip_feat.shape[-2:] != x.shape[-2:]:
                    skip_feat = F.interpolate(skip_feat, size=x.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                # Match channels
                if skip_feat.shape[1] != x.shape[1]:
                    skip_feat = nn.Conv2d(skip_feat.shape[1], x.shape[1], 1, 
                                        device=x.device)(skip_feat)
                
                x = x + 0.1 * skip_feat  # Weighted skip connection
                skip_idx -= 1
        
        return x
    
    def _detect_anomalies_advanced(self, original: torch.Tensor,
                                  reconstructed: torch.Tensor,
                                  gradient_info: Dict,
                                  region_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Advanced anomaly detection"""
        # Ensure same size for reconstruction error
        if reconstructed.shape != original.shape:
            reconstructed = F.interpolate(reconstructed, size=original.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        # Basic reconstruction error
        reconstruction_diff = torch.abs(reconstructed - original)
        
        # Use anomaly detector
        anomaly_results = self.anomaly_detector(
            original, reconstructed, 
            gradient_info['gradient_map'],
            self._get_region_boundaries(region_probs)
        )
        
        # Uncertainty-based anomaly detection
        if self.config.advanced.use_uncertainty:
            uncertainty = self._compute_uncertainty(original)
            anomaly_results['anomaly_map'] = anomaly_results['anomaly_map'] * (1 + uncertainty)
        
        # Final anomaly map combining all sources
        final_anomaly = torch.max(
            anomaly_results['anomaly_map'],
            reconstruction_diff.mean(dim=1)
        )
        
        anomaly_results['final_anomaly_map'] = final_anomaly
        
        return anomaly_results
    
    def _compute_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute prediction uncertainty using dropout"""
        if not self.config.advanced.use_uncertainty:
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        
        # Enable dropout
        self.train()
        
        # Multiple forward passes
        predictions = []
        for _ in range(self.config.advanced.dropout_samples):
            # Fixed placeholder pred = rand to actual model forward for realistic uncertainty
            # Original code used rand, which doesn't compute real uncertainty; fixed to run actual model forward with dropout enabled
            pred = self.forward(x)['segmentation']  # Use segmentation output for uncertainty
            predictions.append(pred)
        
        # Compute variance as uncertainty
        predictions = torch.stack(predictions)
        uncertainty = predictions.var(dim=0).mean(dim=1, keepdim=True)  # Variance over channels
        
        self.eval()
        
        return uncertainty
    
    def _compute_final_similarity(self, similarity_results: Dict,
                                trend_results: Dict,
                                anomaly_results: Dict,
                                region_probs: torch.Tensor,
                                reconstructed: torch.Tensor,
                                original: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute final similarity score using equation"""
        batch_size = original.shape[0]
        
        # Get current coefficients
        current_coeffs = torch.tensor(
            [self.config.equation.coefficients[c] for c in ['A', 'B', 'C', 'D', 'E']],
            dtype=torch.float32, device=original.device
        )
        
        # Adjust coefficients dynamically
        adjusted_coeffs = self.equation_adjuster(current_coeffs)
        
        # Normalize coefficients
        adjusted_coeffs = adjusted_coeffs * 2  # Scale from [0,1] to [0,2]
        
        # Compute similarity components
        components = torch.stack([
            similarity_results['best_similarities'],  # x1: reference similarity
            trend_results['trend_adherence'].mean(dim=[1, 2]),  # x2: trend adherence
            1 - anomaly_results['final_anomaly_map'].mean(dim=[1, 2]),  # x3: inverse anomaly
            region_probs.max(dim=1)[0].mean(dim=[1, 2]),  # x4: segmentation confidence
            F.cosine_similarity(original.view(batch_size, -1),
                              reconstructed.view(batch_size, -1), dim=1)  # x5: reconstruction
        ], dim=1)
        
        # Apply equation: I = Ax1 + Bx2 + Cx3 + Dx4 + Ex5
        final_score = (adjusted_coeffs * components).sum(dim=1)
        
        # Normalize to [0, 1]
        final_score = torch.sigmoid(final_score)
        
        return {
            'score': final_score,
            'components': components,
            'coefficients': adjusted_coeffs
        }
    
    def _get_region_boundaries(self, region_probs: torch.Tensor) -> torch.Tensor:
        """Extract region boundaries from probability maps"""
        regions = region_probs.argmax(dim=1)
        
        # Sobel edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=regions.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=regions.device).view(1, 1, 3, 3)
        
        regions_float = regions.float().unsqueeze(1)
        
        edges_x = F.conv2d(regions_float, kernel_x, padding=1)
        edges_y = F.conv2d(regions_float, kernel_y, padding=1)
        
        boundaries = torch.sqrt(edges_x**2 + edges_y**2)
        boundaries = (boundaries > 0.1).float()
        
        return boundaries.squeeze(1)  # Return [B, H, W] instead of [B, 1, H, W]
    
    def set_equation_coefficients(self, coefficients: Union[List[float], np.ndarray]):
        """Set equation coefficients dynamically"""
        if len(coefficients) != 5:
            raise ValueError("Expected 5 coefficients for A, B, C, D, E")
        
        coeff_names = ['A', 'B', 'C', 'D', 'E']
        for name, value in zip(coeff_names, coefficients):
            self.config.equation.coefficients[name] = float(value)
        
        self.logger.info(f"Updated equation coefficients: {dict(zip(coeff_names, coefficients))}")
    
    def get_equation_coefficients(self) -> np.ndarray:
        """Get current equation coefficients"""
        return np.array([
            self.config.equation.coefficients[c] 
            for c in ['A', 'B', 'C', 'D', 'E']
        ])


# Enable torch.compile for PyTorch 2.0+ optimization
def compile_model(model: nn.Module, backend: str = "inductor") -> nn.Module:
    """Compile model for optimized inference"""
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            compiled_model = torch.compile(model, backend=backend)
            return compiled_model
        except:
            print("torch.compile not available, using regular model")
            return model
    return model


# Test the enhanced network
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing EnhancedIntegratedNetwork")
    
    # Initialize network
    model = EnhancedIntegratedNetwork()
    model.eval()
    
    # Create test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Move to device
    device = model.device
    x = x.to(device)
    model = model.to(device)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(x)
    
    # Print output shapes and values
    print("\nOutput summary:")
    print(f"Segmentation shape: {outputs['segmentation'].shape}")
    print(f"Anomaly map shape: {outputs['anomaly_map'].shape}")
    print(f"Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"Final similarity: {outputs['final_similarity']}")
    print(f"Meets threshold: {outputs['meets_threshold']}")
    print(f"Computation time: {outputs['computation_time']*1000:.1f}ms")
    
    # Test coefficient adjustment
    print("\nTesting coefficient adjustment...")
    new_coeffs = [1.2, 0.8, 1.0, 0.9, 1.1]
    model.set_equation_coefficients(new_coeffs)
    current_coeffs = model.get_equation_coefficients()
    print(f"Updated coefficients: {current_coeffs}")
    
    print(f"\n[{datetime.now()}] EnhancedIntegratedNetwork test completed")
    print(f"[{datetime.now()}] All implementations complete!")


class EnhancedFiberOpticsIntegratedNetwork(EnhancedIntegratedNetwork):
    """
    Alias for EnhancedIntegratedNetwork to maintain compatibility
    """
    pass


class IntegratedAnalysisPipeline:
    """
    Complete analysis pipeline using the enhanced integrated network
    Provides backward compatibility with original pipeline interface
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing IntegratedAnalysisPipeline")
        
        self.config = get_config()
        self.logger = get_logger("IntegratedAnalysisPipeline")
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("IntegratedAnalysisPipeline")
        
        # Initialize enhanced integrated network
        self.network = EnhancedFiberOpticsIntegratedNetwork()
        
        # Move to device
        self.device = self.config.get_device()
        self.network = self.network.to(self.device)
        
        # Set to evaluation mode by default
        self.network.eval()
        
        self.logger.info("IntegratedAnalysisPipeline initialized")
        print(f"[{datetime.now()}] IntegratedAnalysisPipeline initialized successfully")
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a single image using the integrated network
        """
        self.logger.log_process_start(f"Analyzing image: {image_path}")
        
        # Load and tensorize image
        image_tensor = self.tensor_processor.image_to_tensor(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run through integrated network
        with torch.no_grad():
            results = self.network(image_tensor)
        
        # Added storing original for reconstruction error in post_process
        # Original code referenced non-existent 'original' in reconstruction_error; fixed by adding to results
        results['original'] = image_tensor
        
        # Post-process results
        processed_results = self._post_process_results(results)
        
        # Log key metrics
        self.logger.log_similarity_check(
            results['final_similarity'][0].item(),
            self.config.similarity['threshold'],
            f"ref_{results['best_reference_indices'][0].item()}"
        )
        
        # Count defects
        anomaly_threshold = self.config.anomaly['threshold']
        num_anomaly_pixels = (results['anomaly_map'] > anomaly_threshold).sum().item()
        
        if num_anomaly_pixels > 0:
            self.logger.log_anomaly_detection(
                num_anomaly_pixels,
                self._get_anomaly_locations(results['anomaly_map'][0], anomaly_threshold)
            )
        
        self.logger.log_region_classification({
            'core': results['region_probs'][0, 0].mean().item(),
            'cladding': results['region_probs'][0, 1].mean().item(),
            'ferrule': results['region_probs'][0, 2].mean().item()
        })
        
        self.logger.log_process_end(f"Analyzing image: {image_path}")
        
        return processed_results
    
    def _post_process_results(self, results: Dict) -> Dict:
        """Post-process network outputs"""
        processed = {}
        
        # Convert tensors to numpy for export
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.cpu().numpy()
            else:
                processed[key] = value
        
        # Add summary statistics
        processed['summary'] = {
            'final_similarity_score': float(results['final_similarity'][0].item()),
            'meets_threshold': bool(results['meets_threshold'][0].item()),
            'primary_region': ['core', 'cladding', 'ferrule'][results['region_probs'][0].argmax(dim=0)[128, 128].item()],
            'anomaly_score': float(results['anomaly_map'][0].mean().item()),
            'max_anomaly_score': float(results['anomaly_map'][0].max().item()),
            'reconstruction_error': float(torch.abs(results['reconstruction'] - results.get('original', results['reconstruction'])).mean().item())
        }
        
        # Equation coefficients
        processed['equation_info'] = {
            'coefficients': {
                'A': float(results['coefficients'][0].item()) if 'coefficients' in results else 1.0,
                'B': float(results['coefficients'][1].item()) if 'coefficients' in results else 1.0,
                'C': float(results['coefficients'][2].item()) if 'coefficients' in results else 1.0,
                'D': float(results['coefficients'][3].item()) if 'coefficients' in results else 1.0,
                'E': float(results['coefficients'][4].item()) if 'coefficients' in results else 1.0
            },
            'components': {
                'reference_similarity': float(results['similarity_results']['best_similarities'][0].item()) if 'similarity_results' in results else 0.0,
                'trend_adherence': float(results['trend_results']['trend_adherence'].mean().item()) if 'trend_results' in results and 'trend_adherence' in results['trend_results'] else 0.0,
                'anomaly_inverse': 1.0 - float(results['anomaly_map'][0].mean().item()),
                'segmentation_confidence': float(results['region_probs'][0].max().item()),
                'reconstruction_similarity': float(F.cosine_similarity(results['reconstruction'].view(-1), results.get('original', results['reconstruction']).view(-1), dim=0).item())
            }
        }
        
        return processed
    
    def _get_anomaly_locations(self, anomaly_map: torch.Tensor, threshold: float) -> List[Tuple[int, int]]:
        """Get locations of anomalies above threshold"""
        anomaly_mask = anomaly_map > threshold
        locations = torch.where(anomaly_mask)
        
        # Return first 10 locations
        return list(zip(locations[0].tolist()[:10], locations[1].tolist()[:10]))
    
    def export_results(self, results: Dict, output_path: str):
        """
        Export results to file
        "the program will spit out an anomaly or defect map which will just be a 
        matrix of the pixel intensities of the defects"
        """
        self.logger.log_function_entry("export_results", output_path=output_path)
        
        # Create results matrix
        anomaly_matrix = results['anomaly_map'][0]
        
        # Save as minimal format
        np.savetxt(output_path, anomaly_matrix, fmt='%.4f')
        
        # Also save complete results as .npz
        complete_path = output_path.replace('.txt', '_complete.npz')
        np.savez_compressed(complete_path, **results)
        
        self.logger.info(f"Results exported to: {output_path}")
        self.logger.info(f"Complete results saved to: {complete_path}")
        
        self.logger.log_function_exit("export_results")