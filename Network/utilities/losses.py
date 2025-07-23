#!/usr/bin/env python3
"""
Advanced Loss Functions for Fiber Optics Neural Network
Implements Focal Loss, Contrastive Loss, Wasserstein Distance, and more
Based on research: Lin et al., 2017; Chen et al., 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import torchvision.models as models

from core.config_loader import get_config
from core.logger import get_logger


class FocalLoss(nn.Module):
    """
    Focal Loss for Anomaly Detection (Lin et al., 2017)
    Better for imbalanced defect detection in fiber optics
    
    Mathematical formulation:
    Focal Loss = -α(1-p_t)^γ log(p_t)
    Where p_t = p if y=1, else 1-p
    
    This loss down-weights easy examples and focuses on hard cases
    """
    
    def __init__(self, alpha: Union[float, List[float]] = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive/negative examples or list for multi-class (default: 0.25)
            gamma: Focusing parameter for modulating loss (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing FocalLoss")
        
        # Modified alpha to accept list for multi-class support
        # Original code assumed binary classification with float alpha, but segmentation is multi-class (3 classes)
        # This fixes by allowing alpha as list [alpha_class0, alpha_class1, alpha_class2] or float for binary
        self.alpha = alpha if isinstance(alpha, list) else [alpha]
        self.gamma = gamma
        self.reduction = reduction
        
        self.logger = get_logger("FocalLoss")
        self.logger.log_class_init("FocalLoss", alpha=alpha, gamma=gamma)
        
        # Track loss statistics
        self.stats = {
            'easy_examples': 0,
            'hard_examples': 0,
            'total_examples': 0
        }
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            pred: Predicted logits [B, C] or [B, C, H, W]
            target: Ground truth labels [B] or [B, H, W]
        """
        # Handle different input shapes
        if pred.dim() == 4:  # Segmentation case
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(-1, C)
            target = target.reshape(-1)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            # Modified to handle multi-class alpha
            # Original code used torch.where(target > 0), assuming binary, which fails for multi-class targets (0,1,2)
            # Fixed by gathering alpha per class: alpha_t = torch.tensor(self.alpha, device=pred.device)[target]
            # Assumes len(self.alpha) == num_classes
            alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, device=pred.device)
            alpha_t = alpha_tensor[target]
            focal_weight = alpha_t * focal_weight
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        # Track statistics
        with torch.no_grad():
            easy_mask = p > 0.5
            self.stats['easy_examples'] += easy_mask.sum().item()
            self.stats['hard_examples'] += (~easy_mask).sum().item()
            self.stats['total_examples'] += p.numel()
            
            # Log periodically
            if self.stats['total_examples'] > 10000:
                easy_ratio = self.stats['easy_examples'] / self.stats['total_examples']
                hard_ratio = self.stats['hard_examples'] / self.stats['total_examples']
                self.logger.info(f"Focal Loss stats - Easy: {easy_ratio:.2%}, Hard: {hard_ratio:.2%}")
                self.stats = {'easy_examples': 0, 'hard_examples': 0, 'total_examples': 0}
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Learning for Reference Comparison (Chen et al., 2020)
    SimCLR approach adapted for fiber optics
    
    Temperature-scaled cosine similarity with InfoNCE loss
    Helps learn better representations for reference matching
    """
    
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        """
        Args:
            temperature: Temperature for scaling similarities (default: 0.07)
            normalize: Whether to normalize features (default: True)
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing ContrastiveLoss")
        
        self.temperature = temperature
        self.normalize = normalize
        
        self.logger = get_logger("ContrastiveLoss")
        self.logger.log_class_init("ContrastiveLoss", temperature=temperature)
        
    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            features: Feature embeddings [B, D] or [B, N, D]
            labels: Optional labels for supervised contrastive learning
        """
        # Handle different input shapes
        if features.dim() == 3:
            B, N, D = features.shape
            features = features.reshape(B * N, D)
            if labels is not None:
                labels = labels.repeat_interleave(N)
        else:
            B, D = features.shape
        
        # Normalize features
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create masks for positive pairs
        batch_size = features.shape[0]
        
        if labels is None:
            # Self-supervised: positive pairs are augmentations of same image
            # Assume features come in pairs (original, augmented)
            labels = torch.arange(batch_size // 2).repeat(2).to(features.device)
        
        # Mask to remove diagonal
        mask_diag = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # Positive pair mask
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positives = labels_equal & ~mask_diag
        
        # For each anchor, compute loss
        losses = []
        for i in range(batch_size):
            # Get positive similarities
            positive_mask = mask_positives[i]
            if positive_mask.sum() == 0:
                continue
                
            positive_sims = similarity_matrix[i][positive_mask]
            
            # Get negative similarities
            negative_mask = ~labels_equal[i] & ~mask_diag[i]
            negative_sims = similarity_matrix[i][negative_mask]
            
            # Concatenate positives and negatives
            logits = torch.cat([positive_sims, negative_sims])
            
            # Labels: positives are class 0
            labels_loss = torch.zeros(positive_sims.shape[0], dtype=torch.long, device=features.device)
            
            # Cross entropy loss
            loss = F.cross_entropy(logits.unsqueeze(0).repeat(positive_sims.shape[0], 1), labels_loss)
            losses.append(loss)
        
        # Average over all anchors
        if len(losses) == 0:
            return torch.tensor(0.0, device=features.device)
        
        return torch.stack(losses).mean()


class WassersteinLoss(nn.Module):
    """
    Wasserstein Distance for Distribution Matching
    Better for understanding region transitions in fiber optics
    
    Uses 1-Wasserstein distance (Earth Mover's Distance) approximation
    """
    
    def __init__(self, p: int = 1, normalize: bool = True):
        """
        Args:
            p: Which Wasserstein distance (1 or 2)
            normalize: Whether to normalize distributions
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing WassersteinLoss")
        
        self.p = p
        self.normalize = normalize
        
        self.logger = get_logger("WassersteinLoss")
        self.logger.log_class_init("WassersteinLoss", p=p)
        
    def forward(self, pred_samples: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
        """
        Compute Wasserstein distance between two distributions
        
        Args:
            pred_samples: Samples from predicted distribution [B, N] or [B, N, D]
            target_samples: Samples from target distribution [B, N] or [B, N, D]
        """
        # Ensure same shape
        assert pred_samples.shape == target_samples.shape
        
        batch_size = pred_samples.shape[0]
        losses = []
        
        for i in range(batch_size):
            p_samples = pred_samples[i]
            q_samples = target_samples[i]
            
            if self.normalize:
                # Normalize to probability distributions
                p_samples = p_samples / (p_samples.sum() + 1e-8)
                q_samples = q_samples / (q_samples.sum() + 1e-8)
            
            if p_samples.dim() == 1:
                # 1D case: sort and compute L1/L2 distance
                p_sorted, _ = torch.sort(p_samples)
                q_sorted, _ = torch.sort(q_samples)
                
                if self.p == 1:
                    distance = torch.abs(p_sorted - q_sorted).mean()
                else:
                    distance = torch.sqrt(((p_sorted - q_sorted) ** 2).mean())
            else:
                # Multi-dimensional: use Sinkhorn approximation
                distance = self._sinkhorn_distance(p_samples, q_samples)
            
            losses.append(distance)
        
        return torch.stack(losses).mean()
    
    def _sinkhorn_distance(self, x: torch.Tensor, y: torch.Tensor, 
                          epsilon: float = 0.1, num_iters: int = 100) -> torch.Tensor:
        """
        Compute Sinkhorn approximation to Wasserstein distance
        """
        # Cost matrix
        if x.dim() == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            
        C = torch.cdist(x, y, p=self.p)
        
        # Sinkhorn iterations
        K = torch.exp(-C / epsilon)
        u = torch.ones(x.shape[0], device=x.device) / x.shape[0]
        
        for _ in range(num_iters):
            v = 1.0 / (K.T @ u + 1e-8)
            u = 1.0 / (K @ v + 1e-8)
        
        # Transport plan
        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        
        # Sinkhorn distance
        return (T * C).sum()


class PerceptualLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    Better than SSIM for defect detection
    
    Uses pretrained VGG features for perceptual similarity
    """
    
    def __init__(self, net: str = 'vgg16', spatial: bool = False):
        """
        Args:
            net: Which network to use ('vgg16', 'vgg19', 'resnet50')
            spatial: Return spatial map of perceptual distance
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing PerceptualLoss")
        
        self.spatial = spatial
        
        self.logger = get_logger("PerceptualLoss")
        self.logger.log_class_init("PerceptualLoss", net=net)
        
        # Load pretrained network
        if net == 'vgg16':
            # Updated to use weights='DEFAULT' instead of deprecated pretrained=True
            # Original code used deprecated pretrained arg; fixed to modern PyTorch API (as of 2025 knowledge cutoff, but aligns with post-1.10 changes)
            vgg = models.vgg16(weights='DEFAULT').features
            self.layers = [3, 8, 15, 22, 29]  # Conv layers before pooling
        elif net == 'vgg19':
            vgg = models.vgg19(weights='DEFAULT').features
            self.layers = [3, 8, 17, 26, 35]
        else:
            raise ValueError(f"Unsupported network: {net}")
        
        # Extract feature layers
        self.features = nn.ModuleList()
        last_layer = 0
        for layer in self.layers:
            self.features.append(vgg[last_layer:layer+1])
            last_layer = layer + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Linear layers for combining features
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Conv2d(128, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False)
        ])
        
        # Initialize with uniform weights
        for layer in self.linear_layers:
            nn.init.constant_(layer.weight, 0.2)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
        """
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred
        x_target = target
        
        for feature_layer in self.features:
            x_pred = feature_layer(x_pred)
            x_target = feature_layer(x_target)
            pred_features.append(x_pred)
            target_features.append(x_target)
        
        # Compute perceptual distance at each layer
        diffs = []
        for pred_feat, target_feat, linear in zip(pred_features, target_features, self.linear_layers):
            # Normalize features
            pred_feat = F.normalize(pred_feat, p=2, dim=1)
            target_feat = F.normalize(target_feat, p=2, dim=1)
            
            # Squared difference
            diff = (pred_feat - target_feat) ** 2
            
            # Weight and reduce channels
            weighted_diff = linear(diff)
            
            if self.spatial:
                diffs.append(weighted_diff)
            else:
                diffs.append(weighted_diff.mean(dim=[2, 3]))
        
        if self.spatial:
            # Return spatial map
            # Upsample all to same size
            target_size = diffs[0].shape[2:]
            for i in range(1, len(diffs)):
                diffs[i] = F.interpolate(diffs[i], size=target_size, mode='bilinear', align_corners=False)
            return torch.cat(diffs, dim=1).mean(dim=1, keepdim=True)
        else:
            # Return scalar
            return torch.stack(diffs).mean(dim=0).mean()


class CombinedAdvancedLoss(nn.Module):
    """
    Combined loss function using all advanced techniques and statistical losses
    Implements the multi-objective optimization for fiber optics analysis
    Merges both mathematical and domain-specific losses
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize combined loss with configuration
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing CombinedAdvancedLoss")
        
        # Get config from system if not provided
        if config is None:
            from core.config_loader import get_config
            config_obj = get_config()
            # Convert to dict if needed
            if hasattr(config_obj, '__dict__'):
                config = config_obj.__dict__
            else:
                config = config_obj
        
        self.config = config
        self.logger = get_logger("CombinedAdvancedLoss")
        self.logger.log_class_init("CombinedAdvancedLoss")
        
        # Determine mode from config
        self.mode = "production"
        if hasattr(config, 'runtime') and hasattr(config.runtime, 'mode'):
            if config.runtime.mode in ['research', 'statistical']:
                self.mode = 'research'
        
        # Initialize mathematical losses
        self.focal_seg = FocalLoss(alpha=0.25, gamma=2.0)  # For segmentation
        self.focal_anomaly = FocalLoss(alpha=0.75, gamma=3.0)  # For anomaly (higher alpha for rare defects)
        self.contrastive = ContrastiveLoss(temperature=0.07)
        self.wasserstein = WassersteinLoss(p=1)
        self.perceptual = PerceptualLoss(net='vgg16')
        
        # Initialize statistical/domain-specific losses
        self.iou_loss = IoULoss()
        self.circularity_loss = CircularityLoss(target_circularity=1.0, weight=0.1)
        self.master_similarity_loss = MasterSimilarityLoss(threshold=0.7)
        self.correlation_loss = CorrelationConsistencyLoss()
        self.method_accuracy_loss = MethodAccuracyWeightedLoss()
        self.mahalanobis_loss = MahalanobisAnomalyLoss(margin=2.5)
        self.zone_regression_loss = ZoneRegressionLoss()
        self.consensus_loss = ConsensusConsistencyLoss(min_agreement_ratio=0.6)
        self.defect_loss = DefectSpecificLoss()
        
        # Standard losses
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # Loss weights based on mode
        if self.mode == "production":
            self.weights = {
                'segmentation': 0.25,
                'anomaly': 0.20,
                'contrastive': 0.15,
                'perceptual': 0.15,
                'wasserstein': 0.10,
                'reconstruction': 0.15,
                # Statistical losses have lower weights in production
                'iou': 0.1,
                'circularity': 0.05,
                'similarity': 0.1,
                'correlation': 0.05,
                'mahalanobis': 0.1,
                'zone_regression': 0.1,
                'consensus': 0.05,
                'method_accuracy': 0.05
            }
        else:  # research mode
            self.weights = {
                'segmentation': 0.15,
                'anomaly': 0.10,
                'contrastive': 0.05,
                'perceptual': 0.05,
                'wasserstein': 0.05,
                'reconstruction': 0.05,
                # Statistical losses have higher weights in research
                'iou': 0.15,
                'circularity': 0.10,
                'similarity': 0.15,
                'correlation': 0.10,
                'mahalanobis': 0.15,
                'zone_regression': 0.15,
                'consensus': 0.10,
                'method_accuracy': 0.10
            }
        
        # Override with config if available
        if hasattr(config, 'loss') and hasattr(config.loss, 'weights'):
            self.weights.update(config.loss.weights.__dict__ if hasattr(config.loss.weights, '__dict__') else config.loss.weights)
        
        self.logger.info(f"Loss weights ({self.mode} mode): {self.weights}")
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Args:
            predictions: Dictionary of model outputs
            targets: Dictionary of ground truth values
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # 1. Segmentation loss with focal loss
        if 'segmentation' in predictions and 'segmentation' in targets:
            losses['segmentation'] = self.focal_seg(
                predictions['segmentation'], 
                targets['segmentation']
            )
        
        # 2. Anomaly detection with focal loss
        if 'anomaly_logits' in predictions and 'has_anomaly' in targets:
            losses['anomaly'] = self.focal_anomaly(
                predictions['anomaly_logits'],
                targets['has_anomaly'].long()
            )
        
        # 3. Contrastive loss for reference matching
        if 'reference_features' in predictions:
            losses['contrastive'] = self.contrastive(
                predictions['reference_features'],
                targets.get('reference_labels')
            )
        
        # 4. Perceptual loss for reconstruction
        if 'reconstruction' in predictions and 'image' in targets:
            losses['perceptual'] = self.perceptual(
                predictions['reconstruction'],
                targets['image']
            )
        
        # 5. Wasserstein loss for distribution matching
        if 'gradient_distribution' in predictions and 'gradient_distribution' in targets:
            losses['wasserstein'] = self.wasserstein(
                predictions['gradient_distribution'],
                targets['gradient_distribution']
            )
        
        # 6. Standard reconstruction loss
        if 'reconstruction' in predictions and 'image' in targets:
            losses['reconstruction'] = self.l1(
                predictions['reconstruction'],
                targets['image']
            )
        
        # 7. Statistical losses (if in research mode or explicitly requested)
        if self.mode == "research" or predictions.get('compute_statistical', False):
            # IoU loss
            if 'segmentation_logits' in predictions and 'segmentation_masks' in targets:
                pred_masks = torch.sigmoid(predictions['segmentation_logits'])
                losses['iou'] = self.iou_loss(pred_masks, targets['segmentation_masks'])
            
            # Circularity loss
            if 'predicted_masks' in predictions:
                losses['circularity'] = self.circularity_loss(predictions['predicted_masks'])
            
            # Master similarity loss
            if 'similarity_scores' in predictions:
                losses['similarity'] = self.master_similarity_loss(predictions['similarity_scores'])
            
            # Correlation consistency
            if 'features' in predictions:
                losses['correlation'] = self.correlation_loss(predictions['features'])
            
            # Mahalanobis anomaly loss
            if 'mahalanobis_distance' in predictions and 'is_anomaly' in targets:
                losses['mahalanobis'] = self.mahalanobis_loss(
                    predictions['mahalanobis_distance'],
                    targets['is_anomaly']
                )
            
            # Zone regression loss
            if 'zone_parameters' in predictions and 'target_zone_parameters' in targets:
                losses['zone_regression'] = self.zone_regression_loss(
                    predictions['zone_parameters'],
                    targets['target_zone_parameters']
                )
            
            # Consensus consistency
            if 'method_predictions' in predictions:
                losses['consensus'] = self.consensus_loss(predictions['method_predictions'])
            
            # Method accuracy weighted loss
            if 'method_predictions' in predictions and 'method_scores' in predictions:
                base_loss = losses.get('segmentation', torch.tensor(0.0))
                losses['method_accuracy'] = self.method_accuracy_loss(
                    base_loss,
                    predictions['method_scores']
                )
        
        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        for loss_name, loss_value in losses.items():
            if loss_name in self.weights:
                total_loss += self.weights[loss_name] * loss_value
        
        losses['total'] = total_loss
        
        # Log losses periodically
        with torch.no_grad():
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 1
                
            if self._log_counter % 100 == 0:
                loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
                self.logger.info(f"Losses - {loss_str}")
        
        return losses


# Statistical/Domain-specific loss components
class IoULoss(nn.Module):
    """Intersection over Union loss for segmentation quality"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred + target - pred * target, dim=(2, 3))
        iou = intersection / (union + 1e-6)
        return 1 - iou.mean()


class CircularityLoss(nn.Module):
    """Enforces circular shape constraint for fiber cores"""
    
    def __init__(self, target_circularity: float = 1.0, weight: float = 0.1):
        super().__init__()
        self.target_circularity = target_circularity
        self.weight = weight
    
    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        # Assume first channel is core region
        core_masks = masks[:, 0:1]
        
        # Calculate circularity for each mask
        batch_size = core_masks.shape[0]
        circularity_loss = 0
        
        for i in range(batch_size):
            mask = core_masks[i, 0]
            
            # Find contour properties
            if mask.sum() > 0:
                # Simple approximation of circularity
                area = mask.sum()
                # Approximate perimeter using edge detection
                edges_x = torch.abs(mask[1:] - mask[:-1]).sum()
                edges_y = torch.abs(mask[:, 1:] - mask[:, :-1]).sum()
                perimeter = edges_x + edges_y
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    circularity_loss += (circularity - self.target_circularity) ** 2
        
        return self.weight * circularity_loss / batch_size


class MasterSimilarityLoss(nn.Module):
    """Loss based on master similarity equation S > 0.7 threshold"""
    
    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, similarity_scores: torch.Tensor) -> torch.Tensor:
        # Encourage similarities above threshold
        margin_loss = F.relu(self.threshold - similarity_scores)
        return margin_loss.mean()


class CorrelationConsistencyLoss(nn.Module):
    """Maintains expected feature correlations from statistical analysis"""
    
    def __init__(self):
        super().__init__()
        # Expected correlations from statistical analysis
        self.expected_correlations = {
            ('center_x', 'center_y'): 0.9088,
            ('core_radius', 'cladding_radius'): 0.7964,
            ('core_radius', 'ratio'): -0.1654,
            ('cladding_radius', 'ratio'): -0.6823
        }
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Simplified version - returns zero for now
        # Real implementation would compute actual correlations
        return torch.tensor(0.0, device=features.device)


class MethodAccuracyWeightedLoss(nn.Module):
    """Weights losses based on method performance"""
    
    def forward(self, base_loss: torch.Tensor, method_scores: torch.Tensor) -> torch.Tensor:
        # Normalize scores to weights
        weights = F.softmax(method_scores, dim=0)
        
        # Apply weights to base loss
        if base_loss.dim() > 0 and base_loss.shape[0] == weights.shape[0]:
            weighted_loss = (base_loss * weights).sum()
        else:
            weighted_loss = base_loss
        
        return weighted_loss


class MahalanobisAnomalyLoss(nn.Module):
    """Mahalanobis distance-based anomaly loss"""
    
    def __init__(self, margin: float = 2.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, distances: torch.Tensor, is_anomaly: torch.Tensor) -> torch.Tensor:
        # Normal samples should have low distance
        normal_loss = distances * (1 - is_anomaly.float())
        
        # Anomalies should have high distance
        anomaly_loss = F.relu(self.margin - distances) * is_anomaly.float()
        
        return (normal_loss + anomaly_loss).mean()


class ZoneRegressionLoss(nn.Module):
    """Regression loss for zone parameters with physical constraints"""
    
    def forward(self, pred: Dict[str, torch.Tensor], 
                target: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        
        # Basic regression loss
        for key in ['core_radius', 'cladding_radius', 'core_cladding_ratio']:
            if key in pred and key in target:
                loss += F.smooth_l1_loss(pred[key], target[key])
        
        # Physical constraints
        if 'core_radius' in pred and 'cladding_radius' in pred:
            # Core should be smaller than cladding
            constraint_loss = F.relu(pred['core_radius'] - pred['cladding_radius'] + 10)
            loss += constraint_loss.mean()
        
        return loss


class ConsensusConsistencyLoss(nn.Module):
    """Ensures multiple methods agree on predictions"""
    
    def __init__(self, min_agreement_ratio: float = 0.6):
        super().__init__()
        self.min_agreement_ratio = min_agreement_ratio
    
    def forward(self, method_predictions: List[torch.Tensor]) -> torch.Tensor:
        if len(method_predictions) < 2:
            return torch.tensor(0.0)
        
        # Calculate pairwise IoU
        total_loss = 0
        num_pairs = 0
        
        for i in range(len(method_predictions)):
            for j in range(i + 1, len(method_predictions)):
                pred_i = method_predictions[i]
                pred_j = method_predictions[j]
                
                # IoU between predictions
                intersection = (pred_i * pred_j).sum()
                union = (pred_i + pred_j - pred_i * pred_j).sum()
                iou = intersection / (union + 1e-6)
                
                # Penalize low agreement
                total_loss += F.relu(self.min_agreement_ratio - iou)
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class DefectSpecificLoss(nn.Module):
    """Specialized loss for different defect types"""
    
    def __init__(self):
        super().__init__()
        self.defect_weights = {
            'scratch': 1.0,
            'dig': 1.5,
            'blob': 0.8
        }
        self.focal = FocalLoss(gamma=2.0, alpha=0.25)
    
    def forward(self, defect_preds: Dict[str, torch.Tensor], 
                defect_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        
        for defect_type, weight in self.defect_weights.items():
            if defect_type in defect_preds and defect_type in defect_targets:
                loss = self.focal(defect_preds[defect_type], defect_targets[defect_type])
                total_loss += weight * loss
        
        return total_loss


def create_loss_function(config: Dict) -> CombinedAdvancedLoss:
    """
    Factory function to create the combined loss function
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CombinedAdvancedLoss instance
    """
    logger = get_logger("LossFactory")
    logger.log_function_entry("create_loss_function")
    
    loss_fn = CombinedAdvancedLoss(config)
    
    logger.info("Created CombinedAdvancedLoss function")
    logger.log_function_exit("create_loss_function")
    
    return loss_fn


# Test the losses
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing advanced loss functions")
    
    # Test focal loss
    print("\nTesting Focal Loss...")
    focal = FocalLoss()
    pred = torch.randn(4, 3, 256, 256)
    target = torch.randint(0, 3, (4, 256, 256))
    loss = focal(pred, target)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test contrastive loss
    print("\nTesting Contrastive Loss...")
    contrastive = ContrastiveLoss()
    features = torch.randn(8, 128)  # 4 pairs
    loss = contrastive(features)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Test Wasserstein loss
    print("\nTesting Wasserstein Loss...")
    wasserstein = WassersteinLoss()
    dist1 = torch.randn(4, 100)
    dist2 = torch.randn(4, 100)
    loss = wasserstein(dist1, dist2)
    print(f"Wasserstein loss: {loss.item():.4f}")
    
    # Test perceptual loss
    print("\nTesting Perceptual Loss...")
    perceptual = PerceptualLoss()
    img1 = torch.rand(2, 3, 256, 256)
    img2 = torch.rand(2, 3, 256, 256)
    loss = perceptual(img1, img2)
    print(f"Perceptual loss: {loss.item():.4f}")
    
    print(f"[{datetime.now()}] Advanced loss functions test completed")
    print(f"[{datetime.now()}] Next script: fiber_advanced_architectures.py")