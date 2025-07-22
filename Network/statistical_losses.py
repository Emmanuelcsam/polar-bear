#!/usr/bin/env python3
"""
Statistical Loss Functions for Fiber Optics Neural Network
Implements loss functions based on statistical analysis insights
Incorporates IoU, circularity, correlation, and similarity metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

from config_loader import get_config
from logger import get_logger


class IoULoss(nn.Module):
    """
    Intersection over Union Loss for segmentation
    Based on consensus algorithm from separation.py
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU loss
        
        Args:
            pred: Predicted masks [B, C, H, W]
            target: Target masks [B, C, H, W]
            
        Returns:
            IoU loss (1 - IoU)
        """
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Calculate intersection and union
        intersection = torch.sum(pred_flat * target_flat, dim=2)
        union = torch.sum(pred_flat + target_flat - pred_flat * target_flat, dim=2)
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return 1 - torch.mean(iou)


class CircularityLoss(nn.Module):
    """
    Circularity constraint loss for core regions
    Penalizes non-circular core predictions
    """
    
    def __init__(self, target_circularity: float = 1.0, weight: float = 0.1):
        """
        Args:
            target_circularity: Target circularity value (1.0 = perfect circle)
            weight: Weight for the loss
        """
        super().__init__()
        self.target_circularity = target_circularity
        self.weight = weight
        
    def calculate_circularity(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate circularity metric: (4 * π * area) / (perimeter²)
        
        Args:
            mask: Binary mask [B, 1, H, W]
            
        Returns:
            Circularity values [B]
        """
        # Calculate area
        area = torch.sum(mask.view(mask.shape[0], -1), dim=1)
        
        # Calculate perimeter using gradients
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        
        # Apply Sobel filters
        edges_x = F.conv2d(mask, sobel_x, padding=1)
        edges_y = F.conv2d(mask, sobel_y, padding=1)
        
        # Calculate edge magnitude
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Sum to get perimeter
        perimeter = torch.sum(edge_magnitude.view(mask.shape[0], -1), dim=1)
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
        
        return torch.clamp(circularity, 0, 1)
    
    def forward(self, core_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate circularity loss
        
        Args:
            core_mask: Predicted core mask [B, 1, H, W]
            
        Returns:
            Circularity loss
        """
        circularity = self.calculate_circularity(core_mask)
        loss = torch.mean((circularity - self.target_circularity)**2)
        return self.weight * loss


class MasterSimilarityLoss(nn.Module):
    """
    Loss based on the master similarity equation
    Encourages predictions to match reference statistics
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: Minimum similarity threshold (from goal.txt)
        """
        super().__init__()
        self.threshold = threshold
        
    def forward(self, 
                similarity_scores: torch.Tensor,
                target_similar: torch.Tensor) -> torch.Tensor:
        """
        Calculate similarity loss
        
        Args:
            similarity_scores: Predicted similarity scores [B, N]
            target_similar: Binary target (1 if should be similar) [B, N]
            
        Returns:
            Similarity loss
        """
        # For similar pairs, maximize similarity (minimize 1 - similarity)
        similar_loss = target_similar * (1 - similarity_scores)
        
        # For dissimilar pairs, ensure similarity < threshold
        dissimilar_loss = (1 - target_similar) * F.relu(similarity_scores - self.threshold)
        
        # Combine losses
        total_loss = torch.mean(similar_loss + dissimilar_loss)
        
        return total_loss


class CorrelationConsistencyLoss(nn.Module):
    """
    Ensures predicted features maintain expected correlations
    Based on correlation analysis from statistics
    """
    
    def __init__(self):
        super().__init__()
        
        # Key correlations from analysis
        self.expected_correlations = {
            ('center_x', 'center_y'): 0.9088,
            ('core_radius', 'cladding_radius'): 0.7964,
            ('core_radius', 'core_cladding_ratio'): 0.6972,
        }
        
    def calculate_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Pearson correlation coefficient
        
        Args:
            x, y: Feature tensors [B]
            
        Returns:
            Correlation coefficient
        """
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2) + 1e-6)
        
        correlation = numerator / denominator
        return correlation
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate correlation consistency loss
        
        Args:
            features: Dictionary of predicted features
            
        Returns:
            Correlation loss
        """
        total_loss = 0.0
        
        for (feat1, feat2), expected_corr in self.expected_correlations.items():
            if feat1 in features and feat2 in features:
                actual_corr = self.calculate_correlation(features[feat1], features[feat2])
                loss = (actual_corr - expected_corr)**2
                total_loss += loss
        
        return total_loss / len(self.expected_correlations)


class MethodAccuracyWeightedLoss(nn.Module):
    """
    Weights losses based on method accuracy scores
    Better performing methods get higher weight in the loss
    """
    
    def __init__(self, base_loss: nn.Module):
        """
        Args:
            base_loss: Base loss function to weight
        """
        super().__init__()
        self.base_loss = base_loss
        
    def forward(self, 
                predictions: List[torch.Tensor],
                targets: torch.Tensor,
                method_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted loss across methods
        
        Args:
            predictions: List of predictions from each method
            targets: Ground truth targets
            method_scores: Accuracy scores for each method [num_methods]
            
        Returns:
            Weighted loss
        """
        # Normalize method scores to weights
        weights = F.softmax(method_scores, dim=0)
        
        total_loss = 0.0
        for pred, weight in zip(predictions, weights):
            loss = self.base_loss(pred, targets)
            total_loss += weight * loss
        
        return total_loss


class MahalanobisAnomalyLoss(nn.Module):
    """
    Loss based on Mahalanobis distance for anomaly detection
    Encourages normal samples to have low distance, anomalies to have high distance
    """
    
    def __init__(self, margin: float = 2.5):
        """
        Args:
            margin: Margin for normal/anomaly separation
        """
        super().__init__()
        self.margin = margin
        
    def forward(self,
                mahal_distances: torch.Tensor,
                is_anomaly: torch.Tensor) -> torch.Tensor:
        """
        Calculate Mahalanobis-based anomaly loss
        
        Args:
            mahal_distances: Mahalanobis distances [B]
            is_anomaly: Binary labels (1 for anomaly) [B]
            
        Returns:
            Anomaly loss
        """
        # For normal samples, minimize distance
        normal_loss = (1 - is_anomaly) * mahal_distances
        
        # For anomalies, maximize distance (with margin)
        anomaly_loss = is_anomaly * F.relu(self.margin - mahal_distances)
        
        total_loss = torch.mean(normal_loss + anomaly_loss)
        
        return total_loss


class ZoneRegressionLoss(nn.Module):
    """
    Regression loss for zone parameters with physical constraints
    Based on the regression models from statistical analysis
    """
    
    def __init__(self):
        super().__init__()
        
        # Use Smooth L1 loss for robustness
        self.regression_loss = nn.SmoothL1Loss()
        
        # Physical constraint weights
        self.constraint_weight = 0.1
        
    def forward(self,
                pred_params: Dict[str, torch.Tensor],
                target_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate zone parameter regression loss
        
        Args:
            pred_params: Predicted parameters (core_radius, cladding_radius, ratio)
            target_params: Target parameters
            
        Returns:
            Total loss
        """
        # Basic regression loss
        core_loss = self.regression_loss(pred_params['core_radius'], target_params['core_radius'])
        cladding_loss = self.regression_loss(pred_params['cladding_radius'], target_params['cladding_radius'])
        ratio_loss = self.regression_loss(pred_params['core_cladding_ratio'], target_params['core_cladding_ratio'])
        
        regression_loss = core_loss + cladding_loss + ratio_loss
        
        # Physical constraints
        # Core should be smaller than cladding
        constraint_loss = F.relu(pred_params['core_radius'] - pred_params['cladding_radius'] + 10)
        
        # Ratio should be consistent
        predicted_ratio = pred_params['core_radius'] / (pred_params['cladding_radius'] + 1e-6)
        ratio_consistency_loss = torch.mean((predicted_ratio - pred_params['core_cladding_ratio'])**2)
        
        total_loss = regression_loss + self.constraint_weight * (constraint_loss + ratio_consistency_loss)
        
        return total_loss


class StatisticalCompositeLoss(nn.Module):
    """
    Composite loss function combining all statistical insights
    Main loss function for the statistically integrated network
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration dictionary with loss weights
        """
        super().__init__()
        print(f"[{datetime.now()}] Initializing StatisticalCompositeLoss")
        
        self.logger = get_logger("StatisticalCompositeLoss")
        
        # Default weights
        default_weights = {
            'segmentation': 1.0,
            'iou': 0.5,
            'circularity': 0.1,
            'similarity': 0.3,
            'correlation': 0.1,
            'anomaly': 0.5,
            'zone_regression': 0.3,
            'method_accuracy': 0.2
        }
        
        self.weights = config.get('loss_weights', default_weights) if config else default_weights
        
        # Initialize component losses
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.iou_loss = IoULoss()
        self.circularity_loss = CircularityLoss()
        self.similarity_loss = MasterSimilarityLoss()
        self.correlation_loss = CorrelationConsistencyLoss()
        self.anomaly_loss = MahalanobisAnomalyLoss()
        self.zone_regression_loss = ZoneRegressionLoss()
        
        # Method accuracy weighting (wraps segmentation loss)
        self.method_weighted_loss = MethodAccuracyWeightedLoss(self.segmentation_loss)
        
        self.logger.info("StatisticalCompositeLoss initialized with weights: %s", self.weights)
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate composite loss
        
        Args:
            predictions: Dictionary of all predictions from the network
            targets: Dictionary of all target values
            
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # Segmentation loss (if available)
        if 'segmentation_logits' in predictions and 'segmentation_masks' in targets:
            losses['segmentation'] = self.segmentation_loss(
                predictions['segmentation_logits'],
                targets['segmentation_masks']
            ) * self.weights['segmentation']
        
        # IoU loss for masks
        if 'predicted_masks' in predictions and 'target_masks' in targets:
            losses['iou'] = self.iou_loss(
                predictions['predicted_masks'],
                targets['target_masks']
            ) * self.weights['iou']
        
        # Circularity loss for core
        if 'core_mask' in predictions:
            losses['circularity'] = self.circularity_loss(
                predictions['core_mask']
            ) * self.weights['circularity']
        
        # Similarity loss
        if 'similarity_scores' in predictions and 'target_similar' in targets:
            losses['similarity'] = self.similarity_loss(
                predictions['similarity_scores'],
                targets['target_similar']
            ) * self.weights['similarity']
        
        # Correlation consistency loss
        if 'zone_parameters' in predictions:
            losses['correlation'] = self.correlation_loss(
                predictions['zone_parameters']
            ) * self.weights['correlation']
        
        # Anomaly detection loss
        if 'mahalanobis_distance' in predictions and 'is_anomaly' in targets:
            losses['anomaly'] = self.anomaly_loss(
                predictions['mahalanobis_distance'],
                targets['is_anomaly']
            ) * self.weights['anomaly']
        
        # Zone regression loss
        if 'zone_parameters' in predictions and 'target_zone_parameters' in targets:
            losses['zone_regression'] = self.zone_regression_loss(
                predictions['zone_parameters'],
                targets['target_zone_parameters']
            ) * self.weights['zone_regression']
        
        # Method accuracy weighted loss
        if 'method_predictions' in predictions and 'method_scores' in predictions:
            if 'segmentation_masks' in targets:
                losses['method_weighted'] = self.method_weighted_loss(
                    predictions['method_predictions'],
                    targets['segmentation_masks'],
                    predictions['method_scores']
                ) * self.weights['method_accuracy']
        
        # Calculate total loss
        total_loss = sum(losses.values())
        
        # Add total to losses dict
        losses['total'] = total_loss
        
        # Log losses
        self.logger.debug("Loss components: %s", 
                         {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()})
        
        return losses


class ConsensusConsistencyLoss(nn.Module):
    """
    Ensures multiple methods agree on predictions
    Based on consensus algorithm from separation.py
    """
    
    def __init__(self, min_agreement_ratio: float = 0.6):
        """
        Args:
            min_agreement_ratio: Minimum ratio of methods that should agree
        """
        super().__init__()
        self.min_agreement_ratio = min_agreement_ratio
        
    def forward(self, method_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate consensus consistency loss
        
        Args:
            method_masks: List of masks from different methods [B, C, H, W]
            
        Returns:
            Consistency loss
        """
        num_methods = len(method_masks)
        if num_methods < 2:
            return torch.tensor(0.0, device=method_masks[0].device)
        
        # Stack all masks
        stacked_masks = torch.stack(method_masks, dim=1)  # [B, M, C, H, W]
        
        # Calculate pairwise IoU
        total_iou = 0.0
        num_pairs = 0
        
        for i in range(num_methods):
            for j in range(i + 1, num_methods):
                mask_i = stacked_masks[:, i]
                mask_j = stacked_masks[:, j]
                
                intersection = torch.sum(mask_i * mask_j, dim=[1, 2, 3])
                union = torch.sum(mask_i + mask_j - mask_i * mask_j, dim=[1, 2, 3])
                iou = intersection / (union + 1e-6)
                
                total_iou += torch.mean(iou)
                num_pairs += 1
        
        # Average IoU
        avg_iou = total_iou / num_pairs
        
        # Loss is high when agreement is low
        loss = F.relu(self.min_agreement_ratio - avg_iou)
        
        return loss


class DefectSpecificLoss(nn.Module):
    """
    Specialized loss for specific defect types (scratches, digs, blobs)
    Based on detection.py defect classification
    """
    
    def __init__(self):
        super().__init__()
        
        # Different weights for different defect types
        self.defect_weights = {
            'scratch': 1.0,
            'dig': 1.5,      # Digs are more critical
            'blob': 0.8      # Blobs are less critical
        }
        
        # Focal loss for handling class imbalance
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        
    def forward(self,
                defect_predictions: Dict[str, torch.Tensor],
                defect_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate defect-specific loss
        
        Args:
            defect_predictions: Predictions for each defect type
            defect_targets: Target masks for each defect type
            
        Returns:
            Weighted defect loss
        """
        total_loss = 0.0
        
        for defect_type, weight in self.defect_weights.items():
            if f'{defect_type}_map' in defect_predictions and f'{defect_type}_target' in defect_targets:
                pred = defect_predictions[f'{defect_type}_map']
                target = defect_targets[f'{defect_type}_target']
                
                loss = self.focal_loss(pred, target)
                total_loss += weight * loss
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in defect detection
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Focusing parameter
            alpha: Weighting factor
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            pred: Predictions [B, 1, H, W]
            target: Binary targets [B, 1, H, W]
            
        Returns:
            Focal loss
        """
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-bce)
        
        # Focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply focal term and alpha
        loss = self.alpha * focal_term * bce
        
        return torch.mean(loss)


# Create a function to get the appropriate loss based on configuration
def get_statistical_loss(config: Dict) -> nn.Module:
    """
    Get the appropriate loss function based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss module
    """
    loss_type = config.get('loss_type', 'composite')
    
    if loss_type == 'composite':
        return StatisticalCompositeLoss(config)
    elif loss_type == 'iou':
        return IoULoss()
    elif loss_type == 'circularity':
        return CircularityLoss()
    elif loss_type == 'similarity':
        return MasterSimilarityLoss()
    elif loss_type == 'anomaly':
        return MahalanobisAnomalyLoss()
    elif loss_type == 'consensus':
        return ConsensusConsistencyLoss()
    elif loss_type == 'defect':
        return DefectSpecificLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")