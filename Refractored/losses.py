# losses.py
# Loss functions for the fiber optics analysis system

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fiber optic defect detection.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function that handles classification, anomaly detection, and similarity.
    """
    
    def __init__(self, config):
        super().__init__()
        # Convert dict to Box if needed
        if isinstance(config, dict):
            from config import Box
            self.config = Box(config)
        else:
            self.config = config
        
        # Initialize loss functions based on configuration
        if self.config.loss.type == 'focal':
            self.classification_loss = FocalLoss(
                alpha=self.config.loss.focal_alpha, 
                gamma=self.config.loss.focal_gamma
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss()
            
        self.anomaly_loss = nn.BCEWithLogitsLoss()
        self.similarity_loss = nn.CosineEmbeddingLoss()
        
        # Loss weights
        self.weights = self.config.loss.weights
    
    def forward(self, outputs, batch, device):
        """
        Compute combined loss from model outputs and batch data.
        
        Args:
            outputs: Dictionary containing model predictions
            batch: Batch data with labels
            device: Device for tensor operations
            
        Returns:
            Dictionary with individual losses and total loss
        """
        labels = batch['label'].to(device)
        
        # Classification loss
        loss_cls = self.classification_loss(outputs['region_logits'], labels)
        
        # Anomaly detection loss
        anomaly_target = (labels == self.config.data.class_map.defects).float()
        anomaly_target = anomaly_target.view(-1, 1, 1, 1)
        anomaly_target = anomaly_target.expand_as(outputs['anomaly_map'])
        loss_anomaly = self.anomaly_loss(outputs['anomaly_map'], anomaly_target)
        
        # Similarity loss (if reference embedding exists)
        loss_sim = torch.tensor(0.0, device=device)
        if outputs.get('ref_embedding') is not None:
            target_sim = torch.ones(labels.size(0), device=device)
            loss_sim = self.similarity_loss(
                outputs['embedding'], 
                outputs['ref_embedding'], 
                target_sim
            )
        
        # Total weighted loss
        total_loss = (
            self.weights.classification * loss_cls +
            self.weights.anomaly * loss_anomaly +
            self.weights.similarity * loss_sim
        )
        
        return {
            'total': total_loss,
            'classification': loss_cls,
            'anomaly': loss_anomaly,
            'similarity': loss_sim
        }

def get_loss_function(config):
    """Factory function to create the appropriate loss function."""
    return CombinedLoss(config)
