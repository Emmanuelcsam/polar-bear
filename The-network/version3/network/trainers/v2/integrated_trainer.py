import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
from tqdm import tqdm

from ..utils.logger import get_logger
from ..config.config import get_config
from ..models.integrated_fiber_nn import IntegratedFiberOpticsNN
from ..data_loaders.tensor_loader import TensorDataLoader


class IntegratedTrainer:
    """
    Trainer for the integrated neural network that learns everything end-to-end.
    "the neural network does the segmentation and reference comparison and 
    anomaly detection internally"
    """
    
    def __init__(self, model: IntegratedFiberOpticsNN):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        self.model = model.to(self.device)
        self.data_loader = TensorDataLoader()
        
        # Optimizer with different learning rates for different components
        self.optimizer = optim.Adam([
            {'params': model.feature_layers.parameters(), 'lr': self.config.LEARNING_RATE},
            {'params': model.comparison_blocks.parameters(), 'lr': self.config.LEARNING_RATE * 0.5},
            {'params': model.reconstruction_decoder.parameters(), 'lr': self.config.LEARNING_RATE * 0.5},
            {'params': [model.gradient_trend_params, model.pixel_trend_params], 'lr': self.config.LEARNING_RATE * 0.1}
        ])
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            epochs=self.config.NUM_EPOCHS,
            steps_per_epoch=1000  # Will be updated based on actual data
        )
        
        # Loss components
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.similarity_loss = nn.MSELoss()
        
        # Training history
        self.history = {
            'total_loss': [],
            'segmentation_loss': [],
            'reconstruction_loss': [],
            'similarity_scores': [],
            'anomaly_detection_accuracy': []
        }
        
        self.logger.info("Initialized IntegratedTrainer")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Train the integrated model end-to-end.
        "over time the program will get more accurate and faster at computing"
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        self.logger.info(f"Starting integrated training for {num_epochs} epochs")
        
        # Load all reference data
        self.logger.info("Loading reference tensors...")
        self.data_loader.load_all_references(preload=True)
        
        # Create datasets for different image types
        datasets = self._create_datasets()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.model.train()
            
            epoch_metrics = {
                'total_loss': [],
                'seg_loss': [],
                'recon_loss': [],
                'similarities': []
            }
            
            # Train on different image types
            for dataset_name, dataloader in datasets.items():
                self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Training on {dataset_name}")
                
                for batch_idx, (images, region_labels) in enumerate(tqdm(dataloader)):
                    images = images.to(self.device)
                    region_labels = region_labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate losses
                    losses = self._calculate_losses(outputs, images, region_labels)
                    total_loss = losses['total']
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # Track metrics
                    epoch_metrics['total_loss'].append(total_loss.item())
                    epoch_metrics['seg_loss'].append(losses['segmentation'].item())
                    epoch_metrics['recon_loss'].append(losses['reconstruction'].item())
                    epoch_metrics['similarities'].append(outputs['best_similarity'].mean().item())
                    
                    # Log progress
                    if batch_idx % 50 == 0:
                        self._log_batch_metrics(epoch, batch_idx, losses, outputs)
                    
                    # Adjust parameters based on loss
                    # "the program will calculate its losses and try to minimize its 
                    # losses by small percentile adjustments to parameters"
                    if batch_idx % 100 == 0:
                        self._adjust_parameters_based_on_loss(total_loss)
            
            # Epoch summary
            self._log_epoch_summary(epoch, epoch_metrics, time.time() - epoch_start)
            
            # Validation
            if epoch % 5 == 0:
                val_metrics = self._validate()
                self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
    
    def _calculate_losses(self, outputs: Dict[str, torch.Tensor], 
                         original_images: torch.Tensor,
                         region_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components.
        "Im subtracting the resulting classification with the original input to find anomalies"
        """
        losses = {}
        
        # 1. Segmentation loss - how well we identify regions
        # "in the neural network for classification it will be split up into segments"
        segmentation = outputs['segmentation']
        losses['segmentation'] = self.segmentation_loss(segmentation, region_labels)
        
        # 2. Reconstruction loss - for anomaly detection
        # "take the absolute value of the difference between the two"
        reconstruction = outputs['reconstruction']
        losses['reconstruction'] = self.reconstruction_loss(reconstruction, original_images)
        
        # 3. Similarity constraint - must achieve high similarity
        # "the program must achieve over .7(S is always fractional to R)"
        similarity = outputs['best_similarity']
        target_similarity = torch.ones_like(similarity) * 0.8  # Target above threshold
        losses['similarity'] = self.similarity_loss(similarity, target_similarity)
        
        # 4. Anomaly consistency loss - anomalies should be consistent across scales
        if 'all_features' in outputs:
            anomaly_consistency = self._calculate_anomaly_consistency(outputs)
            losses['anomaly_consistency'] = anomaly_consistency
        
        # 5. Trend following loss - features should follow learned trends
        # "when a feature of an image follows this line its classified as region"
        trend_loss = self._calculate_trend_loss(outputs, original_images)
        losses['trend'] = trend_loss
        
        # Total weighted loss
        losses['total'] = (
            losses['segmentation'] * 0.3 +
            losses['reconstruction'] * 0.3 +
            losses['similarity'] * 0.2 +
            losses.get('anomaly_consistency', 0) * 0.1 +
            losses['trend'] * 0.1
        )
        
        return losses
    
    def _calculate_anomaly_consistency(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Ensure anomaly detection is consistent across different feature scales"""
        all_features = outputs['all_features']
        anomaly_map = outputs['anomaly_map']
        
        # Downsample anomaly map to match feature sizes
        consistency_losses = []
        for features in all_features:
            h, w = features.shape[-2:]
            downsampled_anomaly = F.interpolate(
                anomaly_map.unsqueeze(1), 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            # Features should be suppressed where anomalies are high
            feature_magnitude = features.abs().mean(dim=1)
            consistency = torch.abs(feature_magnitude * downsampled_anomaly).mean()
            consistency_losses.append(consistency)
        
        return torch.stack(consistency_losses).mean()
    
    def _calculate_trend_loss(self, outputs: Dict[str, torch.Tensor], 
                            original_images: torch.Tensor) -> torch.Tensor:
        """
        Ensure network learns proper trend lines for regions.
        "the program will forcibly look for all lines of best fit based on gradient 
        trends for all datapoints and pixels"
        """
        segmentation = outputs['segmentation']
        
        # Calculate gradients of original images
        sobel_x = self.model._sobel_x().to(original_images.device)
        sobel_y = self.model._sobel_y().to(original_images.device)
        
        grad_x = F.conv2d(original_images.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y = F.conv2d(original_images.mean(dim=1, keepdim=True), sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)
        
        # Each region should have consistent gradient patterns
        trend_losses = []
        for region_idx in range(3):
            region_mask = segmentation[:, region_idx]
            
            # Calculate mean gradient in this region
            region_gradient = (gradient_magnitude * region_mask).sum() / (region_mask.sum() + 1e-8)
            
            # Variance should be low (consistent gradient)
            gradient_variance = torch.var(gradient_magnitude[region_mask > 0.5])
            trend_losses.append(gradient_variance)
        
        return torch.stack(trend_losses).mean()
    
    def _adjust_parameters_based_on_loss(self, loss: torch.Tensor):
        """
        Dynamically adjust network parameters based on loss.
        "small percentile adjustments to parameters"
        """
        adjustment_rate = self.config.LOSS_ADJUSTMENT_RATE
        
        # Only adjust if loss is high
        if loss.item() > 0.5:
            with torch.no_grad():
                # Adjust gradient and position influences
                for layer in self.model.feature_layers:
                    layer.gradient_influence.data *= (1 - adjustment_rate)
                    layer.position_influence.data *= (1 - adjustment_rate)
                
                # Adjust trend parameters slightly
                self.model.gradient_trend_params.data *= (1 - adjustment_rate * 0.1)
                self.model.pixel_trend_params.data *= (1 - adjustment_rate * 0.1)
            
            self.logger.debug(f"Adjusted parameters by {adjustment_rate} due to loss {loss.item():.4f}")
    
    def _create_datasets(self) -> Dict[str, DataLoader]:
        """Create datasets for different types of images"""
        datasets = {}
        
        # Group by region type
        for region_type in ['core', 'cladding', 'ferrule', 'full']:
            if region_type in self.data_loader.reference_tensors:
                tensor_dict = self.data_loader.reference_tensors[region_type]
                if len(tensor_dict) > 0:
                    # Create dataset
                    tensors = []
                    labels = []
                    
                    for key, tensor_or_path in tensor_dict.items():
                        if isinstance(tensor_or_path, Path):
                            tensor = torch.load(tensor_or_path, map_location='cpu')
                        else:
                            tensor = tensor_or_path
                        
                        tensors.append(tensor)
                        
                        # Create label based on region type
                        if region_type == 'core':
                            labels.append(0)
                        elif region_type == 'cladding':
                            labels.append(1)
                        elif region_type == 'ferrule':
                            labels.append(2)
                        else:
                            labels.append(0)  # Default to core for full images
                    
                    # Create dataloader
                    dataset = TensorDataset(tensors, labels)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,  # Already loaded in memory
                        pin_memory=True
                    )
                    
                    datasets[region_type] = dataloader
                    self.logger.info(f"Created dataset for {region_type} with {len(tensors)} samples")
        
        return datasets
    
    def _validate(self) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        metrics = {
            'avg_similarity': [],
            'threshold_met': [],
            'segmentation_accuracy': [],
            'anomaly_precision': []
        }
        
        with torch.no_grad():
            # Get validation batch
            val_batch, val_keys = self.data_loader.get_tensor_batch(batch_size=32)
            val_batch = val_batch.to(self.device)
            
            # Forward pass
            outputs = self.model(val_batch)
            
            # Calculate metrics
            metrics['avg_similarity'] = outputs['best_similarity'].mean().item()
            metrics['threshold_met'] = outputs['meets_threshold'].float().mean().item()
            
            # Segmentation accuracy (simplified - would need ground truth in practice)
            segmentation = outputs['segmentation'].argmax(dim=1)
            # Assume dominant region based on position (center = core, middle = cladding, outer = ferrule)
            h, w = segmentation.shape[-2:]
            center_mask = self._create_radial_mask(h, w, 0.2).to(self.device)
            middle_mask = self._create_radial_mask(h, w, 0.5).to(self.device) - center_mask
            
            core_accuracy = (segmentation[center_mask > 0.5] == 0).float().mean()
            cladding_accuracy = (segmentation[middle_mask > 0.5] == 1).float().mean()
            
            metrics['segmentation_accuracy'] = (core_accuracy + cladding_accuracy) / 2
        
        return metrics
    
    def _create_radial_mask(self, h: int, w: int, radius_ratio: float) -> torch.Tensor:
        """Create a radial mask for validation"""
        y_center, x_center = h // 2, w // 2
        y_coords = torch.arange(h).float() - y_center
        x_coords = torch.arange(w).float() - x_center
        
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        dist = torch.sqrt(X**2 + Y**2)
        
        max_radius = min(h, w) // 2
        mask = dist <= (max_radius * radius_ratio)
        
        return mask.float()
    
    def _log_batch_metrics(self, epoch: int, batch_idx: int, 
                         losses: Dict[str, torch.Tensor],
                         outputs: Dict[str, torch.Tensor]):
        """Log metrics for current batch"""
        avg_similarity = outputs['best_similarity'].mean().item()
        threshold_met = outputs['meets_threshold'].float().mean().item()
        
        self.logger.info(
            f"Epoch {epoch+1}, Batch {batch_idx}: "
            f"Loss={losses['total'].item():.4f}, "
            f"Seg={losses['segmentation'].item():.4f}, "
            f"Recon={losses['reconstruction'].item():.4f}, "
            f"Sim={avg_similarity:.4f}, "
            f"Threshold={threshold_met:.1%}"
        )
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, List[float]], 
                         epoch_time: float):
        """Log epoch summary"""
        avg_total_loss = np.mean(metrics['total_loss'])
        avg_seg_loss = np.mean(metrics['seg_loss'])
        avg_recon_loss = np.mean(metrics['recon_loss'])
        avg_similarity = np.mean(metrics['similarities'])
        
        self.history['total_loss'].append(avg_total_loss)
        self.history['segmentation_loss'].append(avg_seg_loss)
        self.history['reconstruction_loss'].append(avg_recon_loss)
        self.history['similarity_scores'].append(avg_similarity)
        
        self.logger.info(
            f"\nEpoch {epoch+1} Summary:\n"
            f"  Total Loss: {avg_total_loss:.4f}\n"
            f"  Segmentation Loss: {avg_seg_loss:.4f}\n"
            f"  Reconstruction Loss: {avg_recon_loss:.4f}\n"
            f"  Avg Similarity: {avg_similarity:.4f}\n"
            f"  Time: {epoch_time:.2f}s\n"
        )
        
        # Log current equation parameters
        params = self.model.get_equation_parameters()
        self.logger.info(f"Current equation parameters: {params}")
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"integrated_model_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'equation_parameters': self.model.get_equation_parameters()
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")


class TensorDataset(Dataset):
    """Simple dataset for loaded tensors"""
    
    def __init__(self, tensors: List[torch.Tensor], labels: List[int]):
        self.tensors = tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]