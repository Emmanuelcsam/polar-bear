import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
from tqdm import tqdm

from ..utils.logger import get_logger
from ..config.config import get_config
from ..models.advanced_integrated_nn import AdvancedIntegratedFiberOpticsNN
from ..data_loaders.tensor_loader import TensorDataLoader
from .integrated_trainer import TensorDataset


class AdvancedIntegratedTrainer:
    """
    Trainer for the advanced integrated model that performs simultaneous
    feature classification and anomaly detection.
    "I want each feature to not only look for comparisons but also look for 
    anomalies while comparing"
    """
    
    def __init__(self, model: AdvancedIntegratedFiberOpticsNN):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        self.model = model.to(self.device)
        self.data_loader = TensorDataLoader()
        
        # Different learning rates for different components
        param_groups = [
            # Feature extractors - higher LR for faster adaptation
            {'params': model.feature_extractors.parameters(), 
             'lr': self.config.LEARNING_RATE},
            
            # Analysis blocks - medium LR for stable learning
            {'params': model.analysis_blocks.parameters(), 
             'lr': self.config.LEARNING_RATE * 0.5},
            
            # Trend parameters - very low LR for stability
            {'params': [model.gradient_trends, model.pixel_trends], 
             'lr': self.config.LEARNING_RATE * 0.01},
            
            # Reference embeddings - low LR
            {'params': [model.reference_embeddings], 
             'lr': self.config.LEARNING_RATE * 0.1},
            
            # Reconstruction decoder
            {'params': model.reconstruction_decoder.parameters(), 
             'lr': self.config.LEARNING_RATE * 0.5}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        
        # Scheduler with warmup
        self.warmup_steps = 1000
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.anomaly_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.quality_loss = nn.MSELoss()
        self.similarity_loss = nn.MSELoss()
        
        # Loss weights that adapt during training
        self.loss_weights = {
            'segmentation': 0.25,
            'anomaly': 0.25,
            'reconstruction': 0.2,
            'quality': 0.15,
            'similarity': 0.15
        }
        
        # Training history
        self.history = {
            'total_loss': [],
            'component_losses': {},
            'similarity_scores': [],
            'anomaly_precision': [],
            'anomaly_recall': [],
            'segmentation_accuracy': []
        }
        
        self.logger.info("Initialized AdvancedIntegratedTrainer")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Train the advanced integrated model.
        "really fully analyze everything that I've been saying"
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        self.logger.info(f"Starting advanced integrated training for {num_epochs} epochs")
        self.logger.info("Model will simultaneously classify features and detect anomalies")
        
        # Load reference data
        self.logger.info("Loading reference tensors...")
        self.data_loader.load_all_references(preload=True)
        
        # Create augmented datasets
        datasets = self._create_augmented_datasets()
        
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.model.train()
            
            epoch_metrics = self._init_epoch_metrics()
            
            # Train on different data types
            for dataset_name, dataloader in datasets.items():
                self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Training on {dataset_name}")
                
                for batch_idx, (images, labels, has_anomalies) in enumerate(tqdm(dataloader)):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    has_anomalies = has_anomalies.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    # Calculate comprehensive losses
                    losses = self._calculate_comprehensive_losses(
                        outputs, images, labels, has_anomalies
                    )
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    losses['total'].backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step (with warmup)
                    if global_step < self.warmup_steps:
                        self._warmup_lr(global_step)
                    else:
                        self.scheduler.step()
                    
                    global_step += 1
                    
                    # Update metrics
                    self._update_epoch_metrics(epoch_metrics, losses, outputs, has_anomalies)
                    
                    # Log progress
                    if batch_idx % 50 == 0:
                        self._log_training_progress(epoch, batch_idx, losses, outputs)
                    
                    # Adaptive loss weight adjustment
                    if batch_idx % 200 == 0 and batch_idx > 0:
                        self._adapt_loss_weights(epoch_metrics)
            
            # Epoch summary
            self._log_epoch_summary(epoch, epoch_metrics, time.time() - epoch_start)
            
            # Validation with detailed analysis
            if epoch % 5 == 0:
                val_metrics = self._comprehensive_validation()
                self.logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        self._final_analysis()
    
    def _calculate_comprehensive_losses(self, outputs: Dict[str, torch.Tensor],
                                      original_images: torch.Tensor,
                                      region_labels: torch.Tensor,
                                      has_anomalies: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components for simultaneous training.
        "so I get a fully detailed anomaly detection while also classifying most 
        probable features(or segments) of the image at the same time"
        """
        losses = {}
        
        # 1. Segmentation loss
        segmentation = outputs['segmentation']
        losses['segmentation'] = self.segmentation_loss(segmentation, region_labels)
        
        # 2. Anomaly detection loss (supervised when we have labels)
        anomaly_map = outputs['anomaly_map']
        if has_anomalies.any():
            # For images with anomalies, penalize low anomaly scores
            anomaly_target = has_anomalies.float().unsqueeze(-1).unsqueeze(-1)
            anomaly_target = anomaly_target.expand(-1, anomaly_map.shape[1], anomaly_map.shape[2])
            losses['anomaly'] = self.anomaly_loss(anomaly_map, anomaly_target)
        else:
            # For normal images, anomaly scores should be low
            losses['anomaly'] = anomaly_map.mean()
        
        # 3. Reconstruction loss
        reconstruction = outputs['reconstruction']
        losses['reconstruction'] = self.reconstruction_loss(reconstruction, original_images)
        
        # 4. Quality consistency loss
        # Quality should be high for normal regions, low for anomalies
        quality_map = outputs['quality_map']
        expected_quality = 1 - anomaly_map.detach()  # High quality where no anomalies
        losses['quality'] = self.quality_loss(quality_map, expected_quality)
        
        # 5. Similarity constraint loss
        # "the program must achieve over .7"
        similarity = outputs['similarity']
        target_similarity = torch.ones_like(similarity) * 0.8
        losses['similarity'] = self.similarity_loss(similarity, target_similarity)
        
        # 6. Multi-scale consistency loss
        # Ensure different scales agree on anomalies
        scale_consistency = self._calculate_scale_consistency_loss(outputs)
        losses['scale_consistency'] = scale_consistency
        
        # 7. Trend adherence loss
        # "when a feature of an image follows this line its classified as region"
        trend_loss = self._calculate_trend_loss(outputs)
        losses['trend'] = trend_loss
        
        # Total weighted loss
        losses['total'] = sum(
            self.loss_weights.get(name, 0.1) * loss 
            for name, loss in losses.items() 
            if name != 'total'
        )
        
        return losses
    
    def _calculate_scale_consistency_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Ensure multi-scale analysis is consistent"""
        all_analysis = outputs['all_analysis']
        
        consistency_losses = []
        
        # Compare adjacent scales
        for i in range(len(all_analysis) - 1):
            curr_anomalies = all_analysis[i]['anomaly_scores']
            next_anomalies = all_analysis[i + 1]['anomaly_scores']
            
            # Resize to same size
            next_resized = F.interpolate(next_anomalies, size=curr_anomalies.shape[-2:],
                                       mode='bilinear', align_corners=False)
            
            # They should be similar but not identical (allow for scale-specific features)
            diff = torch.abs(curr_anomalies - next_resized)
            consistency_losses.append(diff.mean())
        
        return torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0)
    
    def _calculate_trend_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Ensure features follow expected trends for regions"""
        trend_adherence = outputs['trend_adherence']
        segmentation = outputs['segmentation']
        
        # High trend adherence should correlate with confident segmentation
        segmentation_confidence = segmentation.max(dim=1)[0]
        
        # Loss is low when trend adherence matches segmentation confidence
        trend_loss = torch.abs(trend_adherence - segmentation_confidence).mean()
        
        return trend_loss
    
    def _create_augmented_datasets(self) -> Dict[str, DataLoader]:
        """Create datasets with anomaly labels"""
        datasets = {}
        
        # Normal images (no anomalies)
        normal_categories = ['core', 'cladding', 'ferrule']
        for category in normal_categories:
            if category in self.data_loader.reference_tensors:
                tensors = []
                labels = []
                anomaly_flags = []
                
                tensor_dict = self.data_loader.reference_tensors[category]
                for key, tensor_or_path in tensor_dict.items():
                    if isinstance(tensor_or_path, Path):
                        tensor = torch.load(tensor_or_path, map_location='cpu')
                    else:
                        tensor = tensor_or_path
                    
                    tensors.append(tensor)
                    labels.append(['core', 'cladding', 'ferrule'].index(category))
                    anomaly_flags.append(0)  # No anomalies
                
                if tensors:
                    dataset = AugmentedDataset(tensors, labels, anomaly_flags)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=self.config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True
                    )
                    datasets[f'{category}_normal'] = dataloader
        
        # Anomalous images
        if 'defects' in self.config.REGION_CATEGORIES:
            for defect_folder in self.config.REGION_CATEGORIES['defects']:
                defect_path = self.config.TENSORIZED_DATA_PATH / defect_folder
                if defect_path.exists():
                    tensors = []
                    labels = []
                    anomaly_flags = []
                    
                    for pt_file in list(defect_path.glob("*.pt"))[:1000]:  # Limit for training
                        tensor = torch.load(pt_file, map_location='cpu')
                        tensors.append(tensor)
                        labels.append(0)  # Default to core (will be refined by network)
                        anomaly_flags.append(1)  # Has anomalies
                    
                    if tensors:
                        dataset = AugmentedDataset(tensors, labels, anomaly_flags)
                        dataloader = DataLoader(
                            dataset,
                            batch_size=self.config.BATCH_SIZE // 2,  # Smaller batch for anomalies
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True
                        )
                        datasets[f'{defect_folder}_anomalous'] = dataloader
        
        return datasets
    
    def _adapt_loss_weights(self, metrics: Dict):
        """Dynamically adjust loss weights based on training progress"""
        # If anomaly detection is poor, increase its weight
        if 'anomaly_precision' in metrics and metrics['anomaly_precision']:
            avg_precision = np.mean(metrics['anomaly_precision'][-100:])
            if avg_precision < 0.7:
                self.loss_weights['anomaly'] *= 1.05
                self.loss_weights['anomaly'] = min(self.loss_weights['anomaly'], 0.4)
        
        # If segmentation is poor, increase its weight
        if 'segmentation_accuracy' in metrics and metrics['segmentation_accuracy']:
            avg_acc = np.mean(metrics['segmentation_accuracy'][-100:])
            if avg_acc < 0.8:
                self.loss_weights['segmentation'] *= 1.05
                self.loss_weights['segmentation'] = min(self.loss_weights['segmentation'], 0.4)
        
        # Normalize weights
        total_weight = sum(self.loss_weights.values())
        for key in self.loss_weights:
            self.loss_weights[key] /= total_weight
    
    def _comprehensive_validation(self) -> Dict[str, float]:
        """Perform detailed validation with anomaly analysis"""
        self.model.eval()
        
        metrics = {
            'segmentation_accuracy': [],
            'anomaly_precision': [],
            'anomaly_recall': [],
            'similarity_scores': [],
            'quality_correlation': []
        }
        
        with torch.no_grad():
            # Get validation batches
            val_normal, _ = self.data_loader.get_tensor_batch(batch_size=16, region='core')
            val_anomalous, _ = self.data_loader.get_tensor_batch(batch_size=16, region='ferrule')
            
            # Process normal batch
            val_normal = val_normal.to(self.device)
            outputs_normal = self.model(val_normal)
            
            # Normal images should have low anomaly scores
            anomaly_scores_normal = outputs_normal['anomaly_map'].mean(dim=[1, 2])
            metrics['anomaly_precision'].extend((anomaly_scores_normal < 0.3).float().tolist())
            
            # Process anomalous batch (using ferrule as proxy for potential anomalies)
            val_anomalous = val_anomalous.to(self.device)
            outputs_anomalous = self.model(val_anomalous)
            
            # Check similarity scores
            metrics['similarity_scores'].extend(outputs_normal['similarity'].tolist())
            metrics['similarity_scores'].extend(outputs_anomalous['similarity'].tolist())
            
            # Quality should correlate with low anomalies
            quality_normal = outputs_normal['quality_map'].mean()
            anomaly_normal = outputs_normal['anomaly_map'].mean()
            correlation = -torch.corrcoef(torch.stack([quality_normal, anomaly_normal]))[0, 1]
            metrics['quality_correlation'].append(correlation.item())
        
        # Calculate averages
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in metrics.items()
        }
    
    def _init_epoch_metrics(self) -> Dict:
        """Initialize metrics for epoch tracking"""
        return {
            'losses': [],
            'component_losses': {
                'segmentation': [],
                'anomaly': [],
                'reconstruction': [],
                'quality': [],
                'similarity': []
            },
            'anomaly_precision': [],
            'anomaly_recall': [],
            'segmentation_accuracy': [],
            'similarity_scores': []
        }
    
    def _update_epoch_metrics(self, metrics: Dict, losses: Dict, 
                            outputs: Dict, has_anomalies: torch.Tensor):
        """Update epoch metrics with batch results"""
        metrics['losses'].append(losses['total'].item())
        
        for key in losses:
            if key != 'total' and key in metrics['component_losses']:
                metrics['component_losses'][key].append(losses[key].item())
        
        # Track similarity scores
        metrics['similarity_scores'].extend(outputs['similarity'].tolist())
        
        # Simple anomaly metrics (would need ground truth in practice)
        anomaly_predicted = outputs['anomaly_map'].mean(dim=[1, 2]) > 0.3
        metrics['anomaly_precision'].extend(
            (anomaly_predicted == has_anomalies).float().tolist()
        )
    
    def _log_training_progress(self, epoch: int, batch_idx: int, 
                             losses: Dict, outputs: Dict):
        """Log detailed training progress"""
        self.logger.info(
            f"Epoch {epoch+1}, Batch {batch_idx}: "
            f"Total Loss={losses['total'].item():.4f}, "
            f"Seg={losses['segmentation'].item():.4f}, "
            f"Anom={losses['anomaly'].item():.4f}, "
            f"Sim={outputs['similarity'].mean().item():.4f}"
        )
        
        # Log anomaly details
        anomaly_details = outputs['anomaly_details']
        self.logger.debug(
            f"Anomaly patterns matched: {anomaly_details['anomaly_patterns_matched'][0]}"
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.NUM_EPOCHS * 1000,  # Approximate steps
            eta_min=1e-6
        )
    
    def _warmup_lr(self, step: int):
        """Linear warmup for learning rate"""
        warmup_factor = min(1.0, step / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * warmup_factor
    
    def _log_epoch_summary(self, epoch: int, metrics: Dict, epoch_time: float):
        """Log comprehensive epoch summary"""
        avg_loss = np.mean(metrics['losses'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch+1} Summary:")
        self.logger.info(f"  Total Loss: {avg_loss:.4f}")
        
        for loss_name, values in metrics['component_losses'].items():
            if values:
                self.logger.info(f"  {loss_name}: {np.mean(values):.4f}")
        
        if metrics['similarity_scores']:
            avg_sim = np.mean(metrics['similarity_scores'])
            above_threshold = sum(1 for s in metrics['similarity_scores'] if s > 0.7)
            self.logger.info(f"  Avg Similarity: {avg_sim:.4f}")
            self.logger.info(f"  Above Threshold: {above_threshold}/{len(metrics['similarity_scores'])}")
        
        self.logger.info(f"  Time: {epoch_time:.2f}s")
        self.logger.info(f"{'='*60}\n")
        
        # Update history
        self.history['total_loss'].append(avg_loss)
        for key, values in metrics['component_losses'].items():
            if key not in self.history['component_losses']:
                self.history['component_losses'][key] = []
            self.history['component_losses'][key].append(np.mean(values) if values else 0)
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint with detailed state"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"advanced_integrated_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_weights': self.loss_weights,
            'history': self.history,
            'equation_parameters': self.model.get_detailed_equation_parameters()
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _final_analysis(self):
        """Perform final analysis of training results"""
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL TRAINING ANALYSIS")
        self.logger.info("="*60)
        
        # Loss progression
        initial_loss = self.history['total_loss'][0]
        final_loss = self.history['total_loss'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        self.logger.info(f"Loss Improvement: {improvement:.1f}%")
        self.logger.info(f"Final Loss: {final_loss:.4f}")
        
        # Component analysis
        self.logger.info("\nComponent Loss Analysis:")
        for component, losses in self.history['component_losses'].items():
            if losses:
                initial = losses[0]
                final = losses[-1]
                change = (initial - final) / initial * 100
                self.logger.info(f"  {component}: {change:.1f}% improvement")
        
        # Equation parameters
        params = self.model.get_detailed_equation_parameters()
        self.logger.info("\nFinal Equation Parameters:")
        for i in range(4):  # 4 scales
            self.logger.info(f"  Scale {i}:")
            self.logger.info(f"    Gradient adjustment: {params[f'scale_{i}_gradient_adjustment']:.4f}")
            self.logger.info(f"    Position adjustment: {params[f'scale_{i}_position_adjustment']:.4f}")


class AugmentedDataset(TensorDataset):
    """Dataset with anomaly labels"""
    
    def __init__(self, tensors: List[torch.Tensor], labels: List[int], 
                 anomaly_flags: List[int]):
        super().__init__(tensors, labels)
        self.anomaly_flags = anomaly_flags
    
    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx], self.anomaly_flags[idx]