#!/usr/bin/env python3
"""
Enhanced Trainer module for Fiber Optics Neural Network
Integrates all advanced optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import time
from collections import defaultdict
import wandb

from core.config_loader import get_config, update_config
from core.logger import get_logger
from logic.integrated_network import EnhancedIntegratedNetwork
from data.data_loader import FiberOpticsDataLoader
from utilities.optimizers import create_advanced_optimizer, SAMWithLookahead
from utilities.losses import create_loss_function
from utilities.hybrid_optimizer import create_hybrid_optimizer
from logic.real_time_optimization import (
    KnowledgeDistillationLoss, 
    EfficientFiberOpticsNetwork,
    ModelCompressor,
    create_student_teacher_models
)
from utilities.distributed_utils import (
    init_distributed, cleanup_distributed, is_main_process,
    wrap_model_ddp, save_checkpoint_distributed, synchronize,
    reduce_tensor, distributed_print, DistributedMetricTracker,
    get_rank, get_world_size
)


class EnhancedTrainer:
    """
    Enhanced trainer with all advanced optimization techniques
    Combines SAM+Lookahead, hybrid optimization, advanced losses, and more
    """
    
    def __init__(self, 
                 model: Optional[EnhancedIntegratedNetwork] = None,
                 teacher_model: Optional[nn.Module] = None,
                 distributed: bool = False):
        """
        Initialize enhanced trainer
        
        Args:
            model: Model to train (creates new if None)
            teacher_model: Teacher model for knowledge distillation
            distributed: Whether to use distributed training
        """
        print(f"[{datetime.now()}] Initializing EnhancedTrainer")
        print(f"[{datetime.now()}] Previous script: enhanced_integrated_network.py")
        
        self.config = get_config()
        self.distributed = distributed
        
        # Initialize distributed training if requested
        if self.distributed:
            self.rank, self.local_rank, self.world_size = init_distributed()
            self.logger = get_logger(f"EnhancedTrainer_rank{self.rank}")
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.logger = get_logger("EnhancedTrainer")
            
        self.logger.log_class_init("EnhancedTrainer")
        
        # Initialize model
        if model is None:
            self.model = EnhancedIntegratedNetwork()
        else:
            self.model = model
        
        # Teacher model for distillation
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model.eval()
            self.distillation_loss = KnowledgeDistillationLoss(
                alpha=self.config.training.distillation_alpha,
                temperature=self.config.training.distillation_temperature
            )
        
        # Move to device
        if self.distributed and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = self.config.get_device()
            
        self.model = self.model.to(self.device)
        if self.teacher_model:
            self.teacher_model = self.teacher_model.to(self.device)
            
        # Wrap model with DDP if distributed
        if self.distributed:
            self.model = wrap_model_ddp(
                self.model, 
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
            if self.teacher_model:
                self.teacher_model = wrap_model_ddp(
                    self.teacher_model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True
                )
        
        # Data loader
        self.data_loader = FiberOpticsDataLoader()
        
        # Create optimizers based on configuration
        self._create_optimizers()
        
        # Loss function with all advanced losses
        self.loss_fn = create_loss_function(self.config)
        
        # Mixed precision training
        self.use_amp = self.config.training.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Learning rate schedulers
        self._create_schedulers()
        
        # Training history
        self.history = defaultdict(list)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        
        # Distributed metric tracker
        if self.distributed:
            self.metric_tracker = DistributedMetricTracker()
        self.best_model_path = Path(self.config.system.checkpoints_path) / "best_model.pth"
        
        # Experiment tracking
        if self.config.monitoring.use_wandb:
            self._init_wandb()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        self.logger.info("EnhancedTrainer initialized with all improvements")
        print(f"[{datetime.now()}] EnhancedTrainer ready")
    
    def _create_optimizers(self):
        """Create optimizers based on configuration"""
        opt_type = self.config.optimizer.type
        
        if opt_type == "hybrid":
            # Hybrid optimizer with gradient descent + evolution
            def fitness_fn(model):
                # Simple fitness based on validation performance
                # In practice, this would evaluate on validation set
                return -self.best_val_loss  # Negative because we maximize fitness
            
            self.optimizer = create_hybrid_optimizer(
                self.model, 
                self.config,
                fitness_fn,
                adaptive=True
            )
            
            # For hybrid, we need special handling
            self.is_hybrid = True
            
        elif opt_type == "sam_lookahead":
            # SAM + Lookahead optimizer
            self.optimizer = create_advanced_optimizer(self.model, self.config)
            self.is_hybrid = False
            
        else:
            # Standard optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay
            )
            self.is_hybrid = False
        
        self.logger.info(f"Created {opt_type} optimizer")
    
    def _create_schedulers(self):
        """Create learning rate schedulers"""
        scheduler_config = self.config.optimizer.scheduler
        
        if scheduler_config.type == "reduce_on_plateau":
            # Get the base optimizer for scheduler
            if hasattr(self.optimizer, 'optimizer'):
                base_optimizer = self.optimizer.optimizer  # SAMWithLookahead
            elif hasattr(self.optimizer, 'gradient_optimizer'):
                base_optimizer = self.optimizer.gradient_optimizer  # HybridOptimizer
            else:
                base_optimizer = self.optimizer
                
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                base_optimizer,
                mode='min',
                patience=scheduler_config.patience,
                factor=scheduler_config.factor,
                min_lr=scheduler_config.min_lr
            )
        elif scheduler_config.type == "cosine":
            # Get the base optimizer for scheduler
            if hasattr(self.optimizer, 'optimizer'):
                base_optimizer = self.optimizer.optimizer  # SAMWithLookahead
            elif hasattr(self.optimizer, 'gradient_optimizer'):
                base_optimizer = self.optimizer.gradient_optimizer  # HybridOptimizer
            else:
                base_optimizer = self.optimizer
                
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                base_optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=scheduler_config.min_lr
            )
        else:
            self.scheduler = None
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        if is_main_process():
            wandb.init(
                project=self.config.monitoring.wandb_project,
                entity=self.config.monitoring.wandb_entity,
                config=self.config._to_dict() if hasattr(self.config, '_to_dict') else vars(self.config)
            )
            wandb.watch(self.model)
    
    def train(self, 
              num_epochs: Optional[int] = None,
              train_loader: Optional[DataLoader] = None,
              val_loader: Optional[DataLoader] = None):
        """
        Train the model with all enhancements
        
        Args:
            num_epochs: Number of epochs to train
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs
        
        self.logger.log_process_start(f"Enhanced training for {num_epochs} epochs")
        
        # Get data loaders if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader = self.data_loader.get_data_loaders(
                distributed=self.distributed
            )
        
        # Training loop
        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Dynamic configuration updates
            if self.config.check_for_updates():
                self.logger.info("Configuration updated, reloading...")
                self.config.reload()
            
            # Log epoch start
            self.logger.log_epoch_start(epoch + 1, num_epochs)
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log progress
            epoch_time = time.time() - epoch_start
            self._log_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoints (only on main process in distributed)
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                if is_main_process():
                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"New best model! Loss: {self.best_val_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0 and is_main_process():
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self._check_early_stopping():
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.log_process_end(f"Enhanced training completed")
        self._save_training_history()
    
    def fine_tune(self, new_data_loader: DataLoader, num_epochs: int = 10):
        """
        Fine-tune the model on new data
        "over time the program will get more accurate and faster at computing"
        """
        self.logger.log_process_start(f"Fine-tuning for {num_epochs} epochs")
        
        # Reduce learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1
        
        # Train with new data
        for epoch in range(num_epochs):
            train_metrics = self._train_epoch(new_data_loader, epoch)
            
            self.logger.info(f"Fine-tune Epoch {epoch + 1}/{num_epochs}: "
                           f"Loss={train_metrics['loss']:.4f}, Similarity={train_metrics['avg_similarity']:.4f}")
        
        self.logger.log_process_end(f"Fine-tuning for {num_epochs} epochs")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with all enhancements"""
        self.model.train()
        
        epoch_metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Training step with enhancements
            batch_metrics = self._training_step(batch)
            
            # Aggregate metrics
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self._log_batch_progress(batch_idx, num_batches, batch_metrics)
            
            self.global_step += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return dict(epoch_metrics)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with all optimizations"""
        # Extract batch data
        images = batch['image']
        targets = self._prepare_targets(batch)
        
        # Define closure for optimizers that need it
        def closure():
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    losses = self.loss_fn(outputs, targets)
                    total_loss = losses['total']
            else:
                outputs = self.model(images)
                losses = self.loss_fn(outputs, targets)
                total_loss = losses['total']
            
            # Add distillation loss if using teacher
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                
                distill_loss = self.distillation_loss(
                    outputs['segmentation'],
                    teacher_outputs['segmentation'],
                    targets.get('segmentation')
                )
                
                total_loss = total_loss + distill_loss
                losses['distillation'] = distill_loss
            
            # Added to store losses for metrics logging
            # Original code referenced non-existent self._last_losses in metrics; fixed by assigning here
            self._last_losses = losses
            
            return total_loss
        
        # Optimization step
        if self.is_hybrid:
            # Hybrid optimizer handles its own zero_grad
            loss = self.optimizer.step(closure)
        else:
            # Standard optimization
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # Mixed precision training
                loss = closure()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
                
                # Optimizer step
                if hasattr(self.optimizer, 'first_step'):
                    # SAM optimizer
                    self.optimizer.first_step(zero_grad=True)
                    
                    # Second forward-backward pass
                    closure()
                    self.scaler.scale(loss).backward()
                    self.optimizer.second_step()
                else:
                    self.scaler.step(self.optimizer)
                
                self.scaler.update()
            else:
                # Regular training
                loss = closure()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
                
                self.optimizer.step()
        
        # Collect metrics
        metrics = {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'lr': self._get_current_lr()
        }
        
        # Add individual loss components
        if hasattr(self, '_last_losses'):
            for key, value in self._last_losses.items():
                metrics[f'loss_{key}'] = value.item() if isinstance(value, torch.Tensor) else value
        
        return metrics
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with all metrics"""
        self.model.eval()
        
        val_metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._prepare_batch(batch)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch['image'])
                        losses = self.loss_fn(outputs, self._prepare_targets(batch))
                else:
                    outputs = self.model(batch['image'])
                    losses = self.loss_fn(outputs, self._prepare_targets(batch))
                
                # Collect metrics
                val_metrics['loss'] += losses['total'].item()
                val_metrics['similarity'] += outputs['final_similarity'].mean().item()
                val_metrics['meets_threshold'] += outputs['meets_threshold'].float().mean().item()
                
                # Component losses
                for key, value in losses.items():
                    if key != 'total':
                        val_metrics[f'loss_{key}'] += value.item()
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return dict(val_metrics)
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training"""
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()}
    
    def _prepare_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare target dictionary for loss calculation"""
        targets = {
            'image': batch['image'],
            'segmentation': batch.get('label', torch.zeros_like(batch['image'][:, 0]).long()),
            'has_anomaly': batch.get('has_anomaly', torch.zeros(batch['image'].shape[0], dtype=torch.bool)),
        }
        
        # Add more targets as needed
        if 'reference_label' in batch:
            targets['reference_labels'] = batch['reference_label']
        
        if 'gradient_distribution' in batch:
            targets['gradient_distribution'] = batch['gradient_distribution']
        
        return targets
    
    def _get_current_lr(self) -> float:
        """Get current learning rate"""
        if self.is_hybrid:
            return self.optimizer.gradient_optimizer.param_groups[0]['lr']
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def _log_batch_progress(self, batch_idx: int, num_batches: int, metrics: Dict[str, float]):
        """Log batch training progress"""
        progress = batch_idx / num_batches * 100
        lr = metrics.get('lr', 0)
        loss = metrics.get('loss', 0)
        
        self.logger.info(
            f"Batch [{batch_idx}/{num_batches}] ({progress:.0f}%) - "
            f"Loss: {loss:.4f}, LR: {lr:.6f}"
        )
        
        # Log to wandb
        if self.config.monitoring.use_wandb:
            wandb.log({
                'batch/loss': loss,
                'batch/lr': lr,
                'batch/step': self.global_step
            })
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, 
                          val_metrics: Dict, epoch_time: float):
        """Log epoch results"""
        self.logger.info(
            f"Epoch [{epoch+1}] - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Similarity: {val_metrics['similarity']:.4f}, "
            f"Meets Threshold: {val_metrics['meets_threshold']:.2%}, "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_similarity'].append(val_metrics['similarity'])
        self.history['epoch_time'].append(epoch_time)
        
        # Log to wandb
        if self.config.monitoring.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'val/loss': val_metrics['loss'],
                'val/similarity': val_metrics['similarity'],
                'val/meets_threshold': val_metrics['meets_threshold'],
                'epoch_time': epoch_time
            })
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered"""
        if len(self.history['val_loss']) < self.config.training.early_stopping_patience:
            return False
        
        recent_losses = self.history['val_loss'][-self.config.training.early_stopping_patience:]
        return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        # Get the actual model if using DDP
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config._to_dict() if hasattr(self.config, '_to_dict') else dict(self.config),
            'history': dict(self.history),
            'distributed': self.distributed,
            'world_size': self.world_size
        }
        
        # Save path
        checkpoint_dir = Path(self.config.system.checkpoints_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = checkpoint_dir / "best_model.pth"
        else:
            path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        
        if self.distributed:
            # Use distributed checkpoint saving
            save_checkpoint_distributed(checkpoint, str(path), is_best=is_best)
        else:
            torch.save(checkpoint, path)
            
        if is_main_process():
            self.logger.info(f"Saved checkpoint to {path}")
    
    def _save_training_history(self):
        """Save training history"""
        history_path = Path(self.config.system.checkpoints_path) / "training_history.npz"
        np.savez_compressed(history_path, **self.history)
        self.logger.info(f"Saved training history to {history_path}")
        
        # Visualize training history if enabled
        if self.config.visualization.save_visualizations and len(self.history['train_loss']) > 0:
            from config.visualizer import FiberOpticsVisualizer
            visualizer = FiberOpticsVisualizer()
            vis_path = Path(self.config.system.results_path) / 'training_history.png'
            
            # Ensure the results directory exists
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            
            visualizer.plot_training_history(dict(self.history), str(vis_path))
            self.logger.info(f"Saved training visualization to {vis_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle loading for DDP models
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Handle state dict keys that might have 'module.' prefix
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.') and not hasattr(self.model, 'module'):
            # Remove 'module.' prefix
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module.') and hasattr(self.model, 'module'):
            # Add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = defaultdict(list, checkpoint['history'])
        
        # Check distributed compatibility
        if checkpoint.get('distributed', False) and not self.distributed:
            self.logger.warning("Checkpoint was saved in distributed mode but loading in single GPU mode")
        elif not checkpoint.get('distributed', False) and self.distributed:
            self.logger.warning("Checkpoint was saved in single GPU mode but loading in distributed mode")
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# Test the enhanced trainer
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing EnhancedTrainer")
    
    # Create trainer
    trainer = EnhancedTrainer()
    
    # Create dummy data loader for testing
    from torch.utils.data import TensorDataset
    
    dummy_images = torch.randn(16, 3, 256, 256)
    dummy_labels = torch.randint(0, 3, (16, 256, 256))
    dummy_anomalies = torch.randint(0, 2, (16,), dtype=torch.bool)
    
    dummy_dataset = TensorDataset(dummy_images, dummy_labels, dummy_anomalies)
    dummy_loader = DataLoader(
        dummy_dataset,
        batch_size=4,
        shuffle=True
    )
    
    # Modify loader to return dict format
    def dict_collate(batch):
        images, labels, anomalies = zip(*batch)
        return {
            'image': torch.stack(images),
            'label': torch.stack(labels),
            'has_anomaly': torch.stack(anomalies)
        }
    
    dummy_loader.collate_fn = dict_collate
    
    # Test one epoch
    print("\nTesting one training epoch...")
    trainer.train(num_epochs=1, train_loader=dummy_loader, val_loader=dummy_loader)
    
    print(f"\n[{datetime.now()}] EnhancedTrainer test completed")
    print(f"[{datetime.now()}] Next script: Run fiber_visualization_ui.py to see the complete system!")