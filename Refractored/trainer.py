# trainer.py
# Training logic and utilities for the fiber optic analysis system

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

from model import create_model
from losses import get_loss_function
from evaluator import Evaluator

class Trainer:
    """Handles the training process for the fiber optic analysis model."""
    
    def __init__(self, config, rank=0, world_size=1, local_rank=0, is_distributed=False):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = is_distributed
        self.device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize model, optimizer, and loss
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler(enabled=config.training.use_amp)
        
        # Initialize evaluator
        self.evaluator = Evaluator(config, self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        
    def _setup_model(self):
        """Initialize and configure the model."""
        self.model = create_model(self.config).to(self.device)
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
            
        self.logger.info(f"Model initialized on device: {self.device}")
        
    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.num_epochs
        )
        
    def _setup_loss(self):
        """Initialize loss function."""
        self.loss_fn = get_loss_function(self.config)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            ref_images = batch['reference'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.use_amp):
                outputs = self.model(
                    images, 
                    ref_images, 
                    equation_coeffs=self.config.equation.coefficients
                )
                
                # Calculate loss
                loss_dict = self.loss_fn(outputs, batch, self.device)
                loss = loss_dict['total']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Log progress
            if self.rank == 0 and batch_idx % self.config.training.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                    f"[{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Cls: {loss_dict['classification']:.3f}, "
                    f"Anom: {loss_dict['anomaly']:.3f}, "
                    f"Sim: {loss_dict['similarity']:.3f})"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        checkpoint_dir = Path(self.config.system.checkpoints_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            avg_loss = self.train_epoch(train_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
                self.logger.info(f"Average training loss: {avg_loss:.4f}")
                
                # Evaluate on validation set
                val_metrics = self.evaluator.evaluate(self.model, val_loader)
                
                # Save checkpoint
                self._save_checkpoint(epoch, avg_loss, val_metrics, checkpoint_dir)
                
                # Log validation results
                self.logger.info(
                    f"Validation - Accuracy: {val_metrics['accuracy']:.3f}, "
                    f"Avg Similarity: {val_metrics['avg_similarity']:.3f}"
                )
        
        self.logger.info("Training completed!")
        
    def _save_checkpoint(self, epoch, train_loss, val_metrics, checkpoint_dir):
        """Save model checkpoint."""
        # Get model state dict (handle DDP)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch+1}.pth")
        
        # Save best checkpoint
        current_score = val_metrics.get('avg_similarity', 0.0)
        if current_score > self.best_score:
            self.best_score = current_score
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            self.logger.info(f"New best model saved with score: {current_score:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            # Load optimizer and scheduler state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False

def create_trainer(config, rank=0, world_size=1, local_rank=0, is_distributed=False):
    """Factory function to create trainer instance."""
    return Trainer(config, rank, world_size, local_rank, is_distributed)
