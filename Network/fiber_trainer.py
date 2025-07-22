#!/usr/bin/env python3
"""
Trainer module for Fiber Optics Neural Network
"the program will run entirely in hpc meaning that it has to be able to run on gpu and in parallel"
"the program will calculate its losses and try to minimize its losses by small percentile adjustments to parameters"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from config import get_config
from logger import get_logger
from integrated_network import FiberOpticsIntegratedNetwork
from data_loader import FiberOpticsDataLoader

class FiberOpticsTrainer:
    """
    Trainer for the integrated fiber optics neural network
    "over time the program will get more accurate and faster at computing"
    """
    
    def __init__(self, model: Optional[FiberOpticsIntegratedNetwork] = None):
        print(f"[{datetime.now()}] Initializing FiberOpticsTrainer")
        print(f"[{datetime.now()}] Previous script: integrated_network.py")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsTrainer")
        
        self.logger.log_class_init("FiberOpticsTrainer")
        
        # Initialize model
        if model is None:
            self.model = FiberOpticsIntegratedNetwork()
        else:
            self.model = model
        
        # Move to device (GPU if available)
        self.device = self.config.get_device()
        self.model = self.model.to(self.device)
        
        # Initialize data loader
        self.data_loader = FiberOpticsDataLoader()
        
        # Loss functions
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.similarity_loss = nn.MSELoss()
        
        # Custom loss weights
        self.loss_weights = {
            'segmentation': 0.25,
            'reconstruction': 0.20,
            'similarity': 0.30,
            'anomaly': 0.15,
            'trend': 0.10
        }
        
        # Optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_similarity': [],
            'val_similarity': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = self.config.CHECKPOINTS_PATH / "best_model.pth"
        
        self.logger.info("FiberOpticsTrainer initialized")
        self.logger.info(f"Using device: {self.device}")
        print(f"[{datetime.now()}] FiberOpticsTrainer initialized successfully")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with parameter groups"""
        # Different learning rates for different components
        param_groups = [
            # Feature extraction layers
            {'params': self.model.feature_extractor.parameters(), 
             'lr': self.config.LEARNING_RATE},
            
            # Segmentation network
            {'params': self.model.segmentation_net.parameters(), 
             'lr': self.config.LEARNING_RATE * 0.5},
            
            # Reference embeddings - slower learning
            {'params': [self.model.reference_embeddings], 
             'lr': self.config.LEARNING_RATE * 0.1},
            
            # Reconstruction decoder
            {'params': self.model.decoder.parameters(), 
             'lr': self.config.LEARNING_RATE * 0.5},
            
            # Other parameters
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'feature_extractor' not in n and 
                          'segmentation_net' not in n and 
                          'reference_embeddings' not in n and 
                          'decoder' not in n],
             'lr': self.config.LEARNING_RATE}
        ]
        
        return optim.AdamW(param_groups, weight_decay=0.01)
    
    def train(self, num_epochs: Optional[int] = None, 
             train_loader: Optional[DataLoader] = None,
             val_loader: Optional[DataLoader] = None):
        """
        Train the model
        "but this is only for the training sequence so that it can be trained fast and effectively"
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        self.logger.log_process_start(f"Training for {num_epochs} epochs")
        
        # Get data loaders if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader = self.data_loader.get_data_loaders()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Log epoch start
            self.logger.log_epoch_start(epoch + 1, num_epochs)
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            self._log_epoch_results(epoch, train_loss, train_metrics, 
                                   val_loss, val_metrics, epoch_time)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_similarity'].append(train_metrics['avg_similarity'])
            self.history['val_similarity'].append(val_metrics['avg_similarity'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_times'].append(epoch_time)
        
        self.logger.log_process_end(f"Training for {num_epochs} epochs")
        self._save_training_history()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        metrics = {
            'avg_similarity': 0,
            'threshold_met_count': 0,
            'avg_anomaly_score': 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Get batch data
            images = batch['image'].to(self.device)
            labels = batch.get('label', torch.zeros(images.shape[0], dtype=torch.long)).to(self.device)
            has_anomalies = batch.get('has_anomaly', torch.zeros(images.shape[0], dtype=torch.bool)).to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses
            losses = self._calculate_losses(outputs, images, labels, has_anomalies)
            total_loss_batch = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # "small percentile adjustments to parameters"
            self.optimizer.step()
            
            # Update metrics
            batch_size = images.shape[0]
            total_loss += total_loss_batch.item() * batch_size
            total_samples += batch_size
            
            metrics['avg_similarity'] += outputs['final_similarity'].sum().item()
            metrics['threshold_met_count'] += outputs['meets_threshold'].sum().item()
            metrics['avg_anomaly_score'] += outputs['anomaly_map'].mean().item() * batch_size
            
            # Log batch progress
            if batch_idx % 10 == 0:
                self.logger.log_batch_progress(
                    batch_idx, 
                    len(train_loader),
                    total_loss_batch.item(),
                    similarity=outputs['final_similarity'].mean().item(),
                    anomaly=outputs['anomaly_map'].mean().item()
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        metrics['avg_similarity'] /= total_samples
        metrics['avg_anomaly_score'] /= total_samples
        metrics['threshold_met_ratio'] = metrics['threshold_met_count'] / total_samples
        
        return avg_loss, metrics
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        metrics = {
            'avg_similarity': 0,
            'threshold_met_count': 0,
            'avg_anomaly_score': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                images = batch['image'].to(self.device)
                labels = batch.get('label', torch.zeros(images.shape[0], dtype=torch.long)).to(self.device)
                has_anomalies = batch.get('has_anomaly', torch.zeros(images.shape[0], dtype=torch.bool)).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate losses
                losses = self._calculate_losses(outputs, images, labels, has_anomalies)
                
                # Update metrics
                batch_size = images.shape[0]
                total_loss += losses['total'].item() * batch_size
                total_samples += batch_size
                
                metrics['avg_similarity'] += outputs['final_similarity'].sum().item()
                metrics['threshold_met_count'] += outputs['meets_threshold'].sum().item()
                metrics['avg_anomaly_score'] += outputs['anomaly_map'].mean().item() * batch_size
        
        # Calculate validation metrics
        avg_loss = total_loss / total_samples
        metrics['avg_similarity'] /= total_samples
        metrics['avg_anomaly_score'] /= total_samples
        metrics['threshold_met_ratio'] = metrics['threshold_met_count'] / total_samples
        
        return avg_loss, metrics
    
    def _calculate_losses(self, outputs: Dict[str, torch.Tensor], 
                         images: torch.Tensor,
                         labels: torch.Tensor,
                         has_anomalies: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components
        "the program will calculate its losses and try to minimize its losses"
        """
        losses = {}
        
        # Segmentation loss
        if labels.max() > 0:  # Only if we have labels
            losses['segmentation'] = self.segmentation_loss(outputs['segmentation'], labels)
        else:
            losses['segmentation'] = torch.tensor(0.0, device=self.device)
        
        # Reconstruction loss
        losses['reconstruction'] = self.reconstruction_loss(outputs['reconstruction'], images)
        
        # Similarity loss - encourage high similarity
        # "the program must achieve over .7"
        target_similarity = torch.ones_like(outputs['final_similarity']) * 0.8
        losses['similarity'] = self.similarity_loss(outputs['final_similarity'], target_similarity)
        
        # Anomaly loss - supervised when we have labels
        if has_anomalies.any():
            # For images with anomalies, anomaly map should have high values
            anomaly_target = has_anomalies.float().view(-1, 1, 1).expand_as(outputs['anomaly_map'])
            losses['anomaly'] = F.binary_cross_entropy(
                outputs['anomaly_map'].clamp(0, 1), 
                anomaly_target
            )
        else:
            # For normal images, anomaly scores should be low
            losses['anomaly'] = outputs['anomaly_map'].mean()
        
        # Trend adherence loss
        # High trend adherence for normal regions
        losses['trend'] = -outputs['trend_adherence'].mean()  # Negative because we want to maximize
        
        # Total weighted loss
        losses['total'] = sum(
            self.loss_weights.get(name, 1.0) * loss 
            for name, loss in losses.items() 
            if name != 'total'
        )
        
        return losses
    
    def _log_epoch_results(self, epoch: int, train_loss: float, train_metrics: Dict,
                          val_loss: float, val_metrics: Dict, epoch_time: float):
        """Log epoch results"""
        self.logger.info(f"\nEpoch {epoch + 1} Results:")
        self.logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        self.logger.info(f"  Train Similarity: {train_metrics['avg_similarity']:.4f}, "
                        f"Val Similarity: {val_metrics['avg_similarity']:.4f}")
        self.logger.info(f"  Train Threshold Met: {train_metrics['threshold_met_ratio']:.2%}, "
                        f"Val Threshold Met: {val_metrics['threshold_met_ratio']:.2%}")
        self.logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        self.logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_weights': self.loss_weights,
            'history': self.history,
            'config': {
                'equation_coefficients': self.config.EQUATION_COEFFICIENTS,
                'gradient_weight_factor': self.config.GRADIENT_WEIGHT_FACTOR,
                'position_weight_factor': self.config.POSITION_WEIGHT_FACTOR
            }
        }
        
        if is_best:
            save_path = self.best_model_path
            self.logger.info(f"Saving best model to {save_path}")
        else:
            save_path = self.config.CHECKPOINTS_PATH / f"checkpoint_epoch_{epoch}.pth"
            self.logger.info(f"Saving checkpoint to {save_path}")
        
        torch.save(checkpoint, save_path)
    
    def _save_training_history(self):
        """Save training history"""
        history_path = self.config.CHECKPOINTS_PATH / "training_history.npz"
        np.savez(history_path, **self.history)
        self.logger.info(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.logger.log_function_entry("load_checkpoint", path=checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'loss_weights' in checkpoint:
            self.loss_weights = checkpoint['loss_weights']
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        if 'config' in checkpoint:
            # Update configuration with saved values
            config_data = checkpoint['config']
            self.config.EQUATION_COEFFICIENTS.update(config_data.get('equation_coefficients', {}))
            self.config.GRADIENT_WEIGHT_FACTOR = config_data.get('gradient_weight_factor', 1.0)
            self.config.POSITION_WEIGHT_FACTOR = config_data.get('position_weight_factor', 1.0)
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        self.logger.log_function_exit("load_checkpoint")
    
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
            train_loss, train_metrics = self._train_epoch(new_data_loader, epoch)
            
            self.logger.info(f"Fine-tune Epoch {epoch + 1}/{num_epochs}: "
                           f"Loss={train_loss:.4f}, Similarity={train_metrics['avg_similarity']:.4f}")
        
        self.logger.log_process_end(f"Fine-tuning for {num_epochs} epochs")

# Test the trainer
if __name__ == "__main__":
    trainer = FiberOpticsTrainer()
    logger = get_logger("TrainerTest")
    
    logger.log_process_start("Trainer Test")
    
    # Create dummy data loader for testing
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    dummy_images = torch.randn(32, 3, 256, 256)
    dummy_labels = torch.randint(0, 3, (32,))
    dummy_anomalies = torch.randint(0, 2, (32,)).bool()
    
    dummy_dataset = TensorDataset(dummy_images, dummy_labels, dummy_anomalies)
    dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)
    
    # Modify loader to return dict format
    class DictDataLoader:
        def __init__(self, loader):
            self.loader = loader
        
        def __iter__(self):
            for images, labels, anomalies in self.loader:
                yield {
                    'image': images,
                    'label': labels,
                    'has_anomaly': anomalies
                }
        
        def __len__(self):
            return len(self.loader)
    
    dict_loader = DictDataLoader(dummy_loader)
    
    # Train for one epoch
    trainer.train(num_epochs=1, train_loader=dict_loader, val_loader=dict_loader)
    
    logger.log_process_end("Trainer Test")
    logger.log_script_transition("trainer.py", "data_loader.py")
    
    print(f"[{datetime.now()}] Trainer test completed")
    print(f"[{datetime.now()}] Next script: data_loader.py")
