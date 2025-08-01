# optimizer.py
# Model optimization utilities (pruning, knowledge distillation)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
from typing import Optional

from model import create_model

class ModelOptimizer:
    """Handles model optimization techniques like pruning and knowledge distillation."""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prune_model(self, model, ratio=0.3):
        """
        Apply global unstructured pruning to the model.
        
        Args:
            model: Model to prune
            ratio: Pruning ratio (fraction of weights to remove)
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Starting model pruning with ratio: {ratio}")
        
        # Collect parameters to prune
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.logger.info("Model pruning completed")
        return model
    
    def calculate_model_size(self, model):
        """Calculate model size in parameters and memory."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (assuming float32)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_mb': memory_mb
        }
    
    def distill_model(self, teacher_model, train_loader, val_loader, save_path=None):
        """
        Perform knowledge distillation to create a smaller student model.
        
        Args:
            teacher_model: Pre-trained teacher model
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save the distilled model
            
        Returns:
            Trained student model
        """
        self.logger.info("Starting knowledge distillation...")
        
        # Create student model
        student_config = self.config.student_model
        student_model = create_model(type('Config', (), {'model': student_config})()).to(self.device)
        
        # Log model sizes
        teacher_size = self.calculate_model_size(teacher_model)
        student_size = self.calculate_model_size(student_model)
        
        self.logger.info(f"Teacher model: {teacher_size['total_parameters']:,} parameters")
        self.logger.info(f"Student model: {student_size['total_parameters']:,} parameters")
        self.logger.info(f"Compression ratio: {teacher_size['total_parameters'] / student_size['total_parameters']:.2f}x")
        
        # Setup training components
        optimizer = optim.AdamW(
            student_model.parameters(),
            lr=self.config.optimizer.learning_rate
        )
        
        distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
        classification_loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.config.training.use_amp)
        
        # Training parameters
        temperature = self.config.optimization.temperature
        alpha = self.config.optimization.alpha
        num_epochs = self.config.optimization.distillation_epochs
        
        teacher_model.eval()
        
        for epoch in range(num_epochs):
            student_model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=self.config.training.use_amp):
                    # Get teacher predictions (no gradients)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(images)
                    
                    # Get student predictions
                    student_outputs = student_model(images)
                    
                    # Calculate distillation loss
                    teacher_logits = teacher_outputs['region_logits'] / temperature
                    student_logits = student_outputs['region_logits'] / temperature
                    
                    soft_targets = F.softmax(teacher_logits, dim=1)
                    soft_predictions = F.log_softmax(student_logits, dim=1)
                    
                    loss_distill = distillation_loss_fn(soft_predictions, soft_targets) * (temperature ** 2)
                    
                    # Calculate student classification loss
                    loss_student = classification_loss_fn(student_outputs['region_logits'], labels)
                    
                    # Combine losses
                    total_loss_batch = alpha * loss_distill + (1 - alpha) * loss_student
                
                # Backward pass
                scaler.scale(total_loss_batch).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Evaluate student model
            student_accuracy = self._evaluate_student(student_model, val_loader)
            
            self.logger.info(
                f"Distillation Epoch {epoch+1}/{num_epochs}: "
                f"Loss = {avg_loss:.4f}, "
                f"Student Accuracy = {student_accuracy:.3f}"
            )
        
        # Save student model
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': student_model.state_dict(),
                'config': student_config,
                'teacher_size': teacher_size,
                'student_size': student_size
            }
            
            torch.save(checkpoint, save_path)
            self.logger.info(f"Student model saved to: {save_path}")
        
        self.logger.info("Knowledge distillation completed")
        return student_model
    
    def _evaluate_student(self, student_model, val_loader):
        """Evaluate student model accuracy."""
        student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.config.training.use_amp):
                    outputs = student_model(images)
                
                _, predicted = torch.max(outputs['region_logits'], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def optimize_model(self, model, train_loader, val_loader, checkpoint_dir):
        """
        Apply all configured optimization techniques.
        
        Args:
            model: Model to optimize
            train_loader: Training data loader
            val_loader: Validation data loader
            checkpoint_dir: Directory to save optimized models
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply pruning if configured
        if self.config.optimization.get('prune_after_training', False):
            self.logger.info("Applying model pruning...")
            pruned_model = self.prune_model(
                model, 
                ratio=self.config.optimization.pruning_ratio
            )
            
            # Save pruned model
            pruned_path = checkpoint_dir / "pruned_model.pth"
            torch.save(pruned_model.state_dict(), pruned_path)
            self.logger.info(f"Pruned model saved to: {pruned_path}")
        
        # Apply knowledge distillation if configured
        if self.config.optimization.get('distill_after_training', False):
            self.logger.info("Starting knowledge distillation...")
            student_model = self.distill_model(
                teacher_model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                save_path=checkpoint_dir / "distilled_student_model.pth"
            )
            
            return student_model
        
        return model

def create_optimizer(config, device='cpu'):
    """Factory function to create model optimizer instance."""
    return ModelOptimizer(config, device)
