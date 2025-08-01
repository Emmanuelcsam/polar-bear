#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed Training Version - Pure CNN-based Fiber Optic Quality Assurance System
Optimized for multi-GPU training on William & Mary Bora HPC cluster
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms as T
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
import argparse

# Import the main architecture components
from fiber_cnn_pure import (
    AttentionGate, MBConvBlock, FiberEncoder, FiberDecoder,
    CombinedLoss, FiberAnalysisNet, FiberDataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_distributed(rank, world_size):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()

def train_model_distributed(rank, world_size, model, train_loader, val_loader, 
                          device, num_epochs=50, lr=1e-3):
    """Distributed training loop with modern best practices"""
    
    # Combined loss for multi-task learning
    zone_criterion = CombinedLoss(alpha=0.25, gamma=2.0, dice_weight=0.6)
    defect_criterion = CombinedLoss(alpha=0.5, gamma=2.0, dice_weight=0.7)  
    quality_criterion = nn.CrossEntropyLoss()
    
    # Modern optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training for efficiency
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_losses = {'zone': 0, 'defect': 0, 'quality': 0, 'total': 0}
        
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            zones_gt = batch['zones'].to(device)
            defects_gt = batch['defects'].to(device) 
            quality_gt = batch['quality'].squeeze().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    zone_loss = zone_criterion(outputs['zones'], zones_gt)
                    defect_loss = defect_criterion(outputs['defects'], defects_gt)
                    quality_loss = quality_criterion(outputs['quality'], quality_gt)
                    total_loss = zone_loss + defect_loss + 0.5 * quality_loss
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                zone_loss = zone_criterion(outputs['zones'], zones_gt)
                defect_loss = defect_criterion(outputs['defects'], defects_gt)
                quality_loss = quality_criterion(outputs['quality'], quality_gt)
                total_loss = zone_loss + defect_loss + 0.5 * quality_loss
                
                total_loss.backward()
                optimizer.step()
            
            # Update running losses
            epoch_losses['zone'] += zone_loss.item()
            epoch_losses['defect'] += defect_loss.item()
            epoch_losses['quality'] += quality_loss.item()
            epoch_losses['total'] += total_loss.item()
            
            if batch_idx % 10 == 0 and rank == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {total_loss.item():.4f}')
        
        scheduler.step()
        
        # Log epoch summary (only on rank 0)
        if rank == 0:
            num_batches = len(train_loader)
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            logger.info(f'Epoch {epoch} Summary - Zone: {avg_losses["zone"]:.4f}, '
                       f'Defect: {avg_losses["defect"]:.4f}, Quality: {avg_losses["quality"]:.4f}')

def main_distributed(rank, world_size, args):
    """Main distributed training function"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'reference_dir': args.reference_dir, 
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'image_size': args.image_size,
        'lr': args.lr,
        'output_dir': args.output_dir
    }
    
    # Create output directory (only on rank 0)
    if rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device(f'cuda:{rank}')
    logger.info(f"Rank {rank}: Using device: {device}")
    
    # Data augmentation pipeline
    train_transform = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets and dataloaders with distributed sampling
    train_dataset = FiberDataset(config['data_dir'], config['reference_dir'], 
                                train_transform, mode='train')
    
    # Split for validation (or use separate validation directory)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_subset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_subset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'], 
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        logger.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Train model
    train_model_distributed(rank, world_size, model, train_loader, val_loader, 
                          device, config['epochs'], config['lr'])
    
    # Save trained model (only on rank 0)
    if rank == 0:
        model_path = os.path.join(config['output_dir'], 'fiber_analysis_model.pth')
        torch.save(model.module.state_dict(), model_path)
        logger.info(f"Model saved successfully to {model_path}!")
    
    # Cleanup
    cleanup_distributed()

def main():
    """Main entry point for distributed training"""
    
    parser = argparse.ArgumentParser(description='Distributed Fiber Optic Quality Assurance CNN')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--reference-dir', type=str, default='reference', help='Reference directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=8, help='Total number of GPUs')
    
    args = parser.parse_args()
    
    # Get distributed training parameters from environment
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    
    # Start distributed training
    main_distributed(local_rank, world_size, args)

if __name__ == "__main__":
    main() 