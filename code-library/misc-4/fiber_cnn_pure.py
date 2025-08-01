#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pure CNN-based Fiber Optic Quality Assurance System
All processing handled within neural network - no classical CV methods
Designed for William & Mary Bora HPC cluster deployment
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern attention mechanism
class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant regions"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# EfficientNet-inspired backbone
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze and Excitation
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//4, hidden_dim, 1),
            nn.Sigmoid()
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers[:-6])
        self.se = nn.Sequential(*layers[-6:-2])
        self.project = nn.Sequential(*layers[-2:])

    def forward(self, x):
        identity = x
        x = self.conv(x)
        
        # Squeeze and Excitation
        se_weight = self.se(x)
        x = x * se_weight
        
        x = self.project(x)
        
        if self.use_residual:
            x = x + identity
        return x

# Modern encoder with EfficientNet-inspired blocks
class FiberEncoder(nn.Module):
    """Multi-scale feature extraction encoder"""
    def __init__(self, in_channels=3):
        super(FiberEncoder, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Multi-scale feature extraction blocks
        self.stage1 = nn.Sequential(
            MBConvBlock(32, 64, stride=1),
            MBConvBlock(64, 64, stride=1)
        )
        
        self.stage2 = nn.Sequential(
            MBConvBlock(64, 128, stride=2),
            MBConvBlock(128, 128, stride=1),
            MBConvBlock(128, 128, stride=1)
        )
        
        self.stage3 = nn.Sequential(
            MBConvBlock(128, 256, stride=2),
            MBConvBlock(256, 256, stride=1),
            MBConvBlock(256, 256, stride=1),
            MBConvBlock(256, 256, stride=1)
        )
        
        self.stage4 = nn.Sequential(
            MBConvBlock(256, 512, stride=2),
            MBConvBlock(512, 512, stride=1),
            MBConvBlock(512, 512, stride=1)
        )
        
        self.stage5 = nn.Sequential(
            MBConvBlock(512, 1024, stride=2),
            MBConvBlock(1024, 1024, stride=1)
        )

    def forward(self, x):
        x0 = self.stem(x)      # 32 x H/2 x W/2
        x1 = self.stage1(x0)   # 64 x H/2 x W/2
        x2 = self.stage2(x1)   # 128 x H/4 x W/4
        x3 = self.stage3(x2)   # 256 x H/8 x W/8
        x4 = self.stage4(x3)   # 512 x H/16 x W/16
        x5 = self.stage5(x4)   # 1024 x H/32 x W/32
        
        return [x1, x2, x3, x4, x5]

# Multi-task decoder with attention
class FiberDecoder(nn.Module):
    """Multi-task decoder for zone segmentation and defect detection"""
    def __init__(self, encoder_channels=[64, 128, 256, 512, 1024]):
        super(FiberDecoder, self).__init__()
        
        # Decoder blocks with attention gates
        self.attention4 = AttentionGate(encoder_channels[4], encoder_channels[3], 256)
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[4], 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 + encoder_channels[3], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.attention3 = AttentionGate(256, encoder_channels[2], 128)
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 + encoder_channels[2], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.attention2 = AttentionGate(128, encoder_channels[1], 64)
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 + encoder_channels[1], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.attention1 = AttentionGate(64, encoder_channels[0], 32)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 + encoder_channels[0], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        
        # Decoder path with attention
        d4 = self.decoder4[0](x5)  # Upsample
        d4 = self.decoder4[1:3](d4)  # BN + ReLU
        att4 = self.attention4(d4, x4)
        d4 = torch.cat([d4, att4], dim=1)
        d4 = self.decoder4[3:](d4)  # Conv + BN + ReLU
        
        d3 = self.decoder3[0](d4)
        d3 = self.decoder3[1:3](d3)
        att3 = self.attention3(d3, x3)
        d3 = torch.cat([d3, att3], dim=1)
        d3 = self.decoder3[3:](d3)
        
        d2 = self.decoder2[0](d3)
        d2 = self.decoder2[1:3](d2)
        att2 = self.attention2(d2, x2)
        d2 = torch.cat([d2, att2], dim=1)
        d2 = self.decoder2[3:](d2)
        
        d1 = self.decoder1[0](d2)
        d1 = self.decoder1[1:3](d1)
        att1 = self.attention1(d1, x1)
        d1 = torch.cat([d1, att1], dim=1)
        d1 = self.decoder1[3:](d1)
        
        return d1

# Combined loss functions based on latest research
class CombinedLoss(nn.Module):
    """Combined Focal + Dice Loss for optimal performance on imbalanced data"""
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance"""
        pred_sigmoid = torch.sigmoid(pred)
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt + 1e-8)
        return loss.mean()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """Dice loss for better boundary detection"""
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice

# Main model architecture
class FiberAnalysisNet(nn.Module):
    """End-to-end fiber optic quality analysis network"""
    def __init__(self, in_channels=3, num_zones=3, num_defect_types=4):
        super(FiberAnalysisNet, self).__init__()
        
        self.encoder = FiberEncoder(in_channels)
        self.decoder = FiberDecoder()
        
        # Multi-task heads
        # Zone segmentation (core, cladding, ferrule)
        self.zone_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_zones, 1)
        )
        
        # Defect detection (scratches, pits, contamination, edge_defects)  
        self.defect_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_defect_types, 1)
        )
        
        # Global quality classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.quality_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # pass, warning, fail
        )

    def forward(self, x):
        # Feature extraction
        features = self.encoder(x)
        
        # Decode features
        decoded = self.decoder(features)
        
        # Multi-task outputs
        zones = self.zone_head(decoded)
        defects = self.defect_head(decoded)
        
        # Global quality assessment
        global_features = self.global_pool(features[-1]).flatten(1)
        quality = self.quality_head(global_features)
        
        return {
            'zones': zones,
            'defects': defects, 
            'quality': quality
        }

# Dataset class for hierarchical data structure
class FiberDataset(Dataset):
    """Dataset handler for hierarchical fiber optic data"""
    def __init__(self, data_dir, reference_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.mode = mode
        
        # Find all images in chunk subdirectories
        self.image_paths = []
        for chunk_dir in sorted(self.data_dir.glob('chunk_*')):
            chunk_images = list(chunk_dir.glob('*.png')) + list(chunk_dir.glob('*.jpg'))
            self.image_paths.extend(chunk_images)
        
        logger.info(f"Found {len(self.image_paths)} images in {mode} mode")
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings()
        
    def _load_reference_embeddings(self):
        """Load reference .pt files as prototype embeddings"""
        embeddings = {}
        for ref_file in self.reference_dir.rglob('*.pt'):
            try:
                embedding = torch.load(ref_file, map_location='cpu')
                embeddings[ref_file.stem] = embedding
            except Exception as e:
                logger.warning(f"Could not load {ref_file}: {e}")
        
        logger.info(f"Loaded {len(embeddings)} reference embeddings")
        return embeddings
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Generate synthetic targets for now (replace with real annotations)
        h, w = image.shape[1], image.shape[2]
        zones_mask = self._generate_synthetic_zones(h, w)
        defects_mask = self._generate_synthetic_defects(h, w)
        quality_label = torch.randint(0, 3, (1,)).long()  # Random quality label
        
        return {
            'image': image,
            'zones': zones_mask,
            'defects': defects_mask,
            'quality': quality_label,
            'path': str(img_path)
        }
    
    def _generate_synthetic_zones(self, h, w):
        """Generate synthetic zone masks (replace with real annotations)"""
        # Create circular zones for demonstration
        center_x, center_y = w // 2, h // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Core (innermost circle)
        core_radius = min(h, w) * 0.1
        core_mask = (dist <= core_radius).float()
        
        # Cladding (middle ring)
        cladding_radius = min(h, w) * 0.25
        cladding_mask = ((dist > core_radius) & (dist <= cladding_radius)).float()
        
        # Ferrule (outer region)
        ferrule_mask = (dist > cladding_radius).float()
        
        return torch.stack([core_mask, cladding_mask, ferrule_mask])
    
    def _generate_synthetic_defects(self, h, w):
        """Generate synthetic defect masks (replace with real annotations)"""
        # Random defect locations for demonstration
        defects = torch.zeros(4, h, w)  # 4 defect types
        
        # Add some random small defects
        for defect_type in range(4):
            num_defects = torch.randint(0, 5, (1,)).item()
            for _ in range(num_defects):
                x = torch.randint(10, w-10, (1,)).item()
                y = torch.randint(10, h-10, (1,)).item()
                size = torch.randint(3, 8, (1,)).item()
                
                y_min, y_max = max(0, y-size), min(h, y+size)
                x_min, x_max = max(0, x-size), min(w, x+size)
                defects[defect_type, y_min:y_max, x_min:x_max] = 1.0
        
        return defects

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3):
    """Training loop with modern best practices"""
    
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
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {total_loss.item():.4f}')
        
        scheduler.step()
        
        # Log epoch summary
        num_batches = len(train_loader)
        avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
        logger.info(f'Epoch {epoch} Summary - Zone: {avg_losses["zone"]:.4f}, '
                   f'Defect: {avg_losses["defect"]:.4f}, Quality: {avg_losses["quality"]:.4f}')

# Main execution
def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Fiber Optic Quality Assurance CNN')
    parser.add_argument('--data-dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--reference-dir', type=str, default='reference', help='Reference directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--image-size', type=int, default=512, help='Image size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
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
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    
    # Datasets and dataloaders
    train_dataset = FiberDataset(config['data_dir'], config['reference_dir'], 
                                train_transform, mode='train')
    
    # Split for validation (or use separate validation directory)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    model = FiberAnalysisNet(in_channels=3, num_zones=3, num_defect_types=4)
    model = model.to(device)
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Train model
    train_model(model, train_loader, val_loader, device, config['epochs'], config['lr'])
    
    # Save trained model
    model_path = os.path.join(config['output_dir'], 'fiber_analysis_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved successfully to {model_path}!")

if __name__ == "__main__":
    main() 