import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import cv2
import numpy as np
import os
import yaml
import logging
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from scipy.stats import jarque_bera
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

class FiberOpticDataset(Dataset):
    """Custom dataset for fiber optic endface images with statistical feature integration"""
    
    def __init__(self, dataset_path, reference_path, config, mode='train'):
        self.dataset_path = Path(dataset_path)
        self.reference_path = Path(reference_path)
        self.config = config
        self.mode = mode
        
        # Load reference tensors for similarity computation
        self.reference_tensors = self._load_reference_tensors()
        
        # Collect image paths from chunk folders
        self.image_paths = []
        for chunk_dir in sorted(self.dataset_path.glob('chunk_*')):
            for img_path in chunk_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.image_paths.append(img_path)
        
        # Statistical parameters from analysis
        self.feature_stats = {
            'center_x': {'mean': 595.8718, 'std': 116.3471},
            'center_y': {'mean': 447.1887, 'std': 87.0680},
            'core_radius': {'mean': 114.4197, 'std': 78.4117},
            'cladding_radius': {'mean': 208.8852, 'std': 100.5034},
            'core_cladding_ratio': {'mean': 0.5302, 'std': 0.1822}
        }
        
        # Initialize transforms based on statistical analysis recommendations
        self.transforms = self._get_transforms()
        
        logging.info(f"Loaded {len(self.image_paths)} images from {len(list(self.dataset_path.glob('chunk_*')))} chunks")
    
    def _load_reference_tensors(self):
        """Load reference .pt files for similarity computation"""
        reference_tensors = {}
        for ref_dir in self.reference_path.iterdir():
            if ref_dir.is_dir():
                ref_tensors = []
                for pt_file in ref_dir.glob('*.pt'):
                    tensor = torch.load(pt_file, map_location='cpu')
                    ref_tensors.append(tensor)
                if ref_tensors:
                    reference_tensors[ref_dir.name] = torch.stack(ref_tensors)
        
        logging.info(f"Loaded reference tensors from {len(reference_tensors)} categories")
        return reference_tensors
    
    def _get_transforms(self):
        """Advanced augmentation strategies based on statistical analysis"""
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.3),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract statistical features using OpenCV
        statistical_features = self._extract_statistical_features(image)
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image']
        
        # Compute similarity with reference images
        similarity_scores = self._compute_similarity_scores(statistical_features)
        
        return {
            'image': image_tensor,
            'features': torch.tensor(statistical_features, dtype=torch.float32),
            'similarity_scores': torch.tensor(similarity_scores, dtype=torch.float32),
            'path': str(img_path)
        }
    
    def _extract_statistical_features(self, image):
        """Extract geometric and statistical features using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic statistical features
        features = []
        features.extend([
            np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        
        # GLCM features (simplified)
        from skimage.feature import graycomatrix, graycoprops
        glcm = graycomatrix(gray, distances=[1, 2, 3], angles=[0, 45, 90, 135])
        for prop in ['energy', 'contrast']:
            features.extend(graycoprops(glcm, prop).flatten()[:8])
        
        # Hu moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)
        
        # Geometric features (circle detection for core/cladding)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=300)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) >= 2:
                # Sort by radius
                circles = circles[circles[:, 2].argsort()]
                core_circle = circles[0]  # Smallest
                cladding_circle = circles[-1]  # Largest
                
                features.extend([
                    core_circle[0], core_circle[1], core_circle[2],  # center_x, center_y, core_radius
                    cladding_circle[0], cladding_circle[1], cladding_circle[2],  # cladding center and radius
                    core_circle[2] / cladding_circle[2] if cladding_circle[2] > 0 else 0  # core_cladding_ratio
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])  # Default values
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0])
        
        return features[:50]  # Limit to 50 features for consistency
    
    def _compute_similarity_scores(self, features):
        """Compute similarity scores using master equation"""
        if len(features) < 7:
            return [0.0] * len(self.reference_tensors)
        
        scores = []
        center_x, center_y = features[0], features[1]
        core_radius, cladding_radius = features[2], features[3]
        core_cladding_ratio = features[6] if len(features) > 6 else 0.5
        num_valid_results = 7.0  # Default from statistical analysis
        
        # Master similarity equation weights
        weights = {
            'center_x': 0.362261,
            'center_y': 0.202874,
            'core_radius': 0.164540,
            'cladding_radius': 0.270316,
            'core_cladding_ratio': 8.887e-07,
            'num_valid_results': 8.298e-06
        }
        
        for ref_name, ref_tensors in self.reference_tensors.items():
            if len(ref_tensors) > 0:
                # Use mean reference values (simplified)
                ref_center_x, ref_center_y = 595.87, 447.19
                ref_core_radius, ref_cladding_radius = 114.42, 208.89
                ref_core_cladding_ratio, ref_num_valid_results = 0.53, 6.84
                
                # Compute weighted distance
                distance = math.sqrt(
                    weights['center_x'] * (center_x - ref_center_x)**2 +
                    weights['center_y'] * (center_y - ref_center_y)**2 +
                    weights['core_radius'] * (core_radius - ref_core_radius)**2 +
                    weights['cladding_radius'] * (cladding_radius - ref_cladding_radius)**2 +
                    weights['core_cladding_ratio'] * (core_cladding_ratio - ref_core_cladding_ratio)**2 +
                    weights['num_valid_results'] * (num_valid_results - ref_num_valid_results)**2
                )
                
                similarity = math.exp(-distance)
                scores.append(similarity)
            else:
                scores.append(0.0)
        
        return scores

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor based on statistical analysis"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone CNN with feature extraction capabilities
        self.backbone = models.efficientnet_b4(pretrained=True)
        self.feature_extractor = create_feature_extractor(
            self.backbone,
            return_nodes={
                'features.2': 'low_level',
                'features.4': 'mid_level', 
                'features.6': 'high_level'
            }
        )
        
        # Adaptive pooling for consistent feature sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(240, 128, 3, padding=1)  # Adjust based on EfficientNet features
        self.fusion_bn = nn.BatchNorm2d(128)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Process multi-scale features
        low = self.adaptive_pool(features['low_level'])
        mid = self.adaptive_pool(features['mid_level'])
        high = self.adaptive_pool(features['high_level'])
        
        # Concatenate features
        fused = torch.cat([low, mid, high], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))
        
        return fused, features

class RegionSegmentationHead(nn.Module):
    """Segmentation head for core, cladding, and ferrule regions"""
    
    def __init__(self, in_channels, num_classes=4):  # background, core, cladding, ferrule
        super().__init__()
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Upsampling to match input resolution
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        seg_logits = self.seg_head(x)
        seg_logits = self.upsample(seg_logits)
        return seg_logits

class DefectDetectionHead(nn.Module):
    """Defect detection head with statistical guidance"""
    
    def __init__(self, in_channels, config):
        super().__init__()
        self.config = config
        
        # Defect classification layers
        self.defect_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Statistical feature integration
        self.stat_fc = nn.Sequential(
            nn.Linear(50, 128),  # 50 statistical features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Final defect classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32 + 64, 128),  # Conv features + statistical features
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, config['num_defect_classes'])
        )
        
    def forward(self, conv_features, stat_features):
        # Process convolutional features
        conv_out = self.defect_conv(conv_features)
        conv_pooled = self.classifier[0](conv_out)  # AdaptiveAvgPool2d
        conv_flattened = self.classifier[1](conv_pooled)  # Flatten
        
        # Process statistical features
        stat_out = self.stat_fc(stat_features)
        
        # Combine features
        combined = torch.cat([conv_flattened, stat_out], dim=1)
        
        # Final classification
        defect_logits = self.classifier[2:](combined)
        
        return defect_logits

class FiberOpticDefectNet(nn.Module):
    """Complete fiber optic defect detection network"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extraction backbone
        self.feature_extractor = MultiScaleFeatureExtractor(config)
        
        # Region segmentation head
        self.segmentation_head = RegionSegmentationHead(128, config['num_region_classes'])
        
        # Defect detection head
        self.defect_head = DefectDetectionHead(128, config)
        
        # Regression heads for geometric parameters (based on statistical analysis)
        self.core_radius_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        self.cladding_radius_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights based on statistical analysis"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch):
        images = batch['image']
        stat_features = batch['features']
        
        # Extract multi-scale features
        fused_features, backbone_features = self.feature_extractor(images)
        
        # Region segmentation
        seg_logits = self.segmentation_head(fused_features)
        
        # Defect detection
        defect_logits = self.defect_head(fused_features, stat_features)
        
        # Geometric parameter regression
        core_radius_pred = self.core_radius_regressor(fused_features)
        cladding_radius_pred = self.cladding_radius_regressor(fused_features)
        
        return {
            'segmentation': seg_logits,
            'defects': defect_logits,
            'core_radius': core_radius_pred,
            'cladding_radius': cladding_radius_pred,
            'features': fused_features
        }

class MultiTaskLoss(nn.Module):
    """Multi-task loss function with statistical weighting"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.seg_loss = nn.CrossEntropyLoss()
        self.defect_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        # Loss weights based on statistical analysis
        self.loss_weights = {
            'segmentation': config['seg_weight'],
            'defects': config['defect_weight'],
            'core_radius': config['core_radius_weight'],
            'cladding_radius': config['cladding_radius_weight']
        }
    
    def forward(self, predictions, targets):
        total_loss = 0
        loss_dict = {}
        
        if 'seg_targets' in targets:
            seg_loss = self.seg_loss(predictions['segmentation'], targets['seg_targets'])
            total_loss += self.loss_weights['segmentation'] * seg_loss
            loss_dict['seg_loss'] = seg_loss
        
        if 'defect_targets' in targets:
            defect_loss = self.defect_loss(predictions['defects'], targets['defect_targets'])
            total_loss += self.loss_weights['defects'] * defect_loss
            loss_dict['defect_loss'] = defect_loss
        
        if 'core_radius_targets' in targets:
            core_loss = self.regression_loss(predictions['core_radius'], targets['core_radius_targets'])
            total_loss += self.loss_weights['core_radius'] * core_loss
            loss_dict['core_radius_loss'] = core_loss
        
        if 'cladding_radius_targets' in targets:
            cladding_loss = self.regression_loss(predictions['cladding_radius'], targets['cladding_radius_targets'])
            total_loss += self.loss_weights['cladding_radius'] * cladding_loss
            loss_dict['cladding_radius_loss'] = cladding_loss
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict

def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Training epoch with comprehensive logging"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch)
        
        # Create dummy targets for demonstration (in practice, these would be real labels)
        targets = {
            'defect_targets': torch.randint(0, config['num_defect_classes'], (batch['image'].size(0),)).to(device)
        }
        
        # Compute loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['gradient_clip_val'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % config['log_interval'] == 0:
            logging.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches

def main():
    """Main training function"""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create datasets
    train_dataset = FiberOpticDataset('dataset', 'reference', config, mode='train')
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = FiberOpticDefectNet(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )
    
    # Create loss function
    criterion = MultiTaskLoss(config)
    
    # Training loop
    for epoch in range(config['epochs']):
        logging.info(f'Starting epoch {epoch+1}/{config["epochs"]}')
        
        # Train epoch
        avg_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        
        logging.info(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            logging.info(f'Checkpoint saved for epoch {epoch+1}')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    logging.info('Training completed and final model saved')

if __name__ == '__main__':
    main()
