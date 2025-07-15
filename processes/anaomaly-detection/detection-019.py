#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


@dataclass
class OmniConfig:
    """Enhanced configuration with deep learning parameters"""
    # Original parameters
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    
    # Deep learning parameters
    use_deep_features: bool = True
    embedding_dim: int = 128
    batch_size: int = 32
    learning_rate: float = 0.001
    margin: float = 1.0
    temperature: float = 0.1
    num_epochs: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone: str = 'resnet50'
    use_correlation_layer: bool = True
    ensemble_size: int = 3
    hard_mining_ratio: float = 0.5
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


class CorrelationLayer(nn.Module):
    """Correlation layer from FlowNet - computes matching scores between feature maps"""
    def __init__(self, max_displacement=20):
        super().__init__()
        self.max_displacement = max_displacement
        
    def forward(self, feat1, feat2):
        """Compute correlation between two feature maps"""
        b, c, h, w = feat1.shape
        
        # Pad feat2 for displacement matching
        pad = self.max_displacement
        feat2_pad = F.pad(feat2, (pad, pad, pad, pad))
        
        # Initialize correlation tensor
        corr = torch.zeros(b, (2*pad+1)**2, h, w).to(feat1.device)
        
        # Compute correlation for each displacement
        for i in range(2*pad+1):
            for j in range(2*pad+1):
                feat2_slice = feat2_pad[:, :, i:i+h, j:j+w]
                corr[:, i*(2*pad+1)+j] = (feat1 * feat2_slice).sum(dim=1)
                
        return corr


class SiameseBackbone(nn.Module):
    """Feature extraction backbone for Siamese network"""
    def __init__(self, backbone='resnet50', embedding_dim=128, pretrained=True):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, x):
        """Extract features and project to embedding space"""
        features = self.backbone(x)
        embedding = self.embedding(features)
        return F.normalize(embedding, p=2, dim=1)  # L2 normalize


class FiberSiameseNet(nn.Module):
    """Siamese network for fiber anomaly detection with correlation layer"""
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        self.backbone = SiameseBackbone(
            backbone=config.backbone,
            embedding_dim=config.embedding_dim
        )
        
        # Correlation layer for local matching
        if config.use_correlation_layer:
            self.correlation = CorrelationLayer(max_displacement=10)
            
            # Additional conv layers after correlation
            self.corr_conv = nn.Sequential(
                nn.Conv2d(441, 256, 3, padding=1),  # 441 = (2*20+1)^2
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
        
        # Decision network
        input_dim = config.embedding_dim * 4 if not config.use_correlation_layer else config.embedding_dim * 4 + 128
        self.decision = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        
    def forward_once(self, x):
        """Forward pass through one branch"""
        return self.backbone(x)
    
    def forward(self, anchor, positive=None, negative=None):
        """Forward pass for triplet or pair"""
        # Extract embeddings
        anchor_emb = self.forward_once(anchor)
        
        if positive is not None and negative is not None:
            # Triplet mode
            positive_emb = self.forward_once(positive)
            negative_emb = self.forward_once(negative)
            return anchor_emb, positive_emb, negative_emb
        elif positive is not None:
            # Pair mode
            positive_emb = self.forward_once(positive)
            
            # Compute similarity features
            features = torch.cat([
                anchor_emb,
                positive_emb,
                torch.abs(anchor_emb - positive_emb),
                anchor_emb * positive_emb
            ], dim=1)
            
            # Add correlation features if enabled
            if self.config.use_correlation_layer and hasattr(self, 'correlation'):
                # Get intermediate feature maps for correlation
                # This requires modifying backbone to return intermediate features
                # For now, we'll skip this in the forward pass
                pass
            
            # Similarity score
            similarity = self.decision(features)
            return anchor_emb, positive_emb, similarity
        else:
            # Single image mode
            return anchor_emb


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, label):
        """
        label: 1 for similar pairs, 0 for dissimilar pairs
        """
        distance = F.pairwise_distance(emb1, emb2)
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)
        return loss.mean()


class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """Compute triplet loss"""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class HierarchicalTripletSampler:
    """Hierarchical sampling strategy for triplet selection"""
    def __init__(self, features, labels, n_clusters=8):
        self.features = features
        self.labels = labels
        self.n_clusters = n_clusters
        self._build_hierarchy()
        
    def _build_hierarchy(self):
        """Build hierarchical tree of classes"""
        # Compute class centers
        unique_labels = np.unique(self.labels)
        class_centers = []
        
        for label in unique_labels:
            mask = self.labels == label
            center = self.features[mask].mean(axis=0)
            class_centers.append(center)
            
        # Hierarchical clustering
        self.kmeans = KMeans(n_clusters=min(self.n_clusters, len(unique_labels)))
        self.cluster_labels = self.kmeans.fit_predict(class_centers)
        
    def sample_triplets(self, n_triplets):
        """Sample triplets using hierarchical strategy"""
        triplets = []
        
        for _ in range(n_triplets):
            # Sample anchor class
            anchor_label = np.random.choice(np.unique(self.labels))
            anchor_indices = np.where(self.labels == anchor_label)[0]
            anchor_idx = np.random.choice(anchor_indices)
            
            # Sample positive from same class
            positive_indices = anchor_indices[anchor_indices != anchor_idx]
            if len(positive_indices) > 0:
                positive_idx = np.random.choice(positive_indices)
            else:
                continue
                
            # Sample hard negative from similar cluster
            anchor_cluster = self.cluster_labels[anchor_label]
            similar_classes = np.where(
                (self.cluster_labels == anchor_cluster) & 
                (np.arange(len(self.cluster_labels)) != anchor_label)
            )[0]
            
            if len(similar_classes) > 0:
                negative_label = np.random.choice(similar_classes)
                negative_indices = np.where(self.labels == negative_label)[0]
                negative_idx = np.random.choice(negative_indices)
                
                triplets.append((anchor_idx, positive_idx, negative_idx))
                
        return triplets


class FiberDataset(Dataset):
    """Dataset for fiber images with similarity labels"""
    def __init__(self, image_paths, labels=None, transform=None, mode='train'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Handle JSON format
            image = self._load_from_json(img_path)
            
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.labels is not None:
            return image, self.labels[idx]
        return image
        
    def _load_from_json(self, json_path):
        """Load image from JSON format"""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        channels = data['image_dimensions'].get('channels', 3)
        
        matrix = np.zeros((height, width, channels), dtype=np.uint8)
        
        for pixel in data['pixels']:
            x = pixel['coordinates']['x']
            y = pixel['coordinates']['y']
            
            if 0 <= x < width and 0 <= y < height:
                bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                if isinstance(bgr, (int, float)):
                    bgr = [bgr] * 3
                matrix[y, x] = bgr[:3]
                
        return matrix


class DeepFiberAnalyzer(OmniFiberAnalyzer):
    """Enhanced analyzer using deep learning and Siamese networks"""
    
    def __init__(self, config: OmniConfig):
        super().__init__(config)
        self.device = torch.device(config.device)
        
        if config.use_deep_features:
            # Initialize Siamese network
            self.siamese_net = FiberSiameseNet(config).to(self.device)
            
            # Initialize ensemble if requested
            if config.ensemble_size > 1:
                self.ensemble = [
                    FiberSiameseNet(config).to(self.device) 
                    for _ in range(config.ensemble_size)
                ]
            else:
                self.ensemble = [self.siamese_net]
                
            # Data transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Load pretrained model if exists
            self._load_deep_models()
            
    def _load_deep_models(self):
        """Load pretrained deep models"""
        model_path = Path(self.knowledge_base_path).parent / 'deep_models'
        if model_path.exists():
            for i, model in enumerate(self.ensemble):
                checkpoint_path = model_path / f'siamese_model_{i}.pth'
                if checkpoint_path.exists():
                    model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    self.logger.info(f"Loaded deep model {i} from {checkpoint_path}")
                    
    def extract_deep_features(self, image):
        """Extract features using deep Siamese network"""
        # Prepare image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Transform and batch
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features from ensemble
        features_list = []
        with torch.no_grad():
            for model in self.ensemble:
                model.eval()
                features = model.forward_once(img_tensor)
                features_list.append(features.cpu().numpy())
                
        # Average ensemble features
        deep_features = np.mean(features_list, axis=0).squeeze()
        
        return deep_features
        
    def compute_deep_similarity(self, img1, img2):
        """Compute similarity using deep features"""
        # Extract deep features
        feat1 = self.extract_deep_features(img1)
        feat2 = self.extract_deep_features(img2)
        
        # Compute similarity metrics
        cosine_sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        
        # Use Siamese network for learned similarity
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.siamese_net.eval()
            _, _, similarity = self.siamese_net(img1_tensor, img2_tensor)
            learned_sim = torch.sigmoid(similarity).item()
            
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_distance': float(euclidean_dist),
            'learned_similarity': float(learned_sim)
        }
        
    def detect_anomalies_comprehensive(self, test_path):
        """Enhanced anomaly detection using both classical and deep features"""
        # Run original detection
        results = super().detect_anomalies_comprehensive(test_path)
        
        if results is None or not self.config.use_deep_features:
            return results
            
        # Enhance with deep learning
        test_image = results['test_image']
        archetype = self.reference_model.get('archetype_image')
        
        if archetype is not None:
            # Compute deep similarity
            deep_sim = self.compute_deep_similarity(test_image, archetype)
            results['deep_analysis'] = deep_sim
            
            # Update verdict based on deep features
            if deep_sim['learned_similarity'] < 0.5:
                results['verdict']['is_anomalous'] = True
                results['verdict']['confidence'] = max(
                    results['verdict']['confidence'],
                    1.0 - deep_sim['learned_similarity']
                )
                
        # Extract deep features for the test image
        deep_features = self.extract_deep_features(test_image)
        results['deep_features'] = deep_features.tolist()
        
        return results
        
    def train_similarity_model(self, train_dir, val_dir=None, epochs=None):
        """Train Siamese network for fiber similarity learning"""
        if epochs is None:
            epochs = self.config.num_epochs
            
        self.logger.info("Training deep similarity model...")
        
        # Prepare datasets
        train_dataset = self._prepare_dataset(train_dir, mode='train')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = None
        if val_dir:
            val_dataset = self._prepare_dataset(val_dir, mode='val')
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4
            )
            
        # Initialize optimizers and losses
        optimizers = []
        for model in self.ensemble:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate
            )
            optimizers.append(optimizer)
            
        contrastive_loss = ContrastiveLoss(margin=self.config.margin)
        triplet_loss = TripletLoss(margin=self.config.margin)
        
        # Training loop
        for epoch in range(epochs):
            # Train each model in ensemble
            for model_idx, (model, optimizer) in enumerate(zip(self.ensemble, optimizers)):
                model.train()
                total_loss = 0
                
                for batch_idx, batch_data in enumerate(train_loader):
                    # Prepare batch based on loss type
                    if len(batch_data) == 3:  # Triplet
                        anchor, positive, negative = batch_data
                        anchor = anchor.to(self.device)
                        positive = positive.to(self.device)
                        negative = negative.to(self.device)
                        
                        # Forward pass
                        anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
                        
                        # Compute loss
                        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
                        
                    else:  # Pairs
                        img1, img2, labels = batch_data
                        img1 = img1.to(self.device)
                        img2 = img2.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        emb1, emb2, _ = model(img1, img2)
                        
                        # Compute loss
                        loss = contrastive_loss(emb1, emb2, labels)
                        
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Model {model_idx}, Loss: {avg_loss:.4f}")
                
            # Validation
            if val_loader:
                self._validate_models(val_loader, contrastive_loss)
                
        # Save trained models
        self._save_deep_models()
        
    def _prepare_dataset(self, data_dir, mode='train'):
        """Prepare dataset for training/validation"""
        # This is a simplified version - in practice, you'd need proper data loading
        # with triplet sampling, augmentation, etc.
        image_paths = []
        labels = []
        
        # Collect all images
        for class_dir in Path(data_dir).iterdir():
            if class_dir.is_dir():
                class_label = class_dir.name
                for img_path in class_dir.glob('*'):
                    if img_path.suffix in ['.png', '.jpg', '.jpeg', '.json']:
                        image_paths.append(str(img_path))
                        labels.append(class_label)
                        
        # Convert labels to integers
        unique_labels = sorted(set(labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_to_int[label] for label in labels]
        
        dataset = FiberDataset(image_paths, labels, self.transform, mode)
        return dataset
        
    def _save_deep_models(self):
        """Save trained deep models"""
        model_path = Path(self.knowledge_base_path).parent / 'deep_models'
        model_path.mkdir(exist_ok=True)
        
        for i, model in enumerate(self.ensemble):
            checkpoint_path = model_path / f'siamese_model_{i}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            self.logger.info(f"Saved model {i} to {checkpoint_path}")


def main():
    """Enhanced main function with deep learning support"""
    print("\n" + "="*80)
    print("ENHANCED OMNIRIBER ANALYZER WITH DEEP LEARNING (v2.0)".center(80))
    print("="*80)
    
    # Create configuration with deep learning enabled
    config = OmniConfig(use_deep_features=True)
    
    # Initialize enhanced analyzer
    analyzer = DeepFiberAnalyzer(config)
    
    while True:
        print("\nOptions:")
        print("1. Analyze single image")
        print("2. Train similarity model")
        print("3. Build reference model")
        print("4. Exit")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            if os.path.isfile(test_path):
                output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                print(f"\nAnalyzing {test_path}...")
                analyzer.analyze_end_face(test_path, output_dir)
                print(f"Results saved to: {output_dir}/")
            else:
                print(f"File not found: {test_path}")
                
        elif choice == '2':
            train_dir = input("Enter training data directory: ").strip().strip('"\'')
            val_dir = input("Enter validation data directory (optional): ").strip().strip('"\'')
            if not val_dir:
                val_dir = None
            epochs = input("Number of epochs (default 50): ").strip()
            epochs = int(epochs) if epochs else None
            
            if os.path.isdir(train_dir):
                analyzer.train_similarity_model(train_dir, val_dir, epochs)
            else:
                print(f"Directory not found: {train_dir}")
                
        elif choice == '3':
            ref_dir = input("Enter reference data directory: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                analyzer.build_comprehensive_reference_model(ref_dir)
            else:
                print(f"Directory not found: {ref_dir}")
                
        elif choice == '4':
            break
            
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
