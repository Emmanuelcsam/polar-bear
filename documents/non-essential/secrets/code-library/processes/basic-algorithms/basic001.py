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
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# Set up device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class OmniConfig:
    """Enhanced configuration for OmniFiberAnalyzer with deep learning parameters"""
    # Original parameters
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    
    # Deep learning parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    image_size: Tuple[int, int] = (128, 128)
    num_workers: int = 2
    model_save_path: str = "fiber_cnn_model.pth"
    use_augmentation: bool = True
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


class FiberDataset(Dataset):
    """Custom Dataset for fiber optic images"""
    def __init__(self, image_paths: List[str], labels: Optional[List[int]] = None, 
                 transform=None, config: OmniConfig = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.config = config or OmniConfig()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        if self.image_paths[idx].endswith('.json'):
            image = self._load_from_json(self.image_paths[idx])
        else:
            image = cv2.imread(self.image_paths[idx])
            
        if image is None:
            # Return a blank image if loading fails
            image = np.zeros((self.config.image_size[0], self.config.image_size[1], 3), dtype=np.uint8)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize
        image = cv2.resize(image, self.config.image_size)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get label
        label = self.labels[idx] if self.labels is not None else 0
        
        return image, label
    
    def _load_from_json(self, json_path):
        """Load image from JSON format"""
        try:
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
            
        except Exception as e:
            logging.error(f"Error loading JSON {json_path}: {e}")
            return None


class FiberCNN(nn.Module):
    """CNN model for fiber anomaly detection inspired by TinyVGG architecture"""
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # Convolutional Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Convolutional Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Convolutional Block 3 (deeper for better feature extraction)
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Calculate size after convolutions
        # For 128x128 input: after 3 MaxPool2d(2): 128/2/2/2 = 16
        self.flatten_size = 128 * 16 * 16
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flatten_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.classifier(x)
        return x


class FiberAnomalyDetector(nn.Module):
    """Advanced anomaly detection model with segmentation capabilities"""
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)  # 512 because of skip connection
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output layer for anomaly segmentation
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        output = self.output(dec1)
        return torch.sigmoid(output)


def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, 
               optimizer: torch.optim.Optimizer, device: torch.device):
    """Performs a training step with model learning on dataloader"""
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Calculate accuracy for classification
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:  # Multi-class
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y)
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Loss backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
    
    # Adjust metrics
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc


def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, 
              device: torch.device):
    """Performs a testing step with model on dataloader"""
    model.eval()
    test_loss = 0
    test_acc = 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            
            # Calculate loss
            test_loss += loss_fn(test_pred, y).item()
            
            # Calculate accuracy
            if len(test_pred.shape) > 1 and test_pred.shape[1] > 1:
                test_pred_class = torch.argmax(test_pred, dim=1)
                test_acc += (test_pred_class == y).sum().item() / len(y)
    
    # Adjust metrics
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc


class EnhancedOmniFiberAnalyzer:
    """Enhanced analyzer using deep learning for fiber anomaly detection"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = device
        
        # Initialize models
        self.classification_model = None
        self.segmentation_model = None
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms for training
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def train_classification_model(self, train_images: List[str], train_labels: List[int],
                                 val_images: List[str], val_labels: List[int]):
        """Train the CNN classification model"""
        self.logger.info("Training CNN classification model...")
        
        # Create datasets
        train_dataset = FiberDataset(train_images, train_labels, 
                                   transform=self.train_transform if self.config.use_augmentation else self.transform,
                                   config=self.config)
        val_dataset = FiberDataset(val_images, val_labels, 
                                 transform=self.transform, config=self.config)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                    shuffle=True, num_workers=self.config.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size,
                                  shuffle=False, num_workers=self.config.num_workers)
        
        # Initialize model
        self.classification_model = FiberCNN(num_classes=2).to(self.device)
        
        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classification_model.parameters(), 
                                   lr=self.config.learning_rate)
        
        # Training loop
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in tqdm(range(self.config.num_epochs)):
            # Train step
            train_loss, train_acc = train_step(self.classification_model, train_dataloader,
                                             loss_fn, optimizer, self.device)
            
            # Test step
            val_loss, val_acc = test_step(self.classification_model, val_dataloader,
                                        loss_fn, self.device)
            
            # Store results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            
            # Print progress
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch: {epoch} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
        
        # Save model
        torch.save(self.classification_model.state_dict(), self.config.model_save_path)
        self.logger.info(f"Model saved to {self.config.model_save_path}")
        
        return results
    
    def analyze_end_face_dl(self, image_path: str, output_dir: str):
        """Analyze fiber end face using deep learning models"""
        self.logger.info(f"Analyzing fiber end face with DL: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        if image_path.endswith('.json'):
            image = self._load_from_json(image_path)
        else:
            image = cv2.imread(image_path)
            
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return None
            
        # Prepare image for model
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize for display
        display_image = cv2.resize(image_rgb, self.config.image_size)
        
        # Transform for model
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Make predictions
        results = {}
        
        # Classification prediction
        if self.classification_model is not None:
            self.classification_model.eval()
            with torch.inference_mode():
                logits = self.classification_model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                
                results['classification'] = {
                    'class': pred_class.item(),
                    'confidence': probs.max().item(),
                    'class_name': 'Anomalous' if pred_class.item() == 1 else 'Normal'
                }
        
        # Segmentation prediction (if model available)
        if self.segmentation_model is not None:
            self.segmentation_model.eval()
            with torch.inference_mode():
                seg_output = self.segmentation_model(image_tensor)
                seg_mask = (seg_output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
                
                results['segmentation'] = {
                    'mask': seg_mask,
                    'anomaly_pixels': np.sum(seg_mask > 0),
                    'total_pixels': seg_mask.size,
                    'anomaly_ratio': np.sum(seg_mask > 0) / seg_mask.size
                }
        
        # Visualize results
        if self.config.enable_visualization:
            self._visualize_dl_results(display_image, results, output_path, Path(image_path).stem)
        
        # Save results
        report_path = output_path / f"{Path(image_path).stem}_dl_report.json"
        with open(report_path, 'w') as f:
            json_results = {k: v for k, v in results.items() if k != 'segmentation'}
            if 'segmentation' in results:
                json_results['segmentation'] = {
                    'anomaly_pixels': results['segmentation']['anomaly_pixels'],
                    'total_pixels': results['segmentation']['total_pixels'],
                    'anomaly_ratio': results['segmentation']['anomaly_ratio']
                }
            json.dump(json_results, f, indent=2)
        
        return results
    
    def _visualize_dl_results(self, image: np.ndarray, results: Dict, output_path: Path, filename: str):
        """Visualize deep learning results"""
        fig, axes = plt.subplots(1, 2 if 'segmentation' in results else 1, figsize=(12, 6))
        
        if 'segmentation' not in results:
            axes = [axes]
        
        # Original image with classification result
        axes[0].imshow(image)
        if 'classification' in results:
            class_info = results['classification']
            color = 'green' if class_info['class'] == 0 else 'red'
            axes[0].set_title(f"{class_info['class_name']} (Confidence: {class_info['confidence']:.2%})", 
                            color=color, fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Segmentation mask if available
        if 'segmentation' in results:
            seg_info = results['segmentation']
            axes[1].imshow(image)
            axes[1].imshow(seg_info['mask'], alpha=0.5, cmap='Reds')
            axes[1].set_title(f"Anomaly Segmentation ({seg_info['anomaly_ratio']:.2%} anomalous)", 
                            fontsize=16)
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f"{filename}_dl_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrix(self, test_images: List[str], test_labels: List[int]):
        """Create confusion matrix for model evaluation"""
        if self.classification_model is None:
            self.logger.error("No trained model available")
            return None
            
        # Create test dataset and dataloader
        test_dataset = FiberDataset(test_images, test_labels, 
                                  transform=self.transform, config=self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size,
                                   shuffle=False, num_workers=self.config.num_workers)
        
        # Make predictions
        all_preds = []
        all_labels = []
        
        self.classification_model.eval()
        with torch.inference_mode():
            for X, y in tqdm(test_dataloader, desc="Making predictions"):
                X = X.to(self.device)
                logits = self.classification_model(X)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, 
                                     target_names=['Normal', 'Anomalous'])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=16)
        plt.colorbar()
        
        # Add labels
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Anomalous'], rotation=45)
        plt.yticks(tick_marks, ['Normal', 'Anomalous'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        return cm, report, plt.gcf()
    
    def visualize_predictions(self, test_images: List[str], test_labels: List[int], n_samples: int = 9):
        """Visualize random predictions from test set"""
        if self.classification_model is None:
            self.logger.error("No trained model available")
            return
            
        # Random sample indices
        indices = np.random.choice(len(test_images), n_samples, replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        self.classification_model.eval()
        
        for idx, ax in enumerate(axes):
            img_idx = indices[idx]
            img_path = test_images[img_idx]
            true_label = test_labels[img_idx]
            
            # Load and preprocess image
            if img_path.endswith('.json'):
                image = self._load_from_json(img_path)
            else:
                image = cv2.imread(img_path)
                
            if image is None:
                continue
                
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # Display image
            display_img = cv2.resize(image_rgb, (128, 128))
            ax.imshow(display_img)
            
            # Make prediction
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                logits = self.classification_model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
            
            # Set title with color based on correctness
            pred_name = 'Anomalous' if pred_class == 1 else 'Normal'
            true_name = 'Anomalous' if true_label == 1 else 'Normal'
            
            color = 'green' if pred_class == true_label else 'red'
            ax.set_title(f'Pred: {pred_name} ({confidence:.2%})\nTrue: {true_name}', 
                        color=color, fontsize=10)
            ax.axis('off')
        
        plt.suptitle('Model Predictions on Test Samples', fontsize=16)
        plt.tight_layout()
        
        return plt.gcf()
    
    def _load_from_json(self, json_path):
        """Load image from JSON format"""
        try:
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
            
        except Exception as e:
            logging.error(f"Error loading JSON {json_path}: {e}")
            return None


def compare_traditional_vs_dl(traditional_results: Dict, dl_results: Dict):
    """Compare traditional CV methods with deep learning results"""
    comparison = {
        'Traditional CV': {
            'Method': 'Statistical + Morphological',
            'Defects Found': len(traditional_results.get('defects', [])),
            'Processing Time': traditional_results.get('processing_time', 'N/A'),
            'Requires Reference': True
        },
        'Deep Learning': {
            'Method': 'CNN Classification',
            'Confidence': dl_results.get('classification', {}).get('confidence', 0),
            'Class': dl_results.get('classification', {}).get('class_name', 'Unknown'),
            'Processing Time': dl_results.get('processing_time', 'N/A'),
            'Requires Reference': False
        }
    }
    
    # Create comparison dataframe
    df = pd.DataFrame(comparison).T
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Method comparison
    methods = list(comparison.keys())
    colors = ['blue', 'orange']
    
    # If we have defect counts
    if 'Defects Found' in df.columns and 'Confidence' in df.columns:
        ax1.bar(methods[0], df.loc['Traditional CV', 'Defects Found'], color=colors[0], label='Defects Found')
        ax1.bar(methods[1], df.loc['Deep Learning', 'Confidence'] * 100, color=colors[1], label='Confidence %')
        ax1.set_ylabel('Count / Percentage')
        ax1.set_title('Detection Results Comparison')
        ax1.legend()
    
    # Create summary text
    ax2.axis('off')
    summary_text = f"""
    Traditional CV Analysis:
    - Method: {df.loc['Traditional CV', 'Method']}
    - Defects Found: {df.loc['Traditional CV', 'Defects Found']}
    - Requires Reference: {df.loc['Traditional CV', 'Requires Reference']}
    
    Deep Learning Analysis:
    - Method: {df.loc['Deep Learning', 'Method']}
    - Prediction: {df.loc['Deep Learning', 'Class']}
    - Confidence: {df.loc['Deep Learning', 'Confidence']:.2%}
    - Requires Reference: {df.loc['Deep Learning', 'Requires Reference']}
    """
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Traditional CV vs Deep Learning Comparison', fontsize=16)
    plt.tight_layout()
    
    return df, plt.gcf()


# Keep the original OmniFiberAnalyzer class but add a method to integrate with DL
class OmniFiberAnalyzer:
    """Original analyzer with added DL integration"""
    
    def __init__(self, config: OmniConfig):
        # ... (keep all original initialization code)
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'timestamp': None
        }
        self.current_metadata = None
        self.logger = logging.getLogger(__name__)
        self.load_knowledge_base()
        
        # Add DL analyzer
        self.dl_analyzer = EnhancedOmniFiberAnalyzer(config)
    
    # ... (keep all original methods)
    
    def analyze_end_face_hybrid(self, image_path: str, output_dir: str):
        """Hybrid analysis using both traditional and DL methods"""
        self.logger.info(f"Performing hybrid analysis on: {image_path}")
        
        # Traditional analysis
        start_time = time.time()
        traditional_results = self.analyze_end_face(image_path, output_dir)
        traditional_results['processing_time'] = time.time() - start_time
        
        # Deep learning analysis
        start_time = time.time()
        dl_results = self.dl_analyzer.analyze_end_face_dl(image_path, output_dir)
        if dl_results:
            dl_results['processing_time'] = time.time() - start_time
        
        # Compare results
        if traditional_results and dl_results:
            comparison_df, comparison_fig = compare_traditional_vs_dl(traditional_results, dl_results)
            
            # Save comparison
            output_path = Path(output_dir)
            comparison_fig.savefig(output_path / f"{Path(image_path).stem}_comparison.png")
            comparison_df.to_csv(output_path / f"{Path(image_path).stem}_comparison.csv")
            
            plt.close(comparison_fig)
        
        return {
            'traditional': traditional_results,
            'deep_learning': dl_results,
            'hybrid_decision': self._make_hybrid_decision(traditional_results, dl_results)
        }
    
    def _make_hybrid_decision(self, traditional_results: Dict, dl_results: Dict) -> Dict:
        """Make a hybrid decision based on both analyses"""
        decision = {
            'is_anomalous': False,
            'confidence': 0.0,
            'reasoning': []
        }
        
        # Traditional analysis vote
        if traditional_results and traditional_results.get('defects', []):
            trad_vote = len(traditional_results['defects']) > 0
            trad_confidence = min(len(traditional_results['defects']) / 10.0, 1.0)
            decision['reasoning'].append(f"Traditional: {len(traditional_results['defects'])} defects found")
        else:
            trad_vote = False
            trad_confidence = 0.0
        
        # DL analysis vote
        if dl_results and 'classification' in dl_results:
            dl_vote = dl_results['classification']['class'] == 1
            dl_confidence = dl_results['classification']['confidence']
            decision['reasoning'].append(f"DL: {dl_results['classification']['class_name']} ({dl_confidence:.2%})")
        else:
            dl_vote = False
            dl_confidence = 0.0
        
        # Weighted decision
        if trad_vote and dl_vote:
            decision['is_anomalous'] = True
            decision['confidence'] = (trad_confidence + dl_confidence) / 2
            decision['reasoning'].append("Both methods agree: ANOMALOUS")
        elif trad_vote or dl_vote:
            decision['is_anomalous'] = True
            decision['confidence'] = max(trad_confidence, dl_confidence) * 0.7  # Lower confidence when disagree
            decision['reasoning'].append("Methods disagree: LIKELY ANOMALOUS")
        else:
            decision['is_anomalous'] = False
            decision['confidence'] = 1.0 - max(trad_confidence, dl_confidence)
            decision['reasoning'].append("Both methods agree: NORMAL")
        
        return decision
    
    # ... (keep all other original methods like load_knowledge_base, save_knowledge_base, etc.)
    def load_knowledge_base(self):
        """Load previously saved knowledge base from JSON."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    loaded_data = json.load(f)
                
                if loaded_data.get('archetype_image'):
                    loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
                
                if loaded_data.get('statistical_model'):
                    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                        if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                            loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)
                
                self.reference_model = loaded_data
                self.logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            except Exception as e:
                self.logger.warning(f"Could not load knowledge base: {e}")


def main():
    """Enhanced main function with DL capabilities"""
    print("\n" + "="*80)
    print("OMNIFIBER ANALYZER - ENHANCED WITH DEEP LEARNING (v2.0)".center(80))
    print("="*80)
    print("\nNow featuring CNN-based anomaly detection!")
    print("Choose an option:")
    print("1. Train a new CNN model")
    print("2. Analyze images with existing model")
    print("3. Traditional analysis only")
    print("4. Hybrid analysis (Traditional + DL)")
    print("5. Exit\n")
    
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    
    while True:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            # Training mode
            data_dir = input("Enter directory containing training images: ").strip()
            if not os.path.isdir(data_dir):
                print("Directory not found!")
                continue
                
            # Collect images and labels
            images = []
            labels = []
            
            print("Collecting images...")
            for class_idx, class_name in enumerate(['normal', 'anomalous']):
                class_dir = os.path.join(data_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.endswith(('.png', '.jpg', '.jpeg', '.json')):
                            images.append(os.path.join(class_dir, img_file))
                            labels.append(class_idx)
            
            if len(images) < 10:
                print("Not enough images for training (need at least 10)")
                continue
            
            # Split data
            from sklearn.model_selection import train_test_split
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            print(f"Training with {len(train_imgs)} images...")
            results = analyzer.dl_analyzer.train_classification_model(
                train_imgs, train_labels, val_imgs, val_labels
            )
            
            print("Training complete!")
            print(f"Final validation accuracy: {results['val_acc'][-1]:.2%}")
            
        elif choice == '2':
            # DL analysis mode
            if analyzer.dl_analyzer.classification_model is None:
                # Try to load existing model
                if os.path.exists(config.model_save_path):
                    print("Loading saved model...")
                    analyzer.dl_analyzer.classification_model = FiberCNN().to(device)
                    analyzer.dl_analyzer.classification_model.load_state_dict(
                        torch.load(config.model_save_path, map_location=device)
                    )
                else:
                    print("No trained model found! Train a model first.")
                    continue
            
            image_path = input("Enter image path to analyze: ").strip()
            if not os.path.isfile(image_path):
                print("File not found!")
                continue
                
            output_dir = f"dl_output_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            analyzer.dl_analyzer.analyze_end_face_dl(image_path, output_dir)
            print(f"Results saved to: {output_dir}/")
            
        elif choice == '3':
            # Traditional analysis
            image_path = input("Enter image path to analyze: ").strip()
            if not os.path.isfile(image_path):
                print("File not found!")
                continue
                
            output_dir = f"traditional_output_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            analyzer.analyze_end_face(image_path, output_dir)
            print(f"Results saved to: {output_dir}/")
            
        elif choice == '4':
            # Hybrid analysis
            image_path = input("Enter image path to analyze: ").strip()
            if not os.path.isfile(image_path):
                print("File not found!")
                continue
                
            # Check if DL model exists
            if analyzer.dl_analyzer.classification_model is None:
                if os.path.exists(config.model_save_path):
                    print("Loading saved model...")
                    analyzer.dl_analyzer.classification_model = FiberCNN().to(device)
                    analyzer.dl_analyzer.classification_model.load_state_dict(
                        torch.load(config.model_save_path, map_location=device)
                    )
                else:
                    print("No DL model found, using traditional analysis only...")
            
            output_dir = f"hybrid_output_{Path(image_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = analyzer.analyze_end_face_hybrid(image_path, output_dir)
            
            print(f"\nHybrid Analysis Results:")
            print(f"Decision: {'ANOMALOUS' if results['hybrid_decision']['is_anomalous'] else 'NORMAL'}")
            print(f"Confidence: {results['hybrid_decision']['confidence']:.2%}")
            print(f"Reasoning: {', '.join(results['hybrid_decision']['reasoning'])}")
            print(f"Results saved to: {output_dir}/")
            
        elif choice == '5':
            break
            
        else:
            print("Invalid choice!")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()