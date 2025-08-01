"""
Auto-generated PyTorch implementation based on data analysis
Generated on: 2025-07-21 13:02:24
Analysis completed after recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import json
import numpy as np

class AnalyzedDataset(Dataset):
    """Dataset class for the tensorized images"""
    
    def __init__(self, tensor_dir, transform=None):
        self.tensor_dir = Path(tensor_dir)
        self.tensor_files = list(self.tensor_dir.rglob('*_color.pt'))
        self.transform = transform
        
        # Load class mapping
        self.classes = ["19700101000222-scratch.jpg", "19700101000223-scratch.jpg", "19700101000237-scratch.jpg", "50", "50-cladding", "50_clean_20250705_0001.png", "50_clean_20250705_0003.jpg", "50_clean_20250705_0004.png", "50_clean_20250705_0005.png", "91", "91-cladding", "91-scratched", "cladding-batch-1", "cladding-batch-3", "cladding-batch-4", "cladding-batch-5", "cladding-features-batch-1", "core-batch-1", "core-batch-2", "core-batch-3", "core-batch-4", "core-batch-5", "core-batch-6", "core-batch-7", "core-batch-8", "dirty-image", "fc-50-clean-full-1.png", "fc-50-clean-full-2.png", "fc-50-clean-full-3.jpg", "fc-50-clean-full.jpg", "fc-50-clean-full.png", "ferrule-batch-1", "ferrule-batch-2", "ferrule-batch-3", "ferrule-batch-4", "large-core-batch", "scratch-library-bmp", "sma", "sma-clean", "visualizations"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        # Load tensor and metadata
        data = torch.load(self.tensor_files[idx], weights_only=False)
        tensor = data['tensor']
        metadata = data['metadata']
        
        # Get label
        class_name = metadata['class_label']
        label = self.class_to_idx.get(class_name, 0)  # Default to 0 if not found
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label

class CustomCNN(nn.Module):
    """Recommended CNN architecture based on data analysis"""
    
    def __init__(self, num_classes=40):
        super(CustomCNN, self).__init__()
        
        # Architecture based on dataset size
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training configuration based on analysis
config = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 4,
    'use_class_weights': True,
}

# Data augmentation based on recommendations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example training loop
def train_model():
    # Load dataset
    dataset = AnalyzedDataset('C:\Users\Saem1001\Documents\GitHub\polar-bear\reference\tesnsorized-data', transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Apply validation transform to val dataset
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    # Initialize model
    model = CustomCNN().to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_acc = 0
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / val_size
        print(f'Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Accuracy: {acc:.2f}%')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    train_model()
