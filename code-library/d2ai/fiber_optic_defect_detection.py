
"""
Fiber Optic Endface Defect Detection Neural Network
Based on (https://d2l.ai/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time

# Based on D2L Chapter 7.2 - Convolutions for Images
class FiberOpticCNN(nn.Module):
    """
    CNN for fiber optic defect detection
    Based on D2L examples from https://d2l.ai/chapter_convolutional-neural-networks/
    """
    def __init__(self, num_classes=4):  # core, cladding, ferrule, defect
        super(FiberOpticCNN, self).__init__()

        # Feature extraction layers - based on D2L CNN architecture
        # Following D2L Chapter 7.2 convolution implementation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Pooling layers - D2L Chapter 7.5 Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Batch normalization - D2L Chapter 8.5 Batch Normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Region classification head
        self.region_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Defect detection head  
        self.defect_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # defect/no defect
        )

    def forward(self, x):
        # Forward pass following D2L CNN structure
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        features = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Multi-task outputs
        region_output = self.region_classifier(features)
        defect_output = self.defect_detector(features)

        return region_output, defect_output, features

# Based on D2L Chapter 14.1 - Image Augmentation
class FiberOpticDataset(Dataset):
    """
    Custom dataset for fiber optic images
    Based on D2L data loading examples
    """
    def __init__(self, data_dir, reference_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.reference_dir = Path(reference_dir)
        self.transform = transform
        self.mode = mode

        # Load image paths from chunk directories
        self.image_paths = []
        for chunk_dir in self.data_dir.glob('chunk_*'):
            chunk_images = list(chunk_dir.glob('*.jpg')) + list(chunk_dir.glob('*.png'))
            self.image_paths.extend(chunk_images)

        # Load reference tensors
        self.reference_tensors = {}
        for ref_dir in self.reference_dir.iterdir():
            if ref_dir.is_dir():
                ref_files = list(ref_dir.glob('*.pt'))
                self.reference_tensors[ref_dir.name] = ref_files

        print(f"Loaded {len(self.image_paths)} images from dataset")
        print(f"Loaded {len(self.reference_tensors)} reference categories")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (D2L style augmentation)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Create synthetic labels for demonstration
        # In practice, these would come from annotations
        region_label = self._extract_region_label(img_path)
        defect_label = self._detect_defects_opencv(image)

        return image, region_label, defect_label

    def _extract_region_label(self, img_path):
        """Extract region classification based on image analysis"""
        # Simplified region classification logic
        # 0: core, 1: cladding, 2: ferrule, 3: mixed
        return torch.randint(0, 4, (1,)).item()  # Placeholder

    def _detect_defects_opencv(self, image_tensor):
        """Use OpenCV-based defect detection as ground truth"""
        # Convert tensor back to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = image_tensor

        # OpenCV-based scratch and defect detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Edge detection for scratches
        edges = cv2.Canny(gray, 50, 150)

        # Blob detection for defects
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 1000
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        # Return binary classification: defect present or not
        has_defect = len(keypoints) > 0 or np.sum(edges) > 1000
        return torch.tensor([1 if has_defect else 0])

# Based on D2L training loops
class FiberOpticTrainer:
    """
    Training class based on D2L training patterns
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss functions
        self.region_criterion = nn.CrossEntropyLoss()
        self.defect_criterion = nn.BCEWithLogitsLoss()

        # Optimizer - following D2L examples
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Metrics tracking
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """Single training epoch following D2L patterns"""
        self.model.train()
        total_loss = 0

        for batch_idx, (images, region_labels, defect_labels) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.device)
            region_labels = region_labels.to(self.device)
            defect_labels = defect_labels.to(self.device).float()

            self.optimizer.zero_grad()

            # Forward pass
            region_out, defect_out, features = self.model(images)

            # Multi-task loss
            region_loss = self.region_criterion(region_out, region_labels)
            defect_loss = self.defect_criterion(defect_out.squeeze(), defect_labels.squeeze())

            total_loss_batch = region_loss + defect_loss
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validation following D2L patterns"""
        self.model.eval()
        correct_region = 0
        correct_defect = 0
        total = 0

        with torch.no_grad():
            for images, region_labels, defect_labels in self.val_loader:
                images = images.to(self.device)
                region_labels = region_labels.to(self.device)
                defect_labels = defect_labels.to(self.device)

                region_out, defect_out, _ = self.model(images)

                # Region accuracy
                _, predicted_region = torch.max(region_out.data, 1)
                correct_region += (predicted_region == region_labels).sum().item()

                # Defect accuracy
                predicted_defect = (torch.sigmoid(defect_out.squeeze()) > 0.5).float()
                correct_defect += (predicted_defect == defect_labels.squeeze()).sum().item()

                total += region_labels.size(0)

        region_acc = 100 * correct_region / total
        defect_acc = 100 * correct_defect / total

        return region_acc, defect_acc

    def train(self, num_epochs):
        """Main training loop based on D2L examples"""
        print("Starting training...")

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation
            region_acc, defect_acc = self.validate()
            self.val_accuracies.append((region_acc, defect_acc))

            # Learning rate scheduling
            self.scheduler.step()

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Region Accuracy: {region_acc:.2f}%')
            print(f'Defect Accuracy: {defect_acc:.2f}%')
            print('-' * 50)

# Feature extraction utilities based on D2L
class FeatureExtractor:
    """
    Feature extraction class based on D2L feature extraction examples
    """
    def __init__(self, model, layer_name='conv4'):
        self.model = model
        self.layer_name = layer_name
        self.features = {}

        # Register hook for feature extraction
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for feature extraction"""
        def hook_fn(module, input, output):
            self.features[self.layer_name] = output.detach()

        # Get the layer by name
        layer = getattr(self.model, self.layer_name)
        layer.register_forward_hook(hook_fn)

    def extract_features(self, images):
        """Extract features from images"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(images)
            return self.features[self.layer_name]

# Statistics and analysis utilities
class FiberOpticAnalyzer:
    """
    Analysis class for generating statistics and reports
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def analyze_dataset(self, dataloader):
        """Analyze entire dataset and generate statistics"""
        self.model.eval()

        stats = {
            'total_images': 0,
            'region_predictions': {'core': 0, 'cladding': 0, 'ferrule': 0, 'mixed': 0},
            'defect_predictions': {'defect': 0, 'clean': 0},
            'confidence_scores': [],
            'processing_times': []
        }

        region_names = ['core', 'cladding', 'ferrule', 'mixed']

        with torch.no_grad():
            for images, _, _ in tqdm(dataloader, desc="Analyzing dataset"):
                start_time = time.time()

                images = images.to(self.device)
                region_out, defect_out, _ = self.model(images)

                # Process predictions
                region_probs = F.softmax(region_out, dim=1)
                defect_probs = torch.sigmoid(defect_out)

                batch_size = images.size(0)
                stats['total_images'] += batch_size

                # Region statistics
                _, region_preds = torch.max(region_out, 1)
                for pred in region_preds:
                    stats['region_predictions'][region_names[pred.item()]] += 1

                # Defect statistics
                defect_preds = (defect_probs.squeeze() > 0.5).float()
                for pred in defect_preds:
                    if pred.item() == 1:
                        stats['defect_predictions']['defect'] += 1
                    else:
                        stats['defect_predictions']['clean'] += 1

                # Confidence scores
                max_region_conf = torch.max(region_probs, dim=1)[0]
                stats['confidence_scores'].extend(max_region_conf.cpu().numpy().tolist())

                # Processing time
                end_time = time.time()
                stats['processing_times'].append(end_time - start_time)

        # Calculate averages
        stats['avg_confidence'] = np.mean(stats['confidence_scores'])
        stats['avg_processing_time'] = np.mean(stats['processing_times'])

        return stats

    def generate_report(self, stats, output_file='defect_analysis_report.json'):
        """Generate comprehensive analysis report"""
        report = {
            'analysis_summary': {
                'total_images_processed': stats['total_images'],
                'average_confidence_score': f"{stats['avg_confidence']:.3f}",
                'average_processing_time_per_batch': f"{stats['avg_processing_time']:.3f}s",
            },
            'region_distribution': stats['region_predictions'],
            'defect_distribution': stats['defect_predictions'],
            'performance_metrics': {
                'images_per_second': f"{stats['total_images'] / sum(stats['processing_times']):.2f}",
                'confidence_std': f"{np.std(stats['confidence_scores']):.3f}"
            }
        }

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Analysis report saved to {output_file}")
        return report

def main():
    """
    Main function to run the fiber optic defect detection system
    Based on D2L main execution patterns
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations based on D2L Chapter 14.1
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check if dataset directory exists
    dataset_dir = Path('dataset')
    reference_dir = Path('reference')

    if not dataset_dir.exists():
        print("Creating sample dataset directory structure...")
        dataset_dir.mkdir(exist_ok=True)
        for i in range(1, 6):  # Create sample chunk directories
            chunk_dir = dataset_dir / f'chunk_{i}'
            chunk_dir.mkdir(exist_ok=True)
            print(f"Created {chunk_dir}")

    if not reference_dir.exists():
        print("Creating sample reference directory structure...")
        reference_dir.mkdir(exist_ok=True)
        for category in ['core_ref', 'cladding_ref', 'ferrule_ref', 'defect_ref']:
            cat_dir = reference_dir / category
            cat_dir.mkdir(exist_ok=True)
            print(f"Created {cat_dir}")

    # Create datasets
    try:
        full_dataset = FiberOpticDataset(dataset_dir, reference_dir, transform=transform)

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        if len(full_dataset) > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

            # Initialize model
            model = FiberOpticCNN(num_classes=4)

            # Training
            trainer = FiberOpticTrainer(model, train_loader, val_loader, device)
            trainer.train(num_epochs=5)  # Reduced for demonstration

            # Analysis
            analyzer = FiberOpticAnalyzer(model, device)
            stats = analyzer.analyze_dataset(val_loader)
            report = analyzer.generate_report(stats)

            print("\nFinal Analysis Report:")
            print(json.dumps(report, indent=2))

        else:
            print("No images found in dataset. Please add images to the dataset/chunk_* directories.")

    except Exception as e:
        print(f"Error during execution: {e}")
        print("This is a demonstration script. Please ensure proper dataset structure:")
        print("project-directory/")
        print("├── dataset/")
        print("│   ├── chunk_1/")
        print("│   ├── chunk_2/")
        print("│   └── ... (up to chunk_135)")
        print("└── reference/")
        print("    ├── reference_category_1/")
        print("    ├── reference_category_2/")
        print("    └── ... (40 subfolders with .pt files)")

if __name__ == "__main__":
    main()
