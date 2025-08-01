
"""
Fiber Optic Endface Defect Detection Neural Network
Based on examples from https://d2l.ai/
Emmanuel Sampson
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
    https://d2l.ai/chapter_convolutional-neural-networks/
    """
    def __init__(self, num_classes=4):  # core, cladding, ferrule, defect
        super(FiberOpticCNN, self).__init__()  # Initialize parent class for neural network

        # Feature extraction layers - based on D2L CNN architecture
        # Following D2L Chapter 7.2 convolution implementation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # First conv layer: 3 input channels (RGB) to 64 feature maps
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Second conv layer: 64 to 128 feature maps
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Third conv layer: 128 to 256 feature maps
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Fourth conv layer: 256 to 512 feature maps

        # Pooling layers - D2L Chapter 7.5 Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 window to reduce spatial dimensions

        # Batch normalization - D2L Chapter 8.5 Batch Normalization
        self.bn1 = nn.BatchNorm2d(64)  # Normalize activations after first conv layer
        self.bn2 = nn.BatchNorm2d(128)  # Normalize activations after second conv layer
        self.bn3 = nn.BatchNorm2d(256)  # Normalize activations after third conv layer
        self.bn4 = nn.BatchNorm2d(512)  # Normalize activations after fourth conv layer

        # Region classification head
        self.region_classifier = nn.Sequential(  # Sequential container for region classification layers
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling to get 7x7 feature maps regardless of input size
            nn.Flatten(),  # Flatten 7x7x512 to 1D vector
            nn.Linear(512 * 7 * 7, 1024),  # Fully connected layer: 25088 to 1024 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.5),  # Dropout for regularization (50% of neurons randomly set to 0)
            nn.Linear(1024, num_classes)  # Output layer: 1024 to 4 classes (core, cladding, ferrule, defect)
        )

        # Defect detection head  
        self.defect_detector = nn.Sequential(  # Sequential container for defect detection layers
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling to get 7x7 feature maps
            nn.Flatten(),  # Flatten 7x7x512 to 1D vector
            nn.Linear(512 * 7 * 7, 512),  # Fully connected layer: 25088 to 512 neurons
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, 1)  # Output layer: 512 to 1 (defect/no defect binary classification)
        )

    def forward(self, x):
        # Forward pass following D2L CNN structure
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BatchNorm -> ReLU -> MaxPool
        features = self.pool(F.relu(self.bn4(self.conv4(x))))  # Conv4 -> BatchNorm -> ReLU -> MaxPool, save features

        # Multi-task outputs
        region_output = self.region_classifier(features)  # Pass features through region classification head
        defect_output = self.defect_detector(features)  # Pass features through defect detection head

        return region_output, defect_output, features  # Return both outputs and intermediate features

# Based on D2L Chapter 14.1 - Image Augmentation
class FiberOpticDataset(Dataset):
    def __init__(self, data_dir, reference_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)  # Convert to Path object for easier file operations
        self.reference_dir = Path(reference_dir)  # Convert to Path object for reference directory
        self.transform = transform  # Store image transformation pipeline
        self.mode = mode  # Store mode (train/val/test)

        # Load image paths from chunk directories
        self.image_paths = []  # Initialize empty list to store image file paths
        for chunk_dir in self.data_dir.glob('chunk_*'):  # Find all directories starting with 'chunk_'
            chunk_images = list(chunk_dir.glob('*.jpg')) + list(chunk_dir.glob('*.png'))  # Get all jpg and png files
            self.image_paths.extend(chunk_images)  # Add found images to the main list

        # Load reference tensors
        self.reference_tensors = {}  # Initialize dictionary to store reference tensors
        for ref_dir in self.reference_dir.iterdir():  # Iterate through reference directory contents
            if ref_dir.is_dir():  # Check if it's a directory
                ref_files = list(ref_dir.glob('*.pt'))  # Get all .pt (PyTorch tensor) files
                self.reference_tensors[ref_dir.name] = ref_files  # Store files with directory name as key

        print(f"Loaded {len(self.image_paths)} images from dataset")  # Print number of loaded images
        print(f"Loaded {len(self.reference_tensors)} reference categories")  # Print number of reference categories

    def __len__(self):
        return len(self.image_paths)  # Return total number of images in dataset

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]  # Get image path at given index
        image = cv2.imread(str(img_path))  # Read image using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB color space

        # Apply transforms (D2L style augmentation)
        if self.transform:  # Check if transformation pipeline is provided
            image = self.transform(image)  # Apply the transformation pipeline
        else:  # If no transform provided, use default tensor conversion
            image = transforms.ToTensor()(image)  # Convert numpy array to PyTorch tensor

        # Create synthetic labels for demonstration
        # In practice, these would come from annotations
        region_label = self._extract_region_label(img_path)  # Generate region classification label
        defect_label = self._detect_defects_opencv(image)  # Generate defect detection label

        return image, region_label, defect_label  # Return image and both labels

    def _extract_region_label(self, img_path):
        """Extract region classification based on image analysis"""
        # Simplified region classification logic
        # 0: core, 1: cladding, 2: ferrule, 3: mixed
        return torch.randint(0, 4, (1,)).item()  # Return random integer 0-3 as placeholder label

    def _detect_defects_opencv(self, image_tensor):
        """Use OpenCV-based defect detection as ground truth"""
        # Convert tensor back to numpy for OpenCV processing
        if isinstance(image_tensor, torch.Tensor):  # Check if input is PyTorch tensor
            img_np = image_tensor.permute(1, 2, 0).numpy()  # Rearrange dimensions from (C,H,W) to (H,W,C)
            img_np = (img_np * 255).astype(np.uint8)  # Scale from [0,1] to [0,255] and convert to uint8
        else:  # If already numpy array
            img_np = image_tensor  # Use as is

        # OpenCV-based scratch and defect detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale for processing

        # Edge detection for scratches
        edges = cv2.Canny(gray, 50, 150)  # Apply Canny edge detection with thresholds 50 and 150

        # Blob detection for defects
        params = cv2.SimpleBlobDetector_Params()  # Create blob detector parameters
        params.filterByArea = True  # Enable area filtering
        params.minArea = 10  # Minimum blob area
        params.maxArea = 1000  # Maximum blob area
        detector = cv2.SimpleBlobDetector_create(params)  # Create blob detector with parameters
        keypoints = detector.detect(gray)  # Detect blobs in grayscale image

        # Return binary classification: defect present or not
        has_defect = len(keypoints) > 0 or np.sum(edges) > 1000  # Check if blobs found or high edge density
        return torch.tensor([1 if has_defect else 0])  # Return tensor with 1 for defect, 0 for clean

# Based on D2L training loops
class FiberOpticTrainer:
    """
    Training class based on D2L training patterns
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)  # Move model to specified device (CPU/GPU)
        self.train_loader = train_loader  # Store training data loader
        self.val_loader = val_loader  # Store validation data loader
        self.device = device  # Store device for later use

        # Loss functions
        self.region_criterion = nn.CrossEntropyLoss()  # Loss function for region classification
        self.defect_criterion = nn.BCEWithLogitsLoss()  # Loss function for binary defect detection

        # Optimizer - following D2L examples
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler

        # Metrics tracking
        self.train_losses = []  # List to store training losses
        self.val_accuracies = []  # List to store validation accuracies

    def train_epoch(self):
        """Single training epoch following D2L patterns"""
        self.model.train()  # Set model to training mode (enables dropout, batch norm updates)
        total_loss = 0  # Initialize total loss accumulator

        for batch_idx, (images, region_labels, defect_labels) in enumerate(tqdm(self.train_loader)):  # Iterate through training batches
            images = images.to(self.device)  # Move images to device
            region_labels = region_labels.to(self.device)  # Move region labels to device
            defect_labels = defect_labels.to(self.device).float()  # Move defect labels to device and convert to float

            self.optimizer.zero_grad()  # Clear gradients from previous iteration

            # Forward pass
            region_out, defect_out, features = self.model(images)  # Forward pass through model

            # Multi-task loss
            region_loss = self.region_criterion(region_out, region_labels)  # Calculate region classification loss
            defect_loss = self.defect_criterion(defect_out.squeeze(), defect_labels.squeeze())  # Calculate defect detection loss

            total_loss_batch = region_loss + defect_loss  # Combine both losses
            total_loss_batch.backward()  # Backward pass to compute gradients
            self.optimizer.step()  # Update model parameters using gradients

            total_loss += total_loss_batch.item()  # Add batch loss to total

        return total_loss / len(self.train_loader)  # Return average loss per batch

    def validate(self):
        """Validation following D2L patterns"""
        self.model.eval()  # Set model to evaluation mode (disables dropout, batch norm uses running stats)
        correct_region = 0  # Initialize correct region predictions counter
        correct_defect = 0  # Initialize correct defect predictions counter
        total = 0  # Initialize total samples counter

        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, region_labels, defect_labels in self.val_loader:  # Iterate through validation batches
                images = images.to(self.device)  # Move images to device
                region_labels = region_labels.to(self.device)  # Move region labels to device
                defect_labels = defect_labels.to(self.device)  # Move defect labels to device

                region_out, defect_out, _ = self.model(images)  # Forward pass (no gradients needed)

                # Region accuracy
                _, predicted_region = torch.max(region_out.data, 1)  # Get predicted class indices
                correct_region += (predicted_region == region_labels).sum().item()  # Count correct predictions

                # Defect accuracy
                predicted_defect = (torch.sigmoid(defect_out.squeeze()) > 0.5).float()  # Apply sigmoid and threshold
                correct_defect += (predicted_defect == defect_labels.squeeze()).sum().item()  # Count correct predictions

                total += region_labels.size(0)  # Add batch size to total

        region_acc = 100 * correct_region / total  # Calculate region accuracy percentage
        defect_acc = 100 * correct_defect / total  # Calculate defect accuracy percentage

        return region_acc, defect_acc  # Return both accuracies

    def train(self, num_epochs):
        """Main training loop """
        print("Starting training...")  # Print training start message

        for epoch in range(num_epochs):  # Iterate through specified number of epochs
            # Training
            train_loss = self.train_epoch()  # Train for one epoch
            self.train_losses.append(train_loss)  # Store training loss

            # Validation
            region_acc, defect_acc = self.validate()  # Validate model performance
            self.val_accuracies.append((region_acc, defect_acc))  # Store validation accuracies

            # Learning rate scheduling
            self.scheduler.step()  # Update learning rate according to scheduler

            print(f'Epoch [{epoch+1}/{num_epochs}]')  # Print current epoch
            print(f'Train Loss: {train_loss:.4f}')  # Print training loss
            print(f'Region Accuracy: {region_acc:.2f}%')  # Print region accuracy
            print(f'Defect Accuracy: {defect_acc:.2f}%')  # Print defect accuracy
            print('-' * 50)  # Print separator line

# Feature extraction utilities based on D2L
class FeatureExtractor:
    """
    Feature extraction class 
    """
    def __init__(self, model, layer_name='conv4'):
        self.model = model  # Store the model
        self.layer_name = layer_name  # Store the layer name to extract features from
        self.features = {}  # Initialize dictionary to store extracted features

        # Register hook for feature extraction
        self._register_hooks()  # Set up hooks to capture intermediate features

    def _register_hooks(self):
        """Register forward hooks for feature extraction"""
        def hook_fn(module, input, output):  # Define hook function
            self.features[self.layer_name] = output.detach()  # Store output tensor (detached from computation graph)

        # Get the layer by name
        layer = getattr(self.model, self.layer_name)  # Get the specified layer from model
        layer.register_forward_hook(hook_fn)  # Register the hook to capture output

    def extract_features(self, images):
        """Extract features from images"""
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            _ = self.model(images)  # Forward pass (output ignored, hook captures features)
            return self.features[self.layer_name]  # Return captured features

# Statistics and analysis utilities
class FiberOpticAnalyzer:
    """
    Analysis class for generating statistics and reports
    """
    def __init__(self, model, device):
        self.model = model  # Store the model
        self.device = device  # Store the device

    def analyze_dataset(self, dataloader):
        """Analyze entire dataset and generate statistics"""
        self.model.eval()  # Set model to evaluation mode

        stats = {  # Initialize statistics dictionary
            'total_images': 0,  # Counter for total images processed
            'region_predictions': {'core': 0, 'cladding': 0, 'ferrule': 0, 'mixed': 0},  # Region prediction counters
            'defect_predictions': {'defect': 0, 'clean': 0},  # Defect prediction counters
            'confidence_scores': [],  # List to store confidence scores
            'processing_times': []  # List to store processing times
        }

        region_names = ['core', 'cladding', 'ferrule', 'mixed']  # List of region class names

        with torch.no_grad():  # Disable gradient computation
            for images, _, _ in tqdm(dataloader, desc="Analyzing dataset"):  # Iterate through dataset
                start_time = time.time()  # Record start time

                images = images.to(self.device)  # Move images to device
                region_out, defect_out, _ = self.model(images)  # Forward pass

                # Process predictions
                region_probs = F.softmax(region_out, dim=1)  # Apply softmax to get probabilities
                defect_probs = torch.sigmoid(defect_out)  # Apply sigmoid to get defect probabilities

                batch_size = images.size(0)  # Get batch size
                stats['total_images'] += batch_size  # Add batch size to total

                # Region statistics
                _, region_preds = torch.max(region_out, 1)  # Get predicted region classes
                for pred in region_preds:  # Iterate through predictions
                    stats['region_predictions'][region_names[pred.item()]] += 1  # Increment counter for predicted class

                # Defect statistics
                defect_preds = (defect_probs.squeeze() > 0.5).float()  # Apply threshold to get binary predictions
                for pred in defect_preds:  # Iterate through predictions
                    if pred.item() == 1:  # If defect predicted
                        stats['defect_predictions']['defect'] += 1  # Increment defect counter
                    else:  # If no defect predicted
                        stats['defect_predictions']['clean'] += 1  # Increment clean counter

                # Confidence scores
                max_region_conf = torch.max(region_probs, dim=1)[0]  # Get maximum confidence for each prediction
                stats['confidence_scores'].extend(max_region_conf.cpu().numpy().tolist())  # Add to confidence list

                # Processing time
                end_time = time.time()  # Record end time
                stats['processing_times'].append(end_time - start_time)  # Calculate and store processing time

        # Calculate averages
        stats['avg_confidence'] = np.mean(stats['confidence_scores'])  # Calculate average confidence
        stats['avg_processing_time'] = np.mean(stats['processing_times'])  # Calculate average processing time

        return stats  # Return complete statistics

    def generate_report(self, stats, output_file='defect_analysis_report.json'):
        """Report"""
        report = {  # Create report dictionary
            'analysis_summary': {  # Summary section
                'total_images_processed': stats['total_images'],  # Total images processed
                'average_confidence_score': f"{stats['avg_confidence']:.3f}",  # Average confidence formatted
                'average_processing_time_per_batch': f"{stats['avg_processing_time']:.3f}s",  # Average time formatted
            },
            'region_distribution': stats['region_predictions'],  # Region prediction distribution
            'defect_distribution': stats['defect_predictions'],  # Defect prediction distribution
            'performance_metrics': {  # Performance metrics section
                'images_per_second': f"{stats['total_images'] / sum(stats['processing_times']):.2f}",  # Throughput
                'confidence_std': f"{np.std(stats['confidence_scores']):.3f}"  # Confidence standard deviation
            }
        }

        # Save report
        with open(output_file, 'w') as f:  # Open file for writing
            json.dump(report, f, indent=2)  # Write JSON report with indentation

        print(f"Analysis report saved to {output_file}")  # Print confirmation message
        return report  # Return the report

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    print(f"Using device: {device}")  # Print which device is being used

    # Data transformations based on D2L Chapter 14.1
    transform = transforms.Compose([  # Create transformation pipeline
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally with 50% probability
        transforms.RandomRotation(10),  # Randomly rotate image by ±10 degrees
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    # Check if dataset directory exists
    dataset_dir = Path('dataset')  # Create Path object for dataset directory
    reference_dir = Path('reference')  # Create Path object for reference directory

    if not dataset_dir.exists():  # Check if dataset directory doesn't exist
        print("Creating sample dataset directory structure...")  # Print creation message
        dataset_dir.mkdir(exist_ok=True)  # Create dataset directory
        for i in range(1, 6):  # Create sample chunk directories
            chunk_dir = dataset_dir / f'chunk_{i}'  # Create chunk directory path
            chunk_dir.mkdir(exist_ok=True)  # Create chunk directory
            print(f"Created {chunk_dir}")  # Print created directory

    if not reference_dir.exists():  # Check if reference directory doesn't exist
        print("Creating sample reference directory structure...")  # Print creation message
        reference_dir.mkdir(exist_ok=True)  # Create reference directory
        for category in ['core_ref', 'cladding_ref', 'ferrule_ref', 'defect_ref']:  # Create sample categories
            cat_dir = reference_dir / category  # Create category directory path
            cat_dir.mkdir(exist_ok=True)  # Create category directory
            print(f"Created {cat_dir}")  # Print created directory

    # Create datasets
    try:  # Try to create and use dataset
        full_dataset = FiberOpticDataset(dataset_dir, reference_dir, transform=transform)  # Create dataset object

        # Split dataset
        train_size = int(0.8 * len(full_dataset))  # Calculate 80% for training
        val_size = len(full_dataset) - train_size  # Calculate 20% for validation

        if len(full_dataset) > 0:  # Check if dataset has images
            train_dataset, val_dataset = torch.utils.data.random_split(  # Split dataset randomly
                full_dataset, [train_size, val_size]  # Split into train and validation sets
            )

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # Training data loader
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)  # Validation data loader

            # Initialize model
            model = FiberOpticCNN(num_classes=4)  # Create CNN model with 4 classes

            # Training
            trainer = FiberOpticTrainer(model, train_loader, val_loader, device)  # Create trainer object
            trainer.train(num_epochs=5)  # Train for 5 epochs

            # Analysis
            analyzer = FiberOpticAnalyzer(model, device)  # Create analyzer object
            stats = analyzer.analyze_dataset(val_loader)  # Analyze validation dataset
            report = analyzer.generate_report(stats)  # Generate analysis report

            print("\nFinal Analysis Report:")  # Print report header
            print(json.dumps(report, indent=2))  # Print formatted JSON report

        else:  # If no images found
            print("No images found in dataset. Please add images to the dataset/chunk_* directories.")  # Print error message

    except Exception as e:  # Catch any exceptions
        print(f"Error during execution: {e}")  # Print error message
        print(" Please ensure proper dataset structure:")  # Print structure requirements
        print("project-directory/")  # Show expected directory structure
        print("├── dataset/")
        print("│   ├── chunk_1/")
        print("│   ├── chunk_2/")
        print("│   └── ... (up to chunk_135)")
        print("└── reference/")
        print("    ├── reference_category_1/")
        print("    ├── reference_category_2/")
        print("    └── ... (40 subfolders with .pt files)")

if __name__ == "__main__":
    main()  # Run main function when script is executed directly
