#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Network Image Classification 
Usage:
python nn-image-matcher-fixed.py --model ResNet --dataset FashionMNIST --lr 0.01 --num_epochs 5
"""

import argparse
import os
import sys
import time
import glob
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Simple fallback implementations if d2l is not available
class SimpleTrainer:
    def __init__(self, max_epochs=10, device=None):
        self.max_epochs = max_epochs  # Store maximum number of training iterations
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose GPU if available, otherwise CPU
    
    def fit(self, model, data_module):
        model = model.to(self.device)  # Move neural network to selected compute device (GPU/CPU)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initialize Adam optimizer for gradient-based weight updates
        criterion = nn.CrossEntropyLoss()  # Define loss function for multi-class classification
        
        train_loader = data_module.get_dataloader(train=True)  # Get batched training data iterator
        val_loader = data_module.get_dataloader(train=False)  # Get batched validation data iterator
        
        for epoch in range(self.max_epochs):  # Iterate through each complete pass over the dataset
            # Training
            model.train()  # Set model to training mode (enables dropout, batch norm updates)
            train_loss = 0.0  # Initialize cumulative loss counter for this epoch
            train_correct = 0  # Initialize correct prediction counter for accuracy calculation
            train_total = 0  # Initialize total sample counter for accuracy calculation
            
            for batch_idx, (data, target) in enumerate(train_loader):  # Process each batch of training samples
                data, target = data.to(self.device), target.to(self.device)  # Move input tensors and labels to compute device
                
                optimizer.zero_grad()  # Clear gradients from previous batch to prevent accumulation
                output = model(data)  # Forward pass: compute predictions from input data
                loss = criterion(output, target)  # Calculate loss between predictions and true labels
                loss.backward()  # Backward pass: compute gradients of loss with respect to model parameters
                optimizer.step()  # Update model weights using computed gradients
                
                train_loss += loss.item()  # Accumulate scalar loss value for epoch statistics
                _, predicted = torch.max(output.data, 1)  # Get class with highest probability as prediction
                train_total += target.size(0)  # Count total number of samples processed
                train_correct += (predicted == target).sum().item()  # Count number of correct predictions
                
                if batch_idx % 100 == 0:  # Print progress every 100 batches to monitor training
                    print(f'Epoch {epoch+1}/{self.max_epochs}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
            val_loss = 0.0  # Initialize validation loss accumulator
            val_correct = 0  # Initialize validation accuracy counter
            val_total = 0  # Initialize validation sample counter
            
            with torch.no_grad():  # Disable gradient computation for memory efficiency during validation
                for data, target in val_loader:  # Process each validation batch
                    data, target = data.to(self.device), target.to(self.device)  # Move validation data to compute device
                    output = model(data)  # Forward pass: generate predictions without gradient tracking
                    loss = criterion(output, target)  # Calculate validation loss for monitoring
                    
                    val_loss += loss.item()  # Accumulate validation loss
                    _, predicted = torch.max(output.data, 1)  # Extract predicted class labels
                    val_total += target.size(0)  # Count total validation samples
                    val_correct += (predicted == target).sum().item()  # Count correct validation predictions
            
            train_acc = 100 * train_correct / train_total  # Calculate training accuracy percentage
            val_acc = 100 * val_correct / val_total  # Calculate validation accuracy percentage
            
            print(f'Epoch {epoch+1}/{self.max_epochs}: '  # Display epoch performance metrics
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

################################################################################
# SECTION 1: DATA MANAGEMENT MODULE
################################################################################

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from a directory."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Store the base directory path containing images
        self.transform = transform  # Store image preprocessing pipeline
        self.image_paths = []  # Initialize list to store all discovered image file paths
        self.labels = []  # Initialize list to store corresponding labels for each image
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']  # Define supported image formats
        for ext in image_extensions:  # Iterate through each supported image extension
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))  # Find images in root directory
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))  # Find images in subdirectories recursively
        
        # Create dummy labels (0 for all images in this simple case)
        self.labels = [0] * len(self.image_paths)  # Assign label 0 to all images (single class dataset)
        self.num_classes = 1  # Set number of classes to 1 for this simple implementation
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")  # Report total number of discovered images
    
    def __len__(self):
        return len(self.image_paths)  # Return total number of images in dataset for PyTorch DataLoader
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get file path for requested image index
        try:
            image = Image.open(img_path).convert('RGB')  # Load image file and ensure 3-channel RGB format
            label = self.labels[idx]  # Get corresponding label for this image
            
            if self.transform:  # Check if image preprocessing transforms are provided
                image = self.transform(image)  # Apply preprocessing pipeline (resize, normalize, etc.)
            
            return image, label  # Return preprocessed image tensor and its label
        except Exception as e:  # Handle corrupted or unreadable image files
            print(f"Error loading image {img_path}: {e}")  # Log error message with problematic file path
            # Return a dummy tensor if image loading fails
            dummy_image = torch.zeros(3, 224, 224)  # Create blank RGB image tensor as fallback
            return dummy_image, 0  # Return dummy data to prevent dataset iteration failure

class DataModule:
    """Data module for handling different datasets."""
    
    def __init__(self, dataset_name='FashionMNIST', batch_size=32, resize=(224, 224), custom_path=None):
        self.dataset_name = dataset_name.lower()  # Convert dataset name to lowercase for consistent comparison
        self.batch_size = batch_size  # Store number of samples per training batch
        self.resize = resize  # Store target image dimensions for resizing
        self.custom_path = custom_path  # Store path to custom image directory if provided
        self.root = './data'  # Set default directory for downloading standard datasets
        self.num_workers = 0  # Set to 0 for Windows compatibility (multiprocessing issues)
        
        # Setup dataset-specific parameters
        if self.dataset_name == 'cifar10':  # Configure parameters for CIFAR-10 dataset
            self.normalize = transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CIFAR-10 channel-wise mean and std normalization
            self.num_classes = 10  # CIFAR-10 has 10 object classes
            self.in_channels = 3  # CIFAR-10 images have 3 color channels (RGB)
        elif self.dataset_name == 'custom' and custom_path:  # Configure parameters for custom image dataset
            self.normalize = transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics for transfer learning compatibility
            self.in_channels = 3  # Assume RGB images for custom datasets
            self.num_classes = 10  # Default value, will be updated when dataset is actually loaded
        else:  # Default configuration for FashionMNIST dataset
            self.normalize = transforms.Normalize([0.1307], [0.3081])  # FashionMNIST grayscale normalization parameters
            self.num_classes = 10  # FashionMNIST has 10 clothing categories
            self.in_channels = 1  # FashionMNIST images are grayscale (single channel)
        
        # Define transformations
        self.train_transform = transforms.Compose([  # Create training data preprocessing pipeline
            transforms.Resize(resize),  # Resize images to target dimensions for model compatibility
            transforms.RandomHorizontalFlip(0.5),  # Randomly flip images horizontally for data augmentation
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor and scale to [0,1]
            self.normalize  # Apply dataset-specific normalization to match training statistics
        ])
        
        self.val_transform = transforms.Compose([  # Create validation data preprocessing pipeline
            transforms.Resize(resize),  # Resize validation images to same dimensions as training
            transforms.ToTensor(),  # Convert PIL image to tensor format
            self.normalize  # Apply same normalization as training data
        ])
    
    def get_dataloader(self, train=True):
        """Create and return a data loader."""
        transform = self.train_transform if train else self.val_transform  # Select appropriate preprocessing based on train/validation mode
        
        if self.dataset_name == 'cifar10':  # Create CIFAR-10 dataset instance
            dataset = torchvision.datasets.CIFAR10(
                root=self.root, train=train, transform=transform, download=True)  # Download CIFAR-10 if not present, apply transforms
        elif self.dataset_name == 'fashionmnist':  # Create FashionMNIST dataset instance
            dataset = torchvision.datasets.FashionMNIST(
                root=self.root, train=train, transform=transform, download=True)  # Download FashionMNIST if not present, apply transforms
        elif self.dataset_name == 'custom' and self.custom_path:  # Create custom dataset from user-provided directory
            dataset = CustomImageDataset(self.custom_path, transform=transform)  # Load images from custom directory with transforms
            self.num_classes = getattr(dataset, 'num_classes', 1)  # Update number of classes based on custom dataset
            # For custom dataset, use same data for train and val
        else:  # Handle unsupported dataset names
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")  # Raise error for invalid dataset specification
        
        return DataLoader(  # Create PyTorch DataLoader for efficient batch processing
            dataset, self.batch_size, shuffle=train,  # Shuffle training data, keep validation data in order
            num_workers=self.num_workers)  # Use specified number of worker processes for data loading

################################################################################
# SECTION 2: CNN MODEL ZOO
################################################################################

class ImageClassifier(nn.Module):
    """Base class for image classification models."""
    
    def __init__(self):
        super().__init__()  # Initialize parent PyTorch Module class
    
    def forward(self, x):
        return self.net(x)  # Pass input through the neural network and return predictions

class LeNet(ImageClassifier):
    """LeNet-5 architecture for image classification."""
    
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()  # Initialize parent ImageClassifier class
        self.net = nn.Sequential(  # Define sequential neural network layers
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),  # First convolutional layer: extract 6 feature maps using 5x5 filters
            nn.Sigmoid(),  # Apply sigmoid activation function for non-linearity
            nn.AvgPool2d(kernel_size=2, stride=2),  # Downsample feature maps by averaging 2x2 regions
            nn.Conv2d(6, 16, kernel_size=5),  # Second convolutional layer: combine features into 16 maps
            nn.Sigmoid(),  # Apply sigmoid activation after second convolution
            nn.AvgPool2d(kernel_size=2, stride=2),  # Second downsampling to reduce spatial dimensions
            nn.Flatten(),  # Convert 2D feature maps to 1D vector for dense layers
            nn.Linear(16 * 54 * 54, 120),  # First fully connected layer: map flattened features to 120 neurons
            nn.Sigmoid(),  # Apply sigmoid activation in dense layer
            nn.Linear(120, 84),  # Second fully connected layer: reduce to 84 neurons
            nn.Sigmoid(),  # Apply sigmoid activation before final layer
            nn.Linear(84, num_classes)  # Output layer: map to number of classes for classification
        )

class AlexNet(ImageClassifier):
    """AlexNet architecture for image classification."""
    
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()  # Initialize parent ImageClassifier class
        self.net = nn.Sequential(  # Define AlexNet architecture as sequential layers
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),  # First conv layer: 96 filters of size 11x11 with stride 4
            nn.ReLU(),  # ReLU activation for faster training than sigmoid
            nn.MaxPool2d(kernel_size=3, stride=2),  # Max pooling to retain strongest activations and reduce size
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # Second conv layer: 256 filters of size 5x5
            nn.ReLU(),  # ReLU activation after second convolution
            nn.MaxPool2d(kernel_size=3, stride=2),  # Second max pooling layer
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # Third conv layer: 384 filters of size 3x3
            nn.ReLU(),  # ReLU activation after third convolution
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # Fourth conv layer: maintain 384 feature maps
            nn.ReLU(),  # ReLU activation after fourth convolution
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # Fifth conv layer: reduce to 256 feature maps
            nn.ReLU(),  # ReLU activation after fifth convolution
            nn.MaxPool2d(kernel_size=3, stride=2),  # Final max pooling before dense layers
            nn.AdaptiveAvgPool2d((6, 6)),  # Adaptive pooling to ensure fixed 6x6 output size
            nn.Flatten(),  # Convert 2D feature maps to 1D vector
            nn.Dropout(0.5),  # Dropout for regularization to prevent overfitting
            nn.Linear(256 * 6 * 6, 4096),  # First dense layer: map to 4096 neurons
            nn.ReLU(),  # ReLU activation in dense layer
            nn.Dropout(0.5),  # Second dropout layer for additional regularization
            nn.Linear(4096, 4096),  # Second dense layer: maintain 4096 neurons
            nn.ReLU(),  # ReLU activation before output layer
            nn.Linear(4096, num_classes)  # Output layer: map to number of classes
        )

class VGG(ImageClassifier):
    """VGG architecture with configurable depth."""
    
    def __init__(self, conv_arch, num_classes=10, in_channels=3):
        super().__init__()  # Initialize parent ImageClassifier class
        
        def vgg_block(num_convs, in_channels, out_channels):  # Helper function to create VGG convolutional blocks
            layers = []  # Initialize list to store block layers
            for _ in range(num_convs):  # Create specified number of conv layers in this block
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # Add 3x3 conv layer with padding
                layers.append(nn.ReLU())  # Add ReLU activation after each convolution
                in_channels = out_channels  # Update input channels for next layer
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Add max pooling at end of block
            return nn.Sequential(*layers)  # Return block as sequential module
        
        conv_blks = []  # Initialize list to store all convolutional blocks
        current_in_channels = in_channels  # Track current number of input channels
        for (num_convs, out_channels) in conv_arch:  # Iterate through architecture specification
            conv_blks.append(vgg_block(num_convs, current_in_channels, out_channels))  # Create and add block
            current_in_channels = out_channels  # Update channel count for next block
        
        self.net = nn.Sequential(  # Combine all blocks and classifier into complete network
            *conv_blks,  # Unpack and add all convolutional blocks
            nn.AdaptiveAvgPool2d((7, 7)),  # Adaptive pooling to ensure 7x7 feature maps
            nn.Flatten(),  # Convert 2D feature maps to 1D vector
            nn.Linear(out_channels * 7 * 7, 4096),  # First classifier layer
            nn.ReLU(),  # ReLU activation in classifier
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(4096, 4096),  # Second classifier layer
            nn.ReLU(),  # ReLU activation before output
            nn.Dropout(0.5),  # Final dropout layer
            nn.Linear(4096, num_classes)  # Output layer for classification
        )

class ResNet(ImageClassifier):
    """Simple ResNet architecture."""
    
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()  # Initialize parent ImageClassifier class
        
        # Use a pre-trained ResNet18 and modify for our needs
        self.net = torchvision.models.resnet18(weights=None)  # Load ResNet18 architecture without pre-trained weights
        
        # Modify first layer if needed
        if in_channels != 3:  # Check if input has different number of channels than RGB
            self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Replace first conv layer for different input channels
        
        # Modify final layer
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)  # Replace final layer to match number of output classes

################################################################################
# SECTION 3: MODEL FACTORY
################################################################################

def get_model(model_name, num_classes, in_channels):
    """Factory function to create models."""
    if model_name.lower() == 'lenet':  # Check if LeNet architecture is requested
        return LeNet(num_classes=num_classes, in_channels=in_channels)  # Create and return LeNet model instance
    elif model_name.lower() == 'alexnet':  # Check if AlexNet architecture is requested
        return AlexNet(num_classes=num_classes, in_channels=in_channels)  # Create and return AlexNet model instance
    elif model_name.lower() == 'vgg':  # Check if VGG architecture is requested
        # VGG-11 architecture
        conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]  # Define VGG-11 block configuration
        return VGG(conv_arch, num_classes=num_classes, in_channels=in_channels)  # Create VGG model with specified architecture
    elif model_name.lower() == 'resnet':  # Check if ResNet architecture is requested
        return ResNet(num_classes=num_classes, in_channels=in_channels)  # Create and return ResNet model instance
    else:  # Handle unsupported model names
        raise ValueError(f"Unsupported model: {model_name}")  # Raise error for invalid model specification

################################################################################
# SECTION 4: MAIN EXECUTION
################################################################################

def print_config(args):
    """Print configuration."""
    print("=" * 60)  # Print top border for configuration display
    print(" " * 15, "Image Classification Pipeline Configuration")  # Print centered title
    print("=" * 60)  # Print separator line
    for k, v in vars(args).items():  # Iterate through all command line arguments
        print(f"{k:<20}: {v}")  # Print each argument name and value with formatting
    print("-" * 60)  # Print separator before device info
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine available compute device
    print(f"{'Device':<20}: {device}")  # Display selected compute device
    print("=" * 60)  # Print bottom border
    return device  # Return device for use in training

def main(args):
    """Main function."""
    device = print_config(args)  # Display configuration and get compute device
    
    # Create data module
    if args.custom_path:  # Check if custom dataset path is provided
        data = DataModule(dataset_name='custom', batch_size=args.batch_size, custom_path=args.custom_path)  # Create data module for custom images
        # Create a dummy dataloader to initialize num_classes
        _ = data.get_dataloader(train=True)  # Initialize dataloader to determine number of classes in custom dataset
    else:  # Use standard dataset
        data = DataModule(dataset_name=args.dataset, batch_size=args.batch_size)  # Create data module for standard dataset
    
    # Create model
    model = get_model(args.model, data.num_classes, data.in_channels)  # Instantiate model with correct architecture and parameters
    print(f"Created {args.model} model with {data.num_classes} classes")  # Display model creation confirmation
    
    # Create trainer and start training
    trainer = SimpleTrainer(max_epochs=args.num_epochs, device=device)  # Initialize trainer with specified epochs and device
    
    print("Starting training...")  # Display training start message
    start_time = time.time()  # Record training start timestamp
    trainer.fit(model, data)  # Execute training loop with model and data
    end_time = time.time()  # Record training completion timestamp
    
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")  # Display total training duration
    print("=" * 60)  # Print final separator line

if __name__ == "__main__":
    parser = argparse.ArgumentParser(  # Create command line argument parser
        description="Neural Network Image Classification Pipeline",  # Set program description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # Show default values in help text
    
    # Model and Dataset Arguments
    parser.add_argument('--model', type=str, default='ResNet',  # Define model architecture argument
                        choices=['LeNet', 'AlexNet', 'VGG', 'ResNet'],  # Restrict to supported model types
                        help='The CNN model architecture to train.')  # Provide help text for model selection
    parser.add_argument('--dataset', type=str, default='FashionMNIST',  # Define dataset selection argument
                        choices=['FashionMNIST', 'CIFAR10'],  # Restrict to supported datasets
                        help='The dataset to use for training and evaluation.')  # Provide help text for dataset selection
    parser.add_argument('--custom_path', type=str, default=None,  # Define custom dataset path argument
                        help='Path to custom image dataset directory.')  # Provide help text for custom dataset
    
    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,  # Define learning rate argument
                        help='Learning rate for the optimizer.')  # Provide help text for learning rate
    parser.add_argument('--num_epochs', type=int, default=5,  # Define number of epochs argument
                        help='Number of epochs to train for.')  # Provide help text for epoch count
    parser.add_argument('--batch_size', type=int, default=32,  # Define batch size argument
                        help='Number of samples per batch.')  # Provide help text for batch size
    
    args = parser.parse_args()  # Parse command line arguments into namespace object
    
    # If custom_path is provided, set dataset to custom
    if args.custom_path:  # Check if custom dataset path was specified
        args.dataset = 'custom'  # Override dataset selection to use custom dataset
    
    main(args)  # Execute main function with parsed arguments
