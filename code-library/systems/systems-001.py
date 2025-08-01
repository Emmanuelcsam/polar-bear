#!/usr/bin/env python3
"""
Convolutional Neural Network Training Script for CIFAR-10 Dataset
Based on PyTorch tutorial - Auto-installs requirements and logs all actions
"""

import sys
import os
import subprocess
import importlib
import time
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if level == "INFO":
        color = Colors.OKBLUE
    elif level == "SUCCESS":
        color = Colors.OKGREEN
    elif level == "WARNING":
        color = Colors.WARNING
    elif level == "ERROR":
        color = Colors.FAIL
    elif level == "HEADER":
        color = Colors.HEADER + Colors.BOLD
    else:
        color = ""
    
    print(f"{color}[{timestamp}] {level}: {message}{Colors.ENDC}")

def check_and_install_requirements():
    """Check for required packages and install if missing"""
    log("CIFAR-10 CNN Training Script Starting", "HEADER")
    log("Checking system requirements...")
    
    required_packages = {
        'numpy': 'numpy',
        'PIL': 'pillow',
        'torch': 'torch',
        'torchvision': 'torchvision'
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        log(f"Checking for {pip_name}...")
        try:
            importlib.import_module(import_name)
            log(f"✓ {pip_name} is already installed", "SUCCESS")
        except ImportError:
            log(f"✗ {pip_name} is not installed", "WARNING")
            missing_packages.append(pip_name)
    
    if missing_packages:
        log(f"Installing missing packages: {', '.join(missing_packages)}", "WARNING")
        
        # Update pip first
        log("Updating pip to latest version...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        log("✓ pip updated successfully", "SUCCESS")
        
        # Install missing packages
        for package in missing_packages:
            log(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                log(f"✓ {package} installed successfully", "SUCCESS")
            except subprocess.CalledProcessError as e:
                log(f"Failed to install {package}: {e}", "ERROR")
                sys.exit(1)
    else:
        log("✓ All required packages are already installed", "SUCCESS")

# Check and install requirements before importing
check_and_install_requirements()

# Now import the required packages
log("Importing required packages...")
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
log("✓ All packages imported successfully", "SUCCESS")

class NeuralNet(nn.Module):
    """Convolutional Neural Network for CIFAR-10 classification"""
    
    def __init__(self):
        super().__init__()
        log("Initializing Neural Network architecture...")
        
        # First convolutional layer: 3 input channels -> 12 feature maps
        self.conv1 = nn.Conv2d(3, 12, 5)
        log("  - Added Conv2D layer 1: 3 channels -> 12 feature maps, 5x5 kernel")
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        log("  - Added MaxPool2D layer: 2x2")
        
        # Second convolutional layer: 12 -> 24 feature maps
        self.conv2 = nn.Conv2d(12, 24, 5)
        log("  - Added Conv2D layer 2: 12 channels -> 24 feature maps, 5x5 kernel")
        
        # Fully connected layers
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        log("  - Added Fully Connected layer 1: 600 -> 120 neurons")
        
        self.fc2 = nn.Linear(120, 84)
        log("  - Added Fully Connected layer 2: 120 -> 84 neurons")
        
        self.fc3 = nn.Linear(84, 10)
        log("  - Added Fully Connected layer 3: 84 -> 10 neurons (output)")
        
        log("✓ Neural Network architecture initialized", "SUCCESS")
    
    def forward(self, x):
        """Forward pass through the network"""
        # First conv layer -> ReLU -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv layer -> ReLU -> pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation - will use CrossEntropyLoss)
        x = self.fc3(x)
        
        return x

def setup_data_transforms():
    """Set up data transformations"""
    log("Setting up data transformations...")
    
    # Transform for training/testing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    log("  - ToTensor: Convert images to tensors")
    log("  - Normalize: Scale to [-1, 1] range (mean=0.5, std=0.5)")
    
    # Transform for loading custom images
    transform_custom = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    log("  - Added Resize for custom images: 32x32 pixels")
    
    log("✓ Data transformations set up", "SUCCESS")
    return transform, transform_custom

def download_and_prepare_data(transform):
    """Download CIFAR-10 dataset and create data loaders"""
    log("Preparing CIFAR-10 dataset...")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
        log("Created ./data directory")
    
    # Download training data
    log("Downloading/loading training data...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    log(f"✓ Training data loaded: {len(trainset)} samples", "SUCCESS")
    
    # Download test data
    log("Downloading/loading test data...")
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    log(f"✓ Test data loaded: {len(testset)} samples", "SUCCESS")
    
    # Create data loaders
    log("Creating data loaders...")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    log("  - Train loader: batch_size=32, shuffle=True, workers=2")
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    log("  - Test loader: batch_size=32, shuffle=False, workers=2")
    
    log("✓ Data loaders created", "SUCCESS")
    
    # Define class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    log(f"Classes: {', '.join(classes)}")
    
    # Check data shape
    images, labels = trainset[0]
    log(f"Image shape: {images.shape} (3 channels, 32x32 pixels)")
    log(f"Number of classes: {len(classes)}")
    
    return trainloader, testloader, classes

def train_network(net, trainloader, num_epochs=30):
    """Train the neural network"""
    log(f"Starting training for {num_epochs} epochs...", "HEADER")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    log("Loss function: CrossEntropyLoss")
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    log("Optimizer: SGD (lr=0.001, momentum=0.9)")
    
    # Training loop
    for epoch in range(num_epochs):
        log(f"\nTraining Epoch {epoch + 1}/{num_epochs}...")
        running_loss = 0.0
        batch_count = 0
        
        for i, data in enumerate(trainloader):
            # Get inputs and labels
            inputs, labels = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            batch_count += 1
            
            # Log progress every 100 batches
            if i % 100 == 99:
                avg_loss = running_loss / 100
                log(f"  Batch {i+1}: Loss = {avg_loss:.4f}")
                running_loss = 0.0
        
        # Epoch statistics
        epoch_loss = running_loss / batch_count if batch_count > 0 else 0
        log(f"✓ Epoch {epoch + 1} completed - Average Loss: {epoch_loss:.4f}", "SUCCESS")
    
    log("✓ Training completed!", "SUCCESS")
    return net

def save_model(net):
    """Save the trained model"""
    log("Saving trained model...")
    
    model_path = 'trained_net.pth'
    torch.save(net.state_dict(), model_path)
    log(f"✓ Model saved to {model_path}", "SUCCESS")
    
    # Verify save by loading
    log("Verifying saved model...")
    test_net = NeuralNet()
    test_net.load_state_dict(torch.load(model_path))
    log("✓ Model verification successful", "SUCCESS")
    
    return model_path

def evaluate_network(net, testloader, classes):
    """Evaluate the network on test data"""
    log("Evaluating network on test data...", "HEADER")
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    # Set to evaluation mode
    net.eval()
    log("Network set to evaluation mode")
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    accuracy = 100 * correct / total
    log(f"\n✓ Overall Accuracy: {accuracy:.2f}%", "SUCCESS")
    log(f"  Correctly classified: {correct}/{total}")
    
    # Per-class accuracy
    log("\nPer-class accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        log(f"  {classes[i]:10s}: {class_acc:.2f}%")
    
    return accuracy

def load_and_classify_images(net, transform_custom, classes):
    """Load and classify example images"""
    log("\nClassifying example images...", "HEADER")
    
    # Set to evaluation mode
    net.eval()
    
    # Check for example images
    image_files = ['example1.jpeg', 'example2.jpeg']
    found_images = []
    
    for img_file in image_files:
        if os.path.exists(img_file):
            found_images.append(img_file)
            log(f"✓ Found image: {img_file}", "SUCCESS")
        else:
            log(f"✗ Image not found: {img_file}", "WARNING")
    
    if not found_images:
        log("No example images found. Skipping image classification.", "WARNING")
        log("To test image classification, add 'example1.jpeg' and/or 'example2.jpeg' to the script directory.")
        return
    
    # Load and classify images
    with torch.no_grad():
        for img_path in found_images:
            log(f"\nProcessing {img_path}...")
            
            # Load image
            image = Image.open(img_path)
            log(f"  Original size: {image.size}")
            
            # Transform image
            image = transform_custom(image)
            image = image.unsqueeze(0)  # Add batch dimension
            log(f"  Transformed shape: {image.shape}")
            
            # Classify
            output = net(image)
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()
            
            # Get confidence scores
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities[0][class_idx].item() * 100
            
            log(f"✓ Prediction: {classes[class_idx]} (confidence: {confidence:.2f}%)", "SUCCESS")

def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        # Set up transforms
        transform, transform_custom = setup_data_transforms()
        
        # Download and prepare data
        trainloader, testloader, classes = download_and_prepare_data(transform)
        
        # Create neural network
        log("\nCreating neural network...")
        net = NeuralNet()
        log("✓ Neural network created", "SUCCESS")
        
        # Check if model already exists
        if os.path.exists('trained_net.pth'):
            log("\nFound existing trained model!", "WARNING")
            response = input("Load existing model? (y/n): ").lower()
            
            if response == 'y':
                log("Loading existing model...")
                net.load_state_dict(torch.load('trained_net.pth'))
                log("✓ Model loaded successfully", "SUCCESS")
            else:
                log("Training new model...")
                net = train_network(net, trainloader)
                save_model(net)
        else:
            # Train the network
            net = train_network(net, trainloader)
            
            # Save the model
            save_model(net)
        
        # Evaluate the network
        evaluate_network(net, testloader, classes)
        
        # Classify example images if available
        load_and_classify_images(net, transform_custom, classes)
        
        # Summary
        elapsed_time = time.time() - start_time
        log(f"\n{'='*50}", "HEADER")
        log(f"Script completed successfully!", "SUCCESS")
        log(f"Total execution time: {elapsed_time:.2f} seconds")
        log(f"{'='*50}", "HEADER")
        
    except KeyboardInterrupt:
        log("\nTraining interrupted by user", "WARNING")
        sys.exit(0)
    except Exception as e:
        log(f"An error occurred: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
