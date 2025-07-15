#!/usr/bin/env python3
"""
PyTorch Computer Vision Tutorial Script
Based on tutorial by Daniel Brooks
Auto-installs dependencies and logs all actions
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Logger function
def log(message):
    """Print timestamped log messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

# Auto-installer function
def install_if_missing(package_name, import_name=None, conda_package=None):
    """Check if package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        log(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        log(f"✗ {package_name} not found. Installing...")
        
        # Try conda first for PyTorch packages
        if conda_package and check_conda_available():
            try:
                log(f"Installing {package_name} via conda...")
                subprocess.check_call([sys.executable, "-m", "conda", "install", "-y", conda_package])
                log(f"✓ Successfully installed {package_name} via conda")
                return True
            except:
                log(f"! Conda installation failed, trying pip...")
        
        # Fall back to pip
        try:
            log(f"Installing {package_name} via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"✓ Successfully installed {package_name} via pip")
            return True
        except Exception as e:
            log(f"✗ Failed to install {package_name}: {e}")
            return False

def check_conda_available():
    """Check if conda is available"""
    try:
        subprocess.check_output(["conda", "--version"])
        return True
    except:
        return False

# Install all required packages
log("=== PyTorch Computer Vision Tutorial ===")
log("Checking and installing dependencies...")

# Install packages as mentioned in the tutorial
packages = [
    ("torch", "torch", "pytorch -c pytorch"),
    ("torchvision", "torchvision", "torchvision -c pytorch"),
    ("numpy", "numpy", None),
    ("Pillow", "PIL", None),
    ("matplotlib", "matplotlib", None),
    ("requests", "requests", None),  # For downloading images
]

for package, import_name, conda_pkg in packages:
    install_if_missing(package, import_name, conda_pkg)

# Now import everything we need
log("\nImporting required libraries...")
try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import requests
    from io import BytesIO
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    log("✓ All imports successful")
except ImportError as e:
    log(f"✗ Import error: {e}")
    sys.exit(1)

# Section 1: Basic PyTorch Operations
log("\n=== Section 1: Basic PyTorch Operations ===")

# Check CUDA availability (as shown in tutorial)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
else:
    log("CUDA not available, using CPU")

# Basic tensor operations from tutorial
log("\nCreating basic tensors...")
x = torch.zeros(3, 3)
log(f"Zeros tensor shape: {x.shape}")
log(f"Zeros tensor:\n{x}")

x_plus_one = x + 1
log(f"\nZeros + 1:\n{x_plus_one}")

# Tensor from list
tensor_from_list = torch.tensor([1, 2, 3])
log(f"\nTensor from list: {tensor_from_list}")

# NumPy interoperability
log("\nNumPy interoperability...")
numpy_array = np.array([3, 2, 1])
log(f"NumPy array: {numpy_array}")

torch_from_numpy = torch.from_numpy(numpy_array)
log(f"Torch tensor from NumPy: {torch_from_numpy}")

# Convert back to NumPy
back_to_numpy = torch_from_numpy.numpy()
log(f"Back to NumPy: {back_to_numpy}")

# Section 2: Working with Images
log("\n=== Section 2: Working with Images ===")

# Download the sample image mentioned in tutorial
image_url = "https://github.com/pytorch/tutorials/raw/master/beginner_source/blitz/cifar10_tutorial.py"
# Using a placeholder image since the exact URL wasn't in the transcript
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/512px-Collage_of_Nine_Dogs.jpg"

log(f"Downloading sample image from: {image_url}")
try:
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')  # Ensure RGB
    img.save('dog.jpg')
    log("✓ Image downloaded and saved as 'dog.jpg'")
except Exception as e:
    log(f"! Could not download image: {e}")
    # Create a dummy image
    log("Creating dummy image instead...")
    img = Image.new('RGB', (256, 256), color='red')
    img.save('dog.jpg')

# Load and display image
log("\nLoading image with PIL...")
img = Image.open('dog.jpg')
log(f"Image size: {img.size}")
log(f"Image mode: {img.mode}")

# Convert to tensor using transforms
log("\nConverting image to tensor...")
transform_to_tensor = transforms.ToTensor()
img_tensor = transform_to_tensor(img)
log(f"Image tensor shape: {img_tensor.shape}")
log(f"Image tensor dtype: {img_tensor.dtype}")

# Add batch dimension (important gotcha mentioned in tutorial)
log("\nAdding batch dimension...")
img_batch = img_tensor.unsqueeze(0)
log(f"Batch tensor shape: {img_batch.shape}")

# Remove batch dimension
img_squeezed = img_batch.squeeze()
log(f"Squeezed tensor shape: {img_squeezed.shape}")

# Section 3: Pre-trained Models (Object Detection with Mask R-CNN)
log("\n=== Section 3: Pre-trained Models (Mask R-CNN) ===")

log("Loading pre-trained Mask R-CNN model...")
try:
    # Load the model as shown in tutorial
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set to evaluation mode
    log("✓ Model loaded successfully")
    
    # COCO class names (80 classes mentioned in tutorial)
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Prepare image for model
    log("\nPreparing image for object detection...")
    transform = transforms.Compose([transforms.ToTensor()])
    img_transformed = transform(img)
    
    # Run prediction
    log("Running object detection...")
    with torch.no_grad():
        prediction = model([img_transformed])
    
    # Extract results
    pred = prediction[0]
    log(f"\nDetected {len(pred['boxes'])} objects")
    
    # Show top predictions (confidence > 0.5 as mentioned in tutorial)
    threshold = 0.5
    high_conf_indices = pred['scores'] > threshold
    
    boxes = pred['boxes'][high_conf_indices]
    labels = pred['labels'][high_conf_indices]
    scores = pred['scores'][high_conf_indices]
    
    log(f"\nObjects with confidence > {threshold}:")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        log(f"  {i+1}. {class_name}: {score:.2f} confidence")
        log(f"     Box: {box.tolist()}")
        
except Exception as e:
    log(f"! Error with object detection: {e}")
    log("This might be due to model download issues or memory constraints")

# Section 4: Building and Training a Classifier (CIFAR-10)
log("\n=== Section 4: Training a Classifier on CIFAR-10 ===")

# Define transforms with normalization as shown in tutorial
log("\nSetting up data transforms...")
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download CIFAR-10 dataset
log("\nDownloading CIFAR-10 dataset...")
try:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)  # Set to 0 for compatibility

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    log("✓ CIFAR-10 dataset loaded")
    log(f"Training samples: {len(trainset)}")
    log(f"Test samples: {len(testset)}")
    
except Exception as e:
    log(f"! Error loading CIFAR-10: {e}")

# Define the neural network from tutorial
log("\nDefining neural network architecture...")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers as shown in tutorial
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Forward pass as shown in tutorial
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize network
net = Net()
log("✓ Network architecture defined")
log(f"Network structure:\n{net}")

# Move to GPU if available
net = net.to(device)
log(f"Network moved to: {device}")

# Define loss function and optimizer as shown in tutorial
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
log("✓ Loss function and optimizer initialized")

# Training loop (simplified version)
log("\nStarting training...")
log("Note: Full training would take a long time. Running a mini demonstration...")

try:
    # Train for just 1 epoch with limited batches for demonstration
    net.train()
    running_loss = 0.0
    
    for epoch in range(1):  # Just 1 epoch for demo
        log(f"\nEpoch {epoch + 1}")
        
        for i, data in enumerate(trainloader):
            if i >= 10:  # Just 10 batches for demo
                break
                
            # Get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # Print every 5 mini-batches
                log(f'  [Batch {i + 1}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0
    
    log("\n✓ Training demonstration complete")
    
    # Save the model
    log("\nSaving trained model...")
    torch.save(net.state_dict(), 'cifar_net.pth')
    log("✓ Model saved as 'cifar_net.pth'")
    
    # Quick evaluation
    log("\nEvaluating on a few test samples...")
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i >= 5:  # Just 5 batches
                break
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total > 0:
        accuracy = 100 * correct / total
        log(f'\nAccuracy on {total} test images: {accuracy:.2f}%')
    
except Exception as e:
    log(f"! Error during training: {e}")

# Section 5: Important PyTorch Tips from Tutorial
log("\n=== Section 5: Important Tips from Tutorial ===")

log("\n1. Batch Dimension:")
log("   - Always add batch dimension with unsqueeze(0) for single images")
log("   - Models expect shape: [batch_size, channels, height, width]")

log("\n2. Device Management:")
log("   - Always check CUDA availability")
log("   - Move both model and data to same device")

log("\n3. eval() vs train() modes:")
log("   - Use model.eval() for inference to disable dropout/batch norm")
log("   - Use model.train() when training")

log("\n4. Gradients:")
log("   - Use torch.no_grad() for inference to save memory")
log("   - Call optimizer.zero_grad() before each training step")

log("\n5. Pre-trained Models:")
log("   - Leverage pre-trained models for better performance")
log("   - Transfer learning often works better than training from scratch")

# Cleanup
log("\n=== Tutorial Complete ===")
log("Files created:")
log("  - dog.jpg (sample image)")
log("  - cifar_net.pth (trained model)")
log("  - data/ (CIFAR-10 dataset)")

log("\nFor more information, refer to:")
log("  - PyTorch documentation: https://pytorch.org")
log("  - Stanford CS231n course")
log("  - Deep Learning book by Goodfellow et al.")

log("\nThank you for running the PyTorch Computer Vision Tutorial!")
