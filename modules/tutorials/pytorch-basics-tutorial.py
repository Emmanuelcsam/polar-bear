#!/usr/bin/env python3
"""
Comprehensive PyTorch Tutorial Script
Based on the All-in-One PyTorch Video Tutorial
Covers: Installation, Tensors, Autograd, Neural Networks, CNNs, Transfer Learning, and more
"""

import os
import sys
import subprocess
import importlib.util
import time
from datetime import datetime
import json

# --- Start of Connector Integration ---
try:
    import connector
    logger = connector.logger
except ImportError:
    print("Connector not found, using basic logging.")
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger(__name__)

CONFIG_FILE = "shared_config.json"
SCRIPT_NAME = "pytorch-basics-tutorial"

# --- End of Connector Integration ---

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    logger.info(f"Checking if {package_name} is installed...")
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        logger.warning(f"{package_name} not found. Installing latest version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logger.info(f"Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e}")
            return False
    else:
        logger.info(f"{package_name} is already installed")
        # Upgrade to latest version
        logger.info(f"Upgrading {package_name} to latest version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            logger.info(f"Successfully upgraded {package_name}")
        except:
            logger.warning(f"Could not upgrade {package_name}, using existing version")
        return True

def setup_environment():
    """Set up the environment by installing all required packages"""
    logger.info("SETTING UP PYTORCH TUTORIAL ENVIRONMENT")
    
    required_packages = {
        "numpy": "numpy",
        "torch": "torch",
        "torchvision": "torchvision",
        "matplotlib": "matplotlib",
        "scikit-learn": "sklearn",
        "tensorboard": "tensorboard",
        "pandas": "pandas",
        "tqdm": "tqdm"
    }
    
    logger.info("Installing/Updating required packages...")
    
    for package, import_name in required_packages.items():
        if not check_and_install_package(package, import_name):
            logger.error(f"Failed to set up {package}. Exiting.")
            sys.exit(1)
        time.sleep(0.5)  # Small delay to avoid overwhelming pip
    
    logger.info("All required packages are installed!")
    return True

# Now import all required packages after installation
def import_packages():
    """Import all required packages"""
    logger.info("Importing required packages...")
    
    packages = {}
    try:
        import torch
        packages['torch'] = torch
        import torch.nn as nn
        packages['nn'] = nn
        import torch.optim as optim
        packages['optim'] = optim
        import torch.nn.functional as F
        packages['F'] = F
        from torch.utils.data import DataLoader, Dataset
        packages['DataLoader'] = DataLoader
        packages['Dataset'] = Dataset
        import torchvision
        packages['torchvision'] = torchvision
        import torchvision.transforms as transforms
        packages['transforms'] = transforms
        import torchvision.datasets as datasets
        packages['datasets'] = datasets
        import torchvision.models as models
        packages['models'] = models
        
        import numpy as np
        packages['np'] = np
        import matplotlib.pyplot as plt
        packages['plt'] = plt
        from sklearn.datasets import make_classification, make_regression
        packages['make_classification'] = make_classification
        packages['make_regression'] = make_regression
        from sklearn.model_selection import train_test_split
        packages['train_test_split'] = train_test_split
        from sklearn.preprocessing import StandardScaler
        packages['StandardScaler'] = StandardScaler
        
        from torch.utils.tensorboard import SummaryWriter
        packages['SummaryWriter'] = SummaryWriter
        from tqdm import tqdm
        packages['tqdm'] = tqdm
        
        logger.info("All packages imported successfully!")
        return packages
    except ImportError as e:
        logger.error(f"Failed to import packages: {e}")
        return None

setup_environment()
packages = import_packages()
if not packages:
    sys.exit(1)

# Make modules available globally for the tutorial functions
torch = packages['torch']
nn = packages['nn']
optim = packages['optim']
F = packages['F']
transforms = packages['transforms']
datasets = packages['datasets']
DataLoader = packages['DataLoader']
Dataset = packages['Dataset']
models = packages['models']
np = packages['np']
plt = packages['plt']
make_classification = packages['make_classification']
make_regression = packages['make_regression']
train_test_split = packages['train_test_split']
StandardScaler = packages['StandardScaler']
SummaryWriter = packages['SummaryWriter']
tqdm = packages['tqdm']
torchvision = packages['torchvision']


# Tutorial 1: Installation and Setup
def tutorial_1_installation():
    """Tutorial 1: Installation and PyTorch Basics"""
    logger.info("\n=== TUTORIAL 1: INSTALLATION AND SETUP ===")
    
    logger.info("Checking PyTorch installation...")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"Using device: {device}")
    
    if cuda_available:
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    return device

# Tutorial 2: Tensor Basics
def tutorial_2_tensor_basics(device):
    """Tutorial 2: Tensor Basics"""
    logger.info("\n=== TUTORIAL 2: TENSOR BASICS ===")
    
    # Creating tensors
    logger.info("Creating empty tensor...")
    x = torch.empty(3)
    logger.info(f"Empty tensor: {x}")
    
    logger.info("Creating random tensor...")
    x = torch.rand(2, 3)
    logger.info(f"Random tensor shape {x.shape}: \n{x}")
    
    logger.info("Creating zeros tensor...")
    x = torch.zeros(2, 3, dtype=torch.int)
    logger.info(f"Zeros tensor: \n{x}")
    
    logger.info("Creating ones tensor...")
    x = torch.ones(2, 3)
    logger.info(f"Ones tensor: \n{x}")
    
    logger.info("Creating tensor from list...")
    x = torch.tensor([2.5, 0.1, 3.0])
    logger.info(f"Tensor from list: {x}")
    
    # Basic operations
    logger.info("\nDemonstrating basic tensor operations...")
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    
    logger.info(f"Tensor x: \n{x}")
    logger.info(f"Tensor y: \n{y}")
    
    logger.info("Addition:")
    z = x + y
    logger.info(f"x + y = \n{z}")
    
    logger.info("In-place addition:")
    y.add_(x)
    logger.info(f"y after in-place addition: \n{y}")
    
    logger.info("Multiplication:")
    z = torch.mul(x, y)
    logger.info(f"x * y = \n{z}")
    
    # Slicing
    logger.info("\nDemonstrating tensor slicing...")
    x = torch.rand(5, 3)
    logger.info(f"Original tensor: \n{x}")
    logger.info(f"First column: {x[:, 0]}")
    logger.info(f"Second row: {x[1, :]}")
    logger.info(f"Element at position [1, 1]: {x[1, 1].item()}")
    
    # Reshaping
    logger.info("\nDemonstrating tensor reshaping...")
    x = torch.rand(4, 4)
    logger.info(f"Original shape: {x.shape}")
    y = x.view(16)
    logger.info(f"Reshaped to 1D: {y.shape}")
    y = x.view(-1, 8)
    logger.info(f"Reshaped with -1: {y.shape}")
    
    # NumPy conversion
    logger.info("\nDemonstrating NumPy conversion...")
    a = torch.ones(5)
    logger.info(f"Torch tensor: {a}")
    b = a.numpy()
    logger.info(f"NumPy array: {b}")
    logger.info(f"Type: {type(b)}")
    
    # Move to GPU if available
    if device.type == "cuda":
        logger.info("\nMoving tensors to GPU...")
        x = torch.ones(5, device=device)
        logger.info(f"Tensor on GPU: {x}")
        logger.info(f"Device: {x.device}")

# Tutorial 3: Autograd
def tutorial_3_autograd():
    """Tutorial 3: Automatic Gradient Computation"""
    logger.info("\n=== TUTORIAL 3: AUTOGRAD ===")
    
    logger.info("Creating tensor with gradient tracking...")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    logger.info(f"Tensor x: {x}")
    logger.info(f"Requires grad: {x.requires_grad}")
    
    logger.info("\nPerforming operations...")
    y = x + 2
    logger.info(f"y = x + 2: {y}")
    logger.info(f"y.grad_fn: {y.grad_fn}")
    
    z = y * y * 2
    logger.info(f"z = y * y * 2: {z}")
    
    out = z.mean()
    logger.info(f"out = z.mean(): {out}")
    
    logger.info("\nComputing gradients...")
    out.backward()
    logger.info(f"Gradients of x: {x.grad}")
    
    # Gradient accumulation example
    logger.warning("\nDemonstrating gradient accumulation...")
    x = torch.ones(4, requires_grad=True)
    
    for epoch in range(3):
        logger.info(f"\nEpoch {epoch + 1}")
        model_output = (x * 3).sum()
        logger.info(f"Model output: {model_output}")
        
        model_output.backward()
        logger.info(f"Gradient after backward: {x.grad}")
        
        if epoch < 2:
            logger.warning("WARNING: Gradients accumulate! Need to zero them.")
        else:
            logger.info("Zeroing gradients...")
            x.grad.zero_()

# Tutorial 4: Backpropagation Example
def tutorial_4_backpropagation(config):
    """Tutorial 4: Backpropagation Example"""
    logger.info("\n=== TUTORIAL 4: BACKPROPAGATION ===")
    
    logger.info("Implementing linear regression example...")
    
    # Training data
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)
    
    # Initialize weight
    w = torch.tensor(1.0, requires_grad=True)
    
    logger.info(f"Input x: {x}")
    logger.info(f"Target y: {y}")
    logger.info(f"Initial weight w: {w}")
    
    # Forward pass
    logger.info("\nForward pass:")
    y_hat = w * x
    logger.info(f"Prediction y_hat = w * x = {y_hat}")
    
    # Compute loss
    loss = (y_hat - y)**2
    logger.info(f"Loss = (y_hat - y)^2 = {loss}")
    
    # Backward pass
    logger.info("\nBackward pass:")
    loss.backward()
    logger.info(f"Gradient dL/dw = {w.grad}")
    logger.info("Expected gradient: -2.0 (manually calculated)")
    
    # Update weights
    logger.info("\nWeight update:")
    learning_rate = config.get('learning_rate', 0.01)
    with torch.no_grad():
        w -= learning_rate * w.grad
    logger.info(f"Updated weight: {w}")

# Tutorial 5: Training Pipeline
def tutorial_5_training_pipeline(config):
    """Tutorial 5: Complete Training Pipeline"""
    logger.info("\n=== TUTORIAL 5: TRAINING PIPELINE ===")
    
    # Generate synthetic data
    logger.info("Generating synthetic regression data...")
    X_numpy, y_numpy = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
    
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y = y.view(y.shape[0], 1)
    
    logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    # Model
    logger.info("\nCreating linear regression model...")
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    logger.info(f"Model: {model}")
    
    # Loss and optimizer
    logger.info("\nSetting up loss and optimizer...")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.get('learning_rate', 0.01))
    logger.info("Loss: MSE, Optimizer: SGD")
    
    # Training loop
    logger.info("\nStarting training...")
    num_epochs = config.get('num_epochs', 100)
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    logger.info("Training completed!")
    
    # Test the model
    logger.info("\nTesting the model...")
    with torch.no_grad():
        test_input = torch.tensor([[5.0]])
        prediction = model(test_input)
        logger.info(f"Input: {test_input.item()}, Prediction: {prediction.item():.2f}")

# Tutorial 6: Custom Dataset
class WineDataset(Dataset):
    """Custom Dataset for Wine data"""
    def __init__(self, transform=None):
        # For demo, we'll create synthetic wine data
        xy, _ = make_classification(n_samples=178, n_features=13, n_informative=10, 
                                  n_redundant=3, n_classes=3, random_state=1)
        self.x = torch.from_numpy(xy[:, 1:].astype(np.float32))
        self.y = torch.from_numpy(xy[:, 0].astype(np.float32))
        self.n_samples = xy.shape[0]
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples

def tutorial_6_datasets_dataloaders(config):
    """Tutorial 6: Datasets and DataLoaders"""
    logger.info("\n=== TUTORIAL 6: DATASETS AND DATALOADERS ===")
    
    logger.info("Creating custom Wine dataset...")
    dataset = WineDataset()
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    first_data = dataset[0]
    features, labels = first_data
    logger.info(f"First sample - Features shape: {features.shape}, Label: {labels}")
    
    # Create DataLoader
    logger.info("\nCreating DataLoader...")
    batch_size = config.get('batch_size', 4)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"DataLoader created with batch size: {batch_size}")
    
    # Iterate through DataLoader
    logger.info("\nIterating through DataLoader...")
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    logger.info(f"Batch features shape: {features.shape}")
    logger.info(f"Batch labels shape: {labels.shape}")
    
    # Training loop simulation
    logger.info("\nSimulating training loop with DataLoader...")
    num_epochs = config.get('num_epochs', 2)
    total_samples = len(dataset)
    n_iterations = int(np.ceil(total_samples / batch_size))
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            if i % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], "
                    f"Batch shape: {inputs.shape}")
                
        if epoch == 0:
            logger.info(f"Completed epoch {epoch+1}")

# Tutorial 7: Neural Network
class NeuralNet(nn.Module):
    """Simple Feed-Forward Neural Network"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        logger.info("Initializing Neural Network...")
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        logger.info(f"Network architecture: {input_size} -> {hidden_size} -> {num_classes}")
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def tutorial_7_neural_network(device, config):
    """Tutorial 7: Feed-Forward Neural Network"""
    logger.info("\n=== TUTORIAL 7: FEED-FORWARD NEURAL NETWORK ===")
    
    # Hyperparameters
    input_size = 784  # 28x28
    hidden_size = config.get('hidden_size', 100)
    num_classes = 10
    num_epochs = config.get('num_epochs', 2)
    batch_size = config.get('batch_size', 100)
    learning_rate = config.get('learning_rate', 0.001)
    
    logger.info("Loading MNIST dataset...")
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True,
                                 transform=transforms.ToTensor(), download=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False,
                                transform=transforms.ToTensor())
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    logger.info("\nCreating neural network model...")
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Loss: CrossEntropy, Optimizer: Adam")
    
    # Training loop
    logger.info("\nStarting training...")
    total_step = len(train_loader)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape images
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], '
                    f'Loss: {loss.item():.4f}')
    
    logger.info("Training completed!")
    
    # Test the model
    logger.info("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f'Test Accuracy: {accuracy:.2f}%')

# Tutorial 8: Convolutional Neural Network
class ConvNet(nn.Module):
    """Convolutional Neural Network"""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        logger.info("Initializing Convolutional Neural Network...")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        logger.info("CNN architecture created successfully")
        
    def forward(self, x):
        # Conv layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv layer 2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def tutorial_8_cnn(device, config):
    """Tutorial 8: Convolutional Neural Network"""
    logger.info("\n=== TUTORIAL 8: CONVOLUTIONAL NEURAL NETWORK ===")
    
    # Hyperparameters
    num_epochs = config.get('num_epochs', 2)
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    logger.info("Loading CIFAR-10 dataset...")
    
    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    logger.info(f"Classes: {classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # Model
    logger.info("\nCreating CNN model...")
    model = ConvNet(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training (abbreviated for demo)
    logger.info("\nStarting CNN training (abbreviated)...")
    model.train()
    
    for epoch in range(1):  # Just 1 epoch for demo
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            if i > 100:  # Limit iterations for demo
                break
                
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 49:
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}')
                running_loss = 0.0
    
    logger.info("CNN training completed!")

# Tutorial 9: Save and Load Models
def tutorial_9_save_load_models(device):
    """Tutorial 9: Saving and Loading Models"""
    logger.info("\n=== TUTORIAL 9: SAVE AND LOAD MODELS ===")
    
    # Create a simple model
    logger.info("Creating a simple model...")
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    logger.info(f"Model created: {model}")
    
    # Method 1: Save entire model (lazy way)
    logger.info("\nMethod 1: Saving entire model...")
    torch.save(model, 'model_complete.pth')
    logger.info("Model saved to 'model_complete.pth'")
    
    # Load entire model
    # Load entire model
    logger.info("Loading entire model...")
    loaded_model = torch.load('model_complete.pth', weights_only=False)
    loaded_model.eval()
    logger.info("Model loaded successfully")
    
    # Method 2: Save state dict (recommended)
    logger.info("\nMethod 2: Saving state dict (recommended)...")
    torch.save(model.state_dict(), 'model_state_dict.pth')
    logger.info("State dict saved to 'model_state_dict.pth'")
    
    # Load state dict
    logger.info("Loading state dict...")
    new_model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    new_model.load_state_dict(torch.load('model_state_dict.pth'))
    new_model.eval()
    logger.info("State dict loaded successfully")
    
    # Save checkpoint with additional info
    logger.info("\nSaving checkpoint with optimizer state...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.4
    }
    torch.save(checkpoint, 'checkpoint.pth')
    logger.info("Checkpoint saved to 'checkpoint.pth'")
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Checkpoint loaded - Epoch: {epoch}, Loss: {loss}")
    
    # Clean up
    import os
    for file in ['model_complete.pth', 'model_state_dict.pth', 'checkpoint.pth']:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"Cleaned up {file}")

# Tutorial 10: Transfer Learning
def tutorial_10_transfer_learning(device):
    """Tutorial 10: Transfer Learning with ResNet"""
    logger.info("\n=== TUTORIAL 10: TRANSFER LEARNING ===")
    
    logger.info("Loading pre-trained ResNet18...")
    model = models.resnet18(pretrained=True)
    logger.info("ResNet18 loaded successfully")
    
    # Freeze all layers
    logger.info("\nFreezing all layers except the last one...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last fully connected layer
    num_features = model.fc.in_features
    logger.info(f"Number of features in last layer: {num_features}")
    
    # New layer for binary classification (e.g., cats vs dogs)
    model.fc = nn.Linear(num_features, 2)
    logger.info("Replaced final layer for binary classification")
    
    # Move to device
    model = model.to(device)
    
    # Only parameters of final layer are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    logger.info("Optimizer set up for final layer only")
    
    # Create dummy data for demonstration
    logger.info("\nCreating dummy data for demonstration...")
    dummy_inputs = torch.randn(4, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1, 1, 0]).to(device)
    
    # Forward pass
    logger.info("Performing forward pass...")
    outputs = model(dummy_inputs)
    logger.info(f"Output shape: {outputs.shape}")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_labels)
    logger.info(f"Loss: {loss.item():.4f}")
    
    logger.info("Transfer learning setup completed!")

# Tutorial 11: TensorBoard Integration
def tutorial_11_tensorboard():
    """Tutorial 11: TensorBoard Integration"""
    logger.info("\n=== TUTORIAL 11: TENSORBOARD INTEGRATION ===")
    
    logger.info("Creating TensorBoard writer...")
    writer = SummaryWriter('runs/pytorch_tutorial')
    logger.info("TensorBoard writer created at 'runs/pytorch_tutorial'")
    
    # Add scalar values
    logger.info("\nAdding scalar values to TensorBoard...")
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    
    logger.info("Scalar values added")
    
    # Add images
    logger.info("\nAdding sample images...")
    img_batch = torch.rand(16, 3, 28, 28)
    img_grid = torchvision.utils.make_grid(img_batch)
    writer.add_image('sample_images', img_grid)
    logger.info("Images added")
    
    # Add model graph
    logger.info("\nAdding model graph...")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    dummy_input = torch.rand(1, 10)
    writer.add_graph(model, dummy_input)
    logger.info("Model graph added")
    
    # Add histogram
    logger.info("\nAdding weight histograms...")
    for n_iter in range(10):
        weights = torch.randn(100, 50)
        writer.add_histogram('weights', weights, n_iter)
    logger.info("Histograms added")
    
    writer.close()
    logger.info("TensorBoard writer closed")
    logger.info("\nTo view TensorBoard, run: tensorboard --logdir=runs")

# Main execution
def main():
    """Main function to run all tutorials"""
    logger.info(f"--- Starting {SCRIPT_NAME} ---")
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PYTORCH TUTORIAL")
    print("From Basics to Advanced Topics")
    print("="*60)
    print("\n")
    
    # Setup environment
    # setup_environment() is called at the beginning of the script.
    
    # Import packages
    # import_packages() is called at the beginning of the script.
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    logger.info("Created data directory")
    
    # Load config
    config = {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            shared_config = json.load(f)
        if SCRIPT_NAME in shared_config:
            config = shared_config[SCRIPT_NAME]
            logger.info(f"Loaded configuration for {SCRIPT_NAME} from {CONFIG_FILE}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load or parse {CONFIG_FILE}, using default parameters. Error: {e}")

    try:
        # Run tutorials
        device = tutorial_1_installation()
        tutorial_2_tensor_basics(device)
        tutorial_3_autograd()
        tutorial_4_backpropagation(config.get('tutorial_4', {}))
        tutorial_5_training_pipeline(config.get('tutorial_5', {}))
        tutorial_6_datasets_dataloaders(config.get('tutorial_6', {}))
        tutorial_7_neural_network(device, config.get('tutorial_7', {}))
        tutorial_8_cnn(device, config.get('tutorial_8', {}))
        tutorial_9_save_load_models(device)
        tutorial_10_transfer_learning(device)
        tutorial_11_tensorboard()
        
        # Summary
        print("\n" + "="*60)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n")
        
        logger.info("All tutorials completed successfully!")
        logger.info("You've covered:")
        tutorials = [
            "1. PyTorch Installation and Setup",
            "2. Tensor Basics and Operations",
            "3. Automatic Gradient Computation (Autograd)",
            "4. Backpropagation",
            "5. Complete Training Pipeline",
            "6. Custom Datasets and DataLoaders",
            "7. Feed-Forward Neural Networks",
            "8. Convolutional Neural Networks",
            "9. Saving and Loading Models",
            "10. Transfer Learning",
            "11. TensorBoard Integration"
        ]
        for tutorial in tutorials:
            logger.info(f"  ✓ {tutorial}")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("\nCleaning up temporary files...")
        # Remove data folder if you want
        # import shutil
        # if os.path.exists('data'):
        #     shutil.rmtree('data')
        #     logger.info("Removed data directory")
        
        if os.path.exists('runs'):
            logger.info("TensorBoard logs saved in 'runs' directory")
        
        logger.info(f"--- Finished {SCRIPT_NAME} ---")


if __name__ == "__main__":
    main()