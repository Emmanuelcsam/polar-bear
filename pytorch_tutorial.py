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

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking if {package_name} is installed...", "INFO")
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        log(f"{package_name} not found. Installing latest version...", "WARNING")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"Successfully installed {package_name}", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            log(f"Failed to install {package_name}: {e}", "ERROR")
            return False
    else:
        log(f"{package_name} is already installed", "SUCCESS")
        # Upgrade to latest version
        log(f"Upgrading {package_name} to latest version...", "INFO")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"Successfully upgraded {package_name}", "SUCCESS")
        except:
            log(f"Could not upgrade {package_name}, using existing version", "WARNING")
        return True

def setup_environment():
    """Set up the environment by installing all required packages"""
    log("SETTING UP PYTORCH TUTORIAL ENVIRONMENT", "HEADER")
    
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
    
    log("Installing/Updating required packages...", "INFO")
    
    for package, import_name in required_packages.items():
        if not check_and_install_package(package, import_name):
            log(f"Failed to set up {package}. Exiting.", "ERROR")
            sys.exit(1)
        time.sleep(0.5)  # Small delay to avoid overwhelming pip
    
    log("All required packages are installed!", "SUCCESS")
    return True

# Now import all required packages after installation
def import_packages():
    """Import all required packages"""
    log("Importing required packages...", "INFO")
    
    global torch, nn, optim, F, transforms, datasets, DataLoader
    global np, plt, make_classification, train_test_split, StandardScaler
    global SummaryWriter, tqdm
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        import torchvision.models as models
        
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_classification, make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        from torch.utils.tensorboard import SummaryWriter
        from tqdm import tqdm
        
        log("All packages imported successfully!", "SUCCESS")
        return True
    except ImportError as e:
        log(f"Failed to import packages: {e}", "ERROR")
        return False

# Tutorial 1: Installation and Setup
def tutorial_1_installation():
    """Tutorial 1: Installation and PyTorch Basics"""
    log("\n=== TUTORIAL 1: INSTALLATION AND SETUP ===", "HEADER")
    
    log("Checking PyTorch installation...", "INFO")
    log(f"PyTorch version: {torch.__version__}", "SUCCESS")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    log(f"CUDA available: {cuda_available}", "INFO")
    log(f"Using device: {device}", "INFO")
    
    if cuda_available:
        log(f"CUDA device count: {torch.cuda.device_count()}", "INFO")
        log(f"CUDA device name: {torch.cuda.get_device_name(0)}", "INFO")
    
    return device

# Tutorial 2: Tensor Basics
def tutorial_2_tensor_basics(device):
    """Tutorial 2: Tensor Basics"""
    log("\n=== TUTORIAL 2: TENSOR BASICS ===", "HEADER")
    
    # Creating tensors
    log("Creating empty tensor...", "INFO")
    x = torch.empty(3)
    log(f"Empty tensor: {x}", "SUCCESS")
    
    log("Creating random tensor...", "INFO")
    x = torch.rand(2, 3)
    log(f"Random tensor shape {x.shape}: \n{x}", "SUCCESS")
    
    log("Creating zeros tensor...", "INFO")
    x = torch.zeros(2, 3, dtype=torch.int)
    log(f"Zeros tensor: \n{x}", "SUCCESS")
    
    log("Creating ones tensor...", "INFO")
    x = torch.ones(2, 3)
    log(f"Ones tensor: \n{x}", "SUCCESS")
    
    log("Creating tensor from list...", "INFO")
    x = torch.tensor([2.5, 0.1, 3.0])
    log(f"Tensor from list: {x}", "SUCCESS")
    
    # Basic operations
    log("\nDemonstrating basic tensor operations...", "INFO")
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    
    log(f"Tensor x: \n{x}", "INFO")
    log(f"Tensor y: \n{y}", "INFO")
    
    log("Addition:", "INFO")
    z = x + y
    log(f"x + y = \n{z}", "SUCCESS")
    
    log("In-place addition:", "INFO")
    y.add_(x)
    log(f"y after in-place addition: \n{y}", "SUCCESS")
    
    log("Multiplication:", "INFO")
    z = torch.mul(x, y)
    log(f"x * y = \n{z}", "SUCCESS")
    
    # Slicing
    log("\nDemonstrating tensor slicing...", "INFO")
    x = torch.rand(5, 3)
    log(f"Original tensor: \n{x}", "INFO")
    log(f"First column: {x[:, 0]}", "SUCCESS")
    log(f"Second row: {x[1, :]}", "SUCCESS")
    log(f"Element at position [1, 1]: {x[1, 1].item()}", "SUCCESS")
    
    # Reshaping
    log("\nDemonstrating tensor reshaping...", "INFO")
    x = torch.rand(4, 4)
    log(f"Original shape: {x.shape}", "INFO")
    y = x.view(16)
    log(f"Reshaped to 1D: {y.shape}", "SUCCESS")
    y = x.view(-1, 8)
    log(f"Reshaped with -1: {y.shape}", "SUCCESS")
    
    # NumPy conversion
    log("\nDemonstrating NumPy conversion...", "INFO")
    a = torch.ones(5)
    log(f"Torch tensor: {a}", "INFO")
    b = a.numpy()
    log(f"NumPy array: {b}", "SUCCESS")
    log(f"Type: {type(b)}", "INFO")
    
    # Move to GPU if available
    if device.type == "cuda":
        log("\nMoving tensors to GPU...", "INFO")
        x = torch.ones(5, device=device)
        log(f"Tensor on GPU: {x}", "SUCCESS")
        log(f"Device: {x.device}", "INFO")

# Tutorial 3: Autograd
def tutorial_3_autograd():
    """Tutorial 3: Automatic Gradient Computation"""
    log("\n=== TUTORIAL 3: AUTOGRAD ===", "HEADER")
    
    log("Creating tensor with gradient tracking...", "INFO")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    log(f"Tensor x: {x}", "SUCCESS")
    log(f"Requires grad: {x.requires_grad}", "INFO")
    
    log("\nPerforming operations...", "INFO")
    y = x + 2
    log(f"y = x + 2: {y}", "SUCCESS")
    log(f"y.grad_fn: {y.grad_fn}", "INFO")
    
    z = y * y * 2
    log(f"z = y * y * 2: {z}", "SUCCESS")
    
    out = z.mean()
    log(f"out = z.mean(): {out}", "SUCCESS")
    
    log("\nComputing gradients...", "INFO")
    out.backward()
    log(f"Gradients of x: {x.grad}", "SUCCESS")
    
    # Gradient accumulation example
    log("\nDemonstrating gradient accumulation...", "WARNING")
    x = torch.ones(4, requires_grad=True)
    
    for epoch in range(3):
        log(f"\nEpoch {epoch + 1}", "INFO")
        model_output = (x * 3).sum()
        log(f"Model output: {model_output}", "INFO")
        
        model_output.backward()
        log(f"Gradient after backward: {x.grad}", "SUCCESS")
        
        if epoch < 2:
            log("WARNING: Gradients accumulate! Need to zero them.", "WARNING")
        else:
            log("Zeroing gradients...", "INFO")
            x.grad.zero_()

# Tutorial 4: Backpropagation Example
def tutorial_4_backpropagation():
    """Tutorial 4: Backpropagation Example"""
    log("\n=== TUTORIAL 4: BACKPROPAGATION ===", "HEADER")
    
    log("Implementing linear regression example...", "INFO")
    
    # Training data
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)
    
    # Initialize weight
    w = torch.tensor(1.0, requires_grad=True)
    
    log(f"Input x: {x}", "INFO")
    log(f"Target y: {y}", "INFO")
    log(f"Initial weight w: {w}", "INFO")
    
    # Forward pass
    log("\nForward pass:", "INFO")
    y_hat = w * x
    log(f"Prediction y_hat = w * x = {y_hat}", "SUCCESS")
    
    # Compute loss
    loss = (y_hat - y)**2
    log(f"Loss = (y_hat - y)^2 = {loss}", "SUCCESS")
    
    # Backward pass
    log("\nBackward pass:", "INFO")
    loss.backward()
    log(f"Gradient dL/dw = {w.grad}", "SUCCESS")
    log("Expected gradient: -2.0 (manually calculated)", "INFO")
    
    # Update weights
    log("\nWeight update:", "INFO")
    learning_rate = 0.01
    with torch.no_grad():
        w -= learning_rate * w.grad
    log(f"Updated weight: {w}", "SUCCESS")

# Tutorial 5: Training Pipeline
def tutorial_5_training_pipeline():
    """Tutorial 5: Complete Training Pipeline"""
    log("\n=== TUTORIAL 5: TRAINING PIPELINE ===", "HEADER")
    
    # Generate synthetic data
    log("Generating synthetic regression data...", "INFO")
    X_numpy, y_numpy = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
    
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    y = y.view(y.shape[0], 1)
    
    log(f"Data shape - X: {X.shape}, y: {y.shape}", "SUCCESS")
    
    # Model
    log("\nCreating linear regression model...", "INFO")
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    log(f"Model: {model}", "SUCCESS")
    
    # Loss and optimizer
    log("\nSetting up loss and optimizer...", "INFO")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    log("Loss: MSE, Optimizer: SGD", "SUCCESS")
    
    # Training loop
    log("\nStarting training...", "INFO")
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}", "INFO")
    
    log("Training completed!", "SUCCESS")
    
    # Test the model
    log("\nTesting the model...", "INFO")
    with torch.no_grad():
        test_input = torch.tensor([[5.0]])
        prediction = model(test_input)
        log(f"Input: {test_input.item()}, Prediction: {prediction.item():.2f}", "SUCCESS")

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

def tutorial_6_datasets_dataloaders():
    """Tutorial 6: Datasets and DataLoaders"""
    log("\n=== TUTORIAL 6: DATASETS AND DATALOADERS ===", "HEADER")
    
    log("Creating custom Wine dataset...", "INFO")
    dataset = WineDataset()
    log(f"Dataset size: {len(dataset)}", "SUCCESS")
    
    # Get first sample
    first_data = dataset[0]
    features, labels = first_data
    log(f"First sample - Features shape: {features.shape}, Label: {labels}", "INFO")
    
    # Create DataLoader
    log("\nCreating DataLoader...", "INFO")
    batch_size = 4
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    log(f"DataLoader created with batch size: {batch_size}", "SUCCESS")
    
    # Iterate through DataLoader
    log("\nIterating through DataLoader...", "INFO")
    dataiter = iter(dataloader)
    data = next(dataiter)
    features, labels = data
    log(f"Batch features shape: {features.shape}", "SUCCESS")
    log(f"Batch labels shape: {labels.shape}", "SUCCESS")
    
    # Training loop simulation
    log("\nSimulating training loop with DataLoader...", "INFO")
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = int(np.ceil(total_samples / batch_size))
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            if i % 20 == 0:
                log(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_iterations}], "
                    f"Batch shape: {inputs.shape}", "INFO")
                
        if epoch == 0:
            log(f"Completed epoch {epoch+1}", "SUCCESS")

# Tutorial 7: Neural Network
class NeuralNet(nn.Module):
    """Simple Feed-Forward Neural Network"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        log("Initializing Neural Network...", "INFO")
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        log(f"Network architecture: {input_size} -> {hidden_size} -> {num_classes}", "SUCCESS")
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def tutorial_7_neural_network(device):
    """Tutorial 7: Feed-Forward Neural Network"""
    log("\n=== TUTORIAL 7: FEED-FORWARD NEURAL NETWORK ===", "HEADER")
    
    # Hyperparameters
    input_size = 784  # 28x28
    hidden_size = 100
    num_classes = 10
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    
    log("Loading MNIST dataset...", "INFO")
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True,
                                 transform=transforms.ToTensor(), download=True)
    
    test_dataset = datasets.MNIST(root='./data', train=False,
                                transform=transforms.ToTensor())
    
    log(f"Training samples: {len(train_dataset)}", "SUCCESS")
    log(f"Test samples: {len(test_dataset)}", "SUCCESS")
    
    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    log("\nCreating neural network model...", "INFO")
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log("Loss: CrossEntropy, Optimizer: Adam", "SUCCESS")
    
    # Training loop
    log("\nStarting training...", "INFO")
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
                log(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], '
                    f'Loss: {loss.item():.4f}', "INFO")
    
    log("Training completed!", "SUCCESS")
    
    # Test the model
    log("\nEvaluating model...", "INFO")
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
        log(f'Test Accuracy: {accuracy:.2f}%', "SUCCESS")

# Tutorial 8: Convolutional Neural Network
class ConvNet(nn.Module):
    """Convolutional Neural Network"""
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        log("Initializing Convolutional Neural Network...", "INFO")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        log("CNN architecture created successfully", "SUCCESS")
        
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

def tutorial_8_cnn(device):
    """Tutorial 8: Convolutional Neural Network"""
    log("\n=== TUTORIAL 8: CONVOLUTIONAL NEURAL NETWORK ===", "HEADER")
    
    # Hyperparameters
    num_epochs = 2
    batch_size = 4
    learning_rate = 0.001
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    log("Loading CIFAR-10 dataset...", "INFO")
    
    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                   download=True, transform=transform)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    log(f"Classes: {classes}", "INFO")
    log(f"Training samples: {len(train_dataset)}", "SUCCESS")
    
    # Model
    log("\nCreating CNN model...", "INFO")
    model = ConvNet(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training (abbreviated for demo)
    log("\nStarting CNN training (abbreviated)...", "INFO")
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
                log(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}', "INFO")
                running_loss = 0.0
    
    log("CNN training completed!", "SUCCESS")

# Tutorial 9: Save and Load Models
def tutorial_9_save_load_models(device):
    """Tutorial 9: Saving and Loading Models"""
    log("\n=== TUTORIAL 9: SAVE AND LOAD MODELS ===", "HEADER")
    
    # Create a simple model
    log("Creating a simple model...", "INFO")
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    log(f"Model created: {model}", "SUCCESS")
    
    # Method 1: Save entire model (lazy way)
    log("\nMethod 1: Saving entire model...", "INFO")
    torch.save(model, 'model_complete.pth')
    log("Model saved to 'model_complete.pth'", "SUCCESS")
    
    # Load entire model
    log("Loading entire model...", "INFO")
    loaded_model = torch.load('model_complete.pth')
    loaded_model.eval()
    log("Model loaded successfully", "SUCCESS")
    
    # Method 2: Save state dict (recommended)
    log("\nMethod 2: Saving state dict (recommended)...", "INFO")
    torch.save(model.state_dict(), 'model_state_dict.pth')
    log("State dict saved to 'model_state_dict.pth'", "SUCCESS")
    
    # Load state dict
    log("Loading state dict...", "INFO")
    new_model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    new_model.load_state_dict(torch.load('model_state_dict.pth'))
    new_model.eval()
    log("State dict loaded successfully", "SUCCESS")
    
    # Save checkpoint with additional info
    log("\nSaving checkpoint with optimizer state...", "INFO")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.4
    }
    torch.save(checkpoint, 'checkpoint.pth')
    log("Checkpoint saved to 'checkpoint.pth'", "SUCCESS")
    
    # Load checkpoint
    log("Loading checkpoint...", "INFO")
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    log(f"Checkpoint loaded - Epoch: {epoch}, Loss: {loss}", "SUCCESS")
    
    # Clean up
    import os
    for file in ['model_complete.pth', 'model_state_dict.pth', 'checkpoint.pth']:
        if os.path.exists(file):
            os.remove(file)
            log(f"Cleaned up {file}", "INFO")

# Tutorial 10: Transfer Learning
def tutorial_10_transfer_learning(device):
    """Tutorial 10: Transfer Learning with ResNet"""
    log("\n=== TUTORIAL 10: TRANSFER LEARNING ===", "HEADER")
    
    log("Loading pre-trained ResNet18...", "INFO")
    model = models.resnet18(pretrained=True)
    log("ResNet18 loaded successfully", "SUCCESS")
    
    # Freeze all layers
    log("\nFreezing all layers except the last one...", "INFO")
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last fully connected layer
    num_features = model.fc.in_features
    log(f"Number of features in last layer: {num_features}", "INFO")
    
    # New layer for binary classification (e.g., cats vs dogs)
    model.fc = nn.Linear(num_features, 2)
    log("Replaced final layer for binary classification", "SUCCESS")
    
    # Move to device
    model = model.to(device)
    
    # Only parameters of final layer are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    log("Optimizer set up for final layer only", "SUCCESS")
    
    # Create dummy data for demonstration
    log("\nCreating dummy data for demonstration...", "INFO")
    dummy_inputs = torch.randn(4, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1, 1, 0]).to(device)
    
    # Forward pass
    log("Performing forward pass...", "INFO")
    outputs = model(dummy_inputs)
    log(f"Output shape: {outputs.shape}", "SUCCESS")
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, dummy_labels)
    log(f"Loss: {loss.item():.4f}", "INFO")
    
    log("Transfer learning setup completed!", "SUCCESS")

# Tutorial 11: TensorBoard Integration
def tutorial_11_tensorboard():
    """Tutorial 11: TensorBoard Integration"""
    log("\n=== TUTORIAL 11: TENSORBOARD INTEGRATION ===", "HEADER")
    
    log("Creating TensorBoard writer...", "INFO")
    writer = SummaryWriter('runs/pytorch_tutorial')
    log("TensorBoard writer created at 'runs/pytorch_tutorial'", "SUCCESS")
    
    # Add scalar values
    log("\nAdding scalar values to TensorBoard...", "INFO")
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    
    log("Scalar values added", "SUCCESS")
    
    # Add images
    log("\nAdding sample images...", "INFO")
    img_batch = torch.rand(16, 3, 28, 28)
    img_grid = torchvision.utils.make_grid(img_batch)
    writer.add_image('sample_images', img_grid)
    log("Images added", "SUCCESS")
    
    # Add model graph
    log("\nAdding model graph...", "INFO")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    dummy_input = torch.rand(1, 10)
    writer.add_graph(model, dummy_input)
    log("Model graph added", "SUCCESS")
    
    # Add histogram
    log("\nAdding weight histograms...", "INFO")
    for n_iter in range(10):
        weights = torch.randn(100, 50)
        writer.add_histogram('weights', weights, n_iter)
    log("Histograms added", "SUCCESS")
    
    writer.close()
    log("TensorBoard writer closed", "SUCCESS")
    log("\nTo view TensorBoard, run: tensorboard --logdir=runs", "INFO")

# Main execution
def main():
    """Main function to run all tutorials"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("="*60)
    print("COMPREHENSIVE PYTORCH TUTORIAL")
    print("From Basics to Advanced Topics")
    print("="*60)
    print(f"{Colors.ENDC}\n")
    
    # Setup environment
    if not setup_environment():
        log("Failed to setup environment. Exiting.", "ERROR")
        return
    
    # Import packages
    if not import_packages():
        log("Failed to import packages. Exiting.", "ERROR")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    log("Created data directory", "SUCCESS")
    
    try:
        # Run tutorials
        device = tutorial_1_installation()
        tutorial_2_tensor_basics(device)
        tutorial_3_autograd()
        tutorial_4_backpropagation()
        tutorial_5_training_pipeline()
        tutorial_6_datasets_dataloaders()
        tutorial_7_neural_network(device)
        tutorial_8_cnn(device)
        tutorial_9_save_load_models(device)
        tutorial_10_transfer_learning(device)
        tutorial_11_tensorboard()
        
        # Summary
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("="*60)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"{Colors.ENDC}\n")
        
        log("All tutorials completed successfully!", "SUCCESS")
        log("You've covered:", "INFO")
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
            log(f"  ✓ {tutorial}", "SUCCESS")
            
    except Exception as e:
        log(f"An error occurred: {e}", "ERROR")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        log("\nCleaning up temporary files...", "INFO")
        # Remove data folder if you want
        # import shutil
        # if os.path.exists('data'):
        #     shutil.rmtree('data')
        #     log("Removed data directory", "INFO")
        
        if os.path.exists('runs'):
            log("TensorBoard logs saved in 'runs' directory", "INFO")

if __name__ == "__main__":
    main()