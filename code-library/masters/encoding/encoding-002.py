#!/usr/bin/env python3
"""
PyTorch Computer Vision Tutorial Script
Based on the comprehensive tutorial covering Fashion MNIST classification
"""

import sys
import subprocess
import importlib
import os
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
SCRIPT_NAME = "cv-training-guide"

# --- End of Connector Integration ---

# Function to check and install packages
def ensure_package_installed(package_name, import_name=None, min_version=None):
    """Check if package is installed and install if not"""
    if import_name is None:
        import_name = package_name
    
    logger.info(f"Checking for {package_name}...")
    
    try:
        module = importlib.import_module(import_name)
        logger.info(f"✓ {package_name} is already installed")
        
        # Check version if specified
        if min_version and hasattr(module, '__version__'):
            current_version = module.__version__
            if current_version < min_version:
                logger.info(f"⚠ {package_name} version {current_version} is older than required {min_version}")
                logger.info(f"Upgrading {package_name}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
                logger.info(f"✓ {package_name} upgraded successfully")
        
        return module
    except ImportError:
        logger.info(f"✗ {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            module = importlib.import_module(import_name)
            logger.info(f"✓ {package_name} installed successfully")
            return module
        except Exception as e:
            logger.error(f"✗ Failed to install {package_name}: {e}")
            sys.exit(1)

# Install all required packages
logger.info("=== Setting up environment ===")
logger.info("Ensuring all required packages are installed...")

# Core packages
torch = ensure_package_installed("torch")
torchvision = ensure_package_installed("torchvision")
np = ensure_package_installed("numpy", "numpy")
matplotlib = ensure_package_installed("matplotlib")
import matplotlib.pyplot as plt
tqdm_auto = ensure_package_installed("tqdm", "tqdm.auto")
pandas = ensure_package_installed("pandas", "pandas")
ensure_package_installed("requests")
ensure_package_installed("pathlib")

# Additional packages for evaluation
torchmetrics = ensure_package_installed("torchmetrics")
mlxtend = ensure_package_installed("mlxtend", min_version="0.19.0")

# Import after installation
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import requests
from pathlib import Path
import random
from timeit import default_timer as timer
import torchmetrics
from mlxtend.plotting import plot_confusion_matrix

logger.info("✓ All packages installed and imported successfully")

# Print versions
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Torchvision version: {torchvision.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Download helper functions
logger.info("\n=== Downloading helper functions ===")
helper_functions_path = Path("helper_functions.py")

if helper_functions_path.exists():
    logger.info("✓ helper_functions.py already exists, skipping download")
else:
    logger.info("Downloading helper_functions.py...")
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    )
    with open(helper_functions_path, "wb") as f:
        f.write(request.content)
    logger.info("✓ helper_functions.py downloaded successfully")

from helper_functions import accuracy_fn

# Setup device agnostic code
logger.info("\n=== Setting up device ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Setup hyperparameters
logger.info("\n=== Setting up hyperparameters ===")
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.1

def load_and_apply_config():
    """Loads configuration from shared file and applies it."""
    global NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    try:
        with open(CONFIG_FILE, 'r') as f:
            shared_config = json.load(f)
        
        if SCRIPT_NAME in shared_config:
            script_config = shared_config[SCRIPT_NAME]
            logger.info(f"Loading configuration for {SCRIPT_NAME} from {CONFIG_FILE}")
            NUM_EPOCHS = script_config.get("NUM_EPOCHS", NUM_EPOCHS)
            BATCH_SIZE = script_config.get("BATCH_SIZE", BATCH_SIZE)
            LEARNING_RATE = script_config.get("LEARNING_RATE", LEARNING_RATE)
            
    except FileNotFoundError:
        logger.info(f"{CONFIG_FILE} not found. Using default hyperparameters.")
    except json.JSONDecodeError:
        logger.error(f"Error decoding {CONFIG_FILE}. Using default hyperparameters.")
    except Exception as e:
        logger.error(f"An error occurred while loading config: {e}")

load_and_apply_config()
logger.info(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")


# Download and prepare data
logger.info("\n=== Preparing Fashion MNIST dataset ===")
logger.info("Downloading training data...")
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

logger.info("Downloading test data...")
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

logger.info(f"✓ Training samples: {len(train_data)}")
logger.info(f"✓ Test samples: {len(test_data)}")

# Get class names
class_names = train_data.classes
logger.info(f"Classes: {class_names}")

# Create DataLoaders
logger.info("\n=== Creating DataLoaders ===")
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

logger.info(f"✓ Train DataLoader: {len(train_dataloader)} batches of {BATCH_SIZE}")
logger.info(f"✓ Test DataLoader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Visualize a sample
logger.info("\n=== Visualizing sample data ===")
image, label = train_data[0]
logger.info(f"Image shape: {image.shape}")
logger.info(f"Image label: {label} ({class_names[label]})")

# Function to plot random images
def plot_random_images(dataset, n=9):
    """Plot n random images from dataset"""
    logger.info(f"Plotting {n} random images...")
    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3
    
    for i in range(1, n + 1):
        random_idx = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[random_idx]
        
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)
    
    plt.tight_layout()
    plt.savefig("random_fashion_mnist_samples.png")
    logger.info("✓ Saved random samples plot")
    plt.close()

plot_random_images(train_data)

# Model 0: Baseline model
logger.info("\n=== Building Model 0: Baseline (Linear) ===")

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# Create model instance
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(
    input_shape=784,  # 28*28
    hidden_units=10,
    output_shape=len(class_names)
)
logger.info(f"✓ Model 0 created: {model_0}")

# Setup loss and optimizer for model 0
loss_fn = nn.CrossEntropyLoss()
optimizer_0 = torch.optim.SGD(params=model_0.parameters(), lr=LEARNING_RATE)

# Training and evaluation functions
def train_step(model, dataloader, loss_fn, optimizer, accuracy_fn, device):
    """Performs a training step with model trying to learn on dataloader"""
    train_loss, train_acc = 0, 0
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model, dataloader, loss_fn, accuracy_fn, device):
    """Performs a testing loop step on model going over dataloader"""
    test_loss, test_acc = 0, 0
    model.eval()
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Train Model 0
logger.info("\n=== Training Model 0 (Baseline) ===")
torch.manual_seed(42)
train_time_start_model_0 = timer()

epochs = NUM_EPOCHS
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model_0, train_dataloader, loss_fn, optimizer_0, accuracy_fn, device="cpu")
    test_step(model_0, test_dataloader, loss_fn, accuracy_fn, device="cpu")

train_time_end_model_0 = timer()
total_train_time_model_0 = print_train_time(
    start=train_time_start_model_0,
    end=train_time_end_model_0,
    device="cpu"
)

# Evaluate Model 0
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y).item()
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss,
        "model_acc": acc
    }

model_0_results = eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)
logger.info(f"✓ Model 0 results: {model_0_results}")

# Model 1: Model with non-linearity
logger.info("\n=== Building Model 1: With Non-linearity (ReLU) ===")

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# Create model instance
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)
logger.info(f"✓ Model 1 created and moved to {device}")

# Setup loss and optimizer for model 1
optimizer_1 = torch.optim.SGD(params=model_1.parameters(), lr=LEARNING_RATE)

# Train Model 1
logger.info("\n=== Training Model 1 (With Non-linearity) ===")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_model_1 = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model_1, train_dataloader, loss_fn, optimizer_1, accuracy_fn, device)
    test_step(model_1, test_dataloader, loss_fn, accuracy_fn, device)

train_time_end_model_1 = timer()
total_train_time_model_1 = print_train_time(
    start=train_time_start_model_1,
    end=train_time_end_model_1,
    device=device
)

# Update eval_model function for GPU
def eval_model_gpu(model, data_loader, loss_fn, accuracy_fn, device):
    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y).item()
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss,
        "model_acc": acc
    }

model_1_results = eval_model_gpu(model_1, test_dataloader, loss_fn, accuracy_fn, device)
logger.info(f"✓ Model 1 results: {model_1_results}")

# Model 2: Convolutional Neural Network (TinyVGG)
logger.info("\n=== Building Model 2: CNN (TinyVGG) ===")

class FashionMNISTModelV2(nn.Module):
    """Model architecture that replicates the TinyVGG model from CNN explainer website"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,  # Calculate from output of conv blocks
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# Create model instance
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(
    input_shape=1,  # 1 color channel (grayscale)
    hidden_units=10,
    output_shape=len(class_names)
).to(device)
logger.info(f"✓ Model 2 (CNN) created and moved to {device}")

# Test with dummy data
logger.info("Testing CNN with dummy data...")
dummy_img = torch.rand(size=(1, 1, 28, 28)).to(device)
logger.info(f"Dummy input shape: {dummy_img.shape}")
dummy_output = model_2(dummy_img)
logger.info(f"Dummy output shape: {dummy_output.shape}")
logger.info("✓ CNN forward pass successful")

# Setup loss and optimizer for model 2
optimizer_2 = torch.optim.SGD(params=model_2.parameters(), lr=LEARNING_RATE)

# Train Model 2
logger.info("\n=== Training Model 2 (CNN) ===")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

train_time_start_model_2 = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    train_step(model_2, train_dataloader, loss_fn, optimizer_2, accuracy_fn, device)
    test_step(model_2, test_dataloader, loss_fn, accuracy_fn, device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(
    start=train_time_start_model_2,
    end=train_time_end_model_2,
    device=device
)

model_2_results = eval_model_gpu(model_2, test_dataloader, loss_fn, accuracy_fn, device)
logger.info(f"✓ Model 2 results: {model_2_results}")

# Compare results
logger.info("\n=== Comparing All Models ===")
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results["training_time"] = [total_train_time_model_0, 
                                    total_train_time_model_1, 
                                    total_train_time_model_2]
logger.info("\nModel Comparison:")
print(compare_results)

# Save comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
compare_results.set_index("model_name")["model_acc"].plot(kind="bar", ax=ax)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Comparison - Fashion MNIST")
ax.set_ylim([0, 100])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("model_comparison.png")
logger.info("✓ Saved model comparison plot")
plt.close()

# Make predictions and visualize
logger.info("\n=== Making Predictions with Best Model ===")

def make_predictions(model, data, device):
    """Make predictions on data using model"""
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

# Get random test samples
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# Make predictions
logger.info("Making predictions on random test samples...")
pred_probs = make_predictions(model=model_2, data=test_samples, device=device)
pred_classes = pred_probs.argmax(dim=1)

# Plot predictions
logger.info("Plotting predictions...")
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"
    
    # Make title green if correct, red if wrong
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c="g")
    else:
        plt.title(title_text, fontsize=10, c="r")
    plt.axis(False)

plt.tight_layout()
plt.savefig("predictions_visualization.png")
logger.info("✓ Saved predictions visualization")
plt.close()

# Create confusion matrix
logger.info("\n=== Creating Confusion Matrix ===")
logger.info("Making predictions on entire test set...")

# Make predictions on entire test dataset
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        X, y = X.to(device), y.to(device)
        y_logit = model_2(X)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

# Concatenate predictions
y_pred_tensor = torch.cat(y_preds)
logger.info(f"Total predictions made: {len(y_pred_tensor)}")

# Create confusion matrix
logger.info("Creating confusion matrix...")
confmat = torchmetrics.ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

# Plot confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)
plt.savefig("confusion_matrix.png")
logger.info("✓ Saved confusion matrix")
plt.close()

# Save the best model
logger.info("\n=== Saving Best Model ===")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "fashion_mnist_tinyvgg.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

logger.info(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)
logger.info(f"✓ Model saved successfully! Size: {MODEL_SAVE_PATH.stat().st_size/1024:.2f} KB")

# Summary
logger.info("\n=== Training Complete! ===")
logger.info("\nFinal Results Summary:")
print(compare_results.to_string())
logger.info(f"\nBest performing model: {compare_results.iloc[compare_results['model_acc'].idxmax()]['model_name']}")
logger.info(f"Best accuracy: {compare_results['model_acc'].max():.2f}%")
logger.info("\nAll visualizations saved to current directory")
logger.info("Model weights saved to ./models/fashion_mnist_tinyvgg.pth")
logger.info("\n✓ Script completed successfully!")

if __name__ == "__main__":
    logger.info(f"--- Starting {SCRIPT_NAME} ---")
    try:
        # The main logic of the script is already executed above.
        # This block is to ensure we log the end of the script.
        pass
    except Exception as e:
        logger.error(f"An error occurred in {SCRIPT_NAME}: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"--- Finished {SCRIPT_NAME} ---")
