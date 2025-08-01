#!/usr/bin/env python3
"""
Neural Network from Scratch
Based on tutorial transcript - No ML libraries, only NumPy!
"""

import sys
import subprocess
import importlib
import os
import time
from datetime import datetime

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, install if not"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking for {package_name}...")
    try:
        importlib.import_module(import_name)
        log(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        log(f"✗ {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            log(f"✗ Failed to install {package_name}")
            return False

# Check and install required packages
log("=== CHECKING DEPENDENCIES ===")
required_packages = [
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("matplotlib", "matplotlib"),
    ("gzip", "gzip"),
    ("pickle", "pickle"),
]

for package, import_name in required_packages:
    if not check_and_install_package(package, import_name):
        log("Failed to install required packages. Exiting.")
        sys.exit(1)

# Now import everything
log("=== IMPORTING MODULES ===")
import numpy as np
import requests
import gzip
import pickle
import matplotlib.pyplot as plt
from typing import List, Tuple
import urllib.request

log("✓ All modules imported successfully")

class NeuralNetwork:
    """Neural Network from scratch - no ML libraries!"""
    
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize neural network with given layer sizes
        layer_sizes: [input_size, hidden1_size, hidden2_size, ..., output_size]
        """
        log(f"Initializing Neural Network with architecture: {layer_sizes}")
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases randomly
        log("Initializing weights and biases...")
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # He initialization for better convergence
            w = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i-1])
            b = np.zeros((1, layer_sizes[i]))
            self.weights.append(w)
            self.biases.append(b)
            log(f"  Layer {i}: Weight shape {w.shape}, Bias shape {b.shape}")
        
        # Store activations for backprop
        self.activations = []
        self.z_values = []
        
        log("✓ Neural Network initialized")
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward pass through the network
        X: input data (batch_size, input_dim)
        """
        self.activations = [X]
        self.z_values = []
        
        activation = X
        
        # Forward through all layers except the last
        for i in range(self.num_layers - 2):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.relu(z)
            self.activations.append(activation)
        
        # Last layer (no ReLU, just linear)
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        # Apply softmax for classification
        output = self.softmax(z)
        self.activations.append(output)
        
        return output
    
    def cross_entropy_loss(self, y_pred, y_true):
        """Calculate cross-entropy loss"""
        n_samples = y_true.shape[0]
        # Clip to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.sum(y_true * np.log(y_pred)) / n_samples
        return loss
    
    def backward(self, X, y_true, learning_rate=0.01):
        """
        Backward pass - the magic of learning!
        Uses backpropagation to update weights
        """
        m = X.shape[0]  # batch size
        
        # Start with the output layer
        y_pred = self.activations[-1]
        
        # Gradient of loss w.r.t output
        delta = y_pred - y_true
        
        # Backpropagate through layers
        for i in reversed(range(self.num_layers - 1)):
            # Gradient for weights and biases
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Prepare for next layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, learning_rate=0.01):
        """Train the neural network"""
        log(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        
        n_samples = X_train.shape[0]
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(batch_X)
                
                # Calculate loss
                loss = self.cross_entropy_loss(y_pred, batch_y)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(batch_X, batch_y, learning_rate)
            
            # Calculate average epoch loss
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            # Evaluate on test set
            test_acc = self.evaluate(X_test, y_test)
            test_accuracies.append(test_acc)
            
            log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_acc:.2f}%")
        
        return train_losses, test_accuracies
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate accuracy"""
        predictions = self.predict(X)
        y_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y_labels) * 100
        return accuracy

def download_mnist():
    """Download and prepare MNIST dataset"""
    log("=== DOWNLOADING MNIST DATASET ===")
    
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    data = {}
    
    for key, filename in files.items():
        filepath = f"mnist_{filename}"
        if not os.path.exists(filepath):
            log(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            log(f"✓ Downloaded {filename}")
        else:
            log(f"✓ {filename} already exists")
        
        # Read the data
        log(f"Reading {filename}...")
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                # Skip header and read images
                f.read(16)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
            else:
                # Skip header and read labels
                f.read(8)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    
    log("✓ MNIST dataset loaded successfully")
    return data

def download_fashion_mnist():
    """Download and prepare Fashion-MNIST dataset"""
    log("=== DOWNLOADING FASHION-MNIST DATASET ===")
    
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    data = {}
    
    for key, filename in files.items():
        filepath = f"fashion_{filename}"
        if not os.path.exists(filepath):
            log(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
            log(f"✓ Downloaded {filename}")
        else:
            log(f"✓ {filename} already exists")
        
        # Read the data
        log(f"Reading {filename}...")
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                # Skip header and read images
                f.read(16)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28*28)
            else:
                # Skip header and read labels
                f.read(8)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    
    log("✓ Fashion-MNIST dataset loaded successfully")
    return data

def prepare_data(images, labels):
    """Normalize images and one-hot encode labels"""
    log("Preparing data...")
    
    # Normalize pixel values to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # One-hot encode labels
    n_classes = 10
    one_hot = np.zeros((labels.shape[0], n_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    
    log(f"✓ Data prepared: {images.shape[0]} samples")
    return images, one_hot

def visualize_predictions(model, X_test, y_test, n_samples=10):
    """Visualize some predictions"""
    log("Visualizing predictions...")
    
    # Random samples
    indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Get prediction
        pred = model.predict(X_test[idx:idx+1])[0]
        true_label = np.argmax(y_test[idx])
        
        # Reshape to 28x28 for display
        img = X_test[idx].reshape(28, 28)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Pred: {pred}, True: {true_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    log("✓ Predictions saved to predictions.png")
    plt.close()

def plot_training_history(train_losses, test_accuracies, dataset_name):
    """Plot training history"""
    log(f"Plotting training history for {dataset_name}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses)
    ax1.set_title(f'{dataset_name} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(test_accuracies)
    ax2.set_title(f'{dataset_name} - Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    filename = f'{dataset_name.lower()}_training_history.png'
    plt.savefig(filename)
    log(f"✓ Training history saved to {filename}")
    plt.close()

def main():
    """Main function - build, train, and test the neural network"""
    log("=== NEURAL NETWORK FROM SCRATCH ===")
    log("Following the tutorial to build a NN with only NumPy!")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    log("✓ Random seed set to 42")
    
    # MNIST Dataset
    log("\n=== TRAINING ON MNIST ===")
    mnist_data = download_mnist()
    
    # Prepare MNIST data
    X_train, y_train = prepare_data(mnist_data['train_images'], mnist_data['train_labels'])
    X_test, y_test = prepare_data(mnist_data['test_images'], mnist_data['test_labels'])
    
    # Create neural network for MNIST
    # Architecture from tutorial: 784 -> hidden layers -> 10
    mnist_nn = NeuralNetwork([784, 128, 64, 10])
    
    # Train on MNIST
    log("Training on MNIST dataset...")
    train_losses, test_accs = mnist_nn.train(
        X_train, y_train, X_test, y_test,
        epochs=20,
        batch_size=64,
        learning_rate=0.1
    )
    
    # Final evaluation
    final_accuracy = mnist_nn.evaluate(X_test, y_test)
    log(f"✓ Final MNIST Test Accuracy: {final_accuracy:.2f}%")
    
    # Visualize some predictions
    visualize_predictions(mnist_nn, X_test, y_test)
    plot_training_history(train_losses, test_accs, "MNIST")
    
    # Fashion-MNIST Dataset
    log("\n=== TRAINING ON FASHION-MNIST ===")
    fashion_data = download_fashion_mnist()
    
    # Prepare Fashion-MNIST data
    X_train_fashion, y_train_fashion = prepare_data(fashion_data['train_images'], fashion_data['train_labels'])
    X_test_fashion, y_test_fashion = prepare_data(fashion_data['test_images'], fashion_data['test_labels'])
    
    # Create neural network for Fashion-MNIST
    fashion_nn = NeuralNetwork([784, 128, 64, 10])
    
    # Train on Fashion-MNIST
    log("Training on Fashion-MNIST dataset...")
    train_losses_fashion, test_accs_fashion = fashion_nn.train(
        X_train_fashion, y_train_fashion, X_test_fashion, y_test_fashion,
        epochs=20,
        batch_size=64,
        learning_rate=0.1
    )
    
    # Final evaluation
    final_accuracy_fashion = fashion_nn.evaluate(X_test_fashion, y_test_fashion)
    log(f"✓ Final Fashion-MNIST Test Accuracy: {final_accuracy_fashion:.2f}%")
    
    # Visualize Fashion-MNIST predictions
    visualize_predictions(fashion_nn, X_test_fashion, y_test_fashion)
    plot_training_history(train_losses_fashion, test_accs_fashion, "Fashion-MNIST")
    
    # Test individual predictions (like in the tutorial)
    log("\n=== TESTING INDIVIDUAL PREDICTIONS ===")
    
    # Test a few random MNIST samples
    log("Testing MNIST predictions...")
    for i in range(3):
        idx = np.random.randint(0, X_test.shape[0])
        pred = mnist_nn.predict(X_test[idx:idx+1])[0]
        true_label = np.argmax(y_test[idx])
        
        # Get confidence
        probs = mnist_nn.forward(X_test[idx:idx+1])[0]
        confidence = probs[pred] * 100
        
        log(f"  Sample {i+1}: Predicted {pred} (confidence: {confidence:.2f}%), True label: {true_label}")
    
    # Fashion-MNIST class names
    fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    log("\nTesting Fashion-MNIST predictions...")
    for i in range(3):
        idx = np.random.randint(0, X_test_fashion.shape[0])
        pred = fashion_nn.predict(X_test_fashion[idx:idx+1])[0]
        true_label = np.argmax(y_test_fashion[idx])
        
        # Get confidence
        probs = fashion_nn.forward(X_test_fashion[idx:idx+1])[0]
        confidence = probs[pred] * 100
        
        log(f"  Sample {i+1}: Predicted {fashion_classes[pred]} (confidence: {confidence:.2f}%), True: {fashion_classes[true_label]}")
    
    log("\n=== NEURAL NETWORK TRAINING COMPLETE ===")
    log(f"MNIST Accuracy: {final_accuracy:.2f}%")
    log(f"Fashion-MNIST Accuracy: {final_accuracy_fashion:.2f}%")
    log("Check the generated PNG files for visualizations!")
    log("Just like the tutorial - we built it from scratch with only NumPy! 🎉")

if __name__ == "__main__":
    main()
