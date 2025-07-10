#!/usr/bin/env python3
"""
Convolutional Neural Network from Scratch
Based on tutorial transcript - implements CNN with automatic dependency management
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Create a logger function
def log(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

log("Starting Convolutional Neural Network from Scratch implementation")

# Define required packages
required_packages = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'tensorflow': 'tensorflow',  # For keras utilities only
}

# Function to install packages
def install_package(package_name):
    """Install a package using pip"""
    log(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        log(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        log(f"ERROR: Failed to install {package_name}")
        return False

# Check and install required packages
log("Checking required dependencies...")
for import_name, package_name in required_packages.items():
    try:
        importlib.import_module(import_name)
        log(f"✓ {import_name} is already installed")
    except ImportError:
        log(f"✗ {import_name} is not installed")
        if not install_package(package_name):
            log(f"FATAL: Cannot proceed without {package_name}")
            sys.exit(1)

# Now import all required modules
log("Importing required modules...")
import numpy as np
from scipy import signal
from tensorflow import keras
log("All modules imported successfully")

# Set random seed for reproducibility
np.random.seed(42)
log("Set random seed to 42 for reproducibility")

class Layer:
    """Base Layer class that all layers inherit from"""
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        """Forward propagation - to be implemented by child classes"""
        raise NotImplementedError
        
    def backward(self, output_gradient, learning_rate):
        """Backward propagation - to be implemented by child classes"""
        raise NotImplementedError

log("Created base Layer class")

class ConvolutionalLayer(Layer):
    """Convolutional Layer implementation from scratch"""
    def __init__(self, input_shape, kernel_size, depth):
        """
        Initialize convolutional layer
        
        Args:
            input_shape: tuple (depth, height, width) of input
            kernel_size: integer size of square kernel
            depth: number of kernels (output depth)
        """
        super().__init__()
        log(f"Initializing ConvolutionalLayer with input_shape={input_shape}, kernel_size={kernel_size}, depth={depth}")
        
        # Unpack input shape
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        
        # Compute output shape using formula: input_size - kernel_size + 1
        self.output_shape = (
            depth,
            self.input_height - kernel_size + 1,
            self.input_width - kernel_size + 1
        )
        log(f"Computed output shape: {self.output_shape}")
        
        # Initialize kernels with shape (depth, input_depth, kernel_size, kernel_size)
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        log(f"Initialized kernels with shape {self.kernels_shape}")
        
        # Initialize biases with shape same as output
        self.biases = np.random.randn(*self.output_shape) * 0.1
        log(f"Initialized biases with shape {self.output_shape}")
        
    def forward(self, input_data):
        """Forward propagation through convolutional layer"""
        log("ConvolutionalLayer forward propagation started")
        self.input = input_data
        
        # Start with biases (copy to avoid modifying original)
        self.output = np.copy(self.biases)
        
        # For each kernel
        for i in range(self.depth):
            # For each channel in input
            for j in range(self.input_depth):
                # Compute cross-correlation and add to output
                self.output[i] += signal.correlate2d(
                    self.input[j], 
                    self.kernels[i, j], 
                    mode='valid'
                )
        
        log(f"ConvolutionalLayer forward propagation completed, output shape: {self.output.shape}")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward propagation through convolutional layer"""
        log("ConvolutionalLayer backward propagation started")
        
        # Initialize gradients
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input.shape)
        
        # Compute kernel gradients
        log("Computing kernel gradients...")
        for i in range(self.depth):
            for j in range(self.input_depth):
                # Kernel gradient = input ⋆ output_gradient
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j],
                    output_gradient[i],
                    mode='valid'
                )
        
        # Compute input gradient
        log("Computing input gradients...")
        for j in range(self.input_depth):
            for i in range(self.depth):
                # Input gradient = output_gradient (full convolution) kernel
                # Full convolution = full correlation with 180° rotated kernel
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i],
                    self.kernels[i, j],
                    mode='full'
                )
        
        # Update parameters using gradient descent
        log("Updating kernels and biases...")
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        
        log("ConvolutionalLayer backward propagation completed")
        return input_gradient

log("Created ConvolutionalLayer class")

class ReshapeLayer(Layer):
    """Reshape layer to transform data between different shapes"""
    def __init__(self, input_shape, output_shape):
        """Initialize reshape layer with input and output shapes"""
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        log(f"Initialized ReshapeLayer: {input_shape} -> {output_shape}")
        
    def forward(self, input_data):
        """Reshape input to output shape"""
        log(f"ReshapeLayer forward: reshaping from {input_data.shape} to {self.output_shape}")
        return np.reshape(input_data, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        """Reshape gradient back to input shape"""
        log(f"ReshapeLayer backward: reshaping gradient from {output_gradient.shape} to {self.input_shape}")
        return np.reshape(output_gradient, self.input_shape)

log("Created ReshapeLayer class")

class Activation(Layer):
    """Base activation class"""
    def __init__(self, activation, activation_prime):
        """Initialize with activation function and its derivative"""
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input_data):
        """Apply activation function"""
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Apply derivative of activation function"""
        return output_gradient * self.activation_prime(self.input)

def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

class SigmoidActivation(Activation):
    """Sigmoid activation layer"""
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)
        log("Initialized SigmoidActivation layer")

log("Created Activation classes")

class DenseLayer(Layer):
    """Fully connected (dense) layer"""
    def __init__(self, input_size, output_size):
        """Initialize dense layer with random weights and biases"""
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size, 1) * 0.1
        log(f"Initialized DenseLayer: {input_size} -> {output_size}")
        
    def forward(self, input_data):
        """Forward propagation: output = weights @ input + bias"""
        log(f"DenseLayer forward propagation, input shape: {input_data.shape}")
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.biases
        log(f"DenseLayer forward completed, output shape: {self.output.shape}")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """Backward propagation with parameter updates"""
        log("DenseLayer backward propagation started")
        
        # Compute gradients
        weights_gradient = np.dot(output_gradient, self.input.T)
        biases_gradient = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        log("DenseLayer backward propagation completed")
        return input_gradient

log("Created DenseLayer class")

def binary_cross_entropy(y_true, y_pred):
    """Binary cross entropy loss function"""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    """Derivative of binary cross entropy loss"""
    # Add small epsilon to prevent division by zero
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

log("Created loss functions")

def preprocess_data(x, y, limit):
    """
    Preprocess MNIST data:
    - Limit to specified classes (0 and 1)
    - Normalize images
    - One-hot encode labels
    """
    log(f"Preprocessing data, limiting to classes: {limit}")
    
    # Get indices for each class
    indices = []
    for class_num in limit:
        class_indices = np.where(y == class_num)[0]
        indices.extend(class_indices)
        log(f"Found {len(class_indices)} samples of class {class_num}")
    
    # Shuffle indices
    np.random.shuffle(indices)
    indices = indices[:1000]  # Limit samples for faster training
    log(f"Using {len(indices)} total samples")
    
    # Extract limited data
    x = x[indices]
    y = y[indices]
    
    # Reshape images from (28, 28) to (1, 28, 28) for convolutional layer
    x = x.reshape(len(x), 1, 28, 28)
    
    # Normalize pixel values to [0, 1]
    x = x.astype("float32") / 255
    log("Normalized pixel values to range [0, 1]")
    
    # One-hot encode labels
    y_encoded = np.zeros((len(y), len(limit), 1))
    for idx, label in enumerate(y):
        class_idx = limit.index(label)
        y_encoded[idx][class_idx] = 1
    
    log("One-hot encoded labels")
    return x, y_encoded

# Build the neural network
def build_network():
    """Build the convolutional neural network architecture"""
    log("Building neural network architecture...")
    
    network = [
        # Convolutional layer: 5 kernels of size 3x3
        ConvolutionalLayer(input_shape=(1, 28, 28), kernel_size=3, depth=5),
        SigmoidActivation(),
        
        # Reshape from (5, 26, 26) to column vector
        ReshapeLayer(input_shape=(5, 26, 26), output_shape=(5 * 26 * 26, 1)),
        
        # Dense layer: 3380 -> 100
        DenseLayer(5 * 26 * 26, 100),
        SigmoidActivation(),
        
        # Output layer: 100 -> 2
        DenseLayer(100, 2),
        SigmoidActivation()
    ]
    
    log("Neural network architecture built successfully")
    return network

# Training function
def train_network(network, x_train, y_train, x_test, y_test, epochs=20, learning_rate=0.1):
    """Train the neural network"""
    log(f"Starting training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        error = 0
        log(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Training loop
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            # Forward propagation
            output = x
            for layer in network:
                output = layer.forward(output)
            
            # Compute error
            error += binary_cross_entropy(y, output)
            
            # Backward propagation
            gradient = binary_cross_entropy_prime(y, output)
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)
            
            # Log progress every 50 samples
            if (i + 1) % 50 == 0:
                log(f"Processed {i + 1}/{len(x_train)} samples")
        
        # Compute average error
        error /= len(x_train)
        log(f"Average training error: {error:.6f}")
        
        # Test on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            log("Evaluating on test set...")
            correct = 0
            for x, y in zip(x_test, y_test):
                output = x
                for layer in network:
                    output = layer.forward(output)
                
                if np.argmax(output) == np.argmax(y):
                    correct += 1
            
            accuracy = correct / len(x_test) * 100
            log(f"Test accuracy: {accuracy:.2f}%")

# Main execution
def main():
    """Main function to run the CNN from scratch"""
    log("\n=== Starting MNIST Classification with CNN from Scratch ===")
    
    # Load MNIST dataset
    log("Loading MNIST dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        log(f"Loaded training data: {x_train.shape}, {y_train.shape}")
        log(f"Loaded test data: {x_test.shape}, {y_test.shape}")
    except Exception as e:
        log(f"ERROR: Failed to load MNIST dataset: {e}")
        return
    
    # Preprocess data (limit to digits 0 and 1)
    log("Preprocessing data for binary classification (0 vs 1)...")
    x_train, y_train = preprocess_data(x_train, y_train, [0, 1])
    x_test, y_test = preprocess_data(x_test, y_test, [0, 1])
    
    log(f"Training data shape: {x_train.shape}")
    log(f"Test data shape: {x_test.shape}")
    
    # Build network
    network = build_network()
    
    # Train network
    train_network(network, x_train, y_train, x_test, y_test, epochs=20, learning_rate=0.1)
    
    log("\n=== Training completed! ===")
    
    # Final evaluation
    log("\nPerforming final evaluation on test set...")
    correct = 0
    for i, (x, y) in enumerate(zip(x_test[:100], y_test[:100])):  # Test on 100 samples
        output = x
        for layer in network:
            output = layer.forward(output)
        
        predicted = np.argmax(output)
        actual = np.argmax(y)
        
        if predicted == actual:
            correct += 1
        
        if i < 5:  # Show first 5 predictions
            log(f"Sample {i + 1}: Predicted={predicted}, Actual={actual}, Output={output.flatten()}")
    
    final_accuracy = correct / 100 * 100
    log(f"\nFinal test accuracy (100 samples): {final_accuracy:.2f}%")
    
    log("\n=== CNN from Scratch implementation completed successfully! ===")

if __name__ == "__main__":
    main()
