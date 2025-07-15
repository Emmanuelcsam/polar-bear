#!/usr/bin/env python3
"""
Convolutional Neural Network Implementation
Based on the tutorial transcript with automatic dependency management
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Function to install packages
def install_package(package_name, import_name=None):
    """Install a package using pip if not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ {package_name} is already installed")
        return True
    except ImportError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠ {package_name} not found. Installing latest version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ Failed to install {package_name}")
            return False

# Check and install required packages
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting dependency check...")

required_packages = [
    ("numpy", "numpy"),
    ("tensorflow", "tensorflow"),
    ("matplotlib", "matplotlib"),
    ("pillow", "PIL")
]

all_installed = True
for package, import_name in required_packages:
    if not install_package(package, import_name):
        all_installed = False

if not all_installed:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ Some packages failed to install. Exiting...")
    sys.exit(1)

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ All dependencies satisfied. Loading libraries...")

# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Libraries loaded successfully")


class CNNFromTutorial:
    """
    Implementation of the CNN architecture from the tutorial:
    - Input: 28x28x1 grayscale image
    - Layer 1: Conv(2 filters, 5x5) + ReLU + MaxPool(2x2, stride 2)
    - Layer 2: Conv(4 filters, 3x3) + Sigmoid + MaxPool(2x2, stride 2)
    - Flatten + Fully Connected (10 outputs)
    - Reported accuracy: 95.4%
    """
    
    def __init__(self):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing CNN model from tutorial...")
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the exact model architecture from the tutorial"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Building model architecture...")
        
        # Create Sequential model
        self.model = keras.Sequential()
        
        # Input layer specification
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding input layer (28x28x1)...")
        
        # Layer 1: First Convolution
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Layer 1 - Convolution...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 2 filters, 5x5 kernel size, stride 1, ReLU activation")
        self.model.add(layers.Conv2D(
            filters=2,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation='relu',
            input_shape=(28, 28, 1),
            padding='valid',
            name='conv1'
        ))
        
        # Layer 1: Max Pooling
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Layer 1 - Max Pooling...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Pool size 2x2, stride 2")
        self.model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name='maxpool1'
        ))
        
        # Layer 2: Second Convolution
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Layer 2 - Convolution...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 4 filters, 3x3 kernel size, stride 1, Sigmoid activation")
        self.model.add(layers.Conv2D(
            filters=4,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='sigmoid',
            padding='valid',
            name='conv2'
        ))
        
        # Layer 2: Max Pooling
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Layer 2 - Max Pooling...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Pool size 2x2, stride 2")
        self.model.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name='maxpool2'
        ))
        
        # Flatten
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Flatten layer...")
        self.model.add(layers.Flatten(name='flatten'))
        
        # Fully Connected Output Layer
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adding Fully Connected layer...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 10 output nodes (for 10 classes)")
        self.model.add(layers.Dense(
            units=10,
            activation=None,  # No activation as mentioned in tutorial
            name='output'
        ))
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Model architecture built successfully")
        
    def compile_model(self):
        """Compile the model with appropriate settings"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Compiling model...")
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Model compiled with Adam optimizer")
        
    def summary(self):
        """Display model summary"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model Summary:")
        self.model.summary()
        
    def visualize_activations(self, image):
        """Visualize the forward propagation through each layer"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Visualizing forward propagation...")
        
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, 28, 28, 1)
            
        # Create models for each layer output
        layer_outputs = []
        layer_names = []
        
        for layer in self.model.layers:
            if 'conv' in layer.name or 'maxpool' in layer.name:
                intermediate_model = keras.Model(inputs=self.model.input, outputs=layer.output)
                layer_outputs.append(intermediate_model.predict(image, verbose=0))
                layer_names.append(layer.name)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - Processed {layer.name}")
        
        return layer_outputs, layer_names
    
    def demonstrate_convolution_math(self, image, layer_idx=0):
        """Demonstrate the mathematical operations in convolution"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Demonstrating convolution mathematics...")
        
        if self.model is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ Model not built yet!")
            return
        
        # Get the first convolutional layer
        conv_layer = self.model.layers[layer_idx]
        if not isinstance(conv_layer, layers.Conv2D):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ Layer {layer_idx} is not a Conv2D layer!")
            return
            
        # Get weights and bias
        weights, bias = conv_layer.get_weights()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Filter dimensions: {weights.shape}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bias shape: {bias.shape}")
        
        # Show example calculation for position (13, 13) as mentioned in tutorial
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Example calculation at position (13, 13):")
        
        # Extract a 5x5 patch (for first conv layer)
        if image.shape[0] >= 18 and image.shape[1] >= 18:
            patch = image[13:18, 13:18]
            filter_0 = weights[:, :, 0, 0]
            
            # Element-wise multiplication
            multiplied = patch * filter_0
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Patch * Filter element-wise multiplication:")
            print(multiplied)
            
            # Sum and add bias
            z_value = np.sum(multiplied) + bias[0]
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Z value (sum + bias): {z_value}")
            
            # Apply ReLU
            activation = max(0, z_value)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] After ReLU activation: {activation}")
        
    def load_pretrained_weights(self):
        """Simulate loading pre-trained weights (as mentioned in tutorial: 95.4% accuracy)"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Note: Tutorial mentions pre-trained model with 95.4% accuracy")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] In practice, you would load weights here")
        
    def predict_single_image(self, image):
        """Predict a single image"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Making prediction...")
        
        # Ensure correct shape
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        
        # Normalize if needed
        if image.max() > 1:
            image = image / 255.0
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Normalized image to [0, 1] range")
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Raw predictions: {predictions[0]}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Predicted class: {predicted_class}")
        
        return predicted_class, predictions[0]


def create_sample_image():
    """Create a sample 28x28 grayscale image for testing"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating sample 28x28 grayscale image...")
    
    # Create a simple pattern or load MNIST sample
    try:
        # Try to load MNIST for a real example
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Loaded MNIST dataset")
        
        # Get a sample that should be "7" as mentioned in tutorial
        seven_indices = np.where(y_test == 7)[0]
        if len(seven_indices) > 0:
            sample_image = x_test[seven_indices[0]]
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Selected a '7' from MNIST dataset")
        else:
            sample_image = x_test[0]
            
        return sample_image, y_test[seven_indices[0] if len(seven_indices) > 0 else 0]
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not load MNIST, creating synthetic image")
        # Create a simple synthetic "7"-like pattern
        image = np.zeros((28, 28))
        # Draw a simple 7
        image[5:8, 8:20] = 255  # Top horizontal line
        image[7:20, 17:20] = 255  # Vertical line
        return image, 7


def main():
    """Main execution function"""
    print(f"\n{'='*60}")
    print(f"CNN Tutorial Implementation")
    print(f"Based on forward propagation visualization tutorial")
    print(f"{'='*60}\n")
    
    # Create CNN instance
    cnn = CNNFromTutorial()
    
    # Build model
    cnn.build_model()
    
    # Compile model
    cnn.compile_model()
    
    # Show model summary
    cnn.summary()
    
    # Create or load sample image
    sample_image, true_label = create_sample_image()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] True label: {true_label}")
    
    # Visualize the image
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Displaying input image...")
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image, cmap='gray')
    plt.title(f'Input Image (True Label: {true_label})')
    plt.colorbar()
    plt.show()
    
    # Initialize with random weights (since we can't load the exact pre-trained weights)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Note: Using random initialization (not the 95.4% accurate pre-trained weights)")
    
    # Demonstrate convolution mathematics
    cnn.demonstrate_convolution_math(sample_image)
    
    # Visualize activations through layers
    layer_outputs, layer_names = cnn.visualize_activations(sample_image)
    
    # Plot activations
    if len(layer_outputs) > 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Plotting layer activations...")
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        plot_idx = 0
        for i, (output, name) in enumerate(zip(layer_outputs, layer_names)):
            if 'conv' in name:
                # Plot all filters for conv layers
                for j in range(output.shape[-1]):
                    if plot_idx < 8:
                        axes[plot_idx].imshow(output[0, :, :, j], cmap='gray')
                        axes[plot_idx].set_title(f'{name} - Filter {j}')
                        axes[plot_idx].axis('off')
                        plot_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    # Make prediction
    predicted_class, predictions = cnn.predict_single_image(sample_image)
    
    # Visualize prediction
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), predictions)
    plt.xlabel('Class')
    plt.ylabel('Output Value')
    plt.title(f'Model Output (Predicted: {predicted_class}, True: {true_label})')
    plt.xticks(range(10))
    plt.show()
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Tutorial implementation complete!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] The model predicted: {predicted_class}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tutorial mentioned prediction: 7")
    
    print(f"\n{'='*60}")
    print(f"Tutorial concepts demonstrated:")
    print(f"- 28x28x1 grayscale input")
    print(f"- Layer 1: Conv(2 filters, 5x5) + ReLU + MaxPool(2x2)")
    print(f"- Layer 2: Conv(4 filters, 3x3) + Sigmoid + MaxPool(2x2)")
    print(f"- Flatten to 100 nodes")
    print(f"- Fully connected to 10 outputs")
    print(f"- Forward propagation visualization")
    print(f"- Convolution mathematics")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
