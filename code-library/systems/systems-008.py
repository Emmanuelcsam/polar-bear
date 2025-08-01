#!/usr/bin/env python3
"""
Convolutional Neural Network (CNN) Implementation based on VGG-16 Architecture
Based on LearnOpenCV tutorial transcript
Features:
- Automatic dependency detection and installation
- Detailed logging of all operations
- Implementation of VGG-16 architecture concepts
- Demonstration of convolution operations and feature extraction
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if level == "INFO":
        print(f"{Colors.BLUE}[{timestamp}] INFO: {message}{Colors.ENDC}")
    elif level == "SUCCESS":
        print(f"{Colors.GREEN}[{timestamp}] SUCCESS: {message}{Colors.ENDC}")
    elif level == "WARNING":
        print(f"{Colors.WARNING}[{timestamp}] WARNING: {message}{Colors.ENDC}")
    elif level == "ERROR":
        print(f"{Colors.FAIL}[{timestamp}] ERROR: {message}{Colors.ENDC}")
    elif level == "HEADER":
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
        print(f"[{timestamp}] {message}")
        print(f"{'='*60}{Colors.ENDC}\n")

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking for package: {package_name}")
    
    try:
        importlib.import_module(import_name)
        log(f"Package {package_name} is already installed", "SUCCESS")
        return True
    except ImportError:
        log(f"Package {package_name} not found. Installing...", "WARNING")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"Successfully installed {package_name}", "SUCCESS")
            return True
        except subprocess.CalledProcessError:
            log(f"Failed to install {package_name}", "ERROR")
            return False

def setup_environment():
    """Setup all required packages"""
    log("SETTING UP ENVIRONMENT", "HEADER")
    
    # Define required packages
    required_packages = [
        ("numpy", "numpy"),
        ("tensorflow", "tensorflow"),
        ("matplotlib", "matplotlib"),
        ("opencv-python", "cv2"),
        ("scikit-learn", "sklearn"),
        ("pillow", "PIL"),
        ("tqdm", "tqdm")
    ]
    
    # Check Python version
    log(f"Python version: {sys.version}")
    
    # Install packages
    for package, import_name in required_packages:
        if not check_and_install_package(package, import_name):
            log(f"Setup failed. Please install {package} manually.", "ERROR")
            sys.exit(1)
    
    log("All dependencies installed successfully!", "SUCCESS")

# Run setup before importing the packages
setup_environment()

# Now import all required packages
log("IMPORTING REQUIRED PACKAGES", "HEADER")

try:
    import numpy as np
    log("Imported numpy", "SUCCESS")
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    log(f"Imported tensorflow (version: {tf.__version__})", "SUCCESS")
    
    import matplotlib.pyplot as plt
    log("Imported matplotlib", "SUCCESS")
    
    import cv2
    log(f"Imported OpenCV (version: {cv2.__version__})", "SUCCESS")
    
    from sklearn.model_selection import train_test_split
    log("Imported scikit-learn", "SUCCESS")
    
    from PIL import Image
    log("Imported PIL", "SUCCESS")
    
    from tqdm import tqdm
    log("Imported tqdm", "SUCCESS")
    
except Exception as e:
    log(f"Failed to import packages: {e}", "ERROR")
    sys.exit(1)

class VGG16Implementation:
    """Implementation of VGG-16 architecture as described in the tutorial"""
    
    def __init__(self):
        log("INITIALIZING VGG-16 IMPLEMENTATION", "HEADER")
        self.input_shape = (224, 224, 3)
        self.num_classes = 1000  # ImageNet classes
        self.model = None
        
    def demonstrate_convolution_operation(self):
        """Demonstrate convolution operation with Sobel kernel as mentioned in tutorial"""
        log("DEMONSTRATING CONVOLUTION OPERATION", "HEADER")
        
        # Create a sample 6x6 input as mentioned in tutorial
        log("Creating 6x6 sample input")
        sample_input = np.array([
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0],
            [10, 10, 10, 0, 0, 0]
        ], dtype=np.float32)
        
        # Sobel kernel for vertical edge detection (3x3)
        log("Creating Sobel kernel for vertical edge detection")
        sobel_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        
        log("Sample input:")
        print(sample_input)
        log("Sobel kernel:")
        print(sobel_kernel)
        
        # Perform convolution with stride=1
        log("Performing convolution operation with stride=1")
        output_size = 4  # (6-3+1) = 4
        output = np.zeros((output_size, output_size))
        
        for i in range(output_size):
            for j in range(output_size):
                # Extract receptive field
                receptive_field = sample_input[i:i+3, j:j+3]
                # Perform element-wise multiplication and sum (dot product)
                conv_result = np.sum(receptive_field * sobel_kernel)
                output[i, j] = conv_result
                log(f"Position ({i},{j}): Convolution result = {conv_result:.2f}")
        
        log("Convolution output:")
        print(output)
        
        # Visualize the convolution
        self._visualize_convolution(sample_input, sobel_kernel, output)
        
    def _visualize_convolution(self, input_data, kernel, output):
        """Visualize convolution operation"""
        log("Visualizing convolution operation")
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input
        axes[0].imshow(input_data, cmap='gray')
        axes[0].set_title('Input (6x6)')
        axes[0].grid(True)
        
        # Kernel
        axes[1].imshow(kernel, cmap='RdBu')
        axes[1].set_title('Sobel Kernel (3x3)')
        axes[1].grid(True)
        
        # Output
        axes[2].imshow(output, cmap='gray')
        axes[2].set_title('Output (4x4)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('convolution_demo.png')
        log("Saved convolution visualization to 'convolution_demo.png'", "SUCCESS")
        plt.close()
        
    def build_vgg16_model(self):
        """Build VGG-16 model architecture as described in tutorial"""
        log("BUILDING VGG-16 ARCHITECTURE", "HEADER")
        
        model = models.Sequential()
        
        # Input layer
        log(f"Adding input layer with shape: {self.input_shape}")
        
        # Conv Block 1 - 2 conv layers + max pool
        log("Building Convolutional Block 1")
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                              input_shape=self.input_shape, name='conv1_1'))
        log("Added Conv1_1: 64 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
        log("Added Conv1_2: 64 filters, 3x3 kernel")
        
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
        log("Added MaxPool1: 2x2 pool size, stride 2")
        
        # Conv Block 2 - 2 conv layers + max pool
        log("Building Convolutional Block 2")
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
        log("Added Conv2_1: 128 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
        log("Added Conv2_2: 128 filters, 3x3 kernel")
        
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))
        log("Added MaxPool2: 2x2 pool size, stride 2")
        
        # Conv Block 3 - 3 conv layers + max pool
        log("Building Convolutional Block 3")
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
        log("Added Conv3_1: 256 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
        log("Added Conv3_2: 256 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
        log("Added Conv3_3: 256 filters, 3x3 kernel")
        
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))
        log("Added MaxPool3: 2x2 pool size, stride 2")
        
        # Conv Block 4 - 3 conv layers + max pool
        log("Building Convolutional Block 4")
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
        log("Added Conv4_1: 512 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
        log("Added Conv4_2: 512 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
        log("Added Conv4_3: 512 filters, 3x3 kernel")
        
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))
        log("Added MaxPool4: 2x2 pool size, stride 2")
        
        # Conv Block 5 - 3 conv layers + max pool
        log("Building Convolutional Block 5")
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
        log("Added Conv5_1: 512 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
        log("Added Conv5_2: 512 filters, 3x3 kernel")
        
        model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
        log("Added Conv5_3: 512 filters, 3x3 kernel")
        
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))
        log("Added MaxPool5: 2x2 pool size, stride 2")
        
        # Flatten - Convert 7x7x512 to 25,088 as mentioned in tutorial
        log("Adding Flatten layer to convert activation maps to 1D vector")
        model.add(layers.Flatten(name='flatten'))
        
        # Fully Connected Layers (Classifier)
        log("Building Fully Connected Classifier")
        model.add(layers.Dense(4096, activation='relu', name='fc1'))
        log("Added FC1: 4096 neurons")
        
        model.add(layers.Dense(4096, activation='relu', name='fc2'))
        log("Added FC2: 4096 neurons")
        
        model.add(layers.Dense(self.num_classes, activation='softmax', name='predictions'))
        log(f"Added Output layer: {self.num_classes} neurons (softmax)")
        
        self.model = model
        log("VGG-16 model built successfully!", "SUCCESS")
        
        return model
    
    def visualize_architecture(self):
        """Visualize the model architecture"""
        log("VISUALIZING MODEL ARCHITECTURE", "HEADER")
        
        if self.model is None:
            log("Model not built yet. Building model first.", "WARNING")
            self.build_vgg16_model()
        
        # Print model summary
        log("Model Summary:")
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        log(f"Total parameters in VGG-16: {total_params:,}")
        
        # Visualize layer shapes
        log("Layer shapes throughout the network:")
        for layer in self.model.layers:
            if hasattr(layer, 'output_shape'):
                log(f"{layer.name}: {layer.output_shape}")
    
    def demonstrate_max_pooling(self):
        """Demonstrate max pooling operation as described in tutorial"""
        log("DEMONSTRATING MAX POOLING OPERATION", "HEADER")
        
        # Create 4x4 input as shown in tutorial
        log("Creating 4x4 sample input")
        sample_input = np.array([
            [1, 3, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4]
        ], dtype=np.float32)
        
        log("Sample input:")
        print(sample_input)
        
        # Perform max pooling with 2x2 filter and stride 2
        log("Performing max pooling with 2x2 filter and stride=2")
        pool_size = 2
        stride = 2
        output_size = 2  # (4-2)/2 + 1 = 2
        output = np.zeros((output_size, output_size))
        
        for i in range(0, 4, stride):
            for j in range(0, 4, stride):
                # Extract pooling window
                window = sample_input[i:i+pool_size, j:j+pool_size]
                # Take maximum value
                max_val = np.max(window)
                output[i//stride, j//stride] = max_val
                log(f"Window at ({i},{j}): max value = {max_val}")
        
        log("Max pooling output:")
        print(output)
        
        # Visualize
        self._visualize_pooling(sample_input, output)
    
    def _visualize_pooling(self, input_data, output):
        """Visualize pooling operation"""
        log("Visualizing max pooling operation")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Input
        im1 = axes[0].imshow(input_data, cmap='viridis')
        axes[0].set_title('Input (4x4)')
        axes[0].grid(True)
        plt.colorbar(im1, ax=axes[0])
        
        # Output
        im2 = axes[1].imshow(output, cmap='viridis')
        axes[1].set_title('After Max Pooling (2x2)')
        axes[1].grid(True)
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('max_pooling_demo.png')
        log("Saved max pooling visualization to 'max_pooling_demo.png'", "SUCCESS")
        plt.close()
    
    def create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        log("CREATING SAMPLE DATASET", "HEADER")
        
        # Create synthetic data
        log("Generating synthetic image data")
        num_samples = 100
        X = np.random.randn(num_samples, 224, 224, 3).astype(np.float32)
        y = np.random.randint(0, 10, num_samples)  # 10 classes for demo
        
        log(f"Created {num_samples} synthetic images")
        log(f"Image shape: {X[0].shape}")
        log(f"Number of classes: {len(np.unique(y))}")
        
        # Split into train/test
        log("Splitting data into train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        log(f"Training samples: {len(X_train)}")
        log(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def demonstrate_feature_extraction(self):
        """Demonstrate how CNNs extract hierarchical features"""
        log("DEMONSTRATING HIERARCHICAL FEATURE EXTRACTION", "HEADER")
        
        if self.model is None:
            log("Building model for feature extraction demo")
            self.build_vgg16_model()
        
        # Create a sample image
        log("Creating sample image for feature extraction")
        sample_image = np.random.randn(1, 224, 224, 3)
        
        # Get outputs from different layers
        log("Extracting features from different layers:")
        
        layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        
        for layer_name in layer_names:
            layer = self.model.get_layer(layer_name)
            intermediate_model = models.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
            
            features = intermediate_model.predict(sample_image, verbose=0)
            log(f"{layer_name} output shape: {features.shape}")
            log(f"  - Spatial dimensions: {features.shape[1]}x{features.shape[2]}")
            log(f"  - Number of channels (depth): {features.shape[3]}")

def main():
    """Main function to run the CNN/VGG-16 demonstration"""
    log("CNN/VGG-16 IMPLEMENTATION BASED ON LEAROPENCV TUTORIAL", "HEADER")
    log("This implementation demonstrates key concepts from the tutorial:")
    log("- Convolutional Neural Networks (CNNs)")
    log("- VGG-16 Architecture")
    log("- Convolution Operations")
    log("- Max Pooling")
    log("- Feature Extraction")
    log("- Fully Connected Classifiers")
    
    # Create VGG-16 implementation
    vgg = VGG16Implementation()
    
    # 1. Demonstrate convolution operation with Sobel kernel
    vgg.demonstrate_convolution_operation()
    
    # 2. Build VGG-16 model
    model = vgg.build_vgg16_model()
    
    # 3. Visualize architecture
    vgg.visualize_architecture()
    
    # 4. Demonstrate max pooling
    vgg.demonstrate_max_pooling()
    
    # 5. Demonstrate feature extraction
    vgg.demonstrate_feature_extraction()
    
    # 6. Additional demonstrations
    log("ADDITIONAL INFORMATION", "HEADER")
    
    # Calculate parameters for MLP comparison (as mentioned in tutorial)
    log("Comparing with MLP (Multi-Layer Perceptron):")
    input_pixels = 224 * 224 * 3
    log(f"Input image flattened: {input_pixels:,} pixels")
    
    # MLP with 3 layers of 128 neurons each
    mlp_params_layer1 = input_pixels * 128
    mlp_params_layer2 = 128 * 128
    mlp_params_layer3 = 128 * 128
    total_mlp_params = mlp_params_layer1 + mlp_params_layer2 + mlp_params_layer3
    
    log(f"MLP parameters (first 3 layers): {total_mlp_params:,}")
    log("This demonstrates why CNNs are more efficient than MLPs for image data")
    
    # Key insights from tutorial
    log("\nKEY INSIGHTS FROM THE TUTORIAL:", "HEADER")
    log("1. CNNs are translation invariant - can detect features regardless of position")
    log("2. CNNs learn hierarchical features:")
    log("   - Early layers: edges, corners, color blobs")
    log("   - Deeper layers: complex structures (e.g., car wheels)")
    log("3. Pooling reduces spatial dimensions while preserving important features")
    log("4. Convolutional layers act as feature extractors")
    log("5. Fully connected layers act as classifiers")
    log("6. VGG-16 was state-of-the-art in 2014 with human-level performance")
    
    log("\nScript execution completed successfully!", "SUCCESS")
    log("Generated files:")
    log("  - convolution_demo.png: Visualization of convolution operation")
    log("  - max_pooling_demo.png: Visualization of max pooling")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nScript interrupted by user", "WARNING")
    except Exception as e:
        log(f"An error occurred: {e}", "ERROR")
        import traceback
        traceback.print_exc()
