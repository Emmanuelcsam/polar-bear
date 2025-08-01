#!/usr/bin/env python3
"""
Image Classification Script using TensorFlow and Convolutional Neural Networks
Based on CIFAR-10 dataset with 10 classification categories
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime

# Function to log actions with timestamp
def log_action(message):
    """Log action with timestamp to terminal"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Function to check and install packages
def check_and_install_package(package_name, import_name=None):
    """Check if package is installed, if not install the latest version"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        log_action(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        log_action(f"✗ {package_name} is not installed. Installing latest version...")
        try:
            # Force upgrade to latest version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log_action(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            log_action(f"✗ Failed to install {package_name}")
            return False

# Check and install all required packages
log_action("Starting dependency check...")
required_packages = [
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("tensorflow", "tensorflow"),
    ("opencv-python", "cv2")
]

all_installed = True
for package, import_name in required_packages:
    if not check_and_install_package(package, import_name):
        all_installed = False

if not all_installed:
    log_action("ERROR: Some packages failed to install. Please check your environment.")
    sys.exit(1)

log_action("All dependencies satisfied. Importing libraries...")

# Import libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

log_action("Libraries imported successfully")

# Define class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
log_action(f"Defined {len(class_names)} classification categories: {', '.join(class_names)}")

# Function to load and prepare CIFAR-10 data
def load_and_prepare_data():
    """Load CIFAR-10 dataset and prepare it for training"""
    log_action("Loading CIFAR-10 dataset from Keras...")
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    
    log_action(f"Dataset loaded: {len(training_images)} training images, {len(testing_images)} testing images")
    
    # Normalize pixel values to range [0, 1]
    log_action("Normalizing pixel values to range [0, 1]...")
    training_images, testing_images = training_images / 255.0, testing_images / 255.0
    
    # Reduce dataset size for faster training (optional)
    log_action("Reducing dataset size for faster training...")
    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]
    
    log_action(f"Dataset reduced to: {len(training_images)} training, {len(testing_images)} testing images")
    
    return training_images, training_labels, testing_images, testing_labels

# Function to visualize sample images
def visualize_samples(images, labels, num_samples=16):
    """Visualize sample images from the dataset"""
    log_action(f"Visualizing {num_samples} sample images...")
    
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i][0]])
    
    plt.tight_layout()
    plt.show()
    log_action("Sample visualization complete")

# Function to build the CNN model
def build_model():
    """Build the Convolutional Neural Network model"""
    log_action("Building CNN model architecture...")
    
    model = models.Sequential()
    
    # First convolutional layer
    log_action("Adding Conv2D layer (32 filters, 3x3 kernel, ReLU activation)...")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    # First max pooling layer
    log_action("Adding MaxPooling2D layer (2x2 pool size)...")
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional layer
    log_action("Adding Conv2D layer (64 filters, 3x3 kernel, ReLU activation)...")
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Second max pooling layer
    log_action("Adding MaxPooling2D layer (2x2 pool size)...")
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional layer
    log_action("Adding Conv2D layer (64 filters, 3x3 kernel, ReLU activation)...")
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Flatten layer
    log_action("Adding Flatten layer...")
    model.add(layers.Flatten())
    
    # Dense hidden layer
    log_action("Adding Dense layer (64 units, ReLU activation)...")
    model.add(layers.Dense(64, activation='relu'))
    
    # Output layer
    log_action("Adding Dense output layer (10 units, Softmax activation)...")
    model.add(layers.Dense(10, activation='softmax'))
    
    log_action("Model architecture complete")
    return model

# Function to compile and train the model
def train_model(model, training_images, training_labels, testing_images, testing_labels, epochs=10):
    """Compile and train the model"""
    log_action("Compiling model (optimizer: Adam, loss: sparse_categorical_crossentropy)...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    log_action(f"Starting training for {epochs} epochs...")
    history = model.fit(training_images, training_labels, epochs=epochs,
                        validation_data=(testing_images, testing_labels),
                        verbose=1)
    
    log_action("Training complete")
    return history

# Function to evaluate the model
def evaluate_model(model, testing_images, testing_labels):
    """Evaluate the model on test data"""
    log_action("Evaluating model on test dataset...")
    loss, accuracy = model.evaluate(testing_images, testing_labels, verbose=0)
    log_action(f"Evaluation complete - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return loss, accuracy

# Function to save the model
def save_model(model, filename='image_classifier.model'):
    """Save the trained model"""
    log_action(f"Saving model to '{filename}'...")
    model.save(filename)
    log_action(f"Model saved successfully")

# Function to load the model
def load_model(filename='image_classifier.model'):
    """Load a saved model"""
    log_action(f"Loading model from '{filename}'...")
    model = models.load_model(filename)
    log_action("Model loaded successfully")
    return model

# Function to preprocess image for prediction
def preprocess_image(image_path):
    """Preprocess an image for prediction"""
    log_action(f"Preprocessing image: {image_path}")
    
    # Read image
    img = cv.imread(image_path)
    if img is None:
        log_action(f"ERROR: Could not read image from {image_path}")
        return None
    
    # Convert BGR to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize to 32x32 if needed
    if img.shape[:2] != (32, 32):
        log_action(f"Resizing image from {img.shape[:2]} to (32, 32)")
        img = cv.resize(img, (32, 32))
    
    # Normalize pixel values
    img = img / 255.0
    
    return img

# Function to predict single image
def predict_image(model, image_path):
    """Make prediction on a single image"""
    img = preprocess_image(image_path)
    if img is None:
        return None
    
    # Make prediction
    log_action("Making prediction...")
    prediction = model.predict(np.array([img]), verbose=0)
    
    # Get predicted class
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_names[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    log_action(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")
    
    # Display image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Prediction: {predicted_class} ({confidence:.2%})')
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence

# Main execution function
def main():
    """Main execution function"""
    log_action("=== Image Classification Script Started ===")
    
    # Check if model already exists
    model_filename = 'image_classifier.model'
    
    if os.path.exists(model_filename):
        log_action(f"Found existing model '{model_filename}'")
        user_input = input("Do you want to use the existing model? (y/n): ").lower()
        
        if user_input == 'y':
            # Load existing model
            model = load_model(model_filename)
        else:
            # Train new model
            log_action("Training new model...")
            training_images, training_labels, testing_images, testing_labels = load_and_prepare_data()
            
            # Visualize samples
            visualize_samples(training_images, training_labels)
            
            # Build and train model
            model = build_model()
            train_model(model, training_images, training_labels, testing_images, testing_labels)
            
            # Evaluate model
            evaluate_model(model, testing_images, testing_labels)
            
            # Save model
            save_model(model, model_filename)
    else:
        # No existing model, train new one
        log_action("No existing model found. Training new model...")
        training_images, training_labels, testing_images, testing_labels = load_and_prepare_data()
        
        # Visualize samples
        visualize_samples(training_images, training_labels)
        
        # Build and train model
        model = build_model()
        train_model(model, training_images, training_labels, testing_images, testing_labels)
        
        # Evaluate model
        evaluate_model(model, testing_images, testing_labels)
        
        # Save model
        save_model(model, model_filename)
    
    # Test on custom images
    log_action("\n=== Testing on Custom Images ===")
    test_images = ['horse.jpg', 'plane.jpg', 'car.jpg', 'deer.jpg']
    
    for image_file in test_images:
        if os.path.exists(image_file):
            log_action(f"\nTesting image: {image_file}")
            predict_image(model, image_file)
        else:
            log_action(f"Image file '{image_file}' not found. Skipping...")
    
    # Interactive prediction loop
    log_action("\n=== Interactive Prediction Mode ===")
    log_action("You can now enter image paths to classify them.")
    log_action("Enter 'quit' to exit.")
    
    while True:
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if os.path.exists(image_path):
            predict_image(model, image_path)
        else:
            log_action(f"File '{image_path}' not found. Please enter a valid path.")
    
    log_action("\n=== Script Completed Successfully ===")

if __name__ == "__main__":
    main()
