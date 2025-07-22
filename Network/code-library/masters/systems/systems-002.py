#!/usr/bin/env python3
"""
CNN-based Fiber Optic Anomaly Detection Module
Based on the tutorial architecture but adapted for fiber optic defect detection
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

@dataclass
class CNNConfig:
    """Configuration for CNN model"""
    # Model architecture
    input_shape: Tuple[int, int, int] = (128, 128, 1)  # Grayscale images
    num_classes: int = 5  # Normal, Scratch, Dig, Contamination, Edge
    
    # Layer 1 parameters (following tutorial)
    layer1_filters: int = 2
    layer1_kernel_size: Tuple[int, int] = (5, 5)
    layer1_activation: str = 'relu'
    layer1_pool_size: Tuple[int, int] = (2, 2)
    
    # Layer 2 parameters (following tutorial)
    layer2_filters: int = 4
    layer2_kernel_size: Tuple[int, int] = (3, 3)
    layer2_activation: str = 'sigmoid'
    layer2_pool_size: Tuple[int, int] = (2, 2)
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Model paths
    model_save_path: str = "fiber_cnn_model.h5"
    weights_save_path: str = "fiber_cnn_weights.h5"
    history_save_path: str = "training_history.json"

class FiberOpticCNN:
    """CNN for fiber optic anomaly detection following tutorial architecture"""
    
    def __init__(self, config: CNNConfig):
        self.config = config
        self.model = None
        self.history = None
        self.logger = logging.getLogger(__name__)
        self.class_names = ['Normal', 'Scratch', 'Dig', 'Contamination', 'Edge']
        
    def build_model(self):
        """Build CNN model following the tutorial architecture"""
        self.logger.info("Building CNN model...")
        
        # Input layer
        inputs = layers.Input(shape=self.config.input_shape)
        
        # Layer 1: First Convolution + ReLU
        conv1 = layers.Conv2D(
            filters=self.config.layer1_filters,
            kernel_size=self.config.layer1_kernel_size,
            strides=(1, 1),
            padding='valid',
            activation=self.config.layer1_activation,
            name='conv1'
        )(inputs)
        
        # Layer 1: Max Pooling
        pool1 = layers.MaxPooling2D(
            pool_size=self.config.layer1_pool_size,
            strides=self.config.layer1_pool_size,  # Tutorial uses stride = pool_size
            name='pool1'
        )(conv1)
        
        # Layer 2: Second Convolution + Sigmoid
        conv2 = layers.Conv2D(
            filters=self.config.layer2_filters,
            kernel_size=self.config.layer2_kernel_size,
            strides=(1, 1),
            padding='valid',
            activation=self.config.layer2_activation,
            name='conv2'
        )(pool1)
        
        # Layer 2: Max Pooling
        pool2 = layers.MaxPooling2D(
            pool_size=self.config.layer2_pool_size,
            strides=self.config.layer2_pool_size,
            name='pool2'
        )(conv2)
        
        # Flatten (as in tutorial)
        flatten = layers.Flatten(name='flatten')(pool2)
        
        # Fully connected layer (output layer)
        # Adding a hidden layer for better performance
        dense1 = layers.Dense(100, activation='relu', name='dense1')(flatten)
        dropout = layers.Dropout(0.5, name='dropout')(dense1)
        
        # Output layer
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',  # For multi-class classification
            name='output'
        )(dropout)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='FiberOpticCNN')
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def visualize_convolution_process(self, image: np.ndarray, layer_name: str = 'conv1'):
        """Visualize convolution process as described in tutorial"""
        if self.model is None:
            raise ValueError("Model not built yet!")
        
        # Get the specific layer
        layer = self.model.get_layer(layer_name)
        
        # Create a model that outputs the activations of this layer
        activation_model = models.Model(
            inputs=self.model.input,
            outputs=layer.output
        )
        
        # Prepare image
        if len(image.shape) == 2:
            image = image.reshape(1, *image.shape, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        # Get activations
        activations = activation_model.predict(image)
        
        # Visualize each filter's output
        n_filters = activations.shape[-1]
        fig, axes = plt.subplots(1, n_filters, figsize=(15, 5))
        
        if n_filters == 1:
            axes = [axes]
        
        for i in range(n_filters):
            axes[i].imshow(activations[0, :, :, i], cmap='gray')
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')
        
        plt.suptitle(f'Convolution Outputs from {layer_name}')
        plt.tight_layout()
        
        return activations
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CNN input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, self.config.input_shape[:2])
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def prepare_training_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from directory structure"""
        self.logger.info(f"Loading training data from {data_dir}")
        
        X, y = [], []
        
        # Expected directory structure:
        # data_dir/
        #   Normal/
        #   Scratch/
        #   Dig/
        #   Contamination/
        #   Edge/
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Load images from class directory
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img_processed = self.preprocess_image(img)
                    
                    X.append(img_processed)
                    y.append(class_idx)
            
            self.logger.info(f"Loaded {len([y_ for y_ in y if y_ == class_idx])} images for class {class_name}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=self.config.num_classes)
        
        self.logger.info(f"Total samples loaded: {len(X)}")
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the CNN model"""
        if self.model is None:
            self.build_model()
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split,
            random_state=42
        )
        
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                self.config.weights_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save model and history
        self.save_model()
        self.plot_training_history()
        
        return self.history
    
    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict single image"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Preprocess image
        img_processed = self.preprocess_image(image)
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get class with highest probability
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        # Create detailed results
        results = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(self.class_names))
            },
            'is_anomaly': predicted_class != 0,  # 0 is Normal class
            'anomaly_type': self.class_names[predicted_class] if predicted_class != 0 else None
        }
        
        return results
    
    def predict_with_visualization(self, image: np.ndarray, save_path: Optional[str] = None):
        """Predict and visualize the convolution process"""
        # Get prediction
        results = self.predict_single(image)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Original image
        ax1 = plt.subplot(3, 4, 1)
        img_display = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ax1.imshow(img_display, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Preprocessed image
        ax2 = plt.subplot(3, 4, 2)
        img_processed = self.preprocess_image(image)
        ax2.imshow(img_processed.squeeze(), cmap='gray')
        ax2.set_title('Preprocessed (128x128)')
        ax2.axis('off')
        
        # Layer 1 convolutions
        conv1_outputs = self.visualize_layer_activations(image, 'conv1')
        for i in range(self.config.layer1_filters):
            ax = plt.subplot(3, 4, 3 + i)
            ax.imshow(conv1_outputs[0, :, :, i], cmap='gray')
            ax.set_title(f'Conv1 Filter {i+1}')
            ax.axis('off')
        
        # Layer 1 pooling
        pool1_outputs = self.visualize_layer_activations(image, 'pool1')
        for i in range(self.config.layer1_filters):
            ax = plt.subplot(3, 4, 5 + i)
            ax.imshow(pool1_outputs[0, :, :, i], cmap='gray')
            ax.set_title(f'Pool1 Filter {i+1}')
            ax.axis('off')
        
        # Layer 2 convolutions (show first 4)
        conv2_outputs = self.visualize_layer_activations(image, 'conv2')
        for i in range(min(4, self.config.layer2_filters)):
            ax = plt.subplot(3, 4, 7 + i)
            ax.imshow(conv2_outputs[0, :, :, i], cmap='gray')
            ax.set_title(f'Conv2 Filter {i+1}')
            ax.axis('off')
        
        # Prediction results
        ax_pred = plt.subplot(3, 4, 11)
        ax_pred.bar(self.class_names, list(results['all_probabilities'].values()))
        ax_pred.set_title(f'Prediction: {results["predicted_class"]} ({results["confidence"]:.2%})')
        ax_pred.set_ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        
        # Summary text
        ax_text = plt.subplot(3, 4, 12)
        ax_text.axis('off')
        summary_text = f"""
Prediction Summary:
------------------
Class: {results['predicted_class']}
Confidence: {results['confidence']:.2%}
Is Anomaly: {results['is_anomaly']}

Probabilities:
"""
        for class_name, prob in results['all_probabilities'].items():
            summary_text += f"\n{class_name}: {prob:.3f}"
        
        ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('CNN Forward Propagation Visualization', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        return results, fig
    
    def visualize_layer_activations(self, image: np.ndarray, layer_name: str) -> np.ndarray:
        """Get activations for a specific layer"""
        layer = self.model.get_layer(layer_name)
        activation_model = models.Model(inputs=self.model.input, outputs=layer.output)
        
        img_processed = self.preprocess_image(image)
        img_batch = np.expand_dims(img_processed, axis=0)
        
        activations = activation_model.predict(img_batch, verbose=0)
        return activations
    
    def integrate_with_detection_pipeline(self, detection_results: Dict) -> Dict:
        """Integrate CNN predictions with existing detection results"""
        # Extract the test image
        test_image = detection_results['test_image']
        
        # Get CNN prediction
        cnn_results = self.predict_single(test_image)
        
        # Add CNN results to detection results
        detection_results['cnn_analysis'] = {
            'predicted_defect_type': cnn_results['predicted_class'],
            'confidence': cnn_results['confidence'],
            'probabilities': cnn_results['all_probabilities'],
            'cnn_anomaly_detected': cnn_results['is_anomaly']
        }
        
        # Update overall verdict based on CNN
        if cnn_results['is_anomaly'] and cnn_results['confidence'] > 0.8:
            detection_results['verdict']['is_anomalous'] = True
            detection_results['verdict']['cnn_triggered'] = True
        
        return detection_results
    
    def save_model(self):
        """Save model and training history"""
        if self.model:
            self.model.save(self.config.model_save_path)
            self.logger.info(f"Model saved to {self.config.model_save_path}")
        
        if self.history:
            history_dict = {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
            }
            with open(self.config.history_save_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load saved model"""
        path = model_path or self.config.model_save_path
        if os.path.exists(path):
            self.model = models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.close()


# Integration function for use with existing detection.py
def enhance_detection_with_cnn(detection_results: Dict, cnn_model_path: Optional[str] = None) -> Dict:
    """Enhance existing detection results with CNN analysis"""
    config = CNNConfig()
    cnn = FiberOpticCNN(config)
    
    # Load pre-trained model
    if cnn_model_path and os.path.exists(cnn_model_path):
        cnn.load_model(cnn_model_path)
    else:
        # Build model for demonstration (would need training in practice)
        cnn.build_model()
    
    # Integrate CNN results
    enhanced_results = cnn.integrate_with_detection_pipeline(detection_results)
    
    return enhanced_results


# Training script
def train_cnn_model(data_directory: str):
    """Train CNN model on fiber optic dataset"""
    config = CNNConfig()
    cnn = FiberOpticCNN(config)
    
    # Build model
    cnn.build_model()
    
    # Prepare data
    X, y = cnn.prepare_training_data(data_directory)
    
    # Train model
    cnn.train(X, y)
    
    return cnn


# Demonstration script
def demonstrate_cnn_forward_propagation():
    """Demonstrate CNN forward propagation as in tutorial"""
    print("\n" + "="*80)
    print("CNN FORWARD PROPAGATION DEMONSTRATION")
    print("Following the tutorial architecture for fiber optic analysis")
    print("="*80 + "\n")
    
    # Create dummy model for demonstration
    config = CNNConfig()
    cnn = FiberOpticCNN(config)
    cnn.build_model()
    
    # Create synthetic fiber optic image
    synthetic_image = np.zeros((128, 128), dtype=np.uint8)
    # Add some defect patterns
    cv2.circle(synthetic_image, (64, 64), 30, 255, -1)  # Core
    cv2.circle(synthetic_image, (64, 64), 25, 200, -1)  # Inner cladding
    cv2.line(synthetic_image, (20, 20), (100, 100), 150, 2)  # Scratch
    cv2.circle(synthetic_image, (90, 40), 5, 100, -1)  # Dig
    
    # Visualize forward propagation
    results, fig = cnn.predict_with_visualization(synthetic_image, 'cnn_demo.png')
    
    print("\nForward Propagation Steps (as per tutorial):")
    print("1. Input Layer: 128x128x1 grayscale image")
    print(f"2. Conv Layer 1: {config.layer1_filters} filters of size {config.layer1_kernel_size}")
    print(f"   - Activation: {config.layer1_activation.upper()}")
    print(f"3. Max Pooling 1: {config.layer1_pool_size} with stride 2")
    print(f"4. Conv Layer 2: {config.layer2_filters} filters of size {config.layer2_kernel_size}")
    print(f"   - Activation: {config.layer2_activation.upper()}")
    print(f"5. Max Pooling 2: {config.layer2_pool_size} with stride 2")
    print("6. Flatten")
    print("7. Fully Connected Layer")
    print(f"8. Output: {config.num_classes} classes")
    
    print(f"\nPrediction: {results['predicted_class']} (Confidence: {results['confidence']:.2%})")
    
    plt.show()


if __name__ == "__main__":
    # Run demonstration
    demonstrate_cnn_forward_propagation()
