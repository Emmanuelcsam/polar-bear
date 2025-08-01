#!/usr/bin/env python3
"""
Training script for CNN-based fiber optic defect detection
Includes data augmentation and synthetic data generation
"""

import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import json
from typing import Tuple, List, Dict

from cnn_fiber_detector import FiberOpticCNN, CNNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiberOpticDataGenerator:
    """Generate synthetic fiber optic images for training"""
    
    def __init__(self, output_dir: str, samples_per_class: int = 1000):
        self.output_dir = output_dir
        self.samples_per_class = samples_per_class
        self.image_size = (128, 128)
        
        # Create directories
        self.classes = ['Normal', 'Scratch', 'Dig', 'Contamination', 'Edge']
        for class_name in self.classes:
            Path(os.path.join(output_dir, class_name)).mkdir(parents=True, exist_ok=True)
    
    def generate_base_fiber(self) -> np.ndarray:
        """Generate base fiber optic image"""
        img = np.zeros(self.image_size, dtype=np.uint8)
        
        # Add background noise
        noise = np.random.normal(20, 5, self.image_size)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Draw fiber core
        center = (self.image_size[0] // 2, self.image_size[1] // 2)
        core_radius = 30
        cv2.circle(img, center, core_radius, 200, -1)
        
        # Draw cladding
        cladding_radius = 50
        cv2.circle(img, center, cladding_radius, 150, 3)
        
        # Add some realistic texture
        texture = np.random.normal(0, 10, self.image_size)
        img = np.clip(img.astype(float) + texture, 0, 255).astype(np.uint8)
        
        # Apply slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def add_scratch(self, img: np.ndarray) -> np.ndarray:
        """Add scratch defect to image"""
        img = img.copy()
        
        # Random scratch parameters
        start_x = np.random.randint(20, img.shape[1] - 20)
        start_y = np.random.randint(20, img.shape[0] - 20)
        length = np.random.randint(30, 70)
        angle = np.random.uniform(0, 2 * np.pi)
        
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        
        # Draw scratch
        thickness = np.random.randint(1, 3)
        color = np.random.randint(50, 100)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        
        return img
    
    def add_dig(self, img: np.ndarray) -> np.ndarray:
        """Add dig/pit defect to image"""
        img = img.copy()
        
        # Random dig parameters
        num_digs = np.random.randint(1, 5)
        
        for _ in range(num_digs):
            x = np.random.randint(30, img.shape[1] - 30)
            y = np.random.randint(30, img.shape[0] - 30)
            radius = np.random.randint(2, 8)
            color = np.random.randint(0, 50)
            
            cv2.circle(img, (x, y), radius, color, -1)
        
        return img
    
    def add_contamination(self, img: np.ndarray) -> np.ndarray:
        """Add contamination/blob defect to image"""
        img = img.copy()
        
        # Random contamination parameters
        num_blobs = np.random.randint(1, 4)
        
        for _ in range(num_blobs):
            x = np.random.randint(20, img.shape[1] - 20)
            y = np.random.randint(20, img.shape[0] - 20)
            
            # Create irregular blob shape
            blob = np.zeros(img.shape, dtype=np.uint8)
            radius = np.random.randint(10, 25)
            
            # Draw multiple overlapping circles for irregular shape
            for _ in range(5):
                dx = np.random.randint(-5, 5)
                dy = np.random.randint(-5, 5)
                r = np.random.randint(radius - 5, radius + 5)
                cv2.circle(blob, (x + dx, y + dy), r, 255, -1)
            
            # Apply blob to image
            blob = cv2.GaussianBlur(blob, (7, 7), 0)
            blob_mask = blob > 128
            img[blob_mask] = np.clip(img[blob_mask] * 0.7 + np.random.randint(100, 150), 0, 255)
        
        return img
    
    def add_edge_defect(self, img: np.ndarray) -> np.ndarray:
        """Add edge irregularity defect to image"""
        img = img.copy()
        
        # Create edge mask
        center = (img.shape[0] // 2, img.shape[1] // 2)
        edge_radius = 50
        
        # Draw irregular edge
        num_points = 20
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = []
        
        for angle in angles:
            # Add random variation to radius
            r = edge_radius + np.random.randint(-10, 10)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points)
        
        # Draw irregular polygon
        cv2.fillPoly(img, [points], 120)
        
        return img
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        logger.info("Generating synthetic fiber optic dataset...")
        
        for class_idx, class_name in enumerate(self.classes):
            logger.info(f"Generating {self.samples_per_class} samples for class: {class_name}")
            
            for i in range(self.samples_per_class):
                # Generate base fiber image
                img = self.generate_base_fiber()
                
                # Add defect based on class
                if class_name == 'Normal':
                    # No defect, just base image
                    pass
                elif class_name == 'Scratch':
                    img = self.add_scratch(img)
                elif class_name == 'Dig':
                    img = self.add_dig(img)
                elif class_name == 'Contamination':
                    img = self.add_contamination(img)
                elif class_name == 'Edge':
                    img = self.add_edge_defect(img)
                
                # Save image
                filename = f"{class_name}_{i:04d}.png"
                filepath = os.path.join(self.output_dir, class_name, filename)
                cv2.imwrite(filepath, img)
            
            logger.info(f"Generated {self.samples_per_class} samples for {class_name}")
        
        logger.info("Dataset generation complete!")
    
    def visualize_samples(self, num_samples: int = 5):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(len(self.classes), num_samples, figsize=(15, 15))
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.output_dir, class_name)
            images = os.listdir(class_dir)[:num_samples]
            
            for img_idx, img_file in enumerate(images):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                ax = axes[class_idx, img_idx] if len(self.classes) > 1 else axes[img_idx]
                ax.imshow(img, cmap='gray')
                if img_idx == 0:
                    ax.set_ylabel(class_name, fontsize=12, fontweight='bold')
                ax.axis('off')
        
        plt.suptitle('Synthetic Fiber Optic Dataset Samples', fontsize=16)
        plt.tight_layout()
        plt.savefig('synthetic_samples.png', dpi=150)
        plt.show()


class FiberOpticTrainer:
    """Enhanced trainer with data augmentation and callbacks"""
    
    def __init__(self, config: CNNConfig):
        self.config = config
        self.cnn = FiberOpticCNN(config)
        self.logger = logging.getLogger(__name__)
    
    def create_data_generators(self, data_dir: str) -> Tuple:
        """Create data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.config.input_shape[:2],
            batch_size=self.config.batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def train_with_generators(self, data_dir: str):
        """Train model using data generators"""
        # Build model
        self.cnn.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(data_dir)
        
        # Calculate steps
        steps_per_epoch = len(train_gen)
        validation_steps = len(val_gen)
        
        self.logger.info(f"Training samples: {train_gen.n}")
        self.logger.info(f"Validation samples: {val_gen.n}")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        
        # Enhanced callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        # Train model
        history = self.cnn.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config.epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.cnn.save_model()
        
        # Plot results
        self.plot_training_results(history)
        
        return history
    
    def plot_training_results(self, history):
        """Enhanced plotting of training results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy plot
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Final metrics
        axes[1, 1].axis('off')
        final_metrics = f"""
Final Training Metrics:
----------------------
Training Accuracy: {history.history['accuracy'][-1]:.4f}
Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
Training Loss: {history.history['loss'][-1]:.4f}
Validation Loss: {history.history['val_loss'][-1]:.4f}

Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}
Best Validation Loss: {min(history.history['val_loss']):.4f}
"""
        axes[1, 1].text(0.1, 0.9, final_metrics, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('CNN Training Results - Fiber Optic Defect Detection', fontsize=16)
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        plt.show()


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("FIBER OPTIC CNN TRAINING PIPELINE")
    print("Following tutorial architecture for defect detection")
    print("="*80 + "\n")
    
    # Configuration
    data_dir = "fiber_optic_dataset"
    synthetic_data = True
    
    # Step 1: Generate or prepare data
    if synthetic_data:
        print("Step 1: Generating synthetic dataset...")
        generator = FiberOpticDataGenerator(data_dir, samples_per_class=500)
        generator.generate_dataset()
        generator.visualize_samples()
    else:
        print("Step 1: Using existing dataset from", data_dir)
    
    # Step 2: Configure model
    print("\nStep 2: Configuring CNN model...")
    config = CNNConfig(
        epochs=30,  # Reduced for demonstration
        batch_size=32,
        learning_rate=0.001
    )
    
    # Step 3: Train model
    print("\nStep 3: Training CNN model...")
    trainer = FiberOpticTrainer(config)
    history = trainer.train_with_generators(data_dir)
    
    # Step 4: Evaluate model
    print("\nStep 4: Model evaluation complete!")
    print("Check the following files:")
    print("  - fiber_cnn_model.h5 (complete model)")
    print("  - fiber_cnn_weights.h5 (weights only)")
    print("  - training_history.json (training metrics)")
    print("  - training_results.png (performance plots)")
    print("  - synthetic_samples.png (dataset samples)")
    
    print("\nTraining complete! The model is ready for integration with detection.py")


if __name__ == "__main__":
    main()
