#!/usr/bin/env python3

import numpy as np
import cv2
import json
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

@dataclass
class OmniConfig:
    """Configuration for neural network anomaly detection"""
    model_path: Optional[str] = "nn_model.npz"
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    hidden_size: int = 128
    image_size: int = 64  # Resize images to 64x64
    anomaly_threshold: float = 0.5
    enable_visualization: bool = True
    
class NeuralNetwork:
    """Neural network implementation from scratch using only numpy"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))
        
        # Store activations for backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None
        
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
        """Forward pass through the network"""
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        
        return self.a3
    
    def backward(self, X, y, output):
        """Backward pass (backpropagation)"""
        m = X.shape[0]
        
        # Calculate gradients for output layer
        dz3 = output - y
        dW3 = (1/m) * np.dot(self.a2.T, dz3)
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # Calculate gradients for layer 2
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Calculate gradients for layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def calculate_loss(self, y_true, y_pred):
        """Calculate cross-entropy loss"""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.forward(X)
    
    def save_weights(self, filepath):
        """Save network weights"""
        np.savez(filepath,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)
    
    def load_weights(self, filepath):
        """Load network weights"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']


class OmniFiberNeuralAnalyzer:
    """Neural network-based fiber optic anomaly detector"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize neural network
        input_size = config.image_size * config.image_size
        hidden_size = config.hidden_size
        output_size = 2  # Normal vs Anomalous
        
        self.network = NeuralNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            learning_rate=config.learning_rate
        )
        
        # Try to load existing model
        if os.path.exists(config.model_path):
            try:
                self.network.load_weights(config.model_path)
                self.logger.info(f"Loaded model from {config.model_path}")
            except:
                self.logger.warning("Could not load model weights")
        
        # Training data storage
        self.training_data = []
        self.training_labels = []
        
    def preprocess_image(self, image):
        """Preprocess image for neural network"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to fixed size
        resized = cv2.resize(gray, (self.config.image_size, self.config.image_size))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Flatten for neural network
        flattened = normalized.flatten()
        
        return flattened, resized
    
    def load_image(self, path):
        """Load image from file"""
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not read image: {path}")
                return None
            return img
    
    def _load_from_json(self, json_path):
        """Load matrix from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            channels = data['image_dimensions'].get('channels', 3)
            
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            
            for pixel in data['pixels']:
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                if 0 <= x < width and 0 <= y < height:
                    bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                    if isinstance(bgr, (int, float)):
                        bgr = [bgr] * 3
                    matrix[y, x] = bgr[:3]
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading JSON {json_path}: {e}")
            return None
    
    def train_on_reference_data(self, ref_dir):
        """Train the neural network on reference (normal) data"""
        self.logger.info(f"Training on reference data from: {ref_dir}")
        
        # Load all reference images as normal samples
        normal_data = []
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        
        for filename in os.listdir(ref_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                filepath = os.path.join(ref_dir, filename)
                image = self.load_image(filepath)
                if image is not None:
                    preprocessed, _ = self.preprocess_image(image)
                    normal_data.append(preprocessed)
        
        if len(normal_data) < 2:
            self.logger.error("Not enough reference data for training")
            return False
        
        self.logger.info(f"Loaded {len(normal_data)} normal samples")
        
        # Create synthetic anomalous samples
        anomalous_data = self._create_synthetic_anomalies(normal_data)
        
        # Prepare training data
        X_normal = np.array(normal_data)
        X_anomalous = np.array(anomalous_data)
        X = np.vstack([X_normal, X_anomalous])
        
        # Create labels (0 = normal, 1 = anomalous)
        y_normal = np.zeros((len(normal_data), 2))
        y_normal[:, 0] = 1  # One-hot encoding for normal
        
        y_anomalous = np.zeros((len(anomalous_data), 2))
        y_anomalous[:, 1] = 1  # One-hot encoding for anomalous
        
        y = np.vstack([y_normal, y_anomalous])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Train the network
        self._train_network(X, y)
        
        # Save the model
        self.network.save_weights(self.config.model_path)
        self.logger.info(f"Model saved to {self.config.model_path}")
        
        return True
    
    def _create_synthetic_anomalies(self, normal_data):
        """Create synthetic anomalous samples from normal data"""
        anomalous_data = []
        
        for normal_sample in normal_data:
            # Reshape to 2D for manipulation
            img = normal_sample.reshape(self.config.image_size, self.config.image_size)
            
            # Create various types of anomalies
            for _ in range(3):  # Create 3 anomalous versions per normal sample
                anomaly = img.copy()
                anomaly_type = np.random.choice(['scratch', 'blob', 'noise', 'dark_spot'])
                
                if anomaly_type == 'scratch':
                    # Add random line
                    x1, y1 = np.random.randint(0, self.config.image_size, 2)
                    x2, y2 = np.random.randint(0, self.config.image_size, 2)
                    cv2.line(anomaly, (x1, y1), (x2, y2), 
                            np.random.uniform(0, 1), thickness=np.random.randint(1, 3))
                
                elif anomaly_type == 'blob':
                    # Add random blob
                    cx, cy = np.random.randint(10, self.config.image_size-10, 2)
                    radius = np.random.randint(3, 10)
                    cv2.circle(anomaly, (cx, cy), radius, 
                              np.random.uniform(0, 1), -1)
                
                elif anomaly_type == 'noise':
                    # Add random noise
                    noise = np.random.normal(0, 0.1, anomaly.shape)
                    anomaly = np.clip(anomaly + noise, 0, 1)
                
                elif anomaly_type == 'dark_spot':
                    # Add dark region
                    x, y = np.random.randint(5, self.config.image_size-15, 2)
                    w, h = np.random.randint(5, 15, 2)
                    anomaly[y:y+h, x:x+w] *= np.random.uniform(0.3, 0.7)
                
                anomalous_data.append(anomaly.flatten())
        
        return anomalous_data
    
    def _train_network(self, X, y):
        """Train the neural network"""
        n_samples = X.shape[0]
        losses = []
        
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            # Mini-batch training
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_X = X[i:i+self.config.batch_size]
                batch_y = y[i:i+self.config.batch_size]
                
                # Forward pass
                output = self.network.forward(batch_X)
                
                # Calculate loss
                loss = self.network.calculate_loss(batch_y, output)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.network.backward(batch_X, batch_y, output)
            
            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            # Decay learning rate
            if epoch % 20 == 0 and epoch > 0:
                self.network.learning_rate *= 0.9
            
            if epoch % 10 == 0:
                # Calculate accuracy
                predictions = self.network.predict(X)
                true_labels = np.argmax(y, axis=1)
                accuracy = np.mean(predictions == true_labels)
                self.logger.info(f"Epoch {epoch}/{self.config.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        
        # Plot training loss
        if self.config.enable_visualization:
            plt.figure(figsize=(10, 6))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('training_loss.png')
            plt.close()
    
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Main analysis method - compatible with pipeline"""
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        image = self.load_image(image_path)
        if image is None:
            return self._create_error_report(image_path, output_dir)
        
        # Get predictions
        results = self.detect_anomalies(image_path)
        
        if results:
            # Convert to pipeline format
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            # Save report
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2)
            self.logger.info(f"Saved detection report to {report_path}")
            
            # Generate visualizations
            if self.config.enable_visualization:
                viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
                self.visualize_results(results, str(viz_path))
                
                # Save activation maps
                activation_path = output_path / f"{Path(image_path).stem}_activations.png"
                self.visualize_activations(results, str(activation_path))
            
            return pipeline_report
        
        return self._create_error_report(image_path, output_dir)
    
    def detect_anomalies(self, test_path):
        """Detect anomalies using neural network"""
        # Load test image
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        # Preprocess
        preprocessed, resized_gray = self.preprocess_image(test_image)
        
        # Get network prediction
        X = preprocessed.reshape(1, -1)
        output_proba = self.network.predict_proba(X)
        prediction = np.argmax(output_proba, axis=1)[0]
        
        # Get anomaly confidence (probability of being anomalous)
        anomaly_confidence = output_proba[0, 1]
        
        # Analyze local regions for anomaly localization
        anomaly_regions = self._localize_anomalies(resized_gray, test_image.shape[:2])
        
        # Get activation maps for visualization
        _ = self.network.forward(X)
        activations = {
            'layer1': self.network.a1,
            'layer2': self.network.a2,
            'output': self.network.a3
        }
        
        return {
            'test_image': test_image,
            'test_gray': resized_gray,
            'preprocessed': preprocessed,
            'prediction': prediction,
            'anomaly_confidence': float(anomaly_confidence),
            'normal_confidence': float(output_proba[0, 0]),
            'is_anomalous': prediction == 1,
            'anomaly_regions': anomaly_regions,
            'activations': activations,
            'threshold': self.config.anomaly_threshold
        }
    
    def _localize_anomalies(self, resized_gray, original_shape):
        """Localize anomalies using sliding window"""
        window_size = 16
        stride = 8
        h, w = resized_gray.shape
        
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                # Extract window
                window = resized_gray[y:y+window_size, x:x+window_size]
                
                # Preprocess window
                window_flat = window.flatten() / 255.0
                
                # Pad to match network input size
                padded = np.zeros(self.config.image_size * self.config.image_size)
                padded[:window_flat.shape[0]] = window_flat
                
                # Get prediction for window
                X = padded.reshape(1, -1)
                proba = self.network.predict_proba(X)
                anomaly_score = proba[0, 1]
                
                # Update anomaly map
                anomaly_map[y:y+window_size, x:x+window_size] = np.maximum(
                    anomaly_map[y:y+window_size, x:x+window_size],
                    anomaly_score
                )
        
        # Find regions with high anomaly scores
        threshold = 0.7
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        # Scale factors to original size
        h_scale = original_shape[0] / h
        w_scale = original_shape[1] / w
        
        regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area > 10:  # Filter small regions
                x_orig = int(x * w_scale)
                y_orig = int(y * h_scale)
                w_orig = int(w * w_scale)
                h_orig = int(h * h_scale)
                
                region_mask = (labels == i)
                confidence = float(np.mean(anomaly_map[region_mask]))
                
                regions.append({
                    'bbox': (x_orig, y_orig, w_orig, h_orig),
                    'area': int(area * h_scale * w_scale),
                    'confidence': confidence,
                    'centroid': (int(centroids[i][0] * w_scale), int(centroids[i][1] * h_scale))
                })
        
        return regions
    
    def visualize_results(self, results, output_path):
        """Visualize detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        test_img = results['test_image']
        if len(test_img.shape) == 3:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        else:
            test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
        
        # Original image
        axes[0, 0].imshow(test_img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        preprocessed = results['preprocessed'].reshape(self.config.image_size, self.config.image_size)
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed (Network Input)')
        axes[0, 1].axis('off')
        
        # Detection result
        overlay = test_img_rgb.copy()
        color = (255, 0, 0) if results['is_anomalous'] else (0, 255, 0)
        text_color = 'red' if results['is_anomalous'] else 'green'
        
        # Draw regions
        for region in results['anomaly_regions']:
            x, y, w, h = region['bbox']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(overlay, f"{region['confidence']:.2f}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(f"Detection: {'ANOMALOUS' if results['is_anomalous'] else 'NORMAL'}", 
                            color=text_color, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Confidence scores
        ax = axes[1, 1]
        classes = ['Normal', 'Anomalous']
        confidences = [results['normal_confidence'], results['anomaly_confidence']]
        bars = ax.bar(classes, confidences, color=['green', 'red'])
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.1%}', ha='center', va='bottom')
        
        ax.set_ylabel('Confidence')
        ax.set_title('Neural Network Output')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=self.config.anomaly_threshold, color='black', linestyle='--', 
                  label=f'Threshold ({self.config.anomaly_threshold})')
        ax.legend()
        
        plt.suptitle(f'Neural Network Anomaly Detection\nFile: {Path(results.get("test_path", "Unknown")).name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")
    
    def visualize_activations(self, results, output_path):
        """Visualize neural network activations"""
        activations = results['activations']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Layer 1 activations
        layer1 = activations['layer1'][0]  # First sample
        n_neurons = min(64, len(layer1))  # Show first 64 neurons
        layer1_img = layer1[:n_neurons].reshape(-1, 1)
        axes[0].imshow(layer1_img, cmap='RdBu', aspect='auto')
        axes[0].set_title('Layer 1 Activations')
        axes[0].set_xlabel('Neuron Index')
        axes[0].set_ylabel('Activation')
        
        # Layer 2 activations
        layer2 = activations['layer2'][0]
        n_neurons = min(64, len(layer2))
        layer2_img = layer2[:n_neurons].reshape(-1, 1)
        axes[1].imshow(layer2_img, cmap='RdBu', aspect='auto')
        axes[1].set_title('Layer 2 Activations')
        axes[1].set_xlabel('Neuron Index')
        
        # Output probabilities
        output = activations['output'][0]
        axes[2].bar(['Normal', 'Anomalous'], output, color=['green', 'red'])
        axes[2].set_title('Output Layer (Softmax)')
        axes[2].set_ylabel('Probability')
        axes[2].set_ylim(0, 1)
        
        plt.suptitle('Neural Network Activation Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Activation visualization saved to {output_path}")
    
    def _convert_to_pipeline_format(self, results, image_path):
        """Convert results to pipeline-expected format"""
        defects = []
        defect_id = 1
        
        # Convert anomaly regions to defects
        for region in results['anomaly_regions']:
            x, y, w, h = region['bbox']
            cx, cy = region['centroid']
            
            defect = {
                'defect_id': f"NN_ANOM_{defect_id:04d}",
                'defect_type': 'NEURAL_ANOMALY',
                'location_xy': [cx, cy],
                'bbox': [x, y, w, h],
                'area_px': region['area'],
                'confidence': float(region['confidence']),
                'severity': self._confidence_to_severity(region['confidence']),
                'orientation': None,
                'contributing_algorithms': ['neural_network_detector'],
                'detection_metadata': {
                    'network_confidence': float(results['anomaly_confidence']),
                    'region_confidence': float(region['confidence'])
                }
            }
            defects.append(defect)
            defect_id += 1
        
        # Overall quality score
        quality_score = float(100 * (1 - results['anomaly_confidence']))
        if len(defects) > 0:
            quality_score = max(0, quality_score - len(defects) * 5)
        
        report = {
            'source_image': image_path,
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'analysis_complete': True,
            'success': True,
            'overall_quality_score': quality_score,
            'defects': defects,
            'summary': {
                'total_defects': len(defects),
                'is_anomalous': results['is_anomalous'],
                'anomaly_confidence': float(results['anomaly_confidence']),
                'normal_confidence': float(results['normal_confidence']),
                'quality_score': quality_score,
                'neural_network_prediction': 'ANOMALOUS' if results['is_anomalous'] else 'NORMAL'
            },
            'analysis_metadata': {
                'analyzer': 'neural_network_anomaly_detector',
                'version': '1.0',
                'model_path': self.config.model_path,
                'network_architecture': f'{self.config.image_size*self.config.image_size}-{self.config.hidden_size}-{self.config.hidden_size}-2',
                'threshold': self.config.anomaly_threshold
            }
        }
        
        return report
    
    def _confidence_to_severity(self, confidence):
        """Convert confidence to severity level"""
        if confidence >= 0.9:
            return 'CRITICAL'
        elif confidence >= 0.7:
            return 'HIGH'
        elif confidence >= 0.5:
            return 'MEDIUM'
        elif confidence >= 0.3:
            return 'LOW'
        else:
            return 'NEGLIGIBLE'
    
    def _create_error_report(self, image_path, output_dir):
        """Create error report"""
        empty_report = {
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'success': False,
            'error': 'Analysis failed',
            'defects': []
        }
        
        report_path = Path(output_dir) / f"{Path(image_path).stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(empty_report, f, indent=2)
        
        return empty_report
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return time.strftime("%Y-%m-%d_%H:%M:%S")


def main():
    """Main function for testing"""
    print("\n" + "="*80)
    print("NEURAL NETWORK FIBER OPTIC ANOMALY DETECTOR".center(80))
    print("="*80)
    print("\nImplemented from scratch using only NumPy!")
    print("Based on the neural network tutorial\n")
    
    config = OmniConfig()
    analyzer = OmniFiberNeuralAnalyzer(config)
    
    while True:
        print("\n1. Train on reference data")
        print("2. Analyze test image")
        print("3. Quit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            ref_dir = input("Enter path to reference data directory: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                analyzer.train_on_reference_data(ref_dir)
            else:
                print(f"Directory not found: {ref_dir}")
        
        elif choice == '2':
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            if os.path.isfile(test_path):
                output_dir = f"nn_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                analyzer.analyze_end_face(test_path, output_dir)
                print(f"\nResults saved to: {output_dir}/")
            else:
                print(f"File not found: {test_path}")
        
        elif choice == '3':
            break
    
    print("\nThank you for using the Neural Network Analyzer!")


if __name__ == "__main__":
    main()
