#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# ==================== CNN COMPONENTS FROM SCRATCH ====================

class Layer:
    """Base layer class for neural network layers."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        raise NotImplementedError
        
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError


class ConvolutionalLayer(Layer):
    """Convolutional layer implementation from scratch."""
    
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        # Initialize kernels and biases
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.random.randn(*self.output_shape) * 0.1
        
    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
                
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class MaxPoolingLayer(Layer):
    """Max pooling layer for downsampling."""
    
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        
    def forward(self, input_data):
        self.input = input_data
        depth, height, width = input_data.shape
        
        self.output_height = (height - self.pool_size) // self.stride + 1
        self.output_width = (width - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((depth, self.output_height, self.output_width))
        
        for d in range(depth):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    
                    pool_region = input_data[d, start_i:end_i, start_j:end_j]
                    self.output[d, i, j] = np.max(pool_region)
                    
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)
        
        depth, height, width = self.input.shape
        
        for d in range(depth):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.stride
                    start_j = j * self.stride
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    
                    pool_region = self.input[d, start_i:end_i, start_j:end_j]
                    max_val = np.max(pool_region)
                    
                    for pi in range(self.pool_size):
                        for pj in range(self.pool_size):
                            if pool_region[pi, pj] == max_val:
                                input_gradient[d, start_i + pi, start_j + pj] = output_gradient[d, i, j]
                                
        return input_gradient


class ReshapeLayer(Layer):
    """Reshape layer to flatten or reshape data."""
    
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, input_data):
        return np.reshape(input_data, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


class DenseLayer(Layer):
    """Fully connected dense layer."""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.random.randn(output_size, 1) * 0.1
        
    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.weights, self.input) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient


class Activation(Layer):
    """Base activation class."""
    
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
    """ReLU activation function."""
    
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)
        
        def relu_prime(x):
            return (x > 0).astype(float)
        
        super().__init__(relu, relu_prime)


# ==================== CNN ANOMALY DETECTOR ====================

class CNNAnomalyDetector:
    """CNN-based anomaly detector for fiber optic images."""
    
    def __init__(self, input_shape=(1, 128, 128)):
        self.input_shape = input_shape
        self.network = self._build_network()
        self.encoder = self._build_encoder()
        self.is_trained = False
        self.reference_encodings = []
        self.threshold = None
        self.logger = logging.getLogger(__name__)
        
    def _build_network(self):
        """Build autoencoder network for anomaly detection."""
        # Encoder
        layers = [
            ConvolutionalLayer(self.input_shape, 3, 8),
            ReLU(),
            MaxPoolingLayer(2, 2),
            ConvolutionalLayer((8, 63, 63), 3, 16),
            ReLU(),
            MaxPoolingLayer(2, 2),
            ConvolutionalLayer((16, 30, 30), 3, 32),
            ReLU(),
            MaxPoolingLayer(2, 2),
            # Flatten
            ReshapeLayer((32, 14, 14), (32 * 14 * 14, 1)),
            DenseLayer(32 * 14 * 14, 128),
            ReLU(),
            DenseLayer(128, 64),  # Bottleneck
            ReLU(),
            # Decoder
            DenseLayer(64, 128),
            ReLU(),
            DenseLayer(128, 32 * 14 * 14),
            ReLU(),
            ReshapeLayer((32 * 14 * 14, 1), (32, 14, 14))
        ]
        return layers
    
    def _build_encoder(self):
        """Build just the encoder part for feature extraction."""
        # Take layers up to the bottleneck
        return self.network[:13]  # Up to the 64-dim bottleneck
    
    def forward(self, input_data):
        """Forward pass through the network."""
        output = input_data
        for layer in self.network:
            output = layer.forward(output)
        return output
    
    def encode(self, input_data):
        """Get encoded representation of input."""
        output = input_data
        for layer in self.encoder:
            output = layer.forward(output)
        return output
    
    def backward(self, grad, learning_rate):
        """Backward pass through the network."""
        for layer in reversed(self.network):
            grad = layer.backward(grad, learning_rate)
        return grad
    
    def train_on_normal_samples(self, normal_images, epochs=10, learning_rate=0.001):
        """Train the autoencoder on normal samples."""
        self.logger.info(f"Training CNN anomaly detector on {len(normal_images)} normal samples...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for img in normal_images:
                # Resize image to expected input size
                img_resized = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_input = img_normalized.reshape(self.input_shape)
                
                # Forward pass
                output = self.forward(img_input)
                
                # Compute reconstruction loss (MSE)
                loss = np.mean((output - img_input[:, :14, :14]) ** 2)  # Compare with downsampled version
                total_loss += loss
                
                # Backward pass
                grad = 2 * (output - img_input[:, :14, :14]) / output.size
                self.backward(grad, learning_rate)
            
            avg_loss = total_loss / len(normal_images)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Store encodings of normal samples
        self.reference_encodings = []
        for img in normal_images:
            img_resized = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_input = img_normalized.reshape(self.input_shape)
            encoding = self.encode(img_input)
            self.reference_encodings.append(encoding.flatten())
        
        # Set threshold based on normal samples
        self._set_threshold()
        self.is_trained = True
        
    def _set_threshold(self):
        """Set anomaly threshold based on reference encodings."""
        if not self.reference_encodings:
            return
        
        # Compute pairwise distances between normal samples
        distances = []
        for i in range(len(self.reference_encodings)):
            for j in range(i + 1, len(self.reference_encodings)):
                dist = np.linalg.norm(self.reference_encodings[i] - self.reference_encodings[j])
                distances.append(dist)
        
        if distances:
            # Set threshold as 95th percentile of normal distances
            self.threshold = np.percentile(distances, 95)
        else:
            self.threshold = 1.0
    
    def detect_anomaly(self, image):
        """Detect if an image is anomalous."""
        if not self.is_trained:
            self.logger.warning("CNN detector not trained yet")
            return 0.0, {}
        
        # Preprocess image
        img_resized = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = img_normalized.reshape(self.input_shape)
        
        # Get encoding
        encoding = self.encode(img_input).flatten()
        
        # Compute distance to nearest normal sample
        min_distance = float('inf')
        for ref_encoding in self.reference_encodings:
            dist = np.linalg.norm(encoding - ref_encoding)
            min_distance = min(min_distance, dist)
        
        # Compute anomaly score
        anomaly_score = min_distance / (self.threshold + 1e-10)
        
        # Get reconstruction
        reconstruction = self.forward(img_input)
        reconstruction_error = np.mean((reconstruction - img_input[:, :14, :14]) ** 2)
        
        return float(anomaly_score), {
            'min_distance': float(min_distance),
            'threshold': float(self.threshold),
            'reconstruction_error': float(reconstruction_error),
            'encoding': encoding
        }


# ==================== ENHANCED OMNICONFIG ====================

@dataclass
class OmniConfig:
    """Enhanced configuration with CNN options."""
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    # New CNN-related options
    use_cnn_detector: bool = True
    cnn_weight: float = 0.3  # Weight of CNN score in final decision
    cnn_model_path: Optional[str] = None
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }


# Keep all the original classes and methods from your script
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class OmniFiberAnalyzer:
    """Enhanced fiber optic anomaly detection system with CNN support."""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.knowledge_base_path = config.knowledge_base_path or "fiber_anomaly_kb.json"
        self.reference_model = {
            'features': [],
            'statistical_model': None,
            'archetype_image': None,
            'feature_names': [],
            'comparison_results': {},
            'learned_thresholds': {},
            'timestamp': None
        }
        self.current_metadata = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize CNN detector if enabled
        self.cnn_detector = None
        if config.use_cnn_detector:
            self.cnn_detector = CNNAnomalyDetector()
            if config.cnn_model_path and os.path.exists(config.cnn_model_path):
                self._load_cnn_model(config.cnn_model_path)
        
        self.load_knowledge_base()
        
    def _load_cnn_model(self, path):
        """Load pre-trained CNN model."""
        try:
            with open(path, 'rb') as f:
                import pickle
                model_data = pickle.load(f)
                # Restore CNN state
                self.cnn_detector.network = model_data['network']
                self.cnn_detector.encoder = model_data['encoder']
                self.cnn_detector.reference_encodings = model_data['reference_encodings']
                self.cnn_detector.threshold = model_data['threshold']
                self.cnn_detector.is_trained = True
                self.logger.info(f"Loaded CNN model from {path}")
        except Exception as e:
            self.logger.warning(f"Could not load CNN model: {e}")
    
    def save_cnn_model(self, path):
        """Save trained CNN model."""
        if self.cnn_detector and self.cnn_detector.is_trained:
            try:
                import pickle
                model_data = {
                    'network': self.cnn_detector.network,
                    'encoder': self.cnn_detector.encoder,
                    'reference_encodings': self.cnn_detector.reference_encodings,
                    'threshold': self.cnn_detector.threshold
                }
                with open(path, 'wb') as f:
                    pickle.dump(model_data, f)
                self.logger.info(f"Saved CNN model to {path}")
            except Exception as e:
                self.logger.error(f"Error saving CNN model: {e}")
    
    def train_cnn_on_reference_images(self, ref_dir):
        """Train CNN detector on reference images."""
        if not self.cnn_detector:
            self.logger.warning("CNN detector not enabled")
            return
        
        # Load reference images
        normal_images = []
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        
        for filename in os.listdir(ref_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                img_path = os.path.join(ref_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    normal_images.append(img)
        
        if normal_images:
            self.logger.info(f"Training CNN on {len(normal_images)} reference images...")
            self.cnn_detector.train_on_normal_samples(normal_images, epochs=20)
            # Save the trained model
            if self.config.cnn_model_path:
                self.save_cnn_model(self.config.cnn_model_path)
    
    def detect_anomalies_comprehensive(self, test_path):
        """Enhanced anomaly detection with CNN support."""
        self.logger.info(f"Analyzing: {test_path}")
        
        if not self.reference_model.get('statistical_model'):
            self.logger.warning("No reference model available. Build one first.")
            return None
        
        test_image = self.load_image(test_path)
        if test_image is None:
            return None
        
        self.logger.info(f"Loaded image: {self.current_metadata}")
        
        if len(test_image.shape) == 3:
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test_image.copy()
        
        # Original statistical analysis
        self.logger.info("Extracting features from test image...")
        test_features, _ = self.extract_ultra_comprehensive_features(test_image)
        
        # ... (keep all the original analysis code) ...
        # Global Analysis
        self.logger.info("Performing global anomaly analysis...")
        stat_model = self.reference_model['statistical_model']
        feature_names = self.reference_model['feature_names']
        
        for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
            if key in stat_model and isinstance(stat_model[key], list):
                stat_model[key] = np.array(stat_model[key], dtype=np.float64)
        
        test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
        
        diff = test_vector - stat_model['robust_mean']
        try:
            mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
        except:
            std_vector = stat_model['std']
            std_vector[std_vector < 1e-6] = 1.0
            normalized_diff = diff / std_vector
            mahalanobis_dist = np.linalg.norm(normalized_diff)
        
        z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
        top_indices = np.argsort(z_scores)[::-1][:10]
        deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                           for i in top_indices]
        
        # Individual Comparisons
        self.logger.info(f"Comparing against {len(self.reference_model['features'])} reference samples...")
        individual_scores = []
        for i, ref_features in enumerate(self.reference_model['features']):
            comp = self.compute_exhaustive_comparison(test_features, ref_features)
            score = self._compute_anomaly_score_from_comparison(comp)
            individual_scores.append(score)
        
        scores_array = np.array(individual_scores)
        comparison_stats = {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
        }
        
        # Structural Analysis
        self.logger.info("Performing structural analysis...")
        archetype = self.reference_model['archetype_image']
        if isinstance(archetype, list):
            archetype = np.array(archetype, dtype=np.uint8)
        if test_gray.shape != archetype.shape:
            test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
        else:
            test_gray_resized = test_gray
        
        structural_comp = self.compute_image_structural_comparison(test_gray_resized, archetype)
        
        # Local Anomaly Detection
        self.logger.info("Detecting local anomalies...")
        anomaly_map = self._compute_local_anomaly_map(test_gray_resized, archetype)
        anomaly_regions = self._find_anomaly_regions(anomaly_map, test_gray.shape)
        
        # Specific Defect Detection
        self.logger.info("Detecting specific defects...")
        specific_defects = self._detect_specific_defects(test_gray)
        
        # CNN-based anomaly detection
        cnn_score = 0.0
        cnn_details = {}
        if self.config.use_cnn_detector and self.cnn_detector and self.cnn_detector.is_trained:
            self.logger.info("Running CNN-based anomaly detection...")
            cnn_score, cnn_details = self.cnn_detector.detect_anomaly(test_gray)
            self.logger.info(f"CNN anomaly score: {cnn_score:.3f}")
        
        # Determine Overall Status (enhanced with CNN)
        thresholds = self.reference_model['learned_thresholds']
        
        # Original criteria
        statistical_anomaly = mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6)
        comparison_anomaly = comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6)
        structural_anomaly = structural_comp['ssim'] < 0.7
        local_anomaly = len(anomaly_regions) > 3 or any(region['confidence'] > 0.8 for region in anomaly_regions)
        
        # CNN criterion
        cnn_anomaly = cnn_score > 1.0 if self.config.use_cnn_detector else False
        
        # Combined decision
        is_anomalous = (
            statistical_anomaly or 
            comparison_anomaly or 
            structural_anomaly or 
            local_anomaly or
            cnn_anomaly
        )
        
        # Combined confidence score
        statistical_confidence = min(1.0, max(
            mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),
            comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),
            1 - structural_comp['ssim'],
            len(anomaly_regions) / 10
        ))
        
        # Weight CNN score if available
        if self.config.use_cnn_detector and self.cnn_detector and self.cnn_detector.is_trained:
            confidence = (1 - self.config.cnn_weight) * statistical_confidence + self.config.cnn_weight * min(1.0, cnn_score)
        else:
            confidence = statistical_confidence
        
        self.logger.info("Analysis Complete!")
        
        return {
            'test_image': test_image,
            'test_gray': test_gray,
            'test_features': test_features,
            'metadata': self.current_metadata,
            
            'global_analysis': {
                'mahalanobis_distance': float(mahalanobis_dist),
                'deviant_features': deviant_features,
                'comparison_stats': comparison_stats,
            },
            
            'structural_analysis': structural_comp,
            
            'local_analysis': {
                'anomaly_map': anomaly_map,
                'anomaly_regions': anomaly_regions,
            },
            
            'specific_defects': specific_defects,
            
            'cnn_analysis': {
                'anomaly_score': float(cnn_score),
                'details': cnn_details,
                'enabled': self.config.use_cnn_detector,
                'trained': self.cnn_detector.is_trained if self.cnn_detector else False
            },
            
            'verdict': {
                'is_anomalous': is_anomalous,
                'confidence': float(confidence),
                'criteria_triggered': {
                    'mahalanobis': statistical_anomaly,
                    'comparison': comparison_anomaly,
                    'structural': structural_anomaly,
                    'local': local_anomaly,
                    'cnn': cnn_anomaly
                }
            }
        }
    
    def _compute_anomaly_score_from_comparison(self, comp):
        """Compute weighted anomaly score from comparison metrics."""
        euclidean_term = min(comp['euclidean_distance'], 1000.0) * 0.2
        manhattan_term = min(comp['manhattan_distance'], 10000.0) * 0.1
        cosine_term = comp['cosine_distance'] * 0.2
        correlation_term = (1 - abs(comp['pearson_correlation'])) * 0.1
        kl_term = min(comp['kl_divergence'], 10.0) * 0.1
        js_term = comp['js_divergence'] * 0.1
        chi_term = min(comp['chi_square'], 10.0) * 0.1
        wasserstein_term = min(comp['wasserstein_distance'], 10.0) * 0.1
        
        score = (euclidean_term + manhattan_term + cosine_term + 
                correlation_term + kl_term + js_term + 
                chi_term + wasserstein_term)
        
        return min(score, 100.0)
    
    # Include all the other methods from the original script...
    # (I'm including key methods here, but all original methods should be preserved)
    
    def load_image(self, path):
        """Load image from JSON or standard image file."""
        self.current_metadata = None
        
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not read image: {path}")
                return None
            self.current_metadata = {'filename': os.path.basename(path)}
            return img
    
    def _load_from_json(self, json_path):
        """Load matrix from JSON file with bounds checking."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            width = data['image_dimensions']['width']
            height = data['image_dimensions']['height']
            channels = data['image_dimensions'].get('channels', 3)
            
            matrix = np.zeros((height, width, channels), dtype=np.uint8)
            oob_count = 0
            
            for pixel in data['pixels']:
                x = pixel['coordinates']['x']
                y = pixel['coordinates']['y']
                
                if 0 <= x < width and 0 <= y < height:
                    bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                    if isinstance(bgr, (int, float)):
                        bgr = [bgr] * 3
                    matrix[y, x] = bgr[:3]
                else:
                    oob_count += 1
            
            if oob_count > 0:
                self.logger.warning(f"Skipped {oob_count} out-of-bounds pixels")
            
            self.current_metadata = {
                'filename': data.get('filename', os.path.basename(json_path)),
                'width': width,
                'height': height,
                'channels': channels,
                'json_path': json_path
            }
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error loading JSON {json_path}: {e}")
            return None
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        return time.strftime("%Y-%m-%d_%H:%M:%S")
    
    # Add all the other methods from your original script here...
    # (extract_ultra_comprehensive_features, compute_exhaustive_comparison, etc.)
    # I'm not including them all to save space, but they should all be preserved


def main():
    """Enhanced main function with CNN training option."""
    print("\n" + "="*80)
    print("ENHANCED OMNIFIBER ANALYZER WITH CNN - DETECTION MODULE (v2.0)".center(80))
    print("="*80)
    print("\nThis enhanced module includes CNN-based anomaly detection.")
    print("For best results, train the CNN on your reference images first.\n")
    
    config = OmniConfig(use_cnn_detector=True)
    analyzer = OmniFiberAnalyzer(config)
    
    while True:
        print("\nOptions:")
        print("1. Train CNN on reference directory")
        print("2. Analyze single image")
        print("3. Quit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            ref_dir = input("Enter path to reference directory: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                analyzer.train_cnn_on_reference_images(ref_dir)
                # Also build statistical reference model
                analyzer.build_comprehensive_reference_model(ref_dir)
            else:
                print(f"✗ Directory not found: {ref_dir}")
                
        elif choice == '2':
            test_path = input("Enter path to test image: ").strip().strip('"\'')
            if os.path.isfile(test_path):
                output_dir = f"detection_output_{Path(test_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
                print(f"\nAnalyzing {test_path}...")
                analyzer.analyze_end_face(test_path, output_dir)
                print(f"\nResults saved to: {output_dir}/")
            else:
                print(f"✗ File not found: {test_path}")
                
        elif choice == '3':
            break
        else:
            print("Invalid choice")
    
    print("\nThank you for using the Enhanced OmniFiber Analyzer!")


if __name__ == "__main__":
    main()
