#!/usr/bin/env python3
"""
Adaptive Feature Detection with PyTorch and OpenCV Integration

This module combines traditional OpenCV Hough transforms with PyTorch deep learning
for adaptive circle and line detection based on intensity profile analysis.
Features real-time learning and parameter optimization for improved detection accuracy.

Author: AI Assistant
Date: August 2025
Version: 1.0.0
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import threading
import time
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature, filters
import warnings
warnings.filterwarnings("ignore")

# Import existing modules
try:
    from hough_lines import HoughLinesDetector, HoughLinesProcessor
    from hough_circles import HoughCirclesDetector, HoughCirclesProcessor
except ImportError as e:
    logging.warning(f"Could not import existing modules: {e}")
    # Create dummy classes if imports fail
    class HoughLinesDetector:
        def __init__(self, **kwargs): pass
        def detect_lines(self, frame): return None, frame
        def update_parameters(self, **kwargs): pass

    class HoughLinesProcessor:
        def __init__(self, detector=None): 
            self.detector = HoughLinesDetector()
        def process_frame(self, frame): return frame

    class HoughCirclesDetector:
        def __init__(self, **kwargs): pass
        def detect_circles(self, frame): return None, frame
        def update_parameters(self, **kwargs): pass

    class HoughCirclesProcessor:
        def __init__(self, detector=None): 
            self.detector = HoughCirclesDetector()
        def process_frame(self, frame): return frame


class IntensityProfileAnalyzer:
    """
    Analyzes intensity profiles for edge detection using first and second derivatives.
    This class implements advanced signal processing techniques for feature detection.
    """

    def __init__(self, window_size: int = 21, sigma: float = 1.0):
        """
        Initialize the intensity profile analyzer.

        Args:
            window_size (int): Size of the analysis window (should be odd)
            sigma (float): Standard deviation for Gaussian smoothing
        """
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel()
        self.first_derivative_kernel = self._create_first_derivative_kernel()
        self.second_derivative_kernel = self._create_second_derivative_kernel()

    def _create_gaussian_kernel(self) -> np.ndarray:
        """Create normalized Gaussian kernel for smoothing."""
        x = np.arange(-self.window_size//2 + 1, self.window_size//2 + 1)
        kernel = np.exp(-x**2 / (2 * self.sigma**2))
        return kernel / np.sum(kernel)

    def _create_first_derivative_kernel(self) -> np.ndarray:
        """Create first derivative of Gaussian kernel."""
        x = np.arange(-self.window_size//2 + 1, self.window_size//2 + 1)
        kernel = -x * np.exp(-x**2 / (2 * self.sigma**2)) / (self.sigma**2)
        return kernel / np.sum(np.abs(kernel))

    def _create_second_derivative_kernel(self) -> np.ndarray:
        """Create second derivative of Gaussian kernel (Laplacian of Gaussian)."""
        x = np.arange(-self.window_size//2 + 1, self.window_size//2 + 1)
        kernel = (x**2 / self.sigma**2 - 1) * np.exp(-x**2 / (2 * self.sigma**2)) / (self.sigma**4)
        return kernel / np.sum(np.abs(kernel))

    def analyze_intensity_profile(self, profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze intensity profile using first and second derivatives.

        Args:
            profile (np.ndarray): 1D intensity profile

        Returns:
            Dict containing original profile, smoothed profile, first and second derivatives
        """
        # Smooth the profile
        smoothed = np.convolve(profile, self.gaussian_kernel, mode='same')

        # Calculate first derivative (gradient)
        first_derivative = np.convolve(smoothed, self.first_derivative_kernel, mode='same')

        # Calculate second derivative (curvature)
        second_derivative = np.convolve(smoothed, self.second_derivative_kernel, mode='same')

        return {
            'original': profile,
            'smoothed': smoothed,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            'edge_strength': np.abs(first_derivative),
            'zero_crossings': self._find_zero_crossings(second_derivative)
        }

    def _find_zero_crossings(self, signal: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Find zero crossings in second derivative for edge detection."""
        zero_crossings = np.zeros_like(signal)
        for i in range(1, len(signal)):
            if signal[i-1] * signal[i] < 0 and abs(signal[i] - signal[i-1]) > threshold:
                zero_crossings[i] = 1
        return zero_crossings

    def extract_radial_profiles(self, image: np.ndarray, center: Tuple[int, int], 
                               num_rays: int = 32, max_radius: int = 100) -> List[np.ndarray]:
        """Extract radial intensity profiles from image center for circle detection."""
        profiles = []
        height, width = image.shape[:2]
        cx, cy = center

        for angle in np.linspace(0, 2*np.pi, num_rays, endpoint=False):
            profile = []
            for r in range(1, max_radius):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))

                if 0 <= x < width and 0 <= y < height:
                    if len(image.shape) == 3:
                        profile.append(np.mean(image[y, x]))
                    else:
                        profile.append(image[y, x])
                else:
                    break

            if len(profile) > self.window_size:
                profiles.append(np.array(profile))

        return profiles


class FeatureDetectionNN(nn.Module):
    """
    Neural Network for adaptive feature detection parameter optimization.
    This network learns optimal parameters based on image characteristics and detection results.
    """

    def __init__(self, input_size: int = 64, hidden_size: int = 128, output_size: int = 10):
        """
        Initialize the neural network.

        Args:
            input_size (int): Size of input feature vector
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output parameter vector
        """
        super(FeatureDetectionNN, self).__init__()

        self.input_norm = nn.LayerNorm(input_size)  # Use LayerNorm instead of BatchNorm1d

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )

        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_size//2, output_size),
            nn.Sigmoid()  # Output parameters in [0, 1] range
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input feature tensor

        Returns:
            Tuple of (parameters, confidence)
        """
        x = self.input_norm(x)
        features = self.feature_extractor(x)
        parameters = self.parameter_head(features)
        confidence = self.confidence_head(features)

        return parameters, confidence


class FeatureDataset(Dataset):
    """PyTorch Dataset for feature detection training data."""

    def __init__(self, features: List[np.ndarray], targets: List[np.ndarray], 
                 rewards: List[float]):
        """
        Initialize the dataset.

        Args:
            features: List of feature vectors
            targets: List of target parameter vectors
            rewards: List of reward values for each sample
        """
        self.features = [torch.FloatTensor(f) for f in features]
        self.targets = [torch.FloatTensor(t) for t in targets]
        self.rewards = torch.FloatTensor(rewards)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.rewards[idx]


class AdaptiveFeatureDetector:
    """
    Main class combining traditional OpenCV methods with PyTorch deep learning
    for adaptive feature detection with continuous learning capabilities.
    """

    def __init__(self, device: str = 'cpu', learning_rate: float = 0.001,
                 memory_size: int = 1000, update_frequency: int = 10):
        """
        Initialize the adaptive feature detector.

        Args:
            device (str): PyTorch device ('cpu' or 'cuda')
            learning_rate (float): Learning rate for neural network
            memory_size (int): Size of experience replay memory
            update_frequency (int): Frequency of neural network updates
        """
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.update_frequency = update_frequency

        # Initialize components
        self.intensity_analyzer = IntensityProfileAnalyzer()
        self.hough_lines_processor = HoughLinesProcessor()
        self.hough_circles_processor = HoughCirclesProcessor()

        # Neural network components
        self.lines_nn = FeatureDetectionNN(input_size=64, output_size=8).to(self.device)
        self.circles_nn = FeatureDetectionNN(input_size=64, output_size=7).to(self.device)

        # Set networks to evaluation mode for inference
        self.lines_nn.eval()
        self.circles_nn.eval()

        self.lines_optimizer = optim.Adam(self.lines_nn.parameters(), lr=learning_rate)
        self.circles_optimizer = optim.Adam(self.circles_nn.parameters(), lr=learning_rate)

        # Experience replay memory
        self.lines_memory = deque(maxlen=memory_size)
        self.circles_memory = deque(maxlen=memory_size)

        # Performance tracking
        self.frame_count = 0
        self.performance_history = {
            'lines': {'precision': [], 'recall': [], 'f1': []},
            'circles': {'precision': [], 'recall': [], 'f1': []}
        }

        # Adaptive thresholds
        self.adaptive_thresholds = {
            'edge_strength': 0.1,
            'zero_crossing': 0.05,
            'gradient_magnitude': 0.2,
            'curvature': 0.1
        }

        # Model persistence
        self.model_save_path = Path("adaptive_models")
        self.model_save_path.mkdir(exist_ok=True)

        self._load_models()

        logging.info("AdaptiveFeatureDetector initialized successfully")

    def extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature vector from image for neural network input.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Feature vector
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        features = []

        # Basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray),
            np.median(gray)
        ])

        # Gradient statistics
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.percentile(gradient_magnitude, 75),
            np.percentile(gradient_magnitude, 90)
        ])

        # Texture features using Local Binary Pattern
        try:
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist[:10])
        except:
            features.extend([0.0] * 10)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)

        # Intensity distribution features
        hist, _ = np.histogram(gray, bins=16)
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        features.extend(hist)

        # Fourier transform features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        # Extract frequency domain statistics
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.max(magnitude_spectrum)
        ])

        # Ensure we have exactly 64 features
        features = features[:64]
        while len(features) < 64:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def detect_lines_adaptive(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect lines using adaptive parameters from neural network.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tuple of (detected lines, processed image)
        """
        # Extract features
        features = self.extract_image_features(image)
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Get adaptive parameters from neural network
        with torch.no_grad():
            params, confidence = self.lines_nn(feature_tensor)
            params = params.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0][0]

        # Map normalized parameters to actual ranges
        adaptive_params = {
            'rho': int(1 + params[0] * 9),  # 1-10
            'theta_degrees': 0.1 + params[1] * 4.9,  # 0.1-5.0
            'threshold': int(10 + params[2] * 290),  # 10-300
            'min_line_length': int(5 + params[3] * 195),  # 5-200
            'max_line_gap': int(1 + params[4] * 49),  # 1-50
            'blur_kernel_size': int(1 + params[5] * 14) | 1,  # 1-15 (odd)
            'blur_sigma': 0.1 + params[6] * 4.9,  # 0.1-5.0
            'canny_low': int(10 + params[7] * 190)  # 10-200
        }

        # Ensure canny_high > canny_low
        adaptive_params['canny_high'] = adaptive_params['canny_low'] + 50

        # Update Hough detector with adaptive parameters
        self.hough_lines_processor.detector.update_parameters(**adaptive_params)

        # Perform detection
        lines, processed_image = self.hough_lines_processor.detector.detect_lines(image)

        # Store experience for learning
        self._store_lines_experience(features, params, lines, confidence)

        return lines, processed_image

    def detect_circles_adaptive(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect circles using adaptive parameters from neural network.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tuple of (detected circles, processed image)
        """
        # Extract features
        features = self.extract_image_features(image)
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Get adaptive parameters from neural network
        with torch.no_grad():
            params, confidence = self.circles_nn(feature_tensor)
            params = params.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0][0]

        # Map normalized parameters to actual ranges
        adaptive_params = {
            'dp': 0.1 + params[0] * 4.9,  # 0.1-5.0
            'min_dist': int(1 + params[1] * 999),  # 1-1000
            'param1': int(1 + params[2] * 499),  # 1-500
            'param2': int(1 + params[3] * 299),  # 1-300
            'min_radius': int(0 + params[4] * 500),  # 0-500
            'max_radius': int(1 + params[5] * 1999),  # 1-2000
            'blur_kernel_size': int(1 + params[6] * 50) | 1  # 1-51 (odd)
        }

        # Update Hough detector with adaptive parameters
        self.hough_circles_processor.detector.update_parameters(**adaptive_params)

        # Perform detection
        circles, processed_image = self.hough_circles_processor.detector.detect_circles(image)

        # Store experience for learning
        self._store_circles_experience(features, params, circles, confidence)

        return circles, processed_image

    def process_frame_comprehensive(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive frame processing with both line and circle detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            Dict containing detection results and analysis
        """
        self.frame_count += 1

        # Intensity profile analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Extract intensity profiles for analysis
        height, width = gray.shape
        center_y, center_x = height // 2, width // 2

        # Horizontal and vertical profiles
        h_profile = gray[center_y, :]
        v_profile = gray[:, center_x]

        # Analyze profiles
        h_analysis = self.intensity_analyzer.analyze_intensity_profile(h_profile)
        v_analysis = self.intensity_analyzer.analyze_intensity_profile(v_profile)

        # Adaptive feature detection
        lines, line_frame = self.detect_lines_adaptive(frame)
        circles, circle_frame = self.detect_circles_adaptive(frame)

        # Combine detection results
        combined_frame = frame.copy()

        # Draw lines
        if lines is not None:
            for line in lines:
                if len(line[0]) == 4:  # Probabilistic Hough
                    x1, y1, x2, y2 = line[0]
                    cv2.line(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw circles
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(combined_frame, (x, y), r, (255, 0, 0), 2)
                cv2.circle(combined_frame, (x, y), 2, (0, 0, 255), 3)

        # Update adaptive learning
        if self.frame_count % self.update_frequency == 0:
            self._update_neural_networks()

        results = {
            'frame': combined_frame,
            'lines': lines,
            'circles': circles,
            'line_count': len(lines) if lines is not None else 0,
            'circle_count': len(circles) if circles is not None else 0,
            'intensity_analysis': {
                'horizontal': h_analysis,
                'vertical': v_analysis
            },
            'frame_count': self.frame_count,
            'adaptive_performance': self._get_performance_metrics()
        }

        return results

    def _store_lines_experience(self, features: np.ndarray, params: np.ndarray, 
                               lines: Optional[np.ndarray], confidence: float):
        """Store experience for lines detection learning."""
        # Calculate reward based on detection quality
        reward = self._calculate_lines_reward(lines, confidence)

        experience = {
            'features': features.copy(),
            'params': params.copy(),
            'reward': reward,
            'timestamp': time.time()
        }

        self.lines_memory.append(experience)

    def _store_circles_experience(self, features: np.ndarray, params: np.ndarray,
                                 circles: Optional[np.ndarray], confidence: float):
        """Store experience for circles detection learning."""
        # Calculate reward based on detection quality
        reward = self._calculate_circles_reward(circles, confidence)

        experience = {
            'features': features.copy(),
            'params': params.copy(),
            'reward': reward,
            'timestamp': time.time()
        }

        self.circles_memory.append(experience)

    def _calculate_lines_reward(self, lines: Optional[np.ndarray], confidence: float) -> float:
        """Calculate reward for lines detection based on quality metrics."""
        if lines is None:
            return 0.1  # Small reward for no false positives

        num_lines = len(lines)

        # Reward based on reasonable number of lines (not too many, not too few)
        if 1 <= num_lines <= 10:
            count_reward = 1.0
        elif num_lines == 0:
            count_reward = 0.2
        else:
            count_reward = max(0.1, 1.0 / (num_lines * 0.1))

        # Confidence component
        confidence_reward = confidence

        # Combine rewards
        total_reward = 0.7 * count_reward + 0.3 * confidence_reward

        return float(total_reward)

    def _calculate_circles_reward(self, circles: Optional[np.ndarray], confidence: float) -> float:
        """Calculate reward for circles detection based on quality metrics."""
        if circles is None:
            return 0.1  # Small reward for no false positives

        num_circles = len(circles)

        # Reward based on reasonable number of circles
        if 1 <= num_circles <= 5:
            count_reward = 1.0
        elif num_circles == 0:
            count_reward = 0.2
        else:
            count_reward = max(0.1, 1.0 / (num_circles * 0.2))

        # Confidence component
        confidence_reward = confidence

        # Combine rewards
        total_reward = 0.7 * count_reward + 0.3 * confidence_reward

        return float(total_reward)

    def _update_neural_networks(self):
        """Update neural networks using experience replay."""
        # Update lines network
        if len(self.lines_memory) > 32:
            self._train_network(self.lines_nn, self.lines_optimizer, self.lines_memory, 'lines')

        # Update circles network
        if len(self.circles_memory) > 32:
            self._train_network(self.circles_nn, self.circles_optimizer, self.circles_memory, 'circles')

    def _train_network(self, network: nn.Module, optimizer: optim.Optimizer, 
                      memory: deque, network_type: str):
        """Train a neural network using experience replay."""
        # Sample batch from memory
        batch_size = min(32, len(memory))
        batch = np.random.choice(list(memory), size=batch_size, replace=False)

        features = torch.FloatTensor([exp['features'] for exp in batch]).to(self.device)
        params = torch.FloatTensor([exp['params'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)

        # Forward pass
        pred_params, pred_confidence = network(features)

        # Calculate losses
        param_loss = F.mse_loss(pred_params, params)
        confidence_loss = F.mse_loss(pred_confidence.squeeze(), rewards)

        total_loss = param_loss + 0.5 * confidence_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        # Log training progress
        if self.frame_count % (self.update_frequency * 10) == 0:
            logging.info(f"{network_type.title()} network - Loss: {total_loss.item():.4f}, "
                        f"Param Loss: {param_loss.item():.4f}, Conf Loss: {confidence_loss.item():.4f}")

    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        lines_avg_reward = np.mean([exp['reward'] for exp in list(self.lines_memory)[-100:]]) if self.lines_memory else 0.0
        circles_avg_reward = np.mean([exp['reward'] for exp in list(self.circles_memory)[-100:]]) if self.circles_memory else 0.0

        return {
            'lines_performance': float(lines_avg_reward),
            'circles_performance': float(circles_avg_reward),
            'memory_usage': {
                'lines': len(self.lines_memory),
                'circles': len(self.circles_memory)
            }
        }

    def save_models(self, path: Optional[str] = None):
        """Save trained models and configuration."""
        if path is None:
            path = self.model_save_path
        else:
            path = Path(path)

        path.mkdir(exist_ok=True)

        # Save PyTorch models
        torch.save(self.lines_nn.state_dict(), path / "lines_nn.pth")
        torch.save(self.circles_nn.state_dict(), path / "circles_nn.pth")

        # Save configuration and performance history
        config = {
            'adaptive_thresholds': self.adaptive_thresholds,
            'performance_history': self.performance_history,
            'frame_count': self.frame_count
        }

        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save recent experience for transfer learning
        recent_lines = list(self.lines_memory)[-100:] if len(self.lines_memory) > 100 else list(self.lines_memory)
        recent_circles = list(self.circles_memory)[-100:] if len(self.circles_memory) > 100 else list(self.circles_memory)

        with open(path / "recent_experience.pkl", 'wb') as f:
            pickle.dump({'lines': recent_lines, 'circles': recent_circles}, f)

        logging.info(f"Models and configuration saved to {path}")

    def _load_models(self, path: Optional[str] = None):
        """Load trained models and configuration."""
        if path is None:
            path = self.model_save_path
        else:
            path = Path(path)

        if not path.exists():
            logging.info("No saved models found, starting fresh")
            return

        try:
            # Load PyTorch models
            lines_model_path = path / "lines_nn.pth"
            circles_model_path = path / "circles_nn.pth"

            if lines_model_path.exists():
                self.lines_nn.load_state_dict(torch.load(lines_model_path, map_location=self.device))
                logging.info("Lines neural network loaded successfully")

            if circles_model_path.exists():
                self.circles_nn.load_state_dict(torch.load(circles_model_path, map_location=self.device))
                logging.info("Circles neural network loaded successfully")

            # Load configuration
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)

                self.adaptive_thresholds = config.get('adaptive_thresholds', self.adaptive_thresholds)
                self.performance_history = config.get('performance_history', self.performance_history)
                self.frame_count = config.get('frame_count', 0)

            # Load recent experience
            experience_path = path / "recent_experience.pkl"
            if experience_path.exists():
                with open(experience_path, 'rb') as f:
                    experience = pickle.load(f)

                for exp in experience['lines']:
                    self.lines_memory.append(exp)

                for exp in experience['circles']:
                    self.circles_memory.append(exp)

                logging.info(f"Loaded {len(experience['lines'])} lines and {len(experience['circles'])} circles experiences")

        except Exception as e:
            logging.warning(f"Error loading models: {e}")

    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about adaptive performance."""
        return {
            'frame_count': self.frame_count,
            'neural_networks': {
                'lines': {
                    'parameters': sum(p.numel() for p in self.lines_nn.parameters()),
                    'memory_size': len(self.lines_memory)
                },
                'circles': {
                    'parameters': sum(p.numel() for p in self.circles_nn.parameters()),
                    'memory_size': len(self.circles_memory)
                }
            },
            'performance_metrics': self._get_performance_metrics(),
            'adaptive_thresholds': self.adaptive_thresholds.copy(),
            'device': str(self.device)
        }

    def reset_learning(self):
        """Reset the learning system to start fresh."""
        self.lines_memory.clear()
        self.circles_memory.clear()
        self.frame_count = 0
        self.performance_history = {
            'lines': {'precision': [], 'recall': [], 'f1': []},
            'circles': {'precision': [], 'recall': [], 'f1': []}
        }

        # Reinitialize neural networks
        self.lines_nn = FeatureDetectionNN(input_size=64, output_size=8).to(self.device)
        self.circles_nn = FeatureDetectionNN(input_size=64, output_size=7).to(self.device)

        self.lines_optimizer = optim.Adam(self.lines_nn.parameters(), lr=self.learning_rate)
        self.circles_optimizer = optim.Adam(self.circles_nn.parameters(), lr=self.learning_rate)

        logging.info("Learning system reset successfully")


class AdaptiveFeatureProcessor:
    """
    High-level processor that integrates with existing systems.
    This provides a simple interface compatible with the existing codebase.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the adaptive processor.

        Args:
            use_gpu (bool): Whether to use GPU if available
        """
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.detector = AdaptiveFeatureDetector(device=device)
        self.processing_enabled = True

        logging.info(f"AdaptiveFeatureProcessor initialized on device: {device}")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with adaptive feature detection.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Processed frame with detections
        """
        if not self.processing_enabled or frame is None:
            return frame

        try:
            results = self.detector.process_frame_comprehensive(frame)
            return results['frame']
        except Exception as e:
            logging.error(f"Error in adaptive processing: {e}")
            return frame

    def toggle_processing(self) -> bool:
        """Toggle processing on/off."""
        self.processing_enabled = not self.processing_enabled
        return self.processing_enabled

    def is_processing_enabled(self) -> bool:
        """Check if processing is enabled."""
        return self.processing_enabled

    def get_detector(self) -> AdaptiveFeatureDetector:
        """Get the underlying adaptive detector."""
        return self.detector

    def save_model(self, path: Optional[str] = None):
        """Save the adaptive model."""
        self.detector.save_models(path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return self.detector.get_adaptive_statistics()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create adaptive processor
    processor = AdaptiveFeatureProcessor(use_gpu=torch.cuda.is_available())

    print("Adaptive Feature Detection System Initialized Successfully!")
    print(f"Device: {processor.get_detector().device}")
    print(f"Neural Networks: {processor.get_statistics()['neural_networks']}")

    # Example with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    processed_frame = processor.process_frame(dummy_frame)

    print(f"Test frame processed successfully. Shape: {processed_frame.shape}")

    # Save model
    processor.save_model()
    print("Model saved successfully!")
