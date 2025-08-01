#!/usr/bin/env python3
"""
Enhanced Geometric Core Detector with PyTorch Learning
Extracts geometric core detection from live_feed.py and implements
PyTorch-based learning for automatic detection alignment.
Enhanced with multi-scale detection for small and large circles.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import pickle

# Import configuration system
try:
    from config_manager import ConfigManager
except ImportError:
    # Fallback configuration
    class ConfigManager:
        def __init__(self, config_file="config.json"):
            self.config = {
                "auto_core_detection": {
                    "detection": {
                        "enable_geometric_detection": True,
                        "enable_improved_detection": True,
                        "enable_manual_learning": True,
                        "min_confidence": 0.3,
                        "max_confidence": 1.0,
                        "detection_timeout": 0.2,
                        "enable_parallel_detection": True,
                        "max_detection_workers": 4
                    },
                    "hough_circles": {
                        "dp": 2.0,
                        "min_dist": 150,
                        "param1": 50,
                        "param2": 25,
                        "min_radius_small": 5,
                        "max_radius_small": 50,
                        "min_radius_medium": 15,
                        "max_radius_medium": 150,
                        "min_radius_large": 50,
                        "max_radius_large": 500,
                        "enable_adaptive_parameters": True,
                        "adaptive_scale_factor": 0.1
                    },
                    "preprocessing": {
                        "enable_clahe": True,
                        "clahe_clip_limit": 2.0,
                        "clahe_tile_grid_size": 8,
                        "enable_gaussian_blur": True,
                        "gaussian_kernel_size": 7,
                        "gaussian_sigma": 1.5,
                        "enable_median_blur": False,
                        "median_kernel_size": 5,
                        "enable_bilateral_filter": False,
                        "bilateral_d": 9,
                        "bilateral_sigma_color": 75,
                        "bilateral_sigma_space": 75
                    }
                }
            }
        
        def get_auto_core_detection_config(self):
            return self.config.get("auto_core_detection", {})


@dataclass
class DetectionResult:
    """Container for detection results"""
    center: Tuple[float, float]
    radius: float
    confidence: float
    method: str
    timestamp: float
    scale: str = "medium"  # small, medium, large
    features: Optional[Dict[str, Any]] = None


class MultiScaleIntensityProfileExtractor:
    """Extract intensity profiles and image characteristics for multi-scale detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_size = config.get("feature_extraction", {}).get("intensity_profile_size", 64)
        
    def extract_intensity_profile(self, image: np.ndarray, center: Tuple[int, int], 
                                radius: int) -> np.ndarray:
        """Extract radial intensity profile"""
        try:
            # Create circular mask
            mask = np.zeros_like(image, dtype=np.uint8)
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(mask, center_int, radius, 255, -1)
            
            # Extract intensity values within circle
            masked_image = cv2.bitwise_and(image, mask)
            intensities = masked_image[mask > 0]
            
            if len(intensities) == 0:
                return np.zeros(self.feature_size)
            
            # Create radial profile
            profile = np.zeros(self.feature_size)
            for i in range(self.feature_size):
                r = int(radius * i / self.feature_size)
                if r < radius:
                    circle_mask = np.zeros_like(image, dtype=np.uint8)
                    cv2.circle(circle_mask, center, r, 255, -1)
                    ring_intensities = masked_image[circle_mask > 0]
                    if len(ring_intensities) > 0:
                        profile[i] = np.mean(ring_intensities)
            
            return profile
            
        except Exception as e:
            print(f"Error extracting intensity profile: {e}")
            return np.zeros(self.feature_size)
    
    def extract_image_characteristics(self, image: np.ndarray, center: Tuple[int, int], 
                                    radius: int) -> Dict[str, float]:
        """Extract image characteristics around the detection"""
        try:
            # Create circular mask
            mask = np.zeros_like(image, dtype=np.uint8)
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(mask, center_int, radius, 255, -1)
            
            # Extract region of interest
            x, y = center
            r = radius
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
            
            roi = image[y1:y2, x1:x2]
            roi_mask = mask[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {}
            
            # Calculate characteristics
            characteristics = {}
            
            # Mean intensity
            characteristics['mean_intensity'] = np.mean(roi[roi_mask > 0]) if np.any(roi_mask > 0) else 0
            
            # Standard deviation
            characteristics['std_intensity'] = np.std(roi[roi_mask > 0]) if np.any(roi_mask > 0) else 0
            
            # Contrast (difference from surrounding area)
            outer_mask = np.zeros_like(image, dtype=np.uint8)
            cv2.circle(outer_mask, center, radius + 10, 255, -1)
            cv2.circle(outer_mask, center, radius, 0, -1)
            outer_intensities = image[outer_mask > 0]
            if len(outer_intensities) > 0:
                characteristics['contrast'] = characteristics['mean_intensity'] - np.mean(outer_intensities)
            else:
                characteristics['contrast'] = 0
            
            # Gradient magnitude
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            characteristics['gradient_magnitude'] = np.mean(gradient_magnitude[roi_mask > 0]) if np.any(roi_mask > 0) else 0
            
            # Local binary pattern (simplified)
            characteristics['texture_variance'] = np.var(roi[roi_mask > 0]) if np.any(roi_mask > 0) else 0
            
            return characteristics
            
        except Exception as e:
            print(f"Error extracting image characteristics: {e}")
            return {}
    
    def extract_pixel_analysis(self, image: np.ndarray, center: Tuple[int, int], 
                              radius: int) -> Dict[str, Any]:
        """Extract detailed pixel analysis"""
        try:
            # Create circular mask
            mask = np.zeros_like(image, dtype=np.uint8)
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(mask, center_int, radius, 255, -1)
            
            # Extract pixel data
            masked_pixels = image[mask > 0]
            
            if len(masked_pixels) == 0:
                return {}
            
            analysis = {}
            
            # Histogram analysis
            hist, _ = np.histogram(masked_pixels, bins=256, range=(0, 255))
            analysis['histogram'] = hist.astype(np.float32) / np.sum(hist)
            
            # Statistical measures
            analysis['pixel_mean'] = np.mean(masked_pixels)
            analysis['pixel_std'] = np.std(masked_pixels)
            analysis['pixel_median'] = np.median(masked_pixels)
            analysis['pixel_min'] = np.min(masked_pixels)
            analysis['pixel_max'] = np.max(masked_pixels)
            
            # Percentiles
            analysis['pixel_25th'] = np.percentile(masked_pixels, 25)
            analysis['pixel_75th'] = np.percentile(masked_pixels, 75)
            
            # Edge density
            edges = cv2.Canny(image, 50, 150)
            edge_mask = cv2.bitwise_and(edges, mask)
            analysis['edge_density'] = np.sum(edge_mask > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
            
            return analysis
            
        except Exception as e:
            print(f"Error extracting pixel analysis: {e}")
            return {}


class EnhancedCoreDetectionNetwork(nn.Module):
    """Enhanced PyTorch neural network for multi-scale core detection learning"""
    
    def __init__(self, input_size: int = 77, config: Dict[str, Any] = None):
        super(EnhancedCoreDetectionNetwork, self).__init__()
        
        self.config = config or {}
        network_config = self.config.get("network", {})
        
        hidden_layers = network_config.get("hidden_layers", [128, 64, 32])
        dropout_rate = network_config.get("dropout_rate", 0.3)
        activation = network_config.get("activation_function", "relu")
        
        # Build feature extractor dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Regression head for x, y, radius
        self.regression_head = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # x, y, radius
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Scale classification head (small, medium, large)
        self.scale_head = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # 3 scales
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        regression = self.regression_head(features)
        confidence = self.confidence_head(features)
        scale = self.scale_head(features)
        return regression, confidence, scale


class EnhancedGeometricCoreDetector:
    """Enhanced geometric core detector with multi-scale detection and PyTorch learning capabilities"""
    
    def __init__(self, model_path: str = "core_detection_model.pth",
                 data_path: str = "detection_data.pkl", config_file: str = "config.json"):
        self.model_path = model_path
        self.data_path = data_path
        
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.get_auto_core_detection_config()
        
        # Feature extractor
        self.feature_extractor = MultiScaleIntensityProfileExtractor(self.config)
        
        # PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedCoreDetectionNetwork(config=self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.config.get("learning", {}).get("learning_rate", 0.001))
        self.criterion = nn.MSELoss()
        
        # Training data
        self.training_data = []
        self.load_model()
        self.load_training_data()
        
        # Detection history for learning
        self.detection_history = deque(maxlen=1000)
        
    def load_model(self):
        """Load trained model if available"""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Loaded model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def save_model(self):
        """Save trained model"""
        try:
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_training_data(self):
        """Load training data if available"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    self.training_data = pickle.load(f)
                print(f"Loaded {len(self.training_data)} training samples")
            except Exception as e:
                print(f"Error loading training data: {e}")
    
    def save_training_data(self):
        """Save training data"""
        try:
            with open(self.data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
            print(f"Training data saved to {self.data_path}")
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def multi_scale_geometric_detection(self, frame: np.ndarray) -> List[DetectionResult]:
        """Enhanced geometric detection with multi-scale detection"""
        start_time = time.time()
        results = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # Get preprocessing configuration
            preprocessing_config = self.config.get("preprocessing", {})
            
            # Enhanced preprocessing
            if preprocessing_config.get("enable_clahe", True):
                clahe = cv2.createCLAHE(
                    clipLimit=preprocessing_config.get("clahe_clip_limit", 2.0),
                    tileGridSize=(preprocessing_config.get("clahe_tile_grid_size", 8), 
                                preprocessing_config.get("clahe_tile_grid_size", 8))
                )
                gray = clahe.apply(gray)
            
            if preprocessing_config.get("enable_gaussian_blur", True):
                kernel_size = preprocessing_config.get("gaussian_kernel_size", 7)
                sigma = preprocessing_config.get("gaussian_sigma", 1.5)
                gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
            
            if preprocessing_config.get("enable_median_blur", False):
                kernel_size = preprocessing_config.get("median_kernel_size", 5)
                gray = cv2.medianBlur(gray, kernel_size)
            
            if preprocessing_config.get("enable_bilateral_filter", False):
                d = preprocessing_config.get("bilateral_d", 9)
                sigma_color = preprocessing_config.get("bilateral_sigma_color", 75)
                sigma_space = preprocessing_config.get("bilateral_sigma_space", 75)
                gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
            
            # Get Hough circles configuration
            hough_config = self.config.get("hough_circles", {})
            
            # Multi-scale detection
            scales = [
                ("small", hough_config.get("min_radius_small", 5), 
                 hough_config.get("max_radius_small", 50)),
                ("medium", hough_config.get("min_radius_medium", 15), 
                 hough_config.get("max_radius_medium", 150)),
                ("large", hough_config.get("min_radius_large", 50), 
                 hough_config.get("max_radius_large", 500))
            ]
            
            for scale_name, min_radius, max_radius in scales:
                # Adaptive parameters based on scale
                if hough_config.get("enable_adaptive_parameters", True):
                    adaptive_factor = hough_config.get("adaptive_scale_factor", 0.1)
                    if scale_name == "small":
                        param2 = hough_config.get("param2", 25) * (1 + adaptive_factor)
                        min_dist = max(min_radius * 2, hough_config.get("min_dist", 150))
                    elif scale_name == "large":
                        param2 = hough_config.get("param2", 25) * (1 - adaptive_factor)
                        min_dist = max(min_radius * 2, hough_config.get("min_dist", 150))
                    else:
                        param2 = hough_config.get("param2", 25)
                        min_dist = hough_config.get("min_dist", 150)
                else:
                    param2 = hough_config.get("param2", 25)
                    min_dist = hough_config.get("min_dist", 150)
                
                # Hough circle detection for this scale
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 
                    dp=hough_config.get("dp", 2.0),
                    minDist=min_dist,
                    param1=hough_config.get("param1", 50),
                    param2=param2,
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        center_x, center_y, radius = circle
                        
                        # Calculate confidence
                        confidence = self._calculate_circle_confidence(gray, center_x, center_y, radius)
                        
                        # Only include if confidence meets threshold
                        min_confidence = self.config.get("detection", {}).get("min_confidence", 0.3)
                        if confidence >= min_confidence:
                            results.append(DetectionResult(
                                center=(float(center_x), float(center_y)),
                                radius=float(radius),
                                confidence=confidence,
                                method=f"geometric_{scale_name}",
                                timestamp=start_time,
                                scale=scale_name
                            ))
            
            # Sort by confidence and remove duplicates
            results = self._remove_duplicate_detections(results)
            
        except Exception as e:
            print(f"Error in multi-scale geometric detection: {e}")
        
        return results
    
    def _remove_duplicate_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Remove duplicate detections based on proximity"""
        if not detections:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        filtered_detections = []
        for detection in sorted_detections:
            # Check if this detection is too close to any already accepted detection
            is_duplicate = False
            for accepted in filtered_detections:
                distance = np.sqrt((detection.center[0] - accepted.center[0])**2 + 
                                 (detection.center[1] - accepted.center[1])**2)
                if distance < min(detection.radius, accepted.radius):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def _calculate_circle_confidence(self, gray: np.ndarray, center_x: int,
                                   center_y: int, radius: int) -> float:
        """Enhanced confidence calculation for circle detection"""
        try:
            mask = np.zeros_like(gray)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            inside_mean = np.mean(gray[mask > 0])
            outside_mask = cv2.circle(np.zeros_like(gray), (center_x, center_y),
                                     radius + 10, 255, -1)
            outside_mask = cv2.circle(outside_mask, (center_x, center_y),
                                     radius, 0, -1)
            outside_mean = np.mean(gray[outside_mask > 0])
            
            # Enhanced confidence calculation
            contrast_ratio = abs(inside_mean - outside_mean) / max(
                inside_mean, outside_mean, 1)
            
            # Additional confidence factors
            confidence_config = self.config.get("confidence", {})
            confidence = contrast_ratio
            
            if confidence_config.get("enable_gradient_analysis", True):
                # Gradient analysis
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_confidence = np.mean(gradient_magnitude[mask > 0]) / 255.0
                confidence = (confidence + gradient_confidence) / 2
            
            if confidence_config.get("enable_edge_density", True):
                # Edge density analysis
                edges = cv2.Canny(gray, 50, 150)
                edge_mask = cv2.bitwise_and(edges, mask)
                edge_density = np.sum(edge_mask > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
                confidence = (confidence + edge_density) / 2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.0
    
    def extract_features(self, frame: np.ndarray, detection: DetectionResult) -> torch.Tensor:
        """Extract features for learning"""
        try:
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Extract intensity profile
            intensity_profile = self.feature_extractor.extract_intensity_profile(
                gray, detection.center, int(detection.radius)
            )
            
            # Extract image characteristics
            characteristics = self.feature_extractor.extract_image_characteristics(
                gray, detection.center, int(detection.radius)
            )
            
            # Extract pixel analysis
            pixel_analysis = self.feature_extractor.extract_pixel_analysis(
                gray, detection.center, int(detection.radius)
            )
            
            # Combine features
            features = []
            
            # Intensity profile (64 features)
            features.extend(intensity_profile)
            
            # Image characteristics (5 features)
            char_features = [
                characteristics.get('mean_intensity', 0),
                characteristics.get('std_intensity', 0),
                characteristics.get('contrast', 0),
                characteristics.get('gradient_magnitude', 0),
                characteristics.get('texture_variance', 0)
            ]
            features.extend(char_features)
            
            # Pixel analysis (8 features)
            pixel_features = [
                pixel_analysis.get('pixel_mean', 0),
                pixel_analysis.get('pixel_std', 0),
                pixel_analysis.get('pixel_median', 0),
                pixel_analysis.get('pixel_min', 0),
                pixel_analysis.get('pixel_max', 0),
                pixel_analysis.get('pixel_25th', 0),
                pixel_analysis.get('pixel_75th', 0),
                pixel_analysis.get('edge_density', 0)
            ]
            features.extend(pixel_features)
            
            # Normalize features
            features = np.array(features, dtype=np.float32)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return torch.tensor(features, dtype=torch.float32).to(self.device)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return torch.zeros(77, dtype=torch.float32).to(self.device)
    
    def learn_from_manual_detection(self, frame: np.ndarray, manual_center: Tuple[float, float], 
                                   manual_radius: float):
        """Learn from manual detection to improve automatic detection"""
        try:
            # Create manual detection result
            manual_detection = DetectionResult(
                center=manual_center,
                radius=manual_radius,
                confidence=1.0,  # Manual detection is considered perfect
                method="manual",
                timestamp=time.time(),
                scale=self._determine_scale(manual_radius)
            )
            
            # Extract features from manual detection
            features = self.extract_features(frame, manual_detection)
            
            # Create target values (normalized)
            height, width = frame.shape[:2]
            target_x = manual_center[0] / width
            target_y = manual_center[1] / height
            target_radius = manual_radius / max(width, height)
            
            target = torch.tensor([target_x, target_y, target_radius], 
                                dtype=torch.float32).to(self.device)
            target_confidence = torch.tensor([1.0], dtype=torch.float32).to(self.device)
            
            # Create scale target
            scale_target = torch.zeros(3, dtype=torch.float32).to(self.device)
            scale_idx = {"small": 0, "medium": 1, "large": 2}[manual_detection.scale]
            scale_target[scale_idx] = 1.0
            
            # Train the model
            self.model.train()
            self.optimizer.zero_grad()
            
            regression_output, confidence_output, scale_output = self.model(features.unsqueeze(0))
            
            # Calculate loss
            regression_loss = self.criterion(regression_output.squeeze(), target)
            confidence_loss = self.criterion(confidence_output.squeeze(), target_confidence)
            scale_loss = self.criterion(scale_output.squeeze(), scale_target)
            total_loss = regression_loss + confidence_loss + scale_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Store training data
            training_sample = {
                'features': features.cpu().numpy(),
                'target': target.cpu().numpy(),
                'scale_target': scale_target.cpu().numpy(),
                'manual_center': manual_center,
                'manual_radius': manual_radius,
                'scale': manual_detection.scale,
                'timestamp': time.time()
            }
            self.training_data.append(training_sample)
            
            print(f"Learned from manual detection: center={manual_center}, "
                  f"radius={manual_radius}, scale={manual_detection.scale}")
            print(f"Training loss: {total_loss.item():.6f}")
            
            # Save model and data periodically
            save_interval = self.config.get("learning", {}).get("save_interval", 10)
            if len(self.training_data) % save_interval == 0:
                self.save_model()
                self.save_training_data()
                
        except Exception as e:
            print(f"Error learning from manual detection: {e}")
    
    def _determine_scale(self, radius: float) -> str:
        """Determine scale based on radius"""
        hough_config = self.config.get("hough_circles", {})
        if radius <= hough_config.get("max_radius_small", 50):
            return "small"
        elif radius <= hough_config.get("max_radius_medium", 150):
            return "medium"
        else:
            return "large"
    
    def improved_detection(self, frame: np.ndarray) -> List[DetectionResult]:
        """Improved detection using learned model"""
        try:
            # First, get geometric detections
            geometric_results = self.multi_scale_geometric_detection(frame)
            
            if not geometric_results:
                return []
            
            improved_results = []
            
            for geometric_result in geometric_results:
                if geometric_result.confidence < 0.3:
                    continue
                
                # Extract features from geometric detection
                features = self.extract_features(frame, geometric_result)
                
                # Use model to improve detection
                self.model.eval()
                with torch.no_grad():
                    regression_output, confidence_output, scale_output = self.model(features.unsqueeze(0))
                    
                    # Denormalize predictions
                    height, width = frame.shape[:2]
                    predicted_x = regression_output[0, 0].item() * width
                    predicted_y = regression_output[0, 1].item() * height
                    predicted_radius = regression_output[0, 2].item() * max(width, height)
                    predicted_confidence = confidence_output[0, 0].item()
                    
                    # Get predicted scale
                    scale_probs = scale_output[0].cpu().numpy()
                    predicted_scale_idx = np.argmax(scale_probs)
                    scale_names = ["small", "medium", "large"]
                    predicted_scale = scale_names[predicted_scale_idx]
                    
                    # Combine with geometric detection
                    combined_center = (
                        (geometric_result.center[0] + predicted_x) / 2,
                        (geometric_result.center[1] + predicted_y) / 2
                    )
                    combined_radius = (geometric_result.radius + predicted_radius) / 2
                    combined_confidence = (geometric_result.confidence + predicted_confidence) / 2
                    
                    improved_results.append(DetectionResult(
                        center=combined_center,
                        radius=combined_radius,
                        confidence=combined_confidence,
                        method="improved",
                        timestamp=time.time(),
                        scale=predicted_scale
                    ))
            
            return improved_results
                
        except Exception as e:
            print(f"Error in improved detection: {e}")
            return self.multi_scale_geometric_detection(frame)
    
    def get_detection_history(self) -> List[DetectionResult]:
        """Get detection history for analysis"""
        return list(self.detection_history)
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
    
    def export_learning_data(self, output_path: str):
        """Export learning data for analysis"""
        try:
            export_data = {
                'training_data': self.training_data,
                'detection_history': list(self.detection_history),
                'model_info': {
                    'model_path': self.model_path,
                    'device': str(self.device),
                    'training_samples': len(self.training_data)
                },
                'config': self.config
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            print(f"Learning data exported to {output_path}")
            
        except Exception as e:
            print(f"Error exporting learning data: {e}")


def main():
    """Test the enhanced geometric core detector with learning"""
    print("Enhanced Geometric Core Detector with PyTorch Learning")
    print("This module provides:")
    print("1. Multi-scale geometric core detection (small, medium, large circles)")
    print("2. Enhanced PyTorch-based learning for automatic detection alignment")
    print("3. Advanced feature extraction for intensity profiles and image characteristics")
    print("4. Manual-to-automatic detection learning with scale classification")
    
    # Create detector
    detector = EnhancedGeometricCoreDetector()
    
    print(f"\nDetector initialized with device: {detector.device}")
    print(f"Model path: {detector.model_path}")
    print(f"Training data path: {detector.data_path}")
    print(f"Configuration loaded: {len(detector.config)} sections")


if __name__ == "__main__":
    main() 