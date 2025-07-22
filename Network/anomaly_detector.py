#!/usr/bin/env python3
"""
Anomaly Detector module for Fiber Optics Neural Network
"take the absolute value of the difference between the two and it will also look at 
the structural similarity index between the two to find the defects or anomalies"
"the change in regions within an image such as the change in pixels between the core 
to cladding or cladding to ferrule are not anomalies"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import cv2

from config_loader import get_config
from logger import get_logger

class AnomalyDetector(nn.Module):
    """
    Main anomaly detection network
    "I get a fully detailed anomaly detection while also classifying most probable features"
    """
    
    def __init__(self):
        super().__init__()
        print(f"[{datetime.now()}] Initializing AnomalyDetector")
        print(f"[{datetime.now()}] Previous script: reference_comparator.py")
        
        self.config = get_config()
        
        # Anomaly detection network
        self.anomaly_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Anomaly classifier
        self.anomaly_classifier = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)  # Anomaly probability map
        )
        
        # Edge-aware anomaly detection
        # "the change in regions within an image... are not anomalies"
        self.edge_suppressor = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # gradient_map + region_boundaries
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Defect type classifier
        self.defect_classifier = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, 1)  # scratch, contamination, misalignment, crack, other
        )
        
        print(f"[{datetime.now()}] AnomalyDetector initialized")
    
    def forward(self, input_tensor: torch.Tensor, reference_tensor: torch.Tensor,
                gradient_map: torch.Tensor, region_boundaries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies using multiple methods
        """
        print(f"[{datetime.now()}] AnomalyDetector.forward: Processing anomaly detection")
        
        # Encode input features
        input_features = self.anomaly_encoder(input_tensor)
        reference_features = self.anomaly_encoder(reference_tensor)
        
        # Feature difference
        feature_diff = torch.abs(input_features - reference_features)
        
        # Anomaly probability map
        anomaly_logits = self.anomaly_classifier(feature_diff)
        anomaly_probs = torch.sigmoid(anomaly_logits)
        
        # Suppress anomalies at region boundaries
        edge_input = torch.cat([gradient_map, region_boundaries], dim=1)
        edge_suppression = self.edge_suppressor(edge_input)
        
        # Apply suppression
        # "the change in regions... are not anomalies"
        suppressed_anomalies = anomaly_probs * (1 - edge_suppression)
        
        # Classify defect types where anomalies are detected
        defect_logits = self.defect_classifier(feature_diff)
        defect_probs = F.softmax(defect_logits, dim=1)
        
        return {
            'anomaly_map': suppressed_anomalies.squeeze(1),
            'raw_anomaly_map': anomaly_probs.squeeze(1),
            'edge_suppression': edge_suppression.squeeze(1),
            'defect_type_logits': defect_logits,
            'defect_type_probs': defect_probs,
            'feature_difference': feature_diff
        }

class StructuralAnomalyDetector:
    """
    Detects anomalies using structural analysis
    "it will also look at the structural similarity index between the two to find the defects"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("StructuralAnomalyDetector")
        
        self.logger.log_class_init("StructuralAnomalyDetector")
    
    def detect_anomalies(self, input_tensor: torch.Tensor, 
                        reference_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies using structural similarity
        """
        self.logger.log_function_entry("detect_anomalies")
        
        # Calculate SSIM
        ssim_map = self._calculate_ssim_map(input_tensor, reference_tensor)
        
        # Low SSIM indicates anomaly
        anomaly_map = 1 - ssim_map
        
        # Calculate absolute difference
        # "take the absolute value of the difference between the two"
        pixel_diff = torch.abs(input_tensor - reference_tensor)
        mean_diff = pixel_diff.mean(dim=1 if input_tensor.dim() == 4 else 0)
        
        # Local anomaly heatmap
        # "it will also look at the local anomaly heat map"
        local_anomaly_map = self._calculate_local_anomalies(mean_diff)
        
        # Combine methods
        # "compare the local anomaly heatmap with the structural similarity index and combine them"
        combined_anomalies = torch.max(anomaly_map, local_anomaly_map)
        
        self.logger.log_function_exit("detect_anomalies")
        
        return {
            'ssim_anomaly_map': anomaly_map,
            'pixel_difference_map': mean_diff,
            'local_anomaly_map': local_anomaly_map,
            'combined_anomaly_map': combined_anomalies
        }
    
    def _calculate_ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate pixel-wise SSIM map"""
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale if needed
        if x.dim() == 4:
            x = x.mean(dim=1, keepdim=True)
            y = y.mean(dim=1, keepdim=True)
        elif x.dim() == 3:
            x = x.mean(dim=0, keepdim=True).unsqueeze(0)
            y = y.mean(dim=0, keepdim=True).unsqueeze(0)
        
        # Local statistics
        kernel_size = 11
        mu_x = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
        mu_y = F.avg_pool2d(y, kernel_size, stride=1, padding=kernel_size//2)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, kernel_size, stride=1, padding=kernel_size//2) - mu_xy
        
        # SSIM
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim.squeeze()
    
    def _calculate_local_anomalies(self, diff_map: torch.Tensor) -> torch.Tensor:
        """Calculate local anomaly heatmap"""
        # Local statistics
        kernel_size = 15
        local_mean = F.avg_pool2d(
            diff_map.unsqueeze(0).unsqueeze(0) if diff_map.dim() == 2 else diff_map.unsqueeze(1),
            kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        
        local_std = torch.sqrt(
            F.avg_pool2d(
                diff_map.unsqueeze(0).unsqueeze(0) ** 2 if diff_map.dim() == 2 else diff_map.unsqueeze(1) ** 2,
                kernel_size,
                stride=1,
                padding=kernel_size//2
            ) - local_mean ** 2
        )
        
        # Z-score
        z_score = (diff_map.unsqueeze(0).unsqueeze(0) if diff_map.dim() == 2 else diff_map.unsqueeze(1) - local_mean) / (local_std + 1e-8)
        
        # High z-score indicates anomaly
        anomaly_map = torch.sigmoid(z_score - 2)  # Threshold at 2 standard deviations
        
        return anomaly_map.squeeze()

class RegionBoundaryDetector:
    """
    Detects region boundaries to avoid false anomaly detection
    "the program will forcibly look for all lines of best fit based on gradient trends"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("RegionBoundaryDetector")
        
        self.logger.log_class_init("RegionBoundaryDetector")
    
    def detect_boundaries(self, region_probs: torch.Tensor, 
                         gradient_map: torch.Tensor) -> torch.Tensor:
        """
        Detect region boundaries
        """
        # Get region predictions
        region_preds = region_probs.argmax(dim=1)
        
        # Detect edges using Sobel
        boundaries = self._detect_edges(region_preds.float())
        
        # Combine with gradient information
        # High gradients at region transitions are expected
        gradient_edges = self._detect_gradient_edges(gradient_map)
        
        # Region transitions have both boundary and gradient changes
        region_boundaries = boundaries * gradient_edges
        
        # Dilate boundaries to account for transition zones
        kernel = torch.ones(1, 1, 5, 5, device=boundaries.device)
        dilated_boundaries = F.conv2d(
            region_boundaries.unsqueeze(1),
            kernel,
            padding=2
        )
        dilated_boundaries = torch.clamp(dilated_boundaries, 0, 1)
        
        return dilated_boundaries.squeeze(1)
    
    def _detect_edges(self, image: torch.Tensor) -> torch.Tensor:
        """Detect edges using Sobel filters"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
        
        # Add channel dimension if needed
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)
        
        # Apply Sobel
        edges_x = F.conv2d(image, sobel_x, padding=1)
        edges_y = F.conv2d(image, sobel_y, padding=1)
        
        # Magnitude
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        
        # Normalize
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        return edges.squeeze()
    
    def _detect_gradient_edges(self, gradient_map: torch.Tensor) -> torch.Tensor:
        """Detect edges in gradient map"""
        # High gradients indicate edges
        gradient_edges = gradient_map > gradient_map.mean() + gradient_map.std()
        
        return gradient_edges.float()

class DefectLocator:
    """
    Locates and classifies specific defects
    "find the total anomalies or defects and their locations"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("DefectLocator")
        
        self.logger.log_class_init("DefectLocator")
        
        # Defect types
        self.defect_types = ['scratch', 'contamination', 'misalignment', 'crack', 'other']
    
    def locate_defects(self, anomaly_map: torch.Tensor, 
                      defect_probs: Optional[torch.Tensor] = None,
                      min_size: int = 10) -> List[Dict]:
        """
        Locate individual defects in anomaly map
        """
        self.logger.log_function_entry("locate_defects")
        
        # Threshold anomaly map
        binary_map = (anomaly_map > self.config.ANOMALY_THRESHOLD).float()
        
        # Convert to numpy for connected components
        if binary_map.dim() > 2:
            binary_map = binary_map.squeeze()
        
        binary_np = binary_map.cpu().numpy().astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_np)
        
        defects = []
        
        for label_id in range(1, num_labels):  # Skip background (0)
            # Get defect mask
            defect_mask = (labels == label_id)
            
            # Check size
            if defect_mask.sum() < min_size:
                continue
            
            # Get bounding box
            coords = np.where(defect_mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Get center
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            
            # Get defect type if available
            defect_type = 'unknown'
            confidence = 0.0
            
            if defect_probs is not None:
                # Get average probabilities in defect region
                defect_region_probs = defect_probs[:, y_min:y_max+1, x_min:x_max+1].mean(dim=[1, 2])
                type_idx = defect_region_probs.argmax().item()
                defect_type = self.defect_types[type_idx]
                confidence = defect_region_probs[type_idx].item()
            
            # Get severity
            severity = anomaly_map[defect_mask].mean().item()
            
            defect_info = {
                'id': label_id,
                'type': defect_type,
                'confidence': confidence,
                'severity': severity,
                'location': (center_x, center_y),
                'bounding_box': (x_min, y_min, x_max, y_max),
                'size': defect_mask.sum(),
                'mask': defect_mask
            }
            
            defects.append(defect_info)
        
        self.logger.info(f"Located {len(defects)} defects")
        self.logger.log_function_exit("locate_defects", f"{len(defects)} defects")
        
        return defects

class ComprehensiveAnomalyDetector:
    """
    Complete anomaly detection system combining all methods
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing ComprehensiveAnomalyDetector")
        
        self.config = get_config()
        self.logger = get_logger("ComprehensiveAnomalyDetector")
        
        self.logger.log_class_init("ComprehensiveAnomalyDetector")
        
        # Initialize components
        self.neural_detector = AnomalyDetector()
        self.structural_detector = StructuralAnomalyDetector()
        self.boundary_detector = RegionBoundaryDetector()
        self.defect_locator = DefectLocator()
        
        # Move neural detector to device
        self.device = self.config.get_device()
        self.neural_detector = self.neural_detector.to(self.device)
        
        self.logger.info("ComprehensiveAnomalyDetector initialized")
        print(f"[{datetime.now()}] ComprehensiveAnomalyDetector initialized successfully")
    
    def detect_anomalies(self, input_tensor: torch.Tensor,
                        reference_tensor: torch.Tensor,
                        region_probs: torch.Tensor,
                        gradient_map: torch.Tensor) -> Dict:
        """
        Comprehensive anomaly detection
        """
        self.logger.log_process_start("Comprehensive Anomaly Detection")
        
        # Detect region boundaries
        region_boundaries = self.boundary_detector.detect_boundaries(region_probs, gradient_map)
        
        # Neural network based detection
        neural_results = self.neural_detector(
            input_tensor, reference_tensor, gradient_map, region_boundaries.unsqueeze(0)
        )
        
        # Structural similarity based detection
        structural_results = self.structural_detector.detect_anomalies(
            input_tensor, reference_tensor
        )
        
        # Combine results
        # "compare the local anomaly heatmap with the structural similarity index and combine them"
        combined_anomaly_map = torch.max(
            neural_results['anomaly_map'],
            structural_results['combined_anomaly_map']
        )
        
        # Locate individual defects
        defects = self.defect_locator.locate_defects(
            combined_anomaly_map,
            neural_results['defect_type_probs'].squeeze(0) if neural_results['defect_type_probs'].dim() > 3 else neural_results['defect_type_probs']
        )
        
        # Log results
        self.logger.log_anomaly_detection(len(defects), [(d['location']) for d in defects[:5]])
        
        results = {
            'anomaly_map': combined_anomaly_map,
            'neural_anomaly_map': neural_results['anomaly_map'],
            'structural_anomaly_map': structural_results['combined_anomaly_map'],
            'defects': defects,
            'defect_count': len(defects),
            'region_boundaries': region_boundaries,
            'defect_type_probs': neural_results['defect_type_probs']
        }
        
        self.logger.log_process_end("Comprehensive Anomaly Detection")
        
        return results

# Test the anomaly detector
if __name__ == "__main__":
    detector = ComprehensiveAnomalyDetector()
    logger = get_logger("AnomalyDetectorTest")
    
    logger.log_process_start("Anomaly Detector Test")
    
    # Create test tensors
    device = detector.device
    test_input = torch.randn(1, 3, 256, 256).to(device)
    test_reference = torch.randn(1, 3, 256, 256).to(device)
    
    # Create test region probabilities
    test_region_probs = torch.softmax(torch.randn(1, 3, 256, 256).to(device), dim=1)
    
    # Create test gradient map
    test_gradient = torch.rand(1, 1, 256, 256).to(device)
    
    # Detect anomalies
    results = detector.detect_anomalies(
        test_input,
        test_reference,
        test_region_probs,
        test_gradient
    )
    
    # Log results
    logger.info(f"Detected {results['defect_count']} defects")
    logger.info(f"Anomaly map range: [{results['anomaly_map'].min():.4f}, {results['anomaly_map'].max():.4f}]")
    
    for i, defect in enumerate(results['defects'][:3]):
        logger.info(f"Defect {i+1}: type={defect['type']}, severity={defect['severity']:.4f}, location={defect['location']}")
    
    logger.log_process_end("Anomaly Detector Test")
    logger.log_script_transition("anomaly_detector.py", "integrated_network.py")
    
    print(f"[{datetime.now()}] Anomaly detector test completed")
    print(f"[{datetime.now()}] Next script: integrated_network.py")
