#!/usr/bin/env python3
"""
Feature Extractor module for Fiber Optics Neural Network
"I want each feature to not only look for comparisons but also look for anomalies while comparing"
Performs simultaneous feature extraction, classification, and anomaly detection

This module now uses the UnifiedFeatureExtractor for all feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from config_loader import get_config
from logger import get_logger
from tensor_processor import TensorProcessor

class SimultaneousFeatureExtractor(nn.Module):
    """
    Extracts features while simultaneously comparing to normal and anomaly patterns
    "Extract features AND simultaneously: - Compare to normal patterns (for classification) 
    - Compare to anomaly patterns (for defect detection) - Assess feature quality - All at the same time"
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_patterns: int = 32):
        super().__init__()
        print(f"[{datetime.now()}] Initializing SimultaneousFeatureExtractor")
        print(f"[{datetime.now()}] Previous script: tensor_processor.py")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patterns = num_patterns
        
        # Feature extraction layers
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Learnable normal patterns for each region
        # "Core has certain expected gradient patterns - Cladding has different patterns"
        self.core_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))
        self.cladding_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))
        self.ferrule_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))
        
        # Learnable anomaly patterns
        self.anomaly_patterns = nn.Parameter(torch.randn(num_patterns, out_channels, 3, 3))
        
        # Pattern matching thresholds
        self.normal_threshold = nn.Parameter(torch.tensor(0.7))
        self.anomaly_threshold = nn.Parameter(torch.tensor(0.3))
        
        # Quality assessment network
        # Calculate actual input channels based on patterns
        # features (out_channels) + all_matches (num_patterns * 3 for each region) + anomaly_matches (num_patterns)
        # But all_matches concatenates individual matches, so it's the sum of individual pattern counts
        quality_input_channels = out_channels + num_patterns * 3 + num_patterns
        
        # Calculate expected input channels for quality assessor
        # features (out_channels) + all_matches (num_patterns * 3) + anomaly_matches (num_patterns)
        quality_input_channels = out_channels + num_patterns * 4
        
        # Create quality assessor with fixed input channels
        self.quality_assessor = nn.Sequential(
            nn.Conv2d(quality_input_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Simultaneous classifier with fixed input channels  
        self.simultaneous_classifier = nn.Conv2d(
            quality_input_channels,
            4,  # core, cladding, ferrule, anomaly
            1
        )
        
        print(f"[{datetime.now()}] SimultaneousFeatureExtractor initialized")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with simultaneous feature extraction and analysis
        """
        print(f"[{datetime.now()}] SimultaneousFeatureExtractor.forward: Processing tensor with shape {x.shape}")
        
        # Extract features
        features = self.feature_conv(x)
        
        # Simultaneously compare to normal patterns
        normal_matches = self._match_normal_patterns(features)
        
        # Simultaneously compare to anomaly patterns
        anomaly_matches = self._match_anomaly_patterns(features)
        
        # Assess feature quality
        quality_input = torch.cat([features, normal_matches['all_matches'], anomaly_matches], dim=1)
        quality_scores = self.quality_assessor(quality_input)
        
        # Simultaneous classification
        classifier_input = torch.cat([features, normal_matches['all_matches'], anomaly_matches], dim=1)
        classifications = self.simultaneous_classifier(classifier_input)
        
        # Separate region and anomaly predictions
        region_logits = classifications[:, :3, :, :]  # core, cladding, ferrule
        anomaly_logits = classifications[:, 3:4, :, :]  # anomaly
        
        return {
            'features': features,
            'region_logits': region_logits,
            'anomaly_logits': anomaly_logits,
            'normal_matches': normal_matches,
            'anomaly_matches': anomaly_matches,
            'quality_scores': quality_scores,
            'region_probs': F.softmax(region_logits, dim=1),
            'anomaly_probs': torch.sigmoid(anomaly_logits)
        }
    
    def _match_normal_patterns(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Match features against normal region patterns"""
        b, c, h, w = features.shape
        
        # Match against each region's patterns
        core_matches = []
        for i in range(self.core_patterns.shape[0]):
            pattern = self.core_patterns[i].unsqueeze(0)
            match = F.conv2d(features, pattern, padding=1)
            match = torch.sigmoid(match)
            core_matches.append(match)
        
        cladding_matches = []
        for i in range(self.cladding_patterns.shape[0]):
            pattern = self.cladding_patterns[i].unsqueeze(0)
            match = F.conv2d(features, pattern, padding=1)
            match = torch.sigmoid(match)
            cladding_matches.append(match)
        
        ferrule_matches = []
        for i in range(self.ferrule_patterns.shape[0]):
            pattern = self.ferrule_patterns[i].unsqueeze(0)
            match = F.conv2d(features, pattern, padding=1)
            match = torch.sigmoid(match)
            ferrule_matches.append(match)
        
        # Stack matches
        core_stack = torch.cat(core_matches, dim=1) if core_matches else torch.zeros(b, 1, h, w, device=features.device)
        cladding_stack = torch.cat(cladding_matches, dim=1) if cladding_matches else torch.zeros(b, 1, h, w, device=features.device)
        ferrule_stack = torch.cat(ferrule_matches, dim=1) if ferrule_matches else torch.zeros(b, 1, h, w, device=features.device)
        
        all_matches = torch.cat([core_stack, cladding_stack, ferrule_stack], dim=1)
        
        return {
            'core_matches': core_stack,
            'cladding_matches': cladding_stack,
            'ferrule_matches': ferrule_stack,
            'all_matches': all_matches
        }
    
    def _match_anomaly_patterns(self, features: torch.Tensor) -> torch.Tensor:
        """Match features against anomaly patterns"""
        anomaly_matches = []
        
        for i in range(self.anomaly_patterns.shape[0]):
            pattern = self.anomaly_patterns[i].unsqueeze(0)
            match = F.conv2d(features, pattern, padding=1)
            match = torch.sigmoid(match)
            anomaly_matches.append(match)
        
        return torch.cat(anomaly_matches, dim=1) if anomaly_matches else torch.zeros_like(features[:, :1, :, :])

# MultiScaleFeatureExtractor 
class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction with simultaneous classification and anomaly detection
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger("MultiScaleFeatureExtractor")
        
        self.logger.log_class_init("MultiScaleFeatureExtractor")
        
        # Feature extractors for each scale
        self.scale_extractors = nn.ModuleList()
        in_channels = 3
        
        # Get feature channels from config or use default
        feature_channels = getattr(self.config, 'feature_channels', [64, 128, 256, 512])
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'feature_channels'):
            feature_channels = self.config.model.feature_channels
        
        for i, out_channels in enumerate(feature_channels):
            extractor = SimultaneousFeatureExtractor(in_channels, out_channels)
            self.scale_extractors.append(extractor)
            in_channels = out_channels
        
        # Cross-scale correlation modules
        self.scale_correlators = nn.ModuleList()
        for i in range(len(feature_channels) - 1):
            correlator = nn.Conv2d(
                feature_channels[i] + feature_channels[i + 1],
                feature_channels[i + 1],
                1
            )
            self.scale_correlators.append(correlator)
        
        # Gradient and position weighting modules
        # "the weights of the neural network will be dependent on the average intensity gradient"
        self.gradient_weight_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, len(feature_channels)),
            nn.Sigmoid()
        )
        
        # "another weight will be dependent on the average pixel position"
        self.position_weight_net = nn.Sequential(
            nn.Linear(3, 16),  # avg_x, avg_y, avg_radial
            nn.ReLU(),
            nn.Linear(16, len(feature_channels)),
            nn.Sigmoid()
        )
        
        self.logger.info("MultiScaleFeatureExtractor initialized")
    
    def forward(self, x: torch.Tensor, gradient_info: Dict, position_info: Dict) -> Dict[str, List[torch.Tensor]]:
        """
        Extract features at multiple scales with correlation
        """
        self.logger.log_function_entry("forward", input_shape=x.shape)
        
        # Calculate gradient and position weights
        avg_gradient = gradient_info['average_gradient'].unsqueeze(0)
        gradient_weights = self.gradient_weight_net(avg_gradient)
        
        avg_positions = torch.stack([
            position_info['average_x'],
            position_info['average_y'],
            position_info['average_radial']
        ]).unsqueeze(0)
        position_weights = self.position_weight_net(avg_positions)
        
        # Combined weights
        scale_weights = gradient_weights * position_weights * self.config.GRADIENT_WEIGHT_FACTOR
        
        # Extract features at each scale
        all_features = []
        all_region_logits = []
        all_anomaly_logits = []
        all_quality_scores = []
        
        current = x
        
        for i, (extractor, weight) in enumerate(zip(self.scale_extractors, scale_weights[0])):
            # Extract features with weight modulation
            scale_results = extractor(current)
            
            # Apply scale weight
            weighted_features = scale_results['features'] * weight.item()
            
            # Correlate with previous scale if not first
            if i > 0 and (i-1) < len(self.scale_correlators):
                # Resize previous features to match current size
                prev_features = F.interpolate(
                    all_features[-1],
                    size=weighted_features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                # Correlate
                correlated = torch.cat([prev_features, weighted_features], dim=1)
                weighted_features = self.scale_correlators[i-1](correlated)
            
            # Store results
            all_features.append(weighted_features)
            all_region_logits.append(scale_results['region_logits'])
            all_anomaly_logits.append(scale_results['anomaly_logits'])
            all_quality_scores.append(scale_results['quality_scores'])
            
            # Use features as input for next scale
            current = weighted_features
            
            self.logger.debug(f"Scale {i}: features shape={weighted_features.shape}, weight={weight.item():.4f}")
        
        self.logger.log_function_exit("forward")
        
        return {
            'features': all_features,
            'region_logits': all_region_logits,
            'anomaly_logits': all_anomaly_logits,
            'quality_scores': all_quality_scores,
            'scale_weights': scale_weights
        }

class TrendAnalyzer(nn.Module):
    """
    Analyzes gradient and position trends to distinguish regions from defects
    "The network learns gradient and position trends for each region"
    "when a feature of an image follows this line its classified as region when it doesn't its classified as a defect"
    """
    
    def __init__(self, num_regions: int = 3):
        super().__init__()
        self.num_regions = num_regions
        
        # Learnable trend parameters for each region
        # Each region has gradient trend (slope, intercept) and position trend
        self.gradient_trends = nn.Parameter(torch.randn(num_regions, 2))
        self.position_trends = nn.Parameter(torch.randn(num_regions, 2))
        
        # Trend deviation network
        self.deviation_analyzer = nn.Sequential(
            nn.Conv2d(4, 32, 1),  # gradient_map, position_map, expected_gradient, expected_position
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, gradient_map: torch.Tensor, position_map: torch.Tensor, 
                region_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze if features follow expected trends
        """
        b, _, h, w = gradient_map.shape
        
        # Calculate expected trends for each region
        expected_gradients = torch.zeros(b, self.num_regions, h, w, device=gradient_map.device)
        expected_positions = torch.zeros(b, self.num_regions, h, w, device=gradient_map.device)
        
        for i in range(self.num_regions):
            # Expected gradient: y = mx + b
            expected_gradients[:, i] = (
                self.gradient_trends[i, 0] * gradient_map.squeeze(1) + 
                self.gradient_trends[i, 1]
            )
            
            # Expected position influence
            expected_positions[:, i] = (
                self.position_trends[i, 0] * position_map.squeeze(1) + 
                self.position_trends[i, 1]
            )
        
        # Weight by region probability
        weighted_expected_gradient = (expected_gradients * region_probs).sum(dim=1, keepdim=True)
        weighted_expected_position = (expected_positions * region_probs).sum(dim=1, keepdim=True)
        
        # Analyze deviation from expected trends
        deviation_input = torch.cat([
            gradient_map,
            position_map,
            weighted_expected_gradient,
            weighted_expected_position
        ], dim=1)
        
        # High deviation = likely defect
        deviation_score = self.deviation_analyzer(deviation_input)
        
        # Trend adherence is inverse of deviation
        trend_adherence = 1 - deviation_score
        
        return {
            'trend_adherence': trend_adherence,
            'deviation_score': deviation_score,
            'expected_gradients': expected_gradients,
            'expected_positions': expected_positions
        }

# Main feature extraction pipeline
class FeatureExtractionPipeline:
    """Complete feature extraction pipeline using UnifiedFeatureExtractor"""
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing FeatureExtractionPipeline")
        
        self.config = get_config()
        self.logger = get_logger("FeatureExtractionPipeline")
        self.tensor_processor = TensorProcessor()
        
        self.logger.log_class_init("FeatureExtractionPipeline")
        
        # Use multi-scale extractor
        self.multi_scale_extractor = MultiScaleFeatureExtractor()
        self.trend_analyzer = TrendAnalyzer()
        
        # Move to device
        self.device = self.config.get_device()
        self.multi_scale_extractor = self.multi_scale_extractor.to(self.device)
        self.trend_analyzer = self.trend_analyzer.to(self.device)
        
        self.logger.info("FeatureExtractionPipeline initialized")
        print(f"[{datetime.now()}] FeatureExtractionPipeline initialized successfully")
    
    def extract_features(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete feature extraction with simultaneous classification and anomaly detection
        """
        self.logger.log_process_start("Feature Extraction")
        
        # Calculate gradient and position information
        gradient_info = self.tensor_processor.calculate_gradient_intensity(tensor)
        position_info = self.tensor_processor.calculate_pixel_positions(tensor.shape)
        
        # Multi-scale feature extraction
        multi_scale_results = self.multi_scale_extractor(tensor, gradient_info, position_info)
        
        # Aggregate results across scales
        # Use the highest resolution for final output
        final_region_logits = multi_scale_results['region_logits'][-1]
        final_anomaly_logits = multi_scale_results['anomaly_logits'][-1]
        final_quality_scores = multi_scale_results['quality_scores'][-1]
        
        # Calculate region probabilities
        region_probs = F.softmax(final_region_logits, dim=1)
        
        # Trend analysis
        trend_results = self.trend_analyzer(
            gradient_info['gradient_map'],
            position_info['radial_positions'],
            region_probs
        )
        
        # Combine anomaly detection from patterns and trends
        # "Deviations from trends indicate anomalies"
        pattern_anomalies = torch.sigmoid(final_anomaly_logits)
        trend_anomalies = trend_results['deviation_score']
        combined_anomalies = torch.max(pattern_anomalies, trend_anomalies)
        
        results = {
            'multi_scale_features': multi_scale_results['features'],
            'region_logits': final_region_logits,
            'region_probs': region_probs,
            'anomaly_map': combined_anomalies.squeeze(1),
            'quality_map': final_quality_scores.squeeze(1),
            'trend_adherence': trend_results['trend_adherence'].squeeze(1),
            'gradient_info': gradient_info,
            'position_info': position_info,
            'scale_weights': multi_scale_results['scale_weights']
        }
        
        # Log results
        self.logger.log_region_classification({
            'core': region_probs[:, 0].mean().item(),
            'cladding': region_probs[:, 1].mean().item(),
            'ferrule': region_probs[:, 2].mean().item()
        })
        
        anomaly_locations = torch.where(combined_anomalies > self.config.ANOMALY_THRESHOLD)
        self.logger.log_anomaly_detection(
            len(anomaly_locations[0]),
            list(zip(anomaly_locations[2].tolist(), anomaly_locations[3].tolist()))[:5]  # First 5
        )
        
        self.logger.log_process_end("Feature Extraction")
        
        return results

# Test the feature extractor
if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    logger = get_logger("FeatureExtractorTest")
    
    logger.log_process_start("Feature Extractor Test")
    
    # Create test tensor
    test_tensor = torch.randn(1, 3, 256, 256).to(pipeline.device)
    
    # Extract features
    results = pipeline.extract_features(test_tensor)
    
    # Log results
    logger.info(f"Extracted {len(results['multi_scale_features'])} scale features")
    logger.info(f"Anomaly map shape: {results['anomaly_map'].shape}")
    logger.info(f"Max anomaly score: {results['anomaly_map'].max().item():.4f}")
    logger.info(f"Mean quality score: {results['quality_map'].mean().item():.4f}")
    
    logger.log_process_end("Feature Extractor Test")
    logger.log_script_transition("feature_extractor.py", "reference_comparator.py")
    
    print(f"[{datetime.now()}] Feature extractor test completed")
    print(f"[{datetime.now()}] Next script: reference_comparator.py")
