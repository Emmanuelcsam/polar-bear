#!/usr/bin/env python3
"""
Feature Extractor module for Fiber Optics Neural Network.
This module performs simultaneous feature extraction, classification, and anomaly detection,
with a focus on comparing input features to both normal and anomalous patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from core.config_loader import get_config
from core.logger import get_logger
from data.tensor_processor import TensorProcessor

class SimultaneousFeatureExtractor(nn.Module):
    """
    Extracts features while simultaneously comparing them to normal and anomaly patterns.

    This module is designed to:
    - Extract salient features from input tensors.
    - Compare features to learned "normal" patterns for classification (e.g., core, cladding).
    - Compare features to learned "anomaly" patterns for defect detection.
    - Assess the quality of the extracted features.
    """

    def __init__(self, in_channels: int, out_channels: int, num_patterns: int = 32):
        """
        Initializes the feature extractor's layers and learnable patterns.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for the feature maps.
            num_patterns (int): Number of patterns to learn for both normal and anomaly matching.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patterns = num_patterns

        # Feature extraction convolutional layers
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Learnable patterns for different fiber optic regions (normal patterns)
        # Each region (core, cladding, ferrule) has its own set of expected patterns.
        self.core_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))
        self.cladding_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))
        self.ferrule_patterns = nn.Parameter(torch.randn(num_patterns // 3, out_channels, 3, 3))

        # Learnable patterns for various types of anomalies
        self.anomaly_patterns = nn.Parameter(torch.randn(num_patterns, out_channels, 3, 3))
        
        # Learnable thresholds for pattern matching
        self.normal_threshold = nn.Parameter(torch.tensor(0.7))
        self.anomaly_threshold = nn.Parameter(torch.tensor(0.3))

        # ## FIX: Input channels are precisely calculated to prevent dimension mismatches.
        # The input to the quality assessor and classifier is a concatenation of the raw features,
        # the normal pattern matches, and the anomaly pattern matches. This calculation ensures
        # the convolutional layers are initialized with the correct input channel size.
        num_normal_patterns = 3 * (num_patterns // 3)  # Exact count for concatenated normal matches
        combined_input_channels = out_channels + num_normal_patterns + num_patterns

        # Quality assessment network to score the relevance of extracted features
        self.quality_assessor = nn.Sequential(
            nn.Conv2d(combined_input_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        # Classifier to distinguish between regions and anomalies simultaneously
        self.simultaneous_classifier = nn.Conv2d(
            combined_input_channels,
            4,  # Outputs: core, cladding, ferrule, anomaly
            1
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass for simultaneous feature extraction and analysis.
        
        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing extracted features, logits,
                                     probabilities, pattern matches, and quality scores.
        """
        # 1. Extract base features
        features = self.feature_conv(x)
        
        # 2. Compare to normal and anomaly patterns in parallel
        normal_matches = self._match_normal_patterns(features)
        anomaly_matches = self._match_anomaly_patterns(features)
        
        # 3. Combine features and match results for further analysis
        combined_input = torch.cat([features, normal_matches['all_matches'], anomaly_matches], dim=1)
        
        # 4. Assess feature quality and perform classification
        quality_scores = self.quality_assessor(combined_input)
        classifications = self.simultaneous_classifier(combined_input)
        
        # 5. Separate the output logits for regions and anomalies
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
    
    def _match_patterns(self, features: torch.Tensor, patterns: nn.Parameter) -> torch.Tensor:
        """Helper function to match features against a set of patterns."""
        matches = []
        for i in range(patterns.shape[0]):
            pattern = patterns[i].unsqueeze(0)
            # Use regular convolution (grouped convolution would require different pattern dimensions)
            match = F.conv2d(features, pattern, padding=1)
            match = torch.sigmoid(match)
            matches.append(match)
        if not matches:
            b, _, h, w = features.shape
            return torch.zeros(b, 0, h, w, device=features.device)
        return torch.cat(matches, dim=1)

    def _match_normal_patterns(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Match features against normal region-specific patterns."""
        b, _, h, w = features.shape
        core_matches = self._match_patterns(features, self.core_patterns)
        cladding_matches = self._match_patterns(features, self.cladding_patterns)
        ferrule_matches = self._match_patterns(features, self.ferrule_patterns)
        
        all_matches = torch.cat([core_matches, cladding_matches, ferrule_matches], dim=1)
        
        return {
            'core_matches': core_matches,
            'cladding_matches': cladding_matches,
            'ferrule_matches': ferrule_matches,
            'all_matches': all_matches
        }

    def _match_anomaly_patterns(self, features: torch.Tensor) -> torch.Tensor:
        """Match features against general anomaly patterns."""
        return self._match_patterns(features, self.anomaly_patterns)

class MultiScaleFeatureExtractor(nn.Module):
    """
    Applies the feature extraction process across multiple scales and correlates the results.
    This allows the network to detect features of varying sizes.
    """
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger("MultiScaleFeatureExtractor")
        self.logger.log_class_init("MultiScaleFeatureExtractor")

        # Dynamically create feature extractors for each scale defined in the config
        self.scale_extractors = nn.ModuleList()
        in_channels = 3  # Initial channels for an RGB image
        feature_channels = self.config.model.get('feature_channels', [64, 128, 256, 512])
        
        for out_channels in feature_channels:
            extractor = SimultaneousFeatureExtractor(in_channels, out_channels)
            self.scale_extractors.append(extractor)
            in_channels = out_channels

        # Modules to correlate features between adjacent scales
        self.scale_correlators = nn.ModuleList()
        for i in range(len(feature_channels) - 1):
            correlator = nn.Conv2d(
                feature_channels[i] + feature_channels[i + 1],
                feature_channels[i + 1],
                kernel_size=1
            )
            self.scale_correlators.append(correlator)

        # Networks to calculate scale weights based on image properties
        # The importance of each scale can be modulated by the image's gradient and pixel positions.
        self.gradient_weight_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, len(feature_channels)), nn.Sigmoid()
        )
        self.position_weight_net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, len(feature_channels)), nn.Sigmoid()
        )
        self.logger.info("MultiScaleFeatureExtractor initialized")
    
    def forward(self, x: torch.Tensor, gradient_info: Dict, position_info: Dict) -> Dict[str, List[torch.Tensor]]:
        """
        Extracts and correlates features across multiple scales.
        
        Args:
            x (torch.Tensor): The input tensor.
            gradient_info (Dict): Dictionary containing gradient information of the input.
            position_info (Dict): Dictionary containing positional information of the input.

        Returns:
            Dict: A dictionary containing lists of features, logits, and scores for each scale.
        """
        self.logger.log_function_entry("forward")
        b = x.shape[0]

        # 1. Calculate weights for each scale based on global image properties
        # ## FIX: Handles both batched and single-item inputs for gradients and positions.
        avg_gradient = gradient_info['average_gradient'].view(b, 1)
        gradient_weights = self.gradient_weight_net(avg_gradient)

        avg_positions = torch.stack([
            position_info['average_x'],
            position_info['average_y'],
            position_info['average_radial']
        ], dim=1)
        position_weights = self.position_weight_net(avg_positions)
        
        factor = self.config.model.get('gradient_weight_factor', 1.0)
        scale_weights = gradient_weights * position_weights * factor
        
        # 2. Process the input through each scale's extractor
        all_features, all_region_logits, all_anomaly_logits, all_quality_scores = [], [], [], []
        current_features = x

        for i, extractor in enumerate(self.scale_extractors):
            scale_results = extractor(current_features)
            
            # Modulate features with the calculated scale weight
            # ## FIX: Correctly reshapes weights for broadcasting across batch, channels, H, and W.
            weight = scale_weights[:, i:i+1].view(b, 1, 1, 1)
            weighted_features = scale_results['features'] * weight
            
            # Correlate with features from the previous scale (if applicable)
            if i > 0:
                prev_features_resized = F.interpolate(
                    all_features[-1], size=weighted_features.shape[-2:], mode='bilinear', align_corners=False
                )
                correlated_input = torch.cat([prev_features_resized, weighted_features], dim=1)
                weighted_features = self.scale_correlators[i-1](correlated_input)
            
            all_features.append(weighted_features)
            all_region_logits.append(scale_results['region_logits'])
            all_anomaly_logits.append(scale_results['anomaly_logits'])
            all_quality_scores.append(scale_results['quality_scores'])
            
            current_features = weighted_features # Input for the next scale
            self.logger.debug(f"Scale {i}: features shape={weighted_features.shape}, weight={weight.mean().item():.4f}")

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
    Analyzes gradient and position trends to distinguish normal regions from defects.
    The core idea is that normal regions follow predictable patterns (trends), while
    defects cause deviations from these trends.
    """
    def __init__(self, num_regions: int = 3):
        super().__init__()
        self.num_regions = num_regions
        
        # Learnable trend parameters (slope, intercept) for each region
        self.gradient_trends = nn.Parameter(torch.randn(num_regions, 2))  # [m, b] for y = mx + b
        self.position_trends = nn.Parameter(torch.randn(num_regions, 2))
        
        # Network to analyze deviations from the expected trends
        self.deviation_analyzer = nn.Sequential(
            nn.Conv2d(4, 32, 1), nn.ReLU(inplace=True),  # Inputs: actual gradient, actual position, expected gradient, expected position
            nn.Conv2d(32, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, gradient_map: torch.Tensor, position_map: torch.Tensor, region_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyzes if pixel-wise features follow the learned trends for their predicted region.
        
        Args:
            gradient_map (torch.Tensor): Map of gradient magnitudes.
            position_map (torch.Tensor): Map of radial positions.
            region_probs (torch.Tensor): Softmax probabilities for each region (core, cladding, ferrule).

        Returns:
            Dict: Contains trend adherence and deviation scores.
        """
        b, _, h, w = gradient_map.shape
        
        # 1. Calculate the expected gradient and position values for each region
        expected_gradients = torch.zeros(b, self.num_regions, h, w, device=gradient_map.device)
        expected_positions = torch.zeros(b, self.num_regions, h, w, device=gradient_map.device)

        for i in range(self.num_regions):
            # y = mx + b, where x is the actual gradient/position value
            expected_gradients[:, i] = self.gradient_trends[i, 0] * gradient_map.squeeze(1) + self.gradient_trends[i, 1]
            expected_positions[:, i] = self.position_trends[i, 0] * position_map.squeeze(1) + self.position_trends[i, 1]

        # 2. Create a single "expected" map by weighting with region probabilities
        weighted_expected_gradient = (expected_gradients * region_probs).sum(dim=1, keepdim=True)
        weighted_expected_position = (expected_positions * region_probs).sum(dim=1, keepdim=True)
        
        # 3. Analyze the deviation between actual and expected values
        deviation_input = torch.cat([
            gradient_map, position_map, weighted_expected_gradient, weighted_expected_position
        ], dim=1)
        
        deviation_score = self.deviation_analyzer(deviation_input) # High score = likely defect
        
        return {
            'trend_adherence': trend_adherence,
            'deviation_score': deviation_score,
            'expected_gradients': expected_gradients,
            'expected_positions': expected_positions
        }

class FeatureExtractionPipeline:
    """Orchestrates the complete feature extraction process."""
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("FeatureExtractionPipeline")
        self.tensor_processor = TensorProcessor()
        self.logger.log_class_init("FeatureExtractionPipeline")

        self.multi_scale_extractor = MultiScaleFeatureExtractor()
        self.trend_analyzer = TrendAnalyzer()

        self.device = self.config.get_device()
        self.multi_scale_extractor = self.multi_scale_extractor.to(self.device)
        self.trend_analyzer = self.trend_analyzer.to(self.device)
        self.logger.info(f"FeatureExtractionPipeline initialized on device: {self.device}")
    
    def extract_features(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        ## FIX: Replaced the internal developer instructions with a proper docstring.
        Runs the full feature extraction pipeline on an input tensor.

        This involves:
        1. Pre-calculating gradient and position maps.
        2. Running the multi-scale feature extractor.
        3. Analyzing feature trends to identify deviations.
        4. Combining pattern-based and trend-based anomaly scores.
        
        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            Dict[str, torch.Tensor]: A comprehensive dictionary of results including features,
                                     region probabilities, anomaly maps, and quality scores.
        """
        self.logger.log_process_start("Feature Extraction")
        tensor = tensor.to(self.device)
        
        # 1. Pre-calculate global information
        gradient_info = self.tensor_processor.calculate_gradient_intensity(tensor)
        position_info = self.tensor_processor.calculate_pixel_positions(tensor.shape)
        
        # 2. Run multi-scale feature extraction
        multi_scale_results = self.multi_scale_extractor(tensor, gradient_info, position_info)
        
        # 3. Use the highest-resolution output for final analysis
        final_region_logits = multi_scale_results['region_logits'][-1]
        final_anomaly_logits = multi_scale_results['anomaly_logits'][-1]
        final_quality_scores = multi_scale_results['quality_scores'][-1]
        region_probs = F.softmax(final_region_logits, dim=1)
        
        # 4. Analyze trends based on the highest-resolution maps
        trend_results = self.trend_analyzer(
            gradient_info['gradient_map'],
            position_info['radial_positions'],
            region_probs
        )
        
        # 5. Combine anomaly scores from pattern matching and trend deviation
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
        
        self.logger.log_process_end("Feature Extraction")
        return results

# Example usage and testing block
if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    logger = get_logger("FeatureExtractorTest")
    logger.log_process_start("Feature Extractor Test")

    # Create a dummy test tensor
    test_tensor = torch.randn(2, 3, 256, 256).to(pipeline.device) # Test with batch size > 1
    
    # Extract features
    results = pipeline.extract_features(test_tensor)
    
    # Log and print results
    logger.info(f"Extracted {len(results['multi_scale_features'])} sets of scale features.")
    logger.info(f"Final anomaly map shape: {results['anomaly_map'].shape}")
    logger.info(f"Max anomaly score: {results['anomaly_map'].max().item():.4f}")
    logger.info(f"Mean quality score: {results['quality_map'].mean().item():.4f}")
    
    logger.log_process_end("Feature Extractor Test")
    print(f"[{datetime.now()}] Feature extractor test completed successfully.")