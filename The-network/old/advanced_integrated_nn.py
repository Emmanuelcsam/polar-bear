import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

from ..utils.logger import get_logger
from ..config.config import get_config


class SimultaneousFeatureAnomalyBlock(nn.Module):
    """
    Simultaneously performs feature classification and anomaly detection.
    "I want each feature to not only look for comparisons but also look for 
    anomalies while comparing"
    """
    
    def __init__(self, in_channels: int, num_patterns: int = 64):
        super().__init__()
        
        # Learnable reference patterns for normal features
        self.normal_patterns = nn.Parameter(torch.randn(num_patterns, in_channels, 3, 3))
        
        # Learnable anomaly patterns (defects, scratches, contamination)
        self.anomaly_patterns = nn.Parameter(torch.randn(num_patterns // 2, in_channels, 3, 3))
        
        # Learnable thresholds for each pattern
        self.normal_thresholds = nn.Parameter(torch.ones(num_patterns) * 0.7)
        self.anomaly_thresholds = nn.Parameter(torch.ones(num_patterns // 2) * 0.3)
        
        # Simultaneous classifier for regions AND anomalies
        self.simultaneous_classifier = nn.Sequential(
            nn.Conv2d(in_channels + num_patterns + num_patterns // 2, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.ReLU()
        )
        
        # Region classification head
        self.region_head = nn.Conv2d(128, 3, 1)  # core, cladding, ferrule
        
        # Anomaly detection head
        self.anomaly_head = nn.Conv2d(128, 1, 1)  # anomaly probability
        
        # Feature quality assessment
        self.quality_head = nn.Conv2d(128, 1, 1)  # how well feature matches any pattern
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simultaneously classify features and detect anomalies.
        "so I get a fully detailed anomaly detection while also classifying most 
        probable features(or segments) of the image at the same time"
        """
        b, c, h, w = features.shape
        
        # Compare to normal patterns
        normal_similarities = []
        normal_anomalies = []
        
        for i in range(self.normal_patterns.shape[0]):
            # Convolve feature with pattern
            pattern = self.normal_patterns[i].unsqueeze(0)
            similarity = F.conv2d(features, pattern, padding=1)
            similarity = torch.sigmoid(similarity)  # Normalize to [0, 1]
            
            # Check if similarity meets threshold (if not, it's anomalous)
            threshold = self.normal_thresholds[i]
            anomaly_from_normal = torch.relu(threshold - similarity)
            
            normal_similarities.append(similarity)
            normal_anomalies.append(anomaly_from_normal)
        
        # Compare to anomaly patterns
        anomaly_detections = []
        
        for i in range(self.anomaly_patterns.shape[0]):
            # Convolve feature with anomaly pattern
            pattern = self.anomaly_patterns[i].unsqueeze(0)
            anomaly_match = F.conv2d(features, pattern, padding=1)
            anomaly_match = torch.sigmoid(anomaly_match)
            
            # High match with anomaly pattern indicates defect
            anomaly_detections.append(anomaly_match)
        
        # Stack all comparisons
        normal_sim_stack = torch.cat(normal_similarities, dim=1)
        normal_anom_stack = torch.cat(normal_anomalies, dim=1)
        anomaly_stack = torch.cat(anomaly_detections, dim=1)
        
        # Combine all information for simultaneous processing
        combined = torch.cat([features, normal_sim_stack, anomaly_stack], dim=1)
        
        # Process through simultaneous classifier
        processed = self.simultaneous_classifier(combined)
        
        # Get region classification
        region_logits = self.region_head(processed)
        
        # Get anomaly detection (combining both sources)
        anomaly_from_patterns = self.anomaly_head(processed)
        total_anomaly = torch.sigmoid(anomaly_from_patterns) + normal_anom_stack.mean(dim=1, keepdim=True)
        
        # Get feature quality (how well it matches expected patterns)
        quality_score = torch.sigmoid(self.quality_head(processed))
        
        return {
            'region_logits': region_logits,
            'anomaly_scores': total_anomaly,
            'normal_similarities': normal_sim_stack,
            'anomaly_matches': anomaly_stack,
            'quality_scores': quality_score,
            'features': processed
        }


class MultiScaleGradientPositionConv(nn.Module):
    """
    Enhanced convolution that processes at multiple scales with gradient/position weighting.
    "the weights of the neural network will be dependent on the average intensity 
    gradient of the images(but It'll adjust the weight of that average as needed)"
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels // 3, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, out_channels // 3, 5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, out_channels // 3, 7, padding=3)
        
        # Gradient and position modulation per scale
        self.gradient_weights = nn.Parameter(torch.ones(3))
        self.position_weights = nn.Parameter(torch.ones(3))
        
        # Dynamic adjustment factors
        self.gradient_adjustment = nn.Parameter(torch.tensor(1.0))
        self.position_adjustment = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract multi-scale features with dynamic weighting"""
        # Calculate gradient and position statistics
        gradient_map = self._calculate_gradient_map(x)
        position_map = self._calculate_position_map(x)
        
        # Apply convolutions at different scales
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        feat_7x7 = self.conv_7x7(x)
        
        # Apply gradient and position modulation to each scale
        avg_gradient = gradient_map.mean(dim=[2, 3], keepdim=True)
        avg_position = position_map.mean(dim=[2, 3], keepdim=True)
        
        # Modulate each scale differently
        feat_3x3 = feat_3x3 * (1 + self.gradient_weights[0] * avg_gradient * self.gradient_adjustment)
        feat_3x3 = feat_3x3 * (1 + self.position_weights[0] * avg_position * self.position_adjustment)
        
        feat_5x5 = feat_5x5 * (1 + self.gradient_weights[1] * avg_gradient * self.gradient_adjustment)
        feat_5x5 = feat_5x5 * (1 + self.position_weights[1] * avg_position * self.position_adjustment)
        
        feat_7x7 = feat_7x7 * (1 + self.gradient_weights[2] * avg_gradient * self.gradient_adjustment)
        feat_7x7 = feat_7x7 * (1 + self.position_weights[2] * avg_position * self.position_adjustment)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)
        
        stats = {
            'gradient_map': gradient_map,
            'position_map': position_map,
            'avg_gradient': avg_gradient.mean().item(),
            'avg_position': avg_position.mean().item()
        }
        
        return multi_scale_features, stats
    
    def _calculate_gradient_map(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate gradient intensity map"""
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # Apply to each channel and average
        grad_x = F.conv2d(x.mean(dim=1, keepdim=True), sobel_x, padding=1)
        grad_y = F.conv2d(x.mean(dim=1, keepdim=True), sobel_y, padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return gradient_magnitude
    
    def _calculate_position_map(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate position-based weighting map"""
        b, c, h, w = x.shape
        
        # Create position encoding
        y_pos = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        x_pos = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        
        # Radial position (distance from center)
        radial_pos = torch.sqrt(x_pos**2 + y_pos**2)
        
        return radial_pos


class AdvancedIntegratedFiberOpticsNN(nn.Module):
    """
    Advanced integrated neural network that performs simultaneous feature classification,
    anomaly detection, and reconstruction all within the network layers.
    "it separates, does anomaly detection and comparison, and then also does reconstruction"
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger()
        
        # Multi-scale feature extraction with gradient/position weighting
        self.feature_extractors = nn.ModuleList([
            MultiScaleGradientPositionConv(3, 96),
            MultiScaleGradientPositionConv(96, 192),
            MultiScaleGradientPositionConv(192, 384),
            MultiScaleGradientPositionConv(384, 768)
        ])
        
        # Simultaneous feature analysis and anomaly detection blocks
        self.analysis_blocks = nn.ModuleList([
            SimultaneousFeatureAnomalyBlock(96, num_patterns=48),
            SimultaneousFeatureAnomalyBlock(192, num_patterns=96),
            SimultaneousFeatureAnomalyBlock(384, num_patterns=192),
            SimultaneousFeatureAnomalyBlock(768, num_patterns=384)
        ])
        
        # Cross-scale correlation module
        # "each segment will be analyzed by multiple comparisons, statistics, and 
        # correlational data from the multiple sources"
        self.cross_scale_correlator = nn.ModuleList([
            nn.Conv2d(96 + 192, 96, 1),
            nn.Conv2d(192 + 384, 192, 1),
            nn.Conv2d(384 + 768, 384, 1)
        ])
        
        # Global context aggregator
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(768, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1)
        )
        
        # Reference comparison module
        # "try to see which of the reference images in the reference folder of the 
        # database that the regions most specifically represent"
        self.reference_embeddings = nn.Parameter(torch.randn(2000, 256))
        self.reference_classifier = nn.Linear(256, 2000)
        
        # Trend analysis module
        # "the program will forcibly look for all lines of best fit based on 
        # gradient trends for all datapoints and pixels"
        self.trend_analyzer = nn.ModuleList([
            nn.Conv2d(96, 32, 1),
            nn.Conv2d(192, 32, 1),
            nn.Conv2d(384, 32, 1),
            nn.Conv2d(768, 32, 1)
        ])
        
        # Trend parameters for each region and scale
        self.gradient_trends = nn.Parameter(torch.randn(3, 4, 2))  # 3 regions, 4 scales, 2 params (slope, intercept)
        self.pixel_trends = nn.Parameter(torch.randn(3, 4, 2))
        
        # Final integration layers
        self.final_integration = nn.Conv2d(3 + 1 + 1, 64, 1)  # regions + anomalies + quality
        
        # Reconstruction decoder (operates on analyzed features)
        self.reconstruction_decoder = nn.ModuleList([
            nn.ConvTranspose2d(768, 384, 4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ConvTranspose2d(96, 3, 4, stride=2, padding=1)
        ])
        
        self.logger.info("Initialized AdvancedIntegratedFiberOpticsNN")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with simultaneous processing.
        "really fully analyze everything that I've been saying"
        """
        batch_size = x.shape[0]
        original_input = x.clone()
        
        # Store all intermediate results
        all_features = []
        all_analysis_results = []
        all_gradients = []
        all_positions = []
        all_trend_scores = []
        
        # Progressive feature extraction and analysis
        current = x
        for i, (extractor, analyzer, trend_layer) in enumerate(
            zip(self.feature_extractors, self.analysis_blocks, self.trend_analyzer)
        ):
            # Extract multi-scale features with gradient/position modulation
            features, stats = extractor(current)
            all_features.append(features)
            all_gradients.append(stats['avg_gradient'])
            all_positions.append(stats['avg_position'])
            
            # Simultaneously analyze features for classification AND anomalies
            # "I want each feature to not only look for comparisons but also look 
            # for anomalies while comparing"
            analysis = analyzer(features)
            all_analysis_results.append(analysis)
            
            # Analyze gradient trends for this scale
            trend_features = trend_layer(features)
            trend_score = self._calculate_trend_adherence(
                trend_features, stats['gradient_map'], stats['position_map'], i
            )
            all_trend_scores.append(trend_score)
            
            # Cross-scale correlation if not first layer
            if i > 0:
                # Correlate with previous scale
                prev_features = F.interpolate(all_features[i-1], size=features.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                correlated = torch.cat([features, prev_features], dim=1)
                features = self.cross_scale_correlator[i-1](correlated)
            
            current = features
        
        # Aggregate multi-scale segmentation and anomaly results
        final_segmentation, final_anomalies, final_quality = self._aggregate_multiscale_analysis(
            all_analysis_results, all_trend_scores
        )
        
        # Global context and reference comparison
        global_features = self.global_context(all_features[-1])
        global_flat = global_features.mean(dim=[2, 3])
        
        # Compare to learned references
        reference_logits = self.reference_classifier(global_flat)
        reference_probs = F.softmax(reference_logits, dim=1)
        best_reference_idx = reference_logits.argmax(dim=1)
        
        # Calculate similarity using learned embeddings
        best_embedding = self.reference_embeddings[best_reference_idx]
        similarity = F.cosine_similarity(global_flat, best_embedding, dim=1)
        
        # Check threshold
        # "the program must achieve over .7"
        meets_threshold = similarity > self.config.SIMILARITY_THRESHOLD
        
        # Reconstruction from analyzed features (not just raw features)
        # "and then also does reconstruction"
        reconstruction = self._reconstruct_from_analyzed_features(
            all_features, all_analysis_results
        )
        
        # Final anomaly refinement using reconstruction
        refined_anomalies = self._refine_anomalies_with_reconstruction(
            final_anomalies, reconstruction, original_input
        )
        
        # Integrate all outputs
        integrated = torch.cat([final_segmentation, refined_anomalies, final_quality], dim=1)
        final_output = self.final_integration(integrated)
        
        outputs = {
            # Segmentation results
            'segmentation': final_segmentation,
            'region_masks': self._extract_region_masks(final_segmentation),
            
            # Anomaly detection results (simultaneous with classification)
            'anomaly_map': refined_anomalies.squeeze(1),
            'anomaly_details': self._extract_anomaly_details(all_analysis_results),
            
            # Reference comparison
            'best_reference_idx': best_reference_idx,
            'reference_distribution': reference_probs,
            'similarity': similarity,
            'meets_threshold': meets_threshold,
            
            # Quality and confidence scores
            'quality_map': final_quality.squeeze(1),
            'feature_quality_scores': [a['quality_scores'] for a in all_analysis_results],
            
            # Reconstruction
            'reconstruction': reconstruction,
            
            # Trend analysis
            'trend_adherence': torch.stack(all_trend_scores).mean(dim=0),
            'gradient_trends': self.gradient_trends,
            'pixel_trends': self.pixel_trends,
            
            # All intermediate results for detailed analysis
            'all_features': all_features,
            'all_analysis': all_analysis_results,
            'final_integrated': final_output
        }
        
        return outputs
    
    def _calculate_trend_adherence(self, features: torch.Tensor, 
                                  gradient_map: torch.Tensor,
                                  position_map: torch.Tensor,
                                  scale_idx: int) -> torch.Tensor:
        """
        Calculate how well features adhere to expected trends.
        "when a feature of an image follows this line its classified as region 
        (core cladding ferrule) when it doesn't its classified as a defect"
        """
        b, c, h, w = features.shape
        
        # Get trend parameters for this scale
        gradient_params = self.gradient_trends[:, scale_idx, :]  # [3 regions, 2 params]
        position_params = self.pixel_trends[:, scale_idx, :]
        
        # Calculate expected values based on trends
        trend_scores = []
        
        for region_idx in range(3):
            # Expected gradient: y = mx + b
            expected_gradient = (gradient_params[region_idx, 0] * gradient_map + 
                               gradient_params[region_idx, 1])
            
            # Expected position influence
            expected_position = (position_params[region_idx, 0] * position_map + 
                               position_params[region_idx, 1])
            
            # How well do features match expected trends?
            feature_magnitude = features.abs().mean(dim=1, keepdim=True)
            trend_match = torch.sigmoid(expected_gradient + expected_position)
            
            # High trend match = likely region, low = likely defect
            trend_scores.append(trend_match)
        
        return torch.cat(trend_scores, dim=1)
    
    def _aggregate_multiscale_analysis(self, 
                                     all_analysis: List[Dict],
                                     all_trends: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate analysis results across all scales.
        "so I get a fully detailed anomaly detection while also classifying most 
        probable features(or segments) of the image at the same time"
        """
        # Resize all to same spatial dimensions
        target_size = all_analysis[0]['region_logits'].shape[-2:]
        
        # Aggregate segmentation
        segmentations = []
        for analysis in all_analysis:
            seg = F.interpolate(analysis['region_logits'], size=target_size, 
                              mode='bilinear', align_corners=False)
            segmentations.append(seg)
        
        # Weighted average (later scales have more weight)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=seg.device)
        weighted_seg = sum(w * s for w, s in zip(weights, segmentations))
        final_segmentation = F.softmax(weighted_seg, dim=1)
        
        # Aggregate anomalies
        anomalies = []
        for analysis, trend in zip(all_analysis, all_trends):
            anom = F.interpolate(analysis['anomaly_scores'], size=target_size,
                               mode='bilinear', align_corners=False)
            # Reduce anomaly scores where trends match
            trend_resized = F.interpolate(trend.max(dim=1, keepdim=True)[0], 
                                        size=target_size, mode='bilinear', align_corners=False)
            anom = anom * (1 - 0.7 * trend_resized)
            anomalies.append(anom)
        
        final_anomalies = sum(w * a for w, a in zip(weights, anomalies))
        
        # Aggregate quality scores
        qualities = []
        for analysis in all_analysis:
            qual = F.interpolate(analysis['quality_scores'], size=target_size,
                               mode='bilinear', align_corners=False)
            qualities.append(qual)
        
        final_quality = sum(w * q for w, q in zip(weights, qualities))
        
        return final_segmentation, final_anomalies, final_quality
    
    def _reconstruct_from_analyzed_features(self, 
                                          all_features: List[torch.Tensor],
                                          all_analysis: List[Dict]) -> torch.Tensor:
        """
        Reconstruct image from analyzed features, not raw features.
        This ensures reconstruction incorporates the anomaly detection results.
        """
        # Start from deepest analyzed features
        current = all_features[-1]
        
        # Modulate by analysis results before reconstruction
        analysis = all_analysis[-1]
        quality = analysis['quality_scores']
        
        # Suppress features where quality is low (likely anomalies)
        current = current * quality
        
        # Progressive reconstruction
        for i, decoder in enumerate(self.reconstruction_decoder):
            current = decoder(current)
            if i < len(self.reconstruction_decoder) - 1:
                current = F.relu(current)
                
                # Skip connections with earlier analysis results
                if i < len(all_analysis) - 1:
                    skip_idx = len(all_analysis) - 2 - i
                    skip_quality = all_analysis[skip_idx]['quality_scores']
                    skip_quality = F.interpolate(skip_quality, size=current.shape[-2:],
                                               mode='bilinear', align_corners=False)
                    current = current * skip_quality
        
        return torch.sigmoid(current)
    
    def _refine_anomalies_with_reconstruction(self, 
                                            anomalies: torch.Tensor,
                                            reconstruction: torch.Tensor,
                                            original: torch.Tensor) -> torch.Tensor:
        """
        Refine anomaly detection using reconstruction difference.
        "Im subtracting the resulting classification with the original input to find anomalies"
        """
        # Direct difference
        direct_diff = torch.abs(reconstruction - original).mean(dim=1, keepdim=True)
        
        # Structural similarity difference
        ssim_diff = 1 - self._calculate_ssim(reconstruction, original)
        
        # Combine with existing anomaly detection
        refined = anomalies + 0.3 * direct_diff + 0.2 * ssim_diff.unsqueeze(1)
        
        # Normalize
        refined = torch.sigmoid(refined)
        
        return refined
    
    def _extract_region_masks(self, segmentation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract individual region masks from segmentation"""
        return {
            'core': segmentation[:, 0],
            'cladding': segmentation[:, 1],
            'ferrule': segmentation[:, 2]
        }
    
    def _extract_anomaly_details(self, all_analysis: List[Dict]) -> Dict:
        """Extract detailed anomaly information from all scales"""
        details = {
            'scale_anomalies': [],
            'anomaly_patterns_matched': [],
            'normal_pattern_deviations': []
        }
        
        for i, analysis in enumerate(all_analysis):
            # Anomaly scores at this scale
            details['scale_anomalies'].append(analysis['anomaly_scores'].mean().item())
            
            # Which anomaly patterns were matched
            anomaly_matches = analysis['anomaly_matches'].mean(dim=[2, 3])  # Average spatial
            top_matches = anomaly_matches.topk(3, dim=1)
            details['anomaly_patterns_matched'].append(top_matches.indices.tolist())
            
            # Deviation from normal patterns
            normal_sims = analysis['normal_similarities'].mean(dim=[2, 3])
            avg_deviation = (1 - normal_sims).mean(dim=1)
            details['normal_pattern_deviations'].append(avg_deviation.tolist())
        
        return details
    
    def _calculate_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM between two images"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Use average across channels
        x_gray = x.mean(dim=1, keepdim=True)
        y_gray = y.mean(dim=1, keepdim=True)
        
        mu_x = F.avg_pool2d(x_gray, 11, stride=1, padding=5)
        mu_y = F.avg_pool2d(y_gray, 11, stride=1, padding=5)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x_gray ** 2, 11, stride=1, padding=5) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y_gray ** 2, 11, stride=1, padding=5) - mu_y_sq
        sigma_xy = F.avg_pool2d(x_gray * y_gray, 11, stride=1, padding=5) - mu_xy
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim.squeeze(1)
    
    def get_detailed_equation_parameters(self) -> Dict:
        """
        Get all equation parameters with per-scale details.
        "I=Ax1+Bx2+Cx3... =S(R)"
        """
        params = {}
        
        # Per-scale gradient and position weights
        for i, extractor in enumerate(self.feature_extractors):
            params[f'scale_{i}_gradient_weights'] = extractor.gradient_weights.detach().cpu().numpy()
            params[f'scale_{i}_position_weights'] = extractor.position_weights.detach().cpu().numpy()
            params[f'scale_{i}_gradient_adjustment'] = extractor.gradient_adjustment.item()
            params[f'scale_{i}_position_adjustment'] = extractor.position_adjustment.item()
        
        # Trend parameters
        params['gradient_trends'] = self.gradient_trends.detach().cpu().numpy()
        params['pixel_trends'] = self.pixel_trends.detach().cpu().numpy()
        
        # Pattern thresholds
        for i, analyzer in enumerate(self.analysis_blocks):
            params[f'scale_{i}_normal_thresholds'] = analyzer.normal_thresholds.detach().cpu().numpy()
            params[f'scale_{i}_anomaly_thresholds'] = analyzer.anomaly_thresholds.detach().cpu().numpy()
        
        return params