import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

from ..utils.logger import get_logger
from ..config.config import get_config


class GradientPixelWeightedConv(nn.Module):
    """
    Convolutional layer with weights modulated by gradient intensity and pixel position.
    "the weights of the neural network will be dependent on the average intensity 
    gradient of the images(but It'll adjust the weight of that average as needed), 
    another weight will be dependent on the average pixel position"
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Learnable modulation parameters for gradient and position
        self.gradient_modulation = nn.Parameter(torch.ones(out_channels))
        self.position_modulation = nn.Parameter(torch.ones(out_channels))
        
        # Adjustable weights for how much gradient/position affects the layer
        self.gradient_influence = nn.Parameter(torch.tensor(0.3))
        self.position_influence = nn.Parameter(torch.tensor(0.2))
        
        # "another weight you will comment out completely will be a manual circle alignment"
        # self.circle_alignment_modulation = nn.Parameter(torch.ones(out_channels))
        # self.circle_influence = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that extracts features/edges and modulates based on statistics.
        "the neural network itself in order to classify the image will separate 
        the image into its features (or edges)"
        """
        batch_size = x.shape[0]
        
        # Calculate gradient intensity for each image in batch
        gradient_maps = self._calculate_gradient_maps(x)
        avg_gradients = gradient_maps.mean(dim=[2, 3])  # Average per channel
        
        # Calculate average pixel positions (center of mass)
        position_weights = self._calculate_position_weights(x)
        
        # Standard convolution to extract features/edges
        features = self.conv(x)
        
        # Modulate features based on gradient and position
        # "I=Ax1+Bx2+Cx3... =S(R)"
        for b in range(batch_size):
            # A * x1 (gradient modulation)
            grad_factor = avg_gradients[b].mean() * self.gradient_influence
            features[b] = features[b] * (1 + grad_factor * self.gradient_modulation.view(-1, 1, 1))
            
            # B * x2 (position modulation)
            pos_factor = position_weights[b].mean() * self.position_influence
            features[b] = features[b] * (1 + pos_factor * self.position_modulation.view(-1, 1, 1))
            
            # C * x3 (circle alignment - commented out)
            # circle_factor = self._calculate_circle_alignment(x[b])
            # features[b] = features[b] * (1 + circle_factor * self.circle_alignment_modulation.view(-1, 1, 1))
        
        # Return features and intermediate calculations for loss computation
        intermediates = {
            'gradient_maps': gradient_maps,
            'avg_gradients': avg_gradients,
            'position_weights': position_weights
        }
        
        return features, intermediates
    
    def _calculate_gradient_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate gradient intensity maps using Sobel filters"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        grad_x = F.conv2d(x, sobel_x.repeat(x.shape[1], 1, 1, 1), 
                         groups=x.shape[1], padding=1)
        grad_y = F.conv2d(x, sobel_y.repeat(x.shape[1], 1, 1, 1), 
                         groups=x.shape[1], padding=1)
        
        # Gradient magnitude
        gradient_maps = torch.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_maps
    
    def _calculate_position_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate position-based weights (center of mass)"""
        b, c, h, w = x.shape
        
        # Create coordinate grids
        y_coords = torch.arange(h, device=x.device).float().view(1, 1, h, 1)
        x_coords = torch.arange(w, device=x.device).float().view(1, 1, 1, w)
        
        # Calculate center of mass for each channel
        position_weights = []
        for i in range(b):
            img = x[i]
            total_intensity = img.sum(dim=[1, 2], keepdim=True) + 1e-8
            
            # Weighted average positions
            avg_y = (img * y_coords).sum(dim=[1, 2]) / total_intensity.squeeze()
            avg_x = (img * x_coords).sum(dim=[1, 2]) / total_intensity.squeeze()
            
            # Normalize to [0, 1]
            pos_weight = torch.sqrt((avg_y/h)**2 + (avg_x/w)**2)
            position_weights.append(pos_weight)
        
        return torch.stack(position_weights)


class FeatureComparisonBlock(nn.Module):
    """
    Compares extracted features to reference patterns learned during training.
    "and then itll compare the edges or features to what they most likely 
    can be classified as"
    """
    
    def __init__(self, in_channels: int, num_patterns: int = 64):
        super().__init__()
        
        # Learnable reference patterns for each region type
        self.reference_patterns = nn.Parameter(torch.randn(num_patterns, in_channels, 1, 1))
        
        # Pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Conv2d(in_channels + num_patterns, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)  # 3 outputs for core, cladding, ferrule
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compare features to learned reference patterns.
        "when a feature of an image follows this line its classified as region 
        (core cladding ferrule) when it doesn't its classified as a defect"
        """
        b, c, h, w = features.shape
        
        # Calculate similarity to each reference pattern
        similarities = []
        for i in range(self.reference_patterns.shape[0]):
            pattern = self.reference_patterns[i].view(1, c, 1, 1)
            # Cosine similarity at each spatial location
            similarity = F.cosine_similarity(features, pattern.expand(b, c, h, w), dim=1)
            similarities.append(similarity.unsqueeze(1))
        
        similarity_maps = torch.cat(similarities, dim=1)
        
        # Concatenate features with similarity maps
        combined = torch.cat([features, similarity_maps], dim=1)
        
        # Classify each pixel into regions
        region_logits = self.pattern_classifier(combined)
        
        return region_logits, similarity_maps


class IntegratedFiberOpticsNN(nn.Module):
    """
    Fully integrated neural network that performs segmentation, comparison, and 
    anomaly detection internally.
    "I want you to create scripts closer to my goal where the neural network does 
    the segmentation and reference comparison and anomaly detection internally"
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger()
        
        # Feature extraction with gradient/position weighting
        self.feature_layers = nn.ModuleList([
            GradientPixelWeightedConv(3, 64),
            GradientPixelWeightedConv(64, 128),
            GradientPixelWeightedConv(128, 256),
            GradientPixelWeightedConv(256, 512)
        ])
        
        # Feature comparison and classification
        self.comparison_blocks = nn.ModuleList([
            FeatureComparisonBlock(64, num_patterns=32),
            FeatureComparisonBlock(128, num_patterns=48),
            FeatureComparisonBlock(256, num_patterns=64),
            FeatureComparisonBlock(512, num_patterns=96)
        ])
        
        # Multi-scale feature fusion for segmentation
        self.segmentation_fusion = nn.Conv2d(3 * 4, 3, 1)  # 3 regions * 4 scales
        
        # Reference learning branch
        # "it will take those three features of an image and try to see which of the 
        # reference images in the reference folder of the database that the regions 
        # most specifically represent"
        self.reference_encoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(256, 128, 1),
            nn.ReLU()
        )
        
        # Learned reference embeddings for comparison
        self.reference_embeddings = nn.Parameter(torch.randn(1000, 128))  # 1000 reference patterns
        
        # Anomaly detection branch
        # "the addition with my program is that Im subtracting the resulting 
        # classification with the original input to find anomalies"
        self.reconstruction_decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        ])
        
        # Line of best fit parameters
        # "the program will forcibly look for all lines of best fit based on 
        # gradient trends for all datapoints and pixels"
        self.gradient_trend_params = nn.Parameter(torch.randn(3, 2))  # Linear fit params for each region
        self.pixel_trend_params = nn.Parameter(torch.randn(3, 2))
        
        self.logger.info("Initialized IntegratedFiberOpticsNN")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass performing all operations internally.
        "that tensor will be compared to other tensors in the correlational data 
        and at the same time it will go through the neural network for classification"
        """
        batch_size = x.shape[0]
        original_input = x.clone()
        
        # Track all intermediate results
        all_features = []
        all_gradients = []
        all_positions = []
        all_region_maps = []
        all_similarities = []
        
        # Feature extraction at multiple scales
        current = x
        for i, (feat_layer, comp_block) in enumerate(zip(self.feature_layers, self.comparison_blocks)):
            # Extract features with gradient/position modulation
            features, intermediates = feat_layer(current)
            all_features.append(features)
            all_gradients.append(intermediates['avg_gradients'])
            all_positions.append(intermediates['position_weights'])
            
            # Compare to learned patterns and classify regions
            region_logits, similarity_maps = comp_block(features)
            all_region_maps.append(F.interpolate(region_logits, size=x.shape[-2:], 
                                                mode='bilinear', align_corners=False))
            all_similarities.append(similarity_maps)
            
            current = features
        
        # Fuse multi-scale segmentation predictions
        # "after the network converges the features of the image to either core 
        # cladding and ferrule"
        fused_regions = torch.cat(all_region_maps, dim=1)
        final_segmentation = self.segmentation_fusion(fused_regions)
        segmentation_probs = F.softmax(final_segmentation, dim=1)
        
        # Extract region masks
        region_masks = {
            'core': segmentation_probs[:, 0],
            'cladding': segmentation_probs[:, 1],
            'ferrule': segmentation_probs[:, 2]
        }
        
        # Reference comparison
        # "now that the process has either the three regions or a most similar 
        # image from the reference bank"
        encoded_features = self.reference_encoder(all_features[-1])
        encoded_flat = encoded_features.view(batch_size, -1)
        
        # Compare to reference embeddings
        reference_similarities = F.cosine_similarity(
            encoded_flat.unsqueeze(1),
            self.reference_embeddings.unsqueeze(0),
            dim=2
        )
        
        best_reference_idx = reference_similarities.argmax(dim=1)
        best_similarity = reference_similarities.max(dim=1)[0]
        
        # Reconstruction for anomaly detection
        # "take the absolute value of the difference between the two"
        reconstruction = all_features[-1]
        for i, decoder in enumerate(self.reconstruction_decoder):
            reconstruction = decoder(reconstruction)
            if i < len(self.reconstruction_decoder) - 1:
                reconstruction = F.relu(reconstruction)
        
        # Ensure reconstruction matches input size
        if reconstruction.shape != original_input.shape:
            reconstruction = F.interpolate(reconstruction, size=original_input.shape[-2:], 
                                         mode='bilinear', align_corners=False)
        
        # Calculate anomaly map
        # "take the absolute value of the difference between the two and it will 
        # also look at the structural similarity index"
        difference_map = torch.abs(reconstruction - original_input)
        
        # Calculate SSIM-like local similarity
        ssim_map = self._calculate_ssim_map(reconstruction, original_input)
        
        # Combine for final anomaly detection
        # "compare the local anomaly heatmap with the structural similarity index 
        # and combine them to find the total anomalies"
        anomaly_map = difference_map.mean(dim=1) * (1 - ssim_map)
        
        # Apply line of best fit to identify true anomalies vs region transitions
        # "when a feature of an image follows this line its classified as region 
        # when it doesn't its classified as a defect"
        anomaly_map = self._apply_trend_filtering(anomaly_map, segmentation_probs, 
                                                 all_gradients, all_positions)
        
        # Check similarity threshold
        # "when classifying and comparing to the reference the program must achieve over .7"
        meets_threshold = best_similarity > self.config.SIMILARITY_THRESHOLD
        
        outputs = {
            'segmentation': segmentation_probs,
            'region_masks': region_masks,
            'reconstruction': reconstruction,
            'anomaly_map': anomaly_map,
            'difference_map': difference_map,
            'best_reference_idx': best_reference_idx,
            'best_similarity': best_similarity,
            'meets_threshold': meets_threshold,
            'all_features': all_features,
            'all_similarities': all_similarities
        }
        
        return outputs
    
    def _calculate_ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate structural similarity index map"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Convert to grayscale
        x_gray = x.mean(dim=1, keepdim=True)
        y_gray = y.mean(dim=1, keepdim=True)
        
        # Local means
        kernel = self._gaussian_kernel(11).to(x.device)
        mu_x = F.conv2d(x_gray, kernel, padding=5)
        mu_y = F.conv2d(y_gray, kernel, padding=5)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Local variances
        sigma_x_sq = F.conv2d(x_gray**2, kernel, padding=5) - mu_x_sq
        sigma_y_sq = F.conv2d(y_gray**2, kernel, padding=5) - mu_y_sq
        sigma_xy = F.conv2d(x_gray * y_gray, kernel, padding=5) - mu_xy
        
        # SSIM
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim.squeeze(1)
    
    def _apply_trend_filtering(self, anomaly_map: torch.Tensor, 
                             segmentation: torch.Tensor,
                             gradients: List[torch.Tensor],
                             positions: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply learned trend lines to distinguish defects from region transitions.
        "it will understand that the regions themselves or the change in regions 
        within an image such as the change in pixels between the core to cladding 
        or cladding to ferrule are not anomalies"
        """
        filtered_map = anomaly_map.clone()
        
        # Get average gradients and positions across scales
        avg_gradient = torch.stack([g.mean(dim=1) for g in gradients]).mean(dim=0)
        avg_position = torch.stack(positions).mean(dim=0).mean(dim=1)
        
        # For each region, check if anomalies follow expected trends
        for region_idx in range(3):
            region_mask = segmentation[:, region_idx]
            
            # Expected gradient trend for this region
            grad_trend = self.gradient_trend_params[region_idx, 0] * avg_gradient + \
                        self.gradient_trend_params[region_idx, 1]
            
            # Expected position trend
            pos_trend = self.pixel_trend_params[region_idx, 0] * avg_position + \
                       self.pixel_trend_params[region_idx, 1]
            
            # Reduce anomaly scores where they match expected trends
            trend_match = torch.sigmoid(grad_trend + pos_trend).unsqueeze(-1).unsqueeze(-1)
            filtered_map = filtered_map * (1 - 0.8 * region_mask * trend_match)
        
        # Suppress anomalies at region boundaries
        boundaries = self._find_region_boundaries(segmentation)
        filtered_map = filtered_map * (1 - 0.9 * boundaries)
        
        return filtered_map
    
    def _find_region_boundaries(self, segmentation: torch.Tensor) -> torch.Tensor:
        """Detect boundaries between regions"""
        # Sobel edge detection on segmentation
        edges = []
        for i in range(segmentation.shape[1]):
            sobel_x = F.conv2d(segmentation[:, i:i+1], 
                              self._sobel_x().to(segmentation.device), padding=1)
            sobel_y = F.conv2d(segmentation[:, i:i+1], 
                              self._sobel_y().to(segmentation.device), padding=1)
            edge = torch.sqrt(sobel_x**2 + sobel_y**2)
            edges.append(edge)
        
        return torch.cat(edges, dim=1).max(dim=1)[0]
    
    def _gaussian_kernel(self, kernel_size: int) -> torch.Tensor:
        """Create Gaussian kernel"""
        sigma = kernel_size / 3.0
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _sobel_x(self) -> torch.Tensor:
        return torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    
    def _sobel_y(self) -> torch.Tensor:
        return torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    
    def adjust_parameters(self, gradient_factor: float = None, position_factor: float = None):
        """
        Allow manual adjustment of weight parameters.
        "the program will allow me to see and tweak the parameters and weights 
        of these fitting equations"
        """
        if gradient_factor is not None:
            for layer in self.feature_layers:
                layer.gradient_influence.data *= gradient_factor
        
        if position_factor is not None:
            for layer in self.feature_layers:
                layer.position_influence.data *= position_factor
        
        self.logger.info(f"Adjusted parameters - Gradient factor: {gradient_factor}, "
                        f"Position factor: {position_factor}")
    
    def get_equation_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get current equation parameters.
        "I=Ax1+Bx2+Cx3... =S(R)"
        """
        params = {}
        
        # Average across all layers
        gradient_influences = torch.stack([layer.gradient_influence for layer in self.feature_layers])
        position_influences = torch.stack([layer.position_influence for layer in self.feature_layers])
        
        params['A_gradient'] = gradient_influences.mean().item()
        params['B_position'] = position_influences.mean().item()
        params['gradient_trends'] = self.gradient_trend_params.detach().cpu()
        params['pixel_trends'] = self.pixel_trend_params.detach().cpu()
        
        return params