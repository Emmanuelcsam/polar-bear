import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np

from ..utils.logger import get_logger
from ..config.config import get_config
from ..processors.segmentation import SegmentationResult


class CustomWeightedLayer(nn.Module):
    """
    Custom layer with weights dependent on image statistics.
    "the weights of the neural network will be dependent on the average intensity gradient"
    "another weight will be dependent on the average pixel position"
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard learnable weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Additional weight modulation parameters
        self.gradient_weight = nn.Parameter(torch.ones(out_features))
        self.position_weight = nn.Parameter(torch.ones(out_features))
        
        # "another weight you will comment out completely will be a manual circle alignment"
        # self.circle_alignment_weight = nn.Parameter(torch.ones(out_features))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x: torch.Tensor, gradient_factor: float, 
                position_factor: Tuple[float, float]) -> torch.Tensor:
        """
        Forward pass with custom weight modulation.
        "I=Ax1+Bx2+Cx3... =S(R)"
        """
        # Base linear transformation
        out = F.linear(x, self.weight, self.bias)
        
        # Apply gradient-based weight modulation
        gradient_modulation = self.gradient_weight * gradient_factor
        out = out * gradient_modulation.unsqueeze(0)
        
        # Apply position-based weight modulation
        position_magnitude = np.sqrt(position_factor[0]**2 + position_factor[1]**2)
        position_modulation = self.position_weight * position_magnitude
        out = out * position_modulation.unsqueeze(0)
        
        # Circle alignment weight (commented out as requested)
        # circle_modulation = self.circle_alignment_weight * circle_factor
        # out = out * circle_modulation.unsqueeze(0)
        
        return out


class FiberOpticsNeuralNetwork(nn.Module):
    """
    Main neural network for fiber optics classification and defect detection.
    "the program is a fiber optics image classifier and defect analyser so its a 
    neural network of classification and anomaly detection"
    """
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = get_logger()
        
        # Input processing layers
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Feature extraction backbone (ResNet-like blocks)
        self.feature_blocks = nn.ModuleList([
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 512)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom weighted layers for processing
        # "then after some hidden layers the network will converge into three specific regions"
        self.custom_layers = nn.ModuleList([
            CustomWeightedLayer(512, self.config.HIDDEN_LAYERS[0]),
            CustomWeightedLayer(self.config.HIDDEN_LAYERS[0], self.config.HIDDEN_LAYERS[1]),
            CustomWeightedLayer(self.config.HIDDEN_LAYERS[1], self.config.HIDDEN_LAYERS[2]),
            CustomWeightedLayer(self.config.HIDDEN_LAYERS[2], self.config.HIDDEN_LAYERS[3])
        ])
        
        # Region-specific heads (core, cladding, ferrule)
        self.core_head = nn.Sequential(
            nn.Linear(self.config.HIDDEN_LAYERS[3], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.cladding_head = nn.Sequential(
            nn.Linear(self.config.HIDDEN_LAYERS[3], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.ferrule_head = nn.Sequential(
            nn.Linear(self.config.HIDDEN_LAYERS[3], 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Defect detection head
        self.defect_head = nn.Sequential(
            nn.Linear(self.config.HIDDEN_LAYERS[3], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Binary defect probability
        )
        
        # Feature fusion for final classification
        self.fusion_layer = nn.Linear(64 * 3, 256)  # 3 regions * 64 features each
        
        self.logger.info("Initialized FiberOpticsNeuralNetwork")
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block for feature extraction"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, image_stats: Dict[str, float],
                segmentation: Optional[SegmentationResult] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        "that tensor will be compared to other tensors in the correlational data and 
        at the same time it will go through the neural network for classification"
        """
        batch_size = x.shape[0]
        
        # Log the forward pass
        self.logger.log_tensor_operation("forward_pass", x.shape, "started")
        
        # Extract convolutional features
        conv_features = self.input_conv(x)
        
        for block in self.feature_blocks:
            conv_features = block(conv_features)
        
        # Global pooling
        pooled_features = self.global_pool(conv_features)
        features = pooled_features.view(batch_size, -1)
        
        # Get gradient and position factors from image statistics
        gradient_factor = image_stats.get('gradient_intensity', 1.0)
        position_factor = (
            image_stats.get('center_x', 128),
            image_stats.get('center_y', 128)
        )
        
        # Pass through custom weighted layers
        # "the weights of the connections between the first layer and next layer will be 
        # dependent on these correlational values and parameters"
        hidden = features
        for i, layer in enumerate(self.custom_layers):
            hidden = layer(hidden, gradient_factor, position_factor)
            hidden = F.relu(hidden)
            
            # Log weight updates for monitoring
            if self.training:
                self.logger.log_weight_update(
                    f"custom_layer_{i}",
                    old_weight=0,  # Placeholder
                    new_weight=layer.gradient_weight.mean().item(),
                    gradient=gradient_factor
                )
        
        # Region-specific processing
        # "the network will converge into three specific regions the core, cladding and ferrule"
        core_features = self.core_head(hidden)
        cladding_features = self.cladding_head(hidden)
        ferrule_features = self.ferrule_head(hidden)
        
        # Defect detection
        defect_score = torch.sigmoid(self.defect_head(hidden))
        
        # Combine region features
        combined_features = torch.cat([core_features, cladding_features, ferrule_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        outputs = {
            'core_features': core_features,
            'cladding_features': cladding_features,
            'ferrule_features': ferrule_features,
            'fused_features': fused_features,
            'defect_probability': defect_score,
            'hidden_features': hidden  # For similarity calculations
        }
        
        return outputs
    
    def compute_region_similarity(self, features: torch.Tensor, 
                                reference_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between extracted features and reference.
        "S(R) where... S is the similarity coefficient the percentage of how similar 
        the input image is from the reference and R is the reference"
        """
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        reference_norm = F.normalize(reference_features, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(features_norm, reference_norm.t())
        
        return similarity
    
    def extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features at multiple scales for better defect detection"""
        features = []
        
        # Initial convolution
        x = self.input_conv(x)
        features.append(x)
        
        # Extract features at each scale
        for block in self.feature_blocks:
            x = block(x)
            features.append(x)
        
        return features
    
    def adjust_weights_based_on_loss(self, loss: torch.Tensor, adjustment_rate: float):
        """
        Adjust network weights based on loss.
        "the program will calculate its losses and try to minimize its losses by 
        small percentile adjustments to parameters"
        """
        # This is typically handled by the optimizer, but we can add custom logic
        with torch.no_grad():
            for layer in self.custom_layers:
                # Adjust gradient weights
                layer.gradient_weight *= (1 - adjustment_rate * loss.item())
                
                # Adjust position weights  
                layer.position_weight *= (1 - adjustment_rate * loss.item())
                
                # Log adjustments
                self.logger.debug(f"Adjusted weights by factor {1 - adjustment_rate * loss.item():.4f}")


class RegionClassifier(nn.Module):
    """
    Specialized classifier for region-specific features.
    "when a feature of an image follows this line its classified as region 
    (core cladding ferrule) when it doesn't its classified as a defect"
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DefectDetector(nn.Module):
    """
    Specialized module for anomaly/defect detection.
    "when it doesn't its classified as a defect, the for every image the program 
    will spit out an anomaly or defect map"
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Encoder for feature compression
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        
        # Reconstruction error indicates anomaly
        reconstruction_error = torch.mean((x - reconstructed) ** 2, dim=1)
        
        return reconstructed, reconstruction_error