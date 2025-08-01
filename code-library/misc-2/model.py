# model.py
# Model architecture for fiber optic analysis

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CBAM(nn.Module):
    """Convolutional Block Attention Module for enhanced feature representation."""
    
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        attn = self.channel_gate(x)
        x = x * attn
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.spatial_gate(torch.cat([avg_pool, max_pool], dim=1))
        x = x * attn
        return x

class FiberOpticNet(nn.Module):
    """
    Unified model for fiber optic analysis handling:
    1. Region classification (core, cladding, ferrule)
    2. Anomaly detection
    3. Similarity comparison via embeddings
    """
    
    def __init__(self, config):
        super().__init__()
        # Convert dict to Box if needed
        if isinstance(config, dict):
            from config import Box
            self.config = Box(config)
        else:
            self.config = config
        
        # Feature extractor backbone
        self._build_backbone()
        
        # Attention mechanism
        self.attention = CBAM(self.backbone_out_channels)

        # Task-specific heads
        self._build_heads()

    def _build_backbone(self):
        """Build the feature extraction backbone."""
        try:
            # Handle different backbone specifications
            if hasattr(models, self.config.backbone):
                backbone_fn = getattr(models, self.config.backbone)
                resnet = backbone_fn(weights='DEFAULT')
            else:
                # Fallback to ResNet18
                resnet = models.resnet18(weights='DEFAULT')
                
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            
            # Determine output channels
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, self.config.image_size, self.config.image_size)
                self.backbone_out_channels = self.encoder(dummy_input).shape[1]
                
        except Exception as e:
            # Fallback backbone
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.backbone_out_channels = 256

    def _build_heads(self):
        """Build task-specific prediction heads."""
        # Region classification head
        self.region_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, self.config.num_classes)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        # Embedding head for similarity comparison
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone_out_channels, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        )

    def forward(self, image, ref_image=None, equation_coeffs=None):
        """
        Forward pass through the network.
        
        Args:
            image: Input fiber optic image
            ref_image: Optional reference image for comparison
            equation_coeffs: Coefficients for final score calculation
            
        Returns:
            Dictionary containing all predictions and scores
        """
        # Extract and attend to features
        features = self.encoder(image)
        attended_features = self.attention(features)
        
        # Generate predictions
        region_logits = self.region_classifier(attended_features)
        anomaly_map = self.anomaly_head(attended_features)
        embedding = self.embedding_head(attended_features)
        
        # Process reference image if provided
        ref_embedding = None
        if ref_image is not None:
            with torch.no_grad():
                ref_features = self.encoder(ref_image)
                ref_attended = self.attention(ref_features)
                ref_embedding = self.embedding_head(ref_attended)
        
        # Calculate intermediate scores
        region_probs = F.softmax(region_logits, dim=1)
        classification_confidence, _ = torch.max(region_probs, dim=1)
        anomaly_score = torch.sigmoid(anomaly_map).mean(dim=[1, 2, 3])
        
        # Embedding similarity
        embedding_similarity = torch.zeros_like(classification_confidence)
        if ref_embedding is not None:
            embedding_similarity = F.cosine_similarity(embedding, ref_embedding)

        # Final similarity score calculation
        coeffs = equation_coeffs or {'A': 0.5, 'B': 0.3, 'C': 0.2}
        final_score = (
            coeffs['A'] * classification_confidence +
            coeffs['B'] * (1 - anomaly_score) +
            coeffs['C'] * embedding_similarity
        )
        
        return {
            "region_logits": region_logits,
            "region_probs": region_probs,
            "anomaly_map": anomaly_map,
            "embedding": embedding,
            "ref_embedding": ref_embedding,
            "classification_confidence": classification_confidence,
            "anomaly_score": anomaly_score,
            "embedding_similarity": embedding_similarity,
            "final_similarity_score": final_score
        }

def create_model(config):
    """Factory function to create model instance."""
    return FiberOpticNet(config.model)

def load_model(config, checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    model = create_model(config)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    return model
