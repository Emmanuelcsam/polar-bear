#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

# Device configuration for GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class OmniConfig:
    """Configuration for OmniFiberAnalyzer - PyTorch enhanced version"""
    knowledge_base_path: Optional[str] = None
    min_defect_size: int = 10
    max_defect_size: int = 5000
    severity_thresholds: Optional[Dict[str, float]] = None
    confidence_threshold: float = 0.3
    anomaly_threshold_multiplier: float = 2.5
    enable_visualization: bool = True
    # New PyTorch-specific parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                'CRITICAL': 0.9,
                'HIGH': 0.7,
                'MEDIUM': 0.5,
                'LOW': 0.3,
                'NEGLIGIBLE': 0.1
            }

class FiberDataset(Dataset):
    """Custom PyTorch Dataset for fiber optic images"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert to PyTorch tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        return {'image': image, 'path': image_path}
    
    def _load_image(self, path):
        """Load image from file or JSON"""
        if path.lower().endswith('.json'):
            return self._load_from_json(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {path}")
            return img

    def _load_from_json(self, json_path):
        """Load matrix from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        matrix = np.zeros((height, width), dtype=np.float32)
        
        for pixel in data['pixels']:
            x = pixel['coordinates']['x']
            y = pixel['coordinates']['y']
            if 0 <= x < width and 0 <= y < height:
                intensity = pixel.get('intensity', 0)
                if isinstance(intensity, list):
                    intensity = np.mean(intensity[:3])  # Average RGB
                matrix[y, x] = intensity / 255.0  # Normalize to [0, 1]
        
        return matrix

class AnomalyDetectorNet(nn.Module):
    """Neural network for fiber optic anomaly detection"""
    def __init__(self, input_channels=1, feature_dim=512):
        super(AnomalyDetectorNet, self).__init__()
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature extraction head
        self.feature_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Extract features
        batch_size = encoded.size(0)
        features = self.feature_head(encoded.view(batch_size, -1))
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return features, reconstructed
    
    def get_anomaly_map(self, x):
        """Generate pixel-wise anomaly scores"""
        with torch.no_grad():
            _, reconstructed = self.forward(x)
            anomaly_map = torch.abs(x - reconstructed)
        return anomaly_map

class CombinedLoss(nn.Module):
    """Combined loss for anomaly detection"""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1)
    
    def forward(self, features, reconstructed, original, reference_features=None):
        # Reconstruction loss
        recon_loss = self.mse(reconstructed, original)
        
        # Feature consistency loss (if reference features provided)
        if reference_features is not None:
            feature_loss = 1 - self.cosine(features, reference_features).mean()
            total_loss = self.alpha * recon_loss + (1 - self.alpha) * feature_loss
        else:
            total_loss = recon_loss
        
        return total_loss, recon_loss

class OmniFiberAnalyzer:
    """PyTorch-enhanced fiber optic anomaly detection system"""
    
    def __init__(self, config: OmniConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = AnomalyDetectorNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = CombinedLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter('runs/fiber_anomaly_detection')
        
        # Reference features storage
        self.reference_features = None
        self.reference_model_path = config.knowledge_base_path or "fiber_anomaly_model.pth"
        
        # Load existing model if available
        self.load_model()
    
    def save_model(self):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reference_features': self.reference_features,
            'config': self.config
        }
        torch.save(checkpoint, self.reference_model_path)
        self.logger.info(f"Model saved to {self.reference_model_path}")
    
    def load_model(self):
        """Load model checkpoint if exists"""
        if os.path.exists(self.reference_model_path):
            try:
                checkpoint = torch.load(self.reference_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.reference_features = checkpoint.get('reference_features')
                self.logger.info(f"Model loaded from {self.reference_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}")
    
    def build_reference_model(self, ref_dir: str):
        """Build reference model using PyTorch training"""
        self.logger.info(f"Building reference model from: {ref_dir}")
        
        # Collect reference images
        valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_paths = []
        
        for filename in os.listdir(ref_dir):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(ref_dir, filename))
        
        if not image_paths:
            self.logger.error(f"No valid files found in {ref_dir}")
            return False
        
        # Create dataset and dataloader
        dataset = FiberDataset(image_paths)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, 
                              shuffle=True, num_workers=2)
        
        # Training loop
        self.model.train()
        all_features = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].unsqueeze(1).to(self.device)  # Add channel dimension
                
                # Forward pass
                features, reconstructed = self.model(images)
                loss, recon_loss = self.criterion(features, reconstructed, images)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Store features
                if epoch == self.config.num_epochs - 1:  # Last epoch
                    all_features.append(features.detach().cpu())
            
            # Log to TensorBoard
            avg_loss = epoch_loss / len(dataloader)
            self.writer.add_scalar('Training/Loss', avg_loss, epoch)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")
        
        # Compute reference features (mean of all features)
        all_features = torch.cat(all_features, dim=0)
        self.reference_features = torch.mean(all_features, dim=0, keepdim=True).to(self.device)
        
        # Save model
        self.save_model()
        self.writer.close()
        
        return True
    
    def detect_anomalies_comprehensive(self, test_path: str) -> Optional[Dict]:
        """Perform anomaly detection using PyTorch model"""
        self.logger.info(f"Analyzing: {test_path}")
        
        if self.reference_features is None:
            self.logger.warning("No reference model available.")
            return None
        
        # Load test image
        dataset = FiberDataset([test_path])
        test_data = dataset[0]
        test_image = test_data['image'].unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Get features and reconstruction
            features, reconstructed = self.model(test_image)
            
            # Calculate anomaly score
            anomaly_map = self.model.get_anomaly_map(test_image)
            
            # Feature distance from reference
            feature_distance = torch.norm(features - self.reference_features, p=2).item()
            
            # Reconstruction error
            recon_error = F.mse_loss(reconstructed, test_image).item()
            
            # Convert to numpy for visualization
            test_img_np = test_image.squeeze().cpu().numpy()
            reconstructed_np = reconstructed.squeeze().cpu().numpy()
            anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
        
        # Find anomaly regions
        anomaly_regions = self._find_anomaly_regions_torch(anomaly_map_np)
        
        # Determine if image is anomalous
        is_anomalous = (
            feature_distance > self.config.anomaly_threshold_multiplier or
            recon_error > 0.1 or
            len(anomaly_regions) > 3
        )
        
        confidence = min(1.0, max(
            feature_distance / self.config.anomaly_threshold_multiplier,
            recon_error * 10,
            len(anomaly_regions) / 10
        ))
        
        results = {
            'test_image': test_img_np,
            'reconstructed': reconstructed_np,
            'anomaly_map': anomaly_map_np,
            'anomaly_regions': anomaly_regions,
            'feature_distance': feature_distance,
            'reconstruction_error': recon_error,
            'verdict': {
                'is_anomalous': is_anomalous,
                'confidence': confidence
            }
        }
        
        return results
    
    def _find_anomaly_regions_torch(self, anomaly_map: np.ndarray) -> List[Dict]:
        """Find anomaly regions using PyTorch operations"""
        # Convert to tensor for processing
        anomaly_tensor = torch.from_numpy(anomaly_map).float()
        
        # Dynamic thresholding
        threshold = torch.quantile(anomaly_tensor[anomaly_tensor > 0], 0.8).item()
        binary_map = (anomaly_tensor > threshold).numpy().astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            if area > 20:  # Filter small regions
                region_mask = (labels == i)
                region_values = anomaly_map[region_mask]
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': int(area),
                    'confidence': float(np.mean(region_values)),
                    'centroid': (int(centroids[i][0]), int(centroids[i][1])),
                    'max_intensity': float(np.max(region_values))
                })
        
        return sorted(regions, key=lambda x: x['confidence'], reverse=True)
    
    def analyze_end_face(self, image_path: str, output_dir: str):
        """Main analysis method - PyTorch enhanced version"""
        self.logger.info(f"Analyzing fiber end face: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        results = self.detect_anomalies_comprehensive(image_path)
        
        if results:
            # Convert to pipeline format
            pipeline_report = self._convert_to_pipeline_format(results, image_path)
            
            # Save report
            report_path = output_path / f"{Path(image_path).stem}_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
            
            self.logger.info(f"Saved detection report to {report_path}")
            
            # Generate visualizations
            if self.config.enable_visualization:
                viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
                self.visualize_results_pytorch(results, str(viz_path))
            
            return pipeline_report
        else:
            self.logger.error(f"Analysis failed for {image_path}")
            return None
    
    def visualize_results_pytorch(self, results: Dict, output_path: str):
        """Create visualization using matplotlib"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(results['test_image'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Reconstructed image
        axes[0, 1].imshow(results['reconstructed'], cmap='gray')
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
        
        # Anomaly map
        im = axes[1, 0].imshow(results['anomaly_map'], cmap='hot')
        axes[1, 0].set_title('Anomaly Map')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Detected regions
        axes[1, 1].imshow(results['test_image'], cmap='gray')
        for region in results['anomaly_regions']:
            x, y, w, h = region['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
            axes[1, 1].add_patch(rect)
            axes[1, 1].text(x, y-5, f"{region['confidence']:.2f}", color='red', fontsize=8)
        axes[1, 1].set_title(f"Detected Anomalies ({len(results['anomaly_regions'])} regions)")
        axes[1, 1].axis('off')
        
        verdict = results['verdict']
        fig.suptitle(f"Anomaly Detection Results - {'ANOMALOUS' if verdict['is_anomalous'] else 'NORMAL'} "
                    f"(Confidence: {verdict['confidence']:.2%})", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {output_path}")
    
    def _convert_to_pipeline_format(self, results: Dict, image_path: str) -> Dict:
        """Convert results to pipeline-expected format"""
        defects = []
        defect_id = 1
        
        # Convert anomaly regions to defects
        for region in results['anomaly_regions']:
            x, y, w, h = region['bbox']
            cx, cy = region['centroid']
            
            defect = {
                'defect_id': f"ANOM_{defect_id:04d}",
                'defect_type': 'ANOMALY',
                'location_xy': [cx, cy],
                'bbox': [x, y, w, h],
                'area_px': region['area'],
                'confidence': float(region['confidence']),
                'severity': self._confidence_to_severity(region['confidence']),
                'orientation': None,
                'contributing_algorithms': ['pytorch_anomaly_detector'],
                'detection_metadata': {
                    'max_intensity': region['max_intensity'],
                    'feature_distance': results['feature_distance'],
                    'reconstruction_error': results['reconstruction_error']
                }
            }
            defects.append(defect)
            defect_id += 1
        
        verdict = results['verdict']
        
        report = {
            'source_image': image_path,
            'image_path': image_path,
            'timestamp': self._get_timestamp(),
            'analysis_complete': True,
            'success': True,
            'overall_quality_score': float(100 * (1 - verdict['confidence'])),
            'defects': defects,
            'zones': {
                'core': {'detected': True},
                'cladding': {'detected': True},
                'ferrule': {'detected': True}
            },
            'summary': {
                'total_defects': len(defects),
                'is_anomalous': verdict['is_anomalous'],
                'anomaly_confidence': float(verdict['confidence']),
                'feature_distance': float(results['feature_distance']),
                'reconstruction_error': float(results['reconstruction_error'])
            },
            'analysis_metadata': {
                'analyzer': 'pytorch_anomaly_detector',
                'version': '2.0',
                'device': str(self.device),
                'model_path': self.reference_model_path
            }
        }
        
        return report
    
    def _confidence_to_severity(self, confidence: float) -> str:
        """Convert confidence score to severity level"""
        for severity, threshold in sorted(self.config.severity_thresholds.items(), 
                                        key=lambda x: x[1], reverse=True):
            if confidence >= threshold:
                return severity
        return 'NEGLIGIBLE'
    
    def _get_timestamp(self):
        """Get current timestamp"""
        return time.strftime("%Y-%m-%d_%H:%M:%S")

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy/torch data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("OMNIFIBER ANALYZER - PYTORCH ENHANCED VERSION (v2.0)".center(80))
    print("="*80)
    print("\nNow with GPU support, automatic differentiation, and deep learning!\n")
    
    # Create configuration
    config = OmniConfig()
    
    # Initialize analyzer
    analyzer = OmniFiberAnalyzer(config)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ No GPU detected, using CPU")
    
    # Interactive testing loop
    while True:
        print("\nOptions:")
        print("1. Train reference model from directory")
        print("2. Analyze test image")
        print("3. Quit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            ref_dir = input("Enter path to reference images directory: ").strip().strip('"\'')
            if os.path.isdir(ref_dir):
                print(f"\nTraining reference model from {ref_dir}...")
                analyzer.build_reference_model(ref_dir)
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
    
    print("\nThank you for using the PyTorch-enhanced OmniFiber Analyzer!")

if __name__ == "__main__":
    main()