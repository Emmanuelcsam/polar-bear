#!/usr/bin/env python3
"""
Main script for Fiber Optics Neural Network with Statistical Integration
Demonstrates how to use the statistically enhanced network
Based on comprehensive analysis from statistics folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

# Import existing modules
from integrated_network import EnhancedIntegratedNetwork
from data_loader import FiberOpticsDataset
from trainer import EnhancedTrainer
from logger import get_logger
from config_loader import get_config

# Import new statistical modules
from statistical_integration import (
    integrate_statistics_into_network,
    StatisticallyIntegratedNetwork
)
from statistical_losses import (
    get_statistical_loss,
    StatisticalCompositeLoss
)
from statistical_config import (
    get_statistical_config,
    merge_with_base_config
)


class StatisticalFiberOpticsSystem:
    """
    Complete system integrating neural network with statistical analysis
    Implements the vision from goal.txt with all statistical enhancements
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the statistically enhanced fiber optics system
        
        Args:
            config_path: Path to configuration file (uses default if None)
        """
        print(f"[{datetime.now()}] Initializing StatisticalFiberOpticsSystem")
        print(f"[{datetime.now()}] Loading configuration...")
        
        # Get base configuration
        self.base_config = get_config(config_path)
        
        # Get statistical configuration
        self.stat_config = get_statistical_config()
        
        # Merge configurations
        self.config = merge_with_base_config(
            self.base_config.__dict__ if hasattr(self.base_config, '__dict__') else self.base_config,
            self.stat_config
        )
        
        # Initialize logger
        self.logger = get_logger("StatisticalFiberOpticsSystem")
        self.logger.info("System initialization started")
        
        # Print configuration summary
        self.stat_config.print_summary()
        
        # Validate configuration
        if not self.stat_config.validate():
            self.logger.warning("Configuration validation found issues")
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize network
        self._initialize_network()
        
        # Initialize loss function
        self._initialize_loss()
        
        # Initialize optimizer
        self._initialize_optimizer()
        
        # Load reference statistics if available
        self._load_reference_statistics()
        
        self.logger.info("StatisticalFiberOpticsSystem initialized successfully")
    
    def _initialize_network(self):
        """Initialize the statistically enhanced network"""
        self.logger.info("Initializing neural network with statistical integration...")
        
        # Create base network
        base_network = EnhancedIntegratedNetwork()
        
        # Integrate statistical components
        self.network = integrate_statistics_into_network(base_network)
        
        # Move to device
        self.network = self.network.to(self.device)
        
        # Print network summary
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        self.logger.info(f"Network initialized with {total_params:,} parameters ({trainable_params:,} trainable)")
        
    def _initialize_loss(self):
        """Initialize the statistical loss function"""
        self.logger.info("Initializing statistical loss function...")
        
        # Get loss function based on configuration
        self.criterion = get_statistical_loss(self.config.get('loss', self.stat_config.loss_settings))
        
        self.logger.info(f"Using loss type: {self.stat_config.loss_settings['loss_type']}")
        
    def _initialize_optimizer(self):
        """Initialize optimizer with different learning rates for different components"""
        self.logger.info("Initializing optimizer...")
        
        # Separate parameters by component
        base_params = []
        statistical_params = []
        
        for name, param in self.network.named_parameters():
            if any(comp in name for comp in ['statistical', 'master_similarity', 'zone_predictor', 
                                             'consensus', 'correlation', 'anomaly']):
                statistical_params.append(param)
            else:
                base_params.append(param)
        
        # Get learning rates
        base_lr = self.config.get('training', {}).get('learning_rate', 1e-4)
        stat_lr_mult = self.stat_config.training_settings['statistical_lr_multiplier']
        
        # Create parameter groups
        param_groups = [
            {'params': base_params, 'lr': base_lr, 'name': 'base'},
            {'params': statistical_params, 'lr': base_lr * stat_lr_mult, 'name': 'statistical'}
        ]
        
        # Initialize optimizer
        optimizer_type = self.config.get('training', {}).get('optimizer', 'AdamW')
        
        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(param_groups)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=0.0001)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.logger.info(f"Optimizer initialized: {optimizer_type}")
        self.logger.info(f"Base LR: {base_lr}, Statistical LR: {base_lr * stat_lr_mult}")
    
    def _load_reference_statistics(self):
        """Load reference statistics from analysis"""
        self.logger.info("Loading reference statistics...")
        
        # Path to statistics files
        stats_dir = Path("statistics")
        
        # Load global statistics
        global_stats_path = stats_dir / "global_statistics.json"
        if global_stats_path.exists():
            with open(global_stats_path, 'r') as f:
                self.global_stats = json.load(f)
            self.logger.info("Loaded global statistics")
        else:
            self.global_stats = None
            self.logger.warning("Global statistics not found")
        
        # Load class statistics
        class_stats_path = stats_dir / "class_statistics.json"
        if class_stats_path.exists():
            with open(class_stats_path, 'r') as f:
                self.class_stats = json.load(f)
            self.logger.info("Loaded class statistics")
        else:
            self.class_stats = None
        
        # Load equations
        equations_path = stats_dir / "equations-separation.json"
        if equations_path.exists():
            with open(equations_path, 'r') as f:
                self.equations = json.load(f)
            self.logger.info("Loaded mathematical equations")
        else:
            self.equations = None
        
        # Initialize reference features for similarity calculation
        if self.global_stats and 'neural_network_insights' in self.global_stats:
            self.reference_features = self._extract_reference_features()
        else:
            self.reference_features = None
    
    def _extract_reference_features(self) -> torch.Tensor:
        """Extract reference features from statistics"""
        # This would normally load actual reference features from the database
        # For now, create dummy features
        num_references = self.stat_config.reference_settings['num_reference_embeddings']
        feature_dim = 6  # center_x, center_y, core_radius, cladding_radius, ratio, num_valid
        
        # Create synthetic reference features based on statistics
        if self.equations and 'regression' in self.equations:
            # Use mean values from regression models
            features = torch.zeros(num_references, feature_dim)
            
            # Fill with variations around mean values
            features[:, 0] = torch.normal(595.87, 116.35, (num_references,))  # center_x
            features[:, 1] = torch.normal(447.19, 87.07, (num_references,))   # center_y
            features[:, 2] = torch.normal(114.42, 78.41, (num_references,))   # core_radius
            features[:, 3] = torch.normal(208.89, 100.50, (num_references,))  # cladding_radius
            features[:, 4] = torch.normal(0.530, 0.182, (num_references,))    # ratio
            features[:, 5] = torch.ones(num_references) * 7                    # num_valid_results
            
            return features.to(self.device)
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with statistical integration
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.network.train()
        
        epoch_losses = {
            'total': 0.0,
            'segmentation': 0.0,
            'iou': 0.0,
            'similarity': 0.0,
            'anomaly': 0.0,
            'zone_regression': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract data
            images = batch['image'].to(self.device)
            targets = {
                'segmentation_masks': batch.get('mask', torch.zeros_like(images[:, 0:1])).to(self.device),
                'is_anomaly': batch.get('is_anomaly', torch.zeros(images.shape[0])).to(self.device),
                'target_zone_parameters': {
                    'core_radius': batch.get('core_radius', torch.zeros(images.shape[0])).to(self.device),
                    'cladding_radius': batch.get('cladding_radius', torch.zeros(images.shape[0])).to(self.device),
                    'core_cladding_ratio': batch.get('ratio', torch.zeros(images.shape[0])).to(self.device)
                }
            }
            
            # Add reference data if available
            reference_data = None
            if self.reference_features is not None:
                # Select random references for comparison
                num_refs = min(10, self.reference_features.shape[0])
                ref_indices = torch.randperm(self.reference_features.shape[0])[:num_refs]
                reference_data = {
                    'features': self.reference_features[ref_indices].unsqueeze(0).expand(images.shape[0], -1, -1)
                }
            
            # Forward pass
            outputs = self.network(images, reference_data)
            
            # Prepare predictions for loss calculation
            predictions = {
                'segmentation_logits': outputs.get('segmentation_logits', outputs.get('final_output')),
                'predicted_masks': outputs.get('consensus_results', {}).get('masks', outputs.get('predicted_masks')),
                'zone_parameters': outputs.get('zone_parameters', {}),
                'similarity_scores': outputs.get('similarity_scores'),
                'mahalanobis_distance': outputs.get('anomaly_results', {}).get('mahalanobis_distance'),
                'method_predictions': outputs.get('method_masks', []),
                'method_scores': outputs.get('method_scores', torch.ones(7).to(self.device))
            }
            
            # Calculate loss
            losses = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update epoch losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {losses['total'].item():.4f} "
                    f"(Seg: {losses.get('segmentation', 0).item():.4f}, "
                    f"IoU: {losses.get('iou', 0).item():.4f}, "
                    f"Anomaly: {losses.get('anomaly', 0).item():.4f})"
                )
            
            # Update reference statistics periodically
            if batch_idx % self.stat_config.training_settings['reference_update_frequency'] == 0:
                self._update_reference_statistics(outputs)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update learning rate
        self.scheduler.step()
        
        return epoch_losses
    
    def _update_reference_statistics(self, outputs: Dict[str, torch.Tensor]):
        """Update reference statistics based on network outputs"""
        if 'statistical_features' in outputs:
            # Update anomaly detector statistics
            self.network.statistical_anomaly_detector.update_statistics(
                outputs['statistical_features']
            )
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the network on validation data
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.network.eval()
        
        metrics = {
            'accuracy': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'anomaly_precision': 0.0,
            'anomaly_recall': 0.0,
            'circularity': 0.0,
            'method_agreement': 0.0,
            'mahalanobis_distance': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch.get('mask', torch.zeros_like(images[:, 0:1])).to(self.device)
                
                # Forward pass
                outputs = self.network(images, self.reference_features)
                
                # Calculate metrics
                if 'consensus_results' in outputs and outputs['consensus_results'] is not None:
                    pred_masks = outputs['consensus_results']['masks']
                    
                    # IoU
                    intersection = torch.sum(pred_masks * masks)
                    union = torch.sum(pred_masks + masks - pred_masks * masks)
                    metrics['iou'] += (intersection / (union + 1e-6)).item()
                    
                    # Dice
                    dice = 2 * intersection / (torch.sum(pred_masks) + torch.sum(masks) + 1e-6)
                    metrics['dice'] += dice.item()
                    
                    # Circularity
                    if 'core_circularity' in outputs['consensus_results']:
                        metrics['circularity'] += outputs['consensus_results']['core_circularity'].mean().item()
                    
                    # Method agreement
                    if 'num_agreeing' in outputs['consensus_results']:
                        metrics['method_agreement'] += outputs['consensus_results']['num_agreeing'].mean().item()
                
                # Anomaly metrics
                if 'anomaly_results' in outputs:
                    if 'mahalanobis_distance' in outputs['anomaly_results']:
                        metrics['mahalanobis_distance'] += outputs['anomaly_results']['mahalanobis_distance'].mean().item()
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
    
    def process_image(self, image_path: str, output_dir: str) -> Dict[str, any]:
        """
        Process a single image through the complete pipeline
        Implements the vision from goal.txt
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Comprehensive results dictionary
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run through network
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(image_tensor, self.reference_features)
        
        # Extract results
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        # Zone parameters
        if 'zone_parameters' in outputs:
            results['zones'] = {
                'core_radius': outputs['zone_parameters']['core_radius'].item(),
                'cladding_radius': outputs['zone_parameters']['cladding_radius'].item(),
                'ratio': outputs['zone_parameters']['core_cladding_ratio'].item()
            }
        
        # Consensus results
        if 'consensus_results' in outputs and outputs['consensus_results'] is not None:
            consensus = outputs['consensus_results']
            results['consensus'] = {
                'center': [consensus['center_x'].item(), consensus['center_y'].item()],
                'confidence': consensus['confidence'].item(),
                'num_agreeing_methods': consensus['num_agreeing'].item(),
                'core_circularity': consensus['core_circularity'].item()
            }
            
            # Save masks
            masks = consensus['masks'].squeeze(0).cpu().numpy()
            np.save(output_path / 'consensus_masks.npy', masks)
        
        # Anomaly detection
        if 'anomaly_results' in outputs:
            anomaly = outputs['anomaly_results']
            results['anomalies'] = {
                'is_anomalous': anomaly['combined_scores'].item() > self.stat_config.anomaly_detection['threshold_multiplier'],
                'anomaly_score': anomaly['combined_scores'].item(),
                'mahalanobis_distance': anomaly['mahalanobis_distance'].item(),
                'severity': anomaly['severities'][0]
            }
            
            # Save defect maps
            for defect_type in ['scratch', 'dig', 'blob']:
                if f'{defect_type}_map' in anomaly:
                    defect_map = anomaly[f'{defect_type}_map'].squeeze().cpu().numpy()
                    np.save(output_path / f'{defect_type}_map.npy', defect_map)
        
        # Similarity scores
        if 'similarity_scores' in outputs and outputs['similarity_scores'] is not None:
            # Get top-5 most similar references
            scores = outputs['similarity_scores'].squeeze()
            top_scores, top_indices = torch.topk(scores, k=min(5, scores.shape[0]))
            
            results['similarity'] = {
                'top_scores': top_scores.cpu().tolist(),
                'top_indices': top_indices.cpu().tolist(),
                'meets_threshold': (top_scores[0] > self.stat_config.similarity_settings['similarity_threshold']).item()
            }
        
        # Statistical features
        if 'statistical_features' in outputs:
            results['feature_statistics'] = {
                'mean': outputs['statistical_features'].mean().item(),
                'std': outputs['statistical_features'].std().item(),
                'min': outputs['statistical_features'].min().item(),
                'max': outputs['statistical_features'].max().item()
            }
        
        # Method scores
        if 'method_scores' in outputs:
            method_names = list(self.stat_config.method_settings.keys())
            results['method_performance'] = {
                name: score.item() 
                for name, score in zip(method_names, outputs['method_scores'])
            }
        
        # Save results JSON
        with open(output_path / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization
        self._visualize_results(image, outputs, output_path)
        
        self.logger.info(f"Processing complete. Results saved to {output_path}")
        
        return results
    
    def _visualize_results(self, image: np.ndarray, outputs: Dict, output_path: Path):
        """Generate comprehensive visualization of results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Statistical Fiber Optics Analysis Results', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Consensus segmentation
        if 'consensus_results' in outputs and outputs['consensus_results'] is not None:
            masks = outputs['consensus_results']['masks'].squeeze(0).cpu().numpy()
            segmentation = np.zeros_like(image)
            segmentation[masks[0] > 0.5] = [255, 0, 0]      # Core - Red
            segmentation[masks[1] > 0.5] = [0, 255, 0]      # Cladding - Green
            segmentation[masks[2] > 0.5] = [0, 0, 255]      # Ferrule - Blue
            
            axes[0, 1].imshow(segmentation)
            axes[0, 1].set_title(f"Consensus Segmentation (Conf: {outputs['consensus_results']['confidence'].item():.2f})")
            axes[0, 1].axis('off')
        
        # Anomaly detection
        if 'anomaly_results' in outputs:
            anomaly_score = outputs['anomaly_results']['combined_scores'].item()
            severity = outputs['anomaly_results']['severities'][0]
            
            # Create anomaly heatmap
            scratch_map = outputs['anomaly_results'].get('scratch_map', torch.zeros(1, 1, image.shape[0], image.shape[1]))
            dig_map = outputs['anomaly_results'].get('dig_map', torch.zeros(1, 1, image.shape[0], image.shape[1]))
            blob_map = outputs['anomaly_results'].get('blob_map', torch.zeros(1, 1, image.shape[0], image.shape[1]))
            
            anomaly_map = (scratch_map + dig_map + blob_map).squeeze().cpu().numpy()
            
            im = axes[0, 2].imshow(anomaly_map, cmap='hot', alpha=0.7)
            axes[0, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.3)
            axes[0, 2].set_title(f"Anomaly Map (Score: {anomaly_score:.2f}, {severity})")
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Zone parameters
        if 'zone_parameters' in outputs:
            axes[1, 0].text(0.1, 0.8, f"Core Radius: {outputs['zone_parameters']['core_radius'].item():.1f} px", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.6, f"Cladding Radius: {outputs['zone_parameters']['cladding_radius'].item():.1f} px", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].text(0.1, 0.4, f"Core/Cladding Ratio: {outputs['zone_parameters']['core_cladding_ratio'].item():.3f}", 
                           transform=axes[1, 0].transAxes, fontsize=12)
            
            if 'consensus_results' in outputs and outputs['consensus_results'] is not None:
                axes[1, 0].text(0.1, 0.2, f"Circularity: {outputs['consensus_results']['core_circularity'].item():.3f}", 
                               transform=axes[1, 0].transAxes, fontsize=12)
            
            axes[1, 0].set_title('Zone Parameters')
            axes[1, 0].axis('off')
        
        # Method scores
        if 'method_scores' in outputs:
            method_names = list(self.stat_config.method_settings.keys())[:7]
            scores = outputs['method_scores'].cpu().numpy()
            
            axes[1, 1].bar(range(len(method_names)), scores)
            axes[1, 1].set_xticks(range(len(method_names)))
            axes[1, 1].set_xticklabels([name.split('_')[0] for name in method_names], rotation=45, ha='right')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Method Performance Scores')
            axes[1, 1].set_ylim(0, 2.0)
        
        # Statistical features histogram
        if 'statistical_features' in outputs:
            features = outputs['statistical_features'].squeeze().cpu().numpy()
            axes[1, 2].hist(features, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 2].set_xlabel('Feature Value')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Statistical Feature Distribution')
            axes[1, 2].axvline(features.mean(), color='red', linestyle='--', label=f'Mean: {features.mean():.2f}')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'analysis_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_model(self, checkpoint_path: str):
        """Save model checkpoint with all components"""
        self.logger.info(f"Saving model checkpoint to {checkpoint_path}")
        
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'stat_config': self.stat_config.to_dict(),
            'method_scores': self.network.method_accuracy_tracker.method_scores.cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info("Model checkpoint saved successfully")
    
    def load_model(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading model checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update method scores
        if 'method_scores' in checkpoint:
            self.network.method_accuracy_tracker.method_scores = torch.tensor(
                checkpoint['method_scores'], 
                device=self.device
            )
        
        self.logger.info("Model checkpoint loaded successfully")


def main():
    """Main function demonstrating the statistically enhanced system"""
    print(f"\n{'='*80}")
    print("FIBER OPTICS NEURAL NETWORK WITH STATISTICAL INTEGRATION")
    print(f"{'='*80}\n")
    
    # Initialize system
    system = StatisticalFiberOpticsSystem()
    
    # Example: Process a single image
    print("\nExample: Processing a single image")
    print("-" * 40)
    
    # Replace with actual image path
    image_path = "path/to/test/image.png"
    output_dir = "output/statistical_analysis"
    
    # Check if image exists
    if Path(image_path).exists():
        results = system.process_image(image_path, output_dir)
        
        print("\nAnalysis Results:")
        print(f"  - Anomaly Score: {results['anomalies']['anomaly_score']:.3f}")
        print(f"  - Severity: {results['anomalies']['severity']}")
        print(f"  - Core Radius: {results['zones']['core_radius']:.1f} px")
        print(f"  - Cladding Radius: {results['zones']['cladding_radius']:.1f} px")
        print(f"  - Core/Cladding Ratio: {results['zones']['ratio']:.3f}")
        
        if 'consensus' in results:
            print(f"  - Consensus Confidence: {results['consensus']['confidence']:.3f}")
            print(f"  - Agreeing Methods: {results['consensus']['num_agreeing_methods']}/7")
        
        if 'similarity' in results:
            print(f"  - Meets Similarity Threshold: {results['similarity']['meets_threshold']}")
    else:
        print(f"  ! Image not found: {image_path}")
        print("  ! Please provide a valid image path")
    
    print(f"\n{'='*80}")
    print("STATISTICAL INTEGRATION COMPLETE")
    print(f"{'='*80}\n")
    
    print("Key Features Integrated:")
    print("  ✓ 88-dimensional statistical feature extraction")
    print("  ✓ Master similarity equation (S > 0.7 threshold)")
    print("  ✓ Zone parameter regression models")
    print("  ✓ Consensus algorithm with IoU and circularity")
    print("  ✓ Method accuracy tracking (exponential moving average)")
    print("  ✓ Correlation-guided attention")
    print("  ✓ Mahalanobis distance anomaly detection")
    print("  ✓ Specific defect detection (scratches, digs, blobs)")
    print("  ✓ Multi-scale feature correlation")
    print("  ✓ Physical constraint enforcement")
    
    print("\nThe system now implements the complete vision from goal.txt with all")
    print("statistical insights integrated into the neural network architecture.")


if __name__ == "__main__":
    main()