import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
from tqdm import tqdm

from ..utils.logger import get_logger
from ..config.config import get_config
from ..models.fiber_optics_nn import FiberOpticsNeuralNetwork
from ..processors.image_processor import ImageProcessor
from ..processors.segmentation import FiberOpticSegmentation
from ..core.similarity import SimilarityCalculator
from ..core.anomaly_detection import AnomalyDetector
from ..data_loaders.tensor_loader import TensorDataLoader


class FiberOpticsDataset(Dataset):
    """Custom dataset for fiber optics images"""
    
    def __init__(self, tensor_paths: List[Path], labels: Optional[List[int]] = None):
        self.tensor_paths = tensor_paths
        self.labels = labels if labels else [0] * len(tensor_paths)
        
    def __len__(self):
        return len(self.tensor_paths)
    
    def __getitem__(self, idx):
        tensor = torch.load(self.tensor_paths[idx], map_location='cpu')
        label = self.labels[idx]
        return tensor, label


class FiberOpticsTrainer:
    """
    Training loop for the fiber optics neural network.
    "the program will calculate its losses and try to minimize its losses by 
    small percentile adjustments to parameters"
    "but this is only for the training sequence so that it can be trained fast 
    and effectively with a large database"
    """
    
    def __init__(self, model: FiberOpticsNeuralNetwork):
        self.config = get_config()
        self.logger = get_logger()
        self.device = self.config.get_device()
        
        self.model = model.to(self.device)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.segmentation = FiberOpticSegmentation()
        self.similarity_calculator = SimilarityCalculator()
        self.anomaly_detector = AnomalyDetector()
        self.data_loader = TensorDataLoader()
        
        # Optimizer with small learning rate for fine adjustments
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss functions
        self.similarity_loss = nn.CosineEmbeddingLoss()
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Track training metrics
        self.training_history = {
            'loss': [],
            'similarity_scores': [],
            'defect_detection_accuracy': [],
            'region_classification_accuracy': []
        }
        
        self.logger.info("Initialized FiberOpticsTrainer")
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop.
        "over time the program will get more accurate and faster at computing"
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        # Load reference data
        self.logger.info("Loading reference tensors...")
        self.data_loader.load_all_references(preload=True)
        
        # Create training dataset
        train_dataset = self._create_training_dataset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            self.model.train()
            epoch_losses = []
            epoch_similarities = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Process batch
                loss, metrics = self._train_step(images, labels)
                
                epoch_losses.append(loss.item())
                epoch_similarities.append(metrics['similarity'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'similarity': f"{metrics['similarity']:.4f}"
                })
                
                # Log batch metrics
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx}: "
                        f"Loss={loss.item():.4f}, Similarity={metrics['similarity']:.4f}"
                    )
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            avg_similarity = np.mean(epoch_similarities)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['similarity_scores'].append(avg_similarity)
            
            # Adjust learning rate
            self.scheduler.step(avg_loss)
            
            # Validation
            if epoch % 5 == 0:
                val_metrics = self._validate()
                self.logger.info(f"Validation metrics: {val_metrics}")
            
            epoch_time = time.time() - epoch_start
            self.logger.log_performance_metric(f"epoch_{epoch+1}_time", epoch_time, "s")
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
        
        self.logger.info("Training completed")
    
    def _train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Single training step.
        "the program will forcibly look for all lines of best fit based on gradient 
        trends for all datapoints and pixels"
        """
        self.optimizer.zero_grad()
        
        batch_size = images.shape[0]
        total_loss = 0
        similarities = []
        
        for i in range(batch_size):
            image = images[i]
            
            # Calculate image statistics
            image_stats = self._calculate_image_statistics(image)
            
            # Segment the image
            segmentation_result = self.segmentation.segment_image(image)
            
            # Forward pass through the network
            outputs = self.model(
                image.unsqueeze(0),
                image_stats,
                segmentation_result
            )
            
            # Get reference for this image
            reference_tensor, ref_key, ref_similarity = self._get_best_reference(
                image, segmentation_result
            )
            
            similarities.append(ref_similarity)
            
            # Calculate losses
            
            # 1. Similarity loss - ensure high similarity to reference
            similarity_target = torch.tensor([1.0]).to(self.device)  # Target high similarity
            similarity_loss = 1 - ref_similarity  # Loss is lower when similarity is higher
            
            # 2. Region consistency loss
            region_loss = self._calculate_region_consistency_loss(
                outputs, segmentation_result
            )
            
            # 3. Defect detection loss (if applicable)
            defect_loss = self._calculate_defect_loss(
                image, reference_tensor, outputs['defect_probability']
            )
            
            # Combined loss with weights
            loss = (0.4 * similarity_loss + 
                   0.4 * region_loss + 
                   0.2 * defect_loss)
            
            total_loss += loss
        
        # Average loss over batch
        avg_loss = total_loss / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Adjust custom weights based on loss
        # "small percentile adjustments to parameters"
        self.model.adjust_weights_based_on_loss(
            avg_loss, 
            self.config.LOSS_ADJUSTMENT_RATE
        )
        
        metrics = {
            'similarity': np.mean(similarities),
            'total_loss': avg_loss.item()
        }
        
        return avg_loss, metrics
    
    def _calculate_image_statistics(self, image: torch.Tensor) -> Dict[str, float]:
        """Calculate statistics needed for network weights"""
        # Use data loader's methods
        gradient_intensity = self.data_loader.calculate_gradient_intensity(image)
        avg_x, avg_y = self.data_loader.calculate_pixel_positions(image)
        
        return {
            'gradient_intensity': gradient_intensity,
            'center_x': avg_x,
            'center_y': avg_y
        }
    
    def _get_best_reference(self, image: torch.Tensor, 
                           segmentation: 'SegmentationResult') -> Tuple[torch.Tensor, str, float]:
        """Find best matching reference for the image"""
        # Get appropriate reference set based on dominant region
        region_sizes = {
            'core': segmentation.core_mask.sum().item(),
            'cladding': segmentation.cladding_mask.sum().item(),
            'ferrule': segmentation.ferrule_mask.sum().item()
        }
        dominant_region = max(region_sizes, key=region_sizes.get)
        
        # Get reference tensor and similarity
        ref_tensor, ref_key, similarity = self.data_loader.get_reference_by_similarity(
            image, dominant_region
        )
        
        return ref_tensor, ref_key, similarity
    
    def _calculate_region_consistency_loss(self, outputs: Dict[str, torch.Tensor],
                                         segmentation: 'SegmentationResult') -> torch.Tensor:
        """
        Ensure network outputs are consistent with segmentation.
        "when a feature of an image follows this line its classified as region"
        """
        # Extract features for each region
        core_features = outputs['core_features']
        cladding_features = outputs['cladding_features']
        ferrule_features = outputs['ferrule_features']
        
        # Calculate feature distinctiveness (regions should have distinct features)
        core_cladding_dist = torch.norm(core_features - cladding_features)
        core_ferrule_dist = torch.norm(core_features - ferrule_features)
        cladding_ferrule_dist = torch.norm(cladding_features - ferrule_features)
        
        # Loss is lower when regions are more distinct
        distinctiveness_loss = 1.0 / (core_cladding_dist + core_ferrule_dist + 
                                     cladding_ferrule_dist + 1e-8)
        
        # Ensure features align with segmentation confidence
        confidence_loss = 1.0 - segmentation.confidence_scores['overall']
        
        return distinctiveness_loss + confidence_loss
    
    def _calculate_defect_loss(self, image: torch.Tensor, 
                             reference: torch.Tensor,
                             defect_probability: torch.Tensor) -> torch.Tensor:
        """Calculate loss for defect detection accuracy"""
        # Detect actual anomalies
        anomaly_result = self.anomaly_detector.detect_anomalies(image, reference)
        
        # Ground truth: 1 if defects present, 0 otherwise
        has_defects = float(len(anomaly_result.defect_locations) > 0)
        ground_truth = torch.tensor([has_defects]).to(self.device)
        
        # Binary cross entropy loss
        defect_loss = nn.functional.binary_cross_entropy(
            defect_probability.squeeze(),
            ground_truth
        )
        
        return defect_loss
    
    def _validate(self) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
        val_metrics = {
            'avg_similarity': [],
            'defect_accuracy': [],
            'region_accuracy': []
        }
        
        with torch.no_grad():
            # Get validation batch
            val_batch, val_keys = self.data_loader.get_tensor_batch(
                batch_size=32,
                region=None
            )
            val_batch = val_batch.to(self.device)
            
            for i in range(val_batch.shape[0]):
                image = val_batch[i]
                
                # Process image
                image_stats = self._calculate_image_statistics(image)
                segmentation_result = self.segmentation.segment_image(image)
                
                # Get model outputs
                outputs = self.model(
                    image.unsqueeze(0),
                    image_stats,
                    segmentation_result
                )
                
                # Calculate metrics
                ref_tensor, _, similarity = self._get_best_reference(image, segmentation_result)
                val_metrics['avg_similarity'].append(similarity)
                
                # Defect detection accuracy
                anomaly_result = self.anomaly_detector.detect_anomalies(image, ref_tensor)
                predicted_defect = outputs['defect_probability'].item() > 0.5
                actual_defect = len(anomaly_result.defect_locations) > 0
                val_metrics['defect_accuracy'].append(float(predicted_defect == actual_defect))
        
        # Average metrics
        return {
            'avg_similarity': np.mean(val_metrics['avg_similarity']),
            'defect_accuracy': np.mean(val_metrics['defect_accuracy'])
        }
    
    def _create_training_dataset(self) -> FiberOpticsDataset:
        """Create dataset from loaded tensors"""
        # Collect all tensor paths
        all_paths = []
        
        for region_tensors in self.data_loader.reference_tensors.values():
            for key, tensor_or_path in region_tensors.items():
                if isinstance(tensor_or_path, Path):
                    all_paths.append(tensor_or_path)
        
        self.logger.info(f"Created training dataset with {len(all_paths)} samples")
        
        return FiberOpticsDataset(all_paths)
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"fiber_optics_nn_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history
        }, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint['epoch']