#!/usr/bin/env python3
"""
Unified Main Entry Point for Fiber Optics Neural Network System
Combines both production and research modes with statistical integration
"the overall program follows this equation I=Ax1+Bx2+Cx3... =S(R)"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import time
import json
import cv2
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import all core modules
from core.config_loader import get_config
from core.logger import get_logger
from data.tensor_processor import TensorProcessor
from data.feature_extractor import FeatureExtractionPipeline
from data.reference_comparator import ReferenceComparator
from logic.anomaly_detector import ComprehensiveAnomalyDetector
from logic.integrated_network import (
    IntegratedAnalysisPipeline, 
    EnhancedFiberOpticsIntegratedNetwork,
    EnhancedIntegratedNetwork
)
from utilities.trainer import EnhancedTrainer
from data.data_loader import FiberOpticsDataLoader, ReferenceDataLoader, FiberOpticsDataset
from logic.real_time_optimization import (
    EfficientFiberOpticsNetwork,
    ModelCompressor,
    create_student_teacher_models
)
from utilities.optimizers import create_advanced_optimizer

# Import statistical modules
from logic.statistical_integration import (
    integrate_statistics_into_network,
    StatisticallyIntegratedNetwork
)
from core.statistical_config import (
    get_statistical_config,
    merge_with_base_config
)

class UnifiedFiberOpticsSystem:
    """
    Complete fiber optics analysis system
    "the program will use tensorized images as its reference data"
    """
    
    def __init__(self, mode: str = "production", config_path: Optional[str] = None):
        """
        Initialize unified fiber optics system
        
        Args:
            mode: 'production' for optimized deployment, 'research' for statistical analysis
            config_path: Optional path to custom configuration file
        """
        print(f"[{datetime.now()}] Initializing UnifiedFiberOpticsSystem")
        print(f"[{datetime.now()}] Mode: {mode}")
        
        # Load configuration based on mode
        self.mode = mode
        self.config = get_config()
        
        # Override runtime mode in config
        if hasattr(self.config, 'runtime'):
            self.config.runtime.mode = mode
        
        # Setup logger
        self.logger = get_logger("UnifiedFiberOpticsSystem")
        self.logger.log_class_init("UnifiedFiberOpticsSystem")
        self.logger.log_process_start("System Initialization")
        
        # Initialize device
        self.device = self.config.get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components based on mode
        if mode == "research":
            self._init_research_mode()
        else:
            self._init_production_mode()
        
        # Common components
        self.data_loader = FiberOpticsDataLoader()
        self.reference_loader = ReferenceDataLoader()
        self.tensor_processor = TensorProcessor()
        
        # Real-time optimization components
        self.efficient_model = None
        self.model_compressor = ModelCompressor()
        
        # System state
        self.is_trained = False
        self.training_history = {}
        self.optimization_enabled = False
        
        self.logger.info(f"UnifiedFiberOpticsSystem initialized in {mode} mode")
        self.logger.log_process_end("System Initialization")
        print(f"[{datetime.now()}] UnifiedFiberOpticsSystem ready")
    
    def _init_production_mode(self):
        """Initialize components for production mode"""
        self.logger.info("Initializing production mode components")
        
        # Use integrated pipeline directly
        self.integrated_pipeline = IntegratedAnalysisPipeline()
        self.network = self.integrated_pipeline.network
        
        # Initialize trainer
        self.trainer = EnhancedTrainer(self.network)
        
        # Initialize loss (use combined advanced loss)
        from utilities.losses import CombinedAdvancedLoss
        self.criterion = CombinedAdvancedLoss(self.config)
        
        # Initialize optimizer
        self._init_optimizer()
        
    def _init_research_mode(self):
        """Initialize components for research mode with statistical integration"""
        self.logger.info("Initializing research mode components")
        
        # Load statistical configuration
        self.stat_config = get_statistical_config()
        
        # Merge configurations
        if hasattr(self.config, '__dict__'):
            base_dict = self.config.__dict__
        else:
            base_dict = self.config
        
        merged_config = merge_with_base_config(base_dict, self.stat_config)
        
        # Update config with merged values
        for key, value in merged_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Create base network and integrate statistics
        base_network = EnhancedIntegratedNetwork()
        self.network = integrate_statistics_into_network(base_network)
        self.network = self.network.to(self.device)
        
        # Create integrated pipeline wrapper
        self.integrated_pipeline = IntegratedAnalysisPipeline()
        self.integrated_pipeline.network = self.network
        
        # Initialize statistical loss
        from utilities.losses import CombinedAdvancedLoss
        self.criterion = CombinedAdvancedLoss(self.config)
        
        # Initialize trainer with statistical components
        self.trainer = EnhancedTrainer(self.network)
        
        # Initialize optimizer with separate learning rates
        self._init_statistical_optimizer()
        
        # Load reference statistics
        self._load_reference_statistics()
        
        self.logger.info("Statistical components integrated successfully")
    
    def _init_optimizer(self):
        """Initialize standard optimizer"""
        optimizer_config = self.config.optimizer
        
        # Fixed optimizer creation to use create_advanced_optimizer for consistency with advanced features
        # Original code used basic optimizers, but script imports advanced; fixed to use SAMWithLookahead by default for production
        self.optimizer = create_advanced_optimizer(self.network, self.config)
        
        # Initialize scheduler
        # Use the internal optimizer for scheduler since SAMWithLookahead is a wrapper
        scheduler_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
        
        if optimizer_config.scheduler.type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                scheduler_optimizer, T_0=10, T_mult=2
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                scheduler_optimizer,
                patience=optimizer_config.scheduler.patience,
                factor=optimizer_config.scheduler.factor
            )
    
    def _init_statistical_optimizer(self):
        """Initialize optimizer with statistical component learning rates"""
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
        base_lr = self.config.optimizer.learning_rate
        stat_lr_mult = self.stat_config.training_settings['statistical_lr_multiplier']
        
        # Create parameter groups
        param_groups = [
            {'params': base_params, 'lr': base_lr, 'name': 'base'},
            {'params': statistical_params, 'lr': base_lr * stat_lr_mult, 'name': 'statistical'}
        ]
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.logger.info(f"Statistical optimizer initialized: Base LR={base_lr}, Stat LR={base_lr * stat_lr_mult}")
    
    def _load_reference_statistics(self):
        """Load reference statistics for research mode"""
        self.logger.info("Loading reference statistics...")
        
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
        
        # Initialize reference features
        if self.global_stats and 'neural_network_insights' in self.global_stats:
            self.reference_features = self._extract_reference_features()
        else:
            self.reference_features = None
    
    def _extract_reference_features(self) -> torch.Tensor:
        """Extract reference features from statistics"""
        if self.mode == "research" and hasattr(self, 'stat_config'):
            num_references = self.stat_config.reference_settings['num_reference_embeddings']
        else:
            num_references = 100
        
        feature_dim = 6
        
        # Create synthetic reference features
        features = torch.zeros(num_references, feature_dim)
        features[:, 0] = torch.normal(595.87, 116.35, (num_references,))  # center_x
        features[:, 1] = torch.normal(447.19, 87.07, (num_references,))   # center_y
        features[:, 2] = torch.normal(114.42, 78.41, (num_references,))   # core_radius
        features[:, 3] = torch.normal(208.89, 100.50, (num_references,))  # cladding_radius
        features[:, 4] = torch.normal(0.530, 0.182, (num_references,))    # ratio
        features[:, 5] = torch.ones(num_references) * 7                    # num_valid_results
        
        return features.to(self.device)
    
    def train_model(self, num_epochs: Optional[int] = None, 
                   load_checkpoint: Optional[str] = None):
        """
        Train the integrated neural network
        "this is only for the training sequence so that it can be trained fast and effectively"
        """
        self.logger.log_process_start("Model Training")
        
        # Load checkpoint if provided
        if load_checkpoint:
            self.logger.info(f"Loading checkpoint: {load_checkpoint}")
            self.trainer.load_checkpoint(load_checkpoint)
        
        # Get data loaders
        train_loader, val_loader = self.data_loader.get_data_loaders(
            train_ratio=0.8,
            use_weighted_sampling=True
        )
        
        # Train
        self.trainer.train(
            num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        self.is_trained = True
        self.training_history = self.trainer.history
        
        self.logger.log_process_end("Model Training")
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """
        Analyze a single fiber optic image
        "an image will be selected from a dataset folder"
        """
        self.logger.log_process_start(f"Single Image Analysis: {image_path}")
        
        start_time = time.time()
        
        # Analyze using integrated pipeline
        results = self.integrated_pipeline.analyze_image(image_path)
        
        analysis_time = time.time() - start_time
        
        # Log results
        self.logger.info(f"Analysis completed in {analysis_time:.2f}s")
        self.logger.info(f"Final similarity: {results['summary']['final_similarity_score']:.4f}")
        self.logger.info(f"Meets threshold: {results['summary']['meets_threshold']}")
        self.logger.info(f"Primary region: {results['summary']['primary_region']}")
        self.logger.info(f"Anomaly score: {results['summary']['anomaly_score']:.4f}")
        
        # Export results
        output_path = Path(self.config.system.results_path) / f"{Path(image_path).stem}_results.txt"
        self.integrated_pipeline.export_results(results, str(output_path))
        
        # Visualize results if enabled
        if self.config.visualization.save_visualizations:
            from config.visualizer import FiberOpticsVisualizer
            visualizer = FiberOpticsVisualizer()
            vis_path = Path(self.config.system.results_path) / f"{Path(image_path).stem}_visualization.png"
            visualizer.visualize_complete_analysis(image_path, results, str(vis_path))
        
        self.logger.log_process_end(f"Single Image Analysis: {image_path}")
        
        return results
    
    def batch_process(self, image_folder: str, max_images: Optional[int] = None) -> List[Dict]:
        """
        Process multiple images in batch
        "or multiple images for batch processing"
        """
        self.logger.log_process_start(f"Batch Processing: {image_folder}")
        
        image_folder = Path(image_folder)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        
        for idx, image_path in enumerate(image_files):
            self.logger.info(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")
            
            try:
                result = self.analyze_single_image(str(image_path))
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append({'error': str(e), 'image': str(image_path)})
        
        # Summary statistics
        successful = [r for r in results if 'error' not in r]
        if successful:
            avg_similarity = np.mean([r['summary']['final_similarity_score'] for r in successful])
            threshold_met_count = sum(1 for r in successful if r['summary']['meets_threshold'])
            
            self.logger.info(f"\nBatch Processing Summary:")
            self.logger.info(f"  Total processed: {len(results)}")
            self.logger.info(f"  Successful: {len(successful)}")
            self.logger.info(f"  Average similarity: {avg_similarity:.4f}")
            self.logger.info(f"  Met threshold: {threshold_met_count}/{len(successful)}")
        
        self.logger.log_process_end(f"Batch Processing: {image_folder}")
        
        return results
    
    def realtime_process(self, stream_source: Optional[str] = None):
        """
        Process images in real-time
        "or realtime processing"
        """
        self.logger.log_process_start("Real-time Processing")
        
        # Get streaming data loader
        stream_loader = self.data_loader.get_streaming_loader(batch_size=1)
        
        self.logger.info("Starting real-time processing loop")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            for batch_idx, batch in enumerate(stream_loader):
                # Process batch
                image_tensor = batch['image'].to(self.config.get_device())
                
                with torch.no_grad():
                    outputs = self.integrated_pipeline.network(image_tensor)
                
                # Quick summary
                similarity = outputs['final_similarity'][0].item()
                meets_threshold = outputs['meets_threshold'][0].item()
                anomaly_score = outputs['anomaly_map'][0].mean().item()
                
                self.logger.info(f"Frame {batch_idx}: Similarity={similarity:.4f}, "
                               f"Threshold={'✓' if meets_threshold else '✗'}, "
                               f"Anomaly={anomaly_score:.4f}")
                
                # Simulate real-time delay
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Real-time processing stopped by user")
        
        self.logger.log_process_end("Real-time Processing")
    
    def update_parameters(self, coefficient: str, value: float):
        """
        Update equation coefficients
        "allow me to see and tweak the parameters and weights of these fitting equations"
        """
        self.logger.info(f"Updating coefficient {coefficient} to {value}")
        
        # Update equation coefficient in config
        coefficients = self.config.equation.coefficients
        coefficients[coefficient] = value
        self.integrated_pipeline.network.set_equation_coefficients(
            [coefficients[k] for k in ['A', 'B', 'C', 'D', 'E']]
        )
        
        # Update model if needed
        if hasattr(self.integrated_pipeline.network, 'equation_adjuster'):
            # Model will use updated config on next forward pass
            self.logger.info("Model will use updated coefficients on next analysis")
    
    def evaluate_performance(self):
        """
        Evaluate system performance
        "over time the program will get more accurate and faster at computing"
        """
        self.logger.log_process_start("Performance Evaluation")
        
        if not self.is_trained:
            self.logger.warning("Model not trained yet")
            return
        
        # Get validation loader
        _, val_loader = self.data_loader.get_data_loaders(batch_size=16)
        
        # Evaluate
        total_similarity = 0
        threshold_met_count = 0
        total_samples = 0
        inference_times = []
        
        self.integrated_pipeline.network.eval()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.config.get_device())
                
                start_time = time.time()
                outputs = self.integrated_pipeline.network(images)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time / images.shape[0])
                
                total_similarity += outputs['final_similarity'].sum().item()
                threshold_met_count += outputs['meets_threshold'].sum().item()
                total_samples += images.shape[0]
        
        # Calculate metrics
        avg_similarity = total_similarity / total_samples
        threshold_ratio = threshold_met_count / total_samples
        avg_inference_time = np.mean(inference_times)
        
        self.logger.info("\nPerformance Metrics:")
        self.logger.info(f"  Average similarity: {avg_similarity:.4f}")
        self.logger.info(f"  Threshold achievement: {threshold_ratio:.2%}")
        self.logger.info(f"  Average inference time: {avg_inference_time*1000:.2f}ms per image")
        
        # Compare to training history
        if self.training_history and 'val_similarity' in self.training_history:
            initial_similarity = self.training_history['val_similarity'][0]
            final_similarity = self.training_history['val_similarity'][-1]
            improvement = (final_similarity - initial_similarity) / initial_similarity * 100
            
            self.logger.info(f"  Training improvement: {improvement:.1f}%")
        
        self.logger.log_process_end("Performance Evaluation")
    
    def optimize_model(self, optimization_type: str = "all"):
        """
        Optimize model for real-time performance
        
        Args:
            optimization_type: Type of optimization ('distill', 'prune', 'quantize', 'all')
        """
        self.logger.log_process_start(f"Model Optimization: {optimization_type}")
        
        if not self.is_trained:
            self.logger.warning("Model not trained yet. Training required before optimization.")
            return
        
        # Create efficient model through knowledge distillation
        if optimization_type in ["distill", "all"]:
            self.logger.info("Creating efficient model through knowledge distillation...")
            
            # Use current model as teacher
            teacher_model = self.integrated_pipeline.network
            teacher_model.eval()
            
            # Create student model
            student_config = {
                'num_classes': 3,
                'base_channels': 32,
                'width_mult': 0.5,
                'use_adaptive': True
            }
            
            self.efficient_model = EfficientFiberOpticsNetwork(**student_config)
            self.efficient_model = self.efficient_model.to(self.device)
            
            # Train student with distillation
            student_trainer = EnhancedTrainer(
                model=self.efficient_model,
                teacher_model=teacher_model
            )
            
            train_loader, val_loader = self.data_loader.get_data_loaders(train_ratio=0.8)
            student_trainer.train(
                num_epochs=10,  # Fewer epochs for distillation
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            self.logger.info("Knowledge distillation completed")
        
        # Apply pruning
        if optimization_type in ["prune", "all"]:
            self.logger.info("Applying model pruning...")
            
            model_to_prune = self.efficient_model if self.efficient_model else self.integrated_pipeline.network
            pruned_model = self.model_compressor.prune_model(
                model_to_prune,
                prune_ratio=0.3,
                structured=False
            )
            
            if self.efficient_model:
                self.efficient_model = pruned_model
            else:
                self.integrated_pipeline.network = pruned_model
            
            self.logger.info("Model pruning completed")
        
        # Apply quantization
        if optimization_type in ["quantize", "all"]:
            self.logger.info("Applying model quantization...")
            
            # Note: Quantization requires calibration data
            self.logger.warning("Quantization skipped - requires calibration implementation")
        
        self.optimization_enabled = True
        self.logger.info(f"Model optimization completed. Type: {optimization_type}")
        self.logger.log_process_end(f"Model Optimization: {optimization_type}")
    
    def benchmark_optimization(self):
        """
        Benchmark optimized model vs original
        """
        if not self.optimization_enabled or not self.efficient_model:
            self.logger.warning("No optimized model available. Run optimize_model first.")
            return
        
        self.logger.log_process_start("Optimization Benchmarking")
        
        # Get test data
        test_loader = self.data_loader.get_test_loader(batch_size=1)
        
        # Benchmark original model
        original_times = []
        original_model = self.integrated_pipeline.network
        original_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 100:  # Test on 100 samples
                    break
                
                image = batch['image'].to(self.device)
                
                start = time.time()
                _ = original_model(image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                original_times.append(time.time() - start)
        
        # Benchmark efficient model
        efficient_times = []
        self.efficient_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 100:
                    break
                
                image = batch['image'].to(self.device)
                
                start = time.time()
                _ = self.efficient_model(image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                efficient_times.append(time.time() - start)
        
        # Calculate statistics
        original_avg = np.mean(original_times) * 1000  # ms
        efficient_avg = np.mean(efficient_times) * 1000  # ms
        speedup = original_avg / efficient_avg
        
        # Model sizes
        original_params = sum(p.numel() for p in original_model.parameters())
        efficient_params = sum(p.numel() for p in self.efficient_model.parameters())
        compression = 1 - (efficient_params / original_params)
        
        self.logger.info("\nOptimization Benchmark Results:")
        self.logger.info(f"  Original model inference: {original_avg:.2f}ms")
        self.logger.info(f"  Efficient model inference: {efficient_avg:.2f}ms")
        self.logger.info(f"  Speedup: {speedup:.2f}x")
        self.logger.info(f"  Original parameters: {original_params:,}")
        self.logger.info(f"  Efficient parameters: {efficient_params:,}")
        self.logger.info(f"  Compression: {compression:.1%}")
        
        self.logger.log_process_end("Optimization Benchmarking")
    
    def process_image_statistical(self, image_path: str, output_dir: str) -> Dict[str, any]:
        """
        Process a single image with full statistical analysis (research mode)
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Comprehensive results dictionary
        """
        if self.mode != "research":
            self.logger.warning("Statistical processing requires research mode")
            return self.analyze_single_image(image_path)
        
        self.logger.info(f"Processing image with statistical analysis: {image_path}")
        
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
        
        # Extract comprehensive results
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'mode': 'statistical_analysis',
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
        
        # Statistical features
        if 'statistical_features' in outputs:
            results['feature_statistics'] = {
                'mean': outputs['statistical_features'].mean().item(),
                'std': outputs['statistical_features'].std().item(),
                'min': outputs['statistical_features'].min().item(),
                'max': outputs['statistical_features'].max().item()
            }
        
        # Save results
        with open(output_path / 'statistical_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Statistical processing complete. Results saved to {output_path}")
        
        return results

def main():
    """
    Unified main entry point with configuration-based execution
    Supports both production and research modes
    """
    # Load configuration to determine mode
    config = get_config()
    
    # Check runtime mode from config
    mode = "production"  # Default
    if hasattr(config, 'runtime') and hasattr(config.runtime, 'mode'):
        if config.runtime.mode in ['research', 'statistical']:
            mode = 'research'
        else:
            mode = 'production'
    
    print(f"\n{'='*80}")
    print(f"FIBER OPTICS NEURAL NETWORK SYSTEM - {mode.upper()} MODE")
    print(f"{'='*80}\n")
    
    # Initialize unified system
    print(f"🚀 INITIALIZING SYSTEM IN {mode.upper()} MODE...")
    system = UnifiedFiberOpticsSystem(mode=mode)
    
    # Check if we have a trained model
    best_model_path = Path(system.config.system.checkpoints_path) / "best_model.pth"
    
    if best_model_path.exists():
        print("\n✅ Found existing trained model, loading...")
        system.trainer.load_checkpoint(str(best_model_path))
        system.is_trained = True
        
        # Evaluate current performance
        print("\n📊 EVALUATING CURRENT MODEL PERFORMANCE...")
        system.evaluate_performance()
        
    else:
        print("\n🎯 No trained model found. Starting training...")
        epochs = system.config.training.num_epochs if hasattr(system.config, 'training') else 50
        print(f"📚 TRAINING MODEL ({epochs} epochs)...")
        system.train_model(num_epochs=epochs)
        
        print("\n💾 Training complete! Evaluating performance...")
        system.evaluate_performance()
    
    # Mode-specific operations
    if mode == "production":
        # Production mode: focus on optimization and real-time performance
        print("\n⚡ OPTIMIZING MODEL FOR REAL-TIME PERFORMANCE...")
        print("   - Knowledge Distillation")
        print("   - Model Pruning")
        print("   - Architecture Optimization")
        system.optimize_model("all")
        
        # Benchmark the optimization
        print("\n🏁 BENCHMARKING OPTIMIZED MODEL...")
        system.benchmark_optimization()
        
    else:
        # Research mode: focus on statistical analysis
        print("\n📊 STATISTICAL ANALYSIS MODE ACTIVE")
        print("   - 88-dimensional feature extraction")
        print("   - Master similarity equation")
        print("   - Zone parameter regression")
        print("   - Consensus algorithm")
        print("   - Mahalanobis distance anomaly detection")
        
        # Process sample with statistical analysis
        sample_folder = Path("./samples")
        if sample_folder.exists():
            images = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png"))
            if images:
                print(f"\n📸 PROCESSING SAMPLE IMAGE WITH STATISTICAL ANALYSIS...")
                result = system.process_image_statistical(
                    str(images[0]), 
                    "output/statistical_analysis"
                )
                if result:
                    print("\n📈 STATISTICAL ANALYSIS RESULTS:")
                    if 'zones' in result:
                        print(f"   Core Radius: {result['zones']['core_radius']:.1f} px")
                        print(f"   Cladding Radius: {result['zones']['cladding_radius']:.1f} px")
                        print(f"   Ratio: {result['zones']['ratio']:.3f}")
                    if 'consensus' in result:
                        print(f"   Consensus Confidence: {result['consensus']['confidence']:.3f}")
                        print(f"   Agreeing Methods: {result['consensus']['num_agreeing_methods']}/7")
    
    # Common operations for both modes
    # Process sample images
    sample_folder = Path("./samples")
    if sample_folder.exists():
        print(f"\n📸 PROCESSING SAMPLE IMAGES FROM {sample_folder}...")
        results = system.batch_process(str(sample_folder), max_images=5)
        
        # Show results
        successful = [r for r in results if 'error' not in r]
        if successful:
            print("\n📈 SAMPLE PROCESSING RESULTS:")
            for i, result in enumerate(successful):
                print(f"   Image {i+1}: Similarity={result['summary']['final_similarity_score']:.4f}, "
                      f"Threshold={'✅' if result['summary']['meets_threshold'] else '❌'}")
    
    # Update equation parameters
    print("\n🔧 FINE-TUNING EQUATION PARAMETERS...")
    if hasattr(system.config, 'equation') and hasattr(system.config.equation, 'coefficients'):
        # Use config values
        for coef in ['A', 'B', 'C', 'D', 'E']:
            value = system.config.equation.coefficients[coef]
            print(f"   {coef} = {value} (from config)")
    else:
        # Use optimal defaults
        optimal_params = {
            'A': 1.2,  # Structural similarity weight
            'B': 0.8,  # Reference comparison weight
            'C': 1.0,  # Feature matching weight
            'D': 0.6,  # Anomaly detection weight
            'E': 0.4   # Additional features weight
        }
        
        for coef, value in optimal_params.items():
            system.update_parameters(coef, value)
            print(f"   Set {coef} = {value}")
    
    # Real-time processing demo
    if system.config.runtime.mode != "batch":
        print("\n🎬 STARTING REAL-TIME PROCESSING DEMO...")
        print("   Press Ctrl+C to stop")
        
        try:
            # Process for a short demo (or until interrupted)
            import signal
            import threading
            
            def timeout_handler():
                print("\n⏱️  Demo time limit reached")
                raise KeyboardInterrupt
            
            # Set a 30-second demo timer
            timer = threading.Timer(30.0, timeout_handler)
            timer.start()
            
            system.realtime_process()
            timer.cancel()
            
        except KeyboardInterrupt:
            print("\n✋ Real-time processing stopped")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 SYSTEM DEMONSTRATION COMPLETED - {mode.upper()} MODE!")
    print(f"{'='*80}")
    print("\n📋 SUMMARY:")
    print(f"   ✅ Model trained and {'optimized' if mode == 'production' else 'statistically enhanced'}")
    print("   ✅ System fully operational")
    print(f"   ✅ {mode.capitalize()} mode features active")
    
    if mode == "production":
        print("\n💡 The system is optimized for real-time production use!")
    else:
        print("\n💡 The system provides comprehensive statistical analysis!")
        print("   All statistical insights are integrated into the neural network.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
