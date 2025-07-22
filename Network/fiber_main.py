#!/usr/bin/env python3
"""
Main entry point for Fiber Optics Neural Network System
Complete integrated system for fiber optic image analysis
"the overall program follows this equation I=Ax1+Bx2+Cx3... =S(R)"
"""

import torch
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import argparse  # Only for documentation, not used

# Import all modules
from config import get_config
from logger import get_logger
from tensor_processor import TensorProcessor
from feature_extractor import FeatureExtractionPipeline
from reference_comparator import ReferenceComparator
from anomaly_detector import ComprehensiveAnomalyDetector
from integrated_network import IntegratedAnalysisPipeline
from trainer import FiberOpticsTrainer
from data_loader import FiberOpticsDataLoader, ReferenceDataLoader

class FiberOpticsSystem:
    """
    Complete fiber optics analysis system
    "the program will use tensorized images as its reference data"
    """
    
    def __init__(self):
        print(f"[{datetime.now()}] Initializing FiberOpticsSystem")
        print(f"[{datetime.now()}] Previous script: data_loader.py")
        print(f"[{datetime.now()}] Starting complete integrated system")
        
        self.config = get_config()
        self.logger = get_logger("FiberOpticsSystem")
        
        self.logger.log_class_init("FiberOpticsSystem")
        self.logger.log_process_start("System Initialization")
        
        # Initialize all components
        self.tensor_processor = TensorProcessor()
        self.integrated_pipeline = IntegratedAnalysisPipeline()
        self.trainer = FiberOpticsTrainer(self.integrated_pipeline.network)
        self.data_loader = FiberOpticsDataLoader()
        self.reference_loader = ReferenceDataLoader()
        
        # System state
        self.is_trained = False
        self.training_history = {}
        
        self.logger.info("FiberOpticsSystem initialized successfully")
        self.logger.info(f"Device: {self.config.get_device()}")
        self.logger.info(f"Tensorized data path: {self.config.TENSORIZED_DATA_PATH}")
        
        self.logger.log_process_end("System Initialization")
        print(f"[{datetime.now()}] FiberOpticsSystem ready")
    
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
        output_path = self.config.RESULTS_PATH / f"{Path(image_path).stem}_results.txt"
        self.integrated_pipeline.export_results(results, str(output_path))
        
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
        
        self.config.update_equation_coefficient(coefficient, value)
        
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

def print_usage():
    """Print usage information (no argparse)"""
    print("\nFiber Optics Neural Network System")
    print("=" * 50)
    print("\nUsage:")
    print("  python main.py train [epochs]        - Train the model")
    print("  python main.py analyze <image>       - Analyze single image")
    print("  python main.py batch <folder>        - Batch process folder")
    print("  python main.py realtime              - Real-time processing")
    print("  python main.py evaluate              - Evaluate performance")
    print("  python main.py update <coef> <value> - Update coefficient")
    print("\nExample:")
    print("  python main.py train 50")
    print("  python main.py analyze sample.jpg")
    print("  python main.py update A 1.5")

def main():
    """
    Main entry point
    "IT WILL NOT USE ARGPARSE OR FLAGS I HATE FLAGS"
    """
    print(f"\n{'='*60}")
    print("FIBER OPTICS NEURAL NETWORK SYSTEM")
    print(f"{'='*60}\n")
    
    # Initialize system
    system = FiberOpticsSystem()
    
    # Parse command line (without argparse)
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "train":
        # Train model
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        checkpoint = sys.argv[3] if len(sys.argv) > 3 else None
        
        system.train_model(num_epochs=epochs, load_checkpoint=checkpoint)
        
    elif command == "analyze":
        # Analyze single image
        if len(sys.argv) < 3:
            print("Error: Please provide image path")
            print_usage()
            return
        
        image_path = sys.argv[2]
        results = system.analyze_single_image(image_path)
        
        # Print summary
        print(f"\nAnalysis Results for {image_path}:")
        print(f"  Final Similarity: {results['summary']['final_similarity_score']:.4f}")
        print(f"  Meets Threshold: {'Yes' if results['summary']['meets_threshold'] else 'No'}")
        print(f"  Primary Region: {results['summary']['primary_region']}")
        print(f"  Anomaly Score: {results['summary']['anomaly_score']:.4f}")
        
    elif command == "batch":
        # Batch process
        if len(sys.argv) < 3:
            print("Error: Please provide folder path")
            print_usage()
            return
        
        folder_path = sys.argv[2]
        max_images = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        results = system.batch_process(folder_path, max_images)
        
    elif command == "realtime":
        # Real-time processing
        system.realtime_process()
        
    elif command == "evaluate":
        # Evaluate performance
        # First load best model
        best_model_path = system.config.CHECKPOINTS_PATH / "best_model.pth"
        if best_model_path.exists():
            system.trainer.load_checkpoint(str(best_model_path))
            system.is_trained = True
        
        system.evaluate_performance()
        
    elif command == "update":
        # Update parameters
        if len(sys.argv) < 4:
            print("Error: Please provide coefficient and value")
            print_usage()
            return
        
        coefficient = sys.argv[2].upper()
        value = float(sys.argv[3])
        
        system.update_parameters(coefficient, value)
        
    else:
        print(f"Unknown command: {command}")
        print_usage()
    
    print(f"\n{'='*60}")
    print("PROCESS COMPLETED")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
