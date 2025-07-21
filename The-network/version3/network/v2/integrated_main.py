#!/usr/bin/env python3
"""
Main script for the integrated fiber optics neural network.
This version uses the neural network to perform all operations internally.
"""

import torch
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from fiber_optics_nn.utils.logger import get_logger
from fiber_optics_nn.config.config import get_config
from fiber_optics_nn.models.integrated_fiber_nn import IntegratedFiberOpticsNN
from fiber_optics_nn.trainers.integrated_trainer import IntegratedTrainer
from fiber_optics_nn.core.integrated_pipeline import IntegratedPipeline
from fiber_optics_nn.data_loaders.tensor_loader import TensorDataLoader
from fiber_optics_nn.processors.image_processor import ImageProcessor


def main():
    """Main entry point with mode selection"""
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("INTEGRATED FIBER OPTICS NEURAL NETWORK")
    logger.info("Neural network performs segmentation, comparison, and anomaly detection internally")
    logger.info("=" * 80)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "train":
            train_integrated_model()
        elif mode == "inference":
            run_inference()
        elif mode == "demo":
            run_demo()
        elif mode == "adjust":
            adjust_parameters_interactive()
        else:
            logger.error(f"Unknown mode: {mode}")
            print_usage()
    else:
        print_usage()


def train_integrated_model():
    """
    Train the integrated neural network.
    "the neural network does the segmentation and reference comparison and 
    anomaly detection internally"
    """
    logger = get_logger()
    config = get_config()
    
    logger.info("Starting integrated model training...")
    
    # Create model
    model = IntegratedFiberOpticsNN()
    
    # Log initial equation parameters
    params = model.get_equation_parameters()
    logger.info(f"Initial equation parameters (I=Ax1+Bx2+...): {params}")
    
    # Create trainer
    trainer = IntegratedTrainer(model)
    
    # Train model
    num_epochs = None
    if len(sys.argv) > 2:
        try:
            num_epochs = int(sys.argv[2])
        except ValueError:
            pass
    
    trainer.train(num_epochs=num_epochs)
    
    # Save final model
    save_integrated_model(model, trainer)


def run_inference():
    """
    Run inference using the trained integrated model.
    "an image will be selected from a dataset folder(or multiple images for 
    batch processing or realtime processing)"
    """
    logger = get_logger()
    
    logger.info("Starting inference mode...")
    
    # Find best checkpoint
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        logger.error("No trained model found. Please train the model first.")
        return
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(checkpoint_path)
    
    # Process test image or batch
    if len(sys.argv) > 2:
        image_path = Path(sys.argv[2])
        if image_path.exists():
            process_single_image_integrated(pipeline, image_path)
        else:
            logger.error(f"Image not found: {image_path}")
    else:
        # Process example batch
        process_example_batch_integrated(pipeline)


def process_single_image_integrated(pipeline: IntegratedPipeline, image_path: Path):
    """
    Process a single image through the integrated pipeline.
    "the image will then be tensorized so it can better be computed by pytorch"
    """
    logger = get_logger()
    logger.info(f"Processing image: {image_path}")
    
    # Load and tensorize image
    image_processor = ImageProcessor()
    image_tensor = image_processor.process_image(image_path)
    
    # Process through integrated pipeline
    results = pipeline.process_image(image_tensor)
    
    # Display results
    display_integrated_results(results, image_path)
    
    # Export results
    output_path = pipeline.export_results(image_path.stem, results, image_tensor)
    logger.info(f"Results exported to: {output_path}")


def process_example_batch_integrated(pipeline: IntegratedPipeline):
    """Process a batch of example images"""
    logger = get_logger()
    
    # Load some example tensors
    data_loader = TensorDataLoader()
    data_loader.load_all_references(preload=False)
    
    # Get a batch
    batch_tensors, batch_keys = data_loader.get_tensor_batch(batch_size=5)
    
    logger.info(f"Processing batch of {len(batch_keys)} images...")
    
    # Process batch
    batch_results = pipeline.process_batch(batch_tensors)
    
    # Display summary
    for i, (key, results) in enumerate(zip(batch_keys, batch_results)):
        logger.info(f"\nImage {i+1}/{len(batch_keys)}: {key}")
        logger.info(f"  Dominant region: {results['segmentation']['dominant_region']}")
        logger.info(f"  Similarity score: {results['reference_matching']['similarity_score']:.4f}")
        logger.info(f"  Meets threshold: {results['reference_matching']['meets_threshold']}")
        logger.info(f"  Anomalies found: {results['anomaly_detection']['num_anomalies']}")
        
        # Export if anomalies found
        if results['anomaly_detection']['num_anomalies'] > 0:
            output_path = pipeline.export_results(f"batch_{key}", results, batch_tensors[i])
            logger.info(f"  Exported to: {output_path}")


def display_integrated_results(results: Dict, image_path: Path):
    """Display comprehensive results"""
    logger = get_logger()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS FOR: {image_path}")
    logger.info("=" * 60)
    
    # Segmentation results
    logger.info("\nSEGMENTATION:")
    logger.info(f"  Dominant region: {results['segmentation']['dominant_region']}")
    for region, prob in zip(['core', 'cladding', 'ferrule'], 
                           results['segmentation']['probabilities'].mean(axis=(1,2))):
        logger.info(f"  {region}: {prob:.1%}")
    
    # Reference matching
    logger.info("\nREFERENCE MATCHING:")
    logger.info(f"  Best reference: #{results['reference_matching']['best_reference_idx']}")
    logger.info(f"  Similarity score: {results['reference_matching']['similarity_score']:.4f}")
    logger.info(f"  Meets threshold (>{get_config().SIMILARITY_THRESHOLD}): "
                f"{'YES' if results['reference_matching']['meets_threshold'] else 'NO'}")
    
    # Anomaly detection
    logger.info("\nANOMALY DETECTION:")
    logger.info(f"  Anomalies found: {results['anomaly_detection']['num_anomalies']}")
    logger.info(f"  Max anomaly score: {results['anomaly_detection']['max_anomaly_score']:.4f}")
    logger.info(f"  Mean anomaly score: {results['anomaly_detection']['mean_anomaly_score']:.4f}")
    
    # Low similarity analysis if applicable
    if 'anomaly_analysis' in results:
        logger.info("\nLOW SIMILARITY ANALYSIS:")
        analysis = results['anomaly_analysis']
        logger.info(f"  Most affected region: {analysis['most_affected_region']}")
        logger.info(f"  Anomaly locations: {analysis['anomaly_locations']['count']} detected")
        logger.info(f"  Likely cause: {analysis['similarity_breakdown']['likely_cause']}")
    
    # Equation parameters
    logger.info("\nEQUATION PARAMETERS (I=Ax1+Bx2+...):")
    params = results['equation_parameters']
    logger.info(f"  A (gradient): {params['A_gradient']:.4f}")
    logger.info(f"  B (position): {params['B_position']:.4f}")
    
    logger.info("=" * 60 + "\n")


def run_demo():
    """
    Run a comprehensive demo showing all capabilities.
    "the neural network itself in order to classify the image will separate 
    the image into its features (or edges)"
    """
    logger = get_logger()
    
    logger.info("Running integrated neural network demo...")
    
    # Create model
    model = IntegratedFiberOpticsNN()
    
    # Create synthetic test image
    test_image = create_synthetic_fiber_image()
    
    # Show how the network processes internally
    logger.info("\nDemonstrating internal processing:")
    
    with torch.no_grad():
        # Get intermediate outputs
        outputs = model(test_image)
        
        # Show feature extraction
        logger.info("\n1. FEATURE EXTRACTION:")
        for i, features in enumerate(outputs['all_features']):
            logger.info(f"   Layer {i+1}: Extracted {features.shape[1]} feature maps "
                       f"of size {features.shape[2]}x{features.shape[3]}")
        
        # Show segmentation
        logger.info("\n2. INTERNAL SEGMENTATION:")
        segmentation = outputs['segmentation']
        logger.info(f"   Network identified {segmentation.shape[1]} regions")
        logger.info(f"   Region predictions shape: {segmentation.shape}")
        
        # Show comparison
        logger.info("\n3. REFERENCE COMPARISON:")
        logger.info(f"   Best matching reference: #{outputs['best_reference_idx'][0]}")
        logger.info(f"   Similarity score: {outputs['best_similarity'][0]:.4f}")
        
        # Show anomaly detection
        logger.info("\n4. ANOMALY DETECTION:")
        logger.info(f"   Reconstruction vs Original difference computed")
        logger.info(f"   Anomaly map shape: {outputs['anomaly_map'].shape}")
        logger.info(f"   Max anomaly score: {outputs['anomaly_map'].max():.4f}")


def create_synthetic_fiber_image() -> torch.Tensor:
    """Create a synthetic fiber optic image for testing"""
    import numpy as np
    
    # Create circular patterns resembling fiber optic image
    size = 256
    center = size // 2
    
    # Create radial distance map
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Create three regions
    core = dist < 20
    cladding = (dist >= 20) & (dist < 60)
    ferrule = dist >= 60
    
    # Create image with different intensities
    image = np.zeros((size, size, 3))
    image[core] = [0.9, 0.9, 0.9]  # Bright core
    image[cladding] = [0.5, 0.5, 0.5]  # Medium cladding
    image[ferrule] = [0.2, 0.2, 0.2]  # Dark ferrule
    
    # Add some noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Add synthetic defect
    defect_x, defect_y = 100, 100
    image[defect_y:defect_y+20, defect_x:defect_x+5] = [1.0, 0.2, 0.2]  # Red scratch
    
    # Convert to tensor
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def adjust_parameters_interactive():
    """
    Interactive parameter adjustment.
    "the program will allow me to see and tweak the parameters and weights 
    of these fitting equations"
    """
    logger = get_logger()
    
    # Load model
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        logger.error("No trained model found")
        return
    
    model = IntegratedFiberOpticsNN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER ADJUSTMENT INTERFACE")
    logger.info("=" * 60)
    
    # Display current parameters
    params = model.get_equation_parameters()
    logger.info("\nCurrent equation parameters (I=Ax1+Bx2+...):")
    logger.info(f"  A (gradient influence): {params['A_gradient']:.4f}")
    logger.info(f"  B (position influence): {params['B_position']:.4f}")
    
    logger.info("\nGradient trend parameters (per region):")
    for i, region in enumerate(['core', 'cladding', 'ferrule']):
        logger.info(f"  {region}: slope={params['gradient_trends'][i,0]:.4f}, "
                   f"intercept={params['gradient_trends'][i,1]:.4f}")
    
    # Simple adjustment interface
    logger.info("\nTo adjust parameters, use:")
    logger.info("  python integrated_main.py train  # Re-train with new config")
    logger.info("  Or modify fiber_optics_nn/config/config.py")
    
    # Example adjustment
    logger.info("\nExample: Adjusting gradient influence by 10%...")
    model.adjust_parameters(gradient_factor=1.1)
    new_params = model.get_equation_parameters()
    logger.info(f"  New A (gradient influence): {new_params['A_gradient']:.4f}")


def save_integrated_model(model: IntegratedFiberOpticsNN, trainer: IntegratedTrainer):
    """Save the trained integrated model"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    final_path = checkpoint_dir / "integrated_model_final.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': trainer.history,
        'equation_parameters': model.get_equation_parameters(),
        'config': trainer.config.__dict__
    }, final_path)
    
    trainer.logger.info(f"Saved final integrated model to {final_path}")


def find_best_checkpoint() -> Optional[Path]:
    """Find the best available checkpoint"""
    checkpoint_dir = Path("checkpoints")
    
    # Look for final model first
    final_path = checkpoint_dir / "integrated_model_final.pth"
    if final_path.exists():
        return final_path
    
    # Otherwise find latest epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("integrated_model_epoch_*.pth"))
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    
    return None


def print_usage():
    """Print usage information"""
    print("\nIntegrated Fiber Optics Neural Network")
    print("=" * 40)
    print("\nUsage:")
    print("  python integrated_main.py train [epochs]     # Train the integrated model")
    print("  python integrated_main.py inference [image]  # Run inference")
    print("  python integrated_main.py demo              # Run demo")
    print("  python integrated_main.py adjust            # Adjust parameters")
    print("\nThe neural network performs segmentation, reference comparison,")
    print("and anomaly detection internally through its layers.")


if __name__ == "__main__":
    main()