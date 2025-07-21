#!/usr/bin/env python3
"""
Advanced integrated neural network that performs simultaneous feature classification,
anomaly detection, and reconstruction all within the network.
"""

import torch
from pathlib import Path
import sys
import numpy as np
import time

sys.path.append(str(Path(__file__).parent))

from fiber_optics_nn.utils.logger import get_logger
from fiber_optics_nn.config.config import get_config
from fiber_optics_nn.models.advanced_integrated_nn import AdvancedIntegratedFiberOpticsNN
from fiber_optics_nn.trainers.advanced_integrated_trainer import AdvancedIntegratedTrainer
from fiber_optics_nn.processors.image_processor import ImageProcessor
from fiber_optics_nn.utils.results_exporter import ResultsExporter
from fiber_optics_nn.data_loaders.tensor_loader import TensorDataLoader


def main():
    """Main entry point for advanced integrated system"""
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("ADVANCED INTEGRATED FIBER OPTICS NEURAL NETWORK")
    logger.info("Simultaneous feature classification and anomaly detection")
    logger.info("=" * 80)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "train":
            train_advanced_model()
        elif mode == "analyze":
            analyze_image_detailed()
        elif mode == "demo":
            run_advanced_demo()
        elif mode == "compare":
            compare_processing_methods()
        else:
            logger.error(f"Unknown mode: {mode}")
            print_advanced_usage()
    else:
        print_advanced_usage()


def train_advanced_model():
    """
    Train the advanced integrated model.
    "I want each feature to not only look for comparisons but also look for 
    anomalies while comparing"
    """
    logger = get_logger()
    logger.info("Starting advanced integrated model training...")
    
    # Create model
    model = AdvancedIntegratedFiberOpticsNN()
    
    # Log model architecture details
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} trainable parameters")
    
    # Log initial equation parameters
    params = model.get_detailed_equation_parameters()
    logger.info("\nInitial equation parameters (I=Ax1+Bx2+...):")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            logger.info(f"  {key}: shape={value.shape}, mean={value.mean():.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Create trainer
    trainer = AdvancedIntegratedTrainer(model)
    
    # Train
    num_epochs = None
    if len(sys.argv) > 2:
        try:
            num_epochs = int(sys.argv[2])
        except ValueError:
            pass
    
    trainer.train(num_epochs=num_epochs)
    
    # Save final model
    save_advanced_model(model, trainer)


def analyze_image_detailed():
    """
    Perform detailed analysis of an image showing simultaneous processing.
    "really fully analyze everything that I've been saying"
    """
    logger = get_logger()
    config = get_config()
    
    # Load model
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        logger.error("No trained model found")
        return
    
    model = AdvancedIntegratedFiberOpticsNN()
    checkpoint = torch.load(checkpoint_path, map_location=config.get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(config.get_device())
    
    # Get image to analyze
    if len(sys.argv) > 2:
        image_path = Path(sys.argv[2])
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return
    else:
        # Create synthetic image for demo
        logger.info("Using synthetic test image...")
        image_tensor = create_detailed_test_image()
        analyze_synthetic = True
    
    if 'image_path' in locals():
        # Load real image
        processor = ImageProcessor()
        image_tensor = processor.process_image(image_path).unsqueeze(0)
        analyze_synthetic = False
    
    # Perform detailed analysis
    logger.info("\n" + "="*60)
    logger.info("DETAILED SIMULTANEOUS ANALYSIS")
    logger.info("="*60)
    
    with torch.no_grad():
        outputs = model(image_tensor.to(config.get_device()))
    
    # 1. Multi-scale feature analysis
    logger.info("\n1. MULTI-SCALE FEATURE EXTRACTION AND ANALYSIS:")
    for i, (features, analysis) in enumerate(zip(outputs['all_features'], 
                                                outputs['all_analysis'])):
        logger.info(f"\n  Scale {i+1}:")
        logger.info(f"    Feature maps: {features.shape[1]} channels, "
                   f"{features.shape[2]}x{features.shape[3]} spatial")
        
        # Region classification at this scale
        region_probs = torch.softmax(analysis['region_logits'], dim=1).mean(dim=[2,3])
        logger.info(f"    Region probabilities: Core={region_probs[0,0]:.2%}, "
                   f"Cladding={region_probs[0,1]:.2%}, Ferrule={region_probs[0,2]:.2%}")
        
        # Anomaly detection at this scale
        anomaly_score = analysis['anomaly_scores'].mean().item()
        logger.info(f"    Anomaly score: {anomaly_score:.4f}")
        
        # Quality assessment
        quality_score = analysis['quality_scores'].mean().item()
        logger.info(f"    Feature quality: {quality_score:.4f}")
        
        # Pattern matching details
        normal_sims = analysis['normal_similarities'].mean(dim=[2,3,0])  # Average over spatial and batch
        logger.info(f"    Normal pattern matches: {normal_sims.shape[0]} patterns, "
                   f"avg similarity={normal_sims.mean():.4f}")
        
        anomaly_matches = analysis['anomaly_matches'].mean(dim=[2,3,0])
        if anomaly_matches.max() > 0.3:
            logger.info(f"    ⚠️  Anomaly patterns detected! Max match={anomaly_matches.max():.4f}")
    
    # 2. Final integrated results
    logger.info("\n2. FINAL INTEGRATED RESULTS:")
    
    # Segmentation
    segmentation = outputs['segmentation']
    dominant_region = ['Core', 'Cladding', 'Ferrule'][segmentation[0].argmax(dim=0)[128, 128]]
    logger.info(f"  Dominant region: {dominant_region}")
    
    # Reference matching
    logger.info(f"  Best reference match: #{outputs['best_reference_idx'][0]}")
    logger.info(f"  Similarity score: {outputs['similarity'][0]:.4f}")
    logger.info(f"  Meets threshold (>0.7): {'✓' if outputs['meets_threshold'][0] else '✗'}")
    
    # Anomaly summary
    anomaly_map = outputs['anomaly_map']
    logger.info(f"  Overall anomaly score: {anomaly_map.mean():.4f}")
    logger.info(f"  Max anomaly location: {torch.where(anomaly_map == anomaly_map.max())}")
    
    # 3. Detailed anomaly analysis
    logger.info("\n3. DETAILED ANOMALY BREAKDOWN:")
    anomaly_details = outputs['anomaly_details']
    
    for i, (scale_score, patterns, deviations) in enumerate(zip(
        anomaly_details['scale_anomalies'],
        anomaly_details['anomaly_patterns_matched'],
        anomaly_details['normal_pattern_deviations']
    )):
        logger.info(f"  Scale {i+1}:")
        logger.info(f"    Anomaly score: {scale_score:.4f}")
        logger.info(f"    Top anomaly patterns: {patterns[0][:3]}")
        logger.info(f"    Avg deviation from normal: {np.mean(deviations):.4f}")
    
    # 4. Trend analysis
    logger.info("\n4. TREND ANALYSIS:")
    trend_adherence = outputs['trend_adherence']
    logger.info(f"  Overall trend adherence: {trend_adherence.mean():.4f}")
    
    gradient_trends = outputs['gradient_trends'].cpu().numpy()
    logger.info("  Gradient trends (per region/scale):")
    for region_idx, region in enumerate(['Core', 'Cladding', 'Ferrule']):
        logger.info(f"    {region}: {gradient_trends[region_idx]}")
    
    # 5. Quality assessment
    logger.info("\n5. FEATURE QUALITY ASSESSMENT:")
    quality_map = outputs['quality_map']
    logger.info(f"  Average quality score: {quality_map.mean():.4f}")
    logger.info(f"  Min quality: {quality_map.min():.4f} at "
               f"{torch.where(quality_map == quality_map.min())}")
    
    # 6. Reconstruction analysis
    logger.info("\n6. RECONSTRUCTION ANALYSIS:")
    reconstruction = outputs['reconstruction']
    reconstruction_error = torch.abs(reconstruction - image_tensor.to(config.get_device())).mean()
    logger.info(f"  Reconstruction error: {reconstruction_error:.4f}")
    
    # Export results
    if not analyze_synthetic:
        export_advanced_results(outputs, image_tensor, image_path.stem)


def run_advanced_demo():
    """
    Interactive demo showing how features are simultaneously analyzed.
    "so I get a fully detailed anomaly detection while also classifying most 
    probable features(or segments) of the image at the same time"
    """
    logger = get_logger()
    config = get_config()
    
    logger.info("Running advanced integrated demo...")
    
    # Create model
    model = AdvancedIntegratedFiberOpticsNN()
    model.eval()
    model = model.to(config.get_device())
    
    # Create test images with different characteristics
    test_cases = [
        ("Normal fiber", create_normal_fiber_image()),
        ("Fiber with scratch", create_fiber_with_scratch()),
        ("Fiber with contamination", create_fiber_with_contamination()),
        ("Misaligned fiber", create_misaligned_fiber())
    ]
    
    logger.info("\nProcessing different fiber optic scenarios:\n")
    
    for name, image in test_cases:
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing: {name}")
        logger.info(f"{'='*40}")
        
        with torch.no_grad():
            outputs = model(image.to(config.get_device()))
        
        # Show simultaneous results
        logger.info("\nSIMULTANEOUS ANALYSIS RESULTS:")
        
        # Classification
        segmentation = outputs['segmentation']
        region_probs = torch.softmax(segmentation, dim=1).mean(dim=[2,3])
        logger.info(f"  Region classification:")
        for i, region in enumerate(['Core', 'Cladding', 'Ferrule']):
            logger.info(f"    {region}: {region_probs[0,i]:.1%}")
        
        # Anomaly detection (happening simultaneously)
        anomaly_score = outputs['anomaly_map'].mean().item()
        max_anomaly = outputs['anomaly_map'].max().item()
        logger.info(f"\n  Anomaly detection:")
        logger.info(f"    Average score: {anomaly_score:.4f}")
        logger.info(f"    Max anomaly: {max_anomaly:.4f}")
        
        if max_anomaly > 0.5:
            logger.info("    ⚠️  ANOMALY DETECTED!")
            
            # Which scales detected it?
            for i, analysis in enumerate(outputs['all_analysis']):
                scale_anomaly = analysis['anomaly_scores'].max().item()
                if scale_anomaly > 0.3:
                    logger.info(f"    - Detected at scale {i+1} with score {scale_anomaly:.4f}")
        
        # Quality assessment
        quality = outputs['quality_map'].mean().item()
        logger.info(f"\n  Overall quality: {quality:.1%}")
        
        # Reference similarity
        similarity = outputs['similarity'][0].item()
        logger.info(f"  Reference similarity: {similarity:.4f}")
        
        time.sleep(0.5)  # Brief pause for readability


def compare_processing_methods():
    """Compare the three different approaches to show the advancement"""
    logger = get_logger()
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON OF PROCESSING APPROACHES")
    logger.info("="*60)
    
    logger.info("\n1. ORIGINAL APPROACH (Separate Steps):")
    logger.info("   - Load image → Preprocess → Segment → Compare → Detect anomalies")
    logger.info("   - Each step is independent")
    logger.info("   - Anomaly detection happens after classification")
    
    logger.info("\n2. INTEGRATED APPROACH (First Improvement):")
    logger.info("   - Neural network performs all steps internally")
    logger.info("   - Features → Classification → Reconstruction → Anomalies")
    logger.info("   - Still sequential within the network")
    
    logger.info("\n3. ADVANCED INTEGRATED (Current):")
    logger.info("   - Simultaneous processing at each feature level")
    logger.info("   - Each feature is analyzed for BOTH classification AND anomalies")
    logger.info("   - Multi-scale correlation and trend analysis")
    logger.info("   - Quality assessment integrated throughout")
    
    logger.info("\nKEY ADVANTAGES:")
    logger.info("- No information loss between steps")
    logger.info("- Anomalies influence classification and vice versa")
    logger.info("- Multiple scales provide redundancy and detail")
    logger.info("- Trend analysis distinguishes defects from transitions")


def create_detailed_test_image() -> torch.Tensor:
    """Create a detailed synthetic fiber optic image"""
    size = 256
    image = np.zeros((size, size, 3), dtype=np.float32)
    center = size // 2
    
    # Create fiber structure
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Core (bright center)
    core_mask = dist < 25
    image[core_mask] = [0.95, 0.95, 0.90]
    
    # Cladding (medium intensity)
    cladding_mask = (dist >= 25) & (dist < 70)
    image[cladding_mask] = [0.5, 0.5, 0.48]
    
    # Ferrule (darker outer region)
    ferrule_mask = dist >= 70
    image[ferrule_mask] = [0.2, 0.2, 0.25]
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    # Add gradient variation
    gradient = np.linspace(0.9, 1.1, size).reshape(1, size, 1)
    image = image * gradient
    
    # Convert to tensor
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return tensor


def create_normal_fiber_image() -> torch.Tensor:
    """Create a normal fiber optic image"""
    return create_detailed_test_image()


def create_fiber_with_scratch() -> torch.Tensor:
    """Create fiber with scratch defect"""
    image = create_detailed_test_image()
    
    # Add scratch
    scratch_start = (80, 80)
    scratch_end = (140, 85)
    
    # Draw scratch line
    for i in range(60):
        y = int(scratch_start[0] + i * (scratch_end[0] - scratch_start[0]) / 60)
        x = int(scratch_start[1] + i * (scratch_end[1] - scratch_start[1]) / 60)
        if 0 <= y < 256 and 0 <= x < 256:
            image[0, :, y-1:y+2, x-1:x+2] = 0.1  # Dark scratch
    
    return image


def create_fiber_with_contamination() -> torch.Tensor:
    """Create fiber with contamination"""
    image = create_detailed_test_image()
    
    # Add contamination spots
    contamination_centers = [(100, 100), (150, 120), (120, 140)]
    
    for cy, cx in contamination_centers:
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        contamination_mask = dist < 8
        
        # Make contamination darker and irregular
        noise = np.random.random(image[0, 0].shape) * 0.3
        image[0, :, contamination_mask] *= (0.4 + noise[contamination_mask])
    
    return image


def create_misaligned_fiber() -> torch.Tensor:
    """Create misaligned fiber (center offset)"""
    size = 256
    image = np.zeros((size, size, 3), dtype=np.float32)
    
    # Offset center
    center_x = size // 2 + 30
    center_y = size // 2 - 20
    
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create offset fiber
    core_mask = dist < 25
    image[core_mask] = [0.95, 0.95, 0.90]
    
    cladding_mask = (dist >= 25) & (dist < 70)
    image[cladding_mask] = [0.5, 0.5, 0.48]
    
    ferrule_mask = dist >= 70
    image[ferrule_mask] = [0.2, 0.2, 0.25]
    
    # Add noise
    noise = np.random.normal(0, 0.02, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return tensor


def export_advanced_results(outputs: Dict, image_tensor: torch.Tensor, image_id: str):
    """Export results from advanced analysis"""
    logger = get_logger()
    exporter = ResultsExporter()
    
    # Create result structure
    from fiber_optics_nn.core.anomaly_detection import AnomalyResult
    
    anomaly_result = AnomalyResult(
        defect_map=(outputs['anomaly_map'] > 0.3).float(),
        anomaly_heatmap=outputs['anomaly_map'],
        defect_locations=[],  # Would extract from anomaly map
        defect_types=[],
        confidence_scores=[],
        combined_anomaly_score=outputs['anomaly_map'].mean().item()
    )
    
    output_path = exporter.export_anomaly_result(
        f"advanced_{image_id}",
        anomaly_result,
        image_tensor.squeeze(0),
        additional_data={
            'segmentation': outputs['segmentation'].cpu().numpy(),
            'quality_map': outputs['quality_map'].cpu().numpy(),
            'similarity': outputs['similarity'].cpu().numpy(),
            'anomaly_details': outputs['anomaly_details'],
            'equation_parameters': outputs['gradient_trends'].cpu().numpy()
        }
    )
    
    logger.info(f"\nResults exported to: {output_path}")


def save_advanced_model(model: AdvancedIntegratedFiberOpticsNN, 
                       trainer: AdvancedIntegratedTrainer):
    """Save the trained advanced model"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    final_path = checkpoint_dir / "advanced_integrated_final.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': trainer.history,
        'loss_weights': trainer.loss_weights,
        'equation_parameters': model.get_detailed_equation_parameters()
    }, final_path)
    
    trainer.logger.info(f"Saved final advanced model to {final_path}")


def find_best_checkpoint() -> Optional[Path]:
    """Find the best available checkpoint"""
    checkpoint_dir = Path("checkpoints")
    
    # Look for final model first
    final_path = checkpoint_dir / "advanced_integrated_final.pth"
    if final_path.exists():
        return final_path
    
    # Otherwise find latest epoch checkpoint
    checkpoints = list(checkpoint_dir.glob("advanced_integrated_epoch_*.pth"))
    if checkpoints:
        return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    
    return None


def print_advanced_usage():
    """Print usage information"""
    print("\nAdvanced Integrated Fiber Optics Neural Network")
    print("=" * 50)
    print("\nUsage:")
    print("  python advanced_integrated_main.py train [epochs]")
    print("  python advanced_integrated_main.py analyze [image]")
    print("  python advanced_integrated_main.py demo")
    print("  python advanced_integrated_main.py compare")
    print("\nThis system performs simultaneous feature classification")
    print("and anomaly detection at multiple scales within the network.")


if __name__ == "__main__":
    main()