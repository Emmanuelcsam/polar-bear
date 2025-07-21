#!/usr/bin/env python3
"""
Main inference script for fiber optics image analysis.
Demonstrates the complete pipeline from image loading to defect detection.
"""

import torch
from pathlib import Path
import time
import sys

# Add the module to path
sys.path.append(str(Path(__file__).parent))

from fiber_optics_nn import (
    get_logger,
    get_config,
    TensorDataLoader,
    ImageProcessor,
    FiberOpticSegmentation,
    FiberOpticsNeuralNetwork,
    SimilarityCalculator,
    AnomalyDetector,
    ResultsExporter
)


def main():
    """
    Main inference pipeline.
    "the process: an image will be selected from a dataset folder(or multiple images 
    for batch processing or realtime processing)"
    """
    # Initialize components
    logger = get_logger()
    config = get_config()
    
    logger.info("=" * 80)
    logger.info("FIBER OPTICS NEURAL NETWORK - INFERENCE MODE")
    logger.info("=" * 80)
    
    # Initialize all modules
    logger.info("Initializing modules...")
    
    data_loader = TensorDataLoader()
    image_processor = ImageProcessor()
    segmentation = FiberOpticSegmentation()
    model = FiberOpticsNeuralNetwork().to(config.get_device())
    similarity_calculator = SimilarityCalculator()
    anomaly_detector = AnomalyDetector()
    results_exporter = ResultsExporter()
    
    # Load model weights if available
    checkpoint_path = Path("checkpoints/fiber_optics_nn_best.pth")
    if checkpoint_path.exists():
        logger.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        logger.warning("No trained model found, using random initialization")
        model.eval()
    
    # Load reference data
    logger.info("Loading reference tensors...")
    num_references = data_loader.load_all_references(preload=True)
    logger.info(f"Loaded {num_references} reference tensors")
    
    # Example: Process a single image
    # You would replace this with your actual image path
    test_image_path = Path("test_image.jpg")
    
    if test_image_path.exists():
        process_single_image(
            test_image_path,
            image_processor,
            segmentation,
            model,
            data_loader,
            similarity_calculator,
            anomaly_detector,
            results_exporter,
            logger
        )
    else:
        # Process a batch from the reference data as example
        logger.info("Processing example batch from reference data...")
        process_batch_example(
            data_loader,
            segmentation,
            model,
            similarity_calculator,
            anomaly_detector,
            results_exporter,
            logger
        )


def process_single_image(image_path, image_processor, segmentation, model, 
                        data_loader, similarity_calculator, anomaly_detector,
                        results_exporter, logger):
    """
    Process a single image through the complete pipeline.
    "the image will then be tensorized so it can better be computed by pytorch"
    """
    logger.info(f"Processing image: {image_path}")
    start_time = time.time()
    
    # Step 1: Load and tensorize image
    logger.info("Step 1: Loading and tensorizing image...")
    image_tensor = image_processor.process_image(image_path)
    
    # Step 2: Calculate image statistics
    logger.info("Step 2: Calculating image statistics...")
    image_stats = image_processor.calculate_image_statistics(image_tensor)
    gradient_intensity = data_loader.calculate_gradient_intensity(image_tensor)
    avg_x, avg_y = data_loader.calculate_pixel_positions(image_tensor)
    
    image_stats['gradient_intensity'] = gradient_intensity
    image_stats['center_x'] = avg_x
    image_stats['center_y'] = avg_y
    
    # Step 3: Segment the image
    logger.info("Step 3: Segmenting image into core/cladding/ferrule...")
    segmentation_result = segmentation.segment_image(image_tensor)
    
    logger.info(f"Segmentation confidence scores: {segmentation_result.confidence_scores}")
    
    # Step 4: Forward pass through neural network
    logger.info("Step 4: Running neural network inference...")
    with torch.no_grad():
        outputs = model(
            image_tensor.unsqueeze(0),
            image_stats,
            segmentation_result
        )
    
    # Step 5: Find best matching reference
    logger.info("Step 5: Finding best matching reference image...")
    
    # Determine dominant region
    region_sizes = {
        'core': segmentation_result.core_mask.sum().item(),
        'cladding': segmentation_result.cladding_mask.sum().item(),
        'ferrule': segmentation_result.ferrule_mask.sum().item()
    }
    dominant_region = max(region_sizes, key=region_sizes.get)
    
    reference_tensor, ref_key, similarity = data_loader.get_reference_by_similarity(
        image_tensor, dominant_region
    )
    
    logger.info(f"Best reference match: {ref_key} with similarity: {similarity:.4f}")
    
    # Check similarity threshold
    if similarity < config.SIMILARITY_THRESHOLD:
        logger.warning(f"Similarity {similarity:.4f} below threshold {config.SIMILARITY_THRESHOLD}")
    
    # Step 6: Detect anomalies
    logger.info("Step 6: Detecting anomalies and defects...")
    anomaly_result = anomaly_detector.detect_anomalies(
        image_tensor,
        reference_tensor,
        {
            'core': segmentation_result.core_mask,
            'cladding': segmentation_result.cladding_mask,
            'ferrule': segmentation_result.ferrule_mask
        }
    )
    
    # Generate defect report
    defect_report = anomaly_detector.generate_defect_report(anomaly_result)
    logger.info(f"Defect report: {defect_report}")
    
    # Step 7: Export results
    logger.info("Step 7: Exporting results...")
    output_path = results_exporter.export_anomaly_result(
        image_path.stem,
        anomaly_result,
        image_tensor,
        {
            'similarity_score': similarity,
            'reference_key': ref_key,
            'segmentation_confidence': segmentation_result.confidence_scores,
            'model_outputs': {k: v.cpu().numpy() for k, v in outputs.items()}
        }
    )
    
    # Also export visualization
    vis_path = results_exporter.export_visualization(
        image_path.stem,
        image_tensor,
        anomaly_result,
        {
            'core': segmentation_result.core_mask,
            'cladding': segmentation_result.cladding_mask,
            'ferrule': segmentation_result.ferrule_mask
        }
    )
    
    # Calculate total processing time
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Image: {image_path}")
    logger.info(f"Similarity to reference: {similarity:.4f}")
    logger.info(f"Number of defects found: {len(anomaly_result.defect_locations)}")
    logger.info(f"Overall anomaly score: {anomaly_result.combined_anomaly_score:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Visualization saved to: {vis_path}")
    logger.info("=" * 60)


def process_batch_example(data_loader, segmentation, model, similarity_calculator,
                         anomaly_detector, results_exporter, logger):
    """
    Process a batch of images as example.
    "or multiple images for batch processing"
    """
    # Get a batch of tensors
    batch_size = 5
    batch_tensors, batch_keys = data_loader.get_tensor_batch(batch_size=batch_size)
    
    logger.info(f"Processing batch of {len(batch_keys)} images...")
    
    results = []
    
    for i, (tensor, key) in enumerate(zip(batch_tensors, batch_keys)):
        logger.info(f"Processing {i+1}/{batch_size}: {key}")
        
        # Calculate statistics
        gradient_intensity = data_loader.calculate_gradient_intensity(tensor)
        avg_x, avg_y = data_loader.calculate_pixel_positions(tensor)
        
        image_stats = {
            'gradient_intensity': gradient_intensity,
            'center_x': avg_x,
            'center_y': avg_y
        }
        
        # Segment
        seg_result = segmentation.segment_image(tensor)
        
        # Model inference
        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0), image_stats, seg_result)
        
        # Get reference
        ref_tensor, ref_key, similarity = data_loader.get_reference_by_similarity(
            tensor, 'core'  # Example region
        )
        
        # Detect anomalies
        anomaly_result = anomaly_detector.detect_anomalies(
            tensor, ref_tensor
        )
        
        results.append({
            'image_id': key,
            'anomaly_result': anomaly_result,
            'input_tensor': tensor,
            'additional_data': {
                'similarity': similarity,
                'reference': ref_key
            }
        })
    
    # Export batch results
    batch_output = results_exporter.export_batch_results(results)
    
    # Generate summary report
    summary_path = results_exporter.export_summary_report(
        results,
        "batch_analysis_summary"
    )
    
    logger.info(f"Batch processing complete. Results saved to: {batch_output}")
    logger.info(f"Summary report saved to: {summary_path}")


if __name__ == "__main__":
    main()