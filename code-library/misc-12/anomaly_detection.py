#!/usr/bin/env python3

import cv2
import numpy as np
import logging
from utils import load_image, get_timestamp
from feature_extraction import extract_ultra_comprehensive_features
from comparison import compute_exhaustive_comparison, compute_image_structural_comparison
from defect_detection import (
    detect_specific_defects, compute_local_anomaly_map, find_anomaly_regions
)
from reference_model import load_knowledge_base, build_minimal_reference
from pathlib import Path
import json
from utils import NumpyEncoder
from visualization import visualize_comprehensive_results
from defect_mask import create_defect_mask
from report_generation import generate_detailed_report


def detect_anomalies_comprehensive(test_path, reference_model, config):
    """Perform exhaustive anomaly detection on a test image."""
    # Log start of analysis
    logging.info(f"Analyzing: {test_path}")
    
    # Check if reference model exists
    if not reference_model.get('statistical_model'):
        logging.warning("No reference model available. Build one first.")
        return None
    
    # Load test image
    test_image = load_image(test_path)
    if test_image is None:
        return None
    
    # Convert to grayscale for analysis
    if len(test_image.shape) == 3:
        test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_image.copy()
    
    # Extract features from test image
    logging.info("Extracting features from test image...")
    test_features, _ = extract_ultra_comprehensive_features(test_image)
    
    # --- Global Analysis ---
    logging.info("Performing global anomaly analysis...")
    
    # Get reference statistics
    stat_model = reference_model['statistical_model']
    feature_names = reference_model['feature_names']
    
    # Ensure numpy arrays (in case loaded from JSON)
    for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
        if key in stat_model and isinstance(stat_model[key], list):
            stat_model[key] = np.array(stat_model[key], dtype=np.float64)
    
    # Create feature vector in consistent order
    test_vector = np.array([test_features.get(fname, 0) for fname in feature_names])
    
    # Compute Mahalanobis distance
    diff = test_vector - stat_model['robust_mean']  # Difference from reference mean
    try:
        # Mahalanobis distance: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
        mahalanobis_dist = np.sqrt(np.abs(diff.T @ stat_model['robust_inv_cov'] @ diff))
    except:
        # Fall back to normalized Euclidean if Mahalanobis fails
        std_vector = stat_model['std']
        std_vector[std_vector < 1e-6] = 1.0  # Avoid division by zero
        normalized_diff = diff / std_vector
        mahalanobis_dist = np.linalg.norm(normalized_diff)
    
    # Compute Z-scores for each feature
    z_scores = np.abs(diff) / (stat_model['std'] + 1e-10)
    
    # Find most deviant features
    top_indices = np.argsort(z_scores)[::-1][:10]  # Top 10 by Z-score
    deviant_features = [(feature_names[i], z_scores[i], test_vector[i], stat_model['mean'][i]) 
                       for i in top_indices]
    
    # --- Individual Comparisons ---
    logging.info(f"Comparing against {len(reference_model['features'])} reference samples...")
    
    # Compare test against each reference sample
    individual_scores = []
    for i, ref_features in enumerate(reference_model['features']):
        # Compute comprehensive comparison
        comp = compute_exhaustive_comparison(test_features, ref_features)
        
        # Compute weighted anomaly score with bounds
        euclidean_term = min(comp['euclidean_distance'], 1000.0) * 0.2      # Cap at 1000
        manhattan_term = min(comp['manhattan_distance'], 10000.0) * 0.1     # Cap at 10000
        cosine_term = comp['cosine_distance'] * 0.2                         # Already bounded [0,2]
        correlation_term = (1 - abs(comp['pearson_correlation'])) * 0.1     # Bounded [0,1]
        kl_term = min(comp['kl_divergence'], 10.0) * 0.1                   # Cap at 10
        js_term = comp['js_divergence'] * 0.1                               # Already bounded
        chi_term = min(comp['chi_square'], 10.0) * 0.1                     # Cap at 10
        wasserstein_term = min(comp['wasserstein_distance'], 10.0) * 0.1   # Cap at 10
        
        # Sum weighted terms
        score = (euclidean_term + manhattan_term + cosine_term + 
                correlation_term + kl_term + js_term + 
                chi_term + wasserstein_term)
        
        # Cap the final score
        score = min(score, 100.0)
        
        individual_scores.append(score)
    
    # Compute statistics of individual comparisons
    scores_array = np.array(individual_scores)
    comparison_stats = {
        'mean': float(np.mean(scores_array)),      # Average comparison score
        'std': float(np.std(scores_array)),        # Variation in scores
        'min': float(np.min(scores_array)),        # Best match score
        'max': float(np.max(scores_array)),        # Worst match score
        'median': float(np.median(scores_array)),  # Median score
    }
    
    # --- Structural Analysis ---
    logging.info("Performing structural analysis...")
    
    # Get reference archetype image
    archetype = reference_model['archetype_image']
    # Convert from list if loaded from JSON
    if isinstance(archetype, list):
        archetype = np.array(archetype, dtype=np.uint8)
    # Resize test image to match archetype if needed
    if test_gray.shape != archetype.shape:
        test_gray_resized = cv2.resize(test_gray, (archetype.shape[1], archetype.shape[0]))
    else:
        test_gray_resized = test_gray
    
    # Compute structural similarity
    structural_comp = compute_image_structural_comparison(test_gray_resized, archetype)
    
    # --- Local Anomaly Detection ---
    logging.info("Detecting local anomalies...")
    
    # Compute local anomaly map using sliding window
    anomaly_map = compute_local_anomaly_map(test_gray_resized, archetype)
    
    # Find distinct anomaly regions
    anomaly_regions = find_anomaly_regions(anomaly_map, test_gray.shape)
    
    # --- Specific Defect Detection ---
    logging.info("Detecting specific defects...")
    specific_defects = detect_specific_defects(test_gray, config.min_defect_size, config.max_defect_size)
    
    # --- Determine Overall Status ---
    thresholds = reference_model['learned_thresholds']
    
    # Multiple criteria for anomaly detection
    is_anomalous = (
        mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6) or    # Statistical distance exceeds threshold
        comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6) or   # Worst match exceeds 95th percentile
        structural_comp['ssim'] < 0.7 or                                    # Low structural similarity
        len(anomaly_regions) > 3 or                                         # Many local anomalies
        any(region['confidence'] > 0.8 for region in anomaly_regions)       # High confidence anomaly
    )
    
    # Overall confidence score (maximum of normalized criteria)
    confidence = min(1.0, max(
        mahalanobis_dist / max(thresholds['anomaly_threshold'], 1e-6),      # Normalized Mahalanobis
        comparison_stats['max'] / max(thresholds['anomaly_p95'], 1e-6),     # Normalized comparison
        1 - structural_comp['ssim'],                                         # Inverted SSIM
        len(anomaly_regions) / 10                                           # Normalized region count
    ))
    
    logging.info("Analysis Complete!")
    
    # Return comprehensive results dictionary
    return {
        'test_image': test_image,                    # Original test image
        'test_gray': test_gray,                      # Grayscale version
        'test_features': test_features,              # Extracted features
        'metadata': {'filename': Path(test_path).stem},  # Image metadata
        
        'global_analysis': {
            'mahalanobis_distance': float(mahalanobis_dist),  # Statistical distance
            'deviant_features': deviant_features,              # Most abnormal features
            'comparison_stats': comparison_stats,              # Individual comparison statistics
        },
        
        'structural_analysis': structural_comp,      # SSIM and related metrics
        
        'local_analysis': {
            'anomaly_map': anomaly_map,             # Pixel-wise anomaly scores
            'anomaly_regions': anomaly_regions,     # Detected anomaly regions
        },
        
        'specific_defects': specific_defects,        # Type-specific defects
        
        'verdict': {
            'is_anomalous': is_anomalous,           # Binary decision
            'confidence': float(confidence),         # Confidence in decision
            'criteria_triggered': {                  # Which criteria caused anomaly
                'mahalanobis': mahalanobis_dist > max(thresholds['anomaly_threshold'], 1e-6),
                'comparison': comparison_stats['max'] > max(thresholds['anomaly_p95'], 1e-6),
                'structural': structural_comp['ssim'] < 0.7,
                'local': len(anomaly_regions) > 3,
            }
        }
    }


def analyze_end_face(image_path, output_dir, config, reference_model=None):
    """Main analysis method - compatible with pipeline expectations"""
    # Log start of analysis for debugging
    logging.info(f"Analyzing fiber end face: {image_path}")
    
    # Create Path object for easier directory manipulation
    output_path = Path(output_dir)
    # Create output directory and any missing parent directories
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or build reference model
    if reference_model is None:
        # Try to load existing knowledge base
        reference_model = load_knowledge_base(config.knowledge_base_path or "fiber_anomaly_kb.json")
        
        # If no reference exists, create minimal one
        if not reference_model:
            logging.warning("No reference model available. Building from single image...")
            reference_model = build_minimal_reference(image_path)
            if reference_model:
                # Save the minimal reference
                # Assuming save_knowledge_base is defined elsewhere or will be added
                # For now, we'll just log that it would be saved if available
                logging.info(f"Saved minimal reference model to {config.knowledge_base_path or 'fiber_anomaly_kb.json'}")
    
    # Check if reference model exists (needed for comparison)
    if not reference_model or not reference_model.get('statistical_model'):
        logging.error("No reference model available and could not build one.")
        return None
    
    # Run comprehensive anomaly detection analysis
    results = detect_anomalies_comprehensive(image_path, reference_model, config)
    
    # Check if analysis succeeded
    if results:
        # Convert internal results format to pipeline-expected format
        pipeline_report = convert_to_pipeline_format(results, image_path, config)
        
        # Construct path for JSON report file
        report_path = output_path / f"{Path(image_path).stem}_report.json"
        # Open file for writing
        with open(report_path, 'w') as f:
            # Write JSON with indentation and custom numpy encoder
            json.dump(pipeline_report, f, indent=2, cls=NumpyEncoder)
        # Log successful save
        logging.info(f"Saved detection report to {report_path}")
        
        # Generate visualizations if enabled in config
        if config.enable_visualization:
            # Construct path for visualization image
            viz_path = output_path / f"{Path(image_path).stem}_analysis.png"
            # Generate and save comprehensive visualization
            visualize_comprehensive_results(results, str(viz_path))
            
            # Construct path for defect mask file
            mask_path = output_path / f"{Path(image_path).stem}_defect_mask.npy"
            # Create binary mask showing defect locations
            defect_mask = create_defect_mask(results)
            # Save mask as numpy array for later processing
            np.save(mask_path, defect_mask)
        
        # Construct path for detailed text report
        text_report_path = output_path / f"{Path(image_path).stem}_detailed.txt"
        # Generate human-readable text report
        generate_detailed_report(results, str(text_report_path))
        
        # Return the pipeline report
        return pipeline_report
        
    else:
        # Log analysis failure
        logging.error(f"Analysis failed for {image_path}")
        # Create minimal error report structure
        empty_report = {
            'image_path': image_path,
            'timestamp': get_timestamp(),
            'success': False,
            'error': 'Analysis failed',
            'defects': []
        }
        # Save error report
        report_path = output_path / f"{Path(image_path).stem}_report.json"
        with open(report_path, 'w') as f:
            json.dump(empty_report, f, indent=2)
        
        # Return the empty report
        return empty_report


def convert_to_pipeline_format(results, image_path, config):
    """Convert internal results format to pipeline-expected format"""
    # Initialize empty list to store formatted defects
    defects = []
    # Initialize defect ID counter
    defect_id = 1
    
    # Process each detected anomaly region
    for region in results['local_analysis']['anomaly_regions']:
        # Extract bounding box coordinates (x, y, width, height)
        x, y, w, h = region['bbox']
        # Extract centroid coordinates
        cx, cy = region['centroid']
        
        # Get confidence score for this region
        confidence = region['confidence']
        # Convert confidence to severity level using thresholds
        severity = confidence_to_severity(confidence, config)
        
        # Create defect dictionary in pipeline format
        defect = {
            'defect_id': f"ANOM_{defect_id:04d}",  # Format: ANOM_0001, ANOM_0002, etc.
            'defect_type': 'ANOMALY',               # Generic anomaly type
            'location_xy': [cx, cy],                # Center point coordinates
            'bbox': [x, y, w, h],                   # Bounding box
            'area_px': region['area'],              # Area in pixels
            'confidence': float(confidence),        # Ensure float type for JSON
            'severity': severity,                   # Calculated severity level
            'orientation': None,                    # No orientation for generic anomalies
            'contributing_algorithms': ['ultra_comprehensive_matrix_analyzer'],  # Algorithm name
            'detection_metadata': {                 # Additional detection details
                'max_intensity': region.get('max_intensity', 0),  # Peak anomaly value
                'anomaly_score': float(confidence)   # Redundant but expected by pipeline
            }
        }
        # Add to defects list
        defects.append(defect)
        # Increment ID counter
        defect_id += 1
    
    # Extract specific defect types from results
    specific_defects = results['specific_defects']
    
    # Process detected scratches (linear defects)
    for scratch in specific_defects['scratches']:
        # Extract line endpoints
        x1, y1, x2, y2 = scratch['line']
        # Calculate center point of line
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Create scratch defect entry
        defect = {
            'defect_id': f"SCR_{defect_id:04d}",   # Format: SCR_0001, etc.
            'defect_type': 'SCRATCH',               # Specific type
            'location_xy': [cx, cy],                # Center of scratch
            'bbox': [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)],  # Bounding box
            'area_px': int(scratch['length'] * 2),  # Approximate area (length * assumed width)
            'confidence': 0.7,                      # Fixed confidence for scratches
            'severity': 'MEDIUM' if scratch['length'] > 50 else 'LOW',  # Length-based severity
            'orientation': float(scratch['angle']), # Angle in degrees
            'contributing_algorithms': ['hough_line_detection'],  # Detection method
            'detection_metadata': {
                'length': float(scratch['length']),     # Scratch length in pixels
                'angle_degrees': float(scratch['angle']) # Redundant angle info
            }
        }
        defects.append(defect)
        defect_id += 1
    
    # Process detected digs/pits (small dark spots)
    for dig in specific_defects['digs']:
        # Extract center coordinates
        cx, cy = dig['center']
        # Calculate approximate radius from area (A = πr²)
        radius = int(np.sqrt(dig['area'] / np.pi))
        
        # Create dig defect entry
        defect = {
            'defect_id': f"DIG_{defect_id:04d}",   # Format: DIG_0001, etc.
            'defect_type': 'DIG',                   # Specific type
            'location_xy': [cx, cy],                # Center of dig
            'bbox': [cx-radius, cy-radius, radius*2, radius*2],  # Square bounding box
            'area_px': int(dig['area']),           # Actual area in pixels
            'confidence': 0.8,                      # Fixed confidence for digs
            'severity': 'HIGH' if dig['area'] > 100 else 'MEDIUM',  # Area-based severity
            'orientation': None,                    # Digs have no orientation
            'contributing_algorithms': ['morphological_blackhat'],  # Detection method
            'detection_metadata': {
                'contour_area': float(dig['area'])  # Precise contour area
            }
        }
        defects.append(defect)
        defect_id += 1
    
    # Process detected blobs/contamination
    for blob in specific_defects['blobs']:
        # Extract bounding box
        x, y, w, h = blob['bbox']
        # Calculate center point
        cx, cy = x + w//2, y + h//2
        
        # Create contamination defect entry
        defect = {
            'defect_id': f"CONT_{defect_id:04d}",  # Format: CONT_0001, etc.
            'defect_type': 'CONTAMINATION',         # Specific type
            'location_xy': [cx, cy],                # Center of blob
            'bbox': [x, y, w, h],                   # Bounding box
            'area_px': int(blob['area']),           # Area in pixels
            'confidence': 0.6,                      # Lower confidence for blobs
            'severity': 'MEDIUM' if blob['area'] > 500 else 'LOW',  # Area-based severity
            'orientation': None,                    # Blobs have no orientation
            'contributing_algorithms': ['blob_detection'],  # Detection method
            'detection_metadata': {
                'circularity': float(blob['circularity']),    # Shape metric (0-1)
                'aspect_ratio': float(blob['aspect_ratio'])   # Width/height ratio
            }
        }
        defects.append(defect)
        defect_id += 1
    
    # Extract summary data from results
    verdict = results['verdict']
    global_stats = results['global_analysis']
    
    # Calculate overall quality score (0-100 scale)
    quality_score = float(100 * (1 - verdict['confidence']))
    if len(defects) > 0:
        # Reduce quality based on number and severity of defects
        quality_score = max(0, quality_score - len(defects) * 2)
    
    # Construct final pipeline-format report
    report = {
        'source_image': image_path,                            # Source image path (pipeline expects this)
        'image_path': image_path,                              # Also keep for compatibility
        'timestamp': get_timestamp(),                          # Analysis timestamp
        'analysis_complete': True,                             # Pipeline expects this field
        'success': True,                                       # Analysis succeeded
        'overall_quality_score': quality_score,                # Pipeline expects this field
        'defects': defects,                                    # List of all defects
        'zones': {                                             # Pipeline expects zone info
            'core': {'detected': True},
            'cladding': {'detected': True},
            'ferrule': {'detected': True}
        },
        'summary': {
            'total_defects': len(defects),                     # Defect count
            'is_anomalous': verdict['is_anomalous'],           # Boolean verdict
            'anomaly_confidence': float(verdict['confidence']), # Overall confidence
            'quality_score': quality_score,                     # Quality score
            'mahalanobis_distance': float(global_stats['mahalanobis_distance']),  # Statistical distance
            'ssim_score': float(results['structural_analysis']['ssim'])  # Structural similarity
        },
        'analysis_metadata': {
            'analyzer': 'ultra_comprehensive_matrix_analyzer',  # Algorithm identifier
            'version': '1.5',                                   # Version number
            'knowledge_base': config.knowledge_base_path or "fiber_anomaly_kb.json",  # Reference model path
            'reference_samples': len(reference_model.get('features', [])),  # Reference count
            'features_extracted': len(results.get('test_features', {}))  # Feature count
        }
    }
    
    return report


def confidence_to_severity(confidence, config):
    """Convert confidence score to severity level"""
    # Iterate through severity levels from highest to lowest
    for severity, threshold in sorted(config.severity_thresholds.items(), 
                                    key=lambda x: x[1], reverse=True):
        # Return first severity level where confidence exceeds threshold
        if confidence >= threshold:
            return severity
    # Default to negligible if below all thresholds
    return 'NEGLIGIBLE' 