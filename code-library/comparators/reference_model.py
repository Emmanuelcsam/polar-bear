#!/usr/bin/env python3

import json
import os
import numpy as np
import logging
from utils import get_timestamp, load_image
from feature_extraction import extract_ultra_comprehensive_features
from comparison import compute_exhaustive_comparison
from config import NumpyEncoder


def compute_robust_statistics(data):
    """Compute robust mean and covariance using custom implementation."""
    # Get data dimensions
    n_samples, n_features = data.shape
    
    # Use median as initial robust mean estimate
    robust_mean = np.median(data, axis=0)
    
    # Compute deviations from median
    deviations = data - robust_mean
    # Compute Median Absolute Deviation for each feature
    mad = np.median(np.abs(deviations), axis=0)
    
    # Scale MAD to approximate standard deviation (1.4826 is consistency factor)
    mad_scaled = mad * 1.4826
    
    # Replace near-zero values to avoid division by zero
    mad_scaled[mad_scaled < 1e-6] = 1.0
    
    # Compute robust covariance using weighted approach
    # Normalize deviations by scaled MAD
    normalized_deviations = deviations / mad_scaled
    # Compute distance from center for each sample
    distances = np.sqrt(np.sum(normalized_deviations**2, axis=1))
    
    # Clip distances to avoid numerical issues
    distances = np.clip(distances, 0, 10)
    
    # Compute weights using Gaussian kernel
    weights = np.exp(-0.5 * distances)
    # Normalize weights
    weight_sum = weights.sum()
    
    # Check if weights are valid
    if weight_sum < 1e-10 or n_samples < 2:
        # Fall back to standard covariance if weights fail
        robust_cov = np.cov(data, rowvar=False)
        # Handle single feature case
        if robust_cov.ndim == 0:
            robust_cov = np.array([[robust_cov]])
    else:
        # Normalize weights
        weights = weights / weight_sum
        
        # Compute weighted covariance
        # Weight data by square root of weights
        weighted_data = data * np.sqrt(weights[:, np.newaxis])
        # Compute covariance of weighted data
        robust_cov = np.dot(weighted_data.T, weighted_data)
        
        # Apply bias correction using effective sample size
        effective_n = 1.0 / np.sum(weights**2)
        if effective_n > 1:
            robust_cov = robust_cov * effective_n / (effective_n - 1)
    
    # Ensure covariance matrix is well-conditioned
    # Add small regularization to diagonal
    reg_value = np.trace(robust_cov) / n_features * 1e-4
    if reg_value < 1e-6:
        reg_value = 1e-6
    robust_cov = robust_cov + np.eye(n_features) * reg_value
    
    # Ensure positive semi-definite through eigenvalue decomposition
    try:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(robust_cov)
        # Clip negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        # Reconstruct covariance matrix
        robust_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    except np.linalg.LinAlgError:
        # Fall back to diagonal matrix if decomposition fails
        var_scale = np.var(data)
        robust_cov = np.eye(n_features) * var_scale
    
    # Compute pseudo-inverse for Mahalanobis distance
    try:
        # Add extra regularization for stable inversion
        robust_inv_cov = np.linalg.pinv(robust_cov + np.eye(n_features) * 1e-4)
    except np.linalg.LinAlgError:
        # Fall back to diagonal approximation
        diag_values = np.diag(robust_cov)
        diag_values[diag_values < 1e-6] = 1e-6
        robust_inv_cov = np.diag(1.0 / diag_values)
    
    return robust_mean, robust_cov, robust_inv_cov


def get_default_thresholds():
    """Return default thresholds when learning fails."""
    return {
        'anomaly_mean': 1.0,                           # Default mean score
        'anomaly_std': 0.5,                            # Default standard deviation
        'anomaly_p90': 1.5,                            # Default 90th percentile
        'anomaly_p95': 2.0,                            # Default 95th percentile
        'anomaly_p99': 3.0,                            # Default 99th percentile
        'anomaly_threshold': 2.5,                      # Config multiplier
    }


def build_comprehensive_reference_model(ref_dir, config):
    """Build an exhaustive reference model from a directory of JSON/image files."""
    # Log start of model building
    logging.info(f"Building Comprehensive Reference Model from: {ref_dir}")
    
    # Define supported file extensions
    valid_extensions = ['.json', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    all_files = []
    
    # List all files in directory
    try:
        for filename in os.listdir(ref_dir):
            # Get file extension
            ext = os.path.splitext(filename)[1].lower()
            # Check if valid format
            if ext in valid_extensions:
                # Add full path to list
                all_files.append(os.path.join(ref_dir, filename))
    except Exception as e:
        # Log error if directory read fails
        logging.error(f"Error reading directory: {e}")
        return None
    
    # Sort files for consistent processing order
    all_files.sort()
    
    # Check if any files found
    if not all_files:
        logging.error(f"No valid files found in {ref_dir}")
        return None
    
    # Log file count
    logging.info(f"Found {len(all_files)} files to process")
    
    # Initialize storage lists
    all_features = []     # Feature dictionaries
    all_images = []       # Grayscale images
    feature_names = []    # Feature name list
    
    # Process each file
    logging.info("Processing files:")
    for i, file_path in enumerate(all_files, 1):
        # Log progress
        logging.info(f"[{i}/{len(all_files)}] {os.path.basename(file_path)}")
        
        # Load image
        image = load_image(file_path)
        if image is None:
            # Log failure and skip
            logging.warning(f"  Failed to load")
            continue
        
        # Convert to grayscale for consistent storage
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Extract features
        features, f_names = extract_ultra_comprehensive_features(image)
        
        # Store feature names from first image
        if not feature_names:
            feature_names = f_names
        
        # Add to collections
        all_features.append(features)
        all_images.append(gray)
        
        # Log success
        logging.info(f"  Processed: {len(features)} features extracted")
    
    # Check if any features extracted
    if not all_features:
        logging.error("No features could be extracted from any file")
        return None
    
    # Check minimum sample requirement
    if len(all_features) < 2:
        logging.error(f"At least 2 reference files are required, but only {len(all_features)} were successfully processed.")
        return None
    
    logging.info("Building Statistical Model...")
    
    # Convert features to matrix (samples x features)
    feature_matrix = np.zeros((len(all_features), len(feature_names)))
    for i, features in enumerate(all_features):
        for j, fname in enumerate(feature_names):
            # Get feature value, default to 0 if missing
            feature_matrix[i, j] = features.get(fname, 0)
    
    # Compute basic statistics
    mean_vector = np.mean(feature_matrix, axis=0)    # Feature means
    std_vector = np.std(feature_matrix, axis=0)      # Feature standard deviations
    median_vector = np.median(feature_matrix, axis=0) # Feature medians
    
    # Compute robust statistics
    logging.info("Computing robust statistics...")
    robust_mean, robust_cov, robust_inv_cov = compute_robust_statistics(feature_matrix)
    
    # Create archetype image (median of all images)
    logging.info("Creating archetype image...")
    # Get target dimensions from first image
    target_shape = all_images[0].shape
    aligned_images = []
    # Resize all images to same size
    for img in all_images:
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]))
        aligned_images.append(img)
    
    # Compute pixel-wise median
    archetype_image = np.median(aligned_images, axis=0).astype(np.uint8)
    
    # Learn anomaly thresholds from pairwise comparisons
    logging.info("Computing pairwise comparisons for threshold learning...")
    # Calculate total number of pairwise comparisons
    n_comparisons = len(all_features) * (len(all_features) - 1) // 2
    logging.info(f"Total comparisons to compute: {n_comparisons}")
    
    # Initialize comparison tracking
    comparison_scores = []
    comparison_count = 0
    
    # Compare all pairs of reference samples
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            # Compute comprehensive comparison
            comp = compute_exhaustive_comparison(all_features[i], all_features[j])
            
            # Compute weighted anomaly score
            score = (comp['euclidean_distance'] * 0.2 +      # Euclidean weight
                    comp['manhattan_distance'] * 0.1 +        # Manhattan weight
                    comp['cosine_distance'] * 0.2 +           # Cosine weight
                    (1 - abs(comp['pearson_correlation'])) * 0.1 +  # Correlation weight
                    min(comp['kl_divergence'], 10.0) * 0.1 +  # KL divergence weight (capped)
                    comp['js_divergence'] * 0.1 +             # JS divergence weight
                    min(comp['chi_square'], 10.0) * 0.1 +     # Chi-square weight (capped)
                    min(comp['wasserstein_distance'], 10.0) * 0.1)  # Wasserstein weight (capped)
            
            # Store score
            comparison_scores.append(score)
            comparison_count += 1
            
            # Log progress every 100 comparisons
            if comparison_count % 100 == 0:
                logging.info(f"  Progress: {comparison_count}/{n_comparisons} ({comparison_count/n_comparisons*100:.1f}%)")
    
    # Learn thresholds from comparison scores
    scores_array = np.array(comparison_scores)
    
    # Check if valid scores exist
    if len(scores_array) > 0 and not np.all(np.isnan(scores_array)):
        # Filter out invalid values
        valid_scores = scores_array[~np.isnan(scores_array)]
        valid_scores = valid_scores[np.isfinite(valid_scores)]
        
        if len(valid_scores) > 0:
            # Clip extreme outliers at 99.9th percentile
            valid_scores = np.clip(valid_scores, 0, np.percentile(valid_scores, 99.9))
            
            # Calculate statistics
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores))
            
            # Create threshold dictionary
            thresholds = {
                'anomaly_mean': mean_score,                    # Mean comparison score
                'anomaly_std': std_score,                      # Std of comparison scores
                'anomaly_p90': float(np.percentile(valid_scores, 90)),   # 90th percentile
                'anomaly_p95': float(np.percentile(valid_scores, 95)),   # 95th percentile
                'anomaly_p99': float(np.percentile(valid_scores, 99)),   # 99th percentile
                'anomaly_threshold': float(min(mean_score + config.anomaly_threshold_multiplier * std_score, # Statistical threshold
                                               np.percentile(valid_scores, 99.5),     # 99.5th percentile
                                               10.0)),                                # Hard cap at 10.0
            }
        else:
            # Use defaults if no valid scores
            thresholds = get_default_thresholds()
    else:
        # Use defaults if no scores computed
        thresholds = get_default_thresholds()
    
    # Store complete reference model
    reference_model = {
        'features': all_features,                      # All feature dictionaries
        'feature_names': feature_names,                # Consistent feature ordering
        'statistical_model': {
            'mean': mean_vector,                       # Feature means
            'std': std_vector,                         # Feature standard deviations
            'median': median_vector,                   # Feature medians
            'robust_mean': robust_mean,                # Robust mean estimate
            'robust_cov': robust_cov,                  # Robust covariance matrix
            'robust_inv_cov': robust_inv_cov,          # Inverse covariance for Mahalanobis
            'n_samples': len(all_features),            # Number of reference samples
        },
        'archetype_image': archetype_image,            # Median reference image
        'learned_thresholds': thresholds,              # Learned anomaly thresholds
        'timestamp': get_timestamp(),                  # Creation timestamp
    }
    
    # Log success summary
    logging.info("Reference Model Built Successfully!")
    logging.info(f"  - Samples: {len(all_features)}")
    logging.info(f"  - Features: {len(feature_names)}")
    logging.info(f"  - Anomaly threshold: {thresholds['anomaly_threshold']:.4f}")
    
    return reference_model


def load_knowledge_base(knowledge_base_path):
    """Load previously saved knowledge base from JSON."""
    # Check if knowledge base file exists
    if os.path.exists(knowledge_base_path):
        try:
            # Open and read JSON file
            with open(knowledge_base_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Convert archetype image list back to numpy array
            if loaded_data.get('archetype_image'):
                loaded_data['archetype_image'] = np.array(loaded_data['archetype_image'], dtype=np.uint8)
            
            # Convert statistical model lists back to numpy arrays
            if loaded_data.get('statistical_model'):
                # Iterate through array fields
                for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                    # Check if field exists and is not None
                    if key in loaded_data['statistical_model'] and loaded_data['statistical_model'][key] is not None:
                        # Convert list to numpy array
                        loaded_data['statistical_model'][key] = np.array(loaded_data['statistical_model'][key], dtype=np.float64)
            
            # Log successful load
            logging.info(f"Loaded knowledge base from {knowledge_base_path}")
            return loaded_data
        except Exception as e:
            # Log warning if load fails
            logging.warning(f"Could not load knowledge base: {e}")
            return None
    return None


def save_knowledge_base(reference_model, knowledge_base_path):
    """Save current knowledge base to JSON."""
    try:
        # Create copy to avoid modifying original
        save_data = reference_model.copy()
        
        # Convert archetype image numpy array to list for JSON
        if isinstance(save_data.get('archetype_image'), np.ndarray):
            save_data['archetype_image'] = save_data['archetype_image'].tolist()
        
        # Convert statistical model numpy arrays to lists
        if save_data.get('statistical_model'):
            # Iterate through array fields
            for key in ['mean', 'std', 'median', 'robust_mean', 'robust_cov', 'robust_inv_cov']:
                # Check if field is numpy array
                if key in save_data['statistical_model'] and isinstance(save_data['statistical_model'][key], np.ndarray):
                    # Convert to list
                    save_data['statistical_model'][key] = save_data['statistical_model'][key].tolist()
        
        # Remove large comparison scores if present to reduce file size
        if 'comparison_scores' in save_data:
            del save_data['comparison_scores']
        
        # Update timestamp to current time
        save_data['timestamp'] = get_timestamp()
        
        # Write to JSON file
        with open(knowledge_base_path, 'w') as f:
            json.dump(save_data, f, indent=2, cls=NumpyEncoder)
        # Log successful save
        logging.info(f"Knowledge base saved to {knowledge_base_path}")
        return True
    except Exception as e:
        # Log error if save fails
        logging.error(f"Error saving knowledge base: {e}")
        return False


def build_minimal_reference(image_path):
    """Build a minimal reference model from a single image"""
    # Log the operation
    logging.info("Building minimal reference model from current image...")
    
    # Load the image from path
    image = load_image(image_path)
    # Check if load succeeded
    if image is None:
        return None
    
    # Extract comprehensive features from image
    features, feature_names = extract_ultra_comprehensive_features(image)
    
    # Convert feature dictionary to numpy array in consistent order
    feature_vector = np.array([features[fname] for fname in feature_names])
    
    # Create minimal statistical model with assumed variance
    reference_model = {
        'features': [features],                    # Single feature set
        'feature_names': feature_names,            # Feature name list
        'statistical_model': {
            'mean': feature_vector,                # Use single sample as mean
            'std': np.ones_like(feature_vector) * 0.1,  # Assume 10% standard deviation
            'median': feature_vector,              # Single sample is also median
            'robust_mean': feature_vector,         # Use as robust mean too
            'robust_cov': np.eye(len(feature_vector)),     # Identity covariance matrix
            'robust_inv_cov': np.eye(len(feature_vector)),  # Identity inverse covariance
            'n_samples': 1,                        # Only one sample
        },
        # Convert to grayscale if needed and store as archetype
        'archetype_image': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
        'learned_thresholds': {                    # Default thresholds
            'anomaly_mean': 1.0,                   # Mean anomaly score
            'anomaly_std': 0.5,                    # Std of anomaly scores
            'anomaly_p90': 1.5,                    # 90th percentile
            'anomaly_p95': 2.0,                    # 95th percentile
            'anomaly_p99': 3.0,                    # 99th percentile
            'anomaly_threshold': 2.5,              # Final threshold
        },
        'timestamp': get_timestamp(),              # Creation time
    }
    
    return reference_model 