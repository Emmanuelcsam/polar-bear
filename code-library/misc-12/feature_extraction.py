#!/usr/bin/env python3

import cv2
import numpy as np
import logging
from utils import sanitize_feature_value
from statistical_functions import (
    compute_skewness, compute_kurtosis, compute_entropy
)


def extract_ultra_comprehensive_features(image):
    """Extract 100+ features using all available methods."""
    # Initialize empty feature dictionary
    features = {}
    
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Log feature extraction start
    logging.info("  Extracting features...")
    
    # Define list of feature extraction methods with names
    feature_extractors = [
        ("Stats", extract_statistical_features),         # Basic statistics
        ("Norms", extract_matrix_norms),                # Matrix norms
        ("LBP", extract_lbp_features),                  # Local Binary Patterns
        ("GLCM", extract_glcm_features),                # Gray-Level Co-occurrence
        ("FFT", extract_fourier_features),              # Fourier Transform
        ("MultiScale", extract_multiscale_features),    # Multi-scale analysis
        ("Morph", extract_morphological_features),      # Morphological features
        ("Shape", extract_shape_features),              # Shape descriptors
        ("SVD", extract_svd_features),                  # Singular Value Decomposition
        ("Entropy", extract_entropy_features),          # Entropy measures
        ("Gradient", extract_gradient_features),        # Gradient features
        ("Topology", extract_topological_proxy_features), # Topological features
    ]
    
    # Execute each feature extractor
    for name, extractor in feature_extractors:
        try:
            # Update features dictionary with new features
            features.update(extractor(gray))
        except Exception as e:
            # Log warning if extraction fails
            logging.warning(f"Feature extraction failed for {name}: {e}")
    
    # Create new dictionary with sanitized values
    sanitized_features = {}
    # Sanitize each feature value to ensure finite numbers
    for key, value in features.items():
        sanitized_features[key] = sanitize_feature_value(value)
    
    # Get sorted list of feature names for consistent ordering
    feature_names = sorted(sanitized_features.keys())
    return sanitized_features, feature_names


def extract_statistical_features(gray):
    """Extract comprehensive statistical features."""
    # Flatten 2D image to 1D array for statistics
    flat = gray.flatten()
    # Calculate percentiles at 10, 25, 50, 75, 90
    percentiles = np.percentile(gray, [10, 25, 50, 75, 90])
    
    return {
        'stat_mean': float(np.mean(gray)),                    # Average pixel value
        'stat_std': float(np.std(gray)),                      # Standard deviation
        'stat_variance': float(np.var(gray)),                 # Variance
        'stat_skew': float(compute_skewness(flat)),          # Distribution skewness
        'stat_kurtosis': float(compute_kurtosis(flat)),       # Distribution kurtosis
        'stat_min': float(np.min(gray)),                      # Minimum value
        'stat_max': float(np.max(gray)),                      # Maximum value
        'stat_range': float(np.max(gray) - np.min(gray)),     # Value range
        'stat_median': float(np.median(gray)),                # Median value
        'stat_mad': float(np.median(np.abs(gray - np.median(gray)))), # Median absolute deviation
        'stat_iqr': float(percentiles[3] - percentiles[1]),   # Interquartile range
        'stat_entropy': float(compute_entropy(gray)),         # Information entropy
        'stat_energy': float(np.sum(gray**2)),                # Energy (sum of squares)
        'stat_p10': float(percentiles[0]),                    # 10th percentile
        'stat_p25': float(percentiles[1]),                    # 25th percentile
        'stat_p50': float(percentiles[2]),                    # 50th percentile (median)
        'stat_p75': float(percentiles[3]),                    # 75th percentile
        'stat_p90': float(percentiles[4]),                    # 90th percentile
    }


def extract_matrix_norms(gray):
    """Extract various matrix norms."""
    return {
        'norm_frobenius': float(np.linalg.norm(gray, 'fro')), # Frobenius norm
        'norm_l1': float(np.sum(np.abs(gray))),               # L1 norm
        'norm_l2': float(np.sqrt(np.sum(gray**2))),           # L2 norm
        'norm_linf': float(np.max(np.abs(gray))),             # L-infinity norm
        'norm_nuclear': float(np.sum(np.linalg.svd(gray, compute_uv=False))), # Nuclear norm
        'norm_trace': float(np.trace(gray)),                  # Trace
    }


def extract_lbp_features(gray):
    """Extract Local Binary Pattern features using custom implementation."""
    # Initialize feature dictionary
    features = {}
    
    # Compute LBP at multiple radii
    for radius in [1, 2, 3, 5]:
        # Initialize LBP result array
        lbp = np.zeros_like(gray, dtype=np.float32)
        
        # Iterate through neighborhood offsets
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Skip center pixel
                if dx == 0 and dy == 0:
                    continue
                
                # Create shifted version of image
                shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                
                # Compare shifted with original (binary pattern)
                lbp += (shifted >= gray).astype(np.float32)
        
        # Compute statistics of LBP
        features[f'lbp_r{radius}_mean'] = float(np.mean(lbp))
        features[f'lbp_r{radius}_std'] = float(np.std(lbp))
        features[f'lbp_r{radius}_entropy'] = float(compute_entropy(lbp))
        features[f'lbp_r{radius}_energy'] = float(np.sum(lbp**2) / lbp.size)
    
    return features


def extract_glcm_features(gray):
    """Extract Gray-Level Co-occurrence Matrix features using custom implementation."""
    # Quantize image to 8 levels for faster computation
    img_q = (gray // 32).astype(np.uint8)
    levels = 8
    
    # Initialize feature dictionary
    features = {}
    # Define distances and angles for GLCM
    distances = [1, 2, 3]
    angles = [0, 45, 90, 135]  # degrees
    
    # Compute GLCM for each distance and angle
    for dist in distances:
        for angle in angles:
            # Initialize GLCM matrix
            glcm = np.zeros((levels, levels), dtype=np.float32)
            
            # Determine pixel offset based on angle
            if angle == 0:
                dy, dx = 0, dist       # Horizontal
            elif angle == 45:
                dy, dx = -dist, dist   # Diagonal up-right
            elif angle == 90:
                dy, dx = -dist, 0      # Vertical
            else:  # 135
                dy, dx = -dist, -dist  # Diagonal up-left
            
            # Build GLCM by counting co-occurrences
            rows, cols = img_q.shape
            for i in range(rows):
                for j in range(cols):
                    # Check if neighbor is within bounds
                    if 0 <= i + dy < rows and 0 <= j + dx < cols:
                        # Increment co-occurrence count
                        glcm[img_q[i, j], img_q[i + dy, j + dx]] += 1
            
            # Normalize GLCM to probabilities
            glcm = glcm / (glcm.sum() + 1e-10)
            
            # Compute GLCM properties
            # Contrast: measure of local variations
            contrast = 0
            for i in range(levels):
                for j in range(levels):
                    contrast += ((i - j) ** 2) * glcm[i, j]
            
            # Energy: measure of uniformity
            energy = np.sum(glcm ** 2)
            
            # Homogeneity: measure of closeness to diagonal
            homogeneity = 0
            for i in range(levels):
                for j in range(levels):
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
            
            # Store features with descriptive names
            features[f'glcm_d{dist}_a{angle}_contrast'] = float(contrast)
            features[f'glcm_d{dist}_a{angle}_energy'] = float(energy)
            features[f'glcm_d{dist}_a{angle}_homogeneity'] = float(homogeneity)
    
    return features


def extract_fourier_features(gray):
    """Extract 2D Fourier Transform features."""
    # Compute 2D FFT
    f = np.fft.fft2(gray)
    # Shift zero frequency to center
    fshift = np.fft.fftshift(f)
    # Compute magnitude spectrum
    magnitude = np.abs(fshift)
    # Compute power spectrum
    power = magnitude**2
    # Compute phase spectrum
    phase = np.angle(fshift)
    
    # Calculate center coordinates
    center = np.array(power.shape) // 2
    # Create coordinate grids
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    # Compute distance from center for each pixel
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    # Compute radial profile (average power at each radius)
    radial_prof = []
    for radius in range(1, min(center)):
        # Create ring mask
        mask = (r >= radius - 1) & (r < radius)
        # Average power in ring
        if mask.any():
            radial_prof.append(np.mean(power[mask]))
    
    # Convert to array
    radial_prof = np.array(radial_prof)
    
    # Compute spectral centroid and spread if profile exists
    if len(radial_prof) > 0:
        # Weighted average of frequencies
        spectral_centroid = float(np.sum(np.arange(len(radial_prof)) * radial_prof) / (np.sum(radial_prof) + 1e-10))
        # Weighted standard deviation of frequencies
        spectral_spread = float(np.sqrt(np.sum((np.arange(len(radial_prof)) - spectral_centroid)**2 * radial_prof) / (np.sum(radial_prof) + 1e-10)))
    else:
        spectral_centroid = 0.0
        spectral_spread = 0.0
    
    return {
        'fft_mean_magnitude': float(np.mean(magnitude)),
        'fft_std_magnitude': float(np.std(magnitude)),
        'fft_max_magnitude': float(np.max(magnitude)),
        'fft_total_power': float(np.sum(power)),
        'fft_dc_component': float(magnitude[center[0], center[1]]),
        'fft_mean_phase': float(np.mean(phase)),
        'fft_std_phase': float(np.std(phase)),
        'fft_spectral_centroid': spectral_centroid,
        'fft_spectral_spread': spectral_spread,
    }


def extract_multiscale_features(gray):
    """Extract multi-scale features using Gaussian pyramids."""
    # Initialize feature dictionary
    features = {}
    
    # Create Gaussian pyramid (progressively downsampled versions)
    pyramid = [gray]
    for i in range(3):
        # Downsample by factor of 2
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    
    # Compute features at each scale
    for level, img in enumerate(pyramid):
        # Basic statistics at this scale
        features[f'pyramid_L{level}_mean'] = float(np.mean(img))
        features[f'pyramid_L{level}_std'] = float(np.std(img))
        features[f'pyramid_L{level}_energy'] = float(np.sum(img**2))
        
        # Compute difference between scales (detail information)
        if level > 0:
            # Upsample current level to match previous level size
            upsampled = cv2.pyrUp(img)
            # Get dimensions of previous level
            h, w = pyramid[level-1].shape
            # Resize to exact dimensions
            upsampled = cv2.resize(upsampled, (w, h))
            
            # Compute difference (approximates wavelet detail coefficients)
            detail = pyramid[level-1].astype(np.float32) - upsampled.astype(np.float32)
            
            # Statistics of detail coefficients
            features[f'pyramid_detail_L{level}_energy'] = float(np.sum(detail**2))
            features[f'pyramid_detail_L{level}_mean'] = float(np.mean(np.abs(detail)))
            features[f'pyramid_detail_L{level}_std'] = float(np.std(detail))
    
    # Laplacian pyramid features (edge information at multiple scales)
    for level in range(2):
        # Compute Laplacian (second derivative)
        laplacian = cv2.Laplacian(pyramid[level], cv2.CV_64F)
        features[f'laplacian_L{level}_energy'] = float(np.sum(laplacian**2))
        features[f'laplacian_L{level}_mean'] = float(np.mean(np.abs(laplacian)))
    
    return features


def extract_morphological_features(gray):
    """Extract morphological features."""
    # Initialize feature dictionary
    features = {}
    
    # Multi-scale morphological operations
    for size in [3, 5, 7, 11]:
        # Create circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        
        # White tophat: bright features smaller than kernel
        wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        # Black tophat: dark features smaller than kernel
        bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Statistics of tophat transforms
        features[f'morph_wth_{size}_mean'] = float(np.mean(wth))
        features[f'morph_wth_{size}_max'] = float(np.max(wth))
        features[f'morph_wth_{size}_sum'] = float(np.sum(wth))
        features[f'morph_bth_{size}_mean'] = float(np.mean(bth))
        features[f'morph_bth_{size}_max'] = float(np.max(bth))
        features[f'morph_bth_{size}_sum'] = float(np.sum(bth))
    
    # Binary morphology analysis
    # Otsu's threshold for automatic binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Define 5x5 square kernel
    kernel = np.ones((5, 5), np.uint8)
    # Erosion: shrinks white regions
    erosion = cv2.erode(binary, kernel, iterations=1)
    # Dilation: expands white regions
    dilation = cv2.dilate(binary, kernel, iterations=1)
    # Morphological gradient: difference between dilation and erosion
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    # Compute morphological statistics
    features['morph_binary_area_ratio'] = float(np.sum(binary) / binary.size)
    features['morph_gradient_sum'] = float(np.sum(gradient))
    features['morph_erosion_ratio'] = float(np.sum(erosion) / (np.sum(binary) + 1e-10))
    features['morph_dilation_ratio'] = float(np.sum(dilation) / (np.sum(binary) + 1e-10))
    
    return features


def extract_shape_features(gray):
    """Extract shape features using Hu moments."""
    # Calculate image moments (statistical measures)
    moments = cv2.moments(gray)
    # Compute 7 Hu moments (rotation invariant)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Initialize feature dictionary
    features = {}
    # Store log-transformed Hu moments for scale invariance
    for i, hu in enumerate(hu_moments):
        # Log transform with sign preservation
        features[f'shape_hu_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
    
    # Additional moment features
    if moments['m00'] > 0:  # Check for non-zero area
        # Calculate centroid coordinates
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        # Normalize centroid to [0,1] range
        features['shape_centroid_x'] = float(cx / gray.shape[1])
        features['shape_centroid_y'] = float(cy / gray.shape[0])
    
    return features


def extract_svd_features(gray):
    """Extract Singular Value Decomposition features."""
    try:
        # Compute SVD (decompose matrix into U*S*V^T)
        _, s, _ = np.linalg.svd(gray, full_matrices=False)
        # Normalize singular values
        s_norm = s / (np.sum(s) + 1e-10)
        
        # Compute cumulative energy
        cumsum = np.cumsum(s_norm)
        # Find components needed for 90% energy
        n_components_90 = np.argmax(cumsum >= 0.9) + 1
        # Find components needed for 95% energy
        n_components_95 = np.argmax(cumsum >= 0.95) + 1
        
        return {
            'svd_largest': float(s[0]) if len(s) > 0 else 0.0,
            'svd_top5_ratio': float(np.sum(s_norm[:5])) if len(s) >= 5 else float(np.sum(s_norm)),
            'svd_top10_ratio': float(np.sum(s_norm[:10])) if len(s) >= 10 else float(np.sum(s_norm)),
            'svd_entropy': float(compute_entropy(s_norm * 1000)),
            'svd_energy_ratio': float(s[0] / (s[1] + 1e-10)) if len(s) > 1 else 0.0,
            'svd_n_components_90': float(n_components_90),
            'svd_n_components_95': float(n_components_95),
            'svd_effective_rank': float(np.exp(compute_entropy(s_norm * 1000))),
        }
    except:
        # Return zeros if SVD fails
        return {f'svd_{k}': 0.0 for k in ['largest', 'top5_ratio', 'top10_ratio', 'entropy', 
                                           'energy_ratio', 'n_components_90', 'n_components_95', 'effective_rank']}


def extract_entropy_features(gray):
    """Extract various entropy measures."""
    # Compute global histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    # Normalize to probability distribution
    hist_norm = hist / (hist.sum() + 1e-10)
    
    # Shannon entropy (already computed)
    shannon = compute_entropy(gray)
    
    # Renyi entropy with parameter alpha = 2
    renyi = -np.log2(np.sum(hist_norm**2) + 1e-10)
    
    # Tsallis entropy with parameter q = 2
    tsallis = (1 - np.sum(hist_norm**2)) / 1
    
    # Local entropy computation
    # Define local window size
    kernel_size = 9
    # Create averaging kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    
    # Compute local statistics for entropy approximation
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    local_ent = np.log2(local_var + 1)
    
    return {
        'entropy_shannon': float(shannon),
        'entropy_renyi': float(renyi),
        'entropy_tsallis': float(tsallis),
        'entropy_local_mean': float(np.mean(local_ent)),
        'entropy_local_std': float(np.std(local_ent)),
        'entropy_local_max': float(np.max(local_ent)),
        'entropy_local_min': float(np.min(local_ent)),
    }


def extract_gradient_features(gray):
    """Extract gradient-based features."""
    # Compute Sobel gradients (first derivatives)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude (edge strength)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Compute gradient orientation (edge direction)
    grad_orient = np.arctan2(grad_y, grad_x)
    
    # Compute Laplacian (second derivative)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Detect edges using Canny algorithm
    edges = cv2.Canny(gray, 50, 150)
    # Calculate edge density
    edge_density = np.sum(edges) / edges.size
    
    return {
        'gradient_magnitude_mean': float(np.mean(grad_mag)),
        'gradient_magnitude_std': float(np.std(grad_mag)),
        'gradient_magnitude_max': float(np.max(grad_mag)),
        'gradient_magnitude_sum': float(np.sum(grad_mag)),
        'gradient_orientation_mean': float(np.mean(grad_orient)),
        'gradient_orientation_std': float(np.std(grad_orient)),
        'laplacian_mean': float(np.mean(np.abs(laplacian))),
        'laplacian_std': float(np.std(laplacian)),
        'laplacian_sum': float(np.sum(np.abs(laplacian))),
        'edge_density': float(edge_density),
        'edge_count': float(np.sum(edges > 0)),
    }


def extract_topological_proxy_features(gray):
    """Extract topological proxy features using connected components analysis."""
    # Initialize feature dictionary
    features = {}
    
    # Create threshold values from 5th to 95th percentile
    thresholds = np.percentile(gray, np.linspace(5, 95, 20))
    
    # Track connected components at each threshold (proxy for Betti 0)
    n_components = []
    for t in thresholds:
        # Create binary image at threshold
        binary = (gray >= t).astype(np.uint8)
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Store count (subtract 1 for background)
        n_components.append(num_labels - 1)
    
    # Track holes at each threshold (proxy for Betti 1)
    n_holes = []
    for t in thresholds:
        # Create inverted binary image
        binary = (gray < t).astype(np.uint8)
        # Find connected components in inverted image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Store count (subtract 1 for background)
        n_holes.append(num_labels - 1)
    
    # Compute statistics for connected components
    if len(n_components) > 1:
        # Compute persistence (changes between thresholds)
        persistence_b0 = np.diff(n_components)
        features['topo_b0_max_components'] = float(np.max(n_components))
        features['topo_b0_mean_components'] = float(np.mean(n_components))
        features['topo_b0_persistence_sum'] = float(np.sum(np.abs(persistence_b0)))
        features['topo_b0_persistence_max'] = float(np.max(np.abs(persistence_b0)))
    else:
        # Handle single threshold case
        features['topo_b0_max_components'] = float(n_components[0]) if n_components else 0.0
        features['topo_b0_mean_components'] = float(n_components[0]) if n_components else 0.0
        features['topo_b0_persistence_sum'] = 0.0
        features['topo_b0_persistence_max'] = 0.0
    
    # Compute statistics for holes
    if len(n_holes) > 1:
        # Compute persistence for holes
        persistence_b1 = np.diff(n_holes)
        features['topo_b1_max_holes'] = float(np.max(n_holes))
        features['topo_b1_mean_holes'] = float(np.mean(n_holes))
        features['topo_b1_persistence_sum'] = float(np.sum(np.abs(persistence_b1)))
        features['topo_b1_persistence_max'] = float(np.max(np.abs(persistence_b1)))
    else:
        # Handle single threshold case
        features['topo_b1_max_holes'] = float(n_holes[0]) if n_holes else 0.0
        features['topo_b1_mean_holes'] = float(n_holes[0]) if n_holes else 0.0
        features['topo_b1_persistence_sum'] = 0.0
        features['topo_b1_persistence_max'] = 0.0
    
    return features 