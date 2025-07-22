"""
Machine Learning Anomaly Detection Module

This module implements multiple machine learning approaches for detecting defects
in fiber optic end face images, including Isolation Forest, One-Class SVM, and
DBSCAN clustering for unsupervised anomaly detection.

Extracted from defect_detection2.py comprehensive detection system.
"""

import numpy as np
import cv2
from scipy import stats, ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage import morphology, feature
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def extract_pixel_features(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Extract comprehensive features for each pixel in the image.
    
    Args:
        image: Input grayscale image
        window_size: Size of local window for feature extraction (default: 5)
    
    Returns:
        Feature array of shape (height, width, num_features)
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    height, width = image.shape
    pad_size = window_size // 2
    
    # Pad image for boundary handling
    padded = np.pad(image, pad_size, mode='reflect')
    
    # Initialize feature array
    num_features = 11  # Number of features per pixel
    features = np.zeros((height, width, num_features))
    
    print(f"Extracting {num_features} features per pixel with window size {window_size}...")
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            # Get local window (padded coordinates)
            pi, pj = i + pad_size, j + pad_size
            window = padded[pi-pad_size:pi+pad_size+1, pj-pad_size:pj+pad_size+1]
            
            # Feature 0: Original intensity
            features[i, j, 0] = padded[pi, pj]
            
            # Feature 1: Local mean
            features[i, j, 1] = np.mean(window)
            
            # Feature 2: Local standard deviation
            features[i, j, 2] = np.std(window)
            
            # Feature 3: Local range (max - min)
            features[i, j, 3] = np.max(window) - np.min(window)
            
            # Feature 4: Local skewness
            flat_window = window.flatten()
            if len(flat_window) > 1 and np.std(flat_window) > 0:
                features[i, j, 4] = stats.skew(flat_window)
            else:
                features[i, j, 4] = 0.0
            
            # Feature 5: Local kurtosis
            if len(flat_window) > 1 and np.std(flat_window) > 0:
                features[i, j, 5] = stats.kurtosis(flat_window)
            else:
                features[i, j, 5] = 0.0
            
            # Feature 6: Local 25th percentile
            features[i, j, 6] = np.percentile(window, 25)
            
            # Feature 7: Local 75th percentile
            features[i, j, 7] = np.percentile(window, 75)
            
            # Feature 8: Gradient magnitude (horizontal)
            if pj > 0 and pj < padded.shape[1] - 1:
                dx = padded[pi, pj+1] - padded[pi, pj-1]
            else:
                dx = 0
            
            # Feature 9: Gradient magnitude (vertical)
            if pi > 0 and pi < padded.shape[0] - 1:
                dy = padded[pi+1, pj] - padded[pi-1, pj]
            else:
                dy = 0
            
            # Feature 10: Gradient magnitude
            features[i, j, 8] = dx
            features[i, j, 9] = dy
            features[i, j, 10] = np.sqrt(dx**2 + dy**2)
    
    print(f"Feature extraction completed. Shape: {features.shape}")
    return features


def ml_anomaly_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                        window_size: int = 5, 
                        isolation_contamination: float = 0.1,
                        svm_nu: float = 0.1,
                        dbscan_eps: float = 0.5,
                        dbscan_min_samples: int = 5,
                        use_pca: bool = True,
                        pca_components: int = 8) -> Dict:
    """
    Detect anomalies using multiple machine learning approaches.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask to limit analysis region
        window_size: Size of local window for feature extraction
        isolation_contamination: Expected fraction of outliers for Isolation Forest
        svm_nu: Nu parameter for One-Class SVM (upper bound on training errors)
        dbscan_eps: Maximum distance between samples in DBSCAN
        dbscan_min_samples: Minimum samples in neighborhood for DBSCAN
        use_pca: Apply PCA for dimensionality reduction
        pca_components: Number of PCA components to keep
    
    Returns:
        Dictionary containing detection results from all methods
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D)")
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    print("Starting ML-based anomaly detection...")
    
    # Extract features
    feature_array = extract_pixel_features(image, window_size)
    height, width, num_features = feature_array.shape
    
    # Get valid pixel features (within mask)
    valid_indices = np.where(mask)
    valid_features = feature_array[valid_indices]
    
    if len(valid_features) < max(dbscan_min_samples, 10):
        print("Warning: Insufficient valid pixels for ML analysis")
        return {
            'isolation_forest': np.zeros_like(mask, dtype=bool),
            'one_class_svm': np.zeros_like(mask, dtype=bool),
            'dbscan_outliers': np.zeros_like(mask, dtype=bool),
            'combined': np.zeros_like(mask, dtype=bool),
            'statistics': {'error': 'Insufficient data'}
        }
    
    print(f"Processing {len(valid_features)} valid pixels with {num_features} features each")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(valid_features)
    
    # Apply PCA if requested
    if use_pca and num_features > pca_components:
        pca = PCA(n_components=pca_components)
        pca_features = pca.fit_transform(scaled_features)
        print(f"Applied PCA: {num_features} -> {pca_components} dimensions")
        print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f}")
        final_features = pca_features
    else:
        final_features = scaled_features
        pca = None
    
    # Initialize result masks
    isolation_mask = np.zeros_like(mask, dtype=bool)
    svm_mask = np.zeros_like(mask, dtype=bool)
    dbscan_mask = np.zeros_like(mask, dtype=bool)
    
    # 1. Isolation Forest
    print("Running Isolation Forest...")
    try:
        iso_forest = IsolationForest(
            contamination=isolation_contamination,
            random_state=42,
            n_estimators=100
        )
        iso_labels = iso_forest.fit_predict(final_features)
        
        # Map back to image space
        anomaly_indices = np.where(iso_labels == -1)[0]
        for idx in anomaly_indices:
            y, x = valid_indices[0][idx], valid_indices[1][idx]
            isolation_mask[y, x] = True
        
        iso_count = np.sum(isolation_mask)
        print(f"  Isolation Forest detected {iso_count} anomalies")
        
    except Exception as e:
        print(f"  Isolation Forest failed: {e}")
        iso_count = 0
    
    # 2. One-Class SVM
    print("Running One-Class SVM...")
    try:
        ocsvm = OneClassSVM(gamma='scale', nu=svm_nu)
        svm_labels = ocsvm.fit_predict(final_features)
        
        # Map back to image space
        anomaly_indices = np.where(svm_labels == -1)[0]
        for idx in anomaly_indices:
            y, x = valid_indices[0][idx], valid_indices[1][idx]
            svm_mask[y, x] = True
        
        svm_count = np.sum(svm_mask)
        print(f"  One-Class SVM detected {svm_count} anomalies")
        
    except Exception as e:
        print(f"  One-Class SVM failed: {e}")
        svm_count = 0
    
    # 3. DBSCAN Clustering
    print("Running DBSCAN clustering...")
    try:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(final_features)
        
        # Consider noise points (label == -1) as anomalies
        anomaly_indices = np.where(cluster_labels == -1)[0]
        for idx in anomaly_indices:
            y, x = valid_indices[0][idx], valid_indices[1][idx]
            dbscan_mask[y, x] = True
        
        dbscan_count = np.sum(dbscan_mask)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"  DBSCAN found {num_clusters} clusters and {dbscan_count} outliers")
        
    except Exception as e:
        print(f"  DBSCAN failed: {e}")
        dbscan_count = 0
        num_clusters = 0
    
    # Combine results using voting
    print("Combining ML detection results...")
    
    # Create voting mask (pixel is anomaly if detected by at least 2 methods)
    vote_count = isolation_mask.astype(int) + svm_mask.astype(int) + dbscan_mask.astype(int)
    combined_mask = vote_count >= 2  # At least 2 methods agree
    
    # Post-processing
    combined_mask = morphology.remove_small_objects(combined_mask, min_size=3)
    combined_mask = morphology.opening(combined_mask, morphology.disk(1))
    
    combined_count = np.sum(combined_mask)
    
    # Compile statistics
    total_pixels = np.sum(mask)
    statistics = {
        'total_pixels': total_pixels,
        'window_size': window_size,
        'num_features': num_features,
        'use_pca': use_pca,
        'pca_components': pca_components if use_pca else num_features,
        'isolation_forest': {
            'count': iso_count,
            'percentage': (iso_count / total_pixels) * 100 if total_pixels > 0 else 0,
            'contamination': isolation_contamination
        },
        'one_class_svm': {
            'count': svm_count,
            'percentage': (svm_count / total_pixels) * 100 if total_pixels > 0 else 0,
            'nu': svm_nu
        },
        'dbscan': {
            'count': dbscan_count,
            'percentage': (dbscan_count / total_pixels) * 100 if total_pixels > 0 else 0,
            'num_clusters': num_clusters,
            'eps': dbscan_eps,
            'min_samples': dbscan_min_samples
        },
        'combined': {
            'count': combined_count,
            'percentage': (combined_count / total_pixels) * 100 if total_pixels > 0 else 0,
            'voting_threshold': 2
        }
    }
    
    if pca is not None:
        statistics['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
    
    print(f"\nML Anomaly Detection Results:")
    print(f"  Isolation Forest: {iso_count} anomalies ({statistics['isolation_forest']['percentage']:.2f}%)")
    print(f"  One-Class SVM: {svm_count} anomalies ({statistics['one_class_svm']['percentage']:.2f}%)")
    print(f"  DBSCAN: {dbscan_count} anomalies ({statistics['dbscan']['percentage']:.2f}%)")
    print(f"  Combined (2+ votes): {combined_count} anomalies ({statistics['combined']['percentage']:.2f}%)")
    
    return {
        'isolation_forest': isolation_mask,
        'one_class_svm': svm_mask,
        'dbscan_outliers': dbscan_mask,
        'combined': combined_mask,
        'feature_array': feature_array,
        'statistics': statistics
    }


def advanced_ml_detection(image: np.ndarray, mask: Optional[np.ndarray] = None,
                         multi_scale: bool = True,
                         window_sizes: List[int] = [3, 5, 7],
                         ensemble_voting: bool = True,
                         adaptive_parameters: bool = True) -> Dict:
    """
    Advanced ML detection with multi-scale analysis and ensemble methods.
    
    Args:
        image: Input grayscale image
        mask: Optional binary mask
        multi_scale: Use multiple window sizes for feature extraction
        window_sizes: List of window sizes to use
        ensemble_voting: Use ensemble voting across scales
        adaptive_parameters: Adapt ML parameters based on image properties
    
    Returns:
        Enhanced ML detection results
    """
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    if not multi_scale:
        # Single scale detection with default parameters
        return ml_anomaly_detection(image, mask)
    
    print("Performing advanced multi-scale ML detection...")
    
    # Adaptive parameter selection based on image properties
    if adaptive_parameters:
        # Analyze image noise and contrast
        masked_pixels = image[mask]
        if len(masked_pixels) > 0:
            noise_level = np.std(masked_pixels) / np.mean(masked_pixels)
            
            # Adapt contamination based on noise level
            base_contamination = 0.05 + min(0.10, noise_level * 0.5)
            
            # Adapt SVM nu parameter
            base_nu = 0.05 + min(0.15, noise_level * 0.3)
            
            print(f"Adaptive parameters: contamination={base_contamination:.3f}, nu={base_nu:.3f}")
        else:
            base_contamination = 0.1
            base_nu = 0.1
    else:
        base_contamination = 0.1
        base_nu = 0.1
    
    # Store results for each scale
    scale_results = []
    combined_votes = np.zeros_like(image, dtype=int)
    
    for window_size in window_sizes:
        print(f"\nProcessing window size {window_size}...")
        
        # Scale-dependent parameter adjustment
        if adaptive_parameters:
            # Larger windows might need more conservative thresholds
            scale_factor = window_size / 5.0  # Normalize to window_size=5
            contamination = base_contamination * (0.8 + 0.4 / scale_factor)
            nu = base_nu * (0.8 + 0.4 / scale_factor)
        else:
            contamination = base_contamination
            nu = base_nu
        
        result = ml_anomaly_detection(
            image, mask,
            window_size=window_size,
            isolation_contamination=contamination,
            svm_nu=nu,
            use_pca=True,
            pca_components=min(8, window_size * 2)
        )
        
        scale_results.append({
            'window_size': window_size,
            'contamination': contamination,
            'nu': nu,
            'result': result
        })
        
        if ensemble_voting:
            # Add votes from this scale
            combined_votes += result['isolation_forest'].astype(int)
            combined_votes += result['one_class_svm'].astype(int)
            combined_votes += result['dbscan_outliers'].astype(int)
    
    if ensemble_voting:
        # Ensemble voting across scales and methods
        max_votes = len(window_sizes) * 3  # 3 methods per scale
        vote_threshold = max(2, max_votes // 3)  # At least 1/3 of votes
        
        ensemble_mask = combined_votes >= vote_threshold
        
        # Advanced post-processing
        ensemble_mask = morphology.remove_small_objects(ensemble_mask, min_size=5)
        ensemble_mask = morphology.opening(ensemble_mask, morphology.disk(1))
        
        ensemble_count = np.sum(ensemble_mask)
        
        print(f"\nEnsemble voting results:")
        print(f"  Vote threshold: {vote_threshold}/{max_votes}")
        print(f"  Ensemble anomalies: {ensemble_count}")
        print(f"  Ensemble percentage: {(ensemble_count / np.sum(mask)) * 100:.2f}%")
    else:
        # Simple combination of final scales
        ensemble_mask = np.zeros_like(image, dtype=bool)
        for scale_result in scale_results:
            ensemble_mask |= scale_result['result']['combined']
        ensemble_count = np.sum(ensemble_mask)
    
    # Comprehensive statistics
    comprehensive_stats = {
        'multi_scale': True,
        'window_sizes': window_sizes,
        'ensemble_voting': ensemble_voting,
        'adaptive_parameters': adaptive_parameters,
        'ensemble_count': ensemble_count,
        'ensemble_percentage': (ensemble_count / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0,
        'scale_results': scale_results,
        'vote_distribution': np.bincount(combined_votes.flatten()) if ensemble_voting else None
    }
    
    return {
        'ensemble': ensemble_mask,
        'vote_map': combined_votes if ensemble_voting else None,
        'scale_results': scale_results,
        'statistics': comprehensive_stats
    }


def visualize_ml_results(image: np.ndarray, results: Dict, save_path: Optional[str] = None):
    """
    Visualize ML anomaly detection results.
    
    Args:
        image: Original input image
        results: Results from ml_anomaly_detection or advanced_ml_detection
        save_path: Optional path to save visualization
    """
    if 'scale_results' in results:
        # Multi-scale results
        num_scales = len(results['scale_results'])
        fig, axes = plt.subplots(3, num_scales + 1, figsize=(4 * (num_scales + 1), 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ensemble result
        axes[1, 0].imshow(results['ensemble'], cmap='hot')
        axes[1, 0].set_title('Ensemble Detection')
        axes[1, 0].axis('off')
        
        # Vote map
        if 'vote_map' in results and results['vote_map'] is not None:
            axes[2, 0].imshow(results['vote_map'], cmap='viridis')
            axes[2, 0].set_title('Vote Map')
            axes[2, 0].axis('off')
        
        # Individual scale results
        for i, scale_result in enumerate(results['scale_results']):
            window_size = scale_result['window_size']
            result = scale_result['result']
            
            # Show combined result for each scale
            axes[0, i + 1].imshow(result['combined'], cmap='hot')
            axes[0, i + 1].set_title(f'Combined (W={window_size})')
            axes[0, i + 1].axis('off')
            
            # Show individual methods
            methods = ['isolation_forest', 'one_class_svm', 'dbscan_outliers']
            for j, method in enumerate(methods):
                if j < 2:  # Only show first 2 methods due to space
                    axes[j+1, i + 1].imshow(result[method], cmap='hot')
                    axes[j+1, i + 1].set_title(f'{method.replace("_", " ").title()} (W={window_size})')
                    axes[j+1, i + 1].axis('off')
    
    else:
        # Single scale results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['isolation_forest'], cmap='hot')
        axes[0, 1].set_title('Isolation Forest')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['one_class_svm'], cmap='hot')
        axes[0, 2].set_title('One-Class SVM')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(results['dbscan_outliers'], cmap='hot')
        axes[1, 0].set_title('DBSCAN Outliers')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['combined'], cmap='hot')
        axes[1, 1].set_title('Combined Detection')
        axes[1, 1].axis('off')
        
        # Feature visualization (first 3 features)
        if 'feature_array' in results:
            feature_composite = np.stack([
                results['feature_array'][:, :, 0],  # Original intensity
                results['feature_array'][:, :, 1],  # Local mean
                results['feature_array'][:, :, 2]   # Local std
            ], axis=2)
            feature_composite = (feature_composite - feature_composite.min()) / (feature_composite.max() - feature_composite.min())
            axes[1, 2].imshow(feature_composite)
            axes[1, 2].set_title('Feature Composite (RGB)')
            axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


# Demo and test code
if __name__ == "__main__":
    print("Machine Learning Anomaly Detection Module - Demo")
    print("=" * 50)
    
    # Create synthetic test image with various defects
    print("Creating synthetic test image with defects...")
    np.random.seed(42)  # For reproducible results
    
    test_image = np.random.randint(120, 140, (200, 200), dtype=np.uint8)
    
    # Add different types of defects
    # Bright circular defects (digs)
    cv2.circle(test_image, (50, 50), 8, 200, -1)
    cv2.circle(test_image, (150, 100), 6, 210, -1)
    
    # Dark circular defects
    cv2.circle(test_image, (100, 150), 7, 80, -1)
    cv2.circle(test_image, (170, 170), 5, 70, -1)
    
    # Linear scratches
    cv2.line(test_image, (20, 80), (80, 120), 90, 2)
    cv2.line(test_image, (120, 30), (160, 70), 85, 2)
    
    # Irregular defect
    test_image[160:170, 40:55] = 60
    
    # Add noise
    noise = np.random.normal(0, 3, test_image.shape)
    test_image = np.clip(test_image.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create circular mask
    center = (100, 100)
    radius = 90
    y, x = np.ogrid[:200, :200]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    print(f"Test image shape: {test_image.shape}")
    print(f"Added various defect types")
    print(f"Mask coverage: {np.sum(mask)} pixels")
    
    # Test single-scale ML detection
    print("\n1. Testing single-scale ML detection...")
    single_results = ml_anomaly_detection(
        test_image, mask,
        window_size=5,
        isolation_contamination=0.1,
        svm_nu=0.1,
        use_pca=True
    )
    
    # Test advanced multi-scale ML detection
    print("\n2. Testing advanced multi-scale ML detection...")
    multi_results = advanced_ml_detection(
        test_image, mask,
        multi_scale=True,
        window_sizes=[3, 5, 7],
        ensemble_voting=True,
        adaptive_parameters=True
    )
    
    # Visualize results
    print("\n3. Visualizing results...")
    
    # Single scale visualization
    fig1 = visualize_ml_results(test_image, single_results)
    plt.suptitle('Single-Scale ML Anomaly Detection', fontsize=16)
    
    # Multi-scale visualization
    fig2 = visualize_ml_results(test_image, multi_results)
    plt.suptitle('Multi-Scale ML Anomaly Detection', fontsize=16)
    
    # Performance comparison
    print("\n4. Performance comparison:")
    single_combined = single_results['statistics']['combined']['count']
    multi_ensemble = multi_results['statistics']['ensemble_count']
    
    print(f"Single-scale combined: {single_combined} anomalies")
    print(f"Multi-scale ensemble: {multi_ensemble} anomalies")
    print(f"Improvement: {multi_ensemble - single_combined} additional anomalies")
    
    # Method comparison
    print("\n5. Method breakdown (single-scale):")
    stats = single_results['statistics']
    print(f"  Isolation Forest: {stats['isolation_forest']['count']} ({stats['isolation_forest']['percentage']:.1f}%)")
    print(f"  One-Class SVM: {stats['one_class_svm']['count']} ({stats['one_class_svm']['percentage']:.1f}%)")
    print(f"  DBSCAN: {stats['dbscan']['count']} ({stats['dbscan']['percentage']:.1f}%)")
    print(f"  Combined: {stats['combined']['count']} ({stats['combined']['percentage']:.1f}%)")
    
    print("\nDemo completed successfully!")
