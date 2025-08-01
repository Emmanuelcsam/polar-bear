#!/usr/bin/env python3

import numpy as np
from typing import Dict


def extract_lbp_features(gray: np.ndarray) -> Dict[str, float]:
    """Extract Local Binary Pattern features."""
    # Compute LBP for different neighborhood sizes
    lbp_3x3 = compute_lbp(gray, 3)
    lbp_5x5 = compute_lbp(gray, 5)
    
    # Compute histograms
    hist_3x3 = compute_lbp_histogram(lbp_3x3)
    hist_5x5 = compute_lbp_histogram(lbp_5x5)
    
    # Compute statistical measures from histograms
    features = {}
    
    # 3x3 LBP features
    features.update(compute_histogram_stats(hist_3x3, 'lbp_3x3'))
    
    # 5x5 LBP features
    features.update(compute_histogram_stats(hist_5x5, 'lbp_5x5'))
    
    # Uniform LBP features (patterns with at most 2 transitions)
    uniform_3x3 = compute_uniform_lbp(gray, 3)
    uniform_5x5 = compute_uniform_lbp(gray, 5)
    
    features['lbp_uniform_3x3'] = float(np.sum(uniform_3x3) / uniform_3x3.size)
    features['lbp_uniform_5x5'] = float(np.sum(uniform_5x5) / uniform_5x5.size)
    
    return features


def compute_lbp(image: np.ndarray, radius: int) -> np.ndarray:
    """Compute Local Binary Pattern for given radius."""
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint8)
    
    # Define neighborhood coordinates
    neighbors = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i != 0 or j != 0:  # Skip center pixel
                neighbors.append((i, j))
    
    # Compute LBP for each pixel
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]
            pattern = 0
            
            for i, (dy, dx) in enumerate(neighbors):
                if i < 8:  # Use only 8 neighbors for standard LBP
                    neighbor = image[y + dy, x + dx]
                    if neighbor >= center:
                        pattern |= (1 << i)
            
            lbp[y, x] = pattern
    
    return lbp


def compute_uniform_lbp(image: np.ndarray, radius: int) -> np.ndarray:
    """Compute uniform LBP patterns (at most 2 transitions)."""
    lbp = compute_lbp(image, radius)
    uniform = np.zeros_like(lbp)
    
    # Define uniform patterns (0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31)
    uniform_patterns = {0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31}
    
    for y in range(lbp.shape[0]):
        for x in range(lbp.shape[1]):
            pattern = lbp[y, x]
            if pattern in uniform_patterns:
                uniform[y, x] = 1
    
    return uniform


def compute_lbp_histogram(lbp: np.ndarray) -> np.ndarray:
    """Compute histogram of LBP patterns."""
    # LBP patterns range from 0 to 255 (8-bit)
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return hist


def compute_histogram_stats(hist: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute statistical measures from histogram."""
    # Normalize histogram
    hist_norm = hist / (hist.sum() + 1e-10)
    
    # Compute statistics
    non_zero = hist_norm[hist_norm > 0]
    
    return {
        f'{prefix}_mean': float(np.mean(hist)),
        f'{prefix}_std': float(np.std(hist)),
        f'{prefix}_entropy': float(-np.sum(non_zero * np.log2(non_zero + 1e-10))),
        f'{prefix}_energy': float(np.sum(hist_norm**2)),
        f'{prefix}_max': float(np.max(hist)),
        f'{prefix}_min': float(np.min(hist)),
    }


def test_lbp_features():
    """Test function for LBP feature extraction."""
    # Create test image with patterns
    test_image = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some patterns
    test_image[20:80, 20:80] = 128  # Gray square
    test_image[30:70, 30:70] = 255  # White square
    test_image[40:60, 40:60] = 0    # Black square
    
    # Add noise
    noise = np.random.normal(0, 20, test_image.shape).astype(np.uint8)
    test_image = np.clip(test_image + noise, 0, 255)
    
    # Extract features
    features = extract_lbp_features(test_image)
    
    # Print results
    print("LBP Features Extracted:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    return features


if __name__ == "__main__":
    test_lbp_features() 