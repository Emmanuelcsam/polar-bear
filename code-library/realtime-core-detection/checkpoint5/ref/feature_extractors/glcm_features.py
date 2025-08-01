#!/usr/bin/env python3

import numpy as np
from typing import Dict, Tuple


def extract_glcm_features(gray: np.ndarray) -> Dict[str, float]:
    """Extract Gray-Level Co-occurrence Matrix features."""
    features = {}
    
    # Define different offsets for GLCM computation
    offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal
    
    for i, (dx, dy) in enumerate(offsets):
        # Compute GLCM for this offset
        glcm = compute_glcm(gray, dx, dy)
        
        # Compute features from GLCM
        glcm_features = compute_glcm_features(glcm)
        
        # Add prefix to feature names
        for key, value in glcm_features.items():
            features[f'glcm_{i}_{key}'] = value
    
    return features


def compute_glcm(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Compute Gray-Level Co-occurrence Matrix."""
    # Quantize image to reduce levels (0-255 -> 0-15)
    levels = 16
    quantized = (image // (256 // levels)).astype(np.uint8)
    
    # Initialize GLCM
    glcm = np.zeros((levels, levels), dtype=np.float64)
    
    height, width = quantized.shape
    
    # Count co-occurrences
    for y in range(height):
        for x in range(width):
            # Current pixel
            i = quantized[y, x]
            
            # Neighbor pixel
            ny, nx = y + dy, x + dx
            
            # Check bounds
            if 0 <= ny < height and 0 <= nx < width:
                j = quantized[ny, nx]
                glcm[i, j] += 1
    
    # Normalize to probabilities
    total = glcm.sum()
    if total > 0:
        glcm = glcm / total
    
    return glcm


def compute_glcm_features(glcm: np.ndarray) -> Dict[str, float]:
    """Compute features from GLCM."""
    features = {}
    
    # Contrast (local variations)
    contrast = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += glcm[i, j] * (i - j) ** 2
    features['contrast'] = float(contrast)
    
    # Homogeneity (closeness to diagonal)
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
    features['homogeneity'] = float(homogeneity)
    
    # Energy (uniformity)
    energy = np.sum(glcm ** 2)
    features['energy'] = float(energy)
    
    # Correlation
    mean_i = np.sum(np.sum(glcm, axis=1) * np.arange(glcm.shape[0]))
    mean_j = np.sum(np.sum(glcm, axis=0) * np.arange(glcm.shape[1]))
    
    std_i = np.sqrt(np.sum(np.sum(glcm, axis=1) * 
                           (np.arange(glcm.shape[0]) - mean_i) ** 2))
    std_j = np.sqrt(np.sum(np.sum(glcm, axis=0) * 
                           (np.arange(glcm.shape[1]) - mean_j) ** 2))
    
    correlation = 0
    if std_i > 0 and std_j > 0:
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                correlation += glcm[i, j] * (i - mean_i) * (j - mean_j)
        correlation /= (std_i * std_j)
    
    features['correlation'] = float(correlation)
    
    # Entropy
    entropy = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            if glcm[i, j] > 0:
                entropy -= glcm[i, j] * np.log2(glcm[i, j])
    features['entropy'] = float(entropy)
    
    # Maximum probability
    max_prob = np.max(glcm)
    features['max_probability'] = float(max_prob)
    
    return features


def test_glcm_features():
    """Test function for GLCM feature extraction."""
    # Create test image with texture
    test_image = np.zeros((100, 100), dtype=np.uint8)
    
    # Create texture pattern
    for i in range(0, 100, 5):
        for j in range(0, 100, 5):
            # Create alternating pattern
            if (i + j) % 10 == 0:
                test_image[i:i+5, j:j+5] = 255
            else:
                test_image[i:i+5, j:j+5] = 0
    
    # Add noise
    noise = np.random.normal(0, 30, test_image.shape).astype(np.uint8)
    test_image = np.clip(test_image + noise, 0, 255)
    
    # Extract features
    features = extract_glcm_features(test_image)
    
    # Print results
    print("GLCM Features Extracted:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    return features


if __name__ == "__main__":
    test_glcm_features() 