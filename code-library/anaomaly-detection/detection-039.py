"""Mock skimage.feature.texture module for testing."""

import numpy as np

def greycomatrix(image, distances, angles, levels, symmetric=True, normed=True):
    """Mock GLCM computation."""
    return np.random.rand(levels, levels, len(distances), len(angles))

def greycoprops(glcm, prop):
    """Mock GLCM property extraction."""
    return np.random.rand(1, 1)

# Make sure functions are available
graycomatrix = greycomatrix  # American spelling variant
graycoprops = greycoprops

__all__ = ['greycomatrix', 'greycoprops', 'graycomatrix', 'graycoprops']
