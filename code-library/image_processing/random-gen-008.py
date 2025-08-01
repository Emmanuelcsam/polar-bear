

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

def _compute_structure_tensor(image, sigma=1.0):
    """
    Compute structure tensor J = ∇I ⊗ ∇I.
    """
    # Gradients
    Ix = ndimage.sobel(image, axis=1)
    Iy = ndimage.sobel(image, axis=0)
    
    # Structure tensor components
    Jxx = gaussian_filter(Ix * Ix, sigma)
    Jxy = gaussian_filter(Ix * Iy, sigma)
    Jyy = gaussian_filter(Iy * Iy, sigma)
    
    return np.stack([Jxx, Jxy, Jxy, Jyy], axis=-1).reshape(*image.shape, 2, 2)

def _eigen_decomposition_2x2(J):
    """
    Efficient eigendecomposition for 2x2 matrices.
    """
    # Extract components
    a = J[..., 0, 0]
    b = J[..., 0, 1]
    c = J[..., 1, 1]
    
    # Trace and determinant
    trace = a + c
    det = a * c - b * b
    
    # Eigenvalues
    discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)
    
    # Eigenvectors
    v1x = lambda1 - c
    v1y = b
    norm1 = np.sqrt(v1x**2 + v1y**2 + 1e-10)
    v1x /= norm1
    v1y /= norm1
    
    v2x = -v1y
    v2y = v1x
    
    eigenvals = np.stack([lambda1, lambda2], axis=-1)
    eigenvecs = np.stack([v1x, v1y, v2x, v2y], axis=-1).reshape(*J.shape[:-2], 2, 2)
    
    return eigenvals, eigenvecs

def _compute_diffusion_tensor(eigenvals, eigenvecs, alpha=0.001):
    """
    Compute diffusion tensor for coherence enhancement.
    """
    lambda1 = eigenvals[..., 0]
    lambda2 = eigenvals[..., 1]
    
    # Coherence measure
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))**2
    
    # Diffusion eigenvalues
    c1 = alpha
    c2 = alpha + (1 - alpha) * np.exp(-1 / (coherence + 1e-10))
    
    # Reconstruct diffusion tensor
    D = np.zeros_like(eigenvecs)
    D[..., 0, 0] = c1 * eigenvecs[..., 0, 0]**2 + c2 * eigenvecs[..., 1, 0]**2
    D[..., 0, 1] = c1 * eigenvecs[..., 0, 0] * eigenvecs[..., 0, 1] + \
                    c2 * eigenvecs[..., 1, 0] * eigenvecs[..., 1, 1]
    D[..., 1, 0] = D[..., 0, 1]
    D[..., 1, 1] = c1 * eigenvecs[..., 0, 1]**2 + c2 * eigenvecs[..., 1, 1]**2
    
    return D

def _apply_tensor_diffusion(image, D, dt=0.1):
    """
    Apply tensor-based diffusion.
    """
    # Compute second derivatives
    Ixx = ndimage.sobel(ndimage.sobel(image, axis=1), axis=1)
    Iyy = ndimage.sobel(ndimage.sobel(image, axis=0), axis=0)
    Ixy = ndimage.sobel(ndimage.sobel(image, axis=1), axis=0)
    
    # Diffusion update
    div = D[..., 0, 0] * Ixx + 2 * D[..., 0, 1] * Ixy + D[..., 1, 1] * Iyy
    
    return image + dt * div

def coherence_enhancing_diffusion(image, iterations=5):
    """
    Coherence-enhancing diffusion for linear structures.
    """
    img = image.copy().astype(np.float64)
    
    for _ in range(iterations):
        # Structure tensor
        J = _compute_structure_tensor(img)
        
        # Eigenvalues and eigenvectors
        eigenvals, eigenvecs = _eigen_decomposition_2x2(J)
        
        # Diffusion tensor
        D = _compute_diffusion_tensor(eigenvals, eigenvecs)
        
        # Apply diffusion
        img = _apply_tensor_diffusion(img, D)
        
    return img

if __name__ == '__main__':
    # Create a dummy image for demonstration
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(image, (50, 50), (200, 200), 128, 5)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    noisy_image = image + np.random.normal(0, 20, image.shape).astype(np.uint8)

    # Apply the coherence enhancing diffusion function
    enhanced_image = coherence_enhancing_diffusion(noisy_image, iterations=5)
    
    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Coherence Enhanced Result', enhanced_image.astype(np.uint8))
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

