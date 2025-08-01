#!/usr/bin/env python3
"""
Anisotropic Diffusion Module for Image Preprocessing
===================================================

This module implements advanced diffusion-based image preprocessing methods
for fiber optic defect detection. Extracted from defect_analysis.py.

Functions:
- Perona-Malik anisotropic diffusion
- Coherence-enhancing diffusion
- Structure tensor computation
- Tensor-based diffusion

Author: Extracted from Advanced Fiber Analysis Team
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')


def perona_malik_diffusion(image, iterations=10, kappa=50, gamma=0.1):
    """
    Perona-Malik anisotropic diffusion for edge-preserving smoothing.
    
    The diffusion equation: ∂I/∂t = div(g(|∇I|) * ∇I)
    where g(x) = 1 / (1 + (x/K)²) is the diffusion function
    
    Args:
        image (np.ndarray): Input grayscale image
        iterations (int): Number of diffusion iterations
        kappa (float): Diffusion constant (edge threshold)
        gamma (float): Time step for numerical integration
    
    Returns:
        np.ndarray: Diffused image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float64 for precision
    img = image.astype(np.float64)
    
    for iteration in range(iterations):
        # Calculate gradients in four directions
        nablaE = np.roll(img, -1, axis=1) - img  # East gradient
        nablaW = np.roll(img, 1, axis=1) - img   # West gradient
        nablaN = np.roll(img, -1, axis=0) - img  # North gradient
        nablaS = np.roll(img, 1, axis=0) - img   # South gradient
        
        # Compute diffusion coefficients g(|∇I|) = 1 / (1 + (|∇I|/K)²)
        cE = 1.0 / (1.0 + (nablaE / kappa) ** 2)
        cW = 1.0 / (1.0 + (nablaW / kappa) ** 2)
        cN = 1.0 / (1.0 + (nablaN / kappa) ** 2)
        cS = 1.0 / (1.0 + (nablaS / kappa) ** 2)
        
        # Update equation: I^(n+1) = I^n + γ * Σ(c_i * ∇_i I)
        img += gamma * (cE * nablaE + cW * nablaW + cN * nablaN + cS * nablaS)
        
        # Ensure values stay within valid range
        img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


def compute_structure_tensor(image, sigma=1.0):
    """
    Compute the structure tensor J = ∇I ⊗ ∇I for coherence analysis.
    
    Args:
        image (np.ndarray): Input grayscale image
        sigma (float): Gaussian smoothing parameter for tensor components
    
    Returns:
        np.ndarray: Structure tensor of shape (H, W, 2, 2)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_float = image.astype(np.float64)
    
    # Compute image gradients using Sobel operators
    Ix = ndimage.sobel(img_float, axis=1)  # Gradient in x-direction
    Iy = ndimage.sobel(img_float, axis=0)  # Gradient in y-direction
    
    # Compute structure tensor components
    Jxx = gaussian_filter(Ix * Ix, sigma)  # J₁₁
    Jxy = gaussian_filter(Ix * Iy, sigma)  # J₁₂ = J₂₁
    Jyy = gaussian_filter(Iy * Iy, sigma)  # J₂₂
    
    # Arrange as 2x2 tensor at each pixel
    h, w = image.shape
    J = np.zeros((h, w, 2, 2))
    J[:, :, 0, 0] = Jxx
    J[:, :, 0, 1] = Jxy
    J[:, :, 1, 0] = Jxy
    J[:, :, 1, 1] = Jyy
    
    return J


def eigen_decomposition_2x2(J):
    """
    Efficient eigendecomposition for 2x2 structure tensors.
    
    For a 2x2 matrix [[a,b],[c,d]], eigenvalues are:
    λ = (trace ± √(trace² - 4*det)) / 2
    
    Args:
        J (np.ndarray): Structure tensor of shape (H, W, 2, 2)
    
    Returns:
        tuple: (eigenvalues, eigenvectors) both of shape (H, W, 2, 2)
    """
    # Extract tensor components
    a = J[:, :, 0, 0]  # Jxx
    b = J[:, :, 0, 1]  # Jxy
    c = J[:, :, 1, 1]  # Jyy
    
    # Compute trace and determinant
    trace = a + c
    det = a * c - b * b
    
    # Compute eigenvalues using quadratic formula
    discriminant = np.sqrt(np.maximum(0, trace**2 - 4 * det))
    lambda1 = 0.5 * (trace + discriminant)  # Larger eigenvalue
    lambda2 = 0.5 * (trace - discriminant)  # Smaller eigenvalue
    
    # Compute eigenvectors
    # For first eigenvector (λ₁)
    v1x = lambda1 - c
    v1y = b
    norm1 = np.sqrt(v1x**2 + v1y**2 + 1e-10)  # Add small epsilon to avoid division by zero
    v1x /= norm1
    v1y /= norm1
    
    # Second eigenvector is perpendicular to first
    v2x = -v1y
    v2y = v1x
    
    # Pack eigenvalues and eigenvectors
    eigenvals = np.stack([lambda1, lambda2], axis=-1)
    eigenvecs = np.zeros_like(J)
    eigenvecs[:, :, 0, 0] = v1x
    eigenvecs[:, :, 0, 1] = v1y
    eigenvecs[:, :, 1, 0] = v2x
    eigenvecs[:, :, 1, 1] = v2y
    
    return eigenvals, eigenvecs


def compute_diffusion_tensor(eigenvals, eigenvecs, alpha=0.001):
    """
    Compute diffusion tensor for coherence-enhancing diffusion.
    
    Args:
        eigenvals (np.ndarray): Eigenvalues of structure tensor
        eigenvecs (np.ndarray): Eigenvectors of structure tensor
        alpha (float): Minimum diffusion coefficient
    
    Returns:
        np.ndarray: Diffusion tensor D of shape (H, W, 2, 2)
    """
    lambda1 = eigenvals[:, :, 0]  # Larger eigenvalue
    lambda2 = eigenvals[:, :, 1]  # Smaller eigenvalue
    
    # Compute coherence measure
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)) ** 2
    
    # Compute diffusion eigenvalues
    c1 = alpha  # Diffusion along gradient (perpendicular to edges)
    c2 = alpha + (1 - alpha) * np.exp(-1 / (coherence + 1e-10))  # Enhanced diffusion along edges
    
    # Reconstruct diffusion tensor D = V * diag(c1, c2) * V^T
    v1x = eigenvecs[:, :, 0, 0]
    v1y = eigenvecs[:, :, 0, 1]
    v2x = eigenvecs[:, :, 1, 0]
    v2y = eigenvecs[:, :, 1, 1]
    
    D = np.zeros_like(eigenvecs)
    D[:, :, 0, 0] = c1 * v1x**2 + c2 * v2x**2
    D[:, :, 0, 1] = c1 * v1x * v1y + c2 * v2x * v2y
    D[:, :, 1, 0] = D[:, :, 0, 1]  # Symmetric
    D[:, :, 1, 1] = c1 * v1y**2 + c2 * v2y**2
    
    return D


def apply_tensor_diffusion(image, D, dt=0.1):
    """
    Apply one step of tensor-based diffusion.
    
    Args:
        image (np.ndarray): Input image
        D (np.ndarray): Diffusion tensor
        dt (float): Time step
    
    Returns:
        np.ndarray: Updated image after one diffusion step
    """
    # Compute second derivatives using Sobel operators
    Ix = ndimage.sobel(image, axis=1)
    Iy = ndimage.sobel(image, axis=0)
    
    Ixx = ndimage.sobel(Ix, axis=1)
    Iyy = ndimage.sobel(Iy, axis=0)
    Ixy = ndimage.sobel(Ix, axis=0)
    
    # Compute divergence: div(D∇I) = D₁₁I_xx + 2D₁₂I_xy + D₂₂I_yy
    divergence = (D[:, :, 0, 0] * Ixx + 
                 2 * D[:, :, 0, 1] * Ixy + 
                 D[:, :, 1, 1] * Iyy)
    
    # Update: I^(n+1) = I^n + dt * div(D∇I)
    updated_image = image + dt * divergence
    
    return np.clip(updated_image, 0, 255)


def coherence_enhancing_diffusion(image, iterations=5, alpha=0.001):
    """
    Coherence-enhancing diffusion for preserving and enhancing linear structures.
    
    Args:
        image (np.ndarray): Input grayscale image
        iterations (int): Number of diffusion iterations
        alpha (float): Minimum diffusion coefficient
    
    Returns:
        np.ndarray: Enhanced image with preserved linear structures
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    
    for iteration in range(iterations):
        # Compute structure tensor
        J = compute_structure_tensor(img)
        
        # Eigendecomposition
        eigenvals, eigenvecs = eigen_decomposition_2x2(J)
        
        # Compute diffusion tensor
        D = compute_diffusion_tensor(eigenvals, eigenvecs, alpha)
        
        # Apply tensor diffusion
        img = apply_tensor_diffusion(img, D)
    
    return img.astype(np.uint8)


def shock_filter(image, iterations=5, dt=0.1):
    """
    Shock filter for edge enhancement and deblurring.
    
    Args:
        image (np.ndarray): Input grayscale image
        iterations (int): Number of filter iterations
        dt (float): Time step
    
    Returns:
        np.ndarray: Shock-filtered image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    
    for iteration in range(iterations):
        # Compute gradients
        Ix = ndimage.sobel(img, axis=1)
        Iy = ndimage.sobel(img, axis=0)
        
        # Compute second derivatives
        Ixx = ndimage.sobel(Ix, axis=1)
        Iyy = ndimage.sobel(Iy, axis=0)
        Ixy = ndimage.sobel(Ix, axis=0)
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(Ix**2 + Iy**2 + 1e-10)
        
        # Compute Laplacian in gradient direction
        # L_ηη = (Ix²Ixx + 2IxIyIxy + Iy²Iyy) / |∇I|²
        numerator = Ix**2 * Ixx + 2 * Ix * Iy * Ixy + Iy**2 * Iyy
        L_eta_eta = numerator / (grad_mag**2 + 1e-10)
        
        # Shock filter update: I_t = -sign(L_ηη) * |∇I|
        shock_term = -np.sign(L_eta_eta) * grad_mag
        
        # Update image
        img += dt * shock_term
        
        # Clamp values
        img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


def bilateral_diffusion(image, iterations=5, sigma_spatial=2.0, sigma_intensity=30.0):
    """
    Bilateral diffusion combining spatial and intensity information.
    
    Args:
        image (np.ndarray): Input grayscale image
        iterations (int): Number of diffusion iterations
        sigma_spatial (float): Spatial standard deviation
        sigma_intensity (float): Intensity standard deviation
    
    Returns:
        np.ndarray: Bilateral diffused image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img = image.astype(np.float64)
    
    for iteration in range(iterations):
        # Apply bilateral filter (approximates bilateral diffusion)
        img = cv2.bilateralFilter(
            img.astype(np.uint8), 
            d=-1,  # Automatic neighborhood size
            sigmaColor=sigma_intensity,
            sigmaSpace=sigma_spatial
        ).astype(np.float64)
    
    return img.astype(np.uint8)


def visualize_diffusion_results(original, results_dict, save_path=None):
    """
    Visualize diffusion results.
    
    Args:
        original (np.ndarray): Original image
        results_dict (dict): Dictionary of diffusion results
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    n_methods = len(results_dict) + 1
    cols = 3
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    # Show original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show diffusion results
    for i, (method_name, result) in enumerate(results_dict.items(), 1):
        if i < len(axes):
            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(f'{method_name.replace("_", " ").title()}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(results_dict) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    """
    Example usage and testing of anisotropic diffusion functions.
    """
    # Create a test image with noise and edges
    test_image = np.random.normal(128, 30, (200, 200)).astype(np.uint8)
    
    # Add some structures
    cv2.rectangle(test_image, (50, 50), (100, 100), 200, -1)
    cv2.line(test_image, (20, 150), (180, 150), 50, 3)
    cv2.circle(test_image, (150, 70), 20, 100, -1)
    
    print("Testing Anisotropic Diffusion Module")
    print("=" * 50)
    
    results = {}
    
    print("\n1. Perona-Malik diffusion:")
    pm_result = perona_malik_diffusion(test_image, iterations=10, kappa=30)
    results['perona_malik'] = pm_result
    print(f"   Applied {10} iterations of Perona-Malik diffusion")
    
    print("\n2. Coherence-enhancing diffusion:")
    ced_result = coherence_enhancing_diffusion(test_image, iterations=5)
    results['coherence_enhancing'] = ced_result
    print(f"   Applied {5} iterations of coherence-enhancing diffusion")
    
    print("\n3. Shock filter:")
    shock_result = shock_filter(test_image, iterations=3)
    results['shock_filter'] = shock_result
    print(f"   Applied {3} iterations of shock filtering")
    
    print("\n4. Bilateral diffusion:")
    bilateral_result = bilateral_diffusion(test_image, iterations=3)
    results['bilateral_diffusion'] = bilateral_result
    print(f"   Applied {3} iterations of bilateral diffusion")
    
    # Visualize results
    visualize_diffusion_results(test_image, results, 'diffusion_test.png')
    
    return results


if __name__ == "__main__":
    results = main()
