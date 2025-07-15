
import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

def hessian_ridge_detection(image, scales=[1, 2, 3]):
    """
    Multi-scale Hessian ridge detection.
    """
    ridge_response = np.zeros_like(image, dtype=np.float64)
    
    for scale in scales:
        # Gaussian smoothing at scale
        smoothed = gaussian_filter(image, scale)
        
        # Hessian matrix
        Hxx = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=1)
        Hyy = ndimage.sobel(ndimage.sobel(smoothed, axis=0), axis=0)
        Hxy = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=0)
        
        # Eigenvalues
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy * Hxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        
        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)
        
        # Ridge measure (Frangi-like)
        Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2)
        
        # Ridge response
        beta = 0.5
        c = 0.5 * np.max(S) if np.max(S) > 0 else 1.0
        
        response = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*c**2)))
        response[lambda2 > 0] = 0  # Only negative λ2 for ridges
        
        # Scale normalization
        ridge_response = np.maximum(ridge_response, scale**2 * response)
        
    return ridge_response

if __name__ == '__main__':
    # Create a dummy image with a line
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(image, (50, 50), (200, 50), 128, 3)
    cv2.line(image, (50, 100), (200, 150), 128, 3)
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)

    # Apply the Hessian ridge detection
    ridge_response = hessian_ridge_detection(noisy_image, scales=[1, 2, 3])
    
    # Normalize for display
    ridge_response_normalized = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Hessian Ridge Detection Result', ridge_response_normalized)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
