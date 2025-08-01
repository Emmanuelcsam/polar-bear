
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

def frangi_vesselness(image, scales=np.arange(1, 4, 0.5)):
    """
    Frangi vesselness filter for line-like structures.
    """
    image = image.astype(np.float64)
    vesselness = np.zeros_like(image)
    
    for scale in scales:
        # Gaussian derivatives
        smoothed = gaussian_filter(image, scale)
        
        # Hessian
        Hxx = gaussian_filter(smoothed, scale, order=[0, 2])
        Hyy = gaussian_filter(smoothed, scale, order=[2, 0])
        Hxy = gaussian_filter(smoothed, scale, order=[1, 1])
        
        # Eigenvalues
        tmp = np.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
        lambda1 = 0.5 * (Hxx + Hyy + tmp)
        lambda2 = 0.5 * (Hxx + Hyy - tmp)
        
        # Sort eigenvalues by absolute value
        idx = np.abs(lambda1) < np.abs(lambda2)
        lambda1[idx], lambda2[idx] = lambda2[idx], lambda1[idx]
        
        # Vesselness measures
        Rb = np.abs(lambda1) / (np.abs(lambda2) + 1e-10)
        S = np.sqrt(lambda1**2 + lambda2**2)
        
        # Parameters
        beta = 0.5
        gamma = 15
        
        # Vesselness
        v = np.exp(-Rb**2 / (2*beta**2)) * (1 - np.exp(-S**2 / (2*gamma**2)))
        v[lambda2 > 0] = 0
        
        # Update maximum response
        vesselness = np.maximum(vesselness, v)
        
    return vesselness

if __name__ == '__main__':
    # Create a dummy image with a line-like structure
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(image, (50, 128), (200, 128), 150, 2)
    noisy_image = cv2.GaussianBlur(image, (5,5), 0) + np.random.normal(0, 10, image.shape).astype(np.uint8)

    # Apply the Frangi vesselness filter
    vesselness_response = frangi_vesselness(noisy_image, scales=np.arange(1, 4, 0.5))
    
    # Normalize for display
    vesselness_normalized = cv2.normalize(vesselness_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Frangi Vesselness Result', vesselness_normalized)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
