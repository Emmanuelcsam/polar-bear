
import numpy as np
import cv2
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

def _radon_transform(image, theta):
    """
    Compute Radon transform.
    """
    # Pad image
    diagonal = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    pad_x = (diagonal - image.shape[1]) // 2
    pad_y = (diagonal - image.shape[0]) // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    
    # Initialize sinogram
    sinogram = np.zeros((diagonal, len(theta)))
    
    # Compute projections
    for i, angle in enumerate(theta):
        rotated = ndimage.rotate(padded, angle, reshape=False, order=1)
        sinogram[:, i] = np.sum(rotated, axis=1)
        
    return sinogram

def _draw_line_from_radon(mask, rho, theta, shape):
    """
    Draw line from Radon parameters.
    """
    h, w = shape
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Handle edge cases for nearly horizontal or vertical lines
    epsilon = 1e-10
    
    if abs(sin_t) < epsilon:
        # Nearly horizontal line (theta ≈ 0 or π)
        x = int(rho / cos_t) if abs(cos_t) > epsilon else 0
        if 0 <= x < w:
            mask[:, x] = 1
    elif abs(cos_t) < epsilon:
        # Nearly vertical line (theta ≈ π/2)
        y = int(rho / sin_t) if abs(sin_t) > epsilon else 0
        if 0 <= y < h:
            mask[y, :] = 1
    elif abs(cos_t) > abs(sin_t):
        # More horizontal than vertical
        for x in range(w):
            y_float = (rho - x * cos_t) / sin_t
            if not np.isfinite(y_float):
                continue
            y = int(round(y_float))
            if 0 <= y < h:
                mask[y, x] = 1
    else:
        # More vertical than horizontal
        for y in range(h):
            x_float = (rho - y * sin_t) / cos_t
            if not np.isfinite(x_float):
                continue
            x = int(round(x_float))
            if 0 <= x < w:
                mask[y, x] = 1

def radon_line_detection(image):
    """
    Radon transform for line detection.
    """
    # Edge detection first
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    
    # Radon transform
    theta = np.linspace(0, 180, 180, endpoint=False)
    sinogram = _radon_transform(edges, theta)
    
    # Find peaks in Radon space
    line_mask = np.zeros_like(image)
    
    # Threshold for peak detection
    threshold = np.mean(sinogram) + 2 * np.std(sinogram)
    
    # Back-project strong lines
    for i in range(sinogram.shape[1]):
        if np.max(sinogram[:, i]) > threshold:
            # Find peak
            rho_idx = np.argmax(sinogram[:, i])
            angle = theta[i] * np.pi / 180
            
            # Draw line
            _draw_line_from_radon(line_mask, rho_idx - sinogram.shape[0]//2, 
                                      angle, image.shape)
            
    return line_mask

if __name__ == '__main__':
    # Create a dummy image with lines
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(image, (50, 50), (200, 50), 255, 2)
    cv2.line(image, (50, 150), (200, 100), 255, 2)
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)

    # Apply Radon line detection
    line_mask = radon_line_detection(noisy_image.astype(np.float64) / 255.0)
    
    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Radon Line Detection Result', line_mask.astype(np.uint8) * 255)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
