

import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def _create_steerable_filter(angle, size=15, sigma=2):
    """
    Create steerable derivative filter.
    """
    x, y = np.meshgrid(np.arange(size) - size//2, 
                      np.arange(size) - size//2)
    
    # Rotate coordinates
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)
    
    # Second derivative of Gaussian
    g = np.exp(-(x_rot**2 + y_rot**2) / (2*sigma**2))
    kernel = -x_rot * g / (sigma**4)
    
    # Normalize the kernel to prevent intensity shifts
    if np.sum(np.abs(kernel)) > 1e-10:
        kernel = kernel / np.sum(np.abs(kernel))
        
    return kernel

def _bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation.
    """
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1
    
    # Boundary check
    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)
    
    # Interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    
    return wa * img[y0, x0] + wb * img[y0, x1] + \
           wc * img[y1, x0] + wd * img[y1, x1]

def _directional_nms(response, angle):
    """
    Non-maximum suppression along direction.
    """
    h, w = response.shape
    suppressed = response.copy()
    
    # Direction vectors
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            # Interpolate along direction
            val1 = _bilinear_interpolate(response, x + dx, y + dy)
            val2 = _bilinear_interpolate(response, x - dx, y - dy)
            
            # Suppress if not maximum
            if response[y, x] < val1 or response[y, x] < val2:
                suppressed[y, x] = 0
                
    return suppressed

def directional_filter_bank(image, n_orientations=16):
    """
    Directional filter bank using steerable filters.
    """
    response_map = np.zeros_like(image, dtype=np.float64)
    
    # Create oriented filters
    for i in range(n_orientations):
        angle = i * np.pi / n_orientations
        
        # Steerable filter coefficients
        kernel = _create_steerable_filter(angle)
        
        # Apply filter
        response = cv2.filter2D(image, -1, kernel)
        
        # Non-maximum suppression along perpendicular direction
        suppressed = _directional_nms(response, angle + np.pi/2)
        
        # Update maximum response
        response_map = np.maximum(response_map, suppressed)
        
    return response_map

if __name__ == '__main__':
    # Create a dummy image with oriented lines
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(image, (50, 50), (200, 80), 255, 2)
    cv2.line(image, (50, 150), (80, 200), 255, 2)
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)

    # Apply the directional filter bank
    response = directional_filter_bank(noisy_image.astype(np.float64), n_orientations=16)
    
    # Normalize for display
    response_normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Directional Filter Bank Result', response_normalized)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

