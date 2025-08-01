
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1):
    """
    Perona-Malik anisotropic diffusion.
    """
    img = image.copy().astype(np.float64)
    
    for _ in range(iterations):
        # Calculate gradients
        nablaE = np.roll(img, -1, axis=1) - img  # East
        nablaW = np.roll(img, 1, axis=1) - img   # West
        nablaN = np.roll(img, -1, axis=0) - img  # North
        nablaS = np.roll(img, 1, axis=0) - img   # South
        
        # Diffusion coefficient g(|∇I|) = 1 / (1 + (|∇I|/K)²)
        cE = 1.0 / (1.0 + (nablaE/kappa)**2)
        cW = 1.0 / (1.0 + (nablaW/kappa)**2)
        cN = 1.0 / (1.0 + (nablaN/kappa)**2)
        cS = 1.0 / (1.0 + (nablaS/kappa)**2)
        
        # Update
        img += gamma * (cE*nablaE + cW*nablaW + cN*nablaN + cS*nablaS)
        
    return img

if __name__ == '__main__':
    # Create a dummy image for demonstration
    # In a real scenario, you would load an image using cv2.imread
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(image, (128, 128), 50, 128, -1)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    noisy_image = image + np.random.normal(0, 20, image.shape).astype(np.uint8)

    # Apply the anisotropic diffusion function
    diffused_image = anisotropic_diffusion(noisy_image, iterations=10, kappa=50, gamma=0.1)
    
    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Anisotropic Diffusion Result', diffused_image.astype(np.uint8))
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
