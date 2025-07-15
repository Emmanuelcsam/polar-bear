
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def phase_congruency(image, nscale=4, norient=6):
    """
    Phase congruency for feature detection.
    """
    rows, cols = image.shape
    
    # Fourier transform
    IM = np.fft.fft2(image)
    
    # Initialize
    PC = np.zeros((rows, cols), dtype=np.float64)
    
    # Frequency coordinates
    u, v = np.meshgrid(np.fft.fftfreq(cols), np.fft.fftfreq(rows))
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1  # Avoid division by zero
    
    # Log-Gabor filters
    wavelength = 6
    for s in range(nscale):
        lambda_s = wavelength * (2**s)
        fo = 1.0 / lambda_s
        
        # Log-Gabor radial component
        logGabor = np.exp(-(np.log(radius/fo))**2 / (2 * np.log(0.65)**2))
        logGabor[radius < fo/3] = 0
        
        for o in range(norient):
            angle = o * np.pi / norient
            
            # Angular component
            theta = np.arctan2(v, u)
            ds = np.sin(theta - angle)
            dc = np.cos(theta - angle)
            spread = np.pi / norient / 1.5
            
            angular = np.exp(-(ds**2) / (2 * spread**2))
            
            # Combined filter
            filter_bank = logGabor * angular
            
            # Apply filter
            response = np.fft.ifft2(IM * filter_bank)
            
            # Phase congruency calculation
            magnitude = np.abs(response)
            phase = np.angle(response)
            
            # Accumulate phase congruency
            PC += magnitude * np.cos(phase - np.mean(phase))
            
    # Normalize
    PC = PC / (nscale * norient)
    if (PC.max() - PC.min()) > 1e-10:
        PC = (PC - PC.min()) / (PC.max() - PC.min())
    
    return PC

if __name__ == '__main__':
    # Create a dummy image with edges
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 200), 255, 5)
    noisy_image = image + np.random.normal(0, 10, image.shape).astype(np.uint8)

    # Apply the phase congruency function
    pc_response = phase_congruency(noisy_image, nscale=4, norient=6)
    
    # Normalize for display
    pc_normalized = cv2.normalize(pc_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the results
    cv2.imshow('Original Noisy Image', noisy_image)
    cv2.imshow('Phase Congruency Result', pc_normalized)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
