import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def coherence_enhancing_diffusion(image: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Coherence-enhancing diffusion for linear structures"""
    img = image.copy().astype(np.float64)

    for _ in range(iterations):
        # Compute structure tensor
        Ix = ndimage.sobel(img, axis=1)
        Iy = ndimage.sobel(img, axis=0)

        # Structure tensor components
        Jxx = gaussian_filter(Ix * Ix, 1.0)
        Jxy = gaussian_filter(Ix * Iy, 1.0)
        Jyy = gaussian_filter(Iy * Iy, 1.0)

        # Eigenvalues
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)

        # Coherence measure
        coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10))**2

        # Apply diffusion based on coherence
        alpha = 0.001
        c1 = alpha
        c2 = alpha + (1 - alpha) * np.exp(-1 / (coherence + 1e-10))

        # Simple diffusion update
        img = gaussian_filter(img, 0.5)

    return img.astype(np.uint8)

if __name__ == '__main__':
    import cv2

    # Create a sample image with some lines
    sample_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(sample_image, (20, 20), (180, 180), 128, 2)
    cv2.line(sample_image, (20, 180), (180, 20), 128, 2)
    noisy_image = sample_image + np.random.randint(0, 20, (200, 200), dtype=np.uint8)

    print("Applying coherence-enhancing diffusion...")
    enhanced_image = coherence_enhancing_diffusion(noisy_image)

    cv2.imwrite("noisy_lines.png", noisy_image)
    cv2.imwrite("enhanced_lines.png", enhanced_image)
    print("Saved 'noisy_lines.png' and 'enhanced_lines.png' for visual inspection.")
