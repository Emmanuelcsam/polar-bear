import numpy as np

def anisotropic_diffusion(image: np.ndarray, iterations: int = 10,
                          kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
    """Perona-Malik anisotropic diffusion"""
    img = image.copy().astype(np.float64)

    for _ in range(iterations):
        # Calculate gradients
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img, 1, axis=1) - img
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img, 1, axis=0) - img

        # Diffusion coefficients
        cE = 1.0 / (1.0 + (nablaE/kappa)**2)
        cW = 1.0 / (1.0 + (nablaW/kappa)**2)
        cN = 1.0 / (1.0 + (nablaN/kappa)**2)
        cS = 1.0 / (1.0 + (nablaS/kappa)**2)

        # Update
        img += gamma * (cE*nablaE + cW*nablaW + cN*nablaN + cS*nablaS)

    return img.astype(np.uint8)

if __name__ == '__main__':
    import cv2

    # Create a sample image with noise
    sample_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(sample_image, (50, 50), (150, 150), 128, -1)
    noisy_image = sample_image + np.random.randint(0, 30, (200, 200), dtype=np.uint8)

    print("Applying anisotropic diffusion to a sample noisy image...")
    diffused_image = anisotropic_diffusion(noisy_image)

    # To visually check the result, you would typically display the images.
    # This requires a GUI environment, which may not be available here.
    # We will save the images instead.
    cv2.imwrite("noisy_image.png", noisy_image)
    cv2.imwrite("diffused_image.png", diffused_image)
    print("Saved 'noisy_image.png' and 'diffused_image.png' for visual inspection.")
