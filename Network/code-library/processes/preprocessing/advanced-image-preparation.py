import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from typing import Dict

from .defect_detection_config import DefectDetectionConfig
from .anisotropic_diffusion import anisotropic_diffusion
from .coherence_enhancing_diffusion import coherence_enhancing_diffusion

def advanced_preprocessing(image: np.ndarray, config: DefectDetectionConfig) -> Dict[str, np.ndarray]:
    """Apply all preprocessing methods"""
    preprocessed = {'original': image.copy()}
    
    # 1. Anisotropic diffusion
    preprocessed['anisotropic'] = anisotropic_diffusion(image)
    
    # 2. Total Variation denoising
    tv_denoised = denoise_tv_chambolle(image.astype(np.float64) / 255.0, weight=0.1) * 255
    preprocessed['tv_denoised'] = tv_denoised.astype(np.uint8)
    
    # 3. Coherence-enhancing diffusion
    preprocessed['coherence'] = coherence_enhancing_diffusion(preprocessed['tv_denoised'])
    
    # 4. Multiple Gaussian blurs
    for size in config.gaussian_blur_sizes:
        preprocessed[f'gaussian_{size[0]}'] = cv2.GaussianBlur(image, size, 0)
    
    # 5. Multiple bilateral filters
    for i, (d, sc, ss) in enumerate(config.bilateral_params):
        preprocessed[f'bilateral_{i}'] = cv2.bilateralFilter(image, d, sc, ss)
    
    # 6. Multiple CLAHE variants
    for i, (clip, grid) in enumerate(config.clahe_params):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        preprocessed[f'clahe_{i}'] = clahe.apply(image)
    
    # 7. Standard preprocessing
    preprocessed['median'] = cv2.medianBlur(image, 5)
    preprocessed['nlmeans'] = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    preprocessed['histeq'] = cv2.equalizeHist(image)
    
    # 8. Morphological
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    preprocessed['morph_gradient'] = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    preprocessed['tophat'] = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    preprocessed['blackhat'] = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    return preprocessed

if __name__ == '__main__':
    config = DefectDetectionConfig()
    sample_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)

    print("Running advanced preprocessing on a sample image...")
    preprocessed_images = advanced_preprocessing(sample_image, config)

    print("\nGenerated preprocessed images:")
    for name, img in preprocessed_images.items():
        print(f"  - {name}: shape {img.shape}, dtype {img.dtype}")
        cv2.imwrite(f"preprocessed_{name}.png", img)
    print("\nSaved preprocessed images for visual inspection.")