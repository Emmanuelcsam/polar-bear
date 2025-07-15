# OpenCV Scripts Collection

This directory contains 100+ individual scripts for various OpenCV image processing operations.
Each script follows the standard format with a `process_image(image: np.ndarray) -> np.ndarray` function.

## Categories

### Filtering (20 scripts)
- Box filter, Gaussian blur (normal and strong)
- Median filter, Bilateral filter
- Motion blur, Sharpening, Unsharp mask
- Emboss, Laplacian, Sobel X/Y, Scharr X/Y
- Prewitt X/Y, High/Low pass filters
- Guided filter, NLM denoising

### Morphology (12 scripts)
- Erosion, Dilation, Opening, Closing
- Gradient, Top Hat, Black Hat
- Hit or Miss, Skeleton, Reconstruction
- Elliptical and Cross kernel operations

### Thresholding (12 scripts)
- Binary (normal and inverse)
- Truncate, To Zero (normal and inverse)
- Otsu's method, Triangle method
- Adaptive (mean and Gaussian)
- Multi-level Otsu
- Niblack's and Sauvola's methods

### Edge Detection (8 scripts)
- Canny (manual and auto threshold)
- Sobel combined, Roberts Cross
- Marr-Hildreth (LoG)
- Difference of Gaussians
- Structured edges
- Phase congruency

### Transformations (18 scripts)
- Resize (half, double), Rotate (45, 90, 180, 270)
- Flip (horizontal, vertical, both)
- Affine and Perspective transforms
- Polar and Log-polar transforms
- Color space conversions (RGB to Gray/HSV/LAB)
- Crop center, Padding

### Histogram Operations (9 scripts)
- Histogram equalization, CLAHE
- Histogram stretching and matching
- Gamma correction
- Log, Exponential, and Power law transforms
- Bit plane slicing

### Feature Detection (12 scripts)
- Harris corners, Shi-Tomasi
- FAST, ORB features
- Blob detection
- Hough lines and circles
- Contour detection, MSER regions
- Template matching
- Watershed and GrabCut segmentation

### Noise Operations (9 scripts)
- Add noise: Gaussian, Salt & Pepper, Poisson, Speckle, Uniform
- Remove noise: Median filter, Bilateral filter, Non-Local Means, Morphological

### Effects (20+ scripts)
- Pyramid operations (Gaussian, Laplacian)
- Inpainting, Super resolution
- Artistic effects: Sepia, Pencil sketch, Cartoon, Oil painting
- Photo effects: HDR tone mapping, Vignette, Vintage
- Pixelate, Mosaic, Color quantization
- Negative, Solarize, Posterize
- Cross processing, 3D Emboss, Kaleidoscope

## Usage

Each script can be used in two ways:

### 1. As a module in your code:
```python
from opencv_scripts.filtering.gaussian_blur import process_image
result = process_image(input_image)
```

### 2. As a standalone script:
```bash
python opencv_scripts/filtering/gaussian_blur.py input_image.jpg
```

## Requirements
- OpenCV (cv2)
- NumPy
- Some effects require opencv-contrib-python for extra modules

## Notes
- All scripts handle both grayscale and color images
- Parameters are set to reasonable defaults
- Output maintains the same type as input when possible
