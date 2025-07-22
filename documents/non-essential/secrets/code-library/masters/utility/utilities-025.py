#!/usr/bin/env python3
"""
Generate individual scripts for every OpenCV image processing function
Creates 100+ scripts covering all major OpenCV operations
"""

import os
import sys
from pathlib import Path
import textwrap

def create_script_directory(base_dir="opencv_scripts"):
    """Create directory structure for scripts"""
    base_path = Path(base_dir)
    categories = [
        "filtering", "morphology", "thresholding", "edge_detection",
        "transformations", "histogram", "features", "noise", "effects"
    ]
    
    for category in categories:
        (base_path / category).mkdir(parents=True, exist_ok=True)
    
    return base_path

def write_script(filepath, content):
    """Write script content to file"""
    with open(filepath, 'w') as f:
        f.write(content)
    os.chmod(filepath, 0o755)  # Make executable

def generate_script_template(name, category, description, process_code, imports=""):
    """Generate a complete script from template"""
    template = f'''#!/usr/bin/env python3
"""
{description}
Category: {category}
"""
import cv2
import numpy as np
{imports}

def process_image(image: np.ndarray) -> np.ndarray:
    """
    {description}
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
{textwrap.indent(process_code, '    ')}
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"{name}_output.png", result)
            print(f"Saved to {name}_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python {name}.py <image_path>")
'''
    return template

def generate_filtering_scripts(base_path):
    """Generate all filtering scripts"""
    scripts = [
        ("box_filter", "Box Filter - Simple averaging filter", 
         "result = cv2.boxFilter(result, -1, (5, 5))"),
        
        ("gaussian_blur", "Gaussian Blur - Smoothing with Gaussian kernel",
         "result = cv2.GaussianBlur(result, (5, 5), 1.0)"),
        
        ("median_filter", "Median Filter - Remove salt and pepper noise",
         "result = cv2.medianBlur(result, 5)"),
        
        ("bilateral_filter", "Bilateral Filter - Edge-preserving smoothing",
         "result = cv2.bilateralFilter(result, 9, 75, 75)"),
        
        ("motion_blur", "Motion Blur - Simulate motion effect",
         """kernel = np.zeros((15, 15))
np.fill_diagonal(kernel, 1)
kernel = kernel / 15
result = cv2.filter2D(result, -1, kernel)"""),
        
        ("gaussian_blur_strong", "Strong Gaussian Blur",
         "result = cv2.GaussianBlur(result, (15, 15), 5.0)"),
        
        ("sharpening_filter", "Sharpening Filter - Enhance edges",
         """kernel = np.array([[-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1]])
result = cv2.filter2D(result, -1, kernel)"""),
        
        ("unsharp_mask", "Unsharp Masking - Sharpen image",
         """gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)"""),
        
        ("emboss_filter", "Emboss Filter - 3D effect",
         """kernel = np.array([[-2, -1, 0],
                     [-1,  1, 1],
                     [ 0,  1, 2]])
result = cv2.filter2D(result, -1, kernel) + 128"""),
        
        ("laplacian_filter", "Laplacian Filter - Highlight edges",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
result = np.uint8(np.absolute(laplacian))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("sobel_x", "Sobel X - Vertical edge detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
result = np.uint8(np.absolute(sobelx))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("sobel_y", "Sobel Y - Horizontal edge detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
result = np.uint8(np.absolute(sobely))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("scharr_x", "Scharr X - More accurate Sobel",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
result = np.uint8(np.absolute(scharrx))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("scharr_y", "Scharr Y - More accurate Sobel",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
result = np.uint8(np.absolute(scharry))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("prewitt_x", "Prewitt X - Alternative edge detection",
         """kernel = np.array([[1, 0, -1],
                     [1, 0, -1],
                     [1, 0, -1]])
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = cv2.filter2D(gray, -1, kernel)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("prewitt_y", "Prewitt Y - Alternative edge detection",
         """kernel = np.array([[1, 1, 1],
                     [0, 0, 0],
                     [-1, -1, -1]])
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = cv2.filter2D(gray, -1, kernel)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("high_pass", "High Pass Filter - Enhance details",
         """kernel = np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])
result = cv2.filter2D(result, -1, kernel)"""),
        
        ("low_pass", "Low Pass Filter - Smooth image",
         """kernel = np.ones((5, 5), np.float32) / 25
result = cv2.filter2D(result, -1, kernel)"""),
        
        ("guided_filter", "Guided Filter - Edge-preserving smoothing",
         """# Simple approximation of guided filter
result = cv2.ximgproc.guidedFilter(result, result, 8, 0.2)""",
         "import cv2.ximgproc"),
        
        ("nlm_denoise", "Non-Local Means Denoising",
         """if len(result.shape) == 3:
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
else:
    result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)"""),
    ]
    
    for name, desc, code, *extra in scripts:
        imports = extra[0] if extra else ""
        content = generate_script_template(name, "filtering", desc, code, imports)
        write_script(base_path / "filtering" / f"{name}.py", content)

def generate_morphology_scripts(base_path):
    """Generate all morphology scripts"""
    scripts = [
        ("erosion", "Morphological Erosion - Shrink bright regions",
         """kernel = np.ones((5, 5), np.uint8)
result = cv2.erode(result, kernel, iterations=1)"""),
        
        ("dilation", "Morphological Dilation - Expand bright regions",
         """kernel = np.ones((5, 5), np.uint8)
result = cv2.dilate(result, kernel, iterations=1)"""),
        
        ("opening", "Morphological Opening - Remove small bright spots",
         """kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)"""),
        
        ("closing", "Morphological Closing - Fill small dark holes",
         """kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)"""),
        
        ("gradient", "Morphological Gradient - Edge detection",
         """kernel = np.ones((5, 5), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)"""),
        
        ("tophat", "Top Hat - Extract small bright features",
         """kernel = np.ones((9, 9), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)"""),
        
        ("blackhat", "Black Hat - Extract small dark features",
         """kernel = np.ones((9, 9), np.uint8)
result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)"""),
        
        ("hitmiss", "Hit or Miss - Detect specific patterns",
         """kernel = np.array([[0, 1, 0],
                     [1, -1, 1],
                     [0, 1, 0]], dtype=np.int8)
if len(result.shape) == 3:
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
else:
    result = cv2.morphologyEx(result, cv2.MORPH_HITMISS, kernel)"""),
        
        ("skeleton", "Morphological Skeleton - Thin objects to lines",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
size = np.size(gray)
skel = np.zeros(gray.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False
while not done:
    eroded = cv2.erode(gray, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(gray, temp)
    skel = cv2.bitwise_or(skel, temp)
    gray = eroded.copy()
    zeros = size - cv2.countNonZero(gray)
    if zeros == size:
        done = True
result = skel
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("reconstruction", "Morphological Reconstruction",
         """marker = result.copy()
marker[5:-5, 5:-5] = 0
kernel = np.ones((5, 5), np.uint8)
while True:
    old_marker = marker.copy()
    marker = cv2.dilate(marker, kernel)
    marker = cv2.min(marker, result)
    if np.array_equal(old_marker, marker):
        break
result = marker"""),
        
        ("elliptical_kernel", "Morphology with Elliptical Kernel",
         """kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)"""),
        
        ("cross_kernel", "Morphology with Cross Kernel",
         """kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "morphology", desc, code)
        write_script(base_path / "morphology" / f"{name}.py", content)

def generate_thresholding_scripts(base_path):
    """Generate all thresholding scripts"""
    scripts = [
        ("threshold_binary", "Binary Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_binary_inv", "Inverse Binary Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_truncate", "Truncate Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_tozero", "To Zero Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_tozero_inv", "To Zero Inverse Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_otsu", "Otsu's Thresholding - Automatic threshold",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("adaptive_mean", "Adaptive Mean Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, 2)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("adaptive_gaussian", "Adaptive Gaussian Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("threshold_triangle", "Triangle Thresholding Method",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("multi_otsu", "Multi-level Otsu Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Calculate histogram
hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
# Find two thresholds
t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
t2 = t1 + (255 - t1) // 2
# Apply multi-level thresholding
result = np.zeros_like(gray)
result[gray <= t1] = 0
result[(gray > t1) & (gray <= t2)] = 127
result[gray > t2] = 255
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("niblack_threshold", "Niblack's Local Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Niblack's method: T = mean + k * std
window_size = 25
k = 0.2
# Calculate local mean and std
mean = cv2.boxFilter(gray.astype(np.float32), -1, (window_size, window_size))
sqmean = cv2.boxFilter((gray.astype(np.float32))**2, -1, (window_size, window_size))
std = np.sqrt(sqmean - mean**2)
threshold = mean + k * std
result = (gray > threshold).astype(np.uint8) * 255
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("sauvola_threshold", "Sauvola's Local Thresholding",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Sauvola's method: T = mean * (1 + k * ((std / R) - 1))
window_size = 25
k = 0.2
R = 128
# Calculate local mean and std
mean = cv2.boxFilter(gray.astype(np.float32), -1, (window_size, window_size))
sqmean = cv2.boxFilter((gray.astype(np.float32))**2, -1, (window_size, window_size))
std = np.sqrt(sqmean - mean**2)
threshold = mean * (1 + k * ((std / R) - 1))
result = (gray > threshold).astype(np.uint8) * 255
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "thresholding", desc, code)
        write_script(base_path / "thresholding" / f"{name}.py", content)

def generate_edge_detection_scripts(base_path):
    """Generate all edge detection scripts"""
    scripts = [
        ("canny_edges", "Canny Edge Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = cv2.Canny(gray, 50, 150)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("canny_auto", "Canny Edge Detection with Auto Thresholds",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Calculate automatic thresholds
v = np.median(gray)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
result = cv2.Canny(gray, lower, upper)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("sobel_combined", "Combined Sobel Edge Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
result = np.uint8(np.clip(magnitude, 0, 255))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("roberts_cross", "Roberts Cross Edge Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
edge_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
edge_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)
magnitude = np.sqrt(edge_x**2 + edge_y**2)
result = np.uint8(np.clip(magnitude, 0, 255))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("marr_hildreth", "Marr-Hildreth Edge Detection (LoG)",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Apply Gaussian blur
gaussian = cv2.GaussianBlur(gray, (5, 5), 1.4)
# Apply Laplacian
laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
# Find zero crossings
result = np.zeros_like(gray)
for i in range(1, laplacian.shape[0]-1):
    for j in range(1, laplacian.shape[1]-1):
        if laplacian[i,j] == 0:
            if (laplacian[i-1,j] * laplacian[i+1,j] < 0) or \
               (laplacian[i,j-1] * laplacian[i,j+1] < 0):
                result[i,j] = 255
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("dog_edges", "Difference of Gaussians Edge Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Apply two Gaussian blurs with different sigmas
g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
# Compute difference
dog = g1.astype(np.float32) - g2.astype(np.float32)
# Normalize and threshold
result = np.uint8(np.clip(np.abs(dog) * 10, 0, 255))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("structured_edges", "Structured Edge Detection",
         """# Simple approximation of structured edges
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Compute gradients
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
# Compute magnitude and orientation
magnitude = np.sqrt(gx**2 + gy**2)
orientation = np.arctan2(gy, gx)
# Non-maximum suppression (simplified)
result = np.zeros_like(gray)
angle = orientation * 180.0 / np.pi
angle[angle < 0] += 180
for i in range(1, gray.shape[0]-1):
    for j in range(1, gray.shape[1]-1):
        q = 255
        r = 255
        # angle 0
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
            q = magnitude[i, j+1]
            r = magnitude[i, j-1]
        # angle 45
        elif (22.5 <= angle[i,j] < 67.5):
            q = magnitude[i+1, j-1]
            r = magnitude[i-1, j+1]
        # angle 90
        elif (67.5 <= angle[i,j] < 112.5):
            q = magnitude[i+1, j]
            r = magnitude[i-1, j]
        # angle 135
        elif (112.5 <= angle[i,j] < 157.5):
            q = magnitude[i-1, j-1]
            r = magnitude[i+1, j+1]
        if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
            result[i,j] = magnitude[i,j]
result = np.uint8(np.clip(result, 0, 255))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("phase_congruency", "Phase Congruency Edge Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Simplified phase congruency using multiple scales
scales = [1, 2, 4]
orientations = [0, 45, 90, 135]
pc = np.zeros_like(gray, dtype=np.float32)
for scale in scales:
    for angle in orientations:
        # Create Gabor kernel
        kernel = cv2.getGaborKernel((21, 21), scale*2, np.radians(angle), 10.0, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        pc += np.abs(filtered)
# Normalize
pc = pc / (len(scales) * len(orientations))
result = np.uint8(np.clip(pc * 255 / pc.max(), 0, 255))
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "edge_detection", desc, code)
        write_script(base_path / "edge_detection" / f"{name}.py", content)

def generate_transformation_scripts(base_path):
    """Generate all transformation scripts"""
    scripts = [
        ("resize_half", "Resize Image to Half Size",
         """height, width = result.shape[:2]
new_height, new_width = height // 2, width // 2
result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_LINEAR)"""),
        
        ("resize_double", "Resize Image to Double Size",
         """height, width = result.shape[:2]
new_height, new_width = height * 2, width * 2
result = cv2.resize(result, (new_width, new_height), interpolation=cv2.INTER_CUBIC)"""),
        
        ("rotate_90", "Rotate Image 90 Degrees",
         "result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)"),
        
        ("rotate_180", "Rotate Image 180 Degrees",
         "result = cv2.rotate(result, cv2.ROTATE_180)"),
        
        ("rotate_270", "Rotate Image 270 Degrees",
         "result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)"),
        
        ("rotate_45", "Rotate Image 45 Degrees",
         """height, width = result.shape[:2]
center = (width // 2, height // 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
result = cv2.warpAffine(result, matrix, (width, height))"""),
        
        ("flip_horizontal", "Flip Image Horizontally",
         "result = cv2.flip(result, 1)"),
        
        ("flip_vertical", "Flip Image Vertically",
         "result = cv2.flip(result, 0)"),
        
        ("flip_both", "Flip Image Both Directions",
         "result = cv2.flip(result, -1)"),
        
        ("affine_transform", "Affine Transformation",
         """height, width = result.shape[:2]
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
matrix = cv2.getAffineTransform(pts1, pts2)
result = cv2.warpAffine(result, matrix, (width, height))"""),
        
        ("perspective_transform", "Perspective Transformation",
         """height, width = result.shape[:2]
pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts2 = np.float32([[0, 0], [width, 0], [int(0.2*width), height], [int(0.8*width), height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(result, matrix, (width, height))"""),
        
        ("polar_transform", "Cartesian to Polar Transform",
         """height, width = result.shape[:2]
center = (width // 2, height // 2)
maxRadius = min(center[0], center[1])
flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
result = cv2.warpPolar(result, (width, height), center, maxRadius, flags)"""),
        
        ("log_polar_transform", "Log-Polar Transform",
         """height, width = result.shape[:2]
center = (width // 2, height // 2)
maxRadius = min(center[0], center[1])
M = maxRadius / np.log(maxRadius)
flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG
result = cv2.warpPolar(result, (width, height), center, M, flags)"""),
        
        ("rgb_to_grayscale", "Convert to Grayscale",
         """if len(result.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)"""),
        
        ("rgb_to_hsv", "Convert RGB to HSV",
         """if len(result.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
else:
    result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)"""),
        
        ("rgb_to_lab", "Convert RGB to LAB",
         """if len(result.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
else:
    result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)"""),
        
        ("crop_center", "Crop Center Region",
         """height, width = result.shape[:2]
crop_h, crop_w = height // 2, width // 2
start_h, start_w = height // 4, width // 4
result = result[start_h:start_h+crop_h, start_w:start_w+crop_w]"""),
        
        ("pad_image", "Pad Image with Border",
         """result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "transformations", desc, code)
        write_script(base_path / "transformations" / f"{name}.py", content)

def generate_histogram_scripts(base_path):
    """Generate all histogram operation scripts"""
    scripts = [
        ("histogram_equalization", "Histogram Equalization",
         """if len(result.shape) == 3:
    # Convert to YCrCb and equalize Y channel
    ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
else:
    result = cv2.equalizeHist(result)"""),
        
        ("clahe", "CLAHE - Contrast Limited Adaptive Histogram Equalization",
         """if len(result.shape) == 3:
    # Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
else:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    result = clahe.apply(result)"""),
        
        ("histogram_stretching", "Histogram Stretching",
         """# Stretch histogram to full range
if len(result.shape) == 3:
    for i in range(3):
        channel = result[:,:,i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            result[:,:,i] = ((channel - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
else:
    min_val = result.min()
    max_val = result.max()
    if max_val > min_val:
        result = ((result - min_val) * 255 / (max_val - min_val)).astype(np.uint8)"""),
        
        ("gamma_correction", "Gamma Correction",
         """gamma = 1.5  # Adjust gamma value
inv_gamma = 1.0 / gamma
table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
result = cv2.LUT(result, table)"""),
        
        ("log_transform", "Logarithmic Transform",
         """# Apply log transform for dynamic range compression
c = 255 / np.log(1 + np.max(result))
result = c * (np.log(result + 1))
result = np.array(result, dtype=np.uint8)"""),
        
        ("exponential_transform", "Exponential Transform",
         """# Apply exponential transform
result = np.array(255 * (result / 255.0) ** 2, dtype=np.uint8)"""),
        
        ("power_law_transform", "Power Law Transform",
         """# Power law (gamma) transformation
gamma = 0.5
result = np.array(255 * (result / 255.0) ** gamma, dtype=np.uint8)"""),
        
        ("histogram_matching", "Histogram Matching",
         """# Match histogram to a Gaussian distribution
if len(result.shape) == 3:
    # Convert to grayscale for simplicity
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
else:
    gray = result
# Create target histogram (Gaussian)
hist_target = np.exp(-0.5 * ((np.arange(256) - 128) / 50) ** 2)
hist_target = (hist_target / hist_target.sum() * gray.size).astype(int)
# Calculate CDF
hist_source, _ = np.histogram(gray.flatten(), 256, [0, 256])
cdf_source = hist_source.cumsum()
cdf_target = hist_target.cumsum()
# Create lookup table
lookup = np.zeros(256, dtype=np.uint8)
j = 0
for i in range(256):
    while j < 255 and cdf_source[i] > cdf_target[j]:
        j += 1
    lookup[i] = j
# Apply lookup table
result = lookup[gray]
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("bit_plane_slicing", "Bit Plane Slicing",
         """# Extract bit plane 7 (most significant)
bit_plane = 7
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
result = ((gray >> bit_plane) & 1) * 255
result = result.astype(np.uint8)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "histogram", desc, code)
        write_script(base_path / "histogram" / f"{name}.py", content)

def generate_feature_scripts(base_path):
    """Generate all feature detection scripts"""
    scripts = [
        ("harris_corners", "Harris Corner Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
corners = cv2.cornerHarris(gray, 2, 3, 0.04)
# Dilate corner points
corners = cv2.dilate(corners, None)
# Threshold and mark corners
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
result_color[corners > 0.01 * corners.max()] = [0, 0, 255]
result = result_color"""),
        
        ("shi_tomasi", "Shi-Tomasi Corner Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
if corners is not None:
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(result_color, (x, y), 3, (0, 255, 0), -1)
result = result_color"""),
        
        ("fast_features", "FAST Feature Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))"""),
        
        ("orb_features", "ORB Feature Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)"""),
        
        ("blob_detection", "Blob Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.1
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)
result = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("hough_lines", "Hough Line Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
result = result_color"""),
        
        ("hough_circles", "Hough Circle Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                          param1=50, param2=30, minRadius=10, maxRadius=0)
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        cv2.circle(result_color, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        cv2.circle(result_color, (circle[0], circle[1]), 2, (0, 0, 255), 3)
result = result_color"""),
        
        ("contour_detection", "Contour Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
cv2.drawContours(result_color, contours, -1, (0, 255, 0), 2)
result = result_color"""),
        
        ("mser_regions", "MSER Region Detection",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(result_color, hulls, 1, (0, 255, 0), 2)
result = result_color"""),
        
        ("template_matching", "Template Matching (using center region)",
         """# Use center region as template
h, w = result.shape[:2]
template_h, template_w = h // 4, w // 4
start_h, start_w = h // 2 - template_h // 2, w // 2 - template_w // 2
template = result[start_h:start_h+template_h, start_w:start_w+template_w]
# Convert to grayscale for matching
if len(result.shape) == 3:
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
else:
    gray = result
    template_gray = template
# Apply template matching
res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
# Draw rectangles
result_color = result.copy() if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
for pt in zip(*loc[::-1]):
    cv2.rectangle(result_color, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 0), 2)
result = result_color"""),
        
        ("watershed_segmentation", "Watershed Segmentation",
         """gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
# Threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
# Find sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Find sure foreground
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
# Find unknown region
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
# Apply watershed
result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
markers = cv2.watershed(result_color, markers)
result_color[markers == -1] = [0, 0, 255]
result = result_color"""),
        
        ("grabcut_segmentation", "GrabCut Foreground Extraction",
         """h, w = result.shape[:2]
# Define rectangle around center
rect = (w//4, h//4, w//2, h//2)
# Initialize mask
mask = np.zeros((h, w), np.uint8)
# Initialize foreground and background models
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# Apply GrabCut
if len(result.shape) == 3:
    cv2.grabCut(result, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    result = cv2.bitwise_and(result, result, mask=mask2)
else:
    # GrabCut needs color image
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.grabCut(result_color, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    result = cv2.bitwise_and(result, result, mask=mask2)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "features", desc, code)
        write_script(base_path / "features" / f"{name}.py", content)

def generate_noise_scripts(base_path):
    """Generate all noise-related scripts"""
    scripts = [
        ("add_gaussian_noise", "Add Gaussian Noise",
         """# Add Gaussian noise
mean = 0
sigma = 25
noise = np.random.normal(mean, sigma, result.shape).astype(np.float32)
result = cv2.add(result.astype(np.float32), noise)
result = np.clip(result, 0, 255).astype(np.uint8)"""),
        
        ("add_salt_pepper", "Add Salt and Pepper Noise",
         """# Add salt and pepper noise
noise_ratio = 0.05
total_pixels = result.size
num_salt = int(total_pixels * noise_ratio / 2)
num_pepper = int(total_pixels * noise_ratio / 2)
# Add salt
coords = [np.random.randint(0, i - 1, num_salt) for i in result.shape]
result[coords[0], coords[1]] = 255
# Add pepper
coords = [np.random.randint(0, i - 1, num_pepper) for i in result.shape]
result[coords[0], coords[1]] = 0"""),
        
        ("add_poisson_noise", "Add Poisson Noise",
         """# Add Poisson noise
noise = np.random.poisson(result).astype(np.float32)
result = np.clip(noise, 0, 255).astype(np.uint8)"""),
        
        ("add_speckle_noise", "Add Speckle Noise",
         """# Add multiplicative speckle noise
noise = np.random.randn(*result.shape) * 0.1
result = result + result * noise
result = np.clip(result, 0, 255).astype(np.uint8)"""),
        
        ("add_uniform_noise", "Add Uniform Noise",
         """# Add uniform noise
noise_level = 50
noise = np.random.uniform(-noise_level, noise_level, result.shape)
result = result.astype(np.float32) + noise
result = np.clip(result, 0, 255).astype(np.uint8)"""),
        
        ("remove_noise_median", "Remove Noise with Median Filter",
         """# Remove noise using median filter
result = cv2.medianBlur(result, 5)"""),
        
        ("remove_noise_bilateral", "Remove Noise with Bilateral Filter",
         """# Remove noise while preserving edges
result = cv2.bilateralFilter(result, 9, 75, 75)"""),
        
        ("remove_noise_nlm", "Remove Noise with Non-Local Means",
         """# Remove noise using Non-Local Means
if len(result.shape) == 3:
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
else:
    result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)"""),
        
        ("remove_noise_morphological", "Remove Noise with Morphological Operations",
         """# Remove noise using morphological operations
kernel = np.ones((3, 3), np.uint8)
# Remove salt noise
result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
# Remove pepper noise
result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)"""),
    ]
    
    for name, desc, code in scripts:
        content = generate_script_template(name, "noise", desc, code)
        write_script(base_path / "noise" / f"{name}.py", content)

def generate_effects_scripts(base_path):
    """Generate additional image effects scripts"""
    scripts = [
        ("gaussian_pyramid", "Gaussian Pyramid (Downscale)",
         """# Create Gaussian pyramid
for i in range(2):
    result = cv2.pyrDown(result)"""),
        
        ("laplacian_pyramid", "Laplacian Pyramid",
         """# Create Laplacian pyramid level
gaussian = cv2.pyrDown(result)
gaussian_up = cv2.pyrUp(gaussian, dstsize=(result.shape[1], result.shape[0]))
result = cv2.subtract(result, gaussian_up)
result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)"""),
        
        ("inpainting", "Inpainting (Remove center region)",
         """# Create mask for center region
h, w = result.shape[:2]
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (w//2, h//2), min(h, w)//8, 255, -1)
# Inpaint
if len(result.shape) == 3:
    result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
else:
    result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)"""),
        
        ("super_resolution", "Super Resolution Upscaling",
         """# Simple super resolution using cubic interpolation
result = cv2.resize(result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# Apply sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
result = cv2.filter2D(result, -1, kernel)"""),
        
        ("sepia_effect", "Sepia Tone Effect",
         """# Apply sepia tone
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
kernel = np.array([[0.272, 0.534, 0.131],
                   [0.349, 0.686, 0.168],
                   [0.393, 0.769, 0.189]])
result = cv2.transform(result, kernel)
result = np.clip(result, 0, 255).astype(np.uint8)"""),
        
        ("pencil_sketch", "Pencil Sketch Effect",
         """# Create pencil sketch effect
if len(result.shape) == 3:
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
else:
    gray = result
# Invert
inv = 255 - gray
# Blur
blur = cv2.GaussianBlur(inv, (21, 21), 0)
# Blend
result = cv2.divide(gray, 255 - blur, scale=256)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("cartoon_effect", "Cartoon Effect",
         """# Apply cartoon effect
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
# Apply bilateral filter
smooth = cv2.bilateralFilter(result, 15, 80, 80)
# Get edges
gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
# Convert edges to color
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# Combine
result = cv2.bitwise_and(smooth, edges)"""),
        
        ("oil_painting", "Oil Painting Effect",
         """# Oil painting effect
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
result = cv2.xphoto.oilPainting(result, 7, 1)""",
         "import cv2.xphoto"),
        
        ("hdr_tone_mapping", "HDR Tone Mapping Effect",
         """# Simulate HDR tone mapping
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
# Create multiple exposures
exposures = []
for ev in [-2, 0, 2]:
    exposure = np.clip(result * (2.0 ** ev), 0, 255).astype(np.uint8)
    exposures.append(exposure)
# Merge exposures
merge_mertens = cv2.createMergeMertens()
result = merge_mertens.process(exposures)
result = np.clip(result * 255, 0, 255).astype(np.uint8)"""),
        
        ("vignette_effect", "Vignette Effect",
         """# Add vignette effect
rows, cols = result.shape[:2]
# Create vignette mask
X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
kernel = Y_resultant_kernel * X_resultant_kernel.T
mask = kernel / kernel.max()
# Apply to each channel
if len(result.shape) == 3:
    for i in range(3):
        result[:,:,i] = result[:,:,i] * mask
else:
    result = (result * mask).astype(np.uint8)"""),
        
        ("pixelate_effect", "Pixelate Effect",
         """# Pixelate effect
pixel_size = 10
h, w = result.shape[:2]
# Resize down
temp = cv2.resize(result, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
# Resize up
result = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)"""),
        
        ("color_quantization", "Color Quantization",
         """# Reduce number of colors
n_colors = 8
if len(result.shape) == 3:
    data = result.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    result = quantized.reshape(result.shape)
else:
    # For grayscale, use simple quantization
    levels = np.linspace(0, 255, n_colors)
    result = np.digitize(result, levels) * (255 // (n_colors - 1))
    result = result.astype(np.uint8)"""),
        
        ("mosaic_effect", "Mosaic Effect",
         """# Create mosaic effect
block_size = 10
h, w = result.shape[:2]
for y in range(0, h, block_size):
    for x in range(0, w, block_size):
        # Get block region
        y2 = min(y + block_size, h)
        x2 = min(x + block_size, w)
        # Calculate average color
        if len(result.shape) == 3:
            avg_color = result[y:y2, x:x2].mean(axis=(0, 1))
        else:
            avg_color = result[y:y2, x:x2].mean()
        # Fill block with average color
        result[y:y2, x:x2] = avg_color"""),
        
        ("negative_effect", "Negative (Invert) Effect",
         """# Create negative effect
result = 255 - result"""),
        
        ("solarize_effect", "Solarize Effect",
         """# Solarize effect
threshold = 128
mask = result > threshold
result[mask] = 255 - result[mask]"""),
        
        ("posterize_effect", "Posterize Effect",
         """# Posterize effect
n_bits = 4
mask = 256 - (1 << n_bits)
result = cv2.bitwise_and(result, mask)"""),
        
        ("cross_process", "Cross Processing Effect",
         """# Simulate cross processing
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
# Adjust individual channels
result[:,:,0] = np.clip(result[:,:,0] * 0.7, 0, 255)  # Blue
result[:,:,1] = np.clip(result[:,:,1] * 1.2, 0, 255)  # Green
result[:,:,2] = np.clip(result[:,:,2] * 1.5, 0, 255)  # Red
result = result.astype(np.uint8)"""),
        
        ("vintage_effect", "Vintage Photo Effect",
         """# Create vintage effect
if len(result.shape) == 2:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
# Add yellow tint
result[:,:,0] = np.clip(result[:,:,0] * 0.7, 0, 255)  # Reduce blue
# Add noise
noise = np.random.normal(0, 10, result.shape)
result = np.clip(result + noise, 0, 255).astype(np.uint8)
# Add vignette
rows, cols = result.shape[:2]
kernel_x = cv2.getGaussianKernel(cols, cols/2.5)
kernel_y = cv2.getGaussianKernel(rows, rows/2.5)
kernel = kernel_y * kernel_x.T
mask = kernel / kernel.max()
for i in range(3):
    result[:,:,i] = (result[:,:,i] * mask).astype(np.uint8)"""),
        
        ("emboss_3d", "3D Emboss Effect",
         """# Create 3D emboss effect
kernel = np.array([[-2, -1, 0],
                   [-1,  1, 1],
                   [ 0,  1, 2]])
if len(result.shape) == 3:
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
else:
    gray = result
embossed = cv2.filter2D(gray, -1, kernel)
embossed = embossed + 128
result = embossed.astype(np.uint8)
if len(image.shape) == 3:
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)"""),
        
        ("kaleidoscope", "Kaleidoscope Effect",
         """# Create kaleidoscope effect
h, w = result.shape[:2]
center = (w // 2, h // 2)
# Get quadrant
quadrant = result[:h//2, :w//2]
# Create mirrored sections
result[:h//2, :w//2] = quadrant
result[:h//2, w//2:] = cv2.flip(quadrant, 1)
result[h//2:, :w//2] = cv2.flip(quadrant, 0)
result[h//2:, w//2:] = cv2.flip(quadrant, -1)"""),
    ]
    
    for name, desc, code, *extra in scripts:
        imports = extra[0] if extra else ""
        content = generate_script_template(name, "effects", desc, code, imports)
        write_script(base_path / "effects" / f"{name}.py", content)

def generate_readme(base_path):
    """Generate README file with all scripts listed"""
    readme_content = """# OpenCV Scripts Collection

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
"""
    
    with open(base_path / "README.md", 'w') as f:
        f.write(readme_content)

def main():
    """Main function to generate all scripts"""
    print("🚀 Generating OpenCV Scripts Collection...")
    
    # Create base directory
    base_path = create_script_directory()
    print(f"✓ Created directory structure at: {base_path}")
    
    # Generate scripts by category
    print("\n📝 Generating scripts...")
    
    print("  → Filtering scripts...")
    generate_filtering_scripts(base_path)
    
    print("  → Morphology scripts...")
    generate_morphology_scripts(base_path)
    
    print("  → Thresholding scripts...")
    generate_thresholding_scripts(base_path)
    
    print("  → Edge detection scripts...")
    generate_edge_detection_scripts(base_path)
    
    print("  → Transformation scripts...")
    generate_transformation_scripts(base_path)
    
    print("  → Histogram scripts...")
    generate_histogram_scripts(base_path)
    
    print("  → Feature detection scripts...")
    generate_feature_scripts(base_path)
    
    print("  → Noise scripts...")
    generate_noise_scripts(base_path)
    
    print("  → Effects scripts...")
    generate_effects_scripts(base_path)
    
    # Generate README
    print("\n📄 Generating README...")
    generate_readme(base_path)
    
    # Count total scripts
    total_scripts = sum(len(list(Path(base_path).glob(f"{cat}/*.py"))) 
                       for cat in ["filtering", "morphology", "thresholding", 
                                  "edge_detection", "transformations", "histogram",
                                  "features", "noise", "effects"])
    
    print(f"\n✅ Successfully generated {total_scripts} OpenCV scripts!")
    print(f"📁 Scripts location: {base_path}")
    print("\n🎯 Next steps:")
    print("1. Copy existing scripts to opencv_scripts/ if needed")
    print("2. Run the automated processing studio")
    print("3. The studio will now have access to all OpenCV operations")

if __name__ == "__main__":
    main()