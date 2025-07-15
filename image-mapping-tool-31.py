"""
Create Intensity Visualization
Converts the input image into a colorized intensity map, similar to a heatmap.
"""
import cv2
import numpy as np

# A dictionary to map user-friendly colormap names to the actual OpenCV constants
# This approach is robust to different OpenCV versions.
COLORMAPS = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "spring": cv2.COLORMAP_SPRING,
    "summer": cv2.COLORMAP_SUMMER,
    "autumn": cv2.COLORMAP_AUTUMN,
    "winter": cv2.COLORMAP_WINTER,
    "rainbow": cv2.COLORMAP_RAINBOW,
    "ocean": cv2.COLORMAP_OCEAN,
}

# Conditionally add colormaps that might not exist in all versions
if hasattr(cv2, 'COLORMAP_GIST_EARTH'):
    COLORMAPS["gist_earth"] = cv2.COLORMAP_GIST_EARTH
if hasattr(cv2, 'COLORMAP_TERRAIN'):
    COLORMAPS["terrain"] = cv2.COLORMAP_TERRAIN


def process_image(image: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """
    Generates a colorized intensity map from the input image.

    This script visualizes the intensity levels of an image by applying a
    colormap. It first converts the image to grayscale, then maps the
    brightness of each pixel to a specific color based on the chosen colormap.

    Args:
        image: The input image (will be converted to grayscale for analysis).
        colormap: The name of the colormap to apply.
                  (e.g., viridis, plasma, inferno, jet, hot).

    Returns:
        A BGR color image representing the intensity map.
    """
    # Create a copy to avoid modifying the original pipeline image
    result = image.copy()

    # --- 1. Ensure the image is grayscale for analysis ---
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result

    # --- 2. Select the colormap ---
    selected_colormap = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_VIRIDIS)

    # --- 3. Apply the colormap to the grayscale image ---
    intensity_map = cv2.applyColorMap(gray, selected_colormap)

    return intensity_map

if __name__ == '__main__':
    # Create a sample black-to-white gradient image for testing
    test_image = np.zeros((400, 400), dtype=np.uint8)
    for i in range(400):
        test_image[:, i] = int((i / 400) * 255)

    # Test the function with a specific colormap
    processed = process_image(test_image, colormap="jet")

    # Display the original and processed images using OpenCV
    cv2.imshow("Original Test Image", test_image)
    cv2.imshow("Processed Intensity Map", processed)

    print("Press any key to close test windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
