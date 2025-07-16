import cv2
import torch
import numpy as np
import pickle
from scipy.signal import correlate2d
from scipy.stats import skew, kurtosis



#Send to Ruleset
#Send operations to logger
#Send to parameter-control


def main():
    # Hardcoded paths for automatic execution (change as needed)
    input_image = "input_image.jpg"
    output_file = "output_data.pkl"

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load image
    img = cv2.imread(input_image)
    if img is None:
        raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Extract characteristics

    # Pixel intensities: the enhanced matrix

    # Statistics
    mean = np.mean(enhanced)
    std = np.std(enhanced)
    min_val = np.min(enhanced)
    max_val = np.max(enhanced)
    median = np.median(enhanced)
    percentiles = np.percentile(enhanced, [25, 50, 75])
    skewness = skew(enhanced.flatten())
    kurt = kurtosis(enhanced.flatten())
    histogram = cv2.calcHist([enhanced], [0], None, [256], [0, 256]).flatten()
    hist_prob = histogram / histogram.sum()
    entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-10))

    # Geometries: contours and properties
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_props = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten().tolist()
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            cx, cy = 0, 0
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        contour_props.append({
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'hu_moments': hu_moments,
            'num_vertices': num_vertices,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        })

    # Gradients for edges/geometries
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Mathematical correlations: 2D autocorrelation
    auto_corr = correlate2d(enhanced, enhanced, mode='same')

    # Send to PyTorch: convert to tensors on GPU
    enhanced_tensor = torch.from_numpy(enhanced).float().to(device)
    sobelx_tensor = torch.from_numpy(sobelx).float().to(device)
    sobely_tensor = torch.from_numpy(sobely).float().to(device)
    grad_mag_tensor = torch.from_numpy(grad_mag).float().to(device)
    auto_corr_tensor = torch.from_numpy(auto_corr).float().to(device)
    histogram_tensor = torch.from_numpy(histogram).float().to(device)

    # Example operation on GPU: normalize enhanced tensor
    enhanced_tensor = enhanced_tensor / 255.0

    # Bring back to CPU for saving (as numpy)
    enhanced = enhanced_tensor.cpu().numpy()
    sobelx = sobelx_tensor.cpu().numpy()
    sobely = sobely_tensor.cpu().numpy()
    grad_mag = grad_mag_tensor.cpu().numpy()
    auto_corr = auto_corr_tensor.cpu().numpy()
    histogram = histogram_tensor.cpu().numpy()

    # Collect all information in a dict (enough to recreate image and more)
    data = {
        'enhanced_image': enhanced,  # Full matrix for recreation
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'median': median,
        'percentiles': percentiles,
        'skewness': skewness,
        'kurtosis': kurt,
        'entropy': entropy,
        'histogram': histogram,
        'contour_properties': contour_props,
        'sobel_x': sobelx,
        'sobel_y': sobely,
        'gradient_magnitude': grad_mag,
        'auto_correlation': auto_corr
    }

    # Send out to other scripts: save as pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
