import cv2
import torch
import numpy as np
import pickle
from scipy.signal import correlate2d
from scipy.stats import skew, kurtosis
import torch.nn.functional as F


#sends information to detection
#sends information to classification
#applies pooling

def extract_features(img, device):
    # img is uint8 grayscale
    # Statistics
    mean = np.mean(img)
    std = np.std(img)
    min_val = np.min(img)
    max_val = np.max(img)
    median = np.median(img)
    percentiles = np.percentile(img, [25, 50, 75])
    skewness = skew(img.flatten())
    kurt = kurtosis(img.flatten())
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_prob = histogram / histogram.sum()
    entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-10))

    # Geometries: contours and properties
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Mathematical correlations: 2D autocorrelation
    auto_corr = correlate2d(img, img, mode='same')

    # Send to PyTorch: convert to tensors on GPU
    img_tensor = torch.from_numpy(img).float().to(device)
    sobelx_tensor = torch.from_numpy(sobelx).float().to(device)
    sobely_tensor = torch.from_numpy(sobely).float().to(device)
    grad_mag_tensor = torch.from_numpy(grad_mag).float().to(device)
    auto_corr_tensor = torch.from_numpy(auto_corr).float().to(device)
    histogram_tensor = torch.from_numpy(histogram).float().to(device)

    # Example operation on GPU: normalize img tensor
    img_tensor = img_tensor / 255.0

    # Bring back to CPU for saving (as numpy)
    processed_image = img_tensor.cpu().numpy()
    sobelx = sobelx_tensor.cpu().numpy()
    sobely = sobely_tensor.cpu().numpy()
    grad_mag = grad_mag_tensor.cpu().numpy()
    auto_corr = auto_corr_tensor.cpu().numpy()
    histogram = histogram_tensor.cpu().numpy()

    # Segment the image into 4 quadrants
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    segments = [
        img[0:h2, 0:w2],
        img[0:h2, w2:w],
        img[h2:h, 0:w2],
        img[h2:h, w2:w]
    ]
    segment_features = []
    for seg in segments:
        if seg.size == 0:
            continue
        mean_seg = np.mean(seg)
        std_seg = np.std(seg)
        min_seg = np.min(seg)
        max_seg = np.max(seg)
        median_seg = np.median(seg)
        percentiles_seg = np.percentile(seg, [25, 50, 75])
        skewness_seg = skew(seg.flatten())
        kurt_seg = kurtosis(seg.flatten())
        # Pooling in PyTorch
        seg_tensor = torch.from_numpy(seg).float().to(device) / 255.0
        avg_pool_seg = F.adaptive_avg_pool2d(seg_tensor.unsqueeze(0).unsqueeze(0), output_size=(1,1)).squeeze().item()
        max_pool_seg = F.adaptive_max_pool2d(seg_tensor.unsqueeze(0).unsqueeze(0), output_size=(1,1)).squeeze().item()
        segment_features.append({
            'mean': mean_seg,
            'std': std_seg,
            'min': min_seg,
            'max': max_seg,
            'median': median_seg,
            'percentiles': percentiles_seg,
            'skewness': skewness_seg,
            'kurtosis': kurt_seg,
            'avg_pool': avg_pool_seg,
            'max_pool': max_pool_seg
        })

    # Collect all information in a dict
    data = {
        'processed_image': processed_image,  # Full matrix, normalized 0-1
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
        'auto_correlation': auto_corr,
        'segment_features': segment_features
    }

    return data

def main():
    # Hardcoded paths for automatic execution (change as needed)
    input_file = "output_data.pkl"
    output_file = "variations_data.pkl"

    # Load data from previous script
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    enhanced_float = data['enhanced_image']
    img = (enhanced_float * 255).astype(np.uint8)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define individual variations
    variations = {
        'original': lambda x: x,
        'gaussian_blur': lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        'median_blur': lambda x: cv2.medianBlur(x, 5),
        'bilateral_filter': lambda x: cv2.bilateralFilter(x, 9, 75, 75),
        'canny_edges': lambda x: cv2.Canny(x, 100, 200),
        'laplacian': lambda x: cv2.convertScaleAbs(cv2.Laplacian(x, cv2.CV_64F)),
        'otsu_threshold': lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'adaptive_threshold': lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        'erode': lambda x: cv2.erode(x, np.ones((3, 3), np.uint8), iterations=1),
        'dilate': lambda x: cv2.dilate(x, np.ones((3, 3), np.uint8), iterations=1),
        'opening': lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
        'closing': lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
    }

    # Define combination variations
    def gaussian_then_canny(x):
        blurred = variations['gaussian_blur'](x)
        return variations['canny_edges'](blurred)

    def median_then_otsu(x):
        med = variations['median_blur'](x)
        return variations['otsu_threshold'](med)

    def bilateral_then_laplacian(x):
        bil = variations['bilateral_filter'](x)
        return cv2.convertScaleAbs(cv2.Laplacian(bil, cv2.CV_64F))

    def opening_then_canny(x):
        opened = variations['opening'](x)
        return variations['canny_edges'](opened)

    def erode_then_dilate(x):
        eroded = variations['erode'](x)
        return variations['dilate'](eroded)

    variations['gaussian_then_canny'] = gaussian_then_canny
    variations['median_then_otsu'] = median_then_otsu
    variations['bilateral_then_laplacian'] = bilateral_then_laplacian
    variations['opening_then_canny'] = opening_then_canny
    variations['erode_then_dilate'] = erode_then_dilate

    # Process each variation
    results = {}
    for name, func in variations.items():
        processed = func(img)
        if processed.dtype != np.uint8:
            processed = cv2.convertScaleAbs(processed).astype(np.uint8)
        features = extract_features(processed, device)
        results[name] = features

    # Save results to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
