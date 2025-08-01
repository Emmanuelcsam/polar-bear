import cv2
import torch
import numpy as np
import pickle
from scipy.signal import correlate2d
from scipy.stats import skew, kurtosis
import torch.nn.functional as F

def compute_glcm(image, dx, dy, levels=256):
    glcm = np.zeros((levels, levels), dtype=np.float64)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if 0 <= i + dy < rows and 0 <= j + dx < cols:
                p1 = image[i, j]
                p2 = image[i + dy, j + dx]
                glcm[p1, p2] += 1
    # Make symmetric
    glcm += glcm.T
    # Normalize
    total = np.sum(glcm)
    if total > 0:
        glcm /= total
    return glcm

def haralick_features(glcm, levels=256):
    i, j = np.meshgrid(np.arange(levels), np.arange(levels))
    contrast = np.sum((i - j) ** 2 * glcm)
    dissimilarity = np.sum(np.abs(i - j) * glcm)
    homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
    asm = np.sum(glcm ** 2)
    energy = np.sqrt(asm)
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    var_i = np.sum((i - mean_i) ** 2 * glcm)
    var_j = np.sum((j - mean_j) ** 2 * glcm)
    std_i = np.sqrt(var_i) if var_i > 0 else 1
    std_j = np.sqrt(var_j) if var_j > 0 else 1
    correlation = np.sum((i - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'asm': asm
    }

def compute_lbp(image, radius=1, points=8):
    rows, cols = image.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = image[i, j]
            code = 0
            for k in range(points):
                angle = 2 * np.pi * k / points
                ni = i - radius * np.sin(angle)
                nj = j + radius * np.cos(angle)
                neighbor = image[int(round(ni)), int(round(nj))]
                if neighbor >= center:
                    code |= 1 << k
            lbp[i, j] = code
    return lbp

def extract_advanced_features(img, device):
    # img is uint8 grayscale
    advanced_features = {}

    # Harris Corner Detection
    harris_params = {'blockSize': 2, 'ksize': 3, 'k': 0.04, 'threshold': 0.01}
    dst = cv2.cornerHarris(img, **{k: v for k, v in harris_params.items() if k != 'threshold'})
    threshold = harris_params['threshold'] * dst.max()
    corners = np.argwhere(dst > threshold)
    num_corners = len(corners)
    advanced_features['harris'] = {
        'method': 'Harris Corner Detection',
        'params': harris_params,
        'features': {'positions': corners.tolist()},
        'stats': {'num_corners': num_corners}
    }

    # SIFT
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp_data = [(p.pt[0], p.pt[1], p.size, p.angle, p.response) for p in kp]
    num_kp = len(kp)
    avg_des = np.mean(des, axis=0).tolist() if des is not None else None
    des_list = des.tolist() if des is not None else None
    advanced_features['sift'] = {
        'method': 'SIFT',
        'params': {},
        'features': {'keypoints': kp_data, 'descriptors': des_list},
        'stats': {'num_keypoints': num_kp, 'avg_descriptor': avg_des}
    }

    # ORB
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    kp_data = [(p.pt[0], p.pt[1], p.size, p.angle, p.response) for p in kp]
    num_kp = len(kp)
    avg_des = np.mean(des, axis=0).tolist() if des is not None else None
    des_list = des.tolist() if des is not None else None
    advanced_features['orb'] = {
        'method': 'ORB',
        'params': {},
        'features': {'keypoints': kp_data, 'descriptors': des_list},
        'stats': {'num_keypoints': num_kp, 'avg_descriptor': avg_des}
    }

    # HOG
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h = h.flatten()
    hog_params = {'winSize': hog.winSize, 'blockSize': hog.blockSize, 'blockStride': hog.blockStride, 'cellSize': hog.cellSize, 'nbins': hog.nbins}
    advanced_features['hog'] = {
        'method': 'HOG',
        'params': hog_params,
        'features': {'hog_vector': h.tolist()},
        'stats': {'length': len(h), 'mean': np.mean(h), 'std': np.std(h)}
    }

    # LBP
    lbp_params = {'radius': 1, 'points': 8}
    lbp_img = compute_lbp(img, **lbp_params)
    lbp_hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256]).flatten()
    advanced_features['lbp'] = {
        'method': 'Local Binary Patterns',
        'params': lbp_params,
        'features': {'lbp_histogram': lbp_hist.tolist()},
        'stats': {'mean': -np.sum((lbp_hist / lbp_hist.sum()) * np.log2(lbp_hist / lbp_hist.sum() + 1e-10)), 'max_bin': np.argmax(lbp_hist)}
    }

    # GLCM Haralick (average over 4 directions)
    directions = [(0,1), (1,1), (1,0), (1,-1)]  # 0, 45, 90, 90 degrees (since -45 same as 135 symmetric)
    glcm_params = {'distances': 1, 'angles': '0, 45, 135, 90 degrees'}
    haralick_avg = {'contrast':0, 'dissimilarity':0, 'homogeneity':0, 'energy':0, 'correlation':0, 'asm':0}
    num_dirs = len(directions)
    for dx, dy in directions:
        glcm = compute_glcm(img, dx, dy)
        har = haralick_features(glcm)
        for k in haralick_avg:
            haralick_avg[k] += har[k] / num_dirs
    advanced_features['haralick'] = {
        'method': 'Haralick Texture Features from GLCM',
        'params': glcm_params,
        'features': {'haralick_values': haralick_avg},
        'stats': haralick_avg  # same
    }

    return advanced_features

def main():
    # Hardcoded paths for automatic execution (change as needed)
    input_file = "variations_data.pkl"
    output_file = "extended_features_data.pkl"

    # Load data from previous script
    with open(input_file, 'rb') as f:
        results = pickle.load(f)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process each variation
    for name, data in results.items():
        # Get the image (convert back to uint8)
        enhanced_float = data['processed_image']
        img = (enhanced_float * 255).astype(np.uint8)

        # Extract advanced features
        advanced = extract_advanced_features(img, device)

        # Add to data
        data['advanced_features'] = advanced

    # Save extended results
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Extended data saved to {output_file}")

if __name__ == "__main__":
    main()
