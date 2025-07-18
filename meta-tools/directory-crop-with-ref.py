import subprocess
import os
import sys
import numpy as np
import time
import logging
import json
import random
import cv2
from PIL import Image
import torch
import torchvision
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import correlation
import pygame
import gc
import platform
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

# Log file setup
log_file_path = 'advanced_crop_learner.log'
log_file = open(log_file_path, 'w')

def log_message(message):
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

log_message("Script started. Checking for required libraries...")

# Check and install missing libraries
required_libs = {
    'numpy': 'numpy',
    'opencv-python': 'cv2',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'scipy': 'scipy',
    'pygame': 'pygame',
    'psutil': 'psutil',
    'pillow': 'PIL'
}

for pkg, module in required_libs.items():
    try:
        __import__(module)
        log_message(f"{pkg} is already installed.")
    except ImportError:
        log_message(f"{pkg} is not installed. Installing now...")
        subprocess.call([sys.executable, '-m', 'pip', 'install', pkg])
        log_message(f"{pkg} installed successfully.")

log_message("All required libraries are installed and ready.")

# Device setup
is_windows = platform.system() == 'Windows'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_message(f"Using device: {device}")

# Feature extractor using ResNet
feature_extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
feature_extractor.fc = torch.nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_deep_features(img: Image.Image) -> np.ndarray:
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(input_tensor).squeeze().cpu().numpy()
    return features

def extract_comprehensive_features(img_path: str) -> Dict[str, Any]:
    """
    Extract every possible statistic and characteristic from the image.
    This includes pixel-level stats, correlations, trends, geometries, intensities, etc.
    """
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to load image.")
        
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        has_alpha = img.shape[2] == 4
        rgb = img[:, :, :3] if has_alpha else img
        alpha = img[:, :, 3] if has_alpha else np.ones(rgb.shape[:2], dtype=np.uint8) * 255
        mask = alpha > 0
        
        # Foreground only
        fg_pixels = rgb[mask]
        if len(fg_pixels) == 0:
            raise ValueError("No foreground pixels.")
        
        # 1. Basic statistics per channel
        stats_dict = {}
        for ch in range(3):
            ch_data = fg_pixels[:, ch].astype(np.float64)
            stats_dict[f'ch{ch}_min'] = np.min(ch_data)
            stats_dict[f'ch{ch}_max'] = np.max(ch_data)
            stats_dict[f'ch{ch}_mean'] = np.mean(ch_data)
            stats_dict[f'ch{ch}_median'] = np.median(ch_data)
            stats_dict[f'ch{ch}_std'] = np.std(ch_data)
            stats_dict[f'ch{ch}_skew'] = skew(ch_data)
            stats_dict[f'ch{ch}_kurtosis'] = kurtosis(ch_data)
        
        # HSV statistics
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        hsv_fg = hsv[mask]
        for ch in range(3):
            ch_data = hsv_fg[:, ch].astype(np.float64)
            stats_dict[f'hsv_ch{ch}_min'] = np.min(ch_data)
            stats_dict[f'hsv_ch{ch}_max'] = np.max(ch_data)
            stats_dict[f'hsv_ch{ch}_mean'] = np.mean(ch_data)
            stats_dict[f'hsv_ch{ch}_median'] = np.median(ch_data)
            stats_dict[f'hsv_ch{ch}_std'] = np.std(ch_data)
            stats_dict[f'hsv_ch{ch}_skew'] = skew(ch_data)
            stats_dict[f'hsv_ch{ch}_kurtosis'] = kurtosis(ch_data)
        
        # 2. Histograms (normalized)
        hist_rgb = [cv2.calcHist([rgb], [i], mask.astype(np.uint8) * 255, [256], [0, 256]).flatten() / np.sum(mask) for i in range(3)]
        stats_dict['hist_r'] = hist_rgb[2].tolist()  # BGR to RGB
        stats_dict['hist_g'] = hist_rgb[1].tolist()
        stats_dict['hist_b'] = hist_rgb[0].tolist()
        
        # 3. Pixel correlations between channels
        corr_rg = correlation(fg_pixels[:, 2], fg_pixels[:, 1])  # R-G
        corr_rb = correlation(fg_pixels[:, 2], fg_pixels[:, 0])  # R-B
        corr_gb = correlation(fg_pixels[:, 1], fg_pixels[:, 0])  # G-B
        stats_dict['corr_rg'] = corr_rg
        stats_dict['corr_rb'] = corr_rb
        stats_dict['corr_gb'] = corr_gb
        
        # 4. Pixel trends (row/col means)
        row_means = np.mean(rgb, axis=1)
        col_means = np.mean(rgb, axis=0)
        stats_dict['row_means_r'] = row_means[:, 2].tolist() if row_means.ndim > 1 else row_means.tolist()
        stats_dict['col_means_r'] = col_means[:, 2].tolist() if col_means.ndim > 1 else col_means.tolist()
        # Similarly for G,B but to save space, only R as example; expand if needed
        
        # 5. Intensity profiles (gradients)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        stats_dict['grad_mean'] = np.mean(grad_mag[mask])
        stats_dict['grad_std'] = np.std(grad_mag[mask])
        
        # 6. Geometries
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten().tolist()
            stats_dict['contour_area'] = area
            stats_dict['contour_perimeter'] = perimeter
            stats_dict['aspect_ratio'] = cv2.boundingRect(largest_contour)[2] / cv2.boundingRect(largest_contour)[3] if len(contours) > 0 else 0
            stats_dict['hu_moments'] = hu_moments
            stats_dict['num_contours'] = len(contours)
        
        # 7. Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges[mask]) / np.sum(mask)
        stats_dict['edge_density'] = edge_density
        
        # 8. Entropy
        hist_gray = cv2.calcHist([gray], [0], mask.astype(np.uint8) * 255, [256], [0, 256]).flatten()
        hist_gray /= hist_gray.sum() + 1e-7
        entropy = -np.sum(hist_gray * np.log2(hist_gray + 1e-7))
        stats_dict['entropy'] = entropy
        
        # 9. Deep features (ResNet)
        pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        deep_feats = extract_deep_features(pil_img)
        stats_dict['deep_features_mean'] = np.mean(deep_feats)
        stats_dict['deep_features_std'] = np.std(deep_feats)
        # To save space, not storing all 2048, but stats; could store PCA if needed
        
        # 10. Covariance matrix for channels
        cov_matrix = np.cov(fg_pixels.T.astype(np.float64))
        stats_dict['cov_matrix'] = cov_matrix.tolist()
        
        return stats_dict
    
    except Exception as e:
        log_message(f"Error extracting features from {img_path}: {e}")
        return {}

def aggregate_features(features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate features from all reference images (average, min, max, etc.)
    """
    if not features_list:
        return {}
    
    agg = {}
    keys = features_list[0].keys()
    for key in keys:
        values = [f[key] for f in features_list if key in f]
        if isinstance(values[0], (int, float)):
            agg[f'{key}_mean'] = np.mean(values)
            agg[f'{key}_std'] = np.std(values)
            agg[f'{key}_min'] = np.min(values)
            agg[f'{key}_max'] = np.max(values)
        elif isinstance(values[0], list):
            # For lists like hist, average
            values_arr = np.array(values)
            agg[f'{key}_mean'] = np.mean(values_arr, axis=0).tolist()
        # For matrices, similar
    return agg

def analyze_reference_directory(ref_dir: str, data_file: str = 'ref_features.json'):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    ref_paths = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith(image_extensions)]
    log_message(f"Found {len(ref_paths)} reference images.")
    
    features_list = []
    with ThreadPoolExecutor(max_workers=4 if is_windows else 8) as executor:
        futures = [executor.submit(extract_comprehensive_features, path) for path in ref_paths]
        for future in futures:
            feat = future.result()
            if feat:
                features_list.append(feat)
    
    aggregated = aggregate_features(features_list)
    
    with open(data_file, 'w') as f:
        json.dump(aggregated, f, indent=4)
    log_message(f"Saved aggregated features to {data_file}")
    
    return aggregated

def load_ref_features(data_file: str = 'ref_features.json') -> Dict[str, Any]:
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    return {}

def generate_mask(image: np.ndarray, ref_features: Dict[str, Any], params: Dict[str, float]) -> np.ndarray:
    """
    Generate foreground mask based on reference features.
    Uses color range in HSV, geometry filtering, etc.
    Params for adjustment: color_multiplier, morph_kernel, etc.
    """
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color range from ref
        h_mean = ref_features.get('hsv_ch0_mean_mean', 0)
        h_std = ref_features.get('hsv_ch0_std_mean', 1)
        s_mean = ref_features.get('hsv_ch1_mean_mean', 0)
        s_std = ref_features.get('hsv_ch1_std_mean', 1)
        v_mean = ref_features.get('hsv_ch2_mean_mean', 0)
        v_std = ref_features.get('hsv_ch2_std_mean', 1)
        
        color_mult = params.get('color_multiplier', 1.0)
        
        lower = np.array([max(0, h_mean - color_mult * h_std), max(0, s_mean - color_mult * s_std), max(0, v_mean - color_mult * v_std)])
        upper = np.array([min(179, h_mean + color_mult * h_std), min(255, s_mean + color_mult * s_std), min(255, v_mean + color_mult * v_std)])
        
        color_mask = cv2.inRange(hsv, lower, upper)
        
        # Morphology
        kernel_size = params.get('morph_kernel', 5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Contours filtering
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        ref_area_mean = ref_features.get('contour_area_mean', 1000)
        ref_aspect = ref_features.get('aspect_ratio_mean', 1.0)
        area_tol = params.get('area_tolerance', 0.5)
        aspect_tol = params.get('aspect_tolerance', 0.5)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if abs(area - ref_area_mean) / ref_area_mean < area_tol:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if abs(aspect - ref_aspect) < aspect_tol:
                    filtered_contours.append(cnt)
        
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        
        # Edge refinement using ref edge density
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask = cv2.bitwise_and(mask, mask, mask=edges)  # Optional refinement
        
        return mask
    
    except Exception as e:
        log_message(f"Error generating mask: {e}")
        return np.zeros(image.shape[:2], np.uint8)

def apply_crop(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA) if original.shape[2] == 3 else original.copy()
    bgra[:, :, 3] = mask
    return bgra

# UI Setup
pygame.init()
screen_width, screen_height = 1200, 800
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
pygame.display.set_caption("Advanced Crop Preview")
font = pygame.font.SysFont(None, 30)

def display_previews(originals, previews):
    screen.fill((0, 0, 0))
    num = len(originals)
    w = screen.get_width() // (2 * num)
    h = screen.get_height() - 100  # Leave space for buttons
    
    for i in range(num):
        # Original
        orig_surf = pygame.surfarray.make_surface(cv2.cvtColor(originals[i], cv2.COLOR_BGR2RGB).swapaxes(0,1))
        scale = min(w / orig_surf.get_width(), h / orig_surf.get_height())
        scaled_orig = pygame.transform.scale(orig_surf, (int(orig_surf.get_width() * scale), int(orig_surf.get_height() * scale)))
        screen.blit(scaled_orig, (i * 2 * w, 0))
        
        # Preview
        prev_img = previews[i]
        prev_rgb = cv2.cvtColor(prev_img, cv2.COLOR_BGRA2RGBA) if prev_img.shape[2] == 4 else prev_img
        prev_surf = pygame.surfarray.make_surface(prev_rgb.swapaxes(0,1))
        scaled_prev = pygame.transform.scale(prev_surf, (int(prev_surf.get_width() * scale), int(prev_surf.get_height() * scale)))
        screen.blit(scaled_prev, ((i * 2 + 1) * w, 0))
    
    # Buttons
    button_height = 50
    confirm_rect = pygame.Rect(0, screen.get_height() - button_height, screen.get_width() // 2, button_height)
    refresh_rect = pygame.Rect(screen.get_width() // 2, screen.get_height() - button_height, screen.get_width() // 2, button_height)
    
    pygame.draw.rect(screen, (0, 255, 0), confirm_rect)
    confirm_text = font.render("Confirm", True, (0, 0, 0))
    screen.blit(confirm_text, (confirm_rect.centerx - confirm_text.get_width() // 2, confirm_rect.centery - confirm_text.get_height() // 2))
    
    pygame.draw.rect(screen, (255, 0, 0), refresh_rect)
    refresh_text = font.render("Refresh (Try Again)", True, (0, 0, 0))
    screen.blit(refresh_text, (refresh_rect.centerx - refresh_text.get_width() // 2, refresh_rect.centery - refresh_text.get_height() // 2))
    
    pygame.display.flip()
    return confirm_rect, refresh_rect

def preview_ui(sample_paths: List[str], ref_features: Dict[str, Any], params: Dict[str, float]):
    originals = [cv2.imread(path) for path in sample_paths]
    masks = [generate_mask(orig, ref_features, params) for orig in originals]
    previews = [apply_crop(orig, m) for orig, m in zip(originals, masks)]
    
    running = True
    while running:
        confirm_rect, refresh_rect = display_previews(originals, previews)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if confirm_rect.collidepoint(pos):
                    running = False
                    return True
                elif refresh_rect.collidepoint(pos):
                    running = False
                    return False  # Refresh
    
    return False

def adjust_params(params: Dict[str, float]) -> Dict[str, float]:
    # Penalize: widen ranges
    params['color_multiplier'] += 0.5
    params['area_tolerance'] += 0.2
    params['aspect_tolerance'] += 0.2
    params['morph_kernel'] += 2
    log_message(f"Adjusted params: {params}")
    return params

def process_all(target_dir: str, output_dir: str, ref_features: Dict[str, Any], params: Dict[str, float]):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    target_paths = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(image_extensions)]
    log_message(f"Processing {len(target_paths)} images...")
    
    def process_one(path):
        try:
            img = cv2.imread(path)
            mask = generate_mask(img, ref_features, params)
            cropped = apply_crop(img, mask)
            output_path = os.path.join(output_dir, os.path.basename(path) + '.png')
            cv2.imwrite(output_path, cropped)
            log_message(f"Processed {path} -> {output_path}")
        except Exception as e:
            log_message(f"Error processing {path}: {e}")
    
    with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
        executor.map(process_one, target_paths)
    
    gc.collect()

# Main flow
log_message("Please enter the full path to the reference directory (cropped photos for learning).")
ref_dir = input().strip()
log_message(f"Reference directory: {ref_dir}")

log_message("Please enter the full path to the target directory (images to crop).")
target_dir = input().strip()
log_message(f"Target directory: {target_dir}")

log_message("Please enter the full path to the output directory.")
output_dir = input().strip()
log_message(f"Output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Analyze references
ref_features = analyze_reference_directory(ref_dir)

# Sample 5 targets for preview
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
all_targets = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(image_extensions)]
sample_paths = random.sample(all_targets, min(5, len(all_targets)))

# Initial params
params = {
    'color_multiplier': 1.5,
    'morph_kernel': 5,
    'area_tolerance': 0.5,
    'aspect_tolerance': 0.5
}

confirmed = False
while not confirmed:
    confirmed = preview_ui(sample_paths, ref_features, params)
    if not confirmed:
        params = adjust_params(params)

# Process all
process_all(target_dir, output_dir, ref_features, params)

pygame.quit()
log_message("Script completed.")
log_file.close()
