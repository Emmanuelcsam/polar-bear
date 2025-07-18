#!/usr/bin/env python3
"""
Advanced Crop Learner with Reference Directory
This tool learns from reference cropped images and applies similar cropping to target images.
"""

import subprocess
import os
import sys
import time
import logging
import json
import random
import gc
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

# Configure logging before any output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_crop_learner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages if missing"""
    required_packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'scipy': 'scipy',
        'pygame': 'pygame',
        'psutil': 'psutil',
        'pillow': 'PIL',
        'tqdm': 'tqdm'
    }
    
    logger.info("Checking required libraries...")
    
    for package, module in required_packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.warning(f"✗ {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                sys.exit(1)

# Install requirements before importing
install_requirements()

# Now import all required libraries
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import correlation
import pygame
import psutil
from tqdm import tqdm

# Global settings
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
MAX_IMAGE_SIZE = 10000  # Maximum dimension
MIN_IMAGE_SIZE = 10     # Minimum dimension

# Device setup
is_windows = platform.system() == 'Windows'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize feature extractor
try:
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
    logger.info("Feature extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize feature extractor: {e}")
    feature_extractor = None

def validate_directory(path: str, purpose: str) -> Path:
    """Validate directory path and return Path object"""
    try:
        dir_path = Path(path).resolve()
        if not dir_path.exists():
            raise ValueError(f"{purpose} directory does not exist: {path}")
        if not dir_path.is_dir():
            raise ValueError(f"{purpose} path is not a directory: {path}")
        if not os.access(dir_path, os.R_OK):
            raise ValueError(f"{purpose} directory is not readable: {path}")
        return dir_path
    except Exception as e:
        logger.error(f"Directory validation failed: {e}")
        raise

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    return a / b if b != 0 else default

def extract_deep_features(img: Image.Image) -> Optional[np.ndarray]:
    """Extract deep features using ResNet"""
    if feature_extractor is None:
        return None
    try:
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(input_tensor).squeeze().cpu().numpy()
        return features
    except Exception as e:
        logger.error(f"Deep feature extraction failed: {e}")
        return None

def extract_comprehensive_features(img_path: str) -> Dict[str, Any]:
    """Extract comprehensive features from image with robust error handling"""
    features = {}
    
    try:
        # Read and validate image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Check image dimensions
        h, w = img.shape[:2]
        if h > MAX_IMAGE_SIZE or w > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized large image from {h}x{w} to {new_h}x{new_w}")
        
        if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
            logger.warning(f"Image too small: {h}x{w}")
            return features
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Extract alpha channel if present
        has_alpha = img.shape[2] == 4
        rgb = img[:, :, :3] if has_alpha else img
        alpha = img[:, :, 3] if has_alpha else np.ones(rgb.shape[:2], dtype=np.uint8) * 255
        mask = alpha > 0
        
        # Get foreground pixels
        fg_pixels = rgb[mask]
        if len(fg_pixels) == 0:
            logger.warning(f"No foreground pixels in {img_path}")
            return features
        
        # 1. Basic RGB statistics
        for ch_idx, ch_name in enumerate(['b', 'g', 'r']):
            ch_data = fg_pixels[:, ch_idx].astype(np.float64)
            if len(ch_data) > 0:
                features[f'{ch_name}_min'] = float(np.min(ch_data))
                features[f'{ch_name}_max'] = float(np.max(ch_data))
                features[f'{ch_name}_mean'] = float(np.mean(ch_data))
                features[f'{ch_name}_median'] = float(np.median(ch_data))
                features[f'{ch_name}_std'] = float(np.std(ch_data))
                if len(ch_data) > 1:
                    features[f'{ch_name}_skew'] = float(skew(ch_data))
                    features[f'{ch_name}_kurtosis'] = float(kurtosis(ch_data))
        
        # 2. HSV statistics
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        hsv_fg = hsv[mask]
        for ch_idx, ch_name in enumerate(['h', 's', 'v']):
            ch_data = hsv_fg[:, ch_idx].astype(np.float64)
            if len(ch_data) > 0:
                features[f'hsv_{ch_name}_mean'] = float(np.mean(ch_data))
                features[f'hsv_{ch_name}_std'] = float(np.std(ch_data))
                features[f'hsv_{ch_name}_min'] = float(np.min(ch_data))
                features[f'hsv_{ch_name}_max'] = float(np.max(ch_data))
        
        # 3. Color histograms (normalized)
        for ch_idx, ch_name in enumerate(['b', 'g', 'r']):
            hist = cv2.calcHist([rgb], [ch_idx], mask.astype(np.uint8), [32], [0, 256])
            hist = hist.flatten() / (np.sum(hist) + 1e-7)
            features[f'hist_{ch_name}'] = hist.tolist()
        
        # 4. Channel correlations
        if len(fg_pixels) > 1:
            try:
                features['corr_rg'] = float(np.corrcoef(fg_pixels[:, 2], fg_pixels[:, 1])[0, 1])
                features['corr_rb'] = float(np.corrcoef(fg_pixels[:, 2], fg_pixels[:, 0])[0, 1])
                features['corr_gb'] = float(np.corrcoef(fg_pixels[:, 1], fg_pixels[:, 0])[0, 1])
            except:
                features['corr_rg'] = features['corr_rb'] = features['corr_gb'] = 0.0
        
        # 5. Gradient statistics
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_fg = grad_mag[mask]
        if len(grad_fg) > 0:
            features['grad_mean'] = float(np.mean(grad_fg))
            features['grad_std'] = float(np.std(grad_fg))
            features['grad_max'] = float(np.max(grad_fg))
        
        # 6. Geometric features
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            features['contour_area'] = float(area)
            features['contour_perimeter'] = float(perimeter)
            features['aspect_ratio'] = safe_divide(float(w), float(h), 1.0)
            features['extent'] = safe_divide(area, float(w * h), 0.0)
            features['solidity'] = safe_divide(area, cv2.contourArea(cv2.convexHull(largest_contour)), 1.0)
            features['num_contours'] = len(contours)
            
            # Hu moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            features['hu_moments'] = [float(m) for m in hu_moments]
        
        # 7. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = edges[mask]
        features['edge_density'] = safe_divide(float(np.sum(edge_pixels > 0)), float(np.sum(mask)), 0.0)
        
        # 8. Entropy
        hist_gray = cv2.calcHist([gray], [0], mask.astype(np.uint8), [256], [0, 256]).flatten()
        hist_gray = hist_gray / (hist_gray.sum() + 1e-7)
        hist_gray = hist_gray[hist_gray > 0]
        if len(hist_gray) > 0:
            features['entropy'] = float(-np.sum(hist_gray * np.log2(hist_gray + 1e-7)))
        
        # 9. Deep features (optional)
        if feature_extractor is not None:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                deep_feats = extract_deep_features(pil_img)
                if deep_feats is not None:
                    features['deep_features_mean'] = float(np.mean(deep_feats))
                    features['deep_features_std'] = float(np.std(deep_feats))
                    # Store first 10 components for more detail
                    features['deep_features_components'] = deep_feats[:10].tolist()
            except Exception as e:
                logger.warning(f"Deep feature extraction failed: {e}")
        
        # 10. Image dimensions
        features['width'] = w
        features['height'] = h
        features['has_alpha'] = has_alpha
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction failed for {img_path}: {e}")
        logger.debug(traceback.format_exc())
        return features

def aggregate_features(features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate features from multiple images with robust handling"""
    if not features_list:
        return {}
    
    aggregated = {}
    
    # Get all unique keys
    all_keys = set()
    for f in features_list:
        all_keys.update(f.keys())
    
    for key in all_keys:
        values = [f[key] for f in features_list if key in f]
        if not values:
            continue
        
        if isinstance(values[0], (int, float)):
            # Numeric values - compute statistics
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
            aggregated[f'{key}_median'] = float(np.median(values))
        elif isinstance(values[0], list):
            # List values - average element-wise
            try:
                values_arr = np.array(values)
                aggregated[f'{key}_mean'] = np.mean(values_arr, axis=0).tolist()
                aggregated[f'{key}_std'] = np.std(values_arr, axis=0).tolist()
            except:
                # If can't convert to array, just store first value
                aggregated[key] = values[0]
        elif isinstance(values[0], bool):
            # Boolean values - count ratio
            aggregated[f'{key}_ratio'] = sum(values) / len(values)
    
    return aggregated

def analyze_reference_directory(ref_dir: Path, cache_file: str = 'ref_features.json') -> Dict[str, Any]:
    """Analyze reference directory with progress tracking"""
    ref_paths = [p for p in ref_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]
    
    if not ref_paths:
        raise ValueError(f"No supported images found in {ref_dir}")
    
    logger.info(f"Found {len(ref_paths)} reference images")
    
    # Check cache
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            logger.info(f"Loaded cached features from {cache_file}")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    # Extract features with progress bar
    features_list = []
    with ThreadPoolExecutor(max_workers=min(4 if is_windows else 8, len(ref_paths))) as executor:
        futures = {executor.submit(extract_comprehensive_features, str(path)): path for path in ref_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing references"):
            path = futures[future]
            try:
                features = future.result()
                if features:
                    features_list.append(features)
                else:
                    logger.warning(f"Empty features for {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
    
    if not features_list:
        raise ValueError("Failed to extract features from any reference image")
    
    # Aggregate features
    aggregated = aggregate_features(features_list)
    aggregated['num_references'] = len(features_list)
    
    # Save cache
    try:
        with open(cache_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Saved aggregated features to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
    
    return aggregated

def generate_mask(image: np.ndarray, ref_features: Dict[str, Any], params: Dict[str, float]) -> np.ndarray:
    """Generate foreground mask with multiple strategies"""
    try:
        h, w = image.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Strategy 1: HSV color range
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mult = params.get('color_multiplier', 2.0)
        
        # Build color ranges
        h_mean = ref_features.get('hsv_h_mean_mean', 90)
        h_std = ref_features.get('hsv_h_std_mean', 30) * color_mult
        s_mean = ref_features.get('hsv_s_mean_mean', 128)
        s_std = ref_features.get('hsv_s_std_mean', 50) * color_mult
        v_mean = ref_features.get('hsv_v_mean_mean', 128)
        v_std = ref_features.get('hsv_v_std_mean', 50) * color_mult
        
        # Handle hue wraparound
        h_low = h_mean - h_std
        h_high = h_mean + h_std
        
        if h_low < 0 or h_high > 179:
            # Hue wraps around
            mask1 = cv2.inRange(hsv, 
                               np.array([0, max(0, s_mean - s_std), max(0, v_mean - v_std)]),
                               np.array([h_high % 180, min(255, s_mean + s_std), min(255, v_mean + v_std)]))
            mask2 = cv2.inRange(hsv,
                               np.array([(h_low + 180) % 180, max(0, s_mean - s_std), max(0, v_mean - v_std)]),
                               np.array([179, min(255, s_mean + s_std), min(255, v_mean + v_std)]))
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:
            color_mask = cv2.inRange(hsv,
                                   np.array([h_low, max(0, s_mean - s_std), max(0, v_mean - v_std)]),
                                   np.array([h_high, min(255, s_mean + s_std), min(255, v_mean + v_std)]))
        
        # Morphological operations
        kernel_size = max(3, int(params.get('morph_kernel', 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Strategy 2: Edge-based segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges to create regions
        edge_kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, edge_kernel, iterations=2)
        
        # Find contours in color mask
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on reference features
        ref_area = ref_features.get('contour_area_mean', 1000)
        ref_aspect = ref_features.get('aspect_ratio_mean', 1.0)
        area_tol = params.get('area_tolerance', 0.7)
        aspect_tol = params.get('aspect_tolerance', 0.7)
        
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Skip tiny contours
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
                
            aspect = w / h
            
            # Relaxed filtering
            area_ratio = safe_divide(area, ref_area, 0)
            if 0.1 < area_ratio < 10:  # Within order of magnitude
                aspect_diff = abs(aspect - ref_aspect)
                if aspect_diff < aspect_tol * ref_aspect:
                    valid_contours.append(cnt)
        
        # Draw filtered contours
        if valid_contours:
            cv2.drawContours(final_mask, valid_contours, -1, 255, -1)
        else:
            # Fallback: use largest contour from color mask
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(final_mask, [largest], -1, 255, -1)
            else:
                final_mask = color_mask
        
        # Post-processing
        # Fill holes
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(final_mask, contours, -1, 255, -1)
        
        # Smooth mask
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        return final_mask
        
    except Exception as e:
        logger.error(f"Mask generation failed: {e}")
        # Return a simple threshold mask as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return mask

def apply_crop(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to create cropped image with alpha channel"""
    try:
        if original.shape[2] == 3:
            bgra = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        else:
            bgra = original.copy()
        
        # Ensure mask is same size
        if mask.shape[:2] != bgra.shape[:2]:
            mask = cv2.resize(mask, (bgra.shape[1], bgra.shape[0]))
        
        bgra[:, :, 3] = mask
        return bgra
    except Exception as e:
        logger.error(f"Failed to apply crop: {e}")
        return original

class CropPreviewUI:
    """Interactive UI for previewing and adjusting crops"""
    
    def __init__(self, screen_width: int = 1400, screen_height: int = 900):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Advanced Crop Preview - Press ESC to exit")
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        
    def create_surface_from_image(self, img: np.ndarray) -> pygame.Surface:
        """Convert OpenCV image to pygame surface"""
        if img.shape[2] == 4:  # BGRA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transpose for pygame
        img_rgb = np.transpose(img_rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(img_rgb)
    
    def display_previews(self, originals: List[np.ndarray], previews: List[np.ndarray], 
                        params: Dict[str, float]) -> Tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        """Display original and preview images with UI controls"""
        self.screen.fill((30, 30, 30))
        
        num_images = len(originals)
        padding = 10
        button_height = 60
        param_height = 100
        
        # Calculate layout
        available_width = self.screen.get_width() - padding * 3
        available_height = self.screen.get_height() - button_height - param_height - padding * 4
        
        cols = min(num_images * 2, 6)  # Max 3 image pairs per row
        rows = (num_images * 2 + cols - 1) // cols
        
        cell_width = available_width // cols
        cell_height = available_height // rows
        
        # Display images
        for i in range(num_images):
            row = (i * 2) // cols
            
            # Original
            orig_col = (i * 2) % cols
            orig_x = padding + orig_col * cell_width
            orig_y = padding + row * cell_height
            
            orig_surf = self.create_surface_from_image(originals[i])
            scale = min(cell_width / orig_surf.get_width(), 
                       cell_height / orig_surf.get_height()) * 0.9
            scaled_orig = pygame.transform.scale(orig_surf, 
                                               (int(orig_surf.get_width() * scale),
                                                int(orig_surf.get_height() * scale)))
            
            # Center in cell
            x_offset = (cell_width - scaled_orig.get_width()) // 2
            y_offset = (cell_height - scaled_orig.get_height()) // 2
            self.screen.blit(scaled_orig, (orig_x + x_offset, orig_y + y_offset))
            
            # Label
            label = self.small_font.render("Original", True, (200, 200, 200))
            self.screen.blit(label, (orig_x + 5, orig_y + 5))
            
            # Preview
            prev_col = orig_col + 1
            prev_x = padding + prev_col * cell_width
            prev_y = orig_y
            
            prev_surf = self.create_surface_from_image(previews[i])
            scaled_prev = pygame.transform.scale(prev_surf,
                                               (int(prev_surf.get_width() * scale),
                                                int(prev_surf.get_height() * scale)))
            self.screen.blit(scaled_prev, (prev_x + x_offset, prev_y + y_offset))
            
            # Label
            label = self.small_font.render("Cropped", True, (200, 200, 200))
            self.screen.blit(label, (prev_x + 5, prev_y + 5))
        
        # Parameters display
        param_y = self.screen.get_height() - button_height - param_height - padding * 2
        param_text = f"Color Range: {params['color_multiplier']:.1f} | " \
                    f"Morph Kernel: {params['morph_kernel']} | " \
                    f"Area Tolerance: {params['area_tolerance']:.1f} | " \
                    f"Aspect Tolerance: {params['aspect_tolerance']:.1f}"
        param_surf = self.small_font.render(param_text, True, (255, 255, 255))
        self.screen.blit(param_surf, (padding, param_y))
        
        # Buttons
        button_y = self.screen.get_height() - button_height - padding
        button_width = (self.screen.get_width() - padding * 4) // 3
        
        # Confirm button
        confirm_rect = pygame.Rect(padding, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (50, 200, 50), confirm_rect)
        confirm_text = self.font.render("Confirm & Process All", True, (255, 255, 255))
        text_rect = confirm_text.get_rect(center=confirm_rect.center)
        self.screen.blit(confirm_text, text_rect)
        
        # Adjust button
        adjust_rect = pygame.Rect(padding * 2 + button_width, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (200, 150, 50), adjust_rect)
        adjust_text = self.font.render("Adjust Parameters", True, (255, 255, 255))
        text_rect = adjust_text.get_rect(center=adjust_rect.center)
        self.screen.blit(adjust_text, text_rect)
        
        # Exit button
        exit_rect = pygame.Rect(padding * 3 + button_width * 2, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, (200, 50, 50), exit_rect)
        exit_text = self.font.render("Exit", True, (255, 255, 255))
        text_rect = exit_text.get_rect(center=exit_rect.center)
        self.screen.blit(exit_text, text_rect)
        
        # Instructions
        inst_text = "ESC to exit | Click buttons or use keyboard: C=Confirm, A=Adjust, Q=Quit"
        inst_surf = self.small_font.render(inst_text, True, (150, 150, 150))
        self.screen.blit(inst_surf, (padding, param_y + 30))
        
        pygame.display.flip()
        return confirm_rect, adjust_rect, exit_rect
    
    def run_preview(self, sample_paths: List[str], ref_features: Dict[str, Any], 
                   params: Dict[str, float]) -> Optional[bool]:
        """Run the preview UI and return user choice"""
        # Load images
        originals = []
        for path in sample_paths:
            img = cv2.imread(str(path))
            if img is not None:
                originals.append(img)
        
        if not originals:
            logger.error("Failed to load sample images")
            return None
        
        # Generate initial previews
        masks = [generate_mask(orig, ref_features, params) for orig in originals]
        previews = [apply_crop(orig, mask) for orig, mask in zip(originals, masks)]
        
        running = True
        result = None
        
        while running:
            confirm_rect, adjust_rect, exit_rect = self.display_previews(originals, previews, params)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    result = None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                        result = None
                    elif event.key == pygame.K_c:
                        running = False
                        result = True
                    elif event.key == pygame.K_a:
                        running = False
                        result = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if confirm_rect.collidepoint(pos):
                        running = False
                        result = True
                    elif adjust_rect.collidepoint(pos):
                        running = False
                        result = False
                    elif exit_rect.collidepoint(pos):
                        running = False
                        result = None
            
            self.clock.tick(30)
        
        return result
    
    def quit(self):
        """Clean up pygame"""
        pygame.quit()

def adjust_parameters(params: Dict[str, float], iteration: int) -> Dict[str, float]:
    """Adjust parameters based on iteration"""
    # Progressive adjustment strategy
    adjustments = {
        'color_multiplier': 0.3 + (iteration * 0.2),
        'area_tolerance': 0.1 + (iteration * 0.15),
        'aspect_tolerance': 0.1 + (iteration * 0.15),
        'morph_kernel': 2
    }
    
    for key, value in adjustments.items():
        if key == 'morph_kernel':
            params[key] = min(15, params.get(key, 5) + value)
        else:
            params[key] = min(3.0, params.get(key, 1.0) + value)
    
    logger.info(f"Adjusted parameters (iteration {iteration}): {params}")
    return params

def process_directory(target_dir: Path, output_dir: Path, ref_features: Dict[str, Any], 
                     params: Dict[str, float], batch_size: int = 100):
    """Process all images in directory with progress tracking"""
    # Find all images
    target_paths = [p for p in target_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]
    
    if not target_paths:
        logger.warning(f"No images found in {target_dir}")
        return
    
    logger.info(f"Processing {len(target_paths)} images...")
    
    # Process in batches to manage memory
    total_processed = 0
    total_failed = 0
    
    for i in range(0, len(target_paths), batch_size):
        batch = target_paths[i:i + batch_size]
        batch_failed = 0
        
        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = {}
            
            for path in batch:
                future = executor.submit(process_single_image, path, output_dir, 
                                       ref_features, params)
                futures[future] = path
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Batch {i//batch_size + 1}"):
                try:
                    success = future.result()
                    if success:
                        total_processed += 1
                    else:
                        total_failed += 1
                        batch_failed += 1
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    total_failed += 1
                    batch_failed += 1
        
        # Garbage collection between batches
        gc.collect()
        
        logger.info(f"Batch complete: {len(batch) - batch_failed}/{len(batch)} successful")
    
    logger.info(f"Processing complete: {total_processed} successful, {total_failed} failed")

def process_single_image(img_path: Path, output_dir: Path, ref_features: Dict[str, Any], 
                        params: Dict[str, float]) -> bool:
    """Process a single image"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Generate mask and crop
        mask = generate_mask(img, ref_features, params)
        cropped = apply_crop(img, mask)
        
        # Save with original name + .png
        output_path = output_dir / f"{img_path.stem}_cropped.png"
        cv2.imwrite(str(output_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {img_path}: {e}")
        return False

def main():
    """Main program flow"""
    logger.info("=== Advanced Crop Learner Started ===")
    
    try:
        # Get directories from user
        print("\n" + "="*60)
        print("ADVANCED CROP LEARNER WITH REFERENCE DIRECTORY")
        print("="*60)
        
        # Reference directory
        while True:
            ref_input = input("\nEnter the path to the REFERENCE directory (cropped examples): ").strip()
            if not ref_input:
                logger.error("No path provided")
                continue
            try:
                ref_dir = validate_directory(ref_input, "Reference")
                break
            except ValueError as e:
                logger.error(str(e))
        
        # Target directory
        while True:
            target_input = input("\nEnter the path to the TARGET directory (images to crop): ").strip()
            if not target_input:
                logger.error("No path provided")
                continue
            try:
                target_dir = validate_directory(target_input, "Target")
                break
            except ValueError as e:
                logger.error(str(e))
        
        # Output directory
        while True:
            output_input = input("\nEnter the path to the OUTPUT directory: ").strip()
            if not output_input:
                logger.error("No path provided")
                continue
            try:
                output_dir = Path(output_input).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                if not os.access(output_dir, os.W_OK):
                    raise ValueError(f"Output directory is not writable: {output_dir}")
                break
            except Exception as e:
                logger.error(f"Output directory error: {e}")
        
        # Analyze reference images
        logger.info("\nAnalyzing reference images...")
        ref_features = analyze_reference_directory(ref_dir)
        logger.info(f"Extracted {len(ref_features)} feature types from references")
        
        # Select sample images for preview
        all_targets = list(target_dir.glob('*'))
        all_targets = [p for p in all_targets if p.suffix.lower() in SUPPORTED_FORMATS]
        
        if not all_targets:
            raise ValueError(f"No supported images found in {target_dir}")
        
        num_samples = min(6, len(all_targets))
        sample_paths = random.sample(all_targets, num_samples)
        logger.info(f"Selected {num_samples} sample images for preview")
        
        # Initial parameters
        params = {
            'color_multiplier': 1.5,
            'morph_kernel': 5,
            'area_tolerance': 0.5,
            'aspect_tolerance': 0.5
        }
        
        # Preview UI loop
        ui = CropPreviewUI()
        confirmed = False
        iteration = 0
        max_iterations = 10
        
        logger.info("\nStarting preview mode...")
        print("\n" + "-"*60)
        print("PREVIEW MODE")
        print("- Review the cropping results")
        print("- Click 'Confirm' if satisfied")
        print("- Click 'Adjust' to try different parameters")
        print("- Press ESC or click 'Exit' to cancel")
        print("-"*60)
        
        while not confirmed and iteration < max_iterations:
            result = ui.run_preview(sample_paths, ref_features, params)
            
            if result is None:
                # User cancelled
                logger.info("User cancelled operation")
                ui.quit()
                return
            elif result:
                # User confirmed
                confirmed = True
                logger.info("User confirmed parameters")
            else:
                # User wants adjustment
                iteration += 1
                params = adjust_parameters(params, iteration)
                logger.info(f"Adjusting parameters (attempt {iteration}/{max_iterations})")
        
        ui.quit()
        
        if not confirmed:
            logger.warning("Maximum adjustment iterations reached")
            return
        
        # Process all images
        print("\n" + "="*60)
        print("PROCESSING ALL IMAGES")
        print("="*60)
        
        process_directory(target_dir, output_dir, ref_features, params)
        
        # Summary
        output_count = len(list(output_dir.glob('*.png')))
        print("\n" + "="*60)
        print(f"PROCESSING COMPLETE")
        print(f"Output directory: {output_dir}")
        print(f"Images processed: {output_count}")
        print("="*60)
        
        logger.info("=== Advanced Crop Learner Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Program error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nERROR: {e}")
        print("Check 'advanced_crop_learner.log' for details")
        sys.exit(1)

if __name__ == "__main__":
    main()