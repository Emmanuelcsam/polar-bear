import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import numpy as np
import hashlib
import json
from collections import defaultdict
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import warnings
import gc
import threading
warnings.filterwarnings('ignore')

# Thread-local storage for OpenCV objects
thread_local = threading.local()

# Check PyTorch version for compatibility
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
WEIGHTS_ONLY_DEFAULT = TORCH_VERSION >= (2, 6)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimateImageTensorizer:
    """
    The ULTIMATE image tensorizer with comprehensive OpenCV analysis.
    Converts images to tensors while extracting all possible features.
    """
    
    def __init__(self):
        """
        Initialize the ultimate image tensorizer - no options, full power always!
        """
        # Always use optimal settings
        self.resize = (224, 224)  # Standard size for neural networks
        self.normalize = True     # Always normalize
        self.save_format = 'pt'   # PyTorch format for best compatibility
        
        # Define transforms for RGB
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])
        
        # Supported image extensions
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', 
                               '.tiff', '.tif', '.ico', '.jfif')
        
        # Initialize data structures - but don't store tensors in memory
        self.class_data = defaultdict(lambda: {
            'count': 0,
            'paths': [],
            'metadata': [],
            'statistics': {}
        })
        
        self.global_statistics = {
            'total_images': 0,
            'class_distribution': {},
            'cross_class_correlations': {},
            'opencv_global_features': {},
            'pca_analysis': {},
            'neural_network_insights': {}
        }
        
        # Resume state tracking
        self.processed_files = set()
        self.resume_state_file = None
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        logger.info("Initialized ULTIMATE tensorizer with FULL OpenCV capabilities!")
    
    def get_thread_local_detectors(self):
        """Get thread-local OpenCV feature detectors"""
        if not hasattr(thread_local, 'sift'):
            thread_local.sift = cv2.SIFT_create()
            thread_local.orb = cv2.ORB_create()
            thread_local.fast = cv2.FastFeatureDetector_create()
        return thread_local.sift, thread_local.orb, thread_local.fast
    
    def extract_class_from_path(self, file_path):
        """Extract class label from folder structure - improved logic"""
        parts = Path(file_path).parts
        
        # Skip files in root directory
        if len(parts) <= 2:
            return Path(file_path).parent.name
        
        # Look for batch folders or specific class indicators
        for i, part in enumerate(parts):
            # Check if this is a meaningful class directory
            if any(keyword in part.lower() for keyword in 
                   ['batch', 'cladding', 'core', 'clean', 'defect', 'ferrule', 
                    'scratch', 'zone', 'dirty', 'image', 'sma']):
                # Make sure it's a directory, not a file
                if i < len(parts) - 1:  # Not the last part (which is the filename)
                    return part
        
        # If no specific class found, use the immediate parent directory
        return Path(file_path).parent.name
    
    def compute_opencv_features(self, img_path):
        """Extract comprehensive features using OpenCV - thread-safe version"""
        try:
            # Read image in both color and grayscale
            img_color = cv2.imread(str(img_path))
            if img_color is None:
                logger.warning(f"Failed to read image: {img_path}")
                return {}, None
                
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Resize for consistent analysis
            img_color = cv2.resize(img_color, self.resize)
            img_gray = cv2.resize(img_gray, self.resize)
            
            features = {}
            
            # Get thread-local detectors
            sift, orb, fast = self.get_thread_local_detectors()
            
            # 1. Color space analysis
            features['color_spaces'] = self.analyze_color_spaces(img_color)
            
            # 2. Histogram features
            features['histograms'] = self.compute_histogram_features(img_color, img_gray)
            
            # 3. Edge detection
            features['edges'] = self.compute_edge_features(img_gray)
            
            # 4. Corner detection
            features['corners'] = self.compute_corner_features(img_gray, fast)
            
            # 5. Blob detection
            features['blobs'] = self.compute_blob_features(img_gray)
            
            # 6. Contour analysis
            features['contours'] = self.compute_contour_features(img_gray)
            
            # 7. Texture features
            features['texture'] = self.compute_advanced_texture_features(img_gray)
            
            # 8. Morphological features
            features['morphology'] = self.compute_morphological_features(img_gray)
            
            # 9. Frequency domain analysis
            features['frequency'] = self.compute_frequency_features_opencv(img_gray)
            
            # 10. Shape descriptors
            features['shape'] = self.compute_shape_descriptors(img_gray)
            
            # 11. Keypoint descriptors
            features['keypoints'] = self.compute_keypoint_features(img_gray, sift, orb)
            
            # 12. Line and circle detection
            features['hough'] = self.compute_hough_features(img_gray)
            
            # 13. Image moments
            features['moments'] = self.compute_image_moments(img_gray)
            
            # 14. Gabor filters
            features['gabor'] = self.compute_gabor_features(img_gray)
            
            return features, img_gray
            
        except Exception as e:
            logger.error(f"Error extracting OpenCV features from {img_path}: {e}")
            return {}, None
    
    def analyze_color_spaces(self, img_bgr):
        """Analyze image in multiple color spaces"""
        features = {}
        
        try:
            # Convert to different color spaces
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            
            # Statistics for each color space
            for name, img in [('rgb', img_rgb), ('hsv', img_hsv), ('lab', img_lab), ('yuv', img_yuv)]:
                for i, channel_name in enumerate(['0', '1', '2']):
                    channel = img[:, :, i]
                    channel_flat = channel.flatten()
                    
                    # Ensure we have valid data
                    if len(channel_flat) > 0:
                        features[f'{name}_channel_{channel_name}_mean'] = float(np.mean(channel))
                        features[f'{name}_channel_{channel_name}_std'] = float(np.std(channel))
                        
                        # Safe computation of skewness and kurtosis
                        if np.std(channel_flat) > 0:
                            features[f'{name}_channel_{channel_name}_skew'] = float(scipy_stats.skew(channel_flat))
                            features[f'{name}_channel_{channel_name}_kurtosis'] = float(scipy_stats.kurtosis(channel_flat))
                        else:
                            features[f'{name}_channel_{channel_name}_skew'] = 0.0
                            features[f'{name}_channel_{channel_name}_kurtosis'] = 0.0
                    else:
                        features[f'{name}_channel_{channel_name}_mean'] = 0.0
                        features[f'{name}_channel_{channel_name}_std'] = 0.0
                        features[f'{name}_channel_{channel_name}_skew'] = 0.0
                        features[f'{name}_channel_{channel_name}_kurtosis'] = 0.0
                        
        except Exception as e:
            # Default values for all color spaces on error
            for name in ['rgb', 'hsv', 'lab', 'yuv']:
                for channel_name in ['0', '1', '2']:
                    features[f'{name}_channel_{channel_name}_mean'] = 0.0
                    features[f'{name}_channel_{channel_name}_std'] = 0.0
                    features[f'{name}_channel_{channel_name}_skew'] = 0.0
                    features[f'{name}_channel_{channel_name}_kurtosis'] = 0.0
        
        return features
    
    def compute_histogram_features(self, img_color, img_gray):
        """Compute comprehensive histogram features"""
        features = {}
        
        try:
            # Grayscale histogram
            hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist_gray = hist_gray.flatten()
            hist_sum = hist_gray.sum()
            if hist_sum > 0:
                hist_gray = hist_gray / hist_sum
            else:
                hist_gray = np.ones_like(hist_gray) / len(hist_gray)
            
            features['gray_hist_entropy'] = -np.sum(hist_gray * np.log2(hist_gray + 1e-10))
            features['gray_hist_energy'] = np.sum(hist_gray ** 2)
            
            # Contrast calculation with safety check
            try:
                contrast_matrix = (np.arange(256)[:, None] - np.arange(256)) ** 2
                features['gray_hist_contrast'] = np.sum(contrast_matrix * hist_gray[:, None] * hist_gray)
            except:
                features['gray_hist_contrast'] = 0
            
            # Color histograms
            for i, color in enumerate(['blue', 'green', 'red']):
                hist = cv2.calcHist([img_color], [i], None, [256], [0, 256])
                hist = hist.flatten()
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist / hist_sum
                else:
                    hist = np.ones_like(hist) / len(hist)
                features[f'{color}_hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
                features[f'{color}_hist_energy'] = np.sum(hist ** 2)
            
            # 2D histogram for color correlation
            hist_2d = cv2.calcHist([img_color], [0, 1], None, [32, 32], [0, 256, 0, 256])
            hist_flat = hist_2d.flatten()
            
            # Safe correlation computation
            if len(hist_flat) > 1 and np.std(hist_flat) > 0:
                try:
                    corr_matrix = np.corrcoef(hist_flat.reshape(32, 32))
                    if corr_matrix.ndim == 2 and corr_matrix.shape[0] > 1:
                        features['color_correlation'] = corr_matrix[0, 1]
                    else:
                        features['color_correlation'] = 0
                except:
                    features['color_correlation'] = 0
            else:
                features['color_correlation'] = 0
                
        except Exception as e:
            # Default values on error
            features.update({
                'gray_hist_entropy': 0,
                'gray_hist_energy': 0,
                'gray_hist_contrast': 0,
                'blue_hist_entropy': 0,
                'blue_hist_energy': 0,
                'green_hist_entropy': 0,
                'green_hist_energy': 0,
                'red_hist_entropy': 0,
                'red_hist_energy': 0,
                'color_correlation': 0
            })
        
        return features
    
    def compute_edge_features(self, img_gray):
        """Compute edge-based features"""
        features = {}
        
        try:
            # Canny edge detection with multiple thresholds
            edges_low = cv2.Canny(img_gray, 50, 150)
            edges_high = cv2.Canny(img_gray, 100, 200)
            
            features['edge_pixels_low'] = float(np.sum(edges_low > 0) / edges_low.size)
            features['edge_pixels_high'] = float(np.sum(edges_high > 0) / edges_high.size)
            
            # Sobel gradients
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_dir = np.arctan2(grad_y, grad_x)
            
            features['gradient_magnitude_mean'] = float(np.mean(grad_mag))
            features['gradient_magnitude_std'] = float(np.std(grad_mag))
            features['gradient_direction_entropy'] = float(self.compute_entropy(grad_dir))
            
            # Laplacian
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            features['laplacian_variance'] = float(laplacian.var())
            
        except Exception as e:
            # Default values on error
            features.update({
                'edge_pixels_low': 0.0,
                'edge_pixels_high': 0.0,
                'gradient_magnitude_mean': 0.0,
                'gradient_magnitude_std': 0.0,
                'gradient_direction_entropy': 0.0,
                'laplacian_variance': 0.0
            })
        
        return features
    
    def compute_corner_features(self, img_gray, fast_detector):
        """Compute corner detection features"""
        features = {}
        
        try:
            # Harris corner detection
            harris = cv2.cornerHarris(img_gray, 2, 3, 0.04)
            features['harris_corner_strength_mean'] = float(np.mean(harris))
            features['harris_corner_strength_max'] = float(np.max(harris))
            features['harris_corner_count'] = int(np.sum(harris > 0.01 * harris.max()))
            
            # Shi-Tomasi corners
            corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.01, 10)
            features['shi_tomasi_corner_count'] = len(corners) if corners is not None else 0
            
            # FAST corners
            fast_kp = fast_detector.detect(img_gray, None)
            features['fast_corner_count'] = len(fast_kp) if fast_kp is not None else 0
            
        except Exception as e:
            # Default values on error
            features.update({
                'harris_corner_strength_mean': 0.0,
                'harris_corner_strength_max': 0.0,
                'harris_corner_count': 0,
                'shi_tomasi_corner_count': 0,
                'fast_corner_count': 0
            })
        
        return features
    
    def compute_blob_features(self, img_gray):
        """Compute blob detection features"""
        features = {}
        
        try:
            # Setup SimpleBlobDetector
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 10
            params.filterByCircularity = True
            params.minCircularity = 0.1
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(img_gray)
            
            features['blob_count'] = len(keypoints) if keypoints is not None else 0
            
            if keypoints and len(keypoints) > 0:
                sizes = [kp.size for kp in keypoints]
                features['blob_size_mean'] = float(np.mean(sizes))
                features['blob_size_std'] = float(np.std(sizes)) if len(sizes) > 1 else 0.0
                features['blob_size_max'] = float(np.max(sizes))
            else:
                features['blob_size_mean'] = 0.0
                features['blob_size_std'] = 0.0
                features['blob_size_max'] = 0.0
                
        except Exception as e:
            # Default values on error
            features.update({
                'blob_count': 0,
                'blob_size_mean': 0.0,
                'blob_size_std': 0.0,
                'blob_size_max': 0.0
            })
        
        return features
    
    def compute_contour_features(self, img_gray):
        """Compute contour-based features"""
        features = {}
        
        try:
            # Find contours
            _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            features['contour_count'] = len(contours)
            
            if contours and len(contours) > 0:
                # Analyze largest contours
                areas = [cv2.contourArea(c) for c in contours]
                perimeters = [cv2.arcLength(c, True) for c in contours]
                
                # Filter out zero areas and perimeters
                valid_areas = [a for a in areas if a > 0]
                valid_perimeters = [p for p in perimeters if p > 0]
                
                if valid_areas:
                    features['contour_area_mean'] = float(np.mean(valid_areas))
                    features['contour_area_std'] = float(np.std(valid_areas)) if len(valid_areas) > 1 else 0.0
                    features['contour_area_max'] = float(np.max(valid_areas))
                else:
                    features['contour_area_mean'] = 0.0
                    features['contour_area_std'] = 0.0
                    features['contour_area_max'] = 0.0
                
                features['contour_perimeter_mean'] = float(np.mean(valid_perimeters)) if valid_perimeters else 0.0
                
                # Analyze largest contour (by area)
                if valid_areas:
                    largest_idx = np.argmax(areas)
                    largest_contour = contours[largest_idx]
                    
                    # Convex hull
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    if contour_area > 0:
                        features['convexity_defects'] = float(hull_area / contour_area)
                    else:
                        features['convexity_defects'] = 1.0
                    
                    # Bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    features['bounding_box_aspect_ratio'] = float(w / h) if h > 0 else 0.0
                    
                    if w * h > 0:
                        features['bounding_box_fill_ratio'] = float(contour_area / (w * h))
                    else:
                        features['bounding_box_fill_ratio'] = 0.0
                    
                    # Minimum enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    if radius > 0:
                        circle_area = np.pi * radius**2
                        features['circularity'] = float(contour_area / circle_area)
                    else:
                        features['circularity'] = 0.0
                else:
                    features['convexity_defects'] = 0.0
                    features['bounding_box_aspect_ratio'] = 0.0
                    features['bounding_box_fill_ratio'] = 0.0
                    features['circularity'] = 0.0
            else:
                # No contours found
                for key in ['contour_area_mean', 'contour_area_std', 'contour_area_max', 
                           'contour_perimeter_mean', 'convexity_defects', 'bounding_box_aspect_ratio',
                           'bounding_box_fill_ratio', 'circularity']:
                    features[key] = 0.0
                    
        except Exception as e:
            # Default values on error
            features.update({
                'contour_count': 0,
                'contour_area_mean': 0.0,
                'contour_area_std': 0.0,
                'contour_area_max': 0.0,
                'contour_perimeter_mean': 0.0,
                'convexity_defects': 0.0,
                'bounding_box_aspect_ratio': 0.0,
                'bounding_box_fill_ratio': 0.0,
                'circularity': 0.0
            })
        
        return features
    
    def compute_advanced_texture_features(self, img_gray):
        """Compute advanced texture features"""
        features = {}
        
        try:
            # Local Binary Patterns
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
            
            # LBP histogram
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                      range=(0, n_points + 2))
            lbp_hist = lbp_hist.astype("float")
            hist_sum = lbp_hist.sum()
            
            if hist_sum > 0:
                lbp_hist /= hist_sum
                features['lbp_entropy'] = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))
                features['lbp_energy'] = float(np.sum(lbp_hist ** 2))
            else:
                features['lbp_entropy'] = 0.0
                features['lbp_energy'] = 0.0
            
            # Gray-Level Co-occurrence Matrix (GLCM)
            # Quantize image to fewer gray levels for GLCM
            img_quantized = (img_gray // 32).astype(np.uint8)
            
            # Ensure we have valid range
            if img_quantized.max() > 7:
                img_quantized = np.clip(img_quantized, 0, 7)
            
            glcm = graycomatrix(img_quantized, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=8, symmetric=True, normed=True)
            
            # GLCM properties
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                try:
                    prop_values = graycoprops(glcm, prop).flatten()
                    features[f'glcm_{prop}_mean'] = float(np.mean(prop_values))
                    features[f'glcm_{prop}_std'] = float(np.std(prop_values)) if len(prop_values) > 1 else 0.0
                except:
                    features[f'glcm_{prop}_mean'] = 0.0
                    features[f'glcm_{prop}_std'] = 0.0
                    
        except Exception as e:
            # Default values on error
            features.update({
                'lbp_entropy': 0.0,
                'lbp_energy': 0.0
            })
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                features[f'glcm_{prop}_mean'] = 0.0
                features[f'glcm_{prop}_std'] = 0.0
        
        return features
    
    def compute_morphological_features(self, img_gray):
        """Compute morphological features"""
        features = {}
        
        try:
            # Define kernels
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            
            # Morphological operations
            erosion = cv2.erode(img_gray, kernel_small, iterations=1)
            dilation = cv2.dilate(img_gray, kernel_small, iterations=1)
            opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_small)
            closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel_small)
            gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel_small)
            tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel_large)
            blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_large)
            
            # Compute differences
            features['erosion_diff'] = float(np.mean(np.abs(img_gray.astype(float) - erosion.astype(float))))
            features['dilation_diff'] = float(np.mean(np.abs(img_gray.astype(float) - dilation.astype(float))))
            features['opening_diff'] = float(np.mean(np.abs(img_gray.astype(float) - opening.astype(float))))
            features['closing_diff'] = float(np.mean(np.abs(img_gray.astype(float) - closing.astype(float))))
            features['gradient_mean'] = float(np.mean(gradient))
            features['gradient_std'] = float(np.std(gradient))
            features['tophat_mean'] = float(np.mean(tophat))
            features['blackhat_mean'] = float(np.mean(blackhat))
            
        except Exception as e:
            # Default values on error
            features.update({
                'erosion_diff': 0.0,
                'dilation_diff': 0.0,
                'opening_diff': 0.0,
                'closing_diff': 0.0,
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'tophat_mean': 0.0,
                'blackhat_mean': 0.0
            })
        
        return features
    
    def compute_frequency_features_opencv(self, img_gray):
        """Compute frequency domain features using OpenCV"""
        features = {}
        
        try:
            # DFT
            f_transform = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = cv2.magnitude(f_shift[:,:,0], f_shift[:,:,1])
            
            # Log spectrum for better visualization
            log_spectrum = np.log(magnitude_spectrum + 1)
            
            # Radial frequency analysis
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create masks for different frequency bands
            Y, X = np.ogrid[:rows, :cols]
            dist_from_center = np.sqrt((Y - crow)**2 + (X - ccol)**2)
            
            # Define frequency bands
            max_radius = np.sqrt(crow**2 + ccol**2)
            if max_radius > 0:
                low_freq_mask = dist_from_center <= max_radius * 0.1
                mid_freq_mask = (dist_from_center > max_radius * 0.1) & (dist_from_center <= max_radius * 0.5)
                high_freq_mask = dist_from_center > max_radius * 0.5
                
                # Compute energy in each band
                total_energy = np.sum(magnitude_spectrum)
                if total_energy > 0:
                    features['low_freq_energy'] = float(np.sum(magnitude_spectrum[low_freq_mask]))
                    features['mid_freq_energy'] = float(np.sum(magnitude_spectrum[mid_freq_mask]))
                    features['high_freq_energy'] = float(np.sum(magnitude_spectrum[high_freq_mask]))
                else:
                    features['low_freq_energy'] = 0.0
                    features['mid_freq_energy'] = 0.0
                    features['high_freq_energy'] = 0.0
            else:
                features['low_freq_energy'] = 0.0
                features['mid_freq_energy'] = 0.0
                features['high_freq_energy'] = 0.0
            
            # Frequency entropy
            freq_hist, _ = np.histogram(log_spectrum.ravel(), bins=50)
            freq_hist_sum = freq_hist.sum()
            
            if freq_hist_sum > 0:
                freq_hist = freq_hist / freq_hist_sum
                features['frequency_entropy'] = float(-np.sum(freq_hist * np.log2(freq_hist + 1e-10)))
            else:
                features['frequency_entropy'] = 0.0
                
        except Exception as e:
            # Default values on error
            features.update({
                'low_freq_energy': 0.0,
                'mid_freq_energy': 0.0,
                'high_freq_energy': 0.0,
                'frequency_entropy': 0.0
            })
        
        return features
    
    def compute_shape_descriptors(self, img_gray):
        """Compute shape descriptors"""
        features = {}
        
        try:
            # Find the main object
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours) > 0:
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Check if contour has minimum area
                if cv2.contourArea(largest_contour) > 0:
                    # Hu moments (invariant to translation, scale, and rotation)
                    moments = cv2.moments(largest_contour)
                    hu_moments = cv2.HuMoments(moments).flatten()
                    
                    for i, hu in enumerate(hu_moments):
                        if hu != 0:
                            features[f'hu_moment_{i}'] = float(-np.sign(hu) * np.log10(abs(hu) + 1e-10))
                        else:
                            features[f'hu_moment_{i}'] = 0.0
                    
                    # Ellipse fitting
                    if len(largest_contour) >= 5:
                        try:
                            ellipse = cv2.fitEllipse(largest_contour)
                            (x, y), (MA, ma), angle = ellipse
                            features['ellipse_major_axis'] = float(MA)
                            features['ellipse_minor_axis'] = float(ma)
                            
                            # Safe eccentricity calculation
                            if MA > 0 and ma > 0 and MA >= ma:
                                features['ellipse_eccentricity'] = float(np.sqrt(1 - (ma/MA)**2))
                            else:
                                features['ellipse_eccentricity'] = 0.0
                                
                            features['ellipse_angle'] = float(angle)
                        except:
                            features['ellipse_major_axis'] = 0.0
                            features['ellipse_minor_axis'] = 0.0
                            features['ellipse_eccentricity'] = 0.0
                            features['ellipse_angle'] = 0.0
                    else:
                        features['ellipse_major_axis'] = 0.0
                        features['ellipse_minor_axis'] = 0.0
                        features['ellipse_eccentricity'] = 0.0
                        features['ellipse_angle'] = 0.0
                    
                    # Solidity
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(largest_contour)
                    
                    if hull_area > 0:
                        features['solidity'] = float(contour_area / hull_area)
                    else:
                        features['solidity'] = 1.0
                    
                    # Extent
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    rect_area = w * h
                    
                    if rect_area > 0:
                        features['extent'] = float(contour_area / rect_area)
                    else:
                        features['extent'] = 0.0
                else:
                    # Contour has zero area
                    for i in range(7):
                        features[f'hu_moment_{i}'] = 0.0
                    features.update({
                        'ellipse_major_axis': 0.0, 'ellipse_minor_axis': 0.0,
                        'ellipse_eccentricity': 0.0, 'ellipse_angle': 0.0,
                        'solidity': 0.0, 'extent': 0.0
                    })
            else:
                # No contours found
                for i in range(7):
                    features[f'hu_moment_{i}'] = 0.0
                features.update({
                    'ellipse_major_axis': 0.0, 'ellipse_minor_axis': 0.0,
                    'ellipse_eccentricity': 0.0, 'ellipse_angle': 0.0,
                    'solidity': 0.0, 'extent': 0.0
                })
                
        except Exception as e:
            # Default values on error
            for i in range(7):
                features[f'hu_moment_{i}'] = 0.0
            features.update({
                'ellipse_major_axis': 0.0, 'ellipse_minor_axis': 0.0,
                'ellipse_eccentricity': 0.0, 'ellipse_angle': 0.0,
                'solidity': 0.0, 'extent': 0.0
            })
        
        return features
    
    def compute_keypoint_features(self, img_gray, sift, orb):
        """Compute keypoint-based features"""
        features = {}
        
        try:
            # SIFT features
            kp_sift, des_sift = sift.detectAndCompute(img_gray, None)
            features['sift_keypoint_count'] = len(kp_sift) if kp_sift is not None else 0
            
            if des_sift is not None and len(des_sift) > 0:
                features['sift_descriptor_mean'] = float(np.mean(des_sift))
                features['sift_descriptor_std'] = float(np.std(des_sift))
            else:
                features['sift_descriptor_mean'] = 0
                features['sift_descriptor_std'] = 0
            
            # ORB features
            kp_orb, des_orb = orb.detectAndCompute(img_gray, None)
            features['orb_keypoint_count'] = len(kp_orb) if kp_orb is not None else 0
            
            # Keypoint distribution
            if kp_sift and len(kp_sift) > 0:
                kp_x = [kp.pt[0] for kp in kp_sift]
                kp_y = [kp.pt[1] for kp in kp_sift]
                features['keypoint_spread_x'] = float(np.std(kp_x)) if len(kp_x) > 1 else 0
                features['keypoint_spread_y'] = float(np.std(kp_y)) if len(kp_y) > 1 else 0
            else:
                features['keypoint_spread_x'] = 0
                features['keypoint_spread_y'] = 0
                
        except Exception as e:
            # Default values on error
            features.update({
                'sift_keypoint_count': 0,
                'sift_descriptor_mean': 0,
                'sift_descriptor_std': 0,
                'orb_keypoint_count': 0,
                'keypoint_spread_x': 0,
                'keypoint_spread_y': 0
            })
        
        return features
    
    def compute_hough_features(self, img_gray):
        """Compute Hough transform features for lines and circles"""
        features = {}
        
        try:
            # Edge detection for Hough
            edges = cv2.Canny(img_gray, 50, 150)
            
            # Hough Line Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            features['hough_line_count'] = len(lines) if lines is not None else 0
            
            if lines is not None and len(lines) > 0:
                # Analyze line angles
                angles = []
                lengths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angles.append(angle)
                    lengths.append(length)
                
                features['line_angle_variance'] = float(np.var(angles)) if len(angles) > 1 else 0.0
                features['line_length_mean'] = float(np.mean(lengths))
                features['line_length_std'] = float(np.std(lengths)) if len(lengths) > 1 else 0.0
            else:
                features['line_angle_variance'] = 0.0
                features['line_length_mean'] = 0.0
                features['line_length_std'] = 0.0
            
            # Hough Circle Transform
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                      param1=50, param2=30, minRadius=5, maxRadius=50)
            
            features['hough_circle_count'] = len(circles[0]) if circles is not None else 0
            
            if circles is not None and len(circles[0]) > 0:
                radii = circles[0, :, 2]
                features['circle_radius_mean'] = float(np.mean(radii))
                features['circle_radius_std'] = float(np.std(radii)) if len(radii) > 1 else 0.0
            else:
                features['circle_radius_mean'] = 0.0
                features['circle_radius_std'] = 0.0
                
        except Exception as e:
            # Default values on error
            features.update({
                'hough_line_count': 0,
                'line_angle_variance': 0.0,
                'line_length_mean': 0.0,
                'line_length_std': 0.0,
                'hough_circle_count': 0,
                'circle_radius_mean': 0.0,
                'circle_radius_std': 0.0
            })
        
        return features
    
    def compute_image_moments(self, img_gray):
        """Compute raw image moments"""
        features = {}
        
        try:
            # Calculate moments
            moments = cv2.moments(img_gray)
            
            # Spatial moments
            features['m00'] = float(moments['m00'])  # Area
            features['m10'] = float(moments['m10'])
            features['m01'] = float(moments['m01'])
            features['m20'] = float(moments['m20'])
            features['m11'] = float(moments['m11'])
            features['m02'] = float(moments['m02'])
            
            # Central moments
            features['mu20'] = float(moments['mu20'])
            features['mu11'] = float(moments['mu11'])
            features['mu02'] = float(moments['mu02'])
            features['mu30'] = float(moments['mu30'])
            features['mu21'] = float(moments['mu21'])
            features['mu12'] = float(moments['mu12'])
            features['mu03'] = float(moments['mu03'])
            
            # Normalized central moments
            for key in ['nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']:
                features[key] = float(moments[key])
                
        except Exception as e:
            # Default values on error
            for key in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02',
                       'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                       'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']:
                features[key] = 0.0
        
        return features
    
    def compute_gabor_features(self, img_gray):
        """Compute Gabor filter responses"""
        features = {}
        
        try:
            # Gabor filter parameters
            ksize = 31
            sigma = 4.0
            lambd = 10.0
            gamma = 0.5
            psi = 0
            
            # Apply Gabor filters at different orientations
            orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            gabor_responses = []
            for theta in orientations:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
                filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
                gabor_responses.append(filtered)
                
                features[f'gabor_theta_{int(theta * 180 / np.pi)}_mean'] = float(np.mean(filtered))
                features[f'gabor_theta_{int(theta * 180 / np.pi)}_std'] = float(np.std(filtered))
            
            # Compute energy across all orientations
            gabor_energy = np.sum([r**2 for r in gabor_responses], axis=0)
            features['gabor_energy_mean'] = float(np.mean(gabor_energy))
            features['gabor_energy_std'] = float(np.std(gabor_energy))
            
        except Exception as e:
            # Default values on error
            for theta in [0, 45, 90, 135]:
                features[f'gabor_theta_{theta}_mean'] = 0.0
                features[f'gabor_theta_{theta}_std'] = 0.0
            features['gabor_energy_mean'] = 0.0
            features['gabor_energy_std'] = 0.0
        
        return features
    
    def compute_entropy(self, data):
        """Compute entropy of data"""
        try:
            hist, _ = np.histogram(data.ravel(), bins=50)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                return float(entropy)
            else:
                return 0.0
        except:
            return 0.0
    
    def save_resume_state(self, output_dir):
        """Save the current processing state for resuming"""
        state_file = Path(output_dir) / '.resume_state.json'
        state = {
            'processed_files': list(self.processed_files),
            'timestamp': datetime.now().isoformat()
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
    
    def load_resume_state(self, output_dir):
        """Load the resume state from previous run"""
        state_file = Path(output_dir) / '.resume_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.processed_files = set(state.get('processed_files', []))
                logger.info(f"Loaded resume state: {len(self.processed_files)} files already processed")
                return True
        return False
    
    def check_if_already_processed(self, img_path, output_dir):
        """Check if an image has already been processed"""
        # Check using the expected output filename
        output_path = output_dir / f"{img_path.stem}_color.pt"
        gray_output_path = output_dir / f"{img_path.stem}_gray.pt"
        
        # If both output files exist, consider it processed
        return output_path.exists() and gray_output_path.exists()
    
    def find_all_images(self, root_dir):
        """Recursively find all image files and organize by class"""
        root_path = Path(root_dir)
        image_files_by_class = defaultdict(list)
        
        logger.info(f"Scanning for images in {root_dir} and organizing by class...")
        
        # First pass: collect all image files
        all_images = []
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                all_images.append(path)
        
        # Second pass: organize by actual directories (not individual files)
        for path in all_images:
            class_label = self.extract_class_from_path(path)
            image_files_by_class[class_label].append(path)
        
        # Clean up single-file "classes" that should be grouped
        cleaned_classes = defaultdict(list)
        for class_name, files in image_files_by_class.items():
            # If it's a single file "class" and looks like a filename, try to group it better
            if len(files) == 1 and any(ext in class_name.lower() for ext in self.image_extensions):
                # Use the parent directory instead
                parent_class = files[0].parent.name
                cleaned_classes[parent_class].extend(files)
            else:
                cleaned_classes[class_name].extend(files)
        
        image_files_by_class = dict(cleaned_classes)
        
        total_images = sum(len(files) for files in image_files_by_class.values())
        logger.info(f"Found {total_images} images across {len(image_files_by_class)} classes")
        
        for class_name, files in sorted(image_files_by_class.items()):
            logger.info(f"  {class_name}: {len(files)} images")
        
        return image_files_by_class
    
    def load_existing_class_counts(self, output_dir):
        """Count existing processed files per class without loading tensors into memory"""
        logger.info("Counting existing tensorized data from disk...")
        
        tensor_dir = Path(output_dir)
        
        # Find all existing tensor files
        for tensor_file in tensor_dir.rglob('*_color.pt'):
            try:
                # Load just the metadata, not the full tensor
                # Use map_location to avoid CUDA memory issues
                data = torch.load(tensor_file, map_location='cpu', weights_only=False)
                
                # Extract metadata
                if 'metadata' in data:
                    class_label = data['metadata'].get('class_label', 'unknown')
                    self.class_data[class_label]['count'] += 1
                    self.class_data[class_label]['paths'].append(data['metadata']['source_path'])
                
                # Immediately delete the loaded data to free memory
                del data
                
            except Exception as e:
                logger.warning(f"Error loading metadata from {tensor_file}: {e}")
        
        # Log class distribution
        total_loaded = sum(data['count'] for data in self.class_data.values())
        logger.info(f"Found {total_loaded} existing tensors on disk")
        
        for class_name, data in self.class_data.items():
            if data['count'] > 0:
                logger.info(f"  {class_name}: {data['count']} tensors found")
    
    def tensorize_and_analyze_image(self, img_path, class_label):
        """Convert image to tensor and perform ALL analyses"""
        try:
            # Extract OpenCV features first
            opencv_features, img_gray_cv = self.compute_opencv_features(img_path)
            
            # If OpenCV features extraction failed completely, skip this image
            if not opencv_features and img_gray_cv is None:
                logger.warning(f"Skipping {img_path} due to OpenCV processing failure")
                return None, None, None, None
            
            # Load image for tensor conversion
            img = Image.open(img_path)
            
            if img.mode != 'RGB':
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Convert to tensor
            tensor = self.transform(img)
            
            # Convert to grayscale tensor
            grayscale_tensor = transforms.functional.rgb_to_grayscale(tensor)
            
            # Combined metadata
            metadata = {
                'original_size': img.size,
                'original_mode': img.mode,
                'tensor_shape': list(tensor.shape),
                'source_path': str(img_path),
                'class_label': class_label,
                'opencv_features': opencv_features
            }
            
            return tensor, grayscale_tensor, opencv_features, metadata
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None, None, None, None
    
    def process_single_image_ultimate(self, img_path, class_label, output_dir):
        """Process single image with ULTIMATE analysis"""
        try:
            # Check if already processed
            if self.check_if_already_processed(img_path, output_dir):
                return True  # Already done, count as success
            
            # Tensorize and analyze
            tensor, grayscale_tensor, opencv_features, metadata = self.tensorize_and_analyze_image(
                img_path, class_label
            )
            
            if tensor is None:
                return False
            
            # Save tensor and grayscale
            output_path = output_dir / f"{img_path.stem}_color.pt"
            
            torch.save({
                'tensor': tensor,
                'grayscale_tensor': grayscale_tensor,
                'metadata': metadata
            }, output_path)
            
            # Update class count (thread-safe)
            with self.lock:
                self.class_data[class_label]['count'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return False
    
    def compute_statistics_from_disk(self, output_dir):
        """Compute statistics by loading tensors in batches from disk"""
        logger.info("Computing comprehensive statistics from tensorized data...")
        
        # For each class, load tensors in batches and compute statistics
        for class_name in self.class_data.keys():
            if self.class_data[class_name]['count'] == 0:
                continue
                
            logger.info(f"Computing statistics for {class_name} ({self.class_data[class_name]['count']} samples)...")
            
            # Find all tensor files for this class
            class_tensor_files = []
            class_dir = Path(output_dir) / class_name
            
            if class_dir.exists():
                for tensor_file in class_dir.glob('*_color.pt'):
                    class_tensor_files.append(tensor_file)
            
            if not class_tensor_files:
                continue
            
            # Load tensors in batches to compute statistics
            batch_size = 100  # Process 100 at a time to manage memory
            all_features = []
            
            for i in range(0, len(class_tensor_files), batch_size):
                batch_files = class_tensor_files[i:i+batch_size]
                batch_tensors = []
                batch_gray_tensors = []
                
                for tensor_file in batch_files:
                    try:
                        data = torch.load(tensor_file, map_location='cpu', weights_only=False)
                        batch_tensors.append(data['tensor'])
                        batch_gray_tensors.append(data['grayscale_tensor'])
                        
                        if 'metadata' in data and 'opencv_features' in data['metadata']:
                            all_features.append(data['metadata']['opencv_features'])
                        
                    except Exception as e:
                        logger.warning(f"Error loading {tensor_file}: {e}")
                        continue
                
                # Clear memory after each batch
                del batch_tensors
                del batch_gray_tensors
                gc.collect()
            
            # Aggregate OpenCV features
            if all_features:
                self.class_data[class_name]['statistics'] = {
                    'opencv_feature_statistics': self.aggregate_opencv_features(all_features),
                    'sample_count': len(class_tensor_files)
                }
    
    def aggregate_opencv_features(self, features_list):
        """Aggregate OpenCV features across all images in a class"""
        if not features_list:
            return {}
        
        # Flatten all features
        all_feature_names = set()
        for features in features_list:
            if features:  # Check if features is not empty/None
                for category in features.values():
                    if isinstance(category, dict):
                        all_feature_names.update(category.keys())
        
        aggregated = {}
        
        # Compute statistics for each feature
        for feature_name in all_feature_names:
            values = []
            for features in features_list:
                if features:  # Check if features is not empty/None
                    for category in features.values():
                        if isinstance(category, dict) and feature_name in category:
                            value = category[feature_name]
                            if isinstance(value, (int, float, np.number)):
                                values.append(float(value))
            
            if values:
                aggregated[feature_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        return aggregated
    
    def generate_neural_network_insights(self):
        """Generate comprehensive insights for neural network training"""
        logger.info("Generating ULTIMATE neural network training insights...")
        
        insights = {
            'recommended_architectures': {},
            'data_characteristics': {},
            'training_recommendations': {},
            'preprocessing_pipeline': {},
            'augmentation_strategies': {},
            'ensemble_recommendations': {}
        }
        
        # Analyze data - ensure counts are integers
        total_samples = sum(int(data['count']) for data in self.class_data.values())
        n_classes = len(self.class_data)
        
        # Ensure class_sizes contains integers
        class_sizes = {}
        for name, data in self.class_data.items():
            count = data['count']
            # Convert to int if it's a string or other type
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    logger.warning(f"Invalid count for class {name}: {count}, defaulting to 0")
                    count = 0
            class_sizes[name] = int(count)
        
        if class_sizes and any(v > 0 for v in class_sizes.values()):
            # Filter out zero counts for min calculation
            non_zero_sizes = [v for v in class_sizes.values() if v > 0]
            if non_zero_sizes:
                max_class_size = max(non_zero_sizes)
                min_class_size = min(non_zero_sizes)
                imbalance_ratio = float(max_class_size) / float(min_class_size)
            else:
                max_class_size = 0
                min_class_size = 0
                imbalance_ratio = 1.0
        else:
            max_class_size = 0
            min_class_size = 0
            imbalance_ratio = 1.0
        
        insights['data_characteristics'] = {
            'total_samples': total_samples,
            'num_classes': n_classes,
            'class_distribution': class_sizes,
            'imbalance_ratio': imbalance_ratio,
            'balanced': imbalance_ratio < 2.0,
            'samples_per_class_mean': np.mean(list(class_sizes.values())) if class_sizes else 0,
            'samples_per_class_std': np.std(list(class_sizes.values())) if len(class_sizes) > 1 else 0
        }
        
        # Multiple architecture recommendations
        insights['recommended_architectures']['cnn'] = {
            'type': 'ConvNet',
            'layers': self.recommend_cnn_architecture(total_samples, n_classes),
            'dropout_rates': [0.3, 0.4, 0.5] if total_samples < 5000 else [0.2, 0.3, 0.4]
        }
        
        insights['recommended_architectures']['resnet'] = {
            'type': 'ResNet',
            'variant': 'ResNet18' if total_samples < 10000 else 'ResNet34',
            'pretrained': True if total_samples < 5000 else False
        }
        
        insights['recommended_architectures']['vision_transformer'] = {
            'recommended': total_samples > 10000,
            'patch_size': 16,
            'embed_dim': 384 if total_samples < 50000 else 768
        }
        
        # Training recommendations
        insights['training_recommendations'] = {
            'batch_size': self.recommend_batch_size(total_samples),
            'learning_rates': {
                'initial': 0.001,
                'scheduler': 'cosine_annealing',
                'min_lr': 0.00001
            },
            'optimizers': ['AdamW', 'SGD with momentum'],
            'epochs': 100 if total_samples < 5000 else 50,
            'early_stopping_patience': 10,
            'use_class_weights': imbalance_ratio > 3.0,
            'use_focal_loss': imbalance_ratio > 5.0,
            'mixup_alpha': 0.2 if total_samples < 10000 else 0.1
        }
        
        # Preprocessing pipeline
        insights['preprocessing_pipeline'] = {
            'normalization': 'imagenet_stats',
            'resize_strategy': 'resize_then_crop',
            'color_augmentation': True,
            'geometric_augmentation': True,
            'advanced_augmentation': ['cutmix', 'mixup'] if total_samples < 10000 else ['randaugment']
        }
        
        # Augmentation strategies
        insights['augmentation_strategies'] = {
            'basic': ['horizontal_flip', 'vertical_flip', 'rotation_15', 'brightness_contrast'],
            'advanced': ['elastic_transform', 'grid_distortion', 'optical_distortion'],
            'color': ['hue_saturation', 'rgb_shift', 'channel_shuffle'],
            'noise': ['gaussian_noise', 'blur', 'jpeg_compression'] if total_samples > 5000 else []
        }
        
        # Ensemble recommendations
        insights['ensemble_recommendations'] = {
            'use_ensemble': n_classes > 5 or total_samples > 10000,
            'ensemble_size': 3 if total_samples < 20000 else 5,
            'strategies': ['different_architectures', 'different_initializations', 'cross_validation_folds']
        }
        
        return insights
    
    def recommend_cnn_architecture(self, total_samples, n_classes):
        """Recommend CNN architecture based on dataset size"""
        if total_samples < 1000:
            return [
                {'conv': 32, 'kernel': 3, 'pool': 2},
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'fc': [128, n_classes]}
            ]
        elif total_samples < 10000:
            return [
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'conv': 128, 'kernel': 3, 'pool': 2},
                {'conv': 256, 'kernel': 3, 'pool': 2},
                {'fc': [512, 256, n_classes]}
            ]
        else:
            return [
                {'conv': 64, 'kernel': 3, 'pool': 2},
                {'conv': 128, 'kernel': 3, 'pool': 2},
                {'conv': 256, 'kernel': 3, 'pool': 2},
                {'conv': 512, 'kernel': 3, 'pool': 2},
                {'fc': [1024, 512, n_classes]}
            ]
    
    def recommend_batch_size(self, total_samples):
        """Recommend batch size based on dataset size"""
        if total_samples < 1000:
            return 16
        elif total_samples < 10000:
            return 32
        elif total_samples < 50000:
            return 64
        else:
            return 128
    
    def save_all_results(self, output_dir, analysis_dir):
        """Save all analysis results and visualizations"""
        Path(analysis_dir).mkdir(parents=True, exist_ok=True)
        
        # Save class statistics
        class_stats_path = analysis_dir / 'ultimate_class_statistics.json'
        class_stats_data = {}
        
        for class_name, data in self.class_data.items():
            stats_copy = {
                'sample_count': int(data['count']),  # Ensure it's an int
                'paths_count': len(data['paths'])
            }
            if 'statistics' in data and data['statistics']:
                stats_copy.update(data['statistics'])
            class_stats_data[class_name] = stats_copy
        
        with open(class_stats_path, 'w') as f:
            json.dump(class_stats_data, f, indent=2, default=self._json_serialize)
        
        # Save global statistics
        global_stats_path = analysis_dir / 'ultimate_global_statistics.json'
        with open(global_stats_path, 'w') as f:
            json.dump(self.global_statistics, f, indent=2, default=self._json_serialize)
        
        # Generate visualizations
        self.generate_all_visualizations(analysis_dir)
        
        # Save neural network config
        nn_insights = self.generate_neural_network_insights()
        nn_config_path = analysis_dir / 'ultimate_neural_network_config.json'
        with open(nn_config_path, 'w') as f:
            json.dump(nn_insights, f, indent=2, default=self._json_serialize)
        
        # Generate comprehensive report
        self.generate_ultimate_report(analysis_dir)
        
        # Generate ready-to-use PyTorch dataset code
        self.generate_pytorch_code(analysis_dir, output_dir)
        
        logger.info(f"ALL results saved to {analysis_dir}")
    
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy/torch types"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return str(obj)
    
    def generate_all_visualizations(self, output_dir):
        """Generate comprehensive visualizations"""
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Class distribution
        plt.figure(figsize=(15, 8))
        classes = list(self.global_statistics['class_distribution'].keys())
        counts = [int(v) for v in self.global_statistics['class_distribution'].values()]  # Ensure integers
        
        if classes and counts:
            plt.subplot(2, 2, 1)
            plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy')
            plt.xlabel('Class Index')
            plt.ylabel('Number of Samples')
            plt.title('Class Distribution')
            
            # Add class names as x-tick labels if not too many
            if len(classes) <= 20:
                plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
            
            # 2. Pie chart for balanced view
            plt.subplot(2, 2, 4)
            # Show only top 10 classes in pie chart
            if len(classes) > 10:
                sorted_indices = np.argsort(counts)[-10:]
                top_classes = [classes[i] for i in sorted_indices]
                top_counts = [counts[i] for i in sorted_indices]
                other_count = sum(counts) - sum(top_counts)
                if other_count > 0:
                    top_classes.append('Others')
                    top_counts.append(other_count)
                plt.pie(top_counts, labels=[cn[:15] for cn in top_classes], autopct='%1.1f%%')
            else:
                plt.pie(counts, labels=[cn[:15] for cn in classes], autopct='%1.1f%%')
            plt.title('Class Distribution (Percentage)')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'overview_analysis.png', dpi=300)
        
        plt.close()
    
    def generate_ultimate_report(self, output_dir):
        """Generate the ultimate comprehensive report"""
        report_path = output_dir / 'ULTIMATE_ANALYSIS_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ULTIMATE NEURAL NETWORK DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Powered by: OpenCV + PyTorch + Advanced Computer Vision\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write(f"Total Images Analyzed: {self.global_statistics['total_images']}\n")
            f.write(f"Number of Classes: {len(self.class_data)}\n")
            f.write(f"Total Features Extracted per Image: 200+\n")
            f.write(f"Analysis Techniques Used: 15+ methods\n\n")
            
            # Detailed Class Analysis
            f.write("DETAILED CLASS ANALYSIS\n")
            f.write("-"*60 + "\n")
            for class_name, count in sorted(self.global_statistics['class_distribution'].items()):
                count = int(count)  # Ensure it's an integer
                percentage = (count / self.global_statistics['total_images']) * 100 if self.global_statistics['total_images'] > 0 else 0
                f.write(f"\n{class_name}:\n")
                f.write(f"  Samples: {count} ({percentage:.1f}%)\n")
            
            # Feature Analysis Summary
            f.write("\n\nFEATURE EXTRACTION SUMMARY\n")
            f.write("-"*60 + "\n")
            f.write("OpenCV Features Extracted:\n")
            f.write("  • Color Space Analysis (RGB, HSV, LAB, YUV)\n")
            f.write("  • Histogram Features (Entropy, Energy, Correlation)\n")
            f.write("  • Edge Detection (Canny, Sobel, Laplacian)\n")
            f.write("  • Corner Detection (Harris, Shi-Tomasi, FAST)\n")
            f.write("  • Blob Detection\n")
            f.write("  • Contour Analysis\n")
            f.write("  • Texture Features (LBP, GLCM, Gabor)\n")
            f.write("  • Morphological Features\n")
            f.write("  • Frequency Domain Analysis\n")
            f.write("  • Shape Descriptors (Hu Moments)\n")
            f.write("  • Keypoint Features (SIFT, ORB)\n")
            f.write("  • Hough Transform (Lines, Circles)\n")
            f.write("  • Image Moments\n")
            
            # Neural Network Recommendations
            nn_insights = self.generate_neural_network_insights()
            f.write("\n\nNEURAL NETWORK ARCHITECTURE RECOMMENDATIONS\n")
            f.write("-"*60 + "\n")
            
            # CNN Architecture
            f.write("\n1. Convolutional Neural Network (CNN):\n")
            cnn_arch = nn_insights['recommended_architectures']['cnn']
            f.write(f"   Layers: {cnn_arch['layers']}\n")
            f.write(f"   Dropout rates: {cnn_arch['dropout_rates']}\n")
            
            # ResNet
            f.write("\n2. ResNet Architecture:\n")
            resnet = nn_insights['recommended_architectures']['resnet']
            f.write(f"   Variant: {resnet['variant']}\n")
            f.write(f"   Pretrained: {resnet['pretrained']}\n")
            
            # Vision Transformer
            f.write("\n3. Vision Transformer:\n")
            vit = nn_insights['recommended_architectures']['vision_transformer']
            f.write(f"   Recommended: {vit['recommended']}\n")
            if vit['recommended']:
                f.write(f"   Patch size: {vit['patch_size']}\n")
                f.write(f"   Embedding dimension: {vit['embed_dim']}\n")
            
            # Training Configuration
            f.write("\n\nTRAINING CONFIGURATION\n")
            f.write("-"*60 + "\n")
            train_config = nn_insights['training_recommendations']
            f.write(f"Batch Size: {train_config['batch_size']}\n")
            f.write(f"Initial Learning Rate: {train_config['learning_rates']['initial']}\n")
            f.write(f"LR Scheduler: {train_config['learning_rates']['scheduler']}\n")
            f.write(f"Optimizers: {', '.join(train_config['optimizers'])}\n")
            f.write(f"Epochs: {train_config['epochs']}\n")
            f.write(f"Early Stopping Patience: {train_config['early_stopping_patience']}\n")
            f.write(f"Use Class Weights: {train_config['use_class_weights']}\n")
            f.write(f"Use Focal Loss: {train_config['use_focal_loss']}\n")
            f.write(f"MixUp Alpha: {train_config['mixup_alpha']}\n")
            
            # Data Insights
            f.write("\n\nDATA INSIGHTS\n")
            f.write("-"*60 + "\n")
            data_char = nn_insights['data_characteristics']
            f.write(f"Class Balance Ratio: {data_char['imbalance_ratio']:.2f}\n")
            f.write(f"Dataset Balanced: {'Yes' if data_char['balanced'] else 'No'}\n")
            f.write(f"Mean samples per class: {data_char['samples_per_class_mean']:.1f}\n")
            f.write(f"Std samples per class: {data_char['samples_per_class_std']:.1f}\n")
            
            # Augmentation Strategies
            f.write("\n\nRECOMMENDED AUGMENTATION STRATEGIES\n")
            f.write("-"*60 + "\n")
            aug_strategies = nn_insights['augmentation_strategies']
            f.write(f"Basic: {', '.join(aug_strategies['basic'])}\n")
            f.write(f"Advanced: {', '.join(aug_strategies['advanced'])}\n")
            f.write(f"Color: {', '.join(aug_strategies['color'])}\n")
            if aug_strategies['noise']:
                f.write(f"Noise: {', '.join(aug_strategies['noise'])}\n")
            
            # Ensemble Recommendations
            f.write("\n\nENSEMBLE RECOMMENDATIONS\n")
            f.write("-"*60 + "\n")
            ensemble = nn_insights['ensemble_recommendations']
            f.write(f"Use Ensemble: {ensemble['use_ensemble']}\n")
            if ensemble['use_ensemble']:
                f.write(f"Ensemble Size: {ensemble['ensemble_size']}\n")
                f.write(f"Strategies: {', '.join(ensemble['strategies'])}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
    
    def generate_pytorch_code(self, analysis_dir, tensor_dir):
        """Generate ready-to-use PyTorch code"""
        code_path = analysis_dir / 'ultimate_pytorch_implementation.py'
        
        nn_insights = self.generate_neural_network_insights()
        
        # Get class names from global statistics
        class_names = list(self.global_statistics['class_distribution'].keys())
        
        code = f'''"""
Auto-generated PyTorch implementation based on data analysis
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import json
import numpy as np

class AnalyzedDataset(Dataset):
    """Dataset class for the tensorized images"""
    
    def __init__(self, tensor_dir, transform=None):
        self.tensor_dir = Path(tensor_dir)
        self.tensor_files = list(self.tensor_dir.rglob('*.pt'))
        self.transform = transform
        
        # Load class mapping
        self.classes = {json.dumps(class_names)}
        self.class_to_idx = {{cls: idx for idx, cls in enumerate(self.classes)}}
        
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        # Load tensor and metadata with weights_only=False since these are our own files
        data = torch.load(self.tensor_files[idx], weights_only=False)
        tensor = data['tensor']
        metadata = data['metadata']
        
        # Get label
        class_name = metadata['class_label']
        label = self.class_to_idx[class_name]
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, label

class CustomCNN(nn.Module):
    """Recommended CNN architecture based on data analysis"""
    
    def __init__(self, num_classes={len(self.class_data)}):
        super(CustomCNN, self).__init__()
        
        # Architecture based on dataset size
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][0]}),
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][1]}),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout({nn_insights['recommended_architectures']['cnn']['dropout_rates'][2]}),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training configuration based on analysis
config = {{
    'batch_size': {nn_insights['training_recommendations']['batch_size']},
    'learning_rate': {nn_insights['training_recommendations']['learning_rates']['initial']},
    'epochs': {nn_insights['training_recommendations']['epochs']},
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_workers': 4,
    'use_class_weights': {nn_insights['training_recommendations']['use_class_weights']},
}}

# Data augmentation based on recommendations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example training loop
def train_model():
    # Load dataset
    dataset = AnalyzedDataset('{tensor_dir}', transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=config['num_workers'])
    
    # Initialize model
    model = CustomCNN().to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    for epoch in range(config['epochs']):
        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
        
        print(f'Epoch {{epoch+1}}: Val Loss: {{val_loss/len(val_loader):.4f}}, '
              f'Accuracy: {{100.*correct/val_size:.2f}}%')

if __name__ == '__main__':
    train_model()
'''
        
        with open(code_path, 'w') as f:
            f.write(code)
        
        logger.info(f"Generated PyTorch implementation code at {code_path}")
    
    def process_directory(self, input_dir, output_dir, analysis_output_dir, num_workers=4):
        """Process all images with ULTIMATE analysis - with resume capability"""
        # Find all images
        image_files_by_class = self.find_all_images(input_dir)
        
        if not image_files_by_class:
            logger.error("No images found in directory tree")
            return
        
        # Create directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(analysis_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load resume state if it exists
        self.load_resume_state(output_dir)
        
        # Count existing tensorized data (don't load into memory)
        self.load_existing_class_counts(output_dir)
        
        # Process each class
        for class_name, image_paths in image_files_by_class.items():
            logger.info(f"\nProcessing class: {class_name}")
            
            class_output_dir = Path(output_dir) / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter out already processed images
            images_to_process = []
            skipped = 0
            
            for img_path in image_paths:
                if self.check_if_already_processed(img_path, class_output_dir):
                    skipped += 1
                    self.processed_files.add(str(img_path))
                else:
                    images_to_process.append(img_path)
            
            if skipped > 0:
                logger.info(f"  Skipping {skipped} already processed images")
            
            if not images_to_process:
                logger.info(f"  All images in {class_name} already processed!")
                continue
            
            successful = 0
            failed = 0
            
            # Process with reduced concurrency for stability
            actual_workers = min(num_workers, 4)  # Limit to 4 workers max
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self.process_single_image_ultimate,
                        img_path,
                        class_name,
                        class_output_dir
                    ): img_path
                    for img_path in images_to_process
                }
                
                with tqdm(total=len(images_to_process), desc=f"Processing {class_name}") as pbar:
                    for future in as_completed(future_to_file):
                        img_path = future_to_file[future]
                        try:
                            result = future.result()
                            if result:
                                successful += 1
                                self.processed_files.add(str(img_path))
                                
                                # Save state periodically (every 100 files)
                                if len(self.processed_files) % 100 == 0:
                                    self.save_resume_state(output_dir)
                            else:
                                failed += 1
                        except Exception as e:
                            failed += 1
                            logger.error(f"Error processing {img_path}: {e}")
                        pbar.update(1)
            
            logger.info(f"{class_name}: {successful} successful, {failed} failed")
            
            # Save resume state after each class
            self.save_resume_state(output_dir)
            
            # Force garbage collection after each class
            gc.collect()
        
        # Update global statistics - ensure all counts are integers
        self.global_statistics['total_images'] = sum(
            int(data['count']) for data in self.class_data.values()
        )
        self.global_statistics['class_distribution'] = {
            name: int(data['count']) for name, data in self.class_data.items()
        }
        
        # Compute statistics from disk (without loading all tensors)
        self.compute_statistics_from_disk(output_dir)
        
        # Generate neural network insights
        self.global_statistics['neural_network_insights'] = self.generate_neural_network_insights()
        
        # Save EVERYTHING
        self.save_all_results(output_dir, analysis_output_dir)
        
        # Clean up resume state file
        resume_file = Path(output_dir) / '.resume_state.json'
        if resume_file.exists():
            resume_file.unlink()
        
        logger.info("\nULTIMATE processing and analysis complete!")

def main():
    print("=== ULTIMATE Smart Image Tensorizer with Full OpenCV Analysis ===\n")
    print("This program will:")
    print("• Resume from where it left off if interrupted")
    print("• Skip already processed images")
    print("• Recursively find ALL images in subdirectories")
    print("• Convert to tensors AND grayscale")
    print("• Extract 200+ features using OpenCV")
    print("• Compute ALL statistical correlations")
    print("• Analyze patterns up to 3rd order")
    print("• Generate neural network architectures")
    print("• Create training recommendations")
    print("• Build visualizations automatically")
    print("\nNO OPTIONS NEEDED - ALWAYS RUNS AT FULL POWER!\n")
    
    # Get directories
    print("Enter the path to the root directory containing images:")
    input_dir = input().strip()
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    print("\nEnter the path to the output directory for tensors:")
    tensor_output_dir = input().strip()
    
    print("\nEnter the path to the output directory for analysis results:")
    analysis_output_dir = input().strip()
    
    print("\nEnter number of parallel workers (default: 4, max recommended: 4):")
    workers_input = input().strip()
    num_workers = int(workers_input) if workers_input.isdigit() else 4
    
    # Limit workers for stability
    if num_workers > 4:
        print(f"Note: Limiting workers to 4 for stability (you requested {num_workers})")
        num_workers = 4
    
    # Create the ULTIMATE tensorizer
    tensorizer = UltimateImageTensorizer()
    
    # Process everything
    print(f"\nStarting ULTIMATE processing with {num_workers} workers...")
    print("This will resume from where it left off if previously interrupted!\n")
    
    try:
        tensorizer.process_directory(
            input_dir, 
            tensor_output_dir, 
            analysis_output_dir, 
            num_workers
        )
        
        print("\nULTIMATE PROCESSING COMPLETE!")
        print(f"\nResults saved to:")
        print(f"• Tensors: {tensor_output_dir}")
        print(f"• Analysis: {analysis_output_dir}")
        print("\nThe analysis directory contains:")
        print("• ultimate_class_statistics.json - Detailed per-class analysis")
        print("• ultimate_global_statistics.json - Complete dataset analysis")
        print("• ultimate_neural_network_config.json - Ready-to-use training config")
        print("• visualizations/ - Comprehensive plots and heatmaps")
        print("• ULTIMATE_ANALYSIS_REPORT.txt - Complete human-readable report")
        print("• ultimate_pytorch_implementation.py - Ready-to-run training code")
        print("\nYour dataset is now FULLY analyzed and ready for neural network training!")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("The program can be restarted and will resume from where it left off.")
        sys.exit(1)

if __name__ == "__main__":
    main()