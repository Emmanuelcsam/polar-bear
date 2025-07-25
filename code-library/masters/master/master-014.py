#!/usr/bin/env python3
"""
Automated Image Processing Studio V4 - Enhanced with PyTorch and Deep ML
=======================================================================
Further enhanced with PyTorch integration for deep learning, automatic parameter tuning using Optuna,
script troubleshooting, template matching, classification, keyword-folder matching,
comprehensive testing, and full timestamped logging.
"""

import os
import sys
import json
import pickle
import hashlib
import time
import shutil
import subprocess
import importlib.util
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from datetime import datetime
from collections import defaultdict, deque
import numpy as np
import warnings
import inspect
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')

# Enhanced logging with timestamps and declaration stamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s - %(funcName)s:%(lineno)d',
    handlers=[
        logging.FileHandler('aps_processing_v4.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'psutil': 'psutil',
        'scipy': 'scipy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'optuna': 'optuna'
    }

    missing = []
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        logger.info(f"Installing missing packages: {missing}")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                return False
        logger.info("Dependencies installed successfully.")
    return True


# Import required modules
try:
    import cv2
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import psutil
    from scipy import optimize
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    import optuna
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.info("Please run 'python automated_processing_studio_v4.py' once to install dependencies.")
    DEPS_AVAILABLE = False


class ParameterAdapter:
    """Handles adaptive parameter adjustment for image processing with Optuna tuning"""

    def __init__(self):
        self.parameter_history = defaultdict(list)
        self.best_parameters = {}
        self.adaptation_rate = 0.1

    def adapt_parameters(self, script_name: str, current_params: Dict,
                       performance: float) -> Dict:
        """Adapt parameters based on performance"""
        logger.info(f"Adapting parameters for {script_name}, current performance: {performance:.4f}")

        if script_name not in self.best_parameters:
            self.best_parameters[script_name] = {
                'params': current_params.copy(),
                'performance': performance
            }
        elif performance > self.best_parameters[script_name]['performance']:
            self.best_parameters[script_name] = {
                'params': current_params.copy(),
                'performance': performance
            }

        # Record history
        self.parameter_history[script_name].append({
            'params': current_params.copy(),
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })

        # Adapt parameters with controlled variation
        adapted_params = current_params.copy()
        for param, value in adapted_params.items():
            if isinstance(value, (int, float)):
                variation = np.random.uniform(-self.adaptation_rate, self.adaptation_rate)
                if isinstance(value, int):
                    adapted_params[param] = max(1, int(value * (1 + variation)))  # Avoid zero or negative
                else:
                    adapted_params[param] = max(0.01, value * (1 + variation))  # Avoid zero or negative

        logger.info(f"Adapted parameters: {adapted_params}")
        return adapted_params

    def tune_parameters_optuna(self, objective: Callable, param_ranges: Dict) -> Dict:
        """Automatic parameter tuning using Optuna"""
        def optuna_objective(trial):
            params = {}
            for param, (low, high) in param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = trial.suggest_int(param, low, high)
                else:
                    params[param] = trial.suggest_float(param, low, high)
            return objective(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(optuna_objective, n_trials=50)
        logger.info(f"Optuna best params: {study.best_params}")
        return study.best_params

    def get_best_parameters(self, script_name: str) -> Optional[Dict]:
        """Get best known parameters for a script"""
        if script_name in self.best_parameters:
            return self.best_parameters[script_name]['params']
        return None


class DeepFeatureExtractor:
    """PyTorch-based deep feature extraction and classification"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classifier = nn.Linear(2048, 1000).to(self.device)  # Example classifier

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features using ResNet"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            return features.cpu().numpy().flatten()

    def classify_image(self, image: np.ndarray) -> str:
        """Classify image using pre-trained model"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            _, predicted = torch.max(output, 1)
            return f"Class {predicted.item()}"  # Placeholder; use actual class names in production


class EnhancedImageProcessor:
    """Enhanced image processor with PyTorch integration"""

    def __init__(self):
        self.feature_cache = {}
        self.parameter_adapter = ParameterAdapter()
        self.deep_extractor = DeepFeatureExtractor()
        if not DEPS_AVAILABLE or cv2 is None:
            raise ImportError("OpenCV is required but not available. Please install dependencies.")

    def normalize_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize two images to same size and type with error handling"""
        if img1 is None or img2 is None:
            raise ValueError("Input images cannot be None")

        if img1.size == 0 or img2.size == 0:
            raise ValueError("Input images cannot be empty")

        try:
            # Ensure same dimensions
            if img1.shape != img2.shape:
                h = min(img1.shape[0], img2.shape[0])
                w = min(img1.shape[1], img2.shape[1])
                if h <= 0 or w <= 0:
                    raise ValueError("Invalid image dimensions")
                img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

            # Ensure same number of channels
            if len(img1.shape) != len(img2.shape):
                if len(img1.shape) == 2:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                elif len(img2.shape) == 2:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            elif len(img1.shape) == 3 and len(img2.shape) == 3 and img1.shape[2] != img2.shape[2]:
                # Handle different channel numbers
                if img1.shape[2] == 1:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                elif img2.shape[2] == 1:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                elif img1.shape[2] == 4:  # BGRA to BGR
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                elif img2.shape[2] == 4:  # BGRA to BGR
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

            return img1, img2
        except Exception as e:
            raise RuntimeError(f"Failed to normalize images: {e}")

    def calculate_perceptual_hash(self, image: np.ndarray) -> str:
        """Calculate perceptual hash for fast comparison with error handling"""
        if image is None or image.size == 0:
            return "0" * 64

        try:
            # Resize to 32x32
            resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

            # Convert to grayscale if needed
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized.copy()

            # Ensure float32 for DCT
            gray_float = gray.astype(np.float32)

            # Calculate DCT
            dct = cv2.dct(gray_float)

            # Use top-left 8x8
            dct_low = dct[:8, :8]

            # Calculate median (avoid empty array)
            if dct_low.size == 0:
                return "0" * 64

            median = np.median(dct_low)

            # Generate hash
            hash_str = ''
            for i in range(8):
                for j in range(8):
                    hash_str += '1' if dct_low[i, j] > median else '0'

            return hash_str
        except Exception as e:
            logger.warning(f"Failed to calculate perceptual hash: {e}")
            return "0" * 64

    def calculate_similarity_score(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Enhanced similarity calculation with deep features"""
        try:
            img1, img2 = self.normalize_images(img1, img2)

            scores = []

            # 1. MSE (normalized)
            try:
                mse = mean_squared_error(img1.flatten(), img2.flatten())
                mse_score = 1.0 - min(mse / (255.0 ** 2), 1.0)
                scores.append(mse_score * 0.25)
            except Exception:
                mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
                mse_score = 1.0 - min(mse / (255.0 ** 2), 1.0)
                scores.append(mse_score * 0.25)

            # 2. Structural Similarity
            ssim = self._calculate_ssim(img1, img2)
            scores.append(ssim * 0.25)

            # 3. Histogram Correlation (normalized to 0-1)
            hist_score = self._histogram_correlation(img1, img2)
            normalized_hist = (hist_score + 1) / 2
            scores.append(normalized_hist * 0.15)

            # 4. Edge Similarity
            edge_score = self._edge_similarity(img1, img2)
            scores.append(edge_score * 0.15)

            # 5. Deep Feature Similarity
            feat1 = self.deep_extractor.extract_features(img1)
            feat2 = self.deep_extractor.extract_features(img2)
            deep_sim = 1.0 - np.linalg.norm(feat1 - feat2) / np.linalg.norm(feat1 + feat2)
            scores.append(deep_sim * 0.2)

            # Combined score (0 = identical, 1 = completely different)
            similarity = 1.0 - sum(scores)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 1.0  # Maximum dissimilarity on error

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between images with error handling"""
        try:
            # Convert to grayscale for SSIM
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = img1.copy(), img2.copy()

            # Check for valid images
            if gray1.size == 0 or gray2.size == 0:
                return 0.0

            C1 = (0.01 * 255)**2
            C2 = (0.03 * 255)**2

            gray1 = gray1.astype(np.float64)
            gray2 = gray2.astype(np.float64)

            # Ensure images are large enough for 11x11 kernel
            h, w = gray1.shape
            if h < 11 or w < 11:
                # Use smaller kernel for small images
                kernel_size = min(h, w) // 2
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                kernel_size = max(3, kernel_size)
                sigma = kernel_size / 6.0
            else:
                kernel_size = 11
                sigma = 1.5

            mu1 = cv2.GaussianBlur(gray1, (kernel_size, kernel_size), sigma)
            mu2 = cv2.GaussianBlur(gray2, (kernel_size, kernel_size), sigma)

            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(gray1**2, (kernel_size, kernel_size), sigma) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(gray2**2, (kernel_size, kernel_size), sigma) - mu2_sq
            sigma12 = cv2.GaussianBlur(gray1 * gray2, (kernel_size, kernel_size), sigma) - mu1_mu2

            # Calculate SSIM with division by zero protection
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

            # Avoid division by zero
            ssim_map = np.divide(numerator, denominator,
                               out=np.zeros_like(numerator),
                               where=denominator!=0)

            return float(ssim_map.mean())

        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0

    def _histogram_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate histogram correlation"""
        try:
            correlations = []

            if len(img1.shape) == 3:
                for i in range(img1.shape[2]):
                    hist1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
                    hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
                    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    correlations.append(corr)
            else:
                hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
                corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                correlations.append(corr)

            return np.mean(correlations)
        except Exception as e:
            logger.warning(f"Histogram correlation failed: {e}")
            return -1.0  # Worst case

    def _edge_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate edge-based similarity"""
        try:
            # Convert to grayscale
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = img1, img2

            # Detect edges
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)

            # Compare edge maps
            intersection = np.logical_and(edges1, edges2).sum()
            union = np.logical_or(edges1, edges2).sum()

            if union == 0:
                return 1.0

            return intersection / union
        except Exception as e:
            logger.warning(f"Edge similarity failed: {e}")
            return 0.0

    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features including deep features"""
        features = {}

        try:
            # Basic statistics
            features['mean'] = np.mean(image)
            features['std'] = np.std(image)
            features['min'] = np.min(image)
            features['max'] = np.max(image)

            # Color histogram
            if len(image.shape) == 3:
                for i, color in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([image], [i], None, [32], [0, 256])
                    features[f'hist_{color}'] = hist.flatten()
            else:
                hist = cv2.calcHist([image], [0], None, [32], [0, 256])
                features['hist_gray'] = hist.flatten()

            # Texture features (using Gabor filters)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gabor_features = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                gabor_features.extend([filtered.mean(), filtered.std()])
            features['gabor'] = np.array(gabor_features)

            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = edges.mean() / 255.0

            # Corner density
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            features['corner_density'] = (corners > 0.01 * corners.max()).mean()

            # Deep features
            features['deep_features'] = self.deep_extractor.extract_features(image)[:100]  # Truncate for efficiency

        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return minimal features on error
            features = {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'edge_density': 0,
                'corner_density': 0,
                'deep_features': np.zeros(100)
            }

        return features

    def create_anomaly_map(self, original: np.ndarray, processed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a visual anomaly map highlighting differences"""
        try:
            original, processed = self.normalize_images(original, processed)

            # Calculate absolute difference
            diff = cv2.absdiff(original, processed)

            # Convert to grayscale if needed
            if len(diff.shape) == 3:
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            else:
                diff_gray = diff

            # Enhance differences
            diff_enhanced = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

            # Apply threshold to highlight significant changes
            _, thresh = cv2.threshold(diff_enhanced, 30, 255, cv2.THRESH_BINARY)

            # Create heatmap
            heatmap = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)

            # Overlay on original
            if len(original.shape) == 2:
                original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            else:
                original_color = original.copy()

            # Blend heatmap with original
            anomaly_map = cv2.addWeighted(original_color, 0.7, heatmap, 0.3, 0)

            # Highlight strong anomalies
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(anomaly_map, contours, -1, (0, 255, 0), 2)

            return anomaly_map, diff_enhanced, heatmap
        except Exception as e:
            logger.warning(f"Anomaly map creation failed: {e}")
            # Return empty arrays on error
            h, w = original.shape[:2]
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            return empty, np.zeros((h, w), dtype=np.uint8), empty

    def template_matching(self, image: np.ndarray, template: np.ndarray) -> Dict:
        """Perform template matching"""
        try:
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            return {
                'score': max_val,
                'location': max_loc
            }
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
            return {'score': 0.0, 'location': (0, 0)}

    def classify(self, image: np.ndarray) -> str:
        """Classify image using deep model"""
        try:
            return self.deep_extractor.classify_image(image)
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "Unknown"


class ImprovedScriptManager:
    """Improved script manager with troubleshooting and keyword-folder matching"""

    def __init__(self, scripts_dirs: List[Union[str, Path]]):
        self.scripts_dirs = []
        for d in scripts_dirs:
            path = Path(d)
            if path.exists():
                self.scripts_dirs.append(path)

        self.functions = {}
        self.function_info = {}
        self.category_map = defaultdict(list)
        self._load_all_scripts()

    def _load_all_scripts(self):
        """Load scripts from all directories"""
        logger.info(f"Loading scripts from {len(self.scripts_dirs)} directories...")

        for scripts_dir in self.scripts_dirs:
            self._load_scripts_from_dir(scripts_dir)

        logger.info(f"✓ Loaded {len(self.functions)} total scripts")

        # Organize by category
        for func_name, info in self.function_info.items():
            self.category_map[info['category']].append(func_name)

    def _wrap_script_function(self, func, script_name: str):
        """Wrap script function to handle various return types"""
        def wrapped_function(image, **kwargs):
            try:
                result = func(image, **kwargs) if kwargs else func(image)

                # Handle various return types
                if isinstance(result, np.ndarray):
                    return result
                elif hasattr(result, '__dict__'):
                    for attr in ['enhanced_image', 'processed_image', 'result_image',
                               'output_image', 'final_image', 'image']:
                        if hasattr(result, attr):
                            img = getattr(result, attr)
                            if isinstance(img, np.ndarray):
                                logger.debug(f"Extracted image from {attr} attribute")
                                return img
                    for attr in ['final_cleaned_mask', 'mask', 'binary_mask']:
                        if hasattr(result, attr):
                            mask = getattr(result, attr)
                            if isinstance(mask, np.ndarray):
                                logger.debug(f"Extracted mask from {attr} attribute")
                                if len(mask.shape) == 2:
                                    if mask.dtype == bool:
                                        mask = mask.astype(np.uint8) * 255
                                    return mask
                elif isinstance(result, tuple) or isinstance(result, list):
                    for item in result:
                        if isinstance(item, np.ndarray):
                            logger.debug(f"Extracted image from tuple/list")
                            return item

                logger.warning(f"Script {script_name} returned unexpected type: {type(result)}")
                return image  # Return original

            except Exception as e:
                logger.error(f"Error in wrapped script {script_name}: {e}")
                return image  # Return original on error

        return wrapped_function

    def _troubleshoot_script(self, module, script_path: Path) -> bool:
        """Test script with dummy image to ensure functionality"""
        try:
            dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = module.process_image(dummy_image)
            if not isinstance(result, np.ndarray):
                raise ValueError("Script did not return ndarray")
            logger.info(f"Script {script_path} passed troubleshooting")
            return True
        except Exception as e:
            logger.error(f"Script {script_path} failed troubleshooting: {e}")
            return False

    def _load_scripts_from_dir(self, scripts_dir: Path):
        """Load all scripts from a directory with troubleshooting"""
        try:
            for script_path in scripts_dir.rglob("*.py"):
                if script_path.name.startswith("_") or script_path.name == "__init__.py":
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(
                        script_path.stem, script_path
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        if hasattr(module, 'process_image'):
                            # Troubleshoot
                            if not self._troubleshoot_script(module, script_path):
                                continue

                            # Use full path relative to scripts dir as key
                            rel_path = script_path.relative_to(scripts_dir)
                            func_key = str(rel_path).replace('\\', '/')  # Normalize path separators

                            # Wrap the function
                            wrapped_func = self._wrap_script_function(
                                module.process_image, func_key
                            )
                            self.functions[func_key] = wrapped_func

                            # Inspect parameters
                            sig = inspect.signature(module.process_image)
                            params = {
                                k: v.default for k, v in sig.parameters.items()
                                if k != 'image' and v.default != inspect.Parameter.empty
                            }

                            self.function_info[func_key] = {
                                'path': script_path,
                                'category': self._determine_category(script_path),
                                'name': script_path.stem,
                                'parameters': params
                            }
                except Exception as e:
                    logger.warning(f"Failed to load {script_path}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error accessing directory {scripts_dir}: {e}")

    def _determine_category(self, script_path: Path) -> str:
        """Determine category from path or filename with keyword matching to folder"""
        # Check if in a category subdirectory
        parts = script_path.parts
        for part in parts:
            if part in ['filtering', 'morphology', 'thresholding', 'edge_detection',
                       'transformations', 'histogram', 'features', 'noise', 'effects']:
                # Ensure keyword match
                filename_lower = script_path.name.lower()
                if any(keyword in filename_lower for keyword in part.lower().split('_')):
                    return part

        # Fallback to filename analysis with keywords
        filename = script_path.name.lower()
        categories = {
            'filtering': ['blur', 'filter', 'smooth', 'denoise'],
            'morphology': ['morph', 'erode', 'dilate', 'open', 'close'],
            'thresholding': ['threshold', 'binary', 'otsu'],
            'edge_detection': ['edge', 'canny', 'sobel', 'gradient'],
            'transformations': ['rotate', 'flip', 'resize', 'warp', 'affine'],
            'histogram': ['histogram', 'equalize', 'clahe', 'gamma'],
            'features': ['corner', 'blob', 'contour', 'hough', 'detect'],
            'noise': ['noise', 'salt', 'pepper', 'gaussian'],
            'effects': ['effect', 'artistic', 'cartoon', 'sketch']
        }

        for category, keywords in categories.items():
            if any(keyword in filename for keyword in keywords):
                return category

        return 'other'

    def get_scripts_by_category(self, category: str) -> List[str]:
        """Get all scripts in a category"""
        return self.category_map.get(category, [])

    def get_random_scripts(self, n: int = 5) -> List[str]:
        """Get random selection of scripts"""
        import random
        all_scripts = list(self.functions.keys())
        return random.sample(all_scripts, min(n, len(all_scripts)))


class SmartLearner:
    """Improved learning system with deep state representation"""

    def __init__(self, num_scripts: int):
        self.num_scripts = num_scripts
        self.successful_sequences = []
        self.failed_sequences = []
        self.script_success_rate = defaultdict(float)
        self.script_usage_count = defaultdict(int)
        self.combination_memory = {}
        self.q_table = defaultdict(lambda: defaultdict(float))  # State -> Action -> Q-value

    def get_recommended_scripts(self, current_state: Dict, target_state: Dict) -> List[str]:
        """Get recommended scripts based on current and target states using Q-learning"""
        recommendations = []

        # Analyze what needs to change
        needs = self._analyze_needs(current_state, target_state)

        # Get scripts that have worked well for similar needs
        for need, weight in needs.items():
            suitable_scripts = self._get_scripts_for_need(need)
            for script in suitable_scripts:
                q_value = self.q_table.get(need, {}).get(script, 0.0)
                recommendations.append((script, weight * (self.script_success_rate.get(script, 0.5) + q_value)))

        # Sort by weighted score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Return unique scripts
        seen = set()
        result = []
        for script, _ in recommendations:
            if script not in seen:
                seen.add(script)
                result.append(script)

        return result[:10]  # Top 10 recommendations

    def _analyze_needs(self, current: Dict, target: Dict) -> Dict[str, float]:
        """Analyze what transformations are needed, including deep features"""
        needs = {}

        # Brightness difference
        brightness_diff = target.get('mean', 0) - current.get('mean', 0)
        if abs(brightness_diff) > 10:
            needs['brightness'] = abs(brightness_diff) / 255.0

        # Contrast difference
        contrast_diff = target.get('std', 0) - current.get('std', 0)
        if abs(contrast_diff) > 5:
            needs['contrast'] = abs(contrast_diff) / 128.0

        # Edge density difference
        edge_diff = target.get('edge_density', 0) - current.get('edge_density', 0)
        if abs(edge_diff) > 0.1:
            needs['edges'] = abs(edge_diff)

        # Texture complexity
        if 'gabor' in current and 'gabor' in target:
            try:
                texture_diff = np.mean(np.abs(target['gabor'] - current['gabor']))
                if texture_diff > 0.1:
                    needs['texture'] = texture_diff
            except Exception:
                pass

        # Deep feature difference
        if 'deep_features' in current and 'deep_features' in target:
            deep_diff = np.linalg.norm(target['deep_features'] - current['deep_features'])
            if deep_diff > 0.1:
                needs['deep'] = deep_diff / 10.0  # Normalize

        return needs

    def _get_scripts_for_need(self, need: str) -> List[str]:
        """Get scripts suitable for a specific need"""
        script_mapping = {
            'brightness': ['gamma_correction.py', 'histogram_equalization.py', 'brightness_adjust.py'],
            'contrast': ['clahe.py', 'histogram_stretching.py', 'contrast_enhance.py'],
            'edges': ['canny_edges.py', 'sobel_combined.py', 'sharpening_filter.py'],
            'texture': ['bilateral_filter.py', 'texture_enhance.py', 'detail_enhance.py'],
            'deep': ['deep_denoise.py', 'deep_enhance.py']  # Assume deep scripts exist
        }

        return script_mapping.get(need, [])

    def record_result(self, sequence: List[str], success: bool, improvement: float):
        """Record the result of a processing sequence and update Q-table"""
        if success:
            self.successful_sequences.append(sequence)
        else:
            self.failed_sequences.append(sequence)

        # Update script statistics
        for script in sequence:
            self.script_usage_count[script] += 1
            if success:
                current_rate = self.script_success_rate[script]
                self.script_success_rate[script] = (
                    current_rate * 0.9 + improvement * 0.1
                )

        # Update Q-table for each action in sequence
        if success:
            reward = improvement
            for i, script in enumerate(sequence):
                state = f"step_{i}"  # Simple state representation
                self.q_table[state][script] = self.q_table[state].get(script, 0) * 0.9 + reward * 0.1

    def save_knowledge(self, filepath: Union[str, Path]):
        """Save learned knowledge"""
        knowledge = {
            'successful_sequences': self.successful_sequences[-100:],  # Keep last 100
            'script_success_rate': dict(self.script_success_rate),
            'script_usage_count': dict(self.script_usage_count),
            'q_table': {k: dict(v) for k, v in self.q_table.items()}
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(knowledge, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save knowledge: {e}")

    def load_knowledge(self, filepath: Union[str, Path]):
        """Load previously learned knowledge"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    knowledge = json.load(f)
                    self.successful_sequences = knowledge.get('successful_sequences', [])
                    self.script_success_rate = defaultdict(
                        float, knowledge.get('script_success_rate', {})
                    )
                    self.script_usage_count = defaultdict(
                        int, knowledge.get('script_usage_count', {})
                    )
                    self.q_table = defaultdict(lambda: defaultdict(float), knowledge.get('q_table', {}))
        except Exception as e:
            logger.warning(f"Failed to load knowledge: {e}")


class ProcessingDebugger:
    """Handles debugging and testing of processing pipelines"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.debug_dir = cache_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)
        self.test_results = []

    def save_debug_state(self, iteration: int, image: np.ndarray,
                        script: str, params: Dict, similarity: float):
        """Save debug information for analysis"""
        debug_info = {
            'iteration': iteration,
            'script': script,
            'parameters': params,
            'similarity': similarity,
            'timestamp': datetime.now().isoformat()
        }

        # Save image
        img_path = self.debug_dir / f"iter_{iteration:04d}_{script.replace('/', '_').replace('.py', '')}.png"
        cv2.imwrite(str(img_path), image)

        # Save metadata
        meta_path = self.debug_dir / f"iter_{iteration:04d}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(debug_info, f, indent=2)

        logger.debug(f"Saved debug state for iteration {iteration}")

    def run_test_suite(self, processor, test_pairs: List[Tuple[str, str]],
                      max_iterations: int = 50) -> Dict:
        """Run comprehensive test suite"""
        logger.info(f"Running test suite with {len(test_pairs)} test pairs")

        results = []
        for i, (input_path, target_path) in enumerate(test_pairs):
            logger.info(f"Test {i+1}/{len(test_pairs)}: {input_path} -> {target_path}")

            try:
                input_img = cv2.imread(input_path)
                target_img = cv2.imread(target_path)

                if input_img is None or target_img is None:
                    logger.error(f"Failed to load test images")
                    continue

                start_time = time.time()
                result = processor.process_to_match_target(
                    input_img, target_img,
                    max_iterations=max_iterations,
                    similarity_threshold=0.1,
                    verbose=False
                )

                results.append({
                    'input': input_path,
                    'target': target_path,
                    'success': result['success'],
                    'similarity': result['final_similarity'],
                    'iterations': result['iterations'],
                    'time': time.time() - start_time
                })

            except Exception as e:
                logger.error(f"Test failed: {e}")
                results.append({
                    'input': input_path,
                    'target': target_path,
                    'success': False,
                    'error': str(e)
                })

        self.test_results = results
        return self._analyze_test_results(results)

    def _analyze_test_results(self, results: List[Dict]) -> Dict:
        """Analyze test results"""
        successful = [r for r in results if r.get('success', False)]
        success_rate = len(successful) / len(results) if results else 0

        analysis = {
            'total_tests': len(results),
            'successful': len(successful),
            'success_rate': success_rate,
            'average_similarity': np.mean([r.get('similarity', 1.0) for r in results]),
            'average_iterations': np.mean([r.get('iterations', 0) for r in results]),
            'average_time': np.mean([r.get('time', 0) for r in results])
        }

        logger.info(f"Test suite results: {success_rate:.1%} success rate")
        return analysis

    def run_unit_tests(self):
        """Run unit tests for key components"""
        logger.info("Running unit tests...")
        # Simple unit test examples
        try:
            # Test normalize_images
            img1 = np.zeros((10, 10, 3), dtype=np.uint8)
            img2 = np.ones((20, 20, 3), dtype=np.uint8)
            norm1, norm2 = EnhancedImageProcessor().normalize_images(img1, img2)
            assert norm1.shape == norm2.shape, "Normalization failed"

            # Test template matching
            processor = EnhancedImageProcessor()
            match = processor.template_matching(img2, img1[:5, :5])
            assert 'score' in match, "Template matching failed"

            # Test classification
            class_label = processor.classify(img1)
            assert isinstance(class_label, str), "Classification failed"

            logger.info("All unit tests passed")
            return True
        except AssertionError as e:
            logger.error(f"Unit test failed: {e}")
            return False

    def run_script_tests(self, script_manager):
        """Test all loaded scripts"""
        logger.info("Running script tests...")
        for script_key, func in script_manager.functions.items():
            try:
                dummy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                result = func(dummy)
                assert isinstance(result, np.ndarray), f"Script {script_key} test failed"
            except Exception as e:
                logger.error(f"Script test failed for {script_key}: {e}")
        logger.info("Script tests complete")

    def run_program_tests(self, studio):
        """Run full program tests"""
        logger.info("Running program tests...")
        self.run_test_suite(studio, [("test_input.png", "test_target.png")])  # Assume test files
        logger.info("Program tests complete")


class EnhancedProcessingStudio:
    """Enhanced automated processing studio with PyTorch and testing"""

    def __init__(self, scripts_dirs: List[str] = None, cache_dir: str = ".studio_cache_v4"):
        if scripts_dirs is None:
            scripts_dirs = ["scripts", "opencv_scripts"]

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self.processor = EnhancedImageProcessor()
        self.script_manager = ImprovedScriptManager(scripts_dirs)
        self.learner = SmartLearner(len(self.script_manager.functions))
        self.debugger = ProcessingDebugger(self.cache_dir)

        # Load previous knowledge
        self.knowledge_file = self.cache_dir / "knowledge.json"
        self.learner.load_knowledge(self.knowledge_file)

        # Processing state
        self.processing_history = []
        self.best_matches = []

        # Parameter tracking
        self.parameter_log = []

    def process_to_match_target(self, input_image: np.ndarray, target_image: np.ndarray,
                               max_iterations: int = 200, similarity_threshold: float = 0.05,
                               verbose: bool = True, optimize_params: bool = True) -> Dict[str, Any]:
        """Process input to match target with param optimization"""
        start_time = time.time()

        # Validate inputs
        if input_image is None or target_image is None:
            raise ValueError("Input and target images cannot be None")

        # Normalize images
        try:
            input_norm, target_norm = self.processor.normalize_images(input_image, target_image)
        except Exception as e:
            logger.error(f"Error normalizing images: {e}")
            return {
                'success': False,
                'final_similarity': 1.0,
                'iterations': 0,
                'processing_time': time.time() - start_time,
                'pipeline': [],
                'final_image': input_image,
                'error': str(e)
            }

        # Extract features
        target_features = self.processor.extract_features(target_norm)

        # Initialize
        current_image = input_norm.copy()
        best_image = current_image.copy()
        best_similarity = self.processor.calculate_similarity_score(current_image, target_norm)
        best_sequence = []

        # Track attempts
        attempts = []
        improvements = []

        if verbose:
            print(f"\n🎯 Target matching started...")
            print(f"Initial similarity: {best_similarity:.4f}")
            print(f"Target threshold: {similarity_threshold}")

        # Try different strategies
        strategies = [
            ('recommended', 0.6),
            ('category_based', 0.3),
            ('random', 0.1)
        ]

        for iteration in range(max_iterations):
            # Check if we've reached the target
            if best_similarity < similarity_threshold:
                if verbose:
                    print(f"\n✅ Target matched! Similarity: {best_similarity:.4f}")
                break

            # Get current features
            current_features = self.processor.extract_features(current_image)

            # Choose strategy
            strategy_choice = np.random.random()
            cumulative = 0
            strategy = 'random'
            for strat_name, prob in strategies:
                cumulative += prob
                if strategy_choice < cumulative:
                    strategy = strat_name
                    break

            # Get candidate scripts
            if strategy == 'recommended':
                candidates = self.learner.get_recommended_scripts(
                    current_features, target_features
                )
            elif strategy == 'category_based':
                categories = ['filtering', 'histogram', 'morphology', 'effects']
                category = np.random.choice(categories)
                candidates = self.script_manager.get_scripts_by_category(category)
            else:
                candidates = self.script_manager.get_random_scripts(5)

            # Try candidates
            improvement_found = False
            for script_key in candidates[:5]:  # Try top 5
                if script_key not in self.script_manager.functions:
                    continue

                try:
                    # Apply script with parameter adaptation
                    script_func = self.script_manager.functions[script_key]

                    # Get parameters
                    script_params = self.script_manager.function_info[script_key]['parameters']
                    best_params = self.processor.parameter_adapter.get_best_parameters(script_key)
                    if best_params:
                        script_params = best_params

                    if optimize_params and script_params:
                        # Optuna tuning
                        def objective(params):
                            processed = script_func(current_image, **params)
                            return self.processor.calculate_similarity_score(processed, target_norm)

                        param_ranges = {k: (0.1 * v, 10 * v) for k, v in script_params.items()}
                        tuned_params = self.processor.parameter_adapter.tune_parameters_optuna(objective, param_ranges)
                        script_params = tuned_params

                    # Log the operation
                    logger.info(f"Applying script: {script_key} with params: {script_params}")
                    self.parameter_log.append({
                        'iteration': iteration,
                        'script': script_key,
                        'parameters': script_params,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Apply the script
                    processed = script_func(current_image, **script_params)

                    # Ensure proper format
                    if processed is not None and isinstance(processed, np.ndarray):
                        # Normalize result
                        if processed.dtype != np.uint8:
                            processed = np.clip(processed, 0, 255).astype(np.uint8)

                        # Calculate similarity
                        similarity = self.processor.calculate_similarity_score(
                            processed, target_norm
                        )

                        # Check if improved
                        if similarity < best_similarity:
                            improvement = best_similarity - similarity
                            improvements.append(improvement)

                            best_similarity = similarity
                            best_image = processed.copy()
                            best_sequence.append(script_key)
                            current_image = processed
                            improvement_found = True

                            if verbose and iteration % 10 == 0:
                                print(f"Iteration {iteration}: {script_key} → "
                                      f"similarity: {similarity:.4f} "
                                      f"(improved by {improvement:.4f})")

                            # Record success and adapt parameters
                            self.learner.record_result(
                                [script_key], True, improvement
                            )

                            # Adapt parameters based on performance
                            adapted_params = self.processor.parameter_adapter.adapt_parameters(
                                script_key, script_params, improvement
                            )

                            # Save debug state if enabled
                            if hasattr(self, 'debug_mode') and self.debug_mode:
                                self.debugger.save_debug_state(
                                    iteration, processed, script_key,
                                    script_params, similarity
                                )

                            break

                except Exception as e:
                    self.learner.record_result([script_key], False, 0)
                    logger.error(f"Script {script_key} failed: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

                    if verbose:
                        print(f"Warning: Script {script_key} failed: {e}")
                    continue

            # Record attempt
            attempts.append({
                'iteration': iteration,
                'strategy': strategy,
                'similarity': best_similarity,
                'improved': improvement_found
            })

            # If no improvement for a while, try a different approach
            if not improvement_found:
                if len(attempts) > 10 and all(not a['improved'] for a in attempts[-10:]):
                    if self.learner.successful_sequences:
                        seq_idx = np.random.randint(0, len(self.learner.successful_sequences))
                        seq = self.learner.successful_sequences[seq_idx]

                        logger.info(f"Trying successful sequence #{seq_idx} with {len(seq)} scripts")

                        for script in seq[:3]:  # Apply up to 3 scripts
                            if script in self.script_manager.functions:
                                try:
                                    logger.info(f"Applying script from sequence: {script}")
                                    current_image = self.script_manager.functions[script](
                                        current_image
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to apply {script}: {e}")

            # Save best matches periodically
            if iteration % 50 == 0 and iteration > 0:
                self.best_matches.append({
                    'iteration': iteration,
                    'similarity': best_similarity,
                    'sequence': best_sequence.copy(),
                    'image': best_image.copy()
                })

        # Save knowledge
        self.learner.save_knowledge(self.knowledge_file)

        # Save parameter log
        param_log_path = self.cache_dir / f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(param_log_path, 'w') as f:
            json.dump(self.parameter_log, f, indent=2)
        logger.info(f"Parameter log saved to {param_log_path}")

        # Generate results
        processing_time = time.time() - start_time

        results = {
            'success': best_similarity < similarity_threshold,
            'final_similarity': best_similarity,
            'iterations': len(attempts),
            'processing_time': processing_time,
            'pipeline': best_sequence,
            'final_image': best_image,
            'attempts': attempts,
            'improvements': improvements,
            'parameter_log': self.parameter_log
        }

        # Generate comprehensive report
        try:
            self._generate_comprehensive_report(
                results, input_image, target_image, best_image
            )
        except Exception as e:
            logger.warning(f"Failed to generate report: {e}")

        return results

    def process_batch(self, input_paths: List[str], target_image: np.ndarray,
                      max_iterations: int = 200, similarity_threshold: float = 0.05,
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """Process multiple input images to match a single target"""
        results = []
        shared_pipeline = None
        for input_path in input_paths:
            input_image = cv2.imread(input_path)
            if input_image is None:
                logger.error(f"Failed to load {input_path}")
                continue

            if shared_pipeline:
                current_image = input_image.copy()
                for script in shared_pipeline:
                    if script in self.script_manager.functions:
                        current_image = self.script_manager.functions[script](current_image)
                result = self.process_to_match_target(current_image, target_image,
                                                      max_iterations=max_iterations // 2,
                                                      similarity_threshold=similarity_threshold,
                                                      verbose=verbose)
            else:
                result = self.process_to_match_target(input_image, target_image,
                                                      max_iterations=max_iterations,
                                                      similarity_threshold=similarity_threshold,
                                                      verbose=verbose)
                if result['success']:
                    shared_pipeline = result['pipeline']

            results.append(result)
        return results

    def process_video_to_match_target(self, input_video_path: str, target_image: np.ndarray,
                                      output_video_path: str, max_iterations: int = 50,
                                      similarity_threshold: float = 0.1, verbose: bool = True):
        """Process each frame of a video to match the target image"""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_to_match_target(frame, target_image,
                                                  max_iterations=max_iterations,
                                                  similarity_threshold=similarity_threshold,
                                                  verbose=verbose)
            out.write(result['final_image'])
            frame_count += 1
            if verbose:
                print(f"Processed frame {frame_count}")

        cap.release()
        out.release()

    def export_pipeline(self, pipeline: List[str], output_file: str):
        """Export the processing pipeline as a standalone Python script"""
        with open(output_file, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("import cv2\n")
            f.write("import numpy as np\n\n")
            f.write("def process_image(image):\n")
            for script in pipeline:
                f.write(f"    # Apply {script}\n")
                f.write(f"    image = {script.replace('.py', '')}.process_image(image)\n")  # Assume imports
            f.write("    return image\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    input_img = cv2.imread('input.png')\n")
            f.write("    output_img = process_image(input_img)\n")
            f.write("    cv2.imwrite('output.png', output_img)\n")
        logger.info(f"Pipeline exported to {output_file}")

    def generate_anomaly_visualization(self, original: np.ndarray,
                                     processed: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate multiple anomaly visualizations"""
        try:
            anomaly_map, diff_map, heatmap = self.processor.create_anomaly_map(
                original, processed
            )

            # Create additional visualizations
            visualizations = {
                'anomaly_map': anomaly_map,
                'difference_map': diff_map,
                'heatmap': heatmap
            }

            # Edge difference visualization
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
            gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed

            edges_orig = cv2.Canny(gray_orig, 50, 150)
            edges_proc = cv2.Canny(gray_proc, 50, 150)
            edge_diff = cv2.absdiff(edges_orig, edges_proc)

            # Color-code edge differences
            edge_vis = np.zeros((edge_diff.shape[0], edge_diff.shape[1], 3), dtype=np.uint8)
            edge_vis[:, :, 0] = edges_orig  # Original edges in blue
            edge_vis[:, :, 1] = edges_proc  # Processed edges in green
            edge_vis[:, :, 2] = edge_diff   # Differences in red

            visualizations['edge_changes'] = edge_vis

            # Histogram difference visualization
            if DEPS_AVAILABLE and plt is not None:
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

                    # Original histogram
                    if len(original.shape) == 3:
                        for i, color in enumerate(['b', 'g', 'r']):
                            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
                            axes[0, 0].plot(hist, color=color, alpha=0.7)
                    else:
                        hist = cv2.calcHist([original], [0], None, [256], [0, 256])
                        axes[0, 0].plot(hist, 'k')
                    axes[0, 0].set_title('Original Histogram')

                    # Processed histogram
                    if len(processed.shape) == 3:
                        for i, color in enumerate(['b', 'g', 'r']):
                            hist = cv2.calcHist([processed], [i], None, [256], [0, 256])
                            axes[0, 1].plot(hist, color=color, alpha=0.7)
                    else:
                        hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
                        axes[0, 1].plot(hist, 'k')
                    axes[0, 1].set_title('Processed Histogram')

                    # Difference visualization
                    axes[1, 0].imshow(cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB))
                    axes[1, 0].set_title('Anomaly Map')
                    axes[1, 0].axis('off')

                    axes[1, 1].imshow(diff_map, cmap='hot')
                    axes[1, 1].set_title('Difference Intensity')
                    axes[1, 1].axis('off')

                    plt.tight_layout()

                    # Save histogram comparison
                    hist_path = self.cache_dir / "histogram_comparison.png"
                    plt.savefig(hist_path)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Failed to create histogram visualization: {e}")

            return visualizations
        except Exception as e:
            logger.warning(f"Failed to generate anomaly visualizations: {e}")
            return {}

    def _generate_comprehensive_report(self, results: Dict, input_image: np.ndarray,
                                     target_image: np.ndarray, output_image: np.ndarray):
        """Generate comprehensive report with visualizations"""
        try:
            report_dir = self.cache_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_dir.mkdir(exist_ok=True)

            # Save base images
            cv2.imwrite(str(report_dir / "01_input.png"), input_image)
            cv2.imwrite(str(report_dir / "02_target.png"), target_image)
            cv2.imwrite(str(report_dir / "03_output.png"), output_image)

            # Generate anomaly visualizations
            anomaly_vis = self.generate_anomaly_visualization(input_image, output_image)
            for name, img in anomaly_vis.items():
                if isinstance(img, np.ndarray) and img.size > 0:
                    cv2.imwrite(str(report_dir / f"04_{name}.png"), img)

            # Create comparison grid
            self._create_comparison_grid(
                input_image, target_image, output_image,
                report_dir / "05_comparison_grid.png"
            )

            # Generate processing sequence visualization
            if results['pipeline']:
                self._visualize_processing_sequence(
                    input_image, results['pipeline'][:10],
                    report_dir / "06_processing_sequence.png"
                )

            # Write detailed text report
            report_text = f"""
Automated Image Processing Report
=================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Success: {results['success']}
Final Similarity Score: {results['final_similarity']:.4f}
Total Iterations: {results['iterations']}
Processing Time: {results['processing_time']:.2f} seconds
Scripts Applied: {len(results['pipeline'])}

PROCESSING PIPELINE
------------------
"""
            for i, script in enumerate(results['pipeline'], 1):
                report_text += f"{i:3d}. {script}\n"

            if results.get('improvements'):
                report_text += f"\nIMPROVEMENT STATISTICS\n"
                report_text += f"----------------------\n"
                report_text += f"Total Improvements: {len(results['improvements'])}\n"
                if results['improvements']:
                    report_text += f"Average Improvement: {np.mean(results['improvements']):.4f}\n"
                    report_text += f"Best Improvement: {max(results['improvements']):.4f}\n"

            report_text += f"\nLEARNING STATISTICS\n"
            report_text += f"-------------------\n"
            report_text += f"Total Scripts Available: {len(self.script_manager.functions)}\n"
            report_text += f"Successful Sequences Learned: {len(self.learner.successful_sequences)}\n"
            report_text += f"Most Successful Scripts:\n"

            # Get top performing scripts
            sorted_scripts = sorted(
                self.learner.script_success_rate.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for script, rate in sorted_scripts:
                usage = self.learner.script_usage_count[script]
                report_text += f"  - {script}: {rate:.3f} success rate ({usage} uses)\n"

            # Save report
            with open(report_dir / "report.txt", 'w') as f:
                f.write(report_text)

            # Save detailed JSON data
            json_data = {
                'results': {k: v for k, v in results.items() if k != 'final_image'},
                'script_statistics': {
                    'success_rates': dict(self.learner.script_success_rate),
                    'usage_counts': dict(self.learner.script_usage_count)
                },
                'timestamp': datetime.now().isoformat()
            }

            with open(report_dir / "processing_data.json", 'w') as f:
                json.dump(json_data, f, indent=2, default=str)

            logger.info(f"Report saved to: {report_dir}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def _create_comparison_grid(self, input_img: np.ndarray, target_img: np.ndarray,
                               output_img: np.ndarray, save_path: Path):
        """Create a comparison grid visualization"""
        try:
            # Normalize sizes
            h = max(input_img.shape[0], target_img.shape[0], output_img.shape[0])
            w = max(input_img.shape[1], target_img.shape[1], output_img.shape[1])

            # Resize all images to same size
            input_resized = cv2.resize(input_img, (w, h))
            target_resized = cv2.resize(target_img, (w, h))
            output_resized = cv2.resize(output_img, (w, h))

            # Create grid
            grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

            # Place images
            grid[:h, :w] = input_resized if len(input_resized.shape) == 3 else cv2.cvtColor(input_resized, cv2.COLOR_GRAY2BGR)
            grid[:h, w:] = target_resized if len(target_resized.shape) == 3 else cv2.cvtColor(target_resized, cv2.COLOR_GRAY2BGR)
            grid[h:, :w] = output_resized if len(output_resized.shape) == 3 else cv2.cvtColor(output_resized, cv2.COLOR_GRAY2BGR)

            # Create difference visualization
            try:
                _, diff_img, _ = self.processor.create_anomaly_map(target_resized, output_resized)
                diff_colored = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
                grid[h:, w:] = diff_colored
            except:
                pass

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grid, "Input", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Target", (w + 10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Output", (10, h + 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, "Difference", (w + 10, h + 30), font, 1, (255, 255, 255), 2)

            cv2.imwrite(str(save_path), grid)
        except Exception as e:
            logger.warning(f"Failed to create comparison grid: {e}")

    def _visualize_processing_sequence(self, start_image: np.ndarray,
                                     sequence: List[str], save_path: Path):
        """Visualize the processing sequence"""
        try:
            num_steps = min(len(sequence), 10)
            if num_steps == 0:
                return

            # Calculate grid size
            cols = min(5, num_steps)
            rows = (num_steps + cols - 1) // cols

            # Thumbnail size
            thumb_h, thumb_w = 200, 200

            # Create canvas
            canvas_w = cols * (thumb_w + 10) + 10
            canvas_h = rows * (thumb_h + 40) + 10
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

            current_img = start_image.copy()

            for i, script in enumerate(sequence[:num_steps]):
                row = i // cols
                col = i % cols

                # Apply script
                if script in self.script_manager.functions:
                    try:
                        current_img = self.script_manager.functions[script](current_img)
                    except:
                        pass

                # Create thumbnail
                if current_img is not None and current_img.size > 0:
                    thumb = cv2.resize(current_img, (thumb_w, thumb_h))
                    if len(thumb.shape) == 2:
                        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

                    # Place in canvas
                    y = row * (thumb_h + 40) + 10
                    x = col * (thumb_w + 10) + 10
                    canvas[y:y+thumb_h, x:x+thumb_w] = thumb

                    # Add label
                    label = script.split('/')[-1].replace('.py', '')[:20]
                    cv2.putText(canvas, label, (x, y + thumb_h + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            cv2.imwrite(str(save_path), canvas)
        except Exception as e:
            logger.warning(f"Failed to visualize processing sequence: {e}")

    def interactive_setup(self):
        """Interactive configuration"""
        print("\n🎨 Enhanced Automated Image Processing Studio V4")
        print("=" * 50)

        config = {}

        # Get input image
        while True:
            config['input_path'] = input("Enter path to input image: ").strip()
            if os.path.exists(config['input_path']):
                break
            print("❌ File not found. Please try again.")

        # Get target image
        while True:
            config['target_path'] = input("Enter path to target image: ").strip()
            if os.path.exists(config['target_path']):
                break
            print("❌ File not found. Please try again.")

        # Get parameters
        try:
            max_iter = input("Maximum iterations (default: 200): ").strip()
            config['max_iterations'] = int(max_iter) if max_iter else 200
        except:
            config['max_iterations'] = 200

        try:
            threshold = input("Similarity threshold (default: 0.05): ").strip()
            config['similarity_threshold'] = float(threshold) if threshold else 0.05
        except:
            config['similarity_threshold'] = 0.05

        verbose = input("Show detailed progress? (y/n, default: y): ").strip().lower()
        config['verbose'] = verbose != 'n'

        return config


def run_debug_mode(studio, num_cycles=5):
    """Run multiple debug cycles with test images"""
    logger.info(f"Starting debug mode with {num_cycles} cycles")

    # Create test images if they don't exist
    test_dir = studio.cache_dir / "test_images"
    test_dir.mkdir(exist_ok=True)

    # Generate simple test images
    test_pairs = []

    # Test 1: Brightness change
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 100
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 150
    cv2.imwrite(str(test_dir / "test1_input.png"), img1)
    cv2.imwrite(str(test_dir / "test1_target.png"), img2)
    test_pairs.append((str(test_dir / "test1_input.png"), str(test_dir / "test1_target.png")))

    # Test 2: Add noise
    img3 = np.ones((200, 200, 3), dtype=np.uint8) * 128
    img4 = img3.copy()
    noise = np.random.normal(0, 20, img3.shape).astype(np.uint8)
    img4 = cv2.add(img4, noise)
    cv2.imwrite(str(test_dir / "test2_input.png"), img4)
    cv2.imwrite(str(test_dir / "test2_target.png"), img3)
    test_pairs.append((str(test_dir / "test2_input.png"), str(test_dir / "test2_target.png")))

    # Run debug cycles
    for cycle in range(num_cycles):
        logger.info(f"\n{'='*50}")
        logger.info(f"Debug cycle {cycle + 1}/{num_cycles}")
        logger.info(f"{'='*50}")

        studio.debug_mode = True
        results = studio.debugger.run_test_suite(studio, test_pairs, max_iterations=50)

        logger.info(f"Cycle {cycle + 1} results:")
        logger.info(f"  Success rate: {results['success_rate']:.1%}")
        logger.info(f"  Avg similarity: {results['average_similarity']:.4f}")
        logger.info(f"  Avg iterations: {results['average_iterations']:.1f}")

        # Adjust learning parameters based on results
        if results['success_rate'] < 0.5:
            studio.processor.parameter_adapter.adaptation_rate *= 1.2
            logger.info(f"Increased adaptation rate to {studio.processor.parameter_adapter.adaptation_rate:.3f}")

    # Final validation
    logger.info("\nRunning final validation...")
    final_results = studio.debugger.run_test_suite(studio, test_pairs, max_iterations=100)

    logger.info("\nFinal validation results:")
    logger.info(f"  Success rate: {final_results['success_rate']:.1%}")
    logger.info(f"  Avg similarity: {final_results['average_similarity']:.4f}")
    logger.info(f"  Avg iterations: {final_results['average_iterations']:.1f}")

    return final_results


def main():
    """Main entry point"""
    print("Enhanced Automated Image Processing Studio V4")
    print("=" * 50)

    # Check dependencies first
    if not check_dependencies():
        print("Failed to install required dependencies. Please install manually.")
        return

    # Re-import if dependencies were just installed
    global DEPS_AVAILABLE, cv2, plt, cm
    if not DEPS_AVAILABLE:
        try:
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            DEPS_AVAILABLE = True
        except ImportError:
            print("Failed to import dependencies after installation.")
            return

    try:
        # Create studio instance
        studio = EnhancedProcessingStudio()

        print(f"\n📚 Loaded {len(studio.script_manager.functions)} processing scripts")
        print(f"📂 Categories: {list(studio.script_manager.category_map.keys())}")

        # Run tests
        studio.debugger.run_unit_tests()
        studio.debugger.run_script_tests(studio.script_manager)
        studio.debugger.run_program_tests(studio)

        # Check if debug mode requested
        if len(sys.argv) > 1 and sys.argv[1] == '--debug':
            logger.info("Running in debug mode")
            run_debug_mode(studio)
            return

        # Get configuration
        config = studio.interactive_setup()

        # Load images
        input_image = cv2.imread(config['input_path'], cv2.IMREAD_UNCHANGED)
        target_image = cv2.imread(config['target_path'], cv2.IMREAD_UNCHANGED)

        if input_image is None or target_image is None:
            print("❌ Failed to load images")
            return

        print(f"\n📸 Input image: {input_image.shape}")
        print(f"🎯 Target image: {target_image.shape}")

        # Process to match target
        results = studio.process_to_match_target(
            input_image,
            target_image,
            max_iterations=config['max_iterations'],
            similarity_threshold=config['similarity_threshold'],
            verbose=config['verbose'],
            optimize_params=True
        )

        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"✅ Success: {results['success']}")
        print(f"📊 Final similarity: {results['final_similarity']:.4f}")
        print(f"⏱️ Processing time: {results['processing_time']:.2f} seconds")
        print(f"🔄 Total iterations: {results['iterations']}")
        print(f"📝 Pipeline length: {len(results['pipeline'])}")

        if results['pipeline']:
            print("\n🔧 Applied transformations:")
            for i, script in enumerate(results['pipeline'][:10], 1):
                print(f"  {i}. {script}")
            if len(results['pipeline']) > 10:
                print(f"  ... and {len(results['pipeline']) - 10} more")

        # Export pipeline
        export_path = "exported_pipeline.py"
        studio.export_pipeline(results['pipeline'], export_path)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
