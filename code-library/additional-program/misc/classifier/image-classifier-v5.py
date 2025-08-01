#!/usr/bin/env python3
"""
Advanced Deep Learning Image Classifier and Renamer with Knowledge Bank
Analyzes, learns, and classifies images with continuous improvement capabilities
Enhanced for hierarchical folder structure and dynamic folder creation
Fixed and improved version combining best features from all versions
"""

import os
import sys
import subprocess
import importlib
import time
import json
import pickle
from datetime import datetime
from pathlib import Path
import hashlib
from collections import defaultdict, Counter
import re
import shutil
import traceback
import logging

# Core functionality setup
def install_package(package_name, import_name=None):
    """Install a package using pip with the latest version"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"[{timestamp()}] {package_name} not found. Installing latest version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            print(f"[{timestamp()}] Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            print(f"[{timestamp()}] ERROR: Failed to install {package_name}")
            return False

def timestamp():
    """Return current timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

print(f"[{timestamp()}] Starting Advanced Image Classifier and Renamer")
print(f"[{timestamp()}] Checking and installing required dependencies...")

# Check and install required packages
required_packages = [
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("opencv-python", "cv2"),
    ("imagehash", "imagehash"),
    ("tqdm", "tqdm")
]

for package, import_name in required_packages:
    if not install_package(package, import_name):
        print(f"[{timestamp()}] ERROR: Cannot proceed without {package}")
        sys.exit(1)

# Import all required modules after installation
print(f"[{timestamp()}] Importing required modules...")
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import warnings
import imagehash
from tqdm import tqdm
warnings.filterwarnings('ignore')

class KnowledgeBank:
    """Persistent knowledge storage for learned classifications"""
    
    def __init__(self, filepath="knowledge_bank.pkl"):
        self.filepath = filepath
        self.features_db = {}  # hash -> features
        self.classifications_db = {}  # hash -> classifications
        self.characteristics_db = {}  # hash -> characteristics dict
        self.relationships = defaultdict(list)  # classification -> [hashes]
        self.classification_weights = defaultdict(lambda: defaultdict(float))
        self.user_feedback = defaultdict(list)  # hash -> [(correct_class, wrong_class)]
        self.characteristic_patterns = defaultdict(Counter)  # characteristic -> Counter of values
        self.feature_dimensions = None  # Track feature dimensions for consistency
        self.folder_structure = defaultdict(set)  # Track known folder structures
        self.custom_keywords = set()  # User-defined keywords
        self.classification_history = defaultdict(list)  # Track classification history
        self.load()
        print(f"[{timestamp()}] Knowledge bank initialized at {filepath}")
    
    def add_image(self, image_hash, features, classifications, characteristics):
        """Add image to knowledge bank"""
        print(f"[{timestamp()}] Adding image to knowledge bank: {image_hash[:8]}...")
        
        # Ensure features are numpy array
        features = np.array(features, dtype=np.float32)
        
        # Check feature dimensions
        if self.feature_dimensions is None:
            self.feature_dimensions = features.shape
        elif features.shape != self.feature_dimensions:
            print(f"[{timestamp()}] WARNING: Feature dimensions mismatch. Expected {self.feature_dimensions}, got {features.shape}")
            # Resize features to match expected dimensions
            if len(features) < self.feature_dimensions[0]:
                # Pad with zeros
                features = np.pad(features, (0, self.feature_dimensions[0] - len(features)), 'constant')
            else:
                # Truncate
                features = features[:self.feature_dimensions[0]]
        
        self.features_db[image_hash] = features
        self.classifications_db[image_hash] = classifications
        self.characteristics_db[image_hash] = characteristics
        
        for classification in classifications:
            if image_hash not in self.relationships[classification]:
                self.relationships[classification].append(image_hash)
        
        # Learn characteristic patterns
        for char_type, char_value in characteristics.items():
            if char_value:  # Only track non-empty values
                self.characteristic_patterns[char_type][str(char_value)] += 1
        
        print(f"[{timestamp()}] Image added with {len(classifications)} classifications")
    
    def add_folder_structure(self, path_components):
        """Learn folder structure patterns"""
        if len(path_components) > 1:
            parent = path_components[0]
            for child in path_components[1:]:
                self.folder_structure[parent].add(child)
                parent = child
    
    def add_custom_keyword(self, keyword):
        """Add user-defined keyword"""
        self.custom_keywords.add(keyword.lower())
        print(f"[{timestamp()}] Added custom keyword: {keyword}")
    
    def update_weights(self, classification, feature_importance):
        """Update classification weights based on learning"""
        print(f"[{timestamp()}] Updating weights for classification: {classification}")
        for idx, importance in enumerate(feature_importance):
            self.classification_weights[classification][idx] = float(importance)
    
    def add_feedback(self, image_hash, correct_class, wrong_class=None):
        """Add user feedback for continuous learning"""
        print(f"[{timestamp()}] Recording user feedback for image {image_hash[:8]}...")
        self.user_feedback[image_hash].append((correct_class, wrong_class))
        self.classification_history[image_hash].append({
            'timestamp': timestamp(),
            'correct': correct_class,
            'wrong': wrong_class
        })
    
    def get_classification_confidence(self, classification):
        """Get confidence score for a classification based on history"""
        total_count = 0
        correct_count = 0
        
        for feedbacks in self.user_feedback.values():
            for correct, wrong in feedbacks:
                if correct == classification:
                    correct_count += 1
                    total_count += 1
                elif wrong == classification:
                    total_count += 1
        
        if total_count == 0:
            return 1.0  # No feedback yet, assume good
        
        return correct_count / total_count
    
    def save(self):
        """Save knowledge bank to disk"""
        print(f"[{timestamp()}] Saving knowledge bank to {self.filepath}...")
        data = {
            'features_db': self.features_db,
            'classifications_db': self.classifications_db,
            'characteristics_db': self.characteristics_db,
            'relationships': dict(self.relationships),
            'classification_weights': dict(self.classification_weights),
            'user_feedback': dict(self.user_feedback),
            'characteristic_patterns': dict(self.characteristic_patterns),
            'feature_dimensions': self.feature_dimensions,
            'folder_structure': dict(self.folder_structure),
            'custom_keywords': self.custom_keywords,
            'classification_history': dict(self.classification_history)
        }
        with open(self.filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[{timestamp()}] Knowledge bank saved successfully")
    
    def load(self):
        """Load knowledge bank from disk"""
        if os.path.exists(self.filepath):
            print(f"[{timestamp()}] Loading existing knowledge bank from {self.filepath}...")
            try:
                with open(self.filepath, 'rb') as f:
                    data = pickle.load(f)
                self.features_db = data.get('features_db', {})
                self.classifications_db = data.get('classifications_db', {})
                self.characteristics_db = data.get('characteristics_db', {})
                self.relationships = defaultdict(list, data.get('relationships', {}))
                self.classification_weights = defaultdict(lambda: defaultdict(float), 
                                                        data.get('classification_weights', {}))
                self.user_feedback = defaultdict(list, data.get('user_feedback', {}))
                self.characteristic_patterns = defaultdict(Counter, data.get('characteristic_patterns', {}))
                self.feature_dimensions = data.get('feature_dimensions', None)
                self.folder_structure = defaultdict(set, data.get('folder_structure', {}))
                self.custom_keywords = data.get('custom_keywords', set())
                self.classification_history = defaultdict(list, data.get('classification_history', {}))
                print(f"[{timestamp()}] Knowledge bank loaded: {len(self.features_db)} images")
            except Exception as e:
                print(f"[{timestamp()}] Error loading knowledge bank: {e}")
                print(f"[{timestamp()}] Starting with empty knowledge bank")
        else:
            print(f"[{timestamp()}] No existing knowledge bank found, starting fresh")

class AdvancedImageClassifier:
    def __init__(self, config_file="classifier_config.json"):
        self.config_file = config_file
        self.load_config()
        
        print(f"[{timestamp()}] Initializing Advanced Image Classifier")
        
        # Initialize knowledge bank
        self.knowledge_bank = KnowledgeBank()
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Classification components pattern
        self.classification_pattern = re.compile(
            r'(\d+)?-?([a-zA-Z]+)?-?([a-zA-Z]+)?-?([a-zA-Z\-]+)?-?([a-zA-Z\-]+)?'
        )
        
        # Initialize similarity tracking
        self.feature_cache = {}  # Cache extracted features
        
        # Connector type patterns
        self.connector_patterns = {
            'fc': ['fc', 'fiber connector'],
            'sma': ['sma', 'sub-miniature-a'],
            'sc': ['sc', 'subscriber connector'],
            'lc': ['lc', 'lucent connector'],
            'st': ['st', 'straight tip'],
            '50': ['50'],
            '91': ['91']
        }
        
        # Defect type patterns
        self.defect_patterns = {
            'scratched': ['scratch', 'scratched', 'scrape'],
            'oil': ['oil', 'oily', 'grease'],
            'wet': ['wet', 'water', 'moisture'],
            'blob': ['blob', 'spot', 'stain'],
            'dig': ['dig', 'dent', 'gouge'],
            'anomaly': ['anomaly', 'abnormal', 'defect']
        }
        
        print(f"[{timestamp()}] Classifier initialized successfully")
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "reference_paths": {},
            "dataset_path": "",
            "similarity_threshold": 0.70,
            "min_confidence": 0.50,
            "classification_components": [
                "core_diameter",
                "connector_type",
                "region",
                "condition",
                "defect_type",
                "additional_characteristics"
            ],
            "custom_keywords": [],
            "auto_create_folders": True,
            "save_learned_references": True,
            "use_adaptive_threshold": True,
            "feature_extraction_method": "combined",
            "max_features_per_method": 256
        }
        
        if os.path.exists(self.config_file):
            print(f"[{timestamp()}] Loading config from {self.config_file}")
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        else:
            print(f"[{timestamp()}] Creating default config file")
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        self.config = default_config
        
        # Add custom keywords to knowledge bank
        for keyword in self.config.get('custom_keywords', []):
            self.knowledge_bank.add_custom_keyword(keyword)
        
        print(f"[{timestamp()}] Config loaded successfully")
    
    def save_config(self):
        """Save configuration to file"""
        print(f"[{timestamp()}] Saving config to {self.config_file}")
        # Update custom keywords from knowledge bank
        self.config['custom_keywords'] = list(self.knowledge_bank.custom_keywords)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def extract_visual_features(self, image_path):
        """Extract comprehensive visual features from image"""
        try:
            print(f"[{timestamp()}] Extracting visual features from: {os.path.basename(image_path)}")
            
            # Check cache first
            cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
            if cache_key in self.feature_cache:
                print(f"[{timestamp()}] Using cached features")
                return self.feature_cache[cache_key]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get perceptual hash
            img_hash = str(imagehash.phash(image))
            
            # Extract different types of features
            features_list = []
            
            # 1. Color histogram features
            color_hist = self._extract_color_histogram(image)
            features_list.append(color_hist)
            
            # 2. Texture features
            texture_features = self._extract_texture_features(image)
            features_list.append(texture_features)
            
            # 3. Edge features
            edge_features = self._extract_edge_features(image)
            features_list.append(edge_features)
            
            # 4. Shape features
            shape_features = self._extract_shape_features(image)
            features_list.append(shape_features)
            
            # 5. Statistical features
            stat_features = self._extract_statistical_features(image)
            features_list.append(stat_features)
            
            # Combine all features with normalization
            visual_features = self._combine_features(features_list)
            
            # Cache the result
            self.feature_cache[cache_key] = (visual_features, img_hash)
            
            # Limit cache size
            if len(self.feature_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self.feature_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.feature_cache[key]
            
            return visual_features, img_hash
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR extracting visual features from {image_path}: {str(e)}")
            logging.error(f"Feature extraction error: {str(e)}", exc_info=True)
            return None, None
    
    def _extract_color_histogram(self, image):
        """Extract color histogram features"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate histograms for each channel
        hist_features = []
        for i in range(3):  # RGB channels
            hist, _ = np.histogram(img_array[:,:,i], bins=32, range=(0, 256))
            hist = hist / (hist.sum() + 1e-8)  # Normalize
            hist_features.extend(hist)
        
        # Also extract HSV histogram for better color representation
        hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        for i in range(3):  # HSV channels
            if i == 0:  # Hue channel
                hist, _ = np.histogram(hsv_image[:,:,i], bins=18, range=(0, 180))
            else:  # Saturation and Value channels
                hist, _ = np.histogram(hsv_image[:,:,i], bins=16, range=(0, 256))
            hist = hist / (hist.sum() + 1e-8)
            hist_features.extend(hist)
        
        return np.array(hist_features, dtype=np.float32)
    
    def _extract_texture_features(self, image):
        """Extract texture features using multiple methods"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. Gabor filters
        for theta in np.arange(0, np.pi, np.pi/4):
            for sigma in [1, 3]:
                for frequency in [0.05, 0.25]:
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, frequency, 0.5, 0)
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    features.extend([filtered.mean(), filtered.var()])
        
        # 2. Local Binary Patterns (simplified)
        radius = 1
        n_points = 8 * radius
        lbp_hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
        features.extend(lbp_hist)
        
        # 3. Gray Level Co-occurrence Matrix features
        # Simplified version - just use mean and variance of different regions
        h, w = gray.shape
        regions = [
            gray[:h//2, :w//2],  # Top-left
            gray[:h//2, w//2:],  # Top-right
            gray[h//2:, :w//2],  # Bottom-left
            gray[h//2:, w//2:]   # Bottom-right
        ]
        for region in regions:
            features.extend([region.mean(), region.var()])
        
        return np.array(features[:self.config['max_features_per_method']], dtype=np.float32)
    
    def _extract_edge_features(self, image):
        """Extract edge features using Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Apply Canny edge detection with auto thresholds
        median_val = np.median(blurred)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(blurred, lower, upper)
        
        # Calculate edge statistics
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Calculate edge distribution
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)
        
        # Edge orientation histogram
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Create orientation histogram
        orient_hist, _ = np.histogram(orientation[magnitude > 20], bins=16, range=(-np.pi, np.pi))
        orient_hist = orient_hist / (orient_hist.sum() + 1e-8)
        
        features = [
            edge_density,
            np.mean(h_projection),
            np.std(h_projection),
            np.mean(v_projection),
            np.std(v_projection),
            np.max(h_projection),
            np.max(v_projection),
            np.median(magnitude),
            np.std(magnitude)
        ]
        features.extend(orient_hist)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_shape_features(self, image):
        """Extract shape-based features"""
        # Convert to grayscale and binary
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
            
            # Bounding box features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / (h + 1e-8)
            extent = area / (w * h + 1e-8)
            
            # Moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            features = [
                area / (gray.shape[0] * gray.shape[1]),  # Normalized area
                perimeter / (2 * (gray.shape[0] + gray.shape[1])),  # Normalized perimeter
                circularity,
                aspect_ratio,
                extent
            ]
            features.extend(hu_moments[:4])  # First 4 Hu moments
        else:
            # Default features if no contours found
            features = [0] * 9
        
        # Add number of contours as a feature
        features.append(min(len(contours), 10) / 10.0)  # Normalized
        
        return np.array(features, dtype=np.float32)
    
    def _extract_statistical_features(self, image):
        """Extract statistical features from the image"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Global statistics
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
            gray.min() / 255.0,
            gray.max() / 255.0
        ])
        
        # Channel statistics
        for i in range(3):  # RGB channels
            channel = img_array[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0
            ])
        
        # Gradient statistics
        gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gradx**2 + grady**2)
        
        features.extend([
            grad_mag.mean() / 255.0,
            grad_mag.std() / 255.0,
            np.percentile(grad_mag, 90) / 255.0
        ])
        
        # Entropy (simplified)
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        features.append(entropy / 5.0)  # Normalized entropy
        
        return np.array(features, dtype=np.float32)
    
    def _combine_features(self, features_list):
        """Combine and normalize features from different extractors"""
        # Concatenate all features
        all_features = np.concatenate(features_list)
        
        # Apply feature scaling
        # Use robust scaling to handle outliers
        median = np.median(all_features)
        mad = np.median(np.abs(all_features - median))
        
        if mad > 0:
            scaled_features = (all_features - median) / (1.4826 * mad)
            # Clip extreme values
            scaled_features = np.clip(scaled_features, -3, 3)
        else:
            scaled_features = all_features
        
        return scaled_features
    
    def parse_classification(self, filename):
        """Parse classification components from filename with improved logic"""
        print(f"[{timestamp()}] Parsing classification from: {filename}")
        
        # Remove extension and clean filename
        base_name = Path(filename).stem
        base_name = re.sub(r'_\d+$', '', base_name)  # Remove trailing numbers
        
        # Extract classification components
        components = {}
        parts = re.split(r'[-_]', base_name)
        
        # Try to identify each component
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            part_lower = part.lower()
            
            # Check if it's a number (likely core diameter)
            if part.isdigit() and len(part) <= 3:
                if 'core_diameter' not in components:
                    components['core_diameter'] = part
            
            # Check for connector type
            connector_found = False
            for conn_type, patterns in self.connector_patterns.items():
                if any(p in part_lower for p in patterns):
                    components['connector_type'] = conn_type
                    connector_found = True
                    break
            
            if connector_found:
                continue
            
            # Check for condition keywords
            if part_lower in ['clean', 'dirty']:
                components['condition'] = part_lower
            # Check for region keywords
            elif part_lower in ['core', 'cladding', 'ferrule']:
                components['region'] = part_lower
            # Check for defect types
            else:
                for defect_type, patterns in self.defect_patterns.items():
                    if any(p in part_lower for p in patterns):
                        if 'defect_type' not in components:
                            components['defect_type'] = []
                        if defect_type not in components['defect_type']:
                            components['defect_type'].append(defect_type)
                        break
                else:
                    # Check custom keywords
                    if part_lower in self.knowledge_bank.custom_keywords:
                        if 'additional_characteristics' not in components:
                            components['additional_characteristics'] = []
                        components['additional_characteristics'].append(part_lower)
                    # Other characteristics
                    elif len(part_lower) > 2:  # Ignore very short parts
                        if 'additional_characteristics' not in components:
                            components['additional_characteristics'] = []
                        components['additional_characteristics'].append(part_lower)
        
        # Join list components
        for key in ['defect_type', 'additional_characteristics']:
            if key in components and isinstance(components[key], list):
                components[key] = '-'.join(sorted(set(components[key])))
        
        print(f"[{timestamp()}] Parsed components: {components}")
        return components
    
    def build_classification_string(self, components):
        """Build classification string from components"""
        parts = []
        
        # Order based on config
        for comp_type in self.config['classification_components']:
            if comp_type in components and components[comp_type]:
                value = str(components[comp_type])
                if value and value not in parts:  # Avoid duplicates
                    parts.append(value)
        
        classification = '-'.join(parts)
        print(f"[{timestamp()}] Built classification: {classification}")
        return classification
    
    def analyze_reference_folder(self, reference_folder):
        """Analyze hierarchical reference folder structure and images"""
        print(f"[{timestamp()}] Analyzing reference folder: {reference_folder}")
        
        reference_data = {}
        total_images = 0
        
        # Walk through entire directory structure
        for root, dirs, files in os.walk(reference_folder):
            # Get relative path from reference folder
            rel_path = os.path.relpath(root, reference_folder)
            if rel_path == '.':
                rel_path = ''
            
            # Count images in this directory
            image_files = [f for f in files if self.is_image_file(f)]
            if image_files:
                print(f"[{timestamp()}] Processing {len(image_files)} images in: {rel_path or 'root'}")
            
            # Process images in current directory
            for file in image_files:
                image_path = os.path.join(root, file)
                
                try:
                    # Extract all features
                    visual_features, img_hash = self.extract_visual_features(image_path)
                    
                    if visual_features is not None:
                        total_images += 1
                        
                        # Parse classification from filename and path
                        components = self.parse_classification(file)
                        
                        # Add path information to components
                        path_parts = rel_path.split(os.sep) if rel_path else []
                        if path_parts:
                            # Learn folder structure
                            self.knowledge_bank.add_folder_structure(path_parts)
                            
                            # Extract information from path hierarchy
                            for i, part in enumerate(path_parts):
                                part_lower = part.lower()
                                
                                # Check for connector types
                                if part_lower in ['fc', 'sma', 'sc', 'lc', 'st']:
                                    components['connector_type'] = part_lower
                                elif part.isdigit() and len(part) <= 3:
                                    components['core_diameter'] = part
                                # Check for regions
                                elif part_lower in ['core', 'cladding', 'ferrule']:
                                    components['region'] = part_lower
                                # Check for conditions/defects
                                elif part_lower in ['clean', 'dirty']:
                                    components['condition'] = part_lower
                                else:
                                    # Check for defect patterns
                                    for defect_type, patterns in self.defect_patterns.items():
                                        if any(p in part_lower for p in patterns):
                                            if 'defect_type' not in components:
                                                components['defect_type'] = defect_type
                                            break
                        
                        # Build classification list from path and filename
                        classifications = []
                        
                        # Use full path as primary classification
                        if path_parts:
                            classifications.append(rel_path.replace(os.sep, '-'))
                        
                        # Add classification from components
                        component_classification = self.build_classification_string(components)
                        if component_classification:
                            classifications.append(component_classification)
                        
                        # Add filename-based classification
                        filename_classification = Path(file).stem
                        if filename_classification not in classifications:
                            classifications.append(filename_classification)
                        
                        # Add to knowledge bank
                        self.knowledge_bank.add_image(
                            img_hash,
                            visual_features,
                            classifications,
                            components
                        )
                        
                        # Store in reference data
                        key = rel_path if rel_path else 'root'
                        if key not in reference_data:
                            reference_data[key] = []
                        
                        reference_data[key].append({
                            'path': image_path,
                            'hash': img_hash,
                            'features': visual_features,
                            'components': components,
                            'classifications': classifications
                        })
                        
                except Exception as e:
                    print(f"[{timestamp()}] ERROR processing {image_path}: {str(e)}")
                    logging.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        
        print(f"[{timestamp()}] Analyzed {total_images} images across {len(reference_data)} folders")
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        return reference_data
    
    def find_similar_images(self, features, threshold=None):
        """Find similar images using cosine similarity with adaptive threshold"""
        if threshold is None:
            threshold = self.config.get('similarity_threshold', 0.70)
        
        if not self.knowledge_bank.features_db:
            print(f"[{timestamp()}] No images in knowledge bank")
            return []
        
        similarities = []
        
        # Convert query features to numpy array
        query_features = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Process in batches for efficiency
        batch_size = 100
        all_hashes = list(self.knowledge_bank.features_db.keys())
        
        for i in range(0, len(all_hashes), batch_size):
            batch_hashes = all_hashes[i:i+batch_size]
            batch_features = []
            
            for img_hash in batch_hashes:
                ref_features = self.knowledge_bank.features_db[img_hash]
                # Ensure same dimensions
                if len(ref_features) != query_features.shape[1]:
                    # Resize to match
                    if len(ref_features) < query_features.shape[1]:
                        ref_features = np.pad(ref_features, (0, query_features.shape[1] - len(ref_features)), 'constant')
                    else:
                        ref_features = ref_features[:query_features.shape[1]]
                batch_features.append(ref_features)
            
            if batch_features:
                batch_features = np.array(batch_features, dtype=np.float32)
                
                # Calculate cosine similarities
                batch_similarities = cosine_similarity(query_features, batch_features)[0]
                
                # Add results above threshold
                for j, similarity in enumerate(batch_similarities):
                    if similarity >= threshold:
                        img_hash = batch_hashes[j]
                        similarities.append({
                            'hash': img_hash,
                            'similarity': float(similarity),
                            'classifications': self.knowledge_bank.classifications_db.get(img_hash, []),
                            'characteristics': self.knowledge_bank.characteristics_db.get(img_hash, {})
                        })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Apply adaptive threshold if enabled
        if self.config.get('use_adaptive_threshold', True) and similarities:
            # If top match is very confident, be more selective
            if similarities[0]['similarity'] > 0.9:
                similarities = [s for s in similarities if s['similarity'] > 0.8]
            # If top match is less confident, be more inclusive
            elif similarities[0]['similarity'] < 0.8:
                min_threshold = max(threshold - 0.1, 0.5)
                similarities = [s for s in similarities if s['similarity'] > min_threshold]
        
        return similarities[:10]  # Return top 10 matches
    
    def classify_image(self, image_path, threshold=None):
        """Classify an image based on learned features with improved logic"""
        if threshold is None:
            threshold = self.config.get('similarity_threshold', 0.70)
        
        print(f"[{timestamp()}] Classifying image: {os.path.basename(image_path)}")
        
        # Extract features
        visual_features, img_hash = self.extract_visual_features(image_path)
        
        if visual_features is None:
            print(f"[{timestamp()}] Failed to extract features")
            return None, None, None
        
        # Find similar images
        similar_images = self.find_similar_images(visual_features, threshold=threshold)
        
        if not similar_images:
            print(f"[{timestamp()}] No similar images found (threshold: {threshold})")
            # Try with lower threshold
            if threshold > 0.5:
                similar_images = self.find_similar_images(visual_features, threshold=0.5)
                if not similar_images:
                    return None, None, None
            else:
                return None, None, None
        
        # Aggregate classifications with weighted voting
        classification_scores = defaultdict(float)
        component_scores = defaultdict(lambda: defaultdict(float))
        total_weight = 0
        
        for similar in similar_images:
            # Weight by similarity and classification confidence
            similarity = similar['similarity']
            
            # Apply confidence from knowledge bank history
            for classification in similar['classifications']:
                confidence_factor = self.knowledge_bank.get_classification_confidence(classification)
                weight = similarity * confidence_factor
                classification_scores[classification] += weight
                total_weight += weight
            
            # Weight components
            for comp_type, comp_value in similar['characteristics'].items():
                if comp_value:
                    component_scores[comp_type][str(comp_value)] += similarity
        
        # Get best classification
        if classification_scores and total_weight > 0:
            # Filter out very low scoring classifications
            min_score = max(classification_scores.values()) * 0.3
            valid_classifications = {k: v for k, v in classification_scores.items() if v >= min_score}
            
            if valid_classifications:
                best_classification = max(valid_classifications.items(), key=lambda x: x[1])
                
                # Get best components
                best_components = {}
                for comp_type, values in component_scores.items():
                    if values:
                        # Only include component if it appears in multiple matches
                        value_counts = {v: count for v, count in values.items() if count > similar_images[0]['similarity'] * 0.5}
                        if value_counts:
                            best_components[comp_type] = max(value_counts.items(), key=lambda x: x[1])[0]
                
                # Calculate confidence based on consistency and similarity
                avg_similarity = sum(s['similarity'] for s in similar_images[:3]) / min(3, len(similar_images))
                classification_consistency = best_classification[1] / total_weight
                confidence = avg_similarity * classification_consistency
                
                # Apply minimum confidence threshold
                if confidence < self.config.get('min_confidence', 0.50):
                    print(f"[{timestamp()}] Classification confidence too low: {confidence:.3f}")
                    return None, None, None
                
                print(f"[{timestamp()}] Classification: {best_classification[0]} (confidence: {confidence:.3f})")
                return best_classification[0], best_components, confidence
        
        return None, None, None
    
    def create_folder_if_needed(self, base_path, components):
        """Create folder structure based on classification components"""
        if not self.config.get('auto_create_folders', True):
            return base_path
        
        # Build folder path from components
        path_parts = []
        
        # Follow the hierarchical structure seen in reference folder
        # Priority order: connector_type -> core_diameter -> region -> condition/defect
        
        if 'connector_type' in components:
            path_parts.append(components['connector_type'])
        
        if 'core_diameter' in components:
            path_parts.append(components['core_diameter'])
        
        if 'region' in components:
            path_parts.append(components['region'])
        
        if 'condition' in components:
            path_parts.append(components['condition'])
        elif 'defect_type' in components:
            path_parts.append('dirty')
            path_parts.append(components['defect_type'])
        
        if path_parts:
            new_path = os.path.join(base_path, *path_parts)
            try:
                os.makedirs(new_path, exist_ok=True)
                print(f"[{timestamp()}] Created/verified folder structure: {new_path}")
                return new_path
            except Exception as e:
                print(f"[{timestamp()}] ERROR creating folder structure: {str(e)}")
                return base_path
        
        return base_path
    
    def save_to_reference(self, image_path, classification, components, reference_folder):
        """Save classified image to reference folder for future learning"""
        if not self.config.get('save_learned_references', True):
            return
        
        try:
            # Create appropriate folder structure
            target_folder = self.create_folder_if_needed(reference_folder, components)
            
            # Generate filename based on classification
            extension = Path(image_path).suffix
            base_classification = self.build_classification_string(components) or classification
            new_filename = f"{base_classification}{extension}"
            
            # Ensure unique filename
            new_filename = self.get_unique_filename(target_folder, Path(new_filename).stem, extension)
            target_path = os.path.join(target_folder, new_filename)
            
            # Copy image to reference folder
            shutil.copy2(image_path, target_path)
            print(f"[{timestamp()}] Saved reference image to: {target_path}")
            
            # Extract features and update knowledge bank
            visual_features, img_hash = self.extract_visual_features(target_path)
            if visual_features is not None:
                self.knowledge_bank.add_image(
                    img_hash,
                    visual_features,
                    [classification],
                    components
                )
            
        except Exception as e:
            print(f"[{timestamp()}] Error saving reference image: {str(e)}")
            logging.error(f"Error saving reference: {str(e)}", exc_info=True)
    
    def is_image_file(self, filepath):
        """Check if file is an image based on extension"""
        return Path(filepath).suffix.lower() in self.image_extensions
    
    def get_unique_filename(self, directory, base_name, extension):
        """Generate unique filename by appending numbers if necessary"""
        counter = 1
        new_filename = f"{base_name}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        while os.path.exists(new_path):
            new_filename = f"{base_name}_{counter}{extension}"
            new_path = os.path.join(directory, new_filename)
            counter += 1
        
        return new_filename
    
    def process_mode(self, reference_folder, dataset_folder):
        """Automated processing mode with improved error handling"""
        print(f"\n[{timestamp()}] === STARTING AUTOMATED PROCESS MODE ===")
        
        # Analyze reference folder
        print(f"[{timestamp()}] Step 1: Analyzing reference folder...")
        reference_data = self.analyze_reference_folder(reference_folder)
        
        if not reference_data:
            print(f"[{timestamp()}] ERROR: No reference data found")
            return
        
        # Process dataset folder
        print(f"[{timestamp()}] Step 2: Processing dataset folder...")
        stats = {
            'processed': 0,
            'classified': 0,
            'already_classified': 0,
            'failed': 0,
            'low_confidence': 0
        }
        
        # Collect all images
        all_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    all_images.append(os.path.join(root, file))
        
        print(f"[{timestamp()}] Found {len(all_images)} images to process")
        
        if not all_images:
            print(f"[{timestamp()}] No images found in dataset folder")
            return
        
        # Process each image with progress bar
        for image_path in tqdm(all_images, desc="Processing images"):
            try:
                print(f"\n[{timestamp()}] Processing: {os.path.basename(image_path)}")
                
                # Check if already has classification
                filename = os.path.basename(image_path)
                existing_components = self.parse_classification(filename)
                
                # Skip if already well-classified
                if len(existing_components) >= 3 and not filename.startswith('IMG_'):
                    print(f"[{timestamp()}] Already classified, skipping: {filename}")
                    stats['already_classified'] += 1
                    continue
                
                # Classify image
                classification, components, confidence = self.classify_image(image_path)
                stats['processed'] += 1
                
                if classification and confidence is not None:
                    if confidence < self.config.get('min_confidence', 0.50):
                        print(f"[{timestamp()}] Low confidence ({confidence:.3f}), skipping")
                        stats['low_confidence'] += 1
                        continue
                    
                    # Build new filename from components
                    if components:
                        new_classification = self.build_classification_string(components)
                    else:
                        new_classification = classification
                    
                    # Generate unique filename
                    directory = os.path.dirname(image_path)
                    extension = Path(image_path).suffix
                    new_filename = self.get_unique_filename(directory, new_classification, extension)
                    new_path = os.path.join(directory, new_filename)
                    
                    # Rename file
                    os.rename(image_path, new_path)
                    stats['classified'] += 1
                    print(f"[{timestamp()}] RENAMED: {filename} -> {new_filename} (confidence: {confidence:.3f})")
                    
                    # Save to reference if high confidence
                    if confidence > 0.85:
                        self.save_to_reference(new_path, classification, components, reference_folder)
                else:
                    print(f"[{timestamp()}] Could not classify: {filename}")
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"[{timestamp()}] ERROR processing {image_path}: {str(e)}")
                logging.error(f"Processing error: {str(e)}", exc_info=True)
                stats['failed'] += 1
        
        # Save knowledge bank
        self.knowledge_bank.save()
        self.save_config()
        
        # Summary
        print(f"\n[{timestamp()}] === PROCESS MODE COMPLETE ===")
        print(f"[{timestamp()}] Total images found: {len(all_images)}")
        print(f"[{timestamp()}] Images processed: {stats['processed']}")
        print(f"[{timestamp()}] Images classified and renamed: {stats['classified']}")
        print(f"[{timestamp()}] Already classified images: {stats['already_classified']}")
        print(f"[{timestamp()}] Low confidence skipped: {stats['low_confidence']}")
        print(f"[{timestamp()}] Failed to classify: {stats['failed']}")
        print(f"[{timestamp()}] Knowledge bank now contains: {len(self.knowledge_bank.features_db)} images")
        print(f"[{timestamp()}] Success rate: {stats['classified'] / (stats['processed'] + 0.001) * 100:.1f}%")
    
    def manual_mode(self, reference_folder, dataset_folder):
        """Interactive manual classification mode with console interface"""
        print(f"\n[{timestamp()}] === STARTING MANUAL MODE ===")
        
        # First analyze reference folder if not done
        print(f"[{timestamp()}] Loading reference data...")
        reference_data = self.analyze_reference_folder(reference_folder)
        
        # Collect dataset images
        dataset_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    image_path = os.path.join(root, file)
                    dataset_images.append(image_path)
        
        print(f"[{timestamp()}] Found {len(dataset_images)} images for manual classification")
        
        if not dataset_images:
            print(f"[{timestamp()}] No images found in dataset folder")
            return
        
        # Statistics
        stats = {
            'processed': 0,
            'classified': 0,
            'skipped': 0
        }
        
        # Process each image interactively
        for i, image_path in enumerate(dataset_images):
            print(f"\n{'='*60}")
            print(f"[{timestamp()}] Image {i+1}/{len(dataset_images)}: {os.path.basename(image_path)}")
            print(f"Path: {image_path}")
            
            # Try automatic classification first
            classification, components, confidence = self.classify_image(image_path)
            
            if classification and confidence is not None:
                print(f"\n[SUGGESTION] Classification: {classification} (confidence: {confidence:.3f})")
                if components:
                    print(f"Components detected:")
                    for comp_type, comp_value in components.items():
                        print(f"  - {comp_type}: {comp_value}")
            else:
                print(f"\n[INFO] No automatic classification available")
            
            # Get user input
            print("\nOptions:")
            print("1. Accept suggested classification")
            print("2. Enter manual classification")
            print("3. Enter components manually")
            print("4. Skip this image")
            print("5. Add custom keyword")
            print("6. Save and exit")
            
            while True:
                choice = input("\nYour choice (1-6): ").strip()
                
                if choice == '1' and classification:
                    # Accept suggestion
                    self._apply_classification(image_path, classification, components, reference_folder)
                    stats['classified'] += 1
                    break
                    
                elif choice == '2':
                    # Manual classification
                    manual_class = input("Enter complete classification (e.g., '50-fc-core-clean'): ").strip()
                    if manual_class:
                        manual_components = self.parse_classification(manual_class)
                        self._apply_classification(image_path, manual_class, manual_components, reference_folder)
                        stats['classified'] += 1
                        
                        # Add feedback to knowledge bank
                        visual_features, img_hash = self.extract_visual_features(image_path)
                        if visual_features is not None and classification:
                            self.knowledge_bank.add_feedback(img_hash, manual_class, classification)
                    break
                    
                elif choice == '3':
                    # Enter components manually
                    print("\nEnter components (press Enter to skip):")
                    manual_components = {}
                    
                    for comp_type in self.config['classification_components']:
                        value = input(f"  {comp_type}: ").strip()
                        if value:
                            manual_components[comp_type] = value
                    
                    if manual_components:
                        manual_class = self.build_classification_string(manual_components)
                        self._apply_classification(image_path, manual_class, manual_components, reference_folder)
                        stats['classified'] += 1
                    break
                    
                elif choice == '4':
                    # Skip
                    print(f"[{timestamp()}] Skipped")
                    stats['skipped'] += 1
                    break
                    
                elif choice == '5':
                    # Add custom keyword
                    keyword = input("Enter custom keyword: ").strip()
                    if keyword:
                        self.knowledge_bank.add_custom_keyword(keyword)
                        print(f"[{timestamp()}] Added custom keyword: {keyword}")
                    continue
                    
                elif choice == '6':
                    # Save and exit
                    print(f"[{timestamp()}] Saving and exiting...")
                    self.knowledge_bank.save()
                    self.save_config()
                    print(f"\n[{timestamp()}] Manual mode statistics:")
                    print(f"  - Processed: {i+1}")
                    print(f"  - Classified: {stats['classified']}")
                    print(f"  - Skipped: {stats['skipped']}")
                    return
                    
                else:
                    print("Invalid choice, please try again.")
                    continue
            
            stats['processed'] += 1
            
            # Show progress
            if stats['processed'] % 10 == 0:
                print(f"\n[PROGRESS] Processed: {stats['processed']}, Classified: {stats['classified']}, Skipped: {stats['skipped']}")
        
        # Save everything
        self.knowledge_bank.save()
        self.save_config()
        
        print(f"\n[{timestamp()}] === MANUAL MODE COMPLETE ===")
        print(f"[{timestamp()}] Total processed: {stats['processed']}")
        print(f"[{timestamp()}] Images classified: {stats['classified']}")
        print(f"[{timestamp()}] Images skipped: {stats['skipped']}")
        print(f"[{timestamp()}] Knowledge bank now contains: {len(self.knowledge_bank.features_db)} images")
    
    def _apply_classification(self, image_path, classification, components, reference_folder):
        """Apply classification to image"""
        try:
            # Generate new filename
            extension = Path(image_path).suffix
            new_filename = f"{classification}{extension}"
            
            # Create unique filename if needed
            directory = os.path.dirname(image_path)
            new_filename = self.get_unique_filename(directory, Path(new_filename).stem, extension)
            new_path = os.path.join(directory, new_filename)
            
            # Rename file
            os.rename(image_path, new_path)
            print(f"[{timestamp()}] RENAMED: {os.path.basename(image_path)} -> {new_filename}")
            
            # Save to reference for future learning
            self.save_to_reference(new_path, classification, components, reference_folder)
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR applying classification: {str(e)}")
            logging.error(f"Classification error: {str(e)}", exc_info=True)

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ADVANCED IMAGE CLASSIFIER AND RENAMER")
    print("Fully functional version with all features")
    print("="*80 + "\n")
    
    # Initialize classifier
    classifier = AdvancedImageClassifier()
    
    # Mode selection
    print("Select operation mode:")
    print("1. Process Mode (Automated)")
    print("2. Manual Mode (Interactive)")
    
    while True:
        mode_choice = input("\nEnter your choice (1 or 2): ").strip()
        if mode_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Path configuration
    print("\nCurrent configuration:")
    print(f"Reference folder: {os.path.abspath('reference')}")
    print(f"Dataset folder: {os.path.abspath('dataset')}")
    
    # Use default paths
    reference_folder = os.path.abspath('reference')
    dataset_folder = os.path.abspath('dataset')
    
    # Verify paths exist
    if not os.path.exists(reference_folder):
        print(f"[{timestamp()}] ERROR: Reference folder not found at {reference_folder}")
        print(f"[{timestamp()}] Please create the folder and add reference images")
        return
    
    if not os.path.exists(dataset_folder):
        print(f"[{timestamp()}] Creating dataset folder at {dataset_folder}")
        os.makedirs(dataset_folder)
    
    # Check if reference folder has images
    ref_images = []
    for root, dirs, files in os.walk(reference_folder):
        for file in files:
            if Path(file).suffix.lower() in classifier.image_extensions:
                ref_images.append(file)
    
    if not ref_images:
        print(f"[{timestamp()}] WARNING: No images found in reference folder")
        print(f"[{timestamp()}] Please add reference images before running")
        return
    
    print(f"[{timestamp()}] Found {len(ref_images)} reference images")
    
    # Update config
    classifier.config['reference_paths']['default'] = reference_folder
    classifier.config['dataset_path'] = dataset_folder
    classifier.save_config()
    
    # Execute chosen mode
    try:
        if mode_choice == '1':
            classifier.process_mode(reference_folder, dataset_folder)
        else:
            classifier.manual_mode(reference_folder, dataset_folder)
    except Exception as e:
        print(f"[{timestamp()}] ERROR during execution: {str(e)}")
        logging.error(f"Execution error: {str(e)}", exc_info=True)
    
    print(f"\n[{timestamp()}] All operations completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Operation cancelled by user")
    except Exception as e:
        print(f"\n[{timestamp()}] FATAL ERROR: {str(e)}")
        traceback.print_exc()