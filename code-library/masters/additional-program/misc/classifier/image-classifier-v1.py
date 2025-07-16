#!/usr/bin/env python3
"""
Advanced Deep Learning Image Classifier and Renamer with Knowledge Bank
Analyzes, learns, and classifies images with continuous improvement capabilities
Enhanced for hierarchical folder structure and dynamic folder creation
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

print(f"[{timestamp()}] Starting Advanced Image Classifier and Renamer with ML")
print(f"[{timestamp()}] Checking and installing required dependencies...")

# Check and install required packages
required_packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("opencv-python", "cv2"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("faiss-cpu", "faiss"),
    ("imagehash", "imagehash")
]

for package, import_name in required_packages:
    if not install_package(package, import_name):
        print(f"[{timestamp()}] ERROR: Cannot proceed without {package}")
        sys.exit(1)

# Import all required modules after installation
print(f"[{timestamp()}] Importing required modules...")
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
import faiss
import imagehash
from scipy.spatial.distance import cdist
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
        self.load()
        print(f"[{timestamp()}] Knowledge bank initialized at {filepath}")
    
    def add_image(self, image_hash, features, classifications, characteristics):
        """Add image to knowledge bank"""
        print(f"[{timestamp()}] Adding image to knowledge bank: {image_hash[:8]}...")
        
        # Ensure features are numpy array
        features = np.array(features)
        
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
            self.characteristic_patterns[char_type][char_value] += 1
        
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
            self.classification_weights[classification][idx] = importance
    
    def add_feedback(self, image_hash, correct_class, wrong_class=None):
        """Add user feedback for continuous learning"""
        print(f"[{timestamp()}] Recording user feedback for image {image_hash[:8]}...")
        self.user_feedback[image_hash].append((correct_class, wrong_class))
    
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
            'custom_keywords': self.custom_keywords
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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{timestamp()}] Using device: {self.device}")
        
        # Initialize knowledge bank
        self.knowledge_bank = KnowledgeBank()
        
        # Initialize multiple pre-trained models for ensemble
        print(f"[{timestamp()}] Loading pre-trained models...")
        self.models = self._load_models()
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # For Inception
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Classification components pattern
        self.classification_pattern = re.compile(
            r'(\d+)?-?([a-zA-Z]+)?-?([a-zA-Z]+)?-?([a-zA-Z\-]+)?-?([a-zA-Z\-]+)?'
        )
        
        # Initialize FAISS index for fast similarity search
        self.faiss_index = None
        self.index_to_hash = {}
        
        # Connector type patterns
        self.connector_patterns = {
            'fc': ['fc', 'fiber connector'],
            'sma': ['sma', 'sub-miniature-a'],
            'sc': ['sc', 'subscriber connector'],
            'lc': ['lc', 'lucent connector'],
            'st': ['st', 'straight tip']
        }
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "reference_paths": {},
            "dataset_path": "",
            "similarity_threshold": 0.75,
            "ensemble_weights": {
                "resnet152": 0.3,
                "densenet201": 0.3,
                "inception_v3": 0.4
            },
            "classification_components": [
                "connector_type",
                "core_diameter",
                "region",
                "condition",
                "defect_type",
                "additional_characteristics"
            ],
            "custom_keywords": [],
            "auto_create_folders": True,
            "save_learned_references": True
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
    
    def _load_models(self):
        """Load multiple pre-trained models for ensemble"""
        models_dict = {}
        
        # ResNet152
        print(f"[{timestamp()}] Loading ResNet152...")
        resnet = models.resnet152(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.to(self.device)
        resnet.eval()
        models_dict['resnet152'] = resnet
        
        # DenseNet201
        print(f"[{timestamp()}] Loading DenseNet201...")
        densenet = models.densenet201(pretrained=True)
        densenet.classifier = nn.Identity()
        densenet.to(self.device)
        densenet.eval()
        models_dict['densenet201'] = densenet
        
        # Inception V3
        print(f"[{timestamp()}] Loading Inception V3...")
        inception = models.inception_v3(pretrained=True)
        inception.aux_logits = False
        inception.fc = nn.Identity()
        inception.to(self.device)
        inception.eval()
        models_dict['inception_v3'] = inception
        
        print(f"[{timestamp()}] All models loaded successfully")
        return models_dict
    
    def extract_visual_features(self, image_path):
        """Extract visual features from image"""
        try:
            print(f"[{timestamp()}] Extracting visual features from: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get perceptual hash
            img_hash = str(imagehash.phash(image))
            
            # Extract color histogram
            color_hist = self._extract_color_histogram(image)
            
            # Extract texture features
            texture_features = self._extract_texture_features(image)
            
            # Extract edge features
            edge_features = self._extract_edge_features(image)
            
            # Combine all features
            visual_features = np.concatenate([
                color_hist,
                texture_features,
                edge_features
            ])
            
            return visual_features, img_hash
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR extracting visual features from {image_path}: {str(e)}")
            return None, None
    
    def _extract_color_histogram(self, image):
        """Extract color histogram features"""
        print(f"[{timestamp()}] Extracting color histogram...")
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate histograms for each channel
        hist_features = []
        for i in range(3):  # RGB channels
            hist, _ = np.histogram(img_array[:,:,i], bins=64, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def _extract_texture_features(self, image):
        """Extract texture features using Gabor filters"""
        print(f"[{timestamp()}] Extracting texture features...")
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Gabor filters
        features = []
        for theta in np.arange(0, np.pi, np.pi/4):
            for sigma in [1, 3]:
                for frequency in [0.05, 0.25]:
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, frequency, 0.5, 0)
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    features.append(filtered.mean())
                    features.append(filtered.var())
        
        return np.array(features)
    
    def _extract_edge_features(self, image):
        """Extract edge features using Canny edge detection"""
        print(f"[{timestamp()}] Extracting edge features...")
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge statistics
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Calculate edge distribution
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)
        
        features = [
            edge_density,
            np.mean(h_projection),
            np.std(h_projection),
            np.mean(v_projection),
            np.std(v_projection)
        ]
        
        return np.array(features)
    
    def extract_deep_features(self, image_path):
        """Extract deep features using ensemble of models"""
        try:
            print(f"[{timestamp()}] Extracting deep features using ensemble...")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            all_features = []
            
            # Extract features from each model
            with torch.no_grad():
                for model_name, model in self.models.items():
                    print(f"[{timestamp()}] Extracting from {model_name}...")
                    features = model(image_tensor)
                    features = features.squeeze().cpu().numpy()
                    
                    # Apply model weight
                    weight = self.config['ensemble_weights'].get(model_name, 1.0)
                    features = features * weight
                    
                    all_features.append(features)
            
            # Combine features
            combined_features = np.concatenate(all_features)
            print(f"[{timestamp()}] Deep feature extraction complete, shape: {combined_features.shape}")
            
            return combined_features
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR extracting deep features from {image_path}: {str(e)}")
            return None
    
    def parse_classification(self, filename):
        """Parse classification components from filename"""
        print(f"[{timestamp()}] Parsing classification from: {filename}")
        
        # Remove extension and any trailing numbers
        base_name = Path(filename).stem
        
        # Extract classification components
        components = {}
        parts = base_name.split('-')
        
        # Try to identify each component
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            
            # Check for connector type
            part_lower = part.lower()
            for conn_type, patterns in self.connector_patterns.items():
                if any(p in part_lower for p in patterns):
                    components['connector_type'] = conn_type
                    break
            
            # Check if it's a number (likely core diameter)
            if part.isdigit():
                components['core_diameter'] = part
            # Check for condition keywords
            elif part_lower in ['clean', 'dirty']:
                components['condition'] = part_lower
            # Check for region keywords
            elif part_lower in ['core', 'cladding', 'ferrule']:
                components['region'] = part_lower
            # Check for defect types
            elif any(defect in part_lower for defect in ['scratched', 'oil', 'wet', 'blob', 'dig', 'anomaly']):
                if 'defect_type' not in components:
                    components['defect_type'] = []
                components['defect_type'].append(part_lower)
            # Check custom keywords
            elif part_lower in self.knowledge_bank.custom_keywords:
                if 'additional_characteristics' not in components:
                    components['additional_characteristics'] = []
                components['additional_characteristics'].append(part_lower)
            # Other characteristics
            else:
                if 'additional_characteristics' not in components:
                    components['additional_characteristics'] = []
                components['additional_characteristics'].append(part_lower)
        
        # Join list components
        for key in ['defect_type', 'additional_characteristics']:
            if key in components and isinstance(components[key], list):
                components[key] = '-'.join(components[key])
        
        print(f"[{timestamp()}] Parsed components: {components}")
        return components
    
    def build_classification_string(self, components):
        """Build classification string from components"""
        parts = []
        
        # Order based on config
        for comp_type in self.config['classification_components']:
            if comp_type in components and components[comp_type]:
                parts.append(str(components[comp_type]))
        
        classification = '-'.join(parts)
        print(f"[{timestamp()}] Built classification: {classification}")
        return classification
    
    def analyze_reference_folder(self, reference_folder):
        """Analyze hierarchical reference folder structure and images"""
        print(f"[{timestamp()}] Analyzing reference folder: {reference_folder}")
        
        reference_data = {}
        
        # Walk through entire directory structure
        for root, dirs, files in os.walk(reference_folder):
            # Get relative path from reference folder
            rel_path = os.path.relpath(root, reference_folder)
            if rel_path == '.':
                rel_path = ''
            
            # Process images in current directory
            for file in files:
                if self.is_image_file(file):
                    image_path = os.path.join(root, file)
                    
                    # Extract all features
                    visual_features, img_hash = self.extract_visual_features(image_path)
                    deep_features = self.extract_deep_features(image_path)
                    
                    if visual_features is not None and deep_features is not None:
                        # Combine all features
                        all_features = np.concatenate([visual_features, deep_features])
                        
                        # Parse classification from filename and path
                        components = self.parse_classification(file)
                        
                        # Add path information to components
                        path_parts = rel_path.split(os.sep) if rel_path else []
                        if path_parts:
                            # Learn folder structure
                            self.knowledge_bank.add_folder_structure(path_parts)
                            
                            # Extract connector type and other info from path
                            for i, part in enumerate(path_parts):
                                part_lower = part.lower()
                                # Check for connector types
                                if part_lower in ['fc', 'sma', 'sc', 'lc', 'st']:
                                    components['connector_type'] = part_lower
                                # Check for core diameter
                                elif part.isdigit():
                                    components['core_diameter'] = part
                                # Check for regions
                                elif part_lower in ['core', 'cladding', 'ferrule']:
                                    components['region'] = part_lower
                                # Check for conditions/defects
                                elif part_lower in ['clean', 'dirty', 'scratched', 'oil', 'blob', 'dig', 'anomaly']:
                                    if part_lower in ['clean', 'dirty']:
                                        components['condition'] = part_lower
                                    else:
                                        if 'defect_type' not in components:
                                            components['defect_type'] = part_lower
                        
                        # Build classification list from path
                        classifications = []
                        if path_parts:
                            # Use the deepest folder as primary classification
                            classifications.append(path_parts[-1])
                            # Also add full path as classification
                            classifications.append(rel_path.replace(os.sep, '-'))
                        
                        # Add to knowledge bank
                        self.knowledge_bank.add_image(
                            img_hash,
                            all_features,
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
                            'features': all_features,
                            'components': components,
                            'classifications': classifications
                        })
        
        print(f"[{timestamp()}] Analyzed {sum(len(v) for v in reference_data.values())} images across {len(reference_data)} folders")
        
        # Build FAISS index for fast search
        self.build_faiss_index()
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        return reference_data
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        print(f"[{timestamp()}] Building FAISS index for fast similarity search...")
        
        if not self.knowledge_bank.features_db:
            print(f"[{timestamp()}] No features to index")
            return
        
        # Collect all features
        features_list = []
        hash_list = []
        
        # Get expected dimensions from first feature
        expected_dim = None
        for img_hash, features in self.knowledge_bank.features_db.items():
            if expected_dim is None:
                expected_dim = len(features)
            
            # Ensure all features have same dimensions
            if len(features) != expected_dim:
                print(f"[{timestamp()}] WARNING: Feature dimension mismatch for {img_hash}")
                if len(features) < expected_dim:
                    # Pad with zeros
                    features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                else:
                    # Truncate
                    features = features[:expected_dim]
            
            features_list.append(features)
            hash_list.append(img_hash)
        
        if not features_list:
            print(f"[{timestamp()}] No valid features to index")
            return
        
        # Convert to numpy array
        features_array = np.array(features_list, dtype=np.float32)
        
        # Normalize features
        faiss.normalize_L2(features_array)
        
        # Create index
        d = features_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        self.faiss_index.add(features_array)
        
        # Create mapping
        self.index_to_hash = {i: h for i, h in enumerate(hash_list)}
        
        print(f"[{timestamp()}] FAISS index built with {len(features_list)} images, dimension: {d}")
    
    def find_similar_images(self, features, k=5, threshold=0.7):
        """Find similar images using FAISS"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print(f"[{timestamp()}] No images in index")
            return []
        
        # Ensure correct dimensions
        expected_dim = self.faiss_index.d
        if len(features) != expected_dim:
            print(f"[{timestamp()}] Adjusting feature dimensions from {len(features)} to {expected_dim}")
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)), 'constant')
            else:
                features = features[:expected_dim]
        
        # Normalize query features
        features = np.array([features], dtype=np.float32)
        faiss.normalize_L2(features)
        
        # Search
        k = min(k, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(features, k)
        
        # Filter by threshold and get results
        results = []
        for i in range(k):
            if similarities[0][i] >= threshold:
                img_hash = self.index_to_hash[indices[0][i]]
                classifications = self.knowledge_bank.classifications_db.get(img_hash, [])
                results.append({
                    'hash': img_hash,
                    'similarity': float(similarities[0][i]),
                    'classifications': classifications
                })
        
        return results
    
    def classify_image(self, image_path, threshold=None):
        """Classify an image based on learned features"""
        if threshold is None:
            threshold = self.config['similarity_threshold']
        
        print(f"[{timestamp()}] Classifying image: {image_path}")
        
        # Extract features
        visual_features, img_hash = self.extract_visual_features(image_path)
        deep_features = self.extract_deep_features(image_path)
        
        if visual_features is None or deep_features is None:
            print(f"[{timestamp()}] Failed to extract features")
            return None, 0.0, {}
        
        # Combine features
        all_features = np.concatenate([visual_features, deep_features])
        
        # Find similar images
        similar_images = self.find_similar_images(all_features, k=10, threshold=threshold)
        
        if not similar_images:
            print(f"[{timestamp()}] No similar images found above threshold")
            return None, 0.0, {}
        
        # Aggregate classifications
        classification_scores = defaultdict(float)
        component_scores = defaultdict(lambda: defaultdict(float))
        
        for similar in similar_images:
            weight = similar['similarity']
            for classification in similar['classifications']:
                classification_scores[classification] += weight
            
            # Get components for this image
            similar_hash = similar['hash']
            if similar_hash in self.knowledge_bank.characteristics_db:
                # Aggregate component information
                characteristics = self.knowledge_bank.characteristics_db.get(similar_hash, {})
                for comp_type, comp_value in characteristics.items():
                    if comp_value:  # Only add if value exists
                        component_scores[comp_type][comp_value] += weight
        
        # Get best classification
        if classification_scores:
            best_classification = max(classification_scores.items(), key=lambda x: x[1])
            classification_name = best_classification[0]
            confidence = best_classification[1] / len(similar_images)
            
            # Get best components
            best_components = {}
            for comp_type, values in component_scores.items():
                if values:
                    best_value = max(values.items(), key=lambda x: x[1])
                    best_components[comp_type] = best_value[0]
            
            print(f"[{timestamp()}] Classification: {classification_name} (confidence: {confidence:.3f})")
            return classification_name, confidence, best_components
        
        return None, 0.0, {}
    
    def create_folder_if_needed(self, base_path, components):
        """Create folder structure based on classification components"""
        if not self.config.get('auto_create_folders', True):
            return base_path
        
        # Build folder path from components
        path_parts = []
        
        # Add connector type if present
        if 'connector_type' in components:
            path_parts.append(components['connector_type'])
        
        # Add core diameter if present
        if 'core_diameter' in components:
            path_parts.append(components['core_diameter'])
        
        # Add region if present
        if 'region' in components:
            path_parts.append(components['region'])
        
        # Add condition/defect if present
        if 'condition' in components:
            path_parts.append(components['condition'])
        elif 'defect_type' in components:
            path_parts.append(components['defect_type'])
        
        if path_parts:
            new_path = os.path.join(base_path, *path_parts)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print(f"[{timestamp()}] Created new folder structure: {new_path}")
            return new_path
        
        return base_path
    
    def save_to_reference(self, image_path, classification, components, reference_folder):
        """Save classified image to reference folder for future learning"""
        if not self.config.get('save_learned_references', True):
            return
        
        try:
            # Create appropriate folder structure
            target_folder = self.create_folder_if_needed(reference_folder, components)
            
            # Copy image to reference folder
            filename = os.path.basename(image_path)
            target_path = os.path.join(target_folder, filename)
            
            # Ensure unique filename
            if os.path.exists(target_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    filename = f"{base}_{counter}{ext}"
                    target_path = os.path.join(target_folder, filename)
                    counter += 1
            
            shutil.copy2(image_path, target_path)
            print(f"[{timestamp()}] Saved reference image to: {target_path}")
            
        except Exception as e:
            print(f"[{timestamp()}] Error saving reference image: {str(e)}")
    
    def is_image_file(self, filepath):
        """Check if file is an image based on extension"""
        return Path(filepath).suffix.lower() in self.image_extensions
    
    def process_mode(self, reference_folder, dataset_folder):
        """Automated processing mode"""
        print(f"\n[{timestamp()}] === STARTING PROCESS MODE ===")
        
        # Analyze reference folder
        print(f"[{timestamp()}] Step 1: Analyzing reference folder...")
        reference_data = self.analyze_reference_folder(reference_folder)
        
        if not reference_data:
            print(f"[{timestamp()}] ERROR: No reference data found")
            return
        
        # Process dataset folder
        print(f"[{timestamp()}] Step 2: Processing dataset folder...")
        processed_count = 0
        classified_count = 0
        already_classified = 0
        failed_count = 0
        
        # Collect all images
        all_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    all_images.append(os.path.join(root, file))
        
        print(f"[{timestamp()}] Found {len(all_images)} images to process")
        
        # Process each image
        for image_path in tqdm(all_images, desc="Processing images"):
            print(f"[{timestamp()}] Processing: {image_path}")
            
            # Check if already has classification
            filename = os.path.basename(image_path)
            if '-' in filename and not filename.startswith('IMG_'):
                print(f"[{timestamp()}] Image appears to be already classified, analyzing anyway...")
                already_classified += 1
            
            # Classify image
            classification, confidence, components = self.classify_image(image_path)
            processed_count += 1
            
            if classification:
                # Build new filename
                new_classification = self.build_classification_string(components) if components else classification
                
                # Generate unique filename
                directory = os.path.dirname(image_path)
                extension = Path(image_path).suffix
                new_filename = self.get_unique_filename(directory, new_classification, extension)
                new_path = os.path.join(directory, new_filename)
                
                # Rename file
                try:
                    os.rename(image_path, new_path)
                    classified_count += 1
                    print(f"[{timestamp()}] RENAMED: {filename} -> {new_filename} (confidence: {confidence:.3f})")
                    
                    # Add to knowledge bank
                    visual_features, img_hash = self.extract_visual_features(new_path)
                    deep_features = self.extract_deep_features(new_path)
                    if visual_features is not None and deep_features is not None:
                        all_features = np.concatenate([visual_features, deep_features])
                        self.knowledge_bank.add_image(
                            img_hash,
                            all_features,
                            [classification],
                            components
                        )
                    
                    # Save to reference if confidence is high
                    if confidence > 0.85:
                        self.save_to_reference(new_path, classification, components, reference_folder)
                        
                except Exception as e:
                    print(f"[{timestamp()}] ERROR renaming {image_path}: {str(e)}")
                    failed_count += 1
            else:
                print(f"[{timestamp()}] Could not classify: {filename}")
                failed_count += 1
        
        # Rebuild index and save
        self.build_faiss_index()
        self.knowledge_bank.save()
        self.save_config()
        
        # Summary
        print(f"\n[{timestamp()}] === PROCESS MODE COMPLETE ===")
        print(f"[{timestamp()}] Total images processed: {processed_count}")
        print(f"[{timestamp()}] Images classified and renamed: {classified_count}")
        print(f"[{timestamp()}] Already classified images: {already_classified}")
        print(f"[{timestamp()}] Failed to classify: {failed_count}")
        print(f"[{timestamp()}] Knowledge bank now contains: {len(self.knowledge_bank.features_db)} images")
    
    def manual_mode(self, reference_folder, dataset_folder):
        """Interactive manual classification mode"""
        print(f"\n[{timestamp()}] === STARTING MANUAL MODE ===")
        
        # First analyze reference folder if not done
        print(f"[{timestamp()}] Loading reference data...")
        reference_data = self.analyze_reference_folder(reference_folder)
        
        # Create GUI
        root = tk.Tk()
        root.title("Manual Image Classification")
        root.geometry("1400x900")
        
        # Variables
        current_image_idx = 0
        dataset_images = []
        
        # Collect dataset images
        for root_dir, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    dataset_images.append(os.path.join(root_dir, file))
        
        print(f"[{timestamp()}] Found {len(dataset_images)} images for manual classification")
        
        if not dataset_images:
            messagebox.showwarning("No Images", "No images found in dataset folder")
            root.destroy()
            return
        
        # GUI Components
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        image_label = ttk.Label(main_frame)
        image_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Info display
        info_text = tk.Text(main_frame, height=10, width=60)
        info_text.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Classification suggestion
        suggestion_var = tk.StringVar()
        suggestion_label = ttk.Label(main_frame, textvariable=suggestion_var)
        suggestion_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Manual classification entry
        ttk.Label(main_frame, text="Manual Classification:").grid(row=3, column=0, sticky=tk.W)
        manual_entry = ttk.Entry(main_frame, width=50)
        manual_entry.grid(row=3, column=1, pady=5)
        
        # Component entries
        component_frame = ttk.LabelFrame(main_frame, text="Classification Components", padding="10")
        component_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        component_vars = {}
        for i, comp in enumerate(self.config['classification_components']):
            ttk.Label(component_frame, text=f"{comp}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            ttk.Entry(component_frame, textvariable=var, width=30).grid(row=i, column=1, pady=2)
            component_vars[comp] = var
        
        # Custom keyword entry
        keyword_frame = ttk.LabelFrame(main_frame, text="Add Custom Keywords", padding="10")
        keyword_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(keyword_frame, text="New Keyword:").grid(row=0, column=0, sticky=tk.W)
        keyword_entry = ttk.Entry(keyword_frame, width=30)
        keyword_entry.grid(row=0, column=1, padx=5)
        
        def add_keyword():
            keyword = keyword_entry.get().strip()
            if keyword:
                self.knowledge_bank.add_custom_keyword(keyword)
                keyword_entry.delete(0, tk.END)
                messagebox.showinfo("Success", f"Added keyword: {keyword}")
        
        ttk.Button(keyword_frame, text="Add Keyword", command=add_keyword).grid(row=0, column=2, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        def load_image(idx):
            """Load and display current image"""
            if 0 <= idx < len(dataset_images):
                image_path = dataset_images[idx]
                print(f"[{timestamp()}] Loading image {idx+1}/{len(dataset_images)}: {image_path}")
                
                # Display image
                img = Image.open(image_path)
                img.thumbnail((600, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                image_label.configure(image=photo)
                image_label.image = photo
                
                # Get classification suggestion
                classification, confidence, components = self.classify_image(image_path)
                
                # Update info
                info_text.delete(1.0, tk.END)
                info_text.insert(tk.END, f"File: {os.path.basename(image_path)}\n")
                info_text.insert(tk.END, f"Path: {image_path}\n")
                info_text.insert(tk.END, f"Image {idx+1} of {len(dataset_images)}\n\n")
                
                if classification:
                    suggestion_var.set(f"Suggested: {classification} (confidence: {confidence:.3f})")
                    info_text.insert(tk.END, f"Suggested classification: {classification}\n")
                    info_text.insert(tk.END, f"Confidence: {confidence:.3f}\n")
                    info_text.insert(tk.END, f"Components: {components}\n")
                    
                    # Fill component fields
                    for comp, var in component_vars.items():
                        var.set(components.get(comp, ''))
                else:
                    suggestion_var.set("No suggestion available")
                    info_text.insert(tk.END, "No classification suggestion available\n")
        
        def save_classification():
            """Save current classification and move to next image"""
            nonlocal current_image_idx
            
            if current_image_idx >= len(dataset_images):
                return
            
            image_path = dataset_images[current_image_idx]
            
            # Get classification from manual entry or build from components
            manual_class = manual_entry.get().strip()
            
            # Get components from fields
            components = {}
            for comp, var in component_vars.items():
                value = var.get().strip()
                if value:
                    components[comp] = value
            
            if not manual_class and components:
                # Build from components
                manual_class = self.build_classification_string(components)
            
            if manual_class:
                # Create folder structure if needed
                directory = self.create_folder_if_needed(os.path.dirname(image_path), components)
                
                # Generate unique filename
                extension = Path(image_path).suffix
                new_filename = self.get_unique_filename(directory, manual_class, extension)
                new_path = os.path.join(directory, new_filename)
                
                try:
                    # Move file to new location
                    shutil.move(image_path, new_path)
                    print(f"[{timestamp()}] RENAMED: {os.path.basename(image_path)} -> {new_filename}")
                    
                    # Extract features and add to knowledge bank
                    visual_features, img_hash = self.extract_visual_features(new_path)
                    deep_features = self.extract_deep_features(new_path)
                    
                    if visual_features is not None and deep_features is not None:
                        all_features = np.concatenate([visual_features, deep_features])
                        
                        # Add to knowledge bank
                        self.knowledge_bank.add_image(
                            img_hash,
                            all_features,
                            [manual_class],
                            components
                        )
                        
                        # Add feedback if there was a suggestion
                        classification, _, _ = self.classify_image(image_path)
                        if classification and classification != manual_class:
                            self.knowledge_bank.add_feedback(img_hash, manual_class, classification)
                        
                        # Save to reference folder
                        self.save_to_reference(new_path, manual_class, components, reference_folder)
                    
                    # Update the image path in the list
                    dataset_images[current_image_idx] = new_path
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to rename: {str(e)}")
                    print(f"[{timestamp()}] ERROR renaming: {str(e)}")
            
            # Move to next image
            next_image()
        
        def next_image():
            """Move to next image"""
            nonlocal current_image_idx
            if current_image_idx < len(dataset_images) - 1:
                current_image_idx += 1
                load_image(current_image_idx)
                manual_entry.delete(0, tk.END)
            else:
                messagebox.showinfo("Complete", "All images have been processed!")
                save_and_exit()
        
        def prev_image():
            """Move to previous image"""
            nonlocal current_image_idx
            if current_image_idx > 0:
                current_image_idx -= 1
                load_image(current_image_idx)
                manual_entry.delete(0, tk.END)
        
        def skip_image():
            """Skip current image without classifying"""
            next_image()
        
        def save_and_exit():
            """Save knowledge bank and exit"""
            print(f"[{timestamp()}] Saving knowledge bank and rebuilding index...")
            self.build_faiss_index()
            self.knowledge_bank.save()
            self.save_config()
            print(f"[{timestamp()}] Manual mode complete")
            root.destroy()
        
        # Add buttons
        ttk.Button(button_frame, text="Previous", command=prev_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Skip", command=skip_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save & Next", command=save_classification).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Save & Exit", command=save_and_exit).grid(row=0, column=3, padx=5)
        
        # Load first image
        load_image(0)
        
        # Start GUI
        root.mainloop()
    
    def get_unique_filename(self, directory, base_name, extension):
        """Generate unique filename by appending numbers if necessary"""
        counter = 1
        new_filename = f"{base_name}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        while os.path.exists(new_path):
            new_filename = f"{base_name}-{counter}{extension}"
            new_path = os.path.join(directory, new_filename)
            counter += 1
        
        return new_filename

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("ADVANCED IMAGE CLASSIFIER AND RENAMER WITH MACHINE LEARNING")
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
        return
    
    if not os.path.exists(dataset_folder):
        print(f"[{timestamp()}] Creating dataset folder at {dataset_folder}")
        os.makedirs(dataset_folder)
    
    # Update config
    classifier.config['reference_paths']['default'] = reference_folder
    classifier.config['dataset_path'] = dataset_folder
    classifier.save_config()
    
    # Execute chosen mode
    if mode_choice == '1':
        classifier.process_mode(reference_folder, dataset_folder)
    else:
        classifier.manual_mode(reference_folder, dataset_folder)
    
    print(f"\n[{timestamp()}] All operations completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Operation cancelled by user")
    except Exception as e:
        print(f"\n[{timestamp()}] FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()