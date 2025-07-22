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
import traceback

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
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("opencv-python", "cv2"),
    ("imagehash", "imagehash")
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
import cv2
import warnings
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
        
        print(f"[{timestamp()}] Using CPU processing (simplified version)")
        
        # Initialize knowledge bank first
        self.knowledge_bank = KnowledgeBank()
        
        # Then load config
        self.load_config()
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Classification components pattern
        self.classification_pattern = re.compile(
            r'(\d+)?-?([a-zA-Z]+)?-?([a-zA-Z]+)?-?([a-zA-Z\-]+)?-?([a-zA-Z\-]+)?'
        )
        
        # Initialize similarity tracking
        self.feature_vectors = {}
        self.classification_index = {}
        
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
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "reference_paths": {},
            "dataset_path": "",
            "similarity_threshold": 0.75,
            "classification_components": [
                "core_diameter",
                "connector_type", 
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
            hist = hist / (hist.sum() + 1e-8)  # Normalize
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
                    kernel = cv2.getGaborKernel((21, 21), sigma, float(theta), frequency, 0.5, 0)
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
                    
                    if visual_features is not None:
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
                                if part_lower in ['fc', 'sma', 'sc', 'lc', 'st', '50', '91']:
                                    if part.isdigit():
                                        components['core_diameter'] = part
                                    else:
                                        components['connector_type'] = part_lower
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
                        
                        # Build classification from filename
                        filename_classification = self.build_classification_string(components)
                        if filename_classification:
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
        
        print(f"[{timestamp()}] Analyzed {sum(len(v) for v in reference_data.values())} images across {len(reference_data)} folders")
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        return reference_data
    
    def find_similar_images(self, features, threshold=0.7):
        """Find similar images using cosine similarity"""
        if not self.knowledge_bank.features_db:
            print(f"[{timestamp()}] No images in knowledge bank")
            return []
        
        similarities = []
        
        for img_hash, ref_features in self.knowledge_bank.features_db.items():
            # Ensure same dimensions
            min_len = min(len(features), len(ref_features))
            feat1 = features[:min_len]
            feat2 = ref_features[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(feat1, feat2)
            norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                if similarity >= threshold:
                    similarities.append({
                        'hash': img_hash,
                        'similarity': similarity,
                        'classifications': self.knowledge_bank.classifications_db.get(img_hash, []),
                        'characteristics': self.knowledge_bank.characteristics_db.get(img_hash, {})
                    })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]  # Return top 5
    
    def classify_image(self, image_path, threshold=None):
        """Classify an image based on learned features"""
        if threshold is None:
            threshold = self.config.get('similarity_threshold', 0.7)
        
        print(f"[{timestamp()}] Classifying image: {image_path}")
        
        # Extract features
        visual_features, img_hash = self.extract_visual_features(image_path)
        
        if visual_features is None:
            print(f"[{timestamp()}] Failed to extract features")
            return None, None, None
        
        # Find similar images
        similar_images = self.find_similar_images(visual_features, threshold=threshold)
        
        if not similar_images:
            print(f"[{timestamp()}] No similar images found (threshold: {threshold})")
            return None, None, None
        
        # Aggregate classifications
        classification_scores = defaultdict(float)
        component_scores = defaultdict(lambda: defaultdict(float))
        
        for similar in similar_images:
            weight = similar['similarity']
            
            # Weight classifications
            for classification in similar['classifications']:
                classification_scores[classification] += weight
            
            # Weight components
            for comp_type, comp_value in similar['characteristics'].items():
                if comp_value:
                    component_scores[comp_type][comp_value] += weight
        
        # Get best classification
        if classification_scores:
            best_classification = max(classification_scores.items(), key=lambda x: x[1])
            
            # Get best components
            best_components = {}
            for comp_type, values in component_scores.items():
                if values:
                    best_components[comp_type] = max(values.items(), key=lambda x: x[1])[0]
            
            confidence = best_classification[1] / len(similar_images)
            
            print(f"[{timestamp()}] Classification: {best_classification[0]} (confidence: {confidence:.3f})")
            return best_classification[0], best_components, confidence
        
        return None, None, None
    
    def create_folder_if_needed(self, base_path, components):
        """Create folder structure based on classification components in reference folder"""
        if not self.config.get('auto_create_folders', True):
            return base_path
        
        # Build folder path from components
        path_parts = []
        
        # Add core diameter if present
        if 'core_diameter' in components:
            path_parts.append(components['core_diameter'])
        
        # Add connector type if present
        if 'connector_type' in components:
            path_parts.append(components['connector_type'])
        
        # Add condition if present
        if 'condition' in components:
            path_parts.append(components['condition'])
        elif 'defect_type' in components:
            path_parts.append('dirty')
            path_parts.append(components['defect_type'])
        
        if path_parts:
            new_folder = os.path.join(base_path, *path_parts)
            os.makedirs(new_folder, exist_ok=True)
            print(f"[{timestamp()}] Created folder structure: {new_folder}")
            return new_folder
        
        return base_path
    
    def save_to_reference(self, image_path, classification, components, reference_folder):
        """Save classified image to reference folder for future learning"""
        if not self.config.get('save_learned_references', True):
            return
        
        try:
            # Create appropriate folder structure in reference folder
            target_folder = self.create_folder_if_needed(reference_folder, components)
            
            # Generate new filename based on classification
            original_name = Path(image_path).stem
            extension = Path(image_path).suffix
            new_filename = f"{classification}{extension}"
            
            # Ensure unique filename
            new_filename = self.get_unique_filename(target_folder, Path(new_filename).stem, extension)
            target_path = os.path.join(target_folder, new_filename)
            
            # Copy file
            shutil.copy2(image_path, target_path)
            print(f"[{timestamp()}] Saved to reference: {target_path}")
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR saving to reference: {str(e)}")
    
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
        """Automated processing mode"""
        print(f"\n[{timestamp()}] === STARTING PROCESS MODE ===")
        
        # Analyze reference folder
        print(f"[{timestamp()}] Step 1: Analyzing reference folder...")
        reference_data = self.analyze_reference_folder(reference_folder)
        
        if not reference_data:
            print(f"[{timestamp()}] No reference data found. Cannot proceed.")
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
                    image_path = os.path.join(root, file)
                    all_images.append(image_path)
        
        print(f"[{timestamp()}] Found {len(all_images)} images to process")
        
        # Process each image
        for image_path in all_images:
            try:
                processed_count += 1
                print(f"\n[{timestamp()}] Processing {processed_count}/{len(all_images)}: {os.path.basename(image_path)}")
                
                # Check if already classified (has recognizable pattern)
                if self.parse_classification(os.path.basename(image_path)):
                    already_classified += 1
                    print(f"[{timestamp()}] Already classified, skipping")
                    continue
                
                # Classify image
                classification, components, confidence = self.classify_image(image_path)
                
                if classification and confidence is not None and confidence > 0.6:  # Minimum confidence threshold
                    # Generate new filename
                    extension = Path(image_path).suffix
                    new_filename = f"{classification}{extension}"
                    
                    # Create unique filename if needed
                    directory = os.path.dirname(image_path)
                    new_filename = self.get_unique_filename(directory, Path(new_filename).stem, extension)
                    new_path = os.path.join(directory, new_filename)
                    
                    # Rename file
                    os.rename(image_path, new_path)
                    print(f"[{timestamp()}] Renamed to: {new_filename}")
                    
                    # Save to reference for future learning
                    self.save_to_reference(new_path, classification, components, reference_folder)
                    
                    classified_count += 1
                else:
                    failed_count += 1
                    print(f"[{timestamp()}] Classification failed or low confidence")
                    
            except Exception as e:
                failed_count += 1
                print(f"[{timestamp()}] ERROR processing {image_path}: {str(e)}")
        
        # Save knowledge bank
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
        """Interactive manual classification mode (simplified)"""
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
        
        # Process each image interactively
        for i, image_path in enumerate(dataset_images):
            print(f"\n[{timestamp()}] Image {i+1}/{len(dataset_images)}: {os.path.basename(image_path)}")
            
            # Try automatic classification first
            classification, components, confidence = self.classify_image(image_path)
            
            if classification and confidence is not None and confidence > 0.5:
                print(f"[{timestamp()}] Suggested classification: {classification} (confidence: {confidence:.3f})")
                print(f"[{timestamp()}] Components: {components}")
            
            # Get user input
            print("\nOptions:")
            print("1. Accept suggested classification")
            print("2. Enter manual classification")
            print("3. Skip this image")
            print("4. Exit manual mode")
            
            choice = input("Your choice (1-4): ").strip()
            
            if choice == '1' and classification:
                # Accept suggestion
                self._apply_classification(image_path, classification, components, reference_folder)
            elif choice == '2':
                # Manual classification
                manual_class = input("Enter classification: ").strip()
                if manual_class:
                    manual_components = self.parse_classification(manual_class)
                    self._apply_classification(image_path, manual_class, manual_components, reference_folder)
            elif choice == '3':
                # Skip
                print(f"[{timestamp()}] Skipped")
                continue
            elif choice == '4':
                # Exit
                break
            else:
                print(f"[{timestamp()}] Invalid choice, skipping")
        
        # Save everything
        self.knowledge_bank.save()
        self.save_config()
        print(f"[{timestamp()}] Manual mode completed")
    
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
            print(f"[{timestamp()}] Renamed to: {new_filename}")
            
            # Save to reference for future learning
            self.save_to_reference(new_path, classification, components, reference_folder)
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR applying classification: {str(e)}")

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
        traceback.print_exc()
