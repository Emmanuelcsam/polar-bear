#!/usr/bin/env python3
"""
Ultimate Image Classifier - Comprehensive Version
Combines all features from previous versions with full functionality
Auto-installs dependencies, detailed logging, GUI support, and robust error handling
"""

import os
import sys
import subprocess
import importlib
import time
import json
import pickle
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import re
import logging
import traceback
import hashlib

# Setup dual logging (console + file) immediately
def setup_logging():
    """Setup logging to both console and file"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f"image_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return logger

# Initialize logging
logger = setup_logging()

def install_package(package_name, import_name=None):
    """Install a package using pip with the latest version"""
    if import_name is None:
        import_name = package_name
    
    logging.info(f"Checking package: {package_name}")
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        logging.info(f"✓ {package_name} already installed (version: {version})")
        return True
    except ImportError:
        logging.warning(f"Package {package_name} not found. Installing latest version...")
        try:
            # Force upgrade to latest version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "--no-cache-dir", "--break-system-packages", package_name
            ])
            logging.info(f"✓ Successfully installed {package_name} (latest version)")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"✗ Failed to install {package_name}: {e}")
            return False

# Core dependencies
print("="*80)
print("ULTIMATE IMAGE CLASSIFIER - COMPREHENSIVE VERSION")
print("="*80)
print("\nChecking and installing required dependencies...")

required_packages = [
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("opencv-python", "cv2"),
    ("imagehash", "imagehash"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib")
]

all_installed = True
for package, import_name in required_packages:
    if not install_package(package, import_name):
        all_installed = False

if not all_installed:
    logging.error("Some packages failed to install. Please check your environment.")
    sys.exit(1)

# Import all modules after installation
logging.info("Importing required modules...")
from PIL import Image, ImageTk
import cv2
import imagehash
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Try to import optional GUI module
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    GUI_AVAILABLE = True
    logging.info("✓ GUI support available (tkinter)")
except ImportError:
    GUI_AVAILABLE = False
    logging.warning("GUI support not available (tkinter missing)")

# Try to import sklearn components
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    logging.info("✓ Advanced ML features available (scikit-learn)")
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Advanced ML features not available (scikit-learn missing)")

class KnowledgeBank:
    """Enhanced persistent knowledge storage with versioning"""
    
    def __init__(self, filepath="knowledge_bank.pkl"):
        self.filepath = filepath
        self.version = "2.0"
        
        # Core databases
        self.features_db = {}  # hash -> features
        self.classifications_db = {}  # hash -> classifications
        self.characteristics_db = {}  # hash -> characteristics dict
        self.file_paths_db = {}  # hash -> original file paths
        
        # Learning data
        self.relationships = defaultdict(list)  # classification -> [hashes]
        self.classification_weights = defaultdict(lambda: defaultdict(float))
        self.user_feedback = defaultdict(list)  # hash -> [(correct_class, wrong_class)]
        self.characteristic_patterns = defaultdict(Counter)  # characteristic -> Counter of values
        
        # Metadata
        self.feature_dimensions = None
        self.folder_structure = defaultdict(set)
        self.custom_keywords = set()
        self.classification_history = defaultdict(list)
        self.learning_stats = {
            'total_images': 0,
            'total_classifications': 0,
            'user_corrections': 0,
            'last_updated': None
        }
        
        self.load()
        logging.info(f"Knowledge bank initialized at {filepath}")
    
    def add_custom_keyword(self, keyword):
        """Add a custom keyword to the knowledge bank"""
        self.custom_keywords.add(keyword)
        logging.debug(f"Added custom keyword: {keyword}")

    def add_folder_structure(self, path_parts):
        """Add folder structure to the knowledge bank"""
        self.folder_structure['/'.join(path_parts)].add(tuple(path_parts))
        logging.debug(f"Added folder structure: {path_parts}")
    
    def add_image(self, image_hash, features, classifications, characteristics, file_path=None):
        """Add image to knowledge bank with comprehensive tracking"""
        logging.debug(f"Adding image to knowledge bank: {image_hash[:8]}...")
        
        # Ensure features are numpy array with correct type
        features = np.array(features, dtype=np.float32)
        
        # Handle feature dimensions
        if self.feature_dimensions is None:
            self.feature_dimensions = features.shape
            logging.debug(f"Set feature dimensions to: {self.feature_dimensions}")
        elif features.shape != self.feature_dimensions:
            logging.warning(f"Feature dimension mismatch. Expected {self.feature_dimensions}, got {features.shape}")
            # Resize features
            if len(features) < self.feature_dimensions[0]:
                features = np.pad(features, (0, self.feature_dimensions[0] - len(features)), 'constant')
            else:
                features = features[:self.feature_dimensions[0]]
        
        # Store data
        self.features_db[image_hash] = features
        self.classifications_db[image_hash] = classifications
        self.characteristics_db[image_hash] = characteristics
        if file_path:
            self.file_paths_db[image_hash] = file_path
        
        # Update relationships
        for classification in classifications:
            if image_hash not in self.relationships[classification]:
                self.relationships[classification].append(image_hash)
        
        # Learn patterns
        for char_type, char_value in characteristics.items():
            if char_value:
                self.characteristic_patterns[char_type][str(char_value)] += 1
        
        # Update stats
        self.learning_stats['total_images'] = len(self.features_db)
        self.learning_stats['total_classifications'] = len(self.relationships)
        self.learning_stats['last_updated'] = datetime.now().isoformat()
        
        logging.info(f"Image added successfully. Total images: {len(self.features_db)}")
    
    def add_feedback(self, image_hash, correct_class, wrong_class=None):
        """Add user feedback for continuous learning"""
        self.user_feedback[image_hash].append((correct_class, wrong_class))
        self.classification_history[image_hash].append({
            'timestamp': datetime.now().isoformat(),
            'correct': correct_class,
            'wrong': wrong_class
        })
        self.learning_stats['user_corrections'] += 1
        logging.info(f"User feedback recorded. Total corrections: {self.learning_stats['user_corrections']}")
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        stats = {
            **self.learning_stats,
            'classifications': dict(Counter(
                cls for clss in self.classifications_db.values() for cls in clss
            ).most_common(10)),
            'characteristics': {
                char_type: dict(counter.most_common(5))
                for char_type, counter in self.characteristic_patterns.items()
            }
        }
        return stats
    
    def save(self):
        """Save knowledge bank with error handling"""
        logging.info("Saving knowledge bank...")
        try:
            # Create backup first
            if os.path.exists(self.filepath):
                backup_path = f"{self.filepath}.backup"
                shutil.copy2(self.filepath, backup_path)
                logging.debug(f"Created backup at {backup_path}")
            
            # Save data
            data = {
                'version': self.version,
                'features_db': self.features_db,
                'classifications_db': self.classifications_db,
                'characteristics_db': self.characteristics_db,
                'file_paths_db': self.file_paths_db,
                'relationships': dict(self.relationships),
                'classification_weights': dict(self.classification_weights),
                'user_feedback': dict(self.user_feedback),
                'characteristic_patterns': dict(self.characteristic_patterns),
                'feature_dimensions': self.feature_dimensions,
                'folder_structure': dict(self.folder_structure),
                'custom_keywords': self.custom_keywords,
                'classification_history': dict(self.classification_history),
                'learning_stats': self.learning_stats
            }
            
            with open(self.filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logging.info(f"Knowledge bank saved successfully ({len(self.features_db)} images)")
            
        except Exception as e:
            logging.error(f"Failed to save knowledge bank: {e}")
            # Restore backup if save failed
            if os.path.exists(f"{self.filepath}.backup"):
                shutil.copy2(f"{self.filepath}.backup", self.filepath)
                logging.info("Restored from backup")
    
    def load(self):
        """Load knowledge bank with migration support"""
        if not os.path.exists(self.filepath):
            logging.info("No existing knowledge bank found, starting fresh")
            return
        
        logging.info("Loading knowledge bank...")
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check version and migrate if needed
            file_version = data.get('version', '1.0')
            if file_version != self.version:
                logging.info(f"Migrating knowledge bank from version {file_version} to {self.version}")
            
            # Load data with defaults for missing fields
            self.features_db = data.get('features_db', {})
            self.classifications_db = data.get('classifications_db', {})
            self.characteristics_db = data.get('characteristics_db', {})
            self.file_paths_db = data.get('file_paths_db', {})
            self.relationships = defaultdict(list, data.get('relationships', {}))
            self.classification_weights = defaultdict(
                lambda: defaultdict(float), 
                data.get('classification_weights', {})
            )
            self.user_feedback = defaultdict(list, data.get('user_feedback', {}))
            self.characteristic_patterns = defaultdict(
                Counter, 
                data.get('characteristic_patterns', {})
            )
            self.feature_dimensions = data.get('feature_dimensions')
            self.folder_structure = defaultdict(set, data.get('folder_structure', {}))
            self.custom_keywords = data.get('custom_keywords', set())
            self.classification_history = defaultdict(
                list, 
                data.get('classification_history', {})
            )
            self.learning_stats = data.get('learning_stats', self.learning_stats)
            
            logging.info(f"Knowledge bank loaded: {len(self.features_db)} images, "
                        f"{len(self.relationships)} classifications")
            
        except Exception as e:
            logging.error(f"Error loading knowledge bank: {e}")
            logging.info("Starting with empty knowledge bank")

class ImageClassifierGUI:
    """Enhanced GUI for manual classification with image preview"""
    
    def __init__(self, classifier, dataset_images, reference_folder):
        self.classifier = classifier
        self.dataset_images = dataset_images
        self.reference_folder = reference_folder
        self.current_index = 0
        self.processed_count = 0
        self.skipped_count = 0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Ultimate Image Classifier - Manual Mode")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create UI
        self.create_widgets()
        
        # Load first image
        self.load_image()
        
        logging.info("GUI initialized for manual classification")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Image display
        left_panel = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Canvas for image
        self.canvas = tk.Canvas(left_panel, width=600, height=400, bg='gray20')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Right panel - Controls
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File info
        info_frame = ttk.LabelFrame(right_panel, text="File Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(info_frame, text="", font=('Arial', 10, 'bold'))
        self.file_label.pack(anchor=tk.W)
        
        self.path_label = ttk.Label(info_frame, text="", font=('Arial', 9))
        self.path_label.pack(anchor=tk.W)
        
        self.progress_label = ttk.Label(info_frame, text="", font=('Arial', 9))
        self.progress_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Classification suggestion
        suggest_frame = ttk.LabelFrame(right_panel, text="Automatic Suggestion", padding="10")
        suggest_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.suggestion_text = tk.Text(suggest_frame, height=4, width=50)
        self.suggestion_text.pack(fill=tk.X)
        
        # Manual classification
        manual_frame = ttk.LabelFrame(right_panel, text="Manual Classification", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(manual_frame, text="Enter classification:").pack(anchor=tk.W)
        self.manual_entry = ttk.Entry(manual_frame, width=50)
        self.manual_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Component inputs
        comp_frame = ttk.LabelFrame(right_panel, text="Classification Components", padding="10")
        comp_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.component_vars = {}
        components = [
            ("Core Diameter", "core_diameter"),
            ("Connector Type", "connector_type"),
            ("Region", "region"),
            ("Condition", "condition"),
            ("Defect Type", "defect_type"),
            ("Additional", "additional_characteristics")
        ]
        
        for label, key in components:
            frame = ttk.Frame(comp_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label}:", width=15).pack(side=tk.LEFT)
            var = tk.StringVar()
            ttk.Entry(frame, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.component_vars[key] = var
        
        # Action buttons
        button_frame = ttk.Frame(right_panel)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="← Previous", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Skip", 
                  command=self.skip_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Accept Suggestion", 
                  command=self.accept_suggestion).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Apply Manual", 
                  command=self.apply_manual).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Build from Components", 
                  command=self.build_from_components).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save & Exit", 
                  command=self.save_and_exit).pack(side=tk.RIGHT, padx=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def load_image(self):
        """Load and display current image"""
        if self.current_index >= len(self.dataset_images):
            messagebox.showinfo("Complete", "All images processed!")
            self.save_and_exit()
            return
        
        image_path = self.dataset_images[self.current_index]
        logging.info(f"Loading image {self.current_index + 1}/{len(self.dataset_images)}: {image_path}")
        
        # Update file info
        self.file_label.config(text=f"File: {os.path.basename(image_path)}")
        self.path_label.config(text=f"Path: {os.path.dirname(image_path)}")
        self.progress_label.config(
            text=f"Progress: {self.current_index + 1} / {len(self.dataset_images)} "
                 f"(Processed: {self.processed_count}, Skipped: {self.skipped_count})"
        )
        
        # Load and display image
        try:
            # Open image
            img = Image.open(image_path)
            
            # Calculate display size (maintain aspect ratio)
            display_width = 600
            display_height = 400
            img_width, img_height = img.size
            
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize for display
            img_display = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img_display)
            
            # Clear canvas and display
            self.canvas.delete("all")
            x = (display_width - new_width) // 2
            y = (display_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
            # Update canvas size
            self.canvas.config(width=display_width, height=display_height)
            
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(300, 200, text="Error loading image", 
                                   fill="red", font=('Arial', 16))
        
        # Get classification suggestion
        self.get_suggestion()
        
        # Update status
        self.status_var.set("Ready")
    
    def get_suggestion(self):
        """Get automatic classification suggestion"""
        self.suggestion_text.delete(1.0, tk.END)
        self.suggestion_text.insert(tk.END, "Analyzing...\n")
        self.root.update()
        
        image_path = self.dataset_images[self.current_index]
        classification, components, confidence = self.classifier.classify_image(image_path)
        
        self.current_classification = classification
        self.current_components = components
        self.current_confidence = confidence
        
        # Update suggestion text
        self.suggestion_text.delete(1.0, tk.END)
        if classification:
            self.suggestion_text.insert(tk.END, 
                f"Classification: {classification}\n"
                f"Confidence: {confidence:.3f}\n"
                f"Components: {json.dumps(components, indent=2)}"
            )
            
            # Fill component fields
            for key, var in self.component_vars.items():
                var.set(components.get(key, ''))
        else:
            self.suggestion_text.insert(tk.END, "No suggestion available")
            self.current_classification = None
    
    def accept_suggestion(self):
        """Accept the automatic suggestion"""
        if self.current_classification:
            self.apply_classification(self.current_classification, self.current_components)
        else:
            messagebox.showwarning("No Suggestion", "No automatic suggestion available")
    
    def apply_manual(self):
        """Apply manual classification from entry"""
        manual_class = self.manual_entry.get().strip()
        if manual_class:
            components = self.classifier.parse_classification(manual_class)
            self.apply_classification(manual_class, components)
        else:
            messagebox.showwarning("Empty Classification", "Please enter a classification")
    
    def build_from_components(self):
        """Build classification from component fields"""
        components = {}
        for key, var in self.component_vars.items():
            value = var.get().strip()
            if value:
                components[key] = value
        
        if components:
            classification = self.classifier.build_classification_string(components)
            self.apply_classification(classification, components)
        else:
            messagebox.showwarning("No Components", "Please fill in at least one component")
    
    def apply_classification(self, classification, components):
        """Apply classification to current image"""
        image_path = self.dataset_images[self.current_index]
        
        try:
            # Apply classification
            success = self.classifier._apply_classification(
                image_path, classification, components, self.reference_folder, is_manual=True
            )
            
            if success:
                self.processed_count += 1
                self.status_var.set(f"Applied: {classification}")
                
                # Move to next image
                self.next_image()
            else:
                self.status_var.set("Failed to apply classification")
                
        except Exception as e:
            logging.error(f"Error applying classification: {e}")
            messagebox.showerror("Error", f"Failed to apply classification: {str(e)}")
    
    def skip_image(self):
        """Skip current image"""
        self.skipped_count += 1
        self.status_var.set("Skipped")
        self.next_image()
    
    def next_image(self):
        """Move to next image"""
        self.current_index += 1
        self.manual_entry.delete(0, tk.END)
        
        # Clear component fields
        for var in self.component_vars.values():
            var.set('')
        
        self.load_image()
    
    def prev_image(self):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.manual_entry.delete(0, tk.END)
            for var in self.component_vars.values():
                var.set('')
            self.load_image()
    
    def save_and_exit(self):
        """Save and close GUI"""
        logging.info(f"Manual mode complete. Processed: {self.processed_count}, "
                    f"Skipped: {self.skipped_count}")
        self.classifier.knowledge_bank.save()
        self.classifier.save_config()
        self.root.destroy()
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

class UltimateImageClassifier:
    """Ultimate image classifier with all features combined"""
    
    def __init__(self, config_file="classifier_config.json", 
                 similarity_threshold=0.65, auto_create_folders=True, custom_keywords=None):
        self.config_file = config_file
        self.knowledge_bank = KnowledgeBank()
        
        # Initialize config with defaults, which can be overridden by constructor args or loaded from file
        self.config = {
            "similarity_threshold": 0.65,
            "min_confidence": 0.50,
            "auto_mode_threshold": 0.70,
            "use_hsv_weight": 0.6,
            "use_rgb_weight": 0.4,
            "save_learned_references": True,
            "auto_create_folders": True,
            "max_cache_size": 1000,
            "classification_components": [
                "core_diameter",
                "connector_type",
                "region", 
                "condition",
                "defect_type",
                "additional_characteristics"
            ],
            "custom_keywords": []
        }

        # Load configuration from file first to get persistent settings
        self.load_config()

        # Override with constructor arguments if provided
        self.config["similarity_threshold"] = similarity_threshold
        self.config["auto_create_folders"] = auto_create_folders
        if custom_keywords is not None:
            self.config["custom_keywords"].extend(custom_keywords)
            self.config["custom_keywords"] = list(set(self.config["custom_keywords"])) # Remove duplicates
        
        # Core settings
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        self.reference_data = {}
        self.feature_cache = {}
        
        # Feature extraction settings
        self.image_size = (128, 128)
        self.color_bins = 32
        self.hsv_bins = 30
        
        # Classification patterns
        self.connector_patterns = {
            'fc': ['fc', 'fiber connector', 'fiber-connector'],
            'sma': ['sma', 'sub-miniature-a'],
            'sc': ['sc', 'subscriber connector'],
            'lc': ['lc', 'lucent connector'],
            'st': ['st', 'straight tip'],
            '50': ['50'],
            '91': ['91']
        }
        
        self.region_patterns = ['core', 'cladding', 'ferrule']
        self.condition_patterns = ['clean', 'dirty']
        self.defect_patterns = {
            'scratched': ['scratch', 'scratched', 'scrape'],
            'oil': ['oil', 'oily', 'grease'],
            'wet': ['wet', 'water', 'moisture'],
            'blob': ['blob', 'spot', 'stain'],
            'dig': ['dig', 'dent', 'gouge'],
            'anomaly': ['anomaly', 'abnormal', 'defect']
        }
        
        logging.info("Ultimate Image Classifier initialized")

        # Add custom keywords to knowledge bank after all config is set
        for keyword in self.config.get('custom_keywords', []):
            self.knowledge_bank.add_custom_keyword(keyword)
        
        self.save_config()
    
    def load_config(self):
        """Load configuration from file and merge with current config"""
        if os.path.exists(self.config_file):
            logging.info(f"Loading configuration from {self.config_file}")
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)
            except Exception as e:
                logging.error(f"Error loading config: {e}")
                logging.info("Using current configuration (possibly defaults)")
        else:
            logging.info("No configuration file found, using current settings")
        
        self.save_config()
        logging.info("Configuration loaded successfully")
    
    def interactive_config_setup(self, config):
        """This method is no longer used as configuration is handled via arguments."""
        logging.warning("interactive_config_setup called but is deprecated. Configuration is now handled via command-line arguments.")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.debug(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def extract_features(self, image_path):
        """Extract comprehensive visual features"""
        try:
            # Check cache first
            cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
            if cache_key in self.feature_cache:
                logging.debug(f"Using cached features for {os.path.basename(image_path)}")
                return self.feature_cache[cache_key]
            
            logging.debug(f"Extracting features from {os.path.basename(image_path)}")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent features
            img_resized = cv2.resize(img_rgb, self.image_size)
            
            # Get image hash
            pil_img = Image.fromarray(img_rgb)
            img_hash = str(imagehash.phash(pil_img))
            
            # Extract different feature types
            features = []
            
            # 1. Color features (RGB + HSV)
            color_features = self._extract_color_features(img_resized)
            features.extend(color_features)
            
            # 2. Texture features
            texture_features = self._extract_texture_features(img_resized)
            features.extend(texture_features)
            
            # 3. Edge features
            edge_features = self._extract_edge_features(img_resized)
            features.extend(edge_features)
            
            # 4. Shape features
            shape_features = self._extract_shape_features(img_resized)
            features.extend(shape_features)
            
            # 5. Statistical features
            stat_features = self._extract_statistical_features(img_resized)
            features.extend(stat_features)
            
            # 6. Dominant colors (if sklearn available)
            if SKLEARN_AVAILABLE:
                dominant_features = self._extract_dominant_colors(img_resized)
                features.extend(dominant_features)
            
            # Convert to numpy array
            features = np.array(features, dtype=np.float32)
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Cache result
            result = (features, img_hash)
            self.feature_cache[cache_key] = result
            
            # Manage cache size
            if len(self.feature_cache) > self.config.get('max_cache_size', 1000):
                # Remove oldest entries
                oldest_keys = list(self.feature_cache.keys())[:100]
                for key in oldest_keys:
                    del self.feature_cache[key]
            
            return result
            
        except Exception as e:
            logging.error(f"Feature extraction failed for {image_path}: {e}")
            return None, None
    
    def _extract_color_features(self, img):
        """Extract color histogram features"""
        features = []
        
        # RGB histograms
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [self.color_bins], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            features.extend(hist)
        
        # HSV histograms
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Hue (most important for color)
        hist_h = cv2.calcHist([img_hsv], [0], None, [self.hsv_bins], [0, 180])
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
        features.extend(hist_h * 2)  # Weight hue more
        
        # Saturation
        hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
        features.extend(hist_s)
        
        # Value
        hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256])
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
        features.extend(hist_v)
        
        # Color moments
        for i in range(3):
            channel = img[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0,
                cv2.moments(channel)['m00'] / (img.shape[0] * img.shape[1] * 255)
            ])
        
        return features
    
    def _extract_texture_features(self, img):
        """Extract texture features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = []
        
        # Gabor filters
        for theta in np.arange(0, np.pi, np.pi/4):
            for sigma in [1, 3]:
                for frequency in [0.05, 0.25]:
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, frequency, 0.5, 0)
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    features.extend([
                        filtered.mean(),
                        filtered.std(),
                        np.percentile(filtered, 25),
                        np.percentile(filtered, 75)
                    ])
        
        # Local Binary Pattern (simplified)
        # Calculate LBP-like features using local gradients
        gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        features.extend([
            gradx.mean(),
            gradx.std(),
            grady.mean(), 
            grady.std(),
            np.sqrt(gradx**2 + grady**2).mean()
        ])
        
        return features
    
    def _extract_edge_features(self, img):
        """Extract edge-based features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Canny edge detection with auto thresholds
        median_val = np.median(blurred)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        edges = cv2.Canny(blurred, lower, upper)
        
        # Edge statistics
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels
        
        # Edge distribution
        h_projection = np.sum(edges, axis=1)
        v_projection = np.sum(edges, axis=0)
        
        # Edge orientation
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        orientation = np.arctan2(sobely, sobelx)
        
        # Orientation histogram
        orient_hist, _ = np.histogram(
            orientation[magnitude > np.percentile(magnitude, 75)],
            bins=8, range=(-np.pi, np.pi)
        )
        orient_hist = orient_hist / (orient_hist.sum() + 1e-7)
        
        features = [
            edge_density,
            h_projection.mean(),
            h_projection.std(),
            v_projection.mean(),
            v_projection.std(),
            np.percentile(h_projection, 90),
            np.percentile(v_projection, 90),
            magnitude.mean(),
            magnitude.std()
        ]
        features.extend(orient_hist)
        
        return features
    
    def _extract_shape_features(self, img):
        """Extract shape-based features"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        
        if contours:
            # Analyze largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic shape properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape descriptors
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-7)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / (h + 1e-7)
            extent = area / (w * h + 1e-7)
            solidity = area / (cv2.contourArea(cv2.convexHull(largest_contour)) + 1e-7)
            
            # Moments
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            features = [
                area / (gray.shape[0] * gray.shape[1]),  # Normalized area
                perimeter / (2 * (gray.shape[0] + gray.shape[1])),  # Normalized perimeter
                circularity,
                aspect_ratio,
                extent,
                solidity,
                len(contours) / 100.0  # Normalized contour count
            ]
            features.extend(hu_moments[:5])  # First 5 Hu moments
        else:
            # Default features if no contours
            features = [0] * 12
        
        return features
    
    def _extract_statistical_features(self, img):
        """Extract statistical features"""
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Global statistics
        features.extend([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.median(gray) / 255.0,
            gray.min() / 255.0,
            gray.max() / 255.0,
            (gray.max() - gray.min()) / 255.0  # Range
        ])
        
        # Channel statistics
        for i in range(3):
            channel = img[:, :, i]
            features.extend([
                channel.mean() / 255.0,
                channel.std() / 255.0,
                np.percentile(channel, 25) / 255.0,
                np.percentile(channel, 75) / 255.0
            ])
        
        # Entropy
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features.append(entropy / 5.0)  # Normalized entropy
        
        return features
    
    def _extract_dominant_colors(self, img):
        """Extract dominant colors using K-means"""
        features = []
        
        try:
            # Reshape image to pixels
            pixels = img.reshape(-1, 3)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_ / 255.0
            
            # Sort by cluster size
            labels = kmeans.labels_
            cluster_sizes = [np.sum(labels == i) for i in range(5)]
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            
            # Add sorted dominant colors
            for idx in sorted_indices:
                features.extend(dominant_colors[idx])
                features.append(cluster_sizes[idx] / len(pixels))  # Cluster proportion
            
        except Exception as e:
            logging.debug(f"Dominant color extraction failed: {e}")
            # Default features
            features = [0] * 20
        
        return features
    
    def _normalize_features(self, features):
        """Normalize feature vector"""
        # Robust normalization
        median = np.median(features)
        mad = np.median(np.abs(features - median))
        
        if mad > 0:
            normalized = (features - median) / (1.4826 * mad)
            # Clip extreme values
            normalized = np.clip(normalized, -3, 3)
            # Scale to [0, 1]
            normalized = (normalized + 3) / 6
        else:
            normalized = features
        
        return normalized
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure same dimensions
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Multiple similarity metrics
        similarities = []
        
        # 1. Histogram intersection (good for histograms)
        hist_intersection = np.minimum(f1, f2).sum() / (np.maximum(f1.sum(), f2.sum()) + 1e-7)
        similarities.append(hist_intersection)
        
        # 2. Correlation coefficient
        if np.std(f1) > 0 and np.std(f2) > 0:
            correlation = np.corrcoef(f1, f2)[0, 1]
            if not np.isnan(correlation):
                similarities.append((correlation + 1.0) / 2.0)
        
        # 3. Cosine similarity
        dot_product = np.dot(f1, f2)
        norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)
        if norm_product > 0:
            cosine_sim = (dot_product / norm_product + 1.0) / 2.0
            similarities.append(cosine_sim)
        
        # 4. Euclidean distance (inverted and normalized)
        euclidean_dist = np.linalg.norm(f1 - f2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        similarities.append(euclidean_sim)
        
        # Weighted combination
        if similarities:
            weights = [0.3, 0.3, 0.2, 0.2][:len(similarities)]
            similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
        else:
            similarity = 0.0
        
        return float(similarity)
    
    def parse_classification(self, text):
        """Parse classification components from text"""
        logging.debug(f"Parsing classification: {text}")
        
        # Clean text
        text = Path(text).stem if '.' in text else text
        text = re.sub(r'_\d+$', '', text)  # Remove trailing numbers
        
        components = {}
        parts = re.split(r'[-_\s]+', text.lower())
        
        for part in parts:
            if not part:
                continue
            
            # Core diameter (numbers)
            if part.isdigit() and len(part) <= 3:
                if 'core_diameter' not in components:
                    components['core_diameter'] = part
                continue
            
            # Connector type
            matched = False
            for conn_type, patterns in self.connector_patterns.items():
                if any(pattern in part for pattern in patterns):
                    components['connector_type'] = conn_type
                    matched = True
                    break
            
            if matched:
                continue
            
            # Region
            if part in self.region_patterns:
                components['region'] = part
                continue
            
            # Condition
            if part in self.condition_patterns:
                components['condition'] = part
                continue
            
            # Defects
            for defect_type, patterns in self.defect_patterns.items():
                if any(pattern in part for pattern in patterns):
                    if 'defect_type' not in components:
                        components['defect_type'] = []
                    if defect_type not in components['defect_type']:
                        components['defect_type'].append(defect_type)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Custom keywords or additional characteristics
            if part in self.knowledge_bank.custom_keywords or len(part) > 2:
                if 'additional_characteristics' not in components:
                    components['additional_characteristics'] = []
                if part not in components['additional_characteristics']:
                    components['additional_characteristics'].append(part)
        
        # Convert lists to strings
        for key in ['defect_type', 'additional_characteristics']:
            if key in components and isinstance(components[key], list):
                components[key] = '-'.join(sorted(components[key]))
        
        logging.debug(f"Parsed components: {components}")
        return components
    
    def build_classification_string(self, components):
        """Build classification string from components"""
        parts = []
        
        # Use configured order
        for comp_type in self.config['classification_components']:
            if comp_type in components and components[comp_type]:
                value = str(components[comp_type])
                if value and value not in parts:
                    parts.append(value)
        
        classification = '-'.join(parts)
        logging.debug(f"Built classification: {classification}")
        return classification
    
    def analyze_reference_folder(self, reference_folder):
        """Analyze reference folder structure and images"""
        logging.info(f"Analyzing reference folder: {reference_folder}")
        
        self.reference_data = {}
        total_images = 0
        failed_images = 0
        
        # Progress tracking
        all_files = []
        for root, dirs, files in os.walk(reference_folder):
            for file in files:
                if self.is_image_file(file):
                    all_files.append((root, file))
        
        logging.info(f"Found {len(all_files)} images to analyze")
        
        # Process with progress bar
        for root, file in tqdm(all_files, desc="Analyzing reference images"):
            image_path = os.path.join(root, file)
            
            try:
                # Get relative path for classification
                rel_path = os.path.relpath(root, reference_folder)
                if rel_path == '.':
                    rel_path = ''
                
                # Extract features
                features, img_hash = self.extract_features(image_path)
                
                if features is not None:
                    total_images += 1
                    
                    # Parse classification from path and filename
                    components = self.parse_classification(file)
                    
                    # Add path information
                    path_parts = rel_path.split(os.sep) if rel_path else []
                    if path_parts:
                        # Learn folder structure
                        self.knowledge_bank.add_folder_structure(path_parts)
                        
                        # Extract info from path
                        for part in path_parts:
                            part_components = self.parse_classification(part)
                            # Merge with existing components (path takes precedence)
                            for key, value in part_components.items():
                                if key not in components or not components[key]:
                                    components[key] = value
                    
                    # Build classifications
                    classifications = []
                    
                    # Full path as classification
                    if rel_path:
                        classifications.append(rel_path.replace(os.sep, '-'))
                    
                    # Component-based classification
                    if components:
                        comp_classification = self.build_classification_string(components)
                        if comp_classification:
                            classifications.append(comp_classification)
                    
                    # Filename as classification
                    classifications.append(Path(file).stem)
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_classifications = []
                    for c in classifications:
                        if c not in seen:
                            seen.add(c)
                            unique_classifications.append(c)
                    
                    # Add to knowledge bank
                    self.knowledge_bank.add_image(
                        img_hash,
                        features,
                        unique_classifications,
                        components,
                        image_path
                    )
                    
                    # Store in reference data
                    key = rel_path if rel_path else 'root'
                    if key not in self.reference_data:
                        self.reference_data[key] = []
                    
                    self.reference_data[key].append({
                        'path': image_path,
                        'hash': img_hash,
                        'features': features,
                        'components': components,
                        'classifications': unique_classifications
                    })
                else:
                    failed_images += 1
                    
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                failed_images += 1
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        # Summary
        logging.info(f"Reference analysis complete:")
        logging.info(f"  - Total images: {total_images}")
        logging.info(f"  - Failed: {failed_images}")
        logging.info(f"  - Folders: {len(self.reference_data)}")
        logging.info(f"  - Classifications: {len(self.knowledge_bank.relationships)}")
        
        # Display statistics
        stats = self.knowledge_bank.get_statistics()
        logging.info(f"Knowledge bank statistics:")
        logging.info(f"  - Total images: {stats['total_images']}")
        logging.info(f"  - User corrections: {stats['user_corrections']}")
        
        return self.reference_data
    
    def find_similar_images(self, features, top_k=10):
        """Find similar images from knowledge bank"""
        if not self.knowledge_bank.features_db:
            logging.warning("No images in knowledge bank")
            return []
        
        similarities = []
        
        for img_hash, ref_features in self.knowledge_bank.features_db.items():
            similarity = self.calculate_similarity(features, ref_features)
            
            if similarity > 0:
                similarities.append({
                    'hash': img_hash,
                    'similarity': similarity,
                    'classifications': self.knowledge_bank.classifications_db.get(img_hash, []),
                    'characteristics': self.knowledge_bank.characteristics_db.get(img_hash, {}),
                    'file_path': self.knowledge_bank.file_paths_db.get(img_hash, 'unknown')
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def classify_image(self, image_path):
        """Classify an image using similarity matching"""
        logging.info(f"Classifying: {os.path.basename(image_path)}")
        
        # Extract features
        features, img_hash = self.extract_features(image_path)
        
        if features is None:
            logging.error("Failed to extract features")
            return None, None, 0.0
        
        # Find similar images
        similar_images = self.find_similar_images(features)
        
        if not similar_images:
            logging.warning("No similar images found")
            return None, None, 0.0
        
        # Filter by threshold
        threshold = self.config.get('similarity_threshold', 0.65)
        similar_images = [s for s in similar_images if s['similarity'] >= threshold]
        
        if not similar_images:
            logging.warning(f"No images above similarity threshold ({threshold})")
            return None, None, 0.0
        
        # Aggregate classifications with weighted voting
        classification_scores = defaultdict(float)
        component_scores = defaultdict(lambda: defaultdict(float))
        
        # Log top matches
        logging.info(f"Top {min(5, len(similar_images))} similar images:")
        for i, similar in enumerate(similar_images[:5]):
            logging.info(f"  {i+1}. {similar['file_path']} (similarity: {similar['similarity']:.3f})")
        
        # Weight by similarity and position
        for i, similar in enumerate(similar_images):
            # Position weight (top matches get more weight)
            position_weight = 1.0 / (i + 1)
            similarity = similar['similarity']
            
            # Combined weight
            weight = similarity * position_weight
            
            # Vote for classifications
            for classification in similar['classifications']:
                classification_scores[classification] += weight
            
            # Vote for components
            for comp_type, comp_value in similar['characteristics'].items():
                if comp_value:
                    component_scores[comp_type][str(comp_value)] += weight
        
        # Get best classification
        if not classification_scores:
            return None, None, 0.0
        
        # Sort classifications by score
        sorted_classifications = sorted(
            classification_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        best_classification = sorted_classifications[0][0]
        
        # Get best components
        best_components = {}
        for comp_type, values in component_scores.items():
            if values:
                best_value = max(values.items(), key=lambda x: x[1])[0]
                # Only include if it has significant support
                if values[best_value] > sum(values.values()) * 0.3:
                    best_components[comp_type] = best_value
        
        # Calculate confidence
        total_weight = sum(classification_scores.values())
        best_score = sorted_classifications[0][1]
        
        # Confidence based on:
        # 1. Best classification score ratio
        # 2. Average similarity of top matches
        # 3. Agreement between top matches
        
        score_ratio = best_score / total_weight if total_weight > 0 else 0
        avg_similarity = np.mean([s['similarity'] for s in similar_images[:3]])
        
        # Check agreement (how many of top 3 agree on classification)
        top_3_agreement = sum(
            1 for s in similar_images[:3]
            if best_classification in s['classifications']
        ) / min(3, len(similar_images))
        
        confidence = (score_ratio * 0.4 + avg_similarity * 0.4 + top_3_agreement * 0.2)
        
        logging.info(f"Classification result: {best_classification} (confidence: {confidence:.3f})")
        logging.debug(f"Components: {best_components}")
        
        return best_classification, best_components, confidence
    
    def process_dataset_auto(self, reference_folder, dataset_folder):
        """Process dataset in automatic mode"""
        logging.info("="*60)
        logging.info("AUTOMATIC PROCESSING MODE")
        logging.info("="*60)
        
        # Analyze reference folder first
        if not self.reference_data:
            self.analyze_reference_folder(reference_folder)
        
        if not self.reference_data:
            logging.error("No reference data available")
            return
        
        # Collect dataset images
        dataset_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            logging.warning("No images found in dataset folder")
            return
        
        logging.info(f"Found {len(dataset_images)} images to process")
        
        # Statistics
        stats = {
            'total': len(dataset_images),
            'success': 0,
            'failed': 0,
            'low_confidence': 0,
            'already_classified': 0
        }
        
        # Process with progress bar
        for image_path in tqdm(dataset_images, desc="Processing images"):
            try:
                logging.debug(f"Processing: {os.path.basename(image_path)}")
                
                # Check if already classified
                filename = os.path.basename(image_path)
                if re.match(r'^[a-zA-Z0-9]+-[a-zA-Z0-9]+-', filename):
                    logging.debug(f"Already classified: {filename}")
                    stats['already_classified'] += 1
                    continue
                
                # Classify
                classification, components, confidence = self.classify_image(image_path)
                
                if classification and confidence >= self.config.get('auto_mode_threshold', 0.70):
                    # Apply classification
                    success = self._apply_classification(
                        image_path, classification, components, reference_folder, is_manual=False
                    )
                    
                    if success:
                        stats['success'] += 1
                        logging.info(f"✓ Classified: {filename} -> {classification}")
                    else:
                        stats['failed'] += 1
                        
                elif classification:
                    logging.info(f"⚠ Low confidence ({confidence:.3f}): {filename}")
                    stats['low_confidence'] += 1
                else:
                    logging.warning(f"✗ Failed to classify: {filename}")
                    stats['failed'] += 1
                    
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                stats['failed'] += 1
        
        # Save knowledge bank
        self.knowledge_bank.save()
        self.save_config()
        
        # Summary
        logging.info("="*60)
        logging.info("PROCESSING COMPLETE")
        logging.info("="*60)
        logging.info(f"Total images:         {stats['total']}")
        logging.info(f"Successfully renamed: {stats['success']}")
        logging.info(f"Already classified:   {stats['already_classified']}")
        logging.info(f"Low confidence:       {stats['low_confidence']}")
        logging.info(f"Failed:               {stats['failed']}")
        
        success_rate = stats['success'] / (stats['total'] - stats['already_classified']) * 100
        logging.info(f"Success rate:         {success_rate:.1f}%")
        
        if stats['low_confidence'] > 0:
            logging.info(f"\nTip: Run manual mode to handle {stats['low_confidence']} low-confidence images")
    
    def process_dataset_manual(self, reference_folder, dataset_folder):
        """Process dataset in manual mode with GUI"""
        logging.info("="*60)
        logging.info("MANUAL PROCESSING MODE")
        logging.info("="*60)
        
        # Check if GUI is available
        if not GUI_AVAILABLE:
            logging.error("GUI not available. Please install tkinter or use console mode.")
            self.process_dataset_manual_console(reference_folder, dataset_folder)
            return
        
        # Analyze reference folder first
        if not self.reference_data:
            self.analyze_reference_folder(reference_folder)
        
        # Collect dataset images
        dataset_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            logging.warning("No images found in dataset folder")
            return
        
        logging.info(f"Found {len(dataset_images)} images for manual classification")
        
        # Create and run GUI
        try:
            gui = ImageClassifierGUI(self, dataset_images, reference_folder)
            gui.run()
        except Exception as e:
            logging.error(f"GUI error: {e}")
            logging.info("Falling back to console mode")
            self.process_dataset_manual_console(reference_folder, dataset_folder)
    
    def process_dataset_manual_console(self, reference_folder, dataset_folder):
        """Fallback console-based manual mode"""
        logging.info("Running in console mode (no GUI)")
        
        # Collect images
        dataset_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if self.is_image_file(file):
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            logging.warning("No images found")
            return
        
        processed = 0
        skipped = 0
        
        for i, image_path in enumerate(dataset_images):
            print(f"\n{'='*60}")
            print(f"Image {i+1}/{len(dataset_images)}: {os.path.basename(image_path)}")
            print(f"Path: {image_path}")
            
            # Get suggestion
            classification, components, confidence = self.classify_image(image_path)
            
            if classification:
                print(f"\nSuggestion: {classification} (confidence: {confidence:.3f})")
                print(f"Components: {json.dumps(components, indent=2)}")
            else:
                print("\nNo automatic suggestion available")
            
            # Menu
            print("\nOptions:")
            print("1. Accept suggestion")
            print("2. Enter custom classification")
            print("3. Skip")
            print("4. Exit")
            
            choice = input("\nYour choice (1-4): ").strip()
            
            if choice == '1' and classification:
                if self._apply_classification(image_path, classification, components, reference_folder):
                    processed += 1
                    
            elif choice == '2':
                custom = input("Enter classification: ").strip()
                if custom:
                    custom_components = self.parse_classification(custom)
                    if self._apply_classification(image_path, custom, custom_components, reference_folder, is_manual=True):
                        processed += 1
                        
            elif choice == '3':
                skipped += 1
                print("Skipped")
                
            elif choice == '4':
                break
        
        # Save
        self.knowledge_bank.save()
        self.save_config()
        
        print(f"\nManual processing complete")
        print(f"Processed: {processed}, Skipped: {skipped}")
    
    def _apply_classification(self, image_path, classification, components, reference_folder, is_manual=False):
        """Apply classification by renaming file"""
        try:
            # Clean classification
            clean_class = classification.replace('/', '-').replace('\\', '-')
            
            # Get paths
            directory = os.path.dirname(image_path)
            extension = Path(image_path).suffix
            base_name = classification
            
            # Create folder structure if enabled and not manual classification
            if self.config.get('auto_create_folders', True) and components and not is_manual:
                target_dir = self.create_folder_structure(directory, components)
            else:
                target_dir = directory
            
            # Generate unique filename
            new_filename = f"{base_name}{extension}"
            new_path = os.path.join(target_dir, new_filename)
            
            counter = 1
            while os.path.exists(new_path) and new_path != image_path:
                new_filename = f"{base_name}_{counter}{extension}"
                new_path = os.path.join(target_dir, new_filename)
                counter += 1
            
            # Move/rename file
            if new_path != image_path:
                shutil.move(image_path, new_path)
                logging.info(f"Moved: {os.path.basename(image_path)} -> {new_path}")
            else:
                logging.debug("File already has correct name and location")
            
            # Extract features for the new location
            features, img_hash = self.extract_features(new_path)
            
            if features is not None:
                # Add to knowledge bank
                self.knowledge_bank.add_image(
                    img_hash,
                    features,
                    [classification],
                    components,
                    new_path
                )
                
                # Save to reference if high confidence
                if self.config.get('save_learned_references', True):
                    self.save_to_reference(new_path, classification, components, reference_folder)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply classification: {e}")
            return False
    
    def create_folder_structure(self, base_path, components):
        """Create hierarchical folder structure based on components"""
        path_parts = []
        
        # Priority order for folder hierarchy
        hierarchy = [
            'connector_type',
            'core_diameter', 
            'region',
            'condition',
            'defect_type'
        ]
        
        for key in hierarchy:
            if key in components and components[key]:
                path_parts.append(str(components[key]))
        
        if path_parts:
            new_path = os.path.join(base_path, *path_parts)
            os.makedirs(new_path, exist_ok=True)
            logging.debug(f"Created folder structure: {new_path}")
            return new_path
        
        return base_path
    
    def save_to_reference(self, image_path, classification, components, reference_folder):
        """Save classified image to reference folder"""
        try:
            # Create folder structure in reference
            target_dir = self.create_folder_structure(reference_folder, components)
            
            # Copy to reference
            filename = os.path.basename(image_path)
            target_path = os.path.join(target_dir, filename)
            
            # Ensure unique
            if os.path.exists(target_path) and target_path != image_path:
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    filename = f"{base}_{counter}{ext}"
                    target_path = os.path.join(target_dir, filename)
                    counter += 1
            
            if target_path != image_path:
                shutil.copy2(image_path, target_path)
                logging.debug(f"Saved to reference: {target_path}")
                
        except Exception as e:
            logging.error(f"Failed to save to reference: {e}")
    
    def is_image_file(self, filepath):
        """Check if file is an image"""
        return Path(filepath).suffix.lower() in self.image_extensions
    
    def get_unique_filename(self, directory, base_name, extension):
        """Generate unique filename"""
        counter = 1
        new_filename = f"{base_name}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        while os.path.exists(new_path):
            new_filename = f"{base_name}_{counter}{extension}"
            new_path = os.path.join(directory, new_filename)
            counter += 1
        
        return new_filename

import argparse

def main():
    """Main entry point with interactive setup"""
    try:
        print("\n" + "="*80)
        print("ULTIMATE IMAGE CLASSIFIER - COMPREHENSIVE VERSION")
        print("="*80 + "\n")
        
        parser = argparse.ArgumentParser(description="Ultimate Image Classifier")
        parser.add_argument("--reference_folder", type=str, default=os.path.abspath('reference'),
                            help="Path to the reference image folder.")
        parser.add_argument("--dataset_folder", type=str, default=os.path.abspath('dataset'),
                            help="Path to the dataset image folder.")
        parser.add_argument("--mode", type=str, choices=["auto", "manual", "exit"], default="auto",
                            help="Operation mode: 'auto' for automatic, 'manual' for GUI-based manual, 'exit' to quit.")
        parser.add_argument("--similarity_threshold", type=float, default=0.65,
                            help="Similarity threshold for classification (0.0-1.0).")
        parser.add_argument("--auto_create_folders", type=bool, default=True,
                            help="Automatically create folders based on classification.")
        parser.add_argument("--custom_keywords", nargs='*', default=[],
                            help="List of custom keywords to add to the knowledge bank.")
        
        args = parser.parse_args()

        # Initialize classifier with parsed arguments
        classifier = UltimateImageClassifier(
            similarity_threshold=args.similarity_threshold,
            auto_create_folders=args.auto_create_folders,
            custom_keywords=args.custom_keywords
        )
        
        reference_folder = args.reference_folder
        dataset_folder = args.dataset_folder
        
        # Verify paths
        if not os.path.exists(reference_folder):
            logging.error(f"Reference folder not found: {reference_folder}")
            print("\nWould you like to create it? (y/n): ", end='')
            if input().strip().lower() == 'y':
                os.makedirs(reference_folder, exist_ok=True)
                logging.info(f"Created reference folder: {reference_folder}")
                print("\nPlease add reference images and run again.")
                return
            else:
                return
        
        if not os.path.exists(dataset_folder):
            logging.info(f"Creating dataset folder: {dataset_folder}")
            os.makedirs(dataset_folder, exist_ok=True)
        
        # Check for reference images
        ref_count = sum(
            1 for root, dirs, files in os.walk(reference_folder)
            for file in files
            if classifier.is_image_file(file)
        )
        
        if ref_count == 0:
            logging.error("No reference images found!")
            print("\nPlease add reference images to the reference folder.")
            return
        
        logging.info(f"Found {ref_count} reference images")
        
        if args.mode == 'auto':
            classifier.process_dataset_auto(reference_folder, dataset_folder)
        elif args.mode == 'manual':
            classifier.process_dataset_manual(reference_folder, dataset_folder)
        elif args.mode == 'exit':
            logging.info("Exiting...")
            classifier.analyze_reference_folder(reference_folder)
        
        # Display final statistics
        stats = classifier.knowledge_bank.get_statistics()
        print(f"\nKnowledge Bank Statistics:")
        print(f"  Total images learned: {stats['total_images']}")
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  User corrections: {stats['user_corrections']}")
        
        if stats['classifications']:
            print(f"\nTop classifications:")
            for cls, count in stats['classifications'].items():
                print(f"  - {cls}: {count}")
        
    except KeyboardInterrupt:
        logging.info("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.debug(traceback.format_exc())

if __name__ == "__main__":
    main()
