#!/usr/bin/env python3
"""
Deep Learning Image Similarity Classifier and Renamer
Analyzes classified images and renames similar images in target directories
"""

import os
import sys
import subprocess
import importlib
import time
from datetime import datetime
from pathlib import Path
import shutil
from collections import defaultdict

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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{timestamp()}] Starting Image Similarity Classifier and Renamer")
print(f"[{timestamp()}] Checking and installing required dependencies...")

# Check and install required packages
required_packages = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("scikit-learn", "sklearn"),
    ("tqdm", "tqdm"),
    ("opencv-python", "cv2")
]

for package, import_name in required_packages:
    if not install_package(package, import_name):
        print(f"[{timestamp()}] ERROR: Cannot proceed without {package}")
        sys.exit(1)

# Import all required modules after installation
print(f"[{timestamp()}] Importing required modules...")
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

class ImageClassifierRenamer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{timestamp()}] Using device: {self.device}")
        
        # Initialize pre-trained model
        print(f"[{timestamp()}] Loading pre-trained ResNet50 model...")
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        print(f"[{timestamp()}] Model loaded successfully")
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Storage for features and classifications
        self.classified_features = {}
        self.classification_map = {}
        self.rename_history = []
        
    def is_image_file(self, filepath):
        """Check if file is an image based on extension"""
        return Path(filepath).suffix.lower() in self.image_extensions
    
    def extract_classification(self, filename):
        """Extract classification from filename"""
        # Remove extension and any trailing numbers
        base_name = Path(filename).stem
        
        # Remove trailing numbers and hyphens/underscores
        import re
        # This regex removes trailing patterns like -1, _2, -001, etc.
        cleaned_name = re.sub(r'[-_]\d+$', '', base_name)
        
        return cleaned_name
    
    def extract_features(self, image_path):
        """Extract feature vector from image using pre-trained model"""
        try:
            print(f"[{timestamp()}] Extracting features from: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            print(f"[{timestamp()}] ERROR processing {image_path}: {str(e)}")
            return None
    
    def analyze_classified_directory(self, directory_path):
        """Analyze all images in the classified directory"""
        print(f"[{timestamp()}] Analyzing classified images in: {directory_path}")
        
        classified_count = 0
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if self.is_image_file(file):
                    filepath = os.path.join(root, file)
                    features = self.extract_features(filepath)
                    
                    if features is not None:
                        classification = self.extract_classification(file)
                        
                        if classification not in self.classified_features:
                            self.classified_features[classification] = []
                        
                        self.classified_features[classification].append(features)
                        classified_count += 1
                        print(f"[{timestamp()}] Added '{classification}' classification from {file}")
        
        # Calculate average features for each classification
        print(f"[{timestamp()}] Computing average features for each classification...")
        for classification in self.classified_features:
            features_array = np.array(self.classified_features[classification])
            avg_features = np.mean(features_array, axis=0)
            self.classification_map[classification] = avg_features
            print(f"[{timestamp()}] Computed average for '{classification}' ({len(self.classified_features[classification])} samples)")
        
        print(f"[{timestamp()}] Analyzed {classified_count} images with {len(self.classification_map)} unique classifications")
        return classified_count
    
    def find_most_similar_classification(self, features, threshold=0.7):
        """Find the most similar classification for given features"""
        if not self.classification_map:
            return None, 0.0
        
        max_similarity = -1
        best_classification = None
        
        for classification, class_features in self.classification_map.items():
            similarity = cosine_similarity(
                features.reshape(1, -1), 
                class_features.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_classification = classification
        
        if max_similarity >= threshold:
            return best_classification, max_similarity
        else:
            return None, max_similarity
    
    def get_unique_filename(self, directory, base_name, extension):
        """Generate unique filename by appending numbers if necessary"""
        counter = 1
        new_name = f"{base_name}{extension}"
        new_path = os.path.join(directory, new_name)
        
        while os.path.exists(new_path):
            new_name = f"{base_name}-{counter}{extension}"
            new_path = os.path.join(directory, new_name)
            counter += 1
        
        return new_name, new_path
    
    def process_target_directory(self, target_directory, similarity_threshold):
        """Process all images in target directory and rename based on similarity"""
        print(f"[{timestamp()}] Starting deep crawl of target directory: {target_directory}")
        
        processed_count = 0
        renamed_count = 0
        unclassified_count = 0
        
        # Collect all image files first
        all_images = []
        for root, dirs, files in os.walk(target_directory):
            for file in files:
                if self.is_image_file(file):
                    all_images.append(os.path.join(root, file))
        
        print(f"[{timestamp()}] Found {len(all_images)} images to process")
        
        # Process each image
        for image_path in tqdm(all_images, desc="Processing images"):
            print(f"[{timestamp()}] Processing: {image_path}")
            
            features = self.extract_features(image_path)
            if features is None:
                print(f"[{timestamp()}] Skipping {image_path} - feature extraction failed")
                continue
            
            classification, similarity = self.find_most_similar_classification(features, similarity_threshold)
            processed_count += 1
            
            if classification:
                # Generate new filename
                directory = os.path.dirname(image_path)
                extension = Path(image_path).suffix
                new_name, new_path = self.get_unique_filename(directory, classification, extension)
                
                # Rename the file
                try:
                    os.rename(image_path, new_path)
                    renamed_count += 1
                    self.rename_history.append({
                        'original': image_path,
                        'new': new_path,
                        'classification': classification,
                        'similarity': similarity
                    })
                    print(f"[{timestamp()}] RENAMED: {os.path.basename(image_path)} -> {new_name} (similarity: {similarity:.3f})")
                except Exception as e:
                    print(f"[{timestamp()}] ERROR renaming {image_path}: {str(e)}")
            else:
                unclassified_count += 1
                print(f"[{timestamp()}] NO MATCH: {os.path.basename(image_path)} (best similarity: {similarity:.3f})")
        
        return processed_count, renamed_count, unclassified_count
    
    def save_rename_history(self, output_file="rename_history.txt"):
        """Save the rename history to a file"""
        print(f"[{timestamp()}] Saving rename history to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Image Rename History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for entry in self.rename_history:
                f.write(f"Original: {entry['original']}\n")
                f.write(f"New: {entry['new']}\n")
                f.write(f"Classification: {entry['classification']}\n")
                f.write(f"Similarity: {entry['similarity']:.3f}\n")
                f.write("-"*40 + "\n")
        
        print(f"[{timestamp()}] Rename history saved")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("IMAGE SIMILARITY CLASSIFIER AND RENAMER")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = ImageClassifierRenamer()
    
    # Get source directory
    while True:
        source_dir = input("\nEnter the path to your classified images directory: ").strip()
        source_dir = os.path.expanduser(source_dir)
        
        if os.path.isdir(source_dir):
            print(f"[{timestamp()}] Source directory confirmed: {source_dir}")
            break
        else:
            print(f"[{timestamp()}] ERROR: Directory not found. Please try again.")
    
    # Analyze classified images
    classified_count = classifier.analyze_classified_directory(source_dir)
    
    if classified_count == 0:
        print(f"[{timestamp()}] ERROR: No images found in source directory")
        return
    
    # Get target directory
    while True:
        target_dir = input("\nEnter the path to the directory to search and rename images: ").strip()
        target_dir = os.path.expanduser(target_dir)
        
        if os.path.isdir(target_dir):
            print(f"[{timestamp()}] Target directory confirmed: {target_dir}")
            break
        else:
            print(f"[{timestamp()}] ERROR: Directory not found. Please try again.")
    
    # Get similarity threshold
    while True:
        threshold_input = input("\nEnter similarity threshold (0.0-1.0, recommended 0.7): ").strip()
        try:
            threshold = float(threshold_input)
            if 0.0 <= threshold <= 1.0:
                print(f"[{timestamp()}] Similarity threshold set to: {threshold}")
                break
            else:
                print(f"[{timestamp()}] ERROR: Threshold must be between 0.0 and 1.0")
        except ValueError:
            print(f"[{timestamp()}] ERROR: Invalid number. Please try again.")
    
    # Ask for confirmation
    print(f"\n[{timestamp()}] Ready to process images")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print(f"  Threshold: {threshold}")
    
    confirm = input("\nProceed with renaming? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print(f"[{timestamp()}] Operation cancelled by user")
        return
    
    # Process target directory
    print(f"\n[{timestamp()}] Starting image processing...")
    processed, renamed, unclassified = classifier.process_target_directory(target_dir, threshold)
    
    # Summary
    print(f"\n[{timestamp()}] PROCESSING COMPLETE")
    print(f"[{timestamp()}] Total images processed: {processed}")
    print(f"[{timestamp()}] Images renamed: {renamed}")
    print(f"[{timestamp()}] Images unclassified: {unclassified}")
    
    # Save history
    if renamed > 0:
        save_history = input("\nSave rename history to file? (yes/no): ").strip().lower()
        if save_history == 'yes':
            history_file = input("Enter filename for history (default: rename_history.txt): ").strip()
            if not history_file:
                history_file = "rename_history.txt"
            classifier.save_rename_history(history_file)
    
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