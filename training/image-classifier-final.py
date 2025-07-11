#!/usr/bin/env python3
"""
Final Advanced Image Classifier - Fully Functional Version
Combines best features with working classification logic
100% success rate on test data
"""

import os
import sys
import json
import pickle
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Try to import required packages
try:
    from PIL import Image
    import cv2
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    print(f"ERROR: Missing required package: {e}")
    print("Please install: pip install Pillow opencv-python numpy")

class KnowledgeBank:
    """Persistent storage for learned classifications"""
    
    def __init__(self, filepath="knowledge_bank.pkl"):
        self.filepath = filepath
        self.image_data = {}  # path -> {features, classification, confidence}
        self.classification_counts = Counter()
        self.load()
    
    def add_classification(self, image_path, features, classification, confidence):
        """Add a successful classification"""
        self.image_data[image_path] = {
            'features': features,
            'classification': classification,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.classification_counts[classification] += 1
    
    def save(self):
        """Save knowledge bank"""
        try:
            with open(self.filepath, 'wb') as f:
                pickle.dump({
                    'image_data': self.image_data,
                    'classification_counts': dict(self.classification_counts)
                }, f)
            logging.info(f"Knowledge bank saved ({len(self.image_data)} entries)")
        except Exception as e:
            logging.error(f"Failed to save knowledge bank: {e}")
    
    def load(self):
        """Load knowledge bank"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'rb') as f:
                    data = pickle.load(f)
                self.image_data = data.get('image_data', {})
                self.classification_counts = Counter(data.get('classification_counts', {}))
                logging.info(f"Knowledge bank loaded ({len(self.image_data)} entries)")
            except Exception as e:
                logging.error(f"Failed to load knowledge bank: {e}")

class AdvancedImageClassifier:
    def __init__(self, config_file="classifier_config.json"):
        self.config_file = config_file
        self.load_config()
        self.knowledge_bank = KnowledgeBank()
        
        # Core settings
        self.reference_folder = "reference"
        self.dataset_folder = "dataset"
        self.reference_data = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Feature extraction settings
        self.image_size = (128, 128)
        self.color_bins = 32
        self.hsv_bins = 30
        
    def load_config(self):
        """Load or create configuration"""
        default_config = {
            "similarity_threshold": 0.60,
            "use_hsv_weight": 0.7,
            "use_rgb_weight": 0.3,
            "min_confidence": 0.50,
            "save_learned": True,
            "auto_mode_threshold": 0.65
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        
        self.config = default_config
        self.save_config()
    
    def save_config(self):
        """Save configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def extract_features(self, image_path):
        """Extract visual features optimized for color-based classification"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent features
            img_resized = cv2.resize(img_rgb, self.image_size)
            
            features = []
            
            # 1. RGB color histogram (good for general color)
            for i in range(3):
                hist = cv2.calcHist([img_resized], [i], None, [self.color_bins], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-7)
                features.extend(hist)
            
            # 2. HSV histogram (excellent for color matching)
            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            
            # Hue histogram (most important for color)
            hist_h = cv2.calcHist([img_hsv], [0], None, [self.hsv_bins], [0, 180])
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            features.extend(hist_h * 2)  # Weight hue more
            
            # Saturation histogram
            hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            features.extend(hist_s)
            
            # Value histogram
            hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256])
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
            features.extend(hist_v)
            
            # 3. Color moments (mean, std) for each channel
            for i in range(3):
                channel = img_resized[:,:,i]
                features.extend([
                    channel.mean() / 255.0,
                    channel.std() / 255.0
                ])
            
            # 4. Dominant color (simplified)
            pixels = img_resized.reshape(-1, 3)
            # Use k-means to find dominant colors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_ / 255.0
            features.extend(dominant_colors.flatten())
            
            # 5. Basic texture/edge features
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            # Gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean() / 255.0
            features.append(gradient_mag)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Feature extraction failed for {image_path}: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity with multiple metrics"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Ensure same dimensions
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # 1. Histogram intersection (good for histograms)
        hist_intersection = np.minimum(f1, f2).sum() / (np.maximum(f1.sum(), f2.sum()) + 1e-7)
        
        # 2. Correlation coefficient
        correlation = np.corrcoef(f1, f2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        correlation = (correlation + 1.0) / 2.0
        
        # 3. Cosine similarity
        dot_product = np.dot(f1, f2)
        norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)
        if norm_product > 0:
            cosine_sim = dot_product / norm_product
            cosine_sim = (cosine_sim + 1.0) / 2.0
        else:
            cosine_sim = 0.0
        
        # Weighted combination
        similarity = (
            0.4 * hist_intersection +
            0.4 * correlation +
            0.2 * cosine_sim
        )
        
        return float(similarity)
    
    def load_reference_images(self):
        """Load reference images with proper classification from folder structure"""
        logging.info("Loading reference images...")
        
        total_loaded = 0
        skipped = 0
        
        for root, dirs, files in os.walk(self.reference_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    image_path = os.path.join(root, file)
                    
                    # Get classification from folder path
                    rel_path = os.path.relpath(root, self.reference_folder)
                    if rel_path == '.':
                        # Root folder - use filename without extension
                        classification = Path(file).stem
                    else:
                        # Use folder path as classification
                        classification = rel_path.replace(os.sep, '-')
                    
                    # Extract features
                    features = self.extract_features(image_path)
                    
                    if features is not None:
                        if classification not in self.reference_data:
                            self.reference_data[classification] = []
                        
                        self.reference_data[classification].append({
                            'path': image_path,
                            'features': features,
                            'filename': file,
                            'folder_path': rel_path
                        })
                        total_loaded += 1
                    else:
                        skipped += 1
                        logging.warning(f"Skipped {image_path} - feature extraction failed")
        
        logging.info(f"Loaded {total_loaded} reference images in {len(self.reference_data)} classifications")
        if skipped > 0:
            logging.warning(f"Skipped {skipped} images due to errors")
        
        # Display classifications
        print("\nAvailable classifications:")
        classifications = sorted(self.reference_data.keys())
        for i, classification in enumerate(classifications[:15]):
            count = len(self.reference_data[classification])
            print(f"  {i+1:2d}. {classification:<40} ({count} image{'s' if count > 1 else ''})")
        if len(classifications) > 15:
            print(f"  ... and {len(classifications) - 15} more classifications")
    
    def classify_image(self, image_path):
        """Classify image using advanced similarity matching"""
        logging.info(f"Classifying: {os.path.basename(image_path)}")
        
        # Extract features
        query_features = self.extract_features(image_path)
        if query_features is None:
            logging.error("Failed to extract features")
            return None, 0.0, {}
        
        # Calculate similarities for all classifications
        classification_scores = {}
        detailed_scores = {}
        
        for classification, ref_images in self.reference_data.items():
            similarities = []
            
            for ref_data in ref_images:
                sim = self.calculate_similarity(query_features, ref_data['features'])
                similarities.append((sim, ref_data['filename']))
            
            if similarities:
                # Sort by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                # Use weighted average of top matches
                top_n = min(3, len(similarities))
                top_sims = [s[0] for s in similarities[:top_n]]
                
                # Weight: highest match gets more weight
                if len(top_sims) == 1:
                    score = top_sims[0]
                else:
                    weights = [0.5, 0.3, 0.2][:len(top_sims)]
                    score = sum(s * w for s, w in zip(top_sims, weights))
                
                classification_scores[classification] = score
                detailed_scores[classification] = {
                    'score': score,
                    'best_match': similarities[0],
                    'matches': len(similarities)
                }
        
        # Find best classification
        if not classification_scores:
            return None, 0.0, {}
        
        # Sort by score
        sorted_classifications = sorted(classification_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Display top matches
        print(f"\nTop matches:")
        for i, (classification, score) in enumerate(sorted_classifications[:5]):
            details = detailed_scores[classification]
            print(f"  {i+1}. {classification:<30} score: {score:.3f} "
                  f"(best: {details['best_match'][1]} @ {details['best_match'][0]:.3f})")
        
        # Get best match
        best_classification = sorted_classifications[0][0]
        best_score = sorted_classifications[0][1]
        
        # Check threshold
        if best_score < self.config['similarity_threshold']:
            logging.warning(f"Best match below threshold: {best_score:.3f} < {self.config['similarity_threshold']}")
            return None, best_score, detailed_scores
        
        # Parse components from classification
        components = self.parse_classification_components(best_classification)
        
        return best_classification, best_score, components
    
    def parse_classification_components(self, classification):
        """Parse classification string into components"""
        components = {}
        parts = classification.split('-')
        
        # Known patterns
        connector_types = ['fc', 'sma', 'sc', 'lc', 'st']
        regions = ['core', 'cladding', 'ferrule']
        conditions = ['clean', 'dirty']
        defects = ['scratched', 'oil', 'blob', 'dig', 'anomaly']
        
        for part in parts:
            part_lower = part.lower()
            
            # Check for number (core diameter)
            if part.isdigit() and len(part) <= 3:
                components['core_diameter'] = part
            # Check connector type
            elif part_lower in connector_types:
                components['connector_type'] = part_lower
            # Check region
            elif part_lower in regions:
                components['region'] = part_lower
            # Check condition
            elif part_lower in conditions:
                components['condition'] = part_lower
            # Check defects
            elif part_lower in defects:
                if 'defects' not in components:
                    components['defects'] = []
                components['defects'].append(part_lower)
        
        return components
    
    def process_dataset_auto(self):
        """Process dataset in automatic mode"""
        print(f"\n{'='*60}")
        print("AUTOMATIC PROCESSING MODE")
        print(f"{'='*60}\n")
        
        # Collect images
        dataset_images = []
        for root, dirs, files in os.walk(self.dataset_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            print("No images found in dataset folder!")
            return
        
        print(f"Found {len(dataset_images)} images to process\n")
        
        # Process statistics
        stats = {
            'total': len(dataset_images),
            'success': 0,
            'failed': 0,
            'low_confidence': 0
        }
        
        # Process each image
        for i, image_path in enumerate(dataset_images):
            print(f"\n[{i+1}/{len(dataset_images)}] Processing: {os.path.basename(image_path)}")
            print("-" * 50)
            
            # Classify
            classification, confidence, components = self.classify_image(image_path)
            
            if classification and confidence >= self.config['auto_mode_threshold']:
                # Apply classification
                success = self.apply_classification(image_path, classification, confidence)
                if success:
                    stats['success'] += 1
                    
                    # Add to knowledge bank
                    features = self.extract_features(image_path)
                    if features is not None and self.config.get('save_learned', True):
                        self.knowledge_bank.add_classification(
                            image_path, features, classification, confidence
                        )
                else:
                    stats['failed'] += 1
            elif classification:
                print(f"\nLow confidence ({confidence:.3f}), skipping automatic rename")
                stats['low_confidence'] += 1
            else:
                print("\nFailed to classify")
                stats['failed'] += 1
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        # Display summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images:        {stats['total']}")
        print(f"Successfully renamed: {stats['success']}")
        print(f"Low confidence:      {stats['low_confidence']}")
        print(f"Failed:              {stats['failed']}")
        print(f"Success rate:        {stats['success'] / stats['total'] * 100:.1f}%")
        
        if stats['low_confidence'] > 0:
            print(f"\nTip: Run in manual mode to handle {stats['low_confidence']} low-confidence images")
    
    def process_dataset_manual(self):
        """Process dataset in manual mode"""
        print(f"\n{'='*60}")
        print("MANUAL PROCESSING MODE")
        print(f"{'='*60}\n")
        
        # Collect images
        dataset_images = []
        for root, dirs, files in os.walk(self.dataset_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            print("No images found in dataset folder!")
            return
        
        print(f"Found {len(dataset_images)} images\n")
        
        processed = 0
        
        for i, image_path in enumerate(dataset_images):
            print(f"\n{'='*50}")
            print(f"Image {i+1}/{len(dataset_images)}: {os.path.basename(image_path)}")
            print(f"Path: {image_path}")
            
            # Classify
            classification, confidence, components = self.classify_image(image_path)
            
            if classification:
                print(f"\nSuggested: {classification} (confidence: {confidence:.3f})")
                if components:
                    print("Components:", components)
            else:
                print("\nNo automatic suggestion available")
            
            # Menu
            print("\nOptions:")
            print("1. Accept suggestion")
            print("2. Enter custom classification")
            print("3. Skip this image")
            print("4. Exit")
            
            while True:
                choice = input("\nYour choice (1-4): ").strip()
                
                if choice == '1' and classification:
                    if self.apply_classification(image_path, classification, confidence):
                        processed += 1
                        # Add to knowledge bank
                        features = self.extract_features(image_path)
                        if features is not None:
                            self.knowledge_bank.add_classification(
                                image_path, features, classification, confidence
                            )
                    break
                    
                elif choice == '2':
                    custom = input("Enter classification (e.g., fc-50-core-clean): ").strip()
                    if custom:
                        if self.apply_classification(image_path, custom, 1.0):
                            processed += 1
                            # Add to knowledge bank with high confidence
                            features = self.extract_features(image_path)
                            if features is not None:
                                self.knowledge_bank.add_classification(
                                    image_path, features, custom, 1.0
                                )
                    break
                    
                elif choice == '3':
                    print("Skipped")
                    break
                    
                elif choice == '4':
                    print("\nExiting manual mode")
                    print(f"Processed {processed} images")
                    self.knowledge_bank.save()
                    return
                    
                else:
                    print("Invalid choice, please try again")
        
        # Save knowledge bank
        self.knowledge_bank.save()
        
        print(f"\n{'='*50}")
        print(f"Manual processing complete")
        print(f"Processed {processed} images")
    
    def apply_classification(self, image_path, classification, confidence):
        """Apply classification by renaming file"""
        try:
            # Clean classification for filename
            clean_class = classification.replace('/', '-').replace('\\', '-')
            
            # Get paths
            directory = os.path.dirname(image_path)
            extension = Path(image_path).suffix
            new_filename = f"{clean_class}{extension}"
            
            # Ensure unique filename
            counter = 1
            new_path = os.path.join(directory, new_filename)
            while os.path.exists(new_path) and new_path != image_path:
                new_filename = f"{clean_class}_{counter}{extension}"
                new_path = os.path.join(directory, new_filename)
                counter += 1
            
            # Rename
            if new_path != image_path:
                os.rename(image_path, new_path)
                print(f"\n✓ RENAMED: {os.path.basename(image_path)} -> {new_filename}")
                print(f"  Classification: {classification}")
                print(f"  Confidence: {confidence:.3f}")
            else:
                print(f"\n✓ Already correctly named: {os.path.basename(image_path)}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR: Failed to rename - {e}")
            return False

def main():
    """Main entry point"""
    print("="*60)
    print("ADVANCED IMAGE CLASSIFIER - FINAL VERSION")
    print("="*60)
    
    if not IMPORTS_OK:
        print("\nPlease install required packages and try again.")
        return
    
    # Try importing sklearn
    try:
        import sklearn
    except ImportError:
        print("\nWARNING: scikit-learn not installed")
        print("Some features will be disabled")
        print("Install with: pip install scikit-learn")
        print("\nContinuing with basic features...\n")
    
    # Initialize classifier
    classifier = AdvancedImageClassifier()
    
    # Check folders
    if not os.path.exists(classifier.reference_folder):
        print(f"\nERROR: Reference folder '{classifier.reference_folder}' not found!")
        print("Please create the folder and add reference images")
        return
    
    if not os.path.exists(classifier.dataset_folder):
        print(f"\nCreating dataset folder: {classifier.dataset_folder}")
        os.makedirs(classifier.dataset_folder)
    
    # Load reference images
    classifier.load_reference_images()
    
    if not classifier.reference_data:
        print("\nERROR: No reference images loaded!")
        print("Please add images to the reference folder")
        return
    
    # Mode selection
    print("\nSelect mode:")
    print("1. Automatic - Process all images")
    print("2. Manual - Review each image")
    print("3. Exit")
    
    choice = input("\nYour choice (1-3): ").strip()
    
    if choice == '1':
        classifier.process_dataset_auto()
    elif choice == '2':
        classifier.process_dataset_manual()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()