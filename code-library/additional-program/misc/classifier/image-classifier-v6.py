#!/usr/bin/env python3
"""
Working Image Classifier - Simplified and Functional Version
Classifies images based on visual similarity to reference images
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Try to import required packages
try:
    from PIL import Image
    import cv2
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Required packages loaded successfully")
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Please install: pip install Pillow opencv-python numpy")
    sys.exit(1)

class SimpleImageClassifier:
    def __init__(self):
        self.reference_folder = "reference"
        self.dataset_folder = "dataset"
        self.reference_data = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
    def extract_simple_features(self, image_path):
        """Extract simple but effective features from image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Convert to RGB (cv2 loads as BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size for consistent features
            img_resized = cv2.resize(img_rgb, (128, 128))
            
            # Extract features
            features = []
            
            # 1. Color histogram (most important for these images)
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
                features.extend(hist)
            
            # 2. HSV histogram (better for color matching)
            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            # Hue is most important for color
            hist_h = cv2.calcHist([img_hsv], [0], None, [30], [0, 180])
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            features.extend(hist_h)
            
            # 3. Basic statistics
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            features.extend([
                gray.mean() / 255.0,
                gray.std() / 255.0,
                np.median(gray) / 255.0
            ])
            
            # 4. Simple edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
            
        # Ensure same dimensions
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Use correlation coefficient (works well for histograms)
        correlation = np.corrcoef(f1, f2)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0
            
        # Convert to 0-1 range
        similarity = (correlation + 1.0) / 2.0
        
        return similarity
    
    def load_reference_images(self):
        """Load and analyze reference images"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading reference images...")
        
        total_loaded = 0
        
        for root, dirs, files in os.walk(self.reference_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    image_path = os.path.join(root, file)
                    
                    # Extract classification from folder path
                    rel_path = os.path.relpath(root, self.reference_folder)
                    if rel_path == '.':
                        classification = Path(file).stem
                    else:
                        # Use folder path as classification
                        classification = rel_path.replace(os.sep, '-')
                    
                    # Extract features
                    features = self.extract_simple_features(image_path)
                    
                    if features is not None:
                        if classification not in self.reference_data:
                            self.reference_data[classification] = []
                        
                        self.reference_data[classification].append({
                            'path': image_path,
                            'features': features,
                            'filename': file
                        })
                        total_loaded += 1
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {total_loaded} reference images in {len(self.reference_data)} classifications")
        
        # Show some classifications
        print("\nAvailable classifications:")
        for i, (classification, images) in enumerate(list(self.reference_data.items())[:10]):
            print(f"  - {classification} ({len(images)} images)")
        if len(self.reference_data) > 10:
            print(f"  ... and {len(self.reference_data) - 10} more classifications")
    
    def classify_image(self, image_path):
        """Classify an image based on reference images"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Classifying: {os.path.basename(image_path)}")
        
        # Extract features
        query_features = self.extract_simple_features(image_path)
        if query_features is None:
            print("Failed to extract features")
            return None, 0.0
        
        # Find best matching classification
        best_classification = None
        best_similarity = 0.0
        all_scores = []
        
        for classification, ref_images in self.reference_data.items():
            # Calculate average similarity to all images in this classification
            similarities = []
            
            for ref_data in ref_images:
                sim = self.calculate_similarity(query_features, ref_data['features'])
                similarities.append(sim)
            
            # Use average similarity
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            # Weight towards max to handle cases where one image is very similar
            combined_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
            
            all_scores.append((classification, combined_similarity, max_similarity, avg_similarity))
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_classification = classification
        
        # Sort scores for debugging
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Show top matches
        print(f"\nTop 5 matches:")
        for i, (classification, combined, max_sim, avg_sim) in enumerate(all_scores[:5]):
            print(f"  {i+1}. {classification}: {combined:.3f} (max: {max_sim:.3f}, avg: {avg_sim:.3f})")
        
        # Apply threshold
        threshold = 0.60  # Lower threshold for better matches
        if best_similarity < threshold:
            print(f"\nBest match ({best_classification}) has low confidence: {best_similarity:.3f}")
            return None, best_similarity
        
        return best_classification, best_similarity
    
    def process_dataset(self):
        """Process all images in dataset folder"""
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET")
        print(f"{'='*60}")
        
        # Collect images
        dataset_images = []
        for root, dirs, files in os.walk(self.dataset_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            print("No images found in dataset folder")
            return
        
        print(f"Found {len(dataset_images)} images to process\n")
        
        # Process each image
        results = {
            'success': 0,
            'failed': 0,
            'total': len(dataset_images)
        }
        
        for image_path in dataset_images:
            classification, confidence = self.classify_image(image_path)
            
            if classification:
                # Build new filename
                # Clean up classification for filename
                clean_classification = classification.replace('/', '-').replace('\\', '-')
                
                # Get file extension
                extension = Path(image_path).suffix
                
                # Create new filename
                directory = os.path.dirname(image_path)
                new_filename = f"{clean_classification}{extension}"
                
                # Ensure unique filename
                counter = 1
                new_path = os.path.join(directory, new_filename)
                while os.path.exists(new_path):
                    new_filename = f"{clean_classification}_{counter}{extension}"
                    new_path = os.path.join(directory, new_filename)
                    counter += 1
                
                # Rename file
                try:
                    os.rename(image_path, new_path)
                    print(f"\n✓ RENAMED: {os.path.basename(image_path)} -> {new_filename}")
                    print(f"  Classification: {classification}")
                    print(f"  Confidence: {confidence:.3f}")
                    results['success'] += 1
                except Exception as e:
                    print(f"\n✗ ERROR renaming {os.path.basename(image_path)}: {e}")
                    results['failed'] += 1
            else:
                print(f"\n✗ FAILED to classify: {os.path.basename(image_path)}")
                results['failed'] += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {results['total']}")
        print(f"Successfully classified: {results['success']}")
        print(f"Failed to classify: {results['failed']}")
        print(f"Success rate: {results['success'] / results['total'] * 100:.1f}%")
    
    def manual_mode(self):
        """Manual classification mode"""
        print(f"\n{'='*60}")
        print(f"MANUAL CLASSIFICATION MODE")
        print(f"{'='*60}")
        
        # Collect images
        dataset_images = []
        for root, dirs, files in os.walk(self.dataset_folder):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    dataset_images.append(os.path.join(root, file))
        
        if not dataset_images:
            print("No images found in dataset folder")
            return
        
        print(f"Found {len(dataset_images)} images to process\n")
        
        for i, image_path in enumerate(dataset_images):
            print(f"\n{'='*40}")
            print(f"Image {i+1}/{len(dataset_images)}: {os.path.basename(image_path)}")
            
            # Try automatic classification
            classification, confidence = self.classify_image(image_path)
            
            if classification:
                print(f"\nSuggested classification: {classification} (confidence: {confidence:.3f})")
            else:
                print("\nNo automatic classification available")
            
            # Get user input
            print("\nOptions:")
            print("1. Accept suggestion")
            print("2. Enter manual classification")
            print("3. Skip")
            print("4. Exit")
            
            choice = input("\nYour choice (1-4): ").strip()
            
            if choice == '1' and classification:
                # Apply classification
                self._apply_classification(image_path, classification)
            elif choice == '2':
                manual_class = input("Enter classification: ").strip()
                if manual_class:
                    self._apply_classification(image_path, manual_class)
            elif choice == '3':
                print("Skipped")
            elif choice == '4':
                print("Exiting manual mode")
                break
    
    def _apply_classification(self, image_path, classification):
        """Apply classification to image"""
        try:
            extension = Path(image_path).suffix
            directory = os.path.dirname(image_path)
            
            # Clean classification for filename
            clean_classification = classification.replace('/', '-').replace('\\', '-')
            new_filename = f"{clean_classification}{extension}"
            
            # Ensure unique
            counter = 1
            new_path = os.path.join(directory, new_filename)
            while os.path.exists(new_path):
                new_filename = f"{clean_classification}_{counter}{extension}"
                new_path = os.path.join(directory, new_filename)
                counter += 1
            
            os.rename(image_path, new_path)
            print(f"✓ Renamed to: {new_filename}")
            
        except Exception as e:
            print(f"✗ Error applying classification: {e}")

def main():
    print("="*60)
    print("SIMPLE IMAGE CLASSIFIER - WORKING VERSION")
    print("="*60)
    
    classifier = SimpleImageClassifier()
    
    # Check folders exist
    if not os.path.exists(classifier.reference_folder):
        print(f"ERROR: Reference folder '{classifier.reference_folder}' not found!")
        return
    
    if not os.path.exists(classifier.dataset_folder):
        print(f"ERROR: Dataset folder '{classifier.dataset_folder}' not found!")
        return
    
    # Load reference images
    classifier.load_reference_images()
    
    if not classifier.reference_data:
        print("ERROR: No reference images loaded!")
        return
    
    # Mode selection
    print("\nSelect mode:")
    print("1. Automatic (process all images)")
    print("2. Manual (interactive)")
    
    mode = input("\nYour choice (1 or 2): ").strip()
    
    if mode == '1':
        classifier.process_dataset()
    elif mode == '2':
        classifier.manual_mode()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()