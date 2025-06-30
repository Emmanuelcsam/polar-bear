#!/usr/bin/env python3
"""
Simple Image Similarity Clustering Script
A lightweight version that uses only perceptual hashing and color histograms
"""

import os
import shutil
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import imagehash
from PIL import Image
from collections import defaultdict

class SimpleImageClusterer:
    def __init__(self, input_dir, output_dir=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or f"{input_dir}_clustered"
        self.images_data = []
        
    def load_images(self):
        """Load all images from directory"""
        print(f"Loading images from {self.input_dir}...")
        
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                filepath = os.path.join(self.input_dir, file)
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        self.images_data.append({
                            'path': filepath,
                            'filename': file,
                            'image': img
                        })
                except:
                    pass
        
        print(f"Found {len(self.images_data)} images")
        
    def extract_features(self):
        """Extract simple features from images"""
        print("Extracting features...")
        features = []
        
        for img_data in self.images_data:
            img = img_data['image']
            
            # 1. Color histogram features
            hist_features = []
            for i in range(3):  # BGR channels
                hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                hist_features.extend(hist.flatten())
            
            # 2. Perceptual hash
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            phash = str(imagehash.phash(pil_img))
            dhash = str(imagehash.dhash(pil_img))
            
            # Convert hashes to binary features
            hash_features = []
            for h in [phash, dhash]:
                binary = bin(int(h, 16))[2:].zfill(64)
                hash_features.extend([int(b) for b in binary])
            
            # 3. Image statistics
            resized = cv2.resize(img, (32, 32))
            stats = [
                np.mean(resized),
                np.std(resized),
                np.median(resized)
            ]
            
            # Combine all features
            all_features = hist_features + hash_features + stats
            features.append(all_features)
        
        # Normalize features
        scaler = StandardScaler()
        return scaler.fit_transform(np.array(features))
    
    def cluster_images(self, features):
        """Cluster images using DBSCAN"""
        print("Clustering images...")
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=2)
        labels = clustering.fit_predict(features)
        
        # Add cluster labels to image data
        for i, label in enumerate(labels):
            self.images_data[i]['cluster'] = label
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        
        print(f"Found {n_clusters} clusters and {n_outliers} outliers")
        return labels
    
    def organize_images(self):
        """Organize images into cluster directories"""
        print(f"Organizing images into {self.output_dir}...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Group by cluster
        clusters = defaultdict(list)
        for img_data in self.images_data:
            clusters[img_data['cluster']].append(img_data)
        
        # Create directories and copy images
        for cluster_id, images in clusters.items():
            if cluster_id == -1:
                cluster_dir = os.path.join(self.output_dir, 'unique_images')
            else:
                cluster_dir = os.path.join(self.output_dir, f'group_{cluster_id + 1}')
            
            os.makedirs(cluster_dir, exist_ok=True)
            
            for img_data in images:
                src = img_data['path']
                dst = os.path.join(cluster_dir, img_data['filename'])
                
                # Handle duplicates
                if os.path.exists(dst):
                    name, ext = os.path.splitext(img_data['filename'])
                    i = 1
                    while os.path.exists(os.path.join(cluster_dir, f"{name}_{i}{ext}")):
                        i += 1
                    dst = os.path.join(cluster_dir, f"{name}_{i}{ext}")
                
                shutil.copy2(src, dst)
        
        # Create summary
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Image Clustering Summary\n")
            f.write("="*30 + "\n\n")
            f.write(f"Total images: {len(self.images_data)}\n")
            f.write(f"Groups created: {len(clusters) - (1 if -1 in clusters else 0)}\n")
            f.write(f"Unique images: {len(clusters.get(-1, []))}\n\n")
            
            for cluster_id, images in sorted(clusters.items()):
                if cluster_id == -1:
                    f.write(f"Unique images: {len(images)} files\n")
                else:
                    f.write(f"Group {cluster_id + 1}: {len(images)} files\n")
        
        print(f"Done! Check {self.output_dir} for results")
        print(f"Summary saved to {summary_path}")
    
    def run(self):
        """Run the clustering pipeline"""
        self.load_images()
        if not self.images_data:
            print("No images found!")
            return
            
        features = self.extract_features()
        self.cluster_images(features)
        self.organize_images()


def main():
    print("\n=== Simple Image Clustering Tool ===\n")
    
    # Get input directory
    while True:
        input_dir = input("Enter path to your images folder: ").strip()
        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            break
        print("Invalid directory, please try again.")
    
    # Get output directory
    default_output = f"{input_dir}_clustered"
    output_dir = input(f"Output folder [{default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output
    
    # Run clustering
    print("\nProcessing...")
    clusterer = SimpleImageClusterer(input_dir, output_dir)
    clusterer.run()


if __name__ == "__main__":
    main()
