#!/usr/bin/env python3
"""
 Image Similarity Clustering Script
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
        self.input_dir = input_dir  # Store the source directory containing images to process
        self.output_dir = output_dir or f"{input_dir}_clustered"  # Set output directory, defaulting to input_dir + "_clustered" suffix
        self.images_data = []  # Initialize empty list to store image metadata and processed data
        
    def load_images(self):
        """Load all images from directory"""
        print(f"Loading images from {self.input_dir}...")  # Display progress message to user
        
        for file in os.listdir(self.input_dir):  # Iterate through each file in the input directory
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Filter for supported image file extensions
                filepath = os.path.join(self.input_dir, file)  # Create full path by joining directory and filename
                try:
                    img = cv2.imread(filepath)  # Load image using OpenCV (returns BGR format)
                    if img is not None:  # Check if image was successfully loaded (None indicates failure)
                        self.images_data.append({  # Add image data to our collection as dictionary
                            'path': filepath,  # Store original file path for later file operations
                            'filename': file,  # Store just the filename for output organization
                            'image': img  # Store the actual image array for feature extraction
                        })
                except:  # Catch any file reading errors without stopping the entire process
                    pass
        
        print(f"Found {len(self.images_data)} images")  # Report how many valid images were successfully loaded
        
    def extract_features(self):
        """Extract simple features from images"""
        print("Extracting features...")  # Inform user about current processing stage
        features = []  # Initialize list to store feature vectors for all images
        
        for img_data in self.images_data:  # Process each loaded image to extract its characteristics
            img = img_data['image']  # Get the image array from our stored data
            
            # 1. Color histogram features
            hist_features = []  # Initialize list for color distribution features
            for i in range(3):  # Loop through BGR color channels (Blue, Green, Red)
                hist = cv2.calcHist([img], [i], None, [32], [0, 256])  # Calculate histogram with 32 bins for pixel intensity distribution
                hist_features.extend(hist.flatten())  # Convert 2D histogram to 1D and add to feature list
            
            # 2. Perceptual hash
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB and create PIL Image for hash calculation
            phash = str(imagehash.phash(pil_img))  # Generate perceptual hash (structural similarity)
            dhash = str(imagehash.dhash(pil_img))  # Generate difference hash (gradient-based similarity)
            
            # Convert hashes to binary features
            hash_features = []  # Initialize list for hash-based binary features
            for h in [phash, dhash]:  # Process both hash types
                binary = bin(int(h, 16))[2:].zfill(64)  # Convert hexadecimal hash to 64-bit binary string
                hash_features.extend([int(b) for b in binary])  # Convert each binary digit to integer and add to features
            
            # 3. Image statistics
            resized = cv2.resize(img, (32, 32))  # Resize image to standard 32x32 for consistent statistical analysis
            stats = [  # Calculate basic statistical measures of pixel intensities
                np.mean(resized),  # Average pixel intensity across all channels
                np.std(resized),   # Standard deviation (measure of contrast/variation)
                np.median(resized) # Median pixel intensity (robust central tendency)
            ]
            
            # Combine all features
            all_features = hist_features + hash_features + stats  # Concatenate all feature types into single vector
            features.append(all_features)  # Add this image's complete feature vector to collection
        
        # Normalize features
        scaler = StandardScaler()  # Create scaler to normalize features (mean=0, std=1)
        return scaler.fit_transform(np.array(features))  # Convert list to numpy array, fit scaler, and return normalized features
    
    def cluster_images(self, features):
        """Cluster images using DBSCAN"""
        print("Clustering images...")  # Update user on current processing step
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=2)  # Create density-based clustering algorithm (eps=neighborhood size, min_samples=core point threshold)
        labels = clustering.fit_predict(features)  # Train on feature data and predict cluster labels for each image
        
        # Add cluster labels to image data
        for i, label in enumerate(labels):  # Iterate through cluster assignments with index
            self.images_data[i]['cluster'] = label  # Store cluster ID in each image's metadata dictionary
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Count unique clusters (excluding -1 which represents outliers)
        n_outliers = list(labels).count(-1)  # Count images classified as outliers/noise
        
        print(f"Found {n_clusters} clusters and {n_outliers} outliers")  # Report clustering results to user
        return labels  # Return cluster labels array for potential further use
    
    def organize_images(self):
        """Organize images into cluster directories"""
        print(f"Organizing images into {self.output_dir}...")  # Inform user about file organization process
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)  # Create main output folder, don't error if it already exists
        
        # Group by cluster
        clusters = defaultdict(list)  # Create dictionary that automatically creates empty lists for new keys
        for img_data in self.images_data:  # Iterate through all processed images
            clusters[img_data['cluster']].append(img_data)  # Group images by their assigned cluster ID
        
        # Create directories and copy images
        for cluster_id, images in clusters.items():  # Process each cluster group
            if cluster_id == -1:  # Handle outlier images (DBSCAN assigns -1 to noise points)
                cluster_dir = os.path.join(self.output_dir, 'unique_images')  # Create special folder for outliers
            else:
                cluster_dir = os.path.join(self.output_dir, f'group_{cluster_id + 1}')  # Create numbered group folders (add 1 for human-readable numbering)
            
            os.makedirs(cluster_dir, exist_ok=True)  # Create the cluster subdirectory
            
            for img_data in images:  # Process each image in the current cluster
                src = img_data['path']  # Get original file location
                dst = os.path.join(cluster_dir, img_data['filename'])  # Build destination path in cluster folder
                
                # Handle duplicates
                if os.path.exists(dst):  # Check if filename already exists in destination
                    name, ext = os.path.splitext(img_data['filename'])  # Split filename into name and extension
                    i = 1  # Start counter for duplicate naming
                    while os.path.exists(os.path.join(cluster_dir, f"{name}_{i}{ext}")):  # Find available numbered filename
                        i += 1  # Increment counter until unique name found
                    dst = os.path.join(cluster_dir, f"{name}_{i}{ext}")  # Set final destination with unique numbered name
                
                shutil.copy2(src, dst)  # Copy file preserving metadata (timestamps, permissions)
        
        # Create summary
        summary_path = os.path.join(self.output_dir, 'summary.txt')  # Build path for summary report file
        with open(summary_path, 'w') as f:  # Open summary file for writing
            f.write("Image Clustering Summary\n")  # Write header for the summary report
            f.write("="*30 + "\n\n")  # Create visual separator line using repeated equals signs
            f.write(f"Total images: {len(self.images_data)}\n")  # Report total number of images processed
            f.write(f"Groups created: {len(clusters) - (1 if -1 in clusters else 0)}\n")  # Count actual groups (excluding outliers)
            f.write(f"Unique images: {len(clusters.get(-1, []))}\n\n")  # Count outlier images using safe dictionary access
            
            for cluster_id, images in sorted(clusters.items()):  # Iterate through clusters in sorted order for consistent output
                if cluster_id == -1:  # Special handling for outlier group
                    f.write(f"Unique images: {len(images)} files\n")  # Report outlier count
                else:
                    f.write(f"Group {cluster_id + 1}: {len(images)} files\n")  # Report each cluster size with human-readable numbering
        
        print(f"Done! Check {self.output_dir} for results")  # Notify user of completion and where to find results
        print(f"Summary saved to {summary_path}")  # Specifically mention summary file location
    
    def run(self):
        """Run the clustering pipeline"""
        self.load_images()  # Load and validate all image files from input directory
        if not self.images_data:  # Check if any valid images were found
            print("No images found!")  # Alert user if no processable images exist
            return  # Exit early if no data to process
            
        features = self.extract_features()  # Extract feature vectors from all loaded images
        self.cluster_images(features)  # Apply clustering algorithm to group similar images
        self.organize_images()  # Copy images into organized folder structure based on clusters


def main():
    print("\n=== Simple Image Clustering Tool ===\n")  # Display application banner to user
    
    # Get input directory
    while True:  # Continue prompting until valid directory is provided
        input_dir = input("Enter path to your images folder: ").strip()  # Get user input and remove whitespace
        if os.path.exists(input_dir) and os.path.isdir(input_dir):  # Validate that path exists and is a directory
            break  # Exit loop when valid directory found
        print("Invalid directory, please try again.")  # Error message for invalid input
    
    # Get output directory
    default_output = f"{input_dir}_clustered"  # Create default output directory name by appending suffix
    output_dir = input(f"Output folder [{default_output}]: ").strip()  # Prompt for output directory with default shown
    if not output_dir:  # Check if user pressed enter without typing anything
        output_dir = default_output  # Use default output directory if none specified
    
    # Run clustering
    print("\nProcessing...")  # Inform user that processing is starting
    clusterer = SimpleImageClusterer(input_dir, output_dir)  # Create clusterer instance with user-specified directories
    clusterer.run()  # Execute the complete clustering pipeline


if __name__ == "__main__":
    main()  # Execute main function only when script is run directly (not imported as module)
