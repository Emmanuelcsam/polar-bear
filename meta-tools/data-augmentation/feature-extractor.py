import cv2
import numpy as np
from PIL import Image
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ReferenceCropper:
    def __init__(self, reference_image_path, debug=False):
        """Initialize with a reference image for cropping"""
        self.debug = debug
        self.reference_image_path = reference_image_path
        self.reference_features = self.extract_reference_features(reference_image_path)
        self.reference_mask = self.create_reference_mask(reference_image_path)
        
        if self.reference_features is None or self.reference_mask is None:
            raise ValueError("Failed to process reference image")
            
        logger.info(f"Loaded reference image: {reference_image_path}")
        logger.info(f"Reference mask brightness: {np.mean(self.reference_mask):.2f}")
        
        # Save debug image of reference mask if debug mode
        if self.debug and self.reference_mask is not None:
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray((self.reference_mask * 255).astype(np.uint8)).save(
                os.path.join(debug_dir, "reference_mask.png")
            )
    
    def extract_reference_features(self, img_path):
        """Extract features from reference image for similarity matching"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract various features
            features = {}
            
            # Color histograms
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            features['color_hist'] = np.concatenate([hist_b, hist_g, hist_r]).flatten()
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            
            # Contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                features['contour_area'] = cv2.contourArea(largest_contour)
                features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            else:
                features['contour_area'] = 0
                features['contour_perimeter'] = 0
                
            return features
        except Exception as e:
            logger.error(f"Error extracting reference features: {e}")
            return None
    
    def create_reference_mask(self, img_path):
        """Create a mask from reference image (assuming it's already cropped/processed)"""
        try:
            img = Image.open(img_path)
            if img.mode == 'RGBA':
                # Use alpha channel as mask
                alpha = np.array(img)[:, :, 3]
                return alpha > 0
            else:
                # For RGB images, create mask based on background detection
                img_array = np.array(img)
                # Simple background detection - assume corners are background
                corner_samples = [
                    img_array[0, 0],
                    img_array[0, -1], 
                    img_array[-1, 0],
                    img_array[-1, -1]
                ]
                bg_color = np.mean(corner_samples, axis=0)
                
                # Create mask where pixels are significantly different from background
                diff = np.linalg.norm(img_array - bg_color, axis=2)
                threshold = np.std(diff) * 2
                return diff > threshold
        except Exception as e:
            logger.error(f"Error creating reference mask: {e}")
            return None
    
    def detect_donut_shape(self, binary_mask):
        """Detect and create mask for donut-shaped objects"""
        # Find all contours
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0 or hierarchy is None:
            return binary_mask
        
        hierarchy = hierarchy[0]
        result_mask = np.zeros_like(binary_mask)
        
        # Find contours that have a parent (i.e., they are holes)
        holes = []
        for i in range(len(contours)):
            if hierarchy[i][3] != -1:  # Has a parent
                holes.append(i)
        
        # Find external contours
        external_contours = []
        for i in range(len(contours)):
            if hierarchy[i][3] == -1:  # No parent
                area = cv2.contourArea(contours[i])
                external_contours.append((i, area))
        
        if not external_contours:
            return binary_mask
        
        # Sort by area and get the largest
        external_contours.sort(key=lambda x: x[1], reverse=True)
        main_idx = external_contours[0][0]
        
        # Draw the main contour
        cv2.drawContours(result_mask, [contours[main_idx]], -1, 255, -1)
        
        # Find holes that belong to the main contour
        for hole_idx in holes:
            # Check if this hole's parent is the main contour
            parent_idx = hierarchy[hole_idx][3]
            if parent_idx == main_idx:
                # This is a direct hole of the main shape
                cv2.drawContours(result_mask, [contours[hole_idx]], -1, 0, -1)
        
        return result_mask
    
    def apply_reference_based_crop(self, img_path):
        """Apply cropping based on reference image similarity"""
        try:
            # Extract features from current image
            current_features = self.extract_reference_features(img_path)
            if current_features is None:
                return None
                
            # Load current image
            img = Image.open(img_path)
            img_array = np.array(img.convert('RGB'))
            
            # Compare color histograms
            ref_hist = self.reference_features['color_hist']
            curr_hist = current_features['color_hist']
            hist_similarity = cv2.compareHist(
                ref_hist.astype(np.float32), 
                curr_hist.astype(np.float32), 
                cv2.HISTCMP_CORREL
            )
            
            # Create mask based on color similarity and reference mask pattern
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Determine if we should look for dark or light objects
            # Calculate the average intensity of the object in the reference
            if self.reference_mask is not None:
                ref_is_bright = np.mean(self.reference_mask) > 0.5
            else:
                ref_is_bright = False
            
            # Get background color (average of corners)
            h, w = gray.shape
            corners = [gray[0,0], gray[0,w-1], gray[h-1,0], gray[h-1,w-1]]
            bg_intensity = np.mean(corners)
            
            # Create initial mask using multiple methods
            masks = []
            
            # Method 1: OTSU thresholding
            _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            masks.append(otsu_mask)
            
            # Method 2: Adaptive thresholding
            adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
            masks.append(adaptive_mask)
            
            # Method 3: Background subtraction
            if bg_intensity > 127:  # Light background
                _, bg_mask = cv2.threshold(gray, bg_intensity - 30, 255, cv2.THRESH_BINARY_INV)
            else:  # Dark background
                _, bg_mask = cv2.threshold(gray, bg_intensity + 30, 255, cv2.THRESH_BINARY)
            masks.append(bg_mask)
            
            # Combine masks using voting
            combined_mask = np.zeros_like(gray)
            for mask in masks:
                combined_mask = combined_mask + (mask > 0).astype(np.uint8)
            
            # Threshold: at least 2 out of 3 methods should agree
            mask = (combined_mask >= 2).astype(np.uint8) * 255
            
            # Clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply donut detection
            mask = self.detect_donut_shape(mask)
            
            # Save debug images if enabled
            if self.debug:
                debug_dir = "debug_output"
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.basename(img_path).split('.')[0]
                
                # Save intermediate masks
                for i, m in enumerate(masks):
                    Image.fromarray(m).save(
                        os.path.join(debug_dir, f"{base_name}_mask_{i}.png")
                    )
                
                # Save combined mask before donut detection
                Image.fromarray(combined_mask * 85).save(
                    os.path.join(debug_dir, f"{base_name}_combined.png")
                )
                
                # Save final mask
                Image.fromarray(mask).save(
                    os.path.join(debug_dir, f"{base_name}_final_mask.png")
                )
            
            # Convert mask to alpha channel
            alpha_array = mask.astype(np.uint8)
            
            # Create RGBA image
            rgba_array = np.dstack((img_array, alpha_array))
            output_image = Image.fromarray(rgba_array, 'RGBA')
            
            # Check if mask is valid
            if np.all(alpha_array == 0):
                logger.warning(f"Empty mask for {img_path}")
                return None
                
            return output_image
            
        except Exception as e:
            logger.error(f"Error applying reference-based crop to {img_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir, num_workers=4):
        """Process all images in input directory and save to output directory"""
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            logger.error("No images found in input directory")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images in parallel
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self.process_single_image, 
                    os.path.join(input_dir, filename),
                    os.path.join(output_dir, filename)
                ): filename 
                for filename in image_files
            }
            
            # Process results with progress bar
            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                        else:
                            failed += 1
                            logger.warning(f"Failed to process {filename}")
                    except Exception as e:
                        failed += 1
                        logger.error(f"Error processing {filename}: {e}")
                    pbar.update(1)
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    def process_single_image(self, input_path, output_path):
        """Process a single image and save the result"""
        try:
            # Apply reference-based cropping
            result = self.apply_reference_based_crop(input_path)
            
            if result is not None:
                # Save as PNG to preserve transparency
                result.save(output_path, 'PNG', optimize=True)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False


def main():
    print("=== Reference-Based Batch Image Cropper ===\n")
    
    # Get input directory
    print("Enter the path to the directory containing images to process:")
    input_dir = input().strip()
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    # Get reference image
    print("\nEnter the path to the reference cropped image:")
    reference_path = input().strip()
    
    if not os.path.exists(reference_path) or not os.path.isfile(reference_path):
        print("Error: Invalid reference image path")
        sys.exit(1)
    
    # Get output directory
    print("\nEnter the path to the output directory:")
    output_dir = input().strip()
    
    # Ask about debug mode
    print("\nEnable debug mode? (y/n, default: n):")
    debug_input = input().strip().lower()
    debug_mode = debug_input == 'y'
    
    # Create cropper instance
    try:
        cropper = ReferenceCropper(reference_path, debug=debug_mode)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Ask for number of workers
    print("\nEnter number of parallel workers (default: 4):")
    workers_input = input().strip()
    num_workers = int(workers_input) if workers_input.isdigit() else 4
    
    # Process the directory
    print(f"\nProcessing images with {num_workers} workers...")
    if debug_mode:
        print("Debug images will be saved to 'debug_output' directory")
    
    cropper.process_directory(input_dir, output_dir, num_workers)
    
    print("\nDone! Check the output directory for results.")


if __name__ == "__main__":
    main()