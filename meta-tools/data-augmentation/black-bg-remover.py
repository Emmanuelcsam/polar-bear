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

class BlackBackgroundRemover:
    def __init__(self, black_threshold=30):
        """
        Initialize the black background remover
        
        Args:
            black_threshold: Maximum RGB value to consider as black (default: 30)
                           Pixels with all RGB values below this are considered black
        """
        self.black_threshold = black_threshold
        logger.info(f"Initialized with black threshold: {black_threshold}")
    
    def remove_black_background(self, img_path):
        """Remove black background from an image"""
        try:
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img.convert('RGB'))
            
            # Create mask where non-black pixels are True
            # A pixel is considered black if all its RGB values are below threshold
            mask = np.any(img_array > self.black_threshold, axis=2)
            
            # Apply some morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # Remove small noise
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            
            # Fill small holes
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # Optionally remove small disconnected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
            if num_labels > 1:
                # Calculate minimum area threshold (1% of image area)
                min_area = 0.01 * img_array.shape[0] * img_array.shape[1]
                
                # Create new mask keeping only components above threshold
                new_mask = np.zeros_like(mask_uint8)
                for i in range(1, num_labels):  # Skip background (label 0)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        new_mask[labels == i] = 255
                mask_uint8 = new_mask
            
            # Convert mask to alpha channel
            alpha_array = mask_uint8.astype(np.uint8)
            
            # Create RGBA image
            rgba_array = np.dstack((img_array, alpha_array))
            output_image = Image.fromarray(rgba_array, 'RGBA')
            
            # Check if mask is valid
            if np.all(alpha_array == 0):
                logger.warning(f"Empty mask for {img_path} - entire image was considered black")
                return None
                
            return output_image
            
        except Exception as e:
            logger.error(f"Error removing black background from {img_path}: {e}")
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
            # Remove black background
            result = self.remove_black_background(input_path)
            
            if result is not None:
                # Save as PNG to preserve transparency
                result.save(output_path, 'PNG', optimize=True)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

def main():
    print("=== Black Background Remover ===\n")
    
    # Get input directory
    print("Enter the path to the directory containing images to process:")
    input_dir = input().strip()
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    # Get output directory
    print("\nEnter the path to the output directory:")
    output_dir = input().strip()
    
    # Ask for black threshold
    print("\nEnter the black threshold (0-255, default: 30):")
    print("Pixels with all RGB values below this will be considered black")
    threshold_input = input().strip()
    
    if threshold_input.isdigit():
        threshold = int(threshold_input)
        threshold = max(0, min(255, threshold))  # Clamp to valid range
    else:
        threshold = 30
    
    # Create remover instance
    remover = BlackBackgroundRemover(black_threshold=threshold)
    
    # Ask for number of workers
    print("\nEnter number of parallel workers (default: 4):")
    workers_input = input().strip()
    num_workers = int(workers_input) if workers_input.isdigit() else 4
    
    # Process the directory
    print(f"\nProcessing images with {num_workers} workers...")
    remover.process_directory(input_dir, output_dir, num_workers)
    
    print("\nDone! Check the output directory for results.")

if __name__ == "__main__":
    main()