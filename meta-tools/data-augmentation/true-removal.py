import cv2
import numpy as np
from PIL import Image
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')  # Configure logging to show INFO level messages with timestamps
logger = logging.getLogger(__name__)  # Create a logger instance for this module

class ReferenceCropper:
    def __init__(self, reference_image_path):
        """Initialize with a reference image for cropping"""
        self.reference_image_path = reference_image_path  # Store the path to the reference image
        self.reference_features = self.extract_reference_features(reference_image_path)  # Extract feature data from reference image for comparison
        self.reference_mask = self.create_reference_mask(reference_image_path)  # Create a binary mask representing the object shape in reference
        
        if self.reference_features is None or self.reference_mask is None:  # Check if feature extraction or mask creation failed
            raise ValueError("Failed to process reference image")  # Raise error if reference processing failed
            
        logger.info(f"Loaded reference image: {reference_image_path}")  # Log successful loading of reference image
        logger.info(f"Reference mask brightness: {np.mean(self.reference_mask):.2f}")  # Log average brightness of the mask (0=dark object, 1=bright object)
    
    def extract_reference_features(self, img_path):
        """Extract features from reference image for similarity matching"""
        try:
            img = cv2.imread(img_path)  # Load image using OpenCV (returns BGR format)
            if img is None:  # Check if image loading failed
                return None
                
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert BGR image to single-channel grayscale
            
            # Extract various features
            features = {}  # Initialize dictionary to store all extracted features
            
            # Color histograms
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])  # Calculate histogram for blue channel (256 bins)
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])  # Calculate histogram for green channel
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])  # Calculate histogram for red channel
            features['color_hist'] = np.concatenate([hist_b, hist_g, hist_r]).flatten()  # Combine all color histograms into single feature vector
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)  # Apply Canny edge detector with low threshold=50, high threshold=150
            features['edge_density'] = np.sum(edges) / (edges.shape[0] * edges.shape[1])  # Calculate ratio of edge pixels to total pixels
            
            # Contour features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find external contours from edge image
            if contours:  # Check if any contours were found
                largest_contour = max(contours, key=cv2.contourArea)  # Find contour with maximum area
                features['contour_area'] = cv2.contourArea(largest_contour)  # Calculate area of largest contour
                features['contour_perimeter'] = cv2.arcLength(largest_contour, True)  # Calculate perimeter of largest contour (closed curve)
            else:
                features['contour_area'] = 0  # Set area to 0 if no contours found
                features['contour_perimeter'] = 0  # Set perimeter to 0 if no contours found
                
            return features  # Return dictionary containing all extracted features
        except Exception as e:
            logger.error(f"Error extracting reference features: {e}")  # Log any errors that occur during feature extraction
            return None  # Return None if feature extraction fails
    
    def create_reference_mask(self, img_path):
        """Create a mask from reference image (assuming it's already cropped/processed)"""
        try:
            img = Image.open(img_path)  # Load image using PIL (supports more formats than OpenCV)
            if img.mode == 'RGBA':  # Check if image has an alpha (transparency) channel
                # Use alpha channel as mask
                alpha = np.array(img)[:, :, 3]  # Extract alpha channel (4th channel, index 3)
                return alpha > 0  # Convert to boolean mask where True=visible, False=transparent
            else:
                # For RGB images, create mask based on background detection
                img_array = np.array(img)  # Convert PIL image to numpy array
                # Simple background detection - assume corners are background
                corner_samples = [  # Sample pixel values from all four corners
                    img_array[0, 0],     # Top-left corner pixel
                    img_array[0, -1],    # Top-right corner pixel  
                    img_array[-1, 0],    # Bottom-left corner pixel
                    img_array[-1, -1]    # Bottom-right corner pixel
                ]
                bg_color = np.mean(corner_samples, axis=0)  # Calculate average background color from corner samples
                
                # Create mask where pixels are significantly different from background
                diff = np.linalg.norm(img_array - bg_color, axis=2)  # Calculate Euclidean distance from each pixel to background color
                threshold = np.std(diff) * 2  # Set threshold as 2 standard deviations of the differences
                return diff > threshold  # Return boolean mask where True=foreground object, False=background
        except Exception as e:
            logger.error(f"Error creating reference mask: {e}")  # Log any errors during mask creation
            return None  # Return None if mask creation fails
    
    def apply_reference_based_crop(self, img_path):
        """Apply cropping based on reference image similarity"""
        try:
            # Extract features from current image
            current_features = self.extract_reference_features(img_path)  # Extract same feature set from current image for comparison
            if current_features is None:  # Check if feature extraction failed
                return None
                
            # Load current image
            img = Image.open(img_path)  # Load current image using PIL
            img_array = np.array(img.convert('RGB'))  # Convert to RGB numpy array (removes alpha if present)
            
            # Compare color histograms
            ref_hist = self.reference_features['color_hist']  # Get reference image color histogram
            curr_hist = current_features['color_hist']  # Get current image color histogram
            hist_similarity = cv2.compareHist(  # Calculate correlation coefficient between histograms
                ref_hist.astype(np.float32),   # Convert to float32 for OpenCV compatibility
                curr_hist.astype(np.float32),  # Convert to float32 for OpenCV compatibility
                cv2.HISTCMP_CORREL  # Use correlation method (returns value between 0 and 1)
            )
            
            # Create mask based on color similarity and reference mask pattern
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert current image to grayscale for processing
            
            # Use adaptive thresholding and morphological operations
            if self.reference_mask is not None:  # Check if reference mask was created successfully
                # Analyze reference mask properties
                ref_is_bright = np.mean(self.reference_mask) > 0.5  # Determine if reference object is bright (>50% mask area is True)
                
                if ref_is_bright:  # If reference object is bright
                    # Looking for bright objects
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Use Otsu's method to find optimal threshold for bright objects
                else:
                    # Looking for dark objects  
                    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Use inverted Otsu's method for dark objects
            else:
                # Default: use edge-based segmentation
                edges = cv2.Canny(gray, 50, 150)  # Apply Canny edge detection as fallback method
                mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)  # Dilate edges to make them thicker and more connected
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10), np.uint8))  # Close gaps in the edges to form solid regions
            
            # Clean up the mask with morphological operations
            kernel = np.ones((3,3), np.uint8)  # Create 3x3 structuring element for morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise pixels (erosion followed by dilation)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes in objects (dilation followed by erosion)
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)  # Find all connected components in the mask
            if num_labels > 1:  # Check if multiple components found (background counts as component 0)
                # Keep only the largest component (excluding background)
                largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Find component with largest area (skip background at index 0)
                mask = (labels == largest_component).astype(np.uint8) * 255  # Create new mask containing only the largest component
            
            # Convert mask to alpha channel
            alpha_array = mask.astype(np.uint8)  # Ensure mask is uint8 format for alpha channel
            
            # Create RGBA image
            rgba_array = np.dstack((img_array, alpha_array))  # Stack RGB channels with alpha channel to create RGBA array
            output_image = Image.fromarray(rgba_array, 'RGBA')  # Convert numpy array back to PIL Image with RGBA mode
            
            # Check if mask is valid
            if np.all(alpha_array == 0):  # Check if entire mask is black (no object detected)
                logger.warning(f"Empty mask for {img_path}")  # Log warning for failed object detection
                return None  # Return None for empty masks
                
            return output_image  # Return the processed RGBA image with transparency mask
            
        except Exception as e:
            logger.error(f"Error applying reference-based crop to {img_path}: {e}")  # Log any errors during processing
            return None  # Return None if processing fails
    
    def process_directory(self, input_dir, output_dir, num_workers=4):
        """Process all images in input directory and save to output directory"""
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')  # Define supported image file extensions
        image_files = [  # Create list of image files in input directory
            f for f in os.listdir(input_dir)   # Iterate through all files in input directory
            if f.lower().endswith(image_extensions)  # Keep only files with supported image extensions (case-insensitive)
        ]
        
        if not image_files:  # Check if no image files were found
            logger.error("No images found in input directory")  # Log error message
            return  # Exit function early if no images to process
        
        logger.info(f"Found {len(image_files)} images to process")  # Log number of images found for processing
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)  # Create output directory structure (exist_ok prevents error if already exists)
        
        # Process images in parallel
        successful = 0  # Initialize counter for successfully processed images
        failed = 0  # Initialize counter for failed image processing attempts
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Create thread pool for parallel processing
            # Submit all tasks
            future_to_file = {  # Create dictionary mapping futures to filenames
                executor.submit(  # Submit processing task to thread pool
                    self.process_single_image,   # Function to execute in parallel
                    os.path.join(input_dir, filename),   # Full path to input image
                    os.path.join(output_dir, filename)   # Full path to output image
                ): filename   # Map future object to filename for tracking
                for filename in image_files  # Create one task for each image file
            }
            
            # Process results with progress bar
            with tqdm(total=len(image_files), desc="Processing images") as pbar:  # Create progress bar with total count
                for future in as_completed(future_to_file):  # Iterate through completed futures as they finish
                    filename = future_to_file[future]  # Get filename associated with completed future
                    try:
                        result = future.result()  # Get result from completed future (blocks until complete)
                        if result:  # Check if processing was successful
                            successful += 1  # Increment success counter
                        else:
                            failed += 1  # Increment failure counter
                            logger.warning(f"Failed to process {filename}")  # Log warning for failed processing
                    except Exception as e:
                        failed += 1  # Increment failure counter for exception
                        logger.error(f"Error processing {filename}: {e}")  # Log error with exception details
                    pbar.update(1)  # Update progress bar by one completed item
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")  # Log final processing statistics
    
    def process_single_image(self, input_path, output_path):
        """Process a single image and save the result"""
        try:
            # Apply reference-based cropping
            result = self.apply_reference_based_crop(input_path)  # Apply the main cropping algorithm to input image
            
            if result is not None:  # Check if cropping was successful
                # Save as PNG to preserve transparency
                result.save(output_path, 'PNG', optimize=True)  # Save processed image as PNG with optimization enabled
                return True  # Return True to indicate successful processing
            return False  # Return False if cropping failed
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")  # Log any errors that occur during processing
            return False  # Return False if exception occurred


def main():
    print("=== Reference-Based Batch Image Cropper ===\n")  # Display program title
    
    # Get input directory
    print("Enter the path to the directory containing images to process:")  # Prompt user for input directory
    input_dir = input().strip()  # Read user input and remove leading/trailing whitespace
    
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):  # Validate that input path exists and is a directory
        print("Error: Invalid input directory")  # Display error message for invalid directory
        sys.exit(1)  # Exit program with error code 1
    
    # Get reference image
    print("\nEnter the path to the reference cropped image:")  # Prompt user for reference image path
    reference_path = input().strip()  # Read user input and remove leading/trailing whitespace
    
    if not os.path.exists(reference_path) or not os.path.isfile(reference_path):  # Validate that reference path exists and is a file
        print("Error: Invalid reference image path")  # Display error message for invalid file
        sys.exit(1)  # Exit program with error code 1
    
    # Get output directory
    print("\nEnter the path to the output directory:")  # Prompt user for output directory path
    output_dir = input().strip()  # Read user input and remove leading/trailing whitespace
    
    # Create cropper instance
    try:
        cropper = ReferenceCropper(reference_path)  # Initialize cropper with reference image
    except ValueError as e:  # Catch ValueError from failed reference processing
        print(f"Error: {e}")  # Display the error message
        sys.exit(1)  # Exit program with error code 1
    
    # Ask for number of workers
    print("\nEnter number of parallel workers (default: 4):")  # Prompt user for thread count
    workers_input = input().strip()  # Read user input and remove leading/trailing whitespace
    num_workers = int(workers_input) if workers_input.isdigit() else 4  # Convert to int if valid digit, otherwise use default 4
    
    # Process the directory
    print(f"\nProcessing images with {num_workers} workers...")  # Display processing start message with worker count
    cropper.process_directory(input_dir, output_dir, num_workers)  # Execute batch processing with specified parameters
    
    print("\nDone! Check the output directory for results.")  # Display completion message


if __name__ == "__main__":
    main()  # Execute main function only when script is run directly (not when imported as module)
