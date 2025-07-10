#!/usr/bin/env python3
"""
Script to overlay scratches from BMP files onto clean fiber optic images.
The scratches are extracted from dark background BMP files and overlaid onto PNG/JPG images.
"""

import sys
import subprocess
import importlib.util

def check_and_install_dependencies():
    """Check for required dependencies and install them if missing."""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (yes/no): ").lower().strip()
        
        if response in ['yes', 'y']:
            for package in missing_packages:
                print(f"Installing {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {package}: {e}")
                    sys.exit(1)
            print("\nAll dependencies installed successfully!")
            print("Please restart the script to use the newly installed packages.")
            sys.exit(0)
        else:
            print("Cannot proceed without required packages.")
            sys.exit(1)

# Check dependencies before importing
check_and_install_dependencies()

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

def extract_scratches(bmp_path: str, threshold: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract scratches from a BMP file with dark background.
    
    Args:
        bmp_path: Path to the BMP file containing scratches
        threshold: Threshold value for separating scratches from background
        
    Returns:
        scratch_mask: Binary mask of the scratches
        scratch_image: The scratch pixels preserved
    """
    # Read the BMP file
    scratch_img = cv2.imread(bmp_path)
    if scratch_img is None:
        raise ValueError(f"Could not read image: {bmp_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(scratch_img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for the scratches (lighter pixels on dark background)
    _, scratch_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply some morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, kernel)
    scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, kernel)
    
    # Create scratch image with transparency
    scratch_image = scratch_img.copy()
    
    return scratch_mask, scratch_image

def overlay_scratches(background_path: str, scratch_mask: np.ndarray, 
                     scratch_image: np.ndarray, opacity: float = 0.7) -> np.ndarray:
    """
    Overlay scratches onto a clean fiber optic image.
    
    Args:
        background_path: Path to the clean fiber optic image
        scratch_mask: Binary mask of the scratches
        scratch_image: The scratch pixels
        opacity: Opacity of the scratch overlay (0.0 to 1.0)
        
    Returns:
        result: The composite image with scratches overlaid
    """
    # Read the background image
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Could not read image: {background_path}")
    
    # Resize scratch mask and image to match background dimensions if needed
    if scratch_mask.shape[:2] != background.shape[:2]:
        scratch_mask = cv2.resize(scratch_mask, (background.shape[1], background.shape[0]))
        scratch_image = cv2.resize(scratch_image, (background.shape[1], background.shape[0]))
    
    # Convert mask to 3-channel for blending
    mask_3channel = cv2.cvtColor(scratch_mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    # Extract only the scratch pixels (make them slightly brighter/visible)
    scratch_pixels = scratch_image.astype(np.float32)
    
    # Enhance scratch visibility by adjusting brightness
    scratch_pixels = np.clip(scratch_pixels * 1.5, 0, 255)
    
    # Blend the images
    result = background.astype(np.float32) * (1 - mask_3channel * opacity) + \
             scratch_pixels * mask_3channel * opacity
    
    return result.astype(np.uint8)

def process_batch(scratch_dir: str = ".", output_dir: str = "output", 
                 threshold: int = 30, opacity: float = 0.7):
    """
    Process all combinations of BMP scratches and clean fiber optic images.
    
    Args:
        scratch_dir: Directory containing BMP and clean images
        output_dir: Directory to save output images
        threshold: Threshold for scratch extraction
        opacity: Opacity of scratch overlay
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all BMP files (scratch files)
    bmp_files = sorted([f for f in os.listdir(scratch_dir) if f.endswith('.bmp')])
    
    # Get all clean fiber optic images (PNG and JPG)
    clean_files = sorted([f for f in os.listdir(scratch_dir) 
                         if (f.endswith('.png') or f.endswith('.jpg')) 
                         and 'clean' in f.lower()])
    
    print(f"Found {len(bmp_files)} scratch files and {len(clean_files)} clean fiber optic images")
    
    # Process each combination
    for i, bmp_file in enumerate(bmp_files):
        bmp_path = os.path.join(scratch_dir, bmp_file)
        
        try:
            # Extract scratches from BMP
            print(f"\nProcessing scratch file: {bmp_file}")
            scratch_mask, scratch_image = extract_scratches(bmp_path, threshold)
            
            # Apply to each clean image
            for j, clean_file in enumerate(clean_files):
                clean_path = os.path.join(scratch_dir, clean_file)
                
                # Create output filename
                scratch_name = Path(bmp_file).stem
                clean_name = Path(clean_file).stem
                output_name = f"{clean_name}_with_{scratch_name}_scratches.jpg"
                output_path = os.path.join(output_dir, output_name)
                
                # Overlay scratches
                result = overlay_scratches(clean_path, scratch_mask, scratch_image, opacity)
                
                # Save result
                cv2.imwrite(output_path, result)
                print(f"  Created: {output_name}")
                
        except Exception as e:
            print(f"  Error processing {bmp_file}: {str(e)}")

def get_user_input(prompt: str, default: str, validation_func=None) -> str:
    """Get input from user with validation."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            user_input = default
        
        if validation_func:
            try:
                validation_func(user_input)
                return user_input
            except ValueError as e:
                print(f"Invalid input: {e}")
        else:
            return user_input

def validate_threshold(value: str):
    """Validate threshold value."""
    try:
        threshold = int(value)
        if not 0 <= threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255")
    except:
        raise ValueError("Threshold must be an integer")

def validate_opacity(value: str):
    """Validate opacity value."""
    try:
        opacity = float(value)
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("Opacity must be between 0.0 and 1.0")
    except:
        raise ValueError("Opacity must be a number")

def main():
    print("=== Scratch Overlay Tool ===")
    print("This tool overlays scratches from BMP files onto clean fiber optic images.")
    print()
    
    # Get user configuration
    input_dir = get_user_input("Enter input directory path", ".")
    output_dir = get_user_input("Enter output directory path", "output")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    # Get processing parameters
    threshold = get_user_input("Enter threshold for scratch extraction (0-255)", "30", validate_threshold)
    opacity = get_user_input("Enter opacity for scratch overlay (0.0-1.0)", "0.7", validate_opacity)
    
    # Ask for processing mode
    mode = get_user_input("Process all combinations or demo mode? (all/demo)", "demo")
    
    if mode.lower() == "demo":
        # Demo mode - process just one example
        bmp_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.bmp')])
        clean_files = sorted([f for f in os.listdir(input_dir) 
                            if (f.endswith('.png') or f.endswith('.jpg')) 
                            and 'clean' in f.lower()])
        
        if not bmp_files:
            print("Error: No BMP files found in the input directory.")
            sys.exit(1)
        
        if not clean_files:
            print("Error: No clean fiber optic images found in the input directory.")
            sys.exit(1)
        
        print(f"\nFound {len(bmp_files)} BMP files and {len(clean_files)} clean images.")
        print(f"Demo will use: {bmp_files[0]} and {clean_files[0]}")
        
        proceed = get_user_input("Proceed with demo?", "yes")
        if proceed.lower() not in ['yes', 'y']:
            print("Demo cancelled.")
            return
        
        Path(output_dir).mkdir(exist_ok=True)
        
        bmp_path = os.path.join(input_dir, bmp_files[0])
        clean_path = os.path.join(input_dir, clean_files[0])
        
        print(f"\nProcessing {bmp_files[0]} with {clean_files[0]}...")
        
        scratch_mask, scratch_image = extract_scratches(bmp_path, int(threshold))
        result = overlay_scratches(clean_path, scratch_mask, scratch_image, float(opacity))
        
        output_path = os.path.join(output_dir, "demo_output.jpg")
        cv2.imwrite(output_path, result)
        print(f"Demo output saved to: {output_path}")
    else:
        # Process all combinations
        proceed = get_user_input("This will process ALL combinations. Continue?", "yes")
        if proceed.lower() not in ['yes', 'y']:
            print("Processing cancelled.")
            return
        
        process_batch(input_dir, output_dir, int(threshold), float(opacity))

if __name__ == "__main__":
    main()