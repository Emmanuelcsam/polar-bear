#!/usr/bin/env python3
"""
Generate all combinations of JPG/PNG images with BMP overlays.
This script overlays each BMP file onto each JPG/PNG file.
"""

import os
import sys
from PIL import Image
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'combination_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def find_images(directory="."):
    """Find all JPG, PNG, and BMP files in directory and subdirectories."""
    jpg_png_files = []
    bmp_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            file_lower = file.lower()
            
            if file_lower.endswith(('.jpg', '.jpeg', '.png')):
                jpg_png_files.append(file_path)
            elif file_lower.endswith('.bmp'):
                bmp_files.append(file_path)
    
    return jpg_png_files, bmp_files

def overlay_images(base_image_path, overlay_image_path, output_path, blend_mode='overlay', opacity=0.7):
    """Overlay one image onto another."""
    try:
        # Open images
        base = Image.open(base_image_path).convert('RGBA')
        overlay = Image.open(overlay_image_path).convert('RGBA')
        
        # Resize overlay to match base image size
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        
        # Apply overlay with opacity
        if blend_mode == 'overlay':
            # Create a new image for the result
            result = Image.new('RGBA', base.size)
            
            # Apply overlay blend mode
            for x in range(base.width):
                for y in range(base.height):
                    base_pixel = base.getpixel((x, y))
                    overlay_pixel = overlay.getpixel((x, y))
                    
                    # Overlay blend mode formula
                    r = overlay_blend_channel(base_pixel[0], overlay_pixel[0], opacity)
                    g = overlay_blend_channel(base_pixel[1], overlay_pixel[1], opacity)
                    b = overlay_blend_channel(base_pixel[2], overlay_pixel[2], opacity)
                    a = base_pixel[3]
                    
                    result.putpixel((x, y), (int(r), int(g), int(b), a))
        else:
            # Simple alpha composite
            overlay_with_opacity = Image.new('RGBA', overlay.size)
            for x in range(overlay.width):
                for y in range(overlay.height):
                    pixel = overlay.getpixel((x, y))
                    overlay_with_opacity.putpixel((x, y), (pixel[0], pixel[1], pixel[2], int(pixel[3] * opacity)))
            
            result = Image.alpha_composite(base, overlay_with_opacity)
        
        # Convert back to RGB for saving as JPG
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            result = result.convert('RGB')
        
        # Save the result
        result.save(output_path, quality=95)
        return True
        
    except Exception as e:
        logging.error(f"Error overlaying {base_image_path} with {overlay_image_path}: {e}")
        return False

def overlay_blend_channel(base, overlay, opacity):
    """Apply overlay blend mode to a single channel."""
    if base < 128:
        result = 2 * base * overlay / 255
    else:
        result = 255 - 2 * (255 - base) * (255 - overlay) / 255
    
    # Apply opacity
    return base * (1 - opacity) + result * opacity

def generate_all_combinations(directory=".", output_dir="combined_output", blend_mode='overlay', opacity=0.7):
    """Generate all combinations of JPG/PNG images with BMP overlays."""
    # Find all images
    logging.info("Searching for images...")
    jpg_png_files, bmp_files = find_images(directory)
    
    logging.info(f"Found {len(jpg_png_files)} JPG/PNG files and {len(bmp_files)} BMP files")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    total_combinations = len(jpg_png_files) * len(bmp_files)
    logging.info(f"Generating {total_combinations} combinations...")
    
    successful = 0
    failed = 0
    
    # Generate all combinations
    for i, base_image in enumerate(jpg_png_files, 1):
        for j, overlay_image in enumerate(bmp_files, 1):
            # Create output filename
            base_name = base_image.stem
            overlay_name = overlay_image.stem
            output_extension = base_image.suffix
            output_filename = f"{base_name}_WITH_{overlay_name}{output_extension}"
            output_full_path = output_path / output_filename
            
            # Log progress
            combination_num = (i - 1) * len(bmp_files) + j
            logging.info(f"[{combination_num}/{total_combinations}] Processing: {base_image.name} + {overlay_image.name}")
            
            # Generate combination
            if overlay_images(base_image, overlay_image, output_full_path, blend_mode, opacity):
                successful += 1
            else:
                failed += 1
    
    logging.info(f"\nCompleted! Successfully generated {successful} combinations, {failed} failed.")
    logging.info(f"Output files saved to: {output_path.absolute()}")

def main():
    """Main function with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate all combinations of JPG/PNG images with BMP overlays')
    parser.add_argument('--directory', '-d', default='.', help='Directory to search for images (default: current directory)')
    parser.add_argument('--output', '-o', default='combined_output', help='Output directory (default: combined_output)')
    parser.add_argument('--blend-mode', '-b', choices=['overlay', 'alpha'], default='overlay', help='Blend mode (default: overlay)')
    parser.add_argument('--opacity', '-p', type=float, default=0.7, help='Overlay opacity 0.0-1.0 (default: 0.7)')
    
    args = parser.parse_args()
    
    # Validate opacity
    if not 0.0 <= args.opacity <= 1.0:
        logging.error("Opacity must be between 0.0 and 1.0")
        sys.exit(1)
    
    generate_all_combinations(args.directory, args.output, args.blend_mode, args.opacity)

if __name__ == "__main__":
    main()