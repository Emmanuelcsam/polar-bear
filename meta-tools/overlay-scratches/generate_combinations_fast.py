#!/usr/bin/env python3
"""
Fast version: Generate all combinations of JPG/PNG images with BMP overlays using NumPy.
This script overlays each BMP file onto each JPG/PNG file with optimized performance.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fast_combination_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

def overlay_blend_numpy(base_array, overlay_array, opacity=0.7):
    """Apply overlay blend mode using NumPy for better performance."""
    # Normalize to 0-1 range
    base_norm = base_array.astype(np.float32) / 255.0
    overlay_norm = overlay_array.astype(np.float32) / 255.0
    
    # Overlay blend mode formula
    mask = base_norm < 0.5
    result = np.zeros_like(base_norm)
    
    # Where base < 0.5: 2 * base * overlay
    result[mask] = 2 * base_norm[mask] * overlay_norm[mask]
    
    # Where base >= 0.5: 1 - 2 * (1 - base) * (1 - overlay)
    result[~mask] = 1 - 2 * (1 - base_norm[~mask]) * (1 - overlay_norm[~mask])
    
    # Apply opacity
    blended = base_norm * (1 - opacity) + result * opacity
    
    # Convert back to 0-255 range
    return (blended * 255).astype(np.uint8)

def process_single_combination(args):
    """Process a single image combination. Used for parallel processing."""
    base_path, overlay_path, output_path, opacity = args
    
    try:
        # Open images
        base = Image.open(base_path).convert('RGBA')
        overlay = Image.open(overlay_path).convert('RGBA')
        
        # Resize overlay to match base image size
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        base_array = np.array(base)
        overlay_array = np.array(overlay)
        
        # Apply overlay blend to RGB channels
        result_array = np.zeros_like(base_array)
        for i in range(3):  # RGB channels
            result_array[:, :, i] = overlay_blend_numpy(base_array[:, :, i], overlay_array[:, :, i], opacity)
        
        # Keep original alpha channel
        result_array[:, :, 3] = base_array[:, :, 3]
        
        # Convert back to PIL Image
        result = Image.fromarray(result_array, 'RGBA')
        
        # Convert to RGB if saving as JPG
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            result = result.convert('RGB')
        
        # Save the result
        result.save(output_path, quality=95)
        return True, str(output_path)
        
    except Exception as e:
        return False, f"Error: {base_path} + {overlay_path}: {str(e)}"

def generate_all_combinations_parallel(directory=".", output_dir="combined_output", opacity=0.7, max_workers=None):
    """Generate all combinations using parallel processing."""
    # Find all images
    logging.info("Searching for images...")
    jpg_png_files, bmp_files = find_images(directory)
    
    logging.info(f"Found {len(jpg_png_files)} JPG/PNG files and {len(bmp_files)} BMP files")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    total_combinations = len(jpg_png_files) * len(bmp_files)
    logging.info(f"Generating {total_combinations} combinations using parallel processing...")
    
    # Prepare all combination arguments
    tasks = []
    for base_image in jpg_png_files:
        for overlay_image in bmp_files:
            # Create output filename
            base_name = base_image.stem
            overlay_name = overlay_image.stem
            output_extension = base_image.suffix
            output_filename = f"{base_name}_WITH_{overlay_name}{output_extension}"
            output_full_path = output_path / output_filename
            
            tasks.append((base_image, overlay_image, output_full_path, opacity))
    
    # Process in parallel
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_combination, task): task for task in tasks}
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_task), 1):
            success, message = future.result()
            
            if success:
                successful += 1
                logging.info(f"[{i}/{total_combinations}] Completed: {Path(message).name}")
            else:
                failed += 1
                logging.error(f"[{i}/{total_combinations}] Failed: {message}")
            
            # Progress update every 10%
            if i % max(1, total_combinations // 10) == 0:
                logging.info(f"Progress: {i}/{total_combinations} ({i*100//total_combinations}%)")
    
    logging.info(f"\nCompleted! Successfully generated {successful} combinations, {failed} failed.")
    logging.info(f"Output files saved to: {output_path.absolute()}")

def main():
    """Main function with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast generation of all JPG/PNG + BMP overlay combinations')
    parser.add_argument('--directory', '-d', default='.', help='Directory to search for images (default: current directory)')
    parser.add_argument('--output', '-o', default='combined_output', help='Output directory (default: combined_output)')
    parser.add_argument('--opacity', '-p', type=float, default=0.7, help='Overlay opacity 0.0-1.0 (default: 0.7)')
    parser.add_argument('--workers', '-w', type=int, help='Number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    # Validate opacity
    if not 0.0 <= args.opacity <= 1.0:
        logging.error("Opacity must be between 0.0 and 1.0")
        sys.exit(1)
    
    generate_all_combinations_parallel(args.directory, args.output, args.opacity, args.workers)

if __name__ == "__main__":
    main()