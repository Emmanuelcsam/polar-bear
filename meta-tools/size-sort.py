#!/usr/bin/env python3
"""
Image Organizer by Resolution
Sorts images into folders based on their pixel dimensions
Interactive version - asks questions instead of using flags
"""

import os
import shutil
from pathlib import Path
from PIL import Image

def get_image_dimensions(image_path):
    """Get the dimensions of an image file."""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def ask_yes_no(question, default="no"):
    """Ask a yes/no question and return boolean result."""
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default == "yes":
        prompt = " [Y/n] "
        default_val = True
    else:
        prompt = " [y/N] "
        default_val = False
    
    while True:
        choice = input(question + prompt).lower().strip()
        if choice == "":
            return default_val
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

def get_directory():
    """Ask user for directory path and validate it exists."""
    while True:
        print("\nWhich directory contains the images you want to organize?")
        print("(Enter '.' for current directory or full path)")
        dir_path = input("Directory path: ").strip()
        
        if dir_path == "":
            print("Please enter a directory path.")
            continue
            
        # Expand user home directory if ~ is used
        dir_path = os.path.expanduser(dir_path)
        
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            return os.path.abspath(dir_path)
        else:
            print(f"Error: '{dir_path}' is not a valid directory!")
            if ask_yes_no("Would you like to try again?", "yes"):
                continue
            else:
                return None

def organize_images_by_resolution(source_dir, move_files, create_subdir, subdir_name):
    """
    Organize images in source_dir into subfolders based on resolution.
    
    Args:
        source_dir: Path to directory containing images
        move_files: If True, move files. If False, copy files.
        create_subdir: If True, create a subdirectory for organized images
        subdir_name: Name of subdirectory if create_subdir is True
    """
    source_path = Path(source_dir)
    
    # Determine target base directory
    if create_subdir:
        base_path = source_path / subdir_name
        base_path.mkdir(exist_ok=True)
        print(f"\nCreated subdirectory: {subdir_name}")
    else:
        base_path = source_path
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}
    
    # First, scan for images
    image_files = []
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Skip files already in organized subdirectory
            if create_subdir and file_path.parent.name == subdir_name:
                continue
            image_files.append(file_path)
    
    if not image_files:
        print("\nNo image files found in the directory!")
        return
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Statistics
    processed = 0
    skipped = 0
    errors = 0
    
    print(f"\nMode: {'Moving' if move_files else 'Copying'} files")
    print("-" * 50)
    
    # Process each image file
    for file_path in image_files:
        # Get image dimensions
        dimensions = get_image_dimensions(file_path)
        
        if dimensions is None:
            errors += 1
            continue
        
        width, height = dimensions
        
        # Create folder name based on dimensions
        folder_name = f"{width}x{height}"
        target_folder = base_path / folder_name
        
        # Create the folder if it doesn't exist
        target_folder.mkdir(exist_ok=True)
        
        # Define target path
        target_path = target_folder / file_path.name
        
        # Handle file name conflicts
        if target_path.exists():
            base_name = file_path.stem
            extension = file_path.suffix
            counter = 1
            
            while target_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                target_path = target_folder / new_name
                counter += 1
        
        try:
            if move_files:
                shutil.move(str(file_path), str(target_path))
                print(f"Moved: {file_path.name} -> {folder_name}/")
            else:
                shutil.copy2(str(file_path), str(target_path))
                print(f"Copied: {file_path.name} -> {folder_name}/")
            processed += 1
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            errors += 1
    
    # Print summary
    print("-" * 50)
    print(f"\nSummary:")
    print(f"  Processed: {processed} images")
    print(f"  Errors: {errors} images")
    
    # Count resolution folders created
    resolution_folders = 0
    if base_path.exists():
        for item in base_path.iterdir():
            if item.is_dir() and 'x' in item.name:
                resolution_folders += 1
    print(f"  Resolution folders: {resolution_folders}")

def main():
    print("=" * 60)
    print("       IMAGE ORGANIZER BY RESOLUTION")
    print("=" * 60)
    print("\nThis script will organize your images into folders")
    print("based on their pixel dimensions (e.g., 1920x1080)")
    
    # Check if Pillow is installed
    try:
        import PIL
    except ImportError:
        print("\nError: Pillow library is not installed!")
        print("This script requires Pillow to read image dimensions.")
        print("\nTo install it, run: pip install Pillow")
        input("\nPress Enter to exit...")
        return
    
    # Get directory
    source_dir = get_directory()
    if source_dir is None:
        print("\nExiting...")
        return
    
    print(f"\nSelected directory: {source_dir}")
    
    # Ask about subdirectory
    print("\nWould you like to create a subdirectory for organized images?")
    print("(This keeps your original directory structure intact)")
    create_subdir = ask_yes_no("Create subdirectory?", "yes")
    
    subdir_name = "organized_by_resolution"
    if create_subdir:
        print(f"\nDefault subdirectory name: '{subdir_name}'")
        if ask_yes_no("Would you like to use a different name?", "no"):
            custom_name = input("Enter subdirectory name: ").strip()
            if custom_name:
                subdir_name = custom_name
    
    # Ask about move vs copy
    print("\nHow would you like to handle the files?")
    print("  1. COPY files (safer - keeps originals in place)")
    print("  2. MOVE files (cleaner - relocates originals)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        if choice == "1":
            move_files = False
            break
        elif choice == "2":
            move_files = True
            break
        else:
            print("Please enter 1 or 2")
    
    # Show preview
    print("\n" + "=" * 60)
    print("READY TO ORGANIZE")
    print("=" * 60)
    print(f"Directory: {source_dir}")
    print(f"Action: {'MOVE' if move_files else 'COPY'} files")
    if create_subdir:
        print(f"Destination: {source_dir}/{subdir_name}/[resolution folders]")
    else:
        print(f"Destination: {source_dir}/[resolution folders]")
    print("=" * 60)
    
    if not ask_yes_no("\nProceed with organizing images?", "yes"):
        print("\nOperation cancelled.")
        return
    
    # Organize the images
    organize_images_by_resolution(source_dir, move_files, create_subdir, subdir_name)
    
    print("\nOrganization complete!")
    
    # Ask if user wants to organize another directory
    if ask_yes_no("\nWould you like to organize another directory?", "no"):
        main()
    else:
        print("\nThank you for using Image Organizer!")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
