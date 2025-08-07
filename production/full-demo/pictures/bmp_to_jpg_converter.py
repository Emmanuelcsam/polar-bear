"""
BMP to JPG Converter
This script converts all BMP images in the current directory to JPG format
and saves them in a 'converted' folder to preserve the original images.
"""

import os
from PIL import Image
import glob

def create_converted_folder():
    """Create the 'converted' folder if it doesn't exist"""
    converted_folder = "converted"
    if not os.path.exists(converted_folder):
        os.makedirs(converted_folder)
        print(f"Created folder: {converted_folder}")
    return converted_folder

def convert_bmp_to_jpg(input_path, output_path, quality=95):
    """
    Convert a single BMP image to JPG format
    
    Args:
        input_path (str): Path to the input BMP file
        output_path (str): Path to the output JPG file
        quality (int): JPG quality (1-100, default 95)
    """
    try:
        # Open the BMP image
        with Image.open(input_path) as img:
            # Convert RGBA to RGB if necessary (JPG doesn't support transparency)
            if img.mode in ('RGBA', 'LA'):
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(img)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            print(f"✓ Converted: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            return True
            
    except Exception as e:
        print(f"✗ Error converting {input_path}: {str(e)}")
        return False

def main():
    """Main function to convert all BMP files in the current directory"""
    print("BMP to JPG Converter")
    print("=" * 40)
    
    # Create the converted folder
    converted_folder = create_converted_folder()
    
    # Find all BMP files in the current directory
    bmp_files = glob.glob("*.bmp")
    
    if not bmp_files:
        print("No BMP files found in the current directory.")
        return
    
    print(f"Found {len(bmp_files)} BMP file(s) to convert:")
    for bmp_file in bmp_files:
        print(f"  - {bmp_file}")
    
    print("\nStarting conversion...")
    print("-" * 40)
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Convert each BMP file
    for bmp_file in bmp_files:
        # Create output filename (replace .bmp with .jpg)
        base_name = os.path.splitext(bmp_file)[0]
        jpg_filename = f"{base_name}.jpg"
        output_path = os.path.join(converted_folder, jpg_filename)
        
        # Convert the file
        if convert_bmp_to_jpg(bmp_file, output_path):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Print summary
    print("-" * 40)
    print(f"Conversion complete!")
    print(f"✓ Successfully converted: {successful_conversions} files")
    if failed_conversions > 0:
        print(f"✗ Failed conversions: {failed_conversions} files")
    
    print(f"\nConverted files saved in: {os.path.abspath(converted_folder)}")

if __name__ == "__main__":
    main()
