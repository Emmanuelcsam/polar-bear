
from pathlib import Path
from typing import List

from log_message import log_message

def get_image_paths_from_user() -> List[Path]:
    """
    Prompts the user for a directory and returns a list of image Paths.
    """
    log_message("Starting image path collection...")
    image_paths: List[Path] = []
    
    while True:
        dir_path_str = input("Enter the path to the directory containing fiber images: ").strip()
        
        # Allow user to type 'exit' to quit
        if dir_path_str.lower() == 'exit':
            log_message("User requested to exit.", level="INFO")
            break

        image_dir = Path(dir_path_str)
        
        if not image_dir.is_dir():
            log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again or type 'exit'.", level="ERROR")
            continue
            
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        image_paths = [
            item for item in image_dir.iterdir() 
            if item.is_file() and item.suffix.lower() in supported_extensions
        ]
        
        if not image_paths:
            log_message(f"No supported images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
            # The loop will continue, asking for another path
        else:
            log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
            break # Exit the loop as we have found images
            
    return image_paths

if __name__ == '__main__':
    # Example of how to use the get_image_paths_from_user function
    
    print("This script will ask you to provide a path to a directory with images.")
    print("You can use the 'fiber_inspection_output' directory that might contain some sample images,")
    print("or any other directory on your system. Type 'exit' to cancel.")
    
    # For demonstration, you can point this to the existing output directory
    # e.g., C:\Users\Saem1001\Documents\GitHub\polar-bear\decrepit-versions\version10
    # or a subdirectory like 'fiber_inspection_output/ima18'
    
    paths = get_image_paths_from_user()
    
    if paths:
        print("\n--- Found Image Files ---")
        for p in paths:
            print(f" - {p.name}")
        print("-------------------------")
    else:
        print("\nNo image paths were collected.")
