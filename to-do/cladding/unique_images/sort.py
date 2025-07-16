import sys
import os
import shutil
import logging
import subprocess
from datetime import datetime

# Function to check and install a package if missing
def install_if_missing(package):
    try:
        __import__(package)
        logging.info(f"Package '{package}' is already installed.")
    except ImportError:
        logging.info(f"Package '{package}' is missing. Installing latest version...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logging.info(f"Successfully installed '{package}'.")
        except Exception as e:
            logging.error(f"Failed to install '{package}': {str(e)}")
            sys.exit(1)

# Set up initial logging to console only
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

logging.info("Starting the image similarity mover script.")

# List of required packages
required_packages = ['PIL', 'numpy', 'scipy', 'imagehash']

# Check and install required packages
for pkg in required_packages:
    if pkg == 'PIL':
        install_if_missing('pillow')  # PIL is imported as PIL, but installed as pillow
    else:
        install_if_missing(pkg)

# Now import the libraries after ensuring they are installed
try:
    from PIL import Image
    import numpy as np
    from scipy import ndimage  # Not directly used, but imagehash may need scipy
    import imagehash
    logging.info("All required libraries imported successfully.")
except ImportError as e:
    logging.error(f"Failed to import required libraries after installation: {str(e)}")
    sys.exit(1)

# Interactive configuration questions
logging.info("Beginning configuration questions.")

reference_image_path = input("What is the path to the reference image? ")
logging.info(f"User provided reference image path: {reference_image_path}")

search_directory = input("What is the directory to search for images? ")
logging.info(f"User provided search directory: {search_directory}")

recursive_search = input("Should the search be recursive (include subfolders)? (yes/no): ").strip().lower() == 'yes'
logging.info(f"Recursive search: {recursive_search}")

subfolder_name = input("What is the name of the subfolder to move similar images to? (default: similar_images): ") or 'similar_images'
logging.info(f"Subfolder name: {subfolder_name}")

threshold = input("What is the similarity threshold (Hamming distance, lower means more similar, suggest 0-20 for somewhat similar)? (default: 10): ") or '10'
try:
    threshold = int(threshold)
    logging.info(f"Similarity threshold: {threshold}")
except ValueError:
    logging.error("Invalid threshold provided. Must be an integer.")
    sys.exit(1)

log_file_path = input("What is the path for the log file? (default: image_mover.log): ") or 'image_mover.log'
logging.info(f"Log file path: {log_file_path}")

# Add file handler to logging now that we have the log file path
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logging.info("Added file handler for logging. Now logging to both terminal and file.")

# Validate inputs
if not os.path.isfile(reference_image_path):
    logging.error(f"Reference image does not exist: {reference_image_path}")
    sys.exit(2)

if not os.path.isdir(search_directory):
    logging.error(f"Search directory does not exist: {search_directory}")
    sys.exit(2)

# Create subfolder in the search directory
subfolder_path = os.path.join(search_directory, subfolder_name)
try:
    os.makedirs(subfolder_path, exist_ok=True)
    logging.info(f"Created subfolder_path: {subfolder_path}")
except Exception as e:
    logging.error(f"Failed to create subfolder: {str(e)}")
    sys.exit(3)

# Compute hash for reference image
try:
    ref_hash = imagehash.dhash(Image.open(reference_image_path))
    logging.info(f"Computed dhash for reference image: {ref_hash}")
except Exception as e:
    logging.error(f"Error processing reference image {reference_image_path}: {str(e)}")
    sys.exit(4)

# Supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

# Function to process a single file
def is_similar(file_path):
    try:
        img_hash = imagehash.dhash(Image.open(file_path))
        distance = ref_hash - img_hash
        logging.info(f"Processed {file_path} - Hash: {img_hash}, Distance: {distance}")
        return distance < threshold
    except Exception as e:
        logging.warning(f"Skipping invalid or corrupted image {file_path}: {str(e)}")
        return False

# Collect files to process
files_to_move = []

if recursive_search:
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                if full_path == reference_image_path:
                    logging.info(f"Skipping the reference image itself: {full_path}")
                    continue
                logging.info(f"Checking similarity for {full_path}")
                if is_similar(full_path):
                    files_to_move.append(full_path)
else:
    for file in os.listdir(search_directory):
        full_path = os.path.join(search_directory, file)
        if not os.path.isfile(full_path):
            continue
        if not file.lower().endswith(image_extensions):
            continue
        if full_path == reference_image_path:
            logging.info(f"Skipping the reference image itself: {full_path}")
            continue
        logging.info(f"Checking similarity for {full_path}")
        if is_similar(full_path):
            files_to_move.append(full_path)

# Move the similar files
for file_path in files_to_move:
    try:
        dest_path = os.path.join(subfolder_path, os.path.basename(file_path))
        # Handle if moving from subdir in recursive, preserve relative path? No, just move to flat subfolder to avoid nesting
        # But if duplicates names, shutil.move will fail if exists, so check
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(os.path.basename(file_path))
            new_base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            dest_path = os.path.join(subfolder_path, new_base)
            logging.info(f"File name conflict, renaming to {new_base}")
        shutil.move(file_path, dest_path)
        logging.info(f"Moved similar image {file_path} to {dest_path}")
    except Exception as e:
        logging.error(f"Failed to move {file_path}: {str(e)}")

logging.info("Script completed successfully.")
