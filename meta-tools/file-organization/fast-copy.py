import os
import shutil
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging immediately
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler for terminal output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# File handler for log file
fh = logging.FileHandler('copy_log.txt')
fh.setLevel(logging.INFO)

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

logger.info("Logging setup complete. Starting the directory copy script.")

# Function to check and install a package if missing
def install_package(package_name):
    try:
        __import__(package_name)
        logger.info(f"Package '{package_name}' is already installed.")
    except ImportError:
        logger.warning(f"Package '{package_name}' not found. Attempting to install the latest version.")
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package_name])
            logger.info(f"Successfully installed the latest version of '{package_name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install '{package_name}': {e}")
            sys.exit(1)

# No external packages needed here, but if we add any in future, we can call install_package('package')

# Prompt for source directory
source = input("Enter the source directory path: ").strip()
logger.info(f"User entered source path: {source}")

while not os.path.isdir(source):
    logger.error(f"Invalid source directory: {source}")
    source = input("Please enter a valid source directory path: ").strip()
    logger.info(f"User entered new source path: {source}")

logger.info(f"Valid source directory confirmed: {source}")

# Prompt for destination directory
dest = input("Enter the destination directory path: ").strip()
logger.info(f"User entered destination path: {dest}")

# Check if destination exists and handle accordingly
if os.path.exists(dest):
    overwrite = input(f"Destination '{dest}' already exists. Do you want to overwrite its contents? (yes/no): ").strip().lower()
    logger.info(f"User response to overwrite: {overwrite}")
    if overwrite != 'yes':
        logger.info("Copy operation cancelled by user.")
        sys.exit("Copy cancelled.")
    logger.info("Proceeding with overwrite of existing destination.")
else:
    try:
        os.makedirs(dest, exist_ok=True)
        logger.info(f"Created new destination directory: {dest}")
    except Exception as e:
        logger.error(f"Error creating destination directory '{dest}': {e}")
        sys.exit(1)

# Function to copy a single file while preserving metadata
def copy_file(src_file, dest_file):
    try:
        shutil.copy2(src_file, dest_file)
        logger.info(f"Successfully copied file '{src_file}' to '{dest_file}'")
    except Exception as e:
        logger.error(f"Error copying file '{src_file}' to '{dest_file}': {e}")

# Collect all files and create directories first (serial, but fast)
files_to_copy = []
logger.info("Starting to walk through source directory to prepare copy tasks.")

try:
    for root, dirs, files in os.walk(source):
        # Create directories
        for d in dirs:
            src_dir = os.path.join(root, d)
            rel_path = os.path.relpath(src_dir, source)
            dest_dir = os.path.join(dest, rel_path)
            try:
                os.makedirs(dest_dir, exist_ok=True)
                logger.info(f"Created directory '{dest_dir}'")
            except Exception as e:
                logger.error(f"Error creating directory '{dest_dir}': {e}")

        # Collect files for parallel copy
        for f in files:
            src_file = os.path.join(root, f)
            rel_path = os.path.relpath(src_file, source)
            dest_file = os.path.join(dest, rel_path)
            files_to_copy.append((src_file, dest_file))
            # Log collection, but not every one to avoid flood; log count periodically if needed
            if len(files_to_copy) % 1000 == 0:
                logger.info(f"Collected {len(files_to_copy)} files for copying so far.")

    logger.info(f"Finished walking directory. Total files to copy: {len(files_to_copy)}")
except Exception as e:
    logger.error(f"Error during directory walk: {e}")
    sys.exit(1)

# Perform parallel file copying if there are files
if files_to_copy:
    max_workers = max(1, os.cpu_count() * 2)  # Adjust based on CPU for IO-bound tasks
    logger.info(f"Starting parallel copy with {max_workers} workers.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_file, src, dst) for src, dst in files_to_copy]
        for future in as_completed(futures):
            try:
                future.result()  # Wait and log any exceptions
            except Exception as e:
                logger.error(f"Exception in copy task: {e}")

    logger.info("All copy tasks completed.")
else:
    logger.info("No files to copy.")

logger.info("Directory copy operation finished.")
