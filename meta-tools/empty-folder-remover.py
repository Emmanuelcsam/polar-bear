#!/usr/bin/env python3
"""
Empty Folder Remover - A robust script to recursively remove empty directories
Author: Assistant
Version: 1.0
"""

import os
import sys
import shutil
import subprocess
import importlib
import importlib.util
from datetime import datetime
from pathlib import Path
import time

# ANSI color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    if level == "INFO":
        color = Colors.CYAN
    elif level == "SUCCESS":
        color = Colors.GREEN
    elif level == "WARNING":
        color = Colors.WARNING
    elif level == "ERROR":
        color = Colors.FAIL
    elif level == "HEADER":
        color = Colors.HEADER + Colors.BOLD
    else:
        color = ""
    
    print(f"{color}[{timestamp}] {level}: {message}{Colors.ENDC}")

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking for package: {package_name}", "INFO")
    
    # Check if the module can be imported
    spec = importlib.util.find_spec(import_name)
    
    if spec is None:
        log(f"Package '{package_name}' not found. Installing...", "WARNING")
        try:
            # Update pip first
            log("Updating pip to latest version...", "INFO")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install the package
            log(f"Installing {package_name}...", "INFO")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            
            log(f"Successfully installed {package_name}", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            log(f"Failed to install {package_name}: {e}", "ERROR")
            return False
    else:
        log(f"Package '{package_name}' is already installed", "SUCCESS")
        return True

def setup_environment():
    """Check and setup all required packages"""
    log("=" * 60, "HEADER")
    log("EMPTY FOLDER REMOVER - ENVIRONMENT SETUP", "HEADER")
    log("=" * 60, "HEADER")
    
    # For this script, we only use built-in modules
    # But I'll include the auto-install functionality as requested
    required_packages = [
        # Adding colorama as an example of external package that could be used
        ("colorama", "colorama"),  # (pip_name, import_name)
    ]
    
    all_installed = True
    
    # Check Python version
    log(f"Python version: {sys.version}", "INFO")
    if sys.version_info < (3, 6):
        log("This script requires Python 3.6 or higher!", "ERROR")
        return False
    
    # Check for required packages
    for package_name, import_name in required_packages:
        if not check_and_install_package(package_name, import_name):
            all_installed = False
    
    log("Environment setup complete!", "SUCCESS")
    return all_installed

def get_user_input(prompt, validation_func=None, error_msg="Invalid input"):
    """Get validated input from user"""
    while True:
        log(prompt, "INFO")
        user_input = input(f"{Colors.BOLD}> {Colors.ENDC}").strip()
        
        if validation_func is None or validation_func(user_input):
            return user_input
        else:
            log(error_msg, "ERROR")

def validate_directory(path):
    """Validate if the given path is a directory"""
    return os.path.exists(path) and os.path.isdir(path)

def is_directory_empty(path):
    """Check if a directory is empty"""
    try:
        # A directory is empty if it contains no files or subdirectories
        return len(os.listdir(path)) == 0
    except PermissionError:
        log(f"Permission denied accessing: {path}", "WARNING")
        return False

def find_empty_directories(root_path):
    """Find all empty directories recursively"""
    empty_dirs = []
    
    log(f"Starting deep crawl from: {root_path}", "INFO")
    
    # Walk the directory tree from bottom to top
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Skip hidden directories if requested
        if skip_hidden and os.path.basename(dirpath).startswith('.'):
            log(f"Skipping hidden directory: {dirpath}", "INFO")
            continue
            
        # Check if directory is empty
        if is_directory_empty(dirpath):
            empty_dirs.append(dirpath)
            log(f"Found empty directory: {dirpath}", "WARNING")
    
    return empty_dirs

def remove_empty_directory(path, dry_run=False):
    """Remove a single empty directory"""
    try:
        if dry_run:
            log(f"[DRY RUN] Would remove: {path}", "INFO")
            return True
        else:
            os.rmdir(path)
            log(f"Successfully removed: {path}", "SUCCESS")
            return True
    except PermissionError:
        log(f"Permission denied removing: {path}", "ERROR")
        return False
    except OSError as e:
        log(f"Error removing {path}: {e}", "ERROR")
        return False

def remove_empty_directories_recursive(root_path, dry_run=False):
    """Remove all empty directories recursively"""
    total_removed = 0
    total_failed = 0
    
    # Keep running until no more empty directories are found
    while True:
        empty_dirs = find_empty_directories(root_path)
        
        if not empty_dirs:
            log("No more empty directories found", "SUCCESS")
            break
        
        log(f"Found {len(empty_dirs)} empty directories in this pass", "INFO")
        
        # Remove directories from deepest to shallowest
        empty_dirs.sort(key=lambda x: x.count(os.sep), reverse=True)
        
        for empty_dir in empty_dirs:
            if remove_empty_directory(empty_dir, dry_run):
                total_removed += 1
            else:
                total_failed += 1
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    return total_removed, total_failed

def main():
    """Main function"""
    global skip_hidden
    
    # Setup environment
    if not setup_environment():
        log("Environment setup failed. Exiting.", "ERROR")
        return 1
    
    print("\n")
    log("=" * 60, "HEADER")
    log("EMPTY FOLDER REMOVER - MAIN PROGRAM", "HEADER")
    log("=" * 60, "HEADER")
    
    # Get directory path from user
    target_dir = get_user_input(
        "Enter the directory path to clean up:",
        validate_directory,
        "Invalid directory path. Please enter a valid directory."
    )
    
    target_dir = os.path.abspath(target_dir)
    log(f"Target directory: {target_dir}", "INFO")
    
    # Ask about hidden directories
    skip_hidden_input = get_user_input(
        "Skip hidden directories (directories starting with '.')? (yes/no):",
        lambda x: x.lower() in ['yes', 'no', 'y', 'n'],
        "Please enter 'yes' or 'no'"
    )
    skip_hidden = skip_hidden_input.lower() in ['yes', 'y']
    
    # Ask about dry run
    dry_run_input = get_user_input(
        "Perform a dry run first (show what would be deleted without actually deleting)? (yes/no):",
        lambda x: x.lower() in ['yes', 'no', 'y', 'n'],
        "Please enter 'yes' or 'no'"
    )
    perform_dry_run = dry_run_input.lower() in ['yes', 'y']
    
    # Perform dry run if requested
    if perform_dry_run:
        log("\n--- DRY RUN MODE ---", "HEADER")
        removed, failed = remove_empty_directories_recursive(target_dir, dry_run=True)
        log(f"\nDry run complete. Would remove {removed} directories.", "INFO")
        
        if removed > 0:
            proceed_input = get_user_input(
                f"\nProceed with actual deletion of {removed} directories? (yes/no):",
                lambda x: x.lower() in ['yes', 'no', 'y', 'n'],
                "Please enter 'yes' or 'no'"
            )
            
            if proceed_input.lower() not in ['yes', 'y']:
                log("Operation cancelled by user.", "WARNING")
                return 0
    
    # Final confirmation
    else:
        confirm_input = get_user_input(
            f"\nAre you sure you want to scan and remove empty folders from:\n{target_dir}\n(yes/no):",
            lambda x: x.lower() in ['yes', 'no', 'y', 'n'],
            "Please enter 'yes' or 'no'"
        )
        
        if confirm_input.lower() not in ['yes', 'y']:
            log("Operation cancelled by user.", "WARNING")
            return 0
    
    # Perform the actual removal
    log("\n--- REMOVING EMPTY DIRECTORIES ---", "HEADER")
    start_time = time.time()
    
    removed, failed = remove_empty_directories_recursive(target_dir, dry_run=False)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    log("\n" + "=" * 60, "HEADER")
    log("OPERATION COMPLETE", "HEADER")
    log("=" * 60, "HEADER")
    log(f"Total directories removed: {removed}", "SUCCESS")
    log(f"Total directories failed: {failed}", "WARNING" if failed > 0 else "INFO")
    log(f"Time taken: {duration:.2f} seconds", "INFO")
    log(f"Target directory: {target_dir}", "INFO")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\nOperation cancelled by user (Ctrl+C)", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)
