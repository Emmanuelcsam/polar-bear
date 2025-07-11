#!/usr/bin/env python3
"""
Folder-Based File Renamer Script
Renames all files in a directory tree based on their parent folder names
"""

import os
import sys
import subprocess
import importlib
import shutil
from datetime import datetime
from pathlib import Path
import json
import time

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    color = Colors.ENDC
    if level == "SUCCESS":
        color = Colors.OKGREEN
    elif level == "WARNING":
        color = Colors.WARNING
    elif level == "ERROR":
        color = Colors.FAIL
    elif level == "INFO":
        color = Colors.OKBLUE
    elif level == "HEADER":
        color = Colors.HEADER + Colors.BOLD
    
    print(f"{color}[{timestamp}] [{level}] {message}{Colors.ENDC}")

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed and install it if not"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking for package: {package_name}", "INFO")
    
    try:
        importlib.import_module(import_name)
        log(f"Package {package_name} is already installed", "SUCCESS")
        return True
    except ImportError:
        log(f"Package {package_name} not found. Installing...", "WARNING")
        
        try:
            # Upgrade pip first
            log("Upgrading pip to latest version...", "INFO")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install the package
            log(f"Installing {package_name}...", "INFO")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            
            # Verify installation
            importlib.import_module(import_name)
            log(f"Successfully installed {package_name}", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            log(f"Failed to install {package_name}: {e}", "ERROR")
            return False
        except ImportError:
            log(f"Package {package_name} installed but cannot be imported", "ERROR")
            return False

def setup_dependencies():
    """Check and install all required dependencies"""
    log("DEPENDENCY CHECK", "HEADER")
    
    # List of required packages (package_name, import_name)
    required_packages = [
        ("colorama", "colorama"),  # For better cross-platform color support
        ("tqdm", "tqdm"),  # For progress bars
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_and_install_package(package_name, import_name):
            all_installed = False
    
    if not all_installed:
        log("Some dependencies could not be installed. The script may not work properly.", "ERROR")
        return False
    
    log("All dependencies are ready", "SUCCESS")
    return True

def get_user_input(prompt, default=None, valid_options=None):
    """Get user input with validation"""
    while True:
        if default:
            user_input = input(f"{Colors.OKCYAN}{prompt} [{default}]: {Colors.ENDC}").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{Colors.OKCYAN}{prompt}: {Colors.ENDC}").strip()
        
        if valid_options:
            if user_input.lower() in [opt.lower() for opt in valid_options]:
                return user_input.lower()
            else:
                log(f"Invalid option. Please choose from: {', '.join(valid_options)}", "WARNING")
        else:
            return user_input

def validate_directory(path):
    """Validate if the given path is a directory"""
    if os.path.exists(path) and os.path.isdir(path):
        return True
    return False

def create_backup_record(original_path, new_path, backup_file):
    """Record file rename operation for potential rollback"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "original": str(original_path),
        "renamed": str(new_path)
    }
    
    with open(backup_file, 'a') as f:
        f.write(json.dumps(record) + '\n')

def get_safe_filename(folder_name, original_filename, existing_files, counter_start=1):
    """Generate a safe filename that doesn't conflict with existing files"""
    # Get file extension
    original_path = Path(original_filename)
    extension = ''.join(original_path.suffixes)  # Handles multiple extensions like .tar.gz
    
    # Base new filename
    new_name = f"{folder_name}{extension}"
    
    # Check if file already exists
    if new_name not in existing_files:
        return new_name
    
    # Add counter if necessary
    counter = counter_start
    while True:
        new_name = f"{folder_name}_{counter}{extension}"
        if new_name not in existing_files:
            return new_name
        counter += 1

def rename_files_in_directory(directory, dry_run=False, backup_file=None, preserve_extensions=True):
    """Rename all files in the directory tree based on parent folder names"""
    log(f"Starting file rename process in: {directory}", "HEADER")
    
    # Import tqdm after installation check
    from tqdm import tqdm
    
    # First, count total files for progress bar
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    
    log(f"Found {total_files} files to process", "INFO")
    
    renamed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Create progress bar
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, dirs, files in os.walk(directory):
            # Get the folder name
            folder_path = Path(root)
            folder_name = folder_path.name
            
            # Skip if we're in the root directory
            if folder_path == Path(directory):
                log(f"Skipping root directory: {root}", "WARNING")
                pbar.update(len(files))
                skipped_count += len(files)
                continue
            
            log(f"Processing folder: {folder_name} ({len(files)} files)", "INFO")
            
            # Track existing files in this folder to avoid conflicts
            existing_files = set(files)
            rename_operations = []
            
            # Plan all renames first to avoid conflicts
            for filename in files:
                file_path = folder_path / filename
                
                # Skip if file doesn't exist (shouldn't happen, but safety check)
                if not file_path.exists():
                    log(f"File not found: {file_path}", "ERROR")
                    error_count += 1
                    pbar.update(1)
                    continue
                
                # Skip if already named correctly
                if filename.startswith(folder_name):
                    log(f"Skipping file (already correctly named): {filename}", "INFO")
                    skipped_count += 1
                    pbar.update(1)
                    continue
                
                # Generate new filename
                new_filename = get_safe_filename(folder_name, filename, existing_files)
                new_path = folder_path / new_filename
                
                rename_operations.append((file_path, new_path, filename, new_filename))
                existing_files.add(new_filename)
            
            # Execute rename operations
            for old_path, new_path, old_name, new_name in rename_operations:
                try:
                    if dry_run:
                        log(f"[DRY RUN] Would rename: {old_name} -> {new_name}", "INFO")
                    else:
                        # Create backup record before renaming
                        if backup_file:
                            create_backup_record(old_path, new_path, backup_file)
                        
                        # Perform the rename
                        old_path.rename(new_path)
                        log(f"Renamed: {old_name} -> {new_name}", "SUCCESS")
                    
                    renamed_count += 1
                    
                except PermissionError:
                    log(f"Permission denied: {old_name}", "ERROR")
                    error_count += 1
                except Exception as e:
                    log(f"Error renaming {old_name}: {str(e)}", "ERROR")
                    error_count += 1
                
                pbar.update(1)
    
    # Summary
    log("RENAME OPERATION COMPLETE", "HEADER")
    log(f"Total files processed: {total_files}", "INFO")
    log(f"Files renamed: {renamed_count}", "SUCCESS")
    log(f"Files skipped: {skipped_count}", "WARNING")
    log(f"Errors encountered: {error_count}", "ERROR" if error_count > 0 else "INFO")
    
    return renamed_count, skipped_count, error_count

def rollback_changes(backup_file):
    """Rollback file renames using backup record"""
    if not os.path.exists(backup_file):
        log("No backup file found for rollback", "ERROR")
        return
    
    log("Starting rollback process", "HEADER")
    
    rollback_count = 0
    error_count = 0
    
    with open(backup_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                renamed_path = Path(record['renamed'])
                original_path = Path(record['original'])
                
                if renamed_path.exists():
                    renamed_path.rename(original_path)
                    log(f"Rolled back: {renamed_path.name} -> {original_path.name}", "SUCCESS")
                    rollback_count += 1
                else:
                    log(f"File not found for rollback: {renamed_path}", "WARNING")
                    
            except Exception as e:
                log(f"Error during rollback: {str(e)}", "ERROR")
                error_count += 1
    
    log(f"Rollback complete. Restored {rollback_count} files with {error_count} errors", "INFO")

def main():
    """Main function"""
    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print("FOLDER-BASED FILE RENAMER")
    print(f"{'='*60}{Colors.ENDC}\n")
    
    # Setup dependencies
    if not setup_dependencies():
        log("Failed to setup dependencies. Exiting.", "ERROR")
        return
    
    # Configuration through user input
    log("\nCONFIGURATION", "HEADER")
    
    # Get directory to process
    while True:
        directory = get_user_input("Enter the directory path to scan")
        if validate_directory(directory):
            directory = os.path.abspath(directory)
            log(f"Valid directory: {directory}", "SUCCESS")
            break
        else:
            log("Invalid directory path. Please try again.", "ERROR")
    
    # Dry run option
    dry_run_choice = get_user_input("Perform a dry run first? (yes/no)", "yes", ["yes", "no"])
    dry_run = dry_run_choice == "yes"
    
    # Backup option
    create_backup = get_user_input("Create backup record for rollback? (yes/no)", "yes", ["yes", "no"])
    
    backup_file = None
    if create_backup == "yes":
        backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rename_backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f"rename_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        log(f"Backup file: {backup_file}", "INFO")
    
    # Show summary
    log("\nOPERATION SUMMARY", "HEADER")
    log(f"Directory: {directory}", "INFO")
    log(f"Dry run: {'Yes' if dry_run else 'No'}", "INFO")
    log(f"Create backup: {'Yes' if create_backup == 'yes' else 'No'}", "INFO")
    
    # Confirmation
    confirm = get_user_input("\nProceed with the operation? (yes/no)", "yes", ["yes", "no"])
    
    if confirm != "yes":
        log("Operation cancelled by user", "WARNING")
        return
    
    # Perform the rename operation
    start_time = time.time()
    renamed, skipped, errors = rename_files_in_directory(directory, dry_run, backup_file)
    elapsed_time = time.time() - start_time
    
    log(f"\nOperation completed in {elapsed_time:.2f} seconds", "INFO")
    
    # If it was a dry run, ask if they want to proceed with actual rename
    if dry_run and renamed > 0:
        proceed = get_user_input("\nDry run complete. Proceed with actual rename? (yes/no)", "no", ["yes", "no"])
        if proceed == "yes":
            log("\nExecuting actual rename operation...", "HEADER")
            renamed, skipped, errors = rename_files_in_directory(directory, False, backup_file)
    
    # Offer rollback option if backup was created and files were renamed
    if backup_file and renamed > 0 and not dry_run:
        rollback = get_user_input("\nWould you like to rollback the changes? (yes/no)", "no", ["yes", "no"])
        if rollback == "yes":
            rollback_changes(backup_file)
    
    log("\nScript execution complete", "HEADER")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n\nOperation interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"\n\nUnexpected error: {str(e)}", "ERROR")
        sys.exit(1)
