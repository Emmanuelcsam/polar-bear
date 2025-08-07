#!/usr/bin/env python3
"""
Location Setter Script
"""

import os
import sys
import glob
from pathlib import Path

def replace_in_file(file_path, old_path, new_path):
    """
    Replace old_path with new_path in the specified file.
    Returns True if replacements were made, False otherwise.
    """
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Check if the old path exists in the file
        if old_path in content:
            # Replace all instances
            new_content = content.replace(old_path, new_path)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def main():
    # Get the current directory
    current_dir = Path.cwd()
    
    # The old path to replace
    old_path = "E:\\GitHub\\polar-bear"
    
    # Get the new path from user input
    if len(sys.argv) > 1:
        new_path = sys.argv[1]
    else:
        new_path = input("Enter the new path to replace 'E:\\GitHub\\polar-bear' with: ").strip()
    
    if not new_path:
        print("Error: No new path provided.")
        return
    
    # Normalize the path (convert forward slashes to backslashes on Windows if needed)
    new_path = os.path.normpath(new_path)
    
    print(f"Replacing '{old_path}' with '{new_path}' in all files...")
    print(f"Working in directory: {current_dir}")
    print("-" * 50)
    
    # Get all files in current directory (excluding this script)
    all_files = []
    for file_path in current_dir.iterdir():
        if file_path.is_file() and file_path.name != 'location-setter.py':
            all_files.append(file_path)
    
    if not all_files:
        print("No files found to process.")
        return
    
    processed_files = 0
    modified_files = 0
    
    # Process each file
    for file_path in all_files:
        try:
            print(f"Processing: {file_path.name}... ", end="")
            
            if replace_in_file(file_path, old_path, new_path):
                print("✓ Modified")
                modified_files += 1
            else:
                print("- No changes needed")
            
            processed_files += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("-" * 50)
    print(f"Summary:")
    print(f"Files processed: {processed_files}")
    print(f"Files modified: {modified_files}")
    print(f"Old path: {old_path}")
    print(f"New path: {new_path}")
    
    if modified_files > 0:
        print("\n✓ Replacement completed successfully!")
    else:
        print("\n- No files needed modification.")

if __name__ == "__main__":
    main()
