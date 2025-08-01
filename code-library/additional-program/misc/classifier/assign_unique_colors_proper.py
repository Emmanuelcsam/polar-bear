#!/usr/bin/env python3
import os
import shutil
import random

def get_all_folders():
    """Get all folders in current directory"""
    folders = []
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            rel_path = os.path.relpath(os.path.join(root, dir_name), '.')
            folders.append(rel_path)
    return sorted(folders)

def get_color_files():
    """Get all color files from the colors directory"""
    colors_dir = '/home/jarvis/Documents/GitHub/polar-bear/training/classifier/colors'
    color_files = []
    for file in os.listdir(colors_dir):
        if file.endswith('.jpg'):
            color_files.append(os.path.join(colors_dir, file))
    return sorted(color_files)

def clean_existing_files():
    """Remove all existing jpg files"""
    print("Cleaning up existing JPG files...")
    removed_count = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")
                removed_count += 1
    print(f"Removed {removed_count} files\n")

def assign_colors_to_folders():
    """Assign one unique color to each folder"""
    folders = get_all_folders()
    color_files = get_color_files()
    
    print(f"Found {len(folders)} folders")
    print(f"Found {len(color_files)} color files")
    
    if len(color_files) < len(folders):
        print(f"Warning: Not enough unique colors ({len(color_files)}) for all folders ({len(folders)})")
        print("Some colors will be reused")
    
    # Shuffle colors for random distribution
    random.shuffle(color_files)
    
    # Create assignment plan
    assignments = []
    for i, folder in enumerate(folders):
        # Use modulo to cycle through colors if we run out
        color_file = color_files[i % len(color_files)]
        color_name = os.path.basename(color_file)
        
        assignments.append({
            'folder': folder,
            'color_file': color_file,
            'color_name': color_name
        })
    
    return assignments

def execute_color_assignment(assignments):
    """Copy color files to their assigned folders with ORIGINAL NAMES"""
    print("\nAssigning colors to folders (keeping original color names)...")
    
    for i, assignment in enumerate(assignments):
        folder = assignment['folder']
        color_file = assignment['color_file']
        color_name = assignment['color_name']
        
        # Destination path - KEEP THE ORIGINAL COLOR NAME
        dest_path = os.path.join(folder, color_name)
        
        try:
            # Copy the color file
            shutil.copy2(color_file, dest_path)
            print(f"{i+1}/{len(assignments)}: {folder} -> {color_name}")
        except Exception as e:
            print(f"Error copying to {folder}: {e}")
    
    print("\nColor assignment complete!")

def create_color_map_file(assignments):
    """Create a reference file showing which color went to which folder"""
    with open('folder_color_assignments.txt', 'w') as f:
        f.write("Folder Color Assignments\n")
        f.write("="*60 + "\n\n")
        f.write(f"{'Folder':<45} | Color File\n")
        f.write("-"*60 + "\n")
        
        # Group by branch
        branch_groups = {}
        for assignment in assignments:
            folder = assignment['folder']
            branch = folder.split('/')[0]
            
            if branch not in branch_groups:
                branch_groups[branch] = []
            branch_groups[branch].append(assignment)
        
        # Write grouped assignments
        for branch, items in sorted(branch_groups.items()):
            f.write(f"\n{branch.upper()} Branch:\n")
            f.write("-"*60 + "\n")
            
            for item in sorted(items, key=lambda x: x['folder']):
                f.write(f"{item['folder']:<45} | {item['color_name']}\n")
    
    print("Created folder_color_assignments.txt reference file")

def create_visual_summary():
    """Create a summary showing color distribution"""
    print("\nCreating visual summary...")
    
    # Count how many folders have each color type
    color_counts = {}
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.jpg'):
                color_base = file.split('_')[0] if '_' in file else file.replace('.jpg', '')
                color_counts[color_base] = color_counts.get(color_base, 0) + 1
    
    print("\nColor Distribution Summary:")
    print("-"*30)
    for color, count in sorted(color_counts.items()):
        print(f"{color:<15}: {count} folders")

def main():
    # Clean up any existing files first
    clean_existing_files()
    
    print("="*60)
    
    # Get assignments
    assignments = assign_colors_to_folders()
    
    # Execute assignment
    execute_color_assignment(assignments)
    
    # Create reference file
    create_color_map_file(assignments)
    
    # Create visual summary
    create_visual_summary()

if __name__ == "__main__":
    main()