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
    """Copy color files to their assigned folders"""
    print("\nAssigning colors to folders...")
    
    for i, assignment in enumerate(assignments):
        folder = assignment['folder']
        color_file = assignment['color_file']
        color_name = assignment['color_name']
        
        # Destination path
        dest_path = os.path.join(folder, 'color.jpg')
        
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
        f.write("="*50 + "\n\n")
        
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
            f.write("-"*30 + "\n")
            
            for item in sorted(items, key=lambda x: x['folder']):
                f.write(f"  {item['folder']:<40} -> {item['color_name']}\n")
    
    print("Created folder_color_assignments.txt reference file")

def main():
    # Clean up any existing color files first
    print("Cleaning up existing color files...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    
    print("\n" + "="*50)
    
    # Get assignments
    assignments = assign_colors_to_folders()
    
    # Execute assignment
    execute_color_assignment(assignments)
    
    # Create reference file
    create_color_map_file(assignments)

if __name__ == "__main__":
    main()