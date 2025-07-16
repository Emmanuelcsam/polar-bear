#!/usr/bin/env python3
import os
import shutil
from collections import defaultdict
import random

# Define color categories and their associated folder branches
COLOR_BRANCH_MAPPING = {
    # Blue/Cyan/Teal colors -> fc branch
    'cyan': 'fc',
    'teal': 'fc',
    'lightgray': 'fc',  # fc gets light grays
    
    # Pink/Purple colors -> sma branch
    'pink': 'sma',
    'purple': 'sma',
    
    # Dark colors -> anomaly folders
    'darkred': 'anomaly',
    'darkgray': 'anomaly',
    'darkgreen': 'anomaly',
    
    # Clean/White colors -> clean folders
    'white': 'clean',
    'lime': 'clean',
    
    # Gray colors -> various folders (will distribute)
    'gray': 'general',
    
    # Orange -> dirty folders
    'orange': 'dirty',
}

def get_color_from_filename(filename):
    """Extract color name from filename"""
    basename = os.path.basename(filename).lower()
    
    # Check for color prefixes
    for color in COLOR_BRANCH_MAPPING.keys():
        if basename.startswith(color):
            return color
    
    return None

def get_all_folders_by_type():
    """Get all folders organized by their type"""
    folders_by_type = defaultdict(list)
    
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            rel_path = os.path.relpath(os.path.join(root, dir_name), '.')
            
            # Categorize by folder name
            folder_basename = os.path.basename(rel_path)
            
            # Special handling for branch-specific folders
            if 'fc' in rel_path:
                folders_by_type['fc'].append(rel_path)
            elif 'sma' in rel_path:
                folders_by_type['sma'].append(rel_path)
            
            # Type-based categorization
            if folder_basename == 'anomaly':
                folders_by_type['anomaly'].append(rel_path)
            elif folder_basename == 'clean':
                folders_by_type['clean'].append(rel_path)
            elif folder_basename == 'dirty':
                folders_by_type['dirty'].append(rel_path)
            elif folder_basename == 'blob':
                folders_by_type['blob'].append(rel_path)
            elif folder_basename == 'oil':
                folders_by_type['oil'].append(rel_path)
            elif folder_basename == 'scratched':
                folders_by_type['scratched'].append(rel_path)
            elif folder_basename == 'cladding':
                folders_by_type['cladding'].append(rel_path)
            elif folder_basename == 'core':
                folders_by_type['core'].append(rel_path)
            elif folder_basename == 'ferrule':
                folders_by_type['ferrule'].append(rel_path)
    
    return folders_by_type

def distribute_colors():
    """Distribute color files to appropriate folders"""
    # Get all JPG files
    jpg_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                jpg_files.append(os.path.join(root, file))
    
    # Group files by color
    files_by_color = defaultdict(list)
    for jpg_file in jpg_files:
        color = get_color_from_filename(jpg_file)
        if color:
            files_by_color[color].append(jpg_file)
    
    # Get folders organized by type
    folders_by_type = get_all_folders_by_type()
    
    # Distribution plan
    distribution_plan = []
    
    # Specific color distributions
    color_distributions = {
        # Blue/Cyan/Teal -> fc branch folders
        'cyan': [f for f in folders_by_type['fc'] if 'blob' in f or 'cladding' in f],
        'teal': [f for f in folders_by_type['fc'] if 'oil' in f or 'ferrule' in f],
        'lightgray': [f for f in folders_by_type['fc'] if 'anomaly' in f],
        
        # Pink/Purple -> sma branch folders
        'pink': [f for f in folders_by_type['sma'] if 'anomaly' in f or 'blob' in f],
        'purple': [f for f in folders_by_type['sma'] if 'cladding' in f],
        
        # Dark colors -> anomaly folders across all branches
        'darkred': folders_by_type['anomaly'],
        'darkgray': folders_by_type['anomaly'] + folders_by_type['dirty'] + folders_by_type['dig'],
        'darkgreen': folders_by_type['oil'],
        
        # Clean colors -> clean folders
        'white': folders_by_type['clean'],
        'lime': folders_by_type['clean'],
        
        # Gray -> distribute across various folders
        'gray': folders_by_type['cladding'] + folders_by_type['core'] + folders_by_type['ferrule'] + folders_by_type['scratched'],
        
        # Orange -> dirty folders
        'orange': folders_by_type['dirty'],
    }
    
    # Create distribution plan
    for color, files in files_by_color.items():
        target_folders = color_distributions.get(color, [])
        
        if not target_folders:
            print(f"Warning: No target folders for color {color}")
            continue
        
        # Distribute files evenly across target folders
        for i, file in enumerate(files):
            target_folder = target_folders[i % len(target_folders)]
            new_filename = f"{color}_{i+1}.jpg"
            new_path = os.path.join(target_folder, new_filename)
            
            distribution_plan.append({
                'source': file,
                'destination': new_path,
                'color': color,
                'target_folder': target_folder
            })
    
    return distribution_plan

def execute_distribution(plan, dry_run=False):
    """Execute the distribution plan"""
    print(f"Distribution plan contains {len(plan)} moves")
    
    if dry_run:
        print("\nDRY RUN - No files will be moved\n")
    
    # Group by color for summary
    moves_by_color = defaultdict(list)
    for move in plan:
        moves_by_color[move['color']].append(move)
    
    # Execute moves
    for color, moves in moves_by_color.items():
        print(f"\n{color.upper()} files ({len(moves)} files):")
        
        for move in moves[:5]:  # Show first 5 examples
            print(f"  {os.path.basename(move['source'])} -> {move['destination']}")
        
        if len(moves) > 5:
            print(f"  ... and {len(moves) - 5} more")
        
        if not dry_run:
            for move in moves:
                try:
                    # Create target directory if it doesn't exist
                    os.makedirs(os.path.dirname(move['destination']), exist_ok=True)
                    
                    # Move the file
                    shutil.move(move['source'], move['destination'])
                    
                except Exception as e:
                    print(f"  Error moving {move['source']}: {e}")
    
    if not dry_run:
        print("\nDistribution complete!")
    else:
        print("\nTo execute the distribution, run with dry_run=False")

def main():
    print("Analyzing color distribution...")
    
    # Create distribution plan
    plan = distribute_colors()
    
    # Show summary
    print(f"\nFound {len(plan)} files to distribute")
    
    # Execute distribution (set dry_run=False to actually move files)
    execute_distribution(plan, dry_run=True)
    
    print("\n" + "="*50)
    print("EXECUTING FILE DISTRIBUTION...")
    print("="*50)
    
    # Execute the distribution
    execute_distribution(plan, dry_run=False)

if __name__ == "__main__":
    main()