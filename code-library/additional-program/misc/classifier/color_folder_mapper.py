#!/usr/bin/env python3
import os
import json
import colorsys
from pathlib import Path

def generate_color_from_hue(hue, saturation=0.7, value=0.9):
    """Generate RGB color from HSV values"""
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(x * 255) for x in rgb)

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string"""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def create_color_mapping(root_dir='.'):
    """Create color mapping for all folders in directory tree"""
    # Get all directories
    all_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            rel_path = os.path.relpath(os.path.join(root, dir_name), root_dir)
            all_dirs.append(rel_path)
    
    # Sort for consistent ordering
    all_dirs.sort()
    
    # Color mapping dictionary
    color_map = {}
    
    # Track top-level directories for base colors
    top_level_dirs = [d for d in all_dirs if '/' not in d]
    
    # Assign base hues to top-level directories
    base_hues = {}
    hue_step = 1.0 / len(top_level_dirs) if top_level_dirs else 1.0
    
    for i, top_dir in enumerate(top_level_dirs):
        base_hue = i * hue_step
        base_hues[top_dir] = base_hue
        # Assign color to top-level directory
        color = generate_color_from_hue(base_hue)
        color_map[top_dir] = {
            'rgb': color,
            'hex': rgb_to_hex(color),
            'depth': 0,
            'branch': top_dir
        }
    
    # Process subdirectories
    for dir_path in all_dirs:
        if dir_path in color_map:
            continue
            
        parts = dir_path.split('/')
        depth = len(parts) - 1
        
        # Find the top-level parent
        top_parent = parts[0]
        
        if top_parent in base_hues:
            base_hue = base_hues[top_parent]
            
            # Adjust hue slightly based on path components
            # This creates similar but distinct colors for same branch
            path_hash = sum(ord(c) for c in dir_path)
            hue_variation = (path_hash % 20 - 10) / 100  # ±10% variation
            adjusted_hue = (base_hue + hue_variation) % 1.0
            
            # Adjust saturation and value based on depth
            # Deeper folders get slightly different saturation/brightness
            saturation = 0.7 - (depth * 0.05)  # Decrease saturation with depth
            saturation = max(0.3, saturation)  # Keep minimum saturation
            
            value = 0.9 - (depth * 0.03)  # Slightly darker with depth
            value = max(0.6, value)  # Keep minimum brightness
            
            color = generate_color_from_hue(adjusted_hue, saturation, value)
            color_map[dir_path] = {
                'rgb': color,
                'hex': rgb_to_hex(color),
                'depth': depth,
                'branch': top_parent
            }
    
    return color_map

def save_color_files(color_map):
    """Save color files in each directory"""
    for dir_path, color_info in color_map.items():
        # Create color.json file in each directory
        full_path = os.path.join('.', dir_path)
        color_file = os.path.join(full_path, 'color.json')
        
        # Ensure directory exists
        os.makedirs(full_path, exist_ok=True)
        
        # Save color information
        with open(color_file, 'w') as f:
            json.dump({
                'path': dir_path,
                'color': color_info,
                'description': f"Color for {dir_path} (branch: {color_info['branch']}, depth: {color_info['depth']})"
            }, f, indent=2)

def create_visualization_html(color_map):
    """Create an HTML visualization of the folder structure with colors"""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Folder Color Map Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .folder {
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .color-box {
            width: 30px;
            height: 30px;
            margin-right: 15px;
            border-radius: 3px;
            border: 1px solid #333;
        }
        .folder-info {
            flex-grow: 1;
        }
        .path {
            font-weight: bold;
            font-size: 14px;
        }
        .details {
            font-size: 12px;
            color: #666;
            margin-top: 3px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .branch-group {
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .branch-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Folder Color Mapping Visualization</h1>
"""
    
    # Group folders by branch
    branches = {}
    for path, info in sorted(color_map.items()):
        branch = info['branch']
        if branch not in branches:
            branches[branch] = []
        branches[branch].append((path, info))
    
    # Generate HTML for each branch
    for branch, folders in sorted(branches.items()):
        branch_color = color_map.get(branch, {}).get('hex', '#000000')
        html_content += f"""
        <div class="branch-group">
            <div class="branch-title" style="background-color: {branch_color};">
                Branch: {branch}
            </div>
"""
        
        for path, info in folders:
            indent = "&nbsp;" * (info['depth'] * 4)
            html_content += f"""
            <div class="folder" style="margin-left: {info['depth'] * 20}px;">
                <div class="color-box" style="background-color: {info['hex']};"></div>
                <div class="folder-info">
                    <div class="path">{indent}{path}</div>
                    <div class="details">
                        RGB: {info['rgb']} | HEX: {info['hex']} | Depth: {info['depth']}
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    print("Creating color mapping for all folders...")
    
    # Create color mapping
    color_map = create_color_mapping()
    
    # Save color files in each directory
    print("Saving color.json files in each directory...")
    save_color_files(color_map)
    
    # Create master mapping file
    print("Creating master color mapping file...")
    with open('folder_color_map.json', 'w') as f:
        json.dump(color_map, f, indent=2)
    
    # Create HTML visualization
    print("Creating HTML visualization...")
    html_content = create_visualization_html(color_map)
    with open('folder_color_visualization.html', 'w') as f:
        f.write(html_content)
    
    # Print summary
    print(f"\nColor mapping complete!")
    print(f"Total folders processed: {len(color_map)}")
    print(f"Branch distribution:")
    
    branch_counts = {}
    for info in color_map.values():
        branch = info['branch']
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    
    for branch, count in sorted(branch_counts.items()):
        print(f"  {branch}: {count} folders")
    
    print("\nFiles created:")
    print("  - color.json in each directory")
    print("  - folder_color_map.json (master mapping)")
    print("  - folder_color_visualization.html (visual reference)")

if __name__ == "__main__":
    main()