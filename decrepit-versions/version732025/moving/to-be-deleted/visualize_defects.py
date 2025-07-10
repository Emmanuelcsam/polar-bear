#!/usr/bin/env python3
"""
Visualize defects from existing numpy arrays
This script attempts to work with minimal dependencies
"""

import os
import sys
from pathlib import Path

# Try to install required packages if not available
def install_package(package_name):
    """Try to install a package using subprocess"""
    import subprocess
    
    print(f"Attempting to install {package_name}...")
    
    # Try different installation methods
    methods = [
        [sys.executable, "-m", "pip", "install", "--user", package_name],
        ["pip3", "install", "--user", package_name],
        ["pip", "install", "--user", package_name],
    ]
    
    for method in methods:
        try:
            result = subprocess.run(method, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully installed {package_name}")
                return True
        except:
            continue
    
    return False

# Forward declaration of the function
def create_simple_visualization():
    """Create visualization without OpenCV using only PIL"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        print("\nUnable to create visualization without required packages.")
        print("Please install manually:")
        print("  python3 -m pip install --user opencv-python numpy pillow")
        return
    
    base_dir = Path(__file__).parent
    test_image_path = base_dir / "test_image" / "img(303).jpg"
    
    # Load test image
    try:
        img = Image.open(test_image_path)
        draw = ImageDraw.Draw(img)
        
        # Read the detailed report to get defect locations
        report_path = base_dir / "results" / "img (303)" / "3_detected" / "img (303)" / "img (303)_detailed.txt"
        
        if report_path.exists():
            with open(report_path, 'r') as f:
                content = f.read()
            
            # Parse defect regions
            regions = []
            lines = content.split('\n')
            in_region_section = False
            
            for i, line in enumerate(lines):
                if "LOCAL ANOMALY REGIONS" in line:
                    in_region_section = True
                elif "SPECIFIC DEFECTS DETECTED" in line:
                    in_region_section = False
                elif in_region_section and "Location:" in line:
                    # Extract coordinates
                    coords_str = line.split("(")[1].split(")")[0]
                    coords = [int(x.strip()) for x in coords_str.split(",")]
                    x, y, w, h = coords
                    regions.append((x, y, w, h))
            
            # Draw rectangles for each region
            for x, y, w, h in regions[:10]:  # Limit to first 10 for visibility
                draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
            
            # Add text
            draw.text((10, 10), f"Defects Found: {len(regions)}", fill='red')
            
            # Save result
            output_path = base_dir / "img303_defect_overlay_simple.jpg"
            img.save(output_path)
            print(f"\n✓ Simple visualization saved to: {output_path}")
            
    except Exception as e:
        print(f"Error creating simple visualization: {e}")

# Check and install dependencies
packages_needed = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'PIL': 'pillow'
}

missing_packages = []

for module_name, package_name in packages_needed.items():
    try:
        if module_name == 'cv2':
            import cv2
        elif module_name == 'numpy':
            import numpy
        elif module_name == 'PIL':
            from PIL import Image
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("Missing packages detected. Attempting to install...")
    for package in missing_packages:
        if not install_package(package):
            print(f"✗ Failed to install {package}")
    
    # Try to import again
    try:
        import cv2
        import numpy as np
        from PIL import Image
        print("\n✓ All packages successfully imported!")
    except ImportError as e:
        print(f"\n✗ Still missing packages after installation attempt: {e}")
        print("\nTrying alternative approach without OpenCV...")
        
        # Alternative: Create a simple visualization without OpenCV
        create_simple_visualization()
        sys.exit(0)
else:
    import cv2
    import numpy as np
    from PIL import Image

def main():
    """Main visualization function"""
    print("="*60)
    print("DEFECT VISUALIZATION FOR img(303).jpg")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    # Paths
    test_image_path = base_dir / "test_image" / "img(303).jpg"
    results_dir = base_dir / "results" / "img (303)" / "3_detected"
    
    # Load original image
    print(f"\nLoading test image: {test_image_path}")
    image = cv2.imread(str(test_image_path))
    
    if image is None:
        print(f"✗ Failed to load image: {test_image_path}")
        return
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Create overlays
    defect_overlay = image.copy()
    zone_overlay = image.copy()
    combined_overlay = image.copy()
    
    # Load and apply defect masks
    print("\nLoading defect masks...")
    
    regions = {
        "img (303)": {"color": (0, 0, 255), "label": "Full Image"},
        "region_core": {"color": (0, 255, 0), "label": "Core"},
        "region_cladding": {"color": (255, 255, 0), "label": "Cladding"},
        "region_ferrule": {"color": (255, 0, 0), "label": "Ferrule"}
    }
    
    for region_name, info in regions.items():
        mask_path = results_dir / region_name / f"{region_name}_defect_mask.npy"
        
        if mask_path.exists():
            print(f"  Loading {region_name} mask...")
            try:
                # Load numpy array
                defect_mask = np.load(str(mask_path))
                print(f"    ✓ Mask shape: {defect_mask.shape}")
                
                # Apply color to defect regions
                if region_name == "img (303)":
                    # For full image, use red overlay
                    defect_regions = defect_mask > 0
                    defect_overlay[defect_regions] = cv2.addWeighted(
                        defect_overlay[defect_regions], 0.5,
                        np.full_like(defect_overlay[defect_regions], info["color"]), 0.5, 0
                    )
                
            except Exception as e:
                print(f"    ✗ Error loading mask: {e}")
    
    # Parse detailed report for defect locations
    detailed_report = results_dir / "img (303)" / "img (303)_detailed.txt"
    
    if detailed_report.exists():
        print(f"\nParsing detailed report...")
        with open(detailed_report, 'r') as f:
            content = f.read()
        
        # Extract defect statistics
        defect_stats = {}
        for line in content.split('\n'):
            if "Scratches:" in line and "DEFECTS" in content[:content.index(line)]:
                defect_stats['scratches'] = int(line.split(':')[1].strip())
            elif "Digs:" in line and "DEFECTS" in content[:content.index(line)]:
                defect_stats['digs'] = int(line.split(':')[1].strip())
            elif "Blobs:" in line and "DEFECTS" in content[:content.index(line)]:
                defect_stats['blobs'] = int(line.split(':')[1].strip())
            elif "Edge Irregularities:" in line:
                defect_stats['edge_irregularities'] = int(line.split(':')[1].strip())
        
        # Extract region coordinates
        regions_found = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if "Location:" in line and "(" in line and ")" in line:
                try:
                    coords_str = line.split("(")[1].split(")")[0]
                    coords = [int(x.strip()) for x in coords_str.split(",")]
                    if len(coords) == 4:
                        x, y, w, h = coords
                        
                        # Get confidence if available
                        confidence = 1.0
                        for j in range(i, min(i+5, len(lines))):
                            if "Confidence:" in lines[j]:
                                conf_str = lines[j].split(":")[1].strip()
                                confidence = float(conf_str)
                                break
                        
                        regions_found.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence
                        })
                except:
                    continue
        
        print(f"  ✓ Found {len(regions_found)} defect regions")
        
        # Draw bounding boxes on combined overlay
        for i, region in enumerate(regions_found[:20]):  # Limit to first 20 for visibility
            x, y, w, h = region['bbox']
            conf = region['confidence']
            
            # Color based on confidence
            if conf > 0.8:
                color = (0, 0, 255)  # Red for high confidence
            elif conf > 0.6:
                color = (0, 165, 255)  # Orange for medium
            else:
                color = (0, 255, 255)  # Yellow for low
            
            # Draw rectangle
            cv2.rectangle(combined_overlay, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"#{i+1} ({conf:.2f})"
            cv2.putText(combined_overlay, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Add summary text to overlays
    summary_text = [
        f"Total Defects: 99",
        f"Scratches: 27",
        f"Digs: 68",
        f"Blobs: 1",
        f"Edge Irregularities: 3",
        f"Status: FAIL"
    ]
    
    y_offset = 30
    for text in summary_text:
        cv2.putText(combined_overlay, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    
    # Save outputs
    output_dir = base_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    outputs = {
        "img303_original.jpg": image,
        "img303_defect_overlay.jpg": defect_overlay,
        "img303_defect_boxes.jpg": combined_overlay
    }
    
    print(f"\nSaving visualizations to {output_dir}...")
    for filename, img in outputs.items():
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), img)
        print(f"  ✓ Saved: {filename}")
    
    # Also save in the main directory for easy access
    main_output = base_dir / "img303_defect_visualization.jpg"
    cv2.imwrite(str(main_output), combined_overlay)
    print(f"\n✓ Main output saved to: {main_output}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError in main visualization: {e}")
        print("Attempting simple visualization...")
        create_simple_visualization()