#!/usr/bin/env python3
"""
Process img(303).jpg and create visualization
This script will attempt to process the image and generate outputs
"""

import subprocess
import sys
import os
from pathlib import Path

def install_and_run():
    """Install dependencies and run processing"""
    
    base_dir = Path(__file__).parent
    test_image = base_dir / "test_image" / "img(303).jpg"
    
    print("="*60)
    print("PROCESSING img(303).jpg WITH DEFECT DETECTION")
    print("="*60)
    
    # Check if test image exists
    if not test_image.exists():
        print(f"Error: Test image not found at {test_image}")
        return
    
    print(f"✓ Found test image: {test_image}")
    print(f"  Size: {test_image.stat().st_size:,} bytes")
    
    # Try to set up a virtual environment
    venv_dir = base_dir / "venv"
    
    print("\n1. Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print("✓ Virtual environment created")
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return
    
    # Determine the pip path in venv
    if os.name == 'nt':  # Windows
        pip_path = venv_dir / "Scripts" / "pip"
        python_path = venv_dir / "Scripts" / "python"
    else:  # Unix/Linux/Mac
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
    
    # Install required packages
    print("\n2. Installing required packages...")
    packages = ["opencv-python", "numpy", "pillow", "matplotlib"]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.run([str(pip_path), "install", package], 
                         capture_output=True, check=True)
            print(f"   ✓ {package} installed")
        except Exception as e:
            print(f"   ✗ Failed to install {package}: {e}")
    
    # Create a processing script
    process_script = base_dir / "run_processing.py"
    
    script_content = '''#!/usr/bin/env python3
"""Process img(303).jpg and create visualization"""

import cv2
import numpy as np
from pathlib import Path
import json

def process_image():
    base_dir = Path(__file__).parent
    test_image_path = base_dir / "test_image" / "img(303).jpg"
    
    # Load image
    img = cv2.imread(str(test_image_path))
    if img is None:
        print("Failed to load image")
        return
    
    print(f"Image loaded: {img.shape}")
    
    # Simple defect detection using edge detection and thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply various filters to detect defects
    # 1. Edge detection for scratches
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Threshold for surface defects
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphological operations to find blobs
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (defect regions)
    contours, _ = cv2.findContours(edges | morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization
    result = img.copy()
    defect_mask = np.zeros_like(gray)
    
    defects = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify defect type based on shape
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 3 or aspect_ratio < 0.3:
                defect_type = "scratch"
                color = (0, 0, 255)  # Red
            elif area > 1000:
                defect_type = "blob"
                color = (255, 0, 0)  # Blue
            else:
                defect_type = "dig"
                color = (0, 255, 0)  # Green
            
            # Draw on result
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result, f"{defect_type} #{i}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Fill defect mask
            cv2.drawContours(defect_mask, [contour], -1, 255, -1)
            
            defects.append({
                "type": defect_type,
                "bbox": [x, y, w, h],
                "area": area
            })
    
    # Add summary text
    summary = [
        f"Total Defects: {len(defects)}",
        f"Scratches: {sum(1 for d in defects if d['type'] == 'scratch')}",
        f"Digs: {sum(1 for d in defects if d['type'] == 'dig')}",
        f"Blobs: {sum(1 for d in defects if d['type'] == 'blob')}",
        "Status: FAIL" if len(defects) > 5 else "Status: PASS"
    ]
    
    # Draw summary box
    cv2.rectangle(result, (10, 10), (250, 120), (255, 255, 255), -1)
    cv2.rectangle(result, (10, 10), (250, 120), (0, 0, 0), 2)
    
    y_offset = 30
    for text in summary:
        cv2.putText(result, text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_offset += 20
    
    # Save outputs
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "img303_original.jpg"), img)
    cv2.imwrite(str(output_dir / "img303_defect_overlay.jpg"), result)
    cv2.imwrite(str(output_dir / "img303_defect_mask.jpg"), defect_mask)
    cv2.imwrite(str(output_dir / "img303_edges.jpg"), edges)
    
    # Save defect data
    with open(output_dir / "defects.json", 'w') as f:
        json.dump({
            "image": "img(303).jpg",
            "total_defects": len(defects),
            "defects": defects,
            "summary": summary
        }, f, indent=2)
    
    print(f"\\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\\nSummary:")
    for line in summary:
        print(f"  {line}")

if __name__ == "__main__":
    process_image()
'''
    
    with open(process_script, 'w') as f:
        f.write(script_content)
    
    # Run the processing script
    print("\n3. Running defect detection...")
    try:
        result = subprocess.run([str(python_path), str(process_script)], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"✗ Failed to run processing: {e}")
    
    # Check outputs
    output_dir = base_dir / "output"
    if output_dir.exists():
        print("\n4. Generated outputs:")
        for file in output_dir.iterdir():
            print(f"   ✓ {file.name} ({file.stat().st_size:,} bytes)")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print("\nTo view results:")
    print(f"1. Check the 'output' directory for generated images")
    print(f"2. Open img303_defect_overlay.jpg to see defects highlighted")
    print(f"3. View defects.json for detailed defect information")

if __name__ == "__main__":
    install_and_run()