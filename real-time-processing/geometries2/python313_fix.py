#!/usr/bin/env python3
"""
Python 3.13 OpenCV Installation Helper
======================================

Python 3.13 is very new and OpenCV might not have wheels yet.
This script helps find and install compatible versions.
"""

import subprocess
import sys
import json
import urllib.request

def get_opencv_versions():
    """Fetch available OpenCV versions from PyPI"""
    print("Checking available OpenCV versions...")
    
    try:
        url = "https://pypi.org/pypi/opencv-python/json"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            versions = list(data["releases"].keys())
            
        # Filter for recent stable versions
        stable_versions = [v for v in versions if not any(x in v for x in ['rc', 'a', 'b'])]
        stable_versions.sort(reverse=True)
        
        return stable_versions[:10]  # Latest 10 versions
    except:
        return []

def try_install_opencv(version=None):
    """Try to install OpenCV with specific version"""
    package = "opencv-python" if version is None else f"opencv-python=={version}"
    
    print(f"\nTrying to install {package}...")
    
    result = subprocess.run(['uv', 'pip', 'install', package], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Successfully installed {package}")
        return True
    else:
        print(f"✗ Failed to install {package}")
        if "No matching distribution" in result.stderr:
            print("  (No compatible wheel for Python 3.13)")
        return False

def main():
    print("Python 3.13 OpenCV Installation Helper")
    print("="*40)
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {py_version}")
    
    if sys.version_info.minor < 13:
        print("You're not using Python 3.13. Regular installation should work.")
        return
    
    # Get available versions
    versions = get_opencv_versions()
    
    if not versions:
        print("Could not fetch version list from PyPI")
        versions = ["4.10.0.84", "4.9.0.80", "4.8.1.78"]  # Fallback list
    
    print(f"\nFound {len(versions)} OpenCV versions")
    
    # Try latest version first
    if try_install_opencv():
        return
    
    # Try specific versions
    print("\nTrying specific versions...")
    for version in versions:
        if try_install_opencv(version):
            # Also install contrib
            contrib_pkg = f"opencv-contrib-python=={version}"
            subprocess.run(['uv', 'pip', 'install', contrib_pkg])
            return
    
    # If all fails, suggest alternatives
    print("\n" + "="*40)
    print("❌ Could not find compatible OpenCV for Python 3.13")
    print("\nSuggested solutions:")
    print("1. Use Python 3.11 or 3.12:")
    print("   uv venv --python 3.11")
    print("   uv pip install opencv-python")
    print("\n2. Build from source (advanced):")
    print("   https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html")
    print("\n3. Wait for OpenCV to release Python 3.13 wheels")
    print("   Check: https://pypi.org/project/opencv-python/")
    print("\n4. Try installing without version constraints:")
    print("   uv pip install --pre opencv-python")

if __name__ == "__main__":
    main()
