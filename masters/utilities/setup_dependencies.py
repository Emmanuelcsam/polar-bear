#!/usr/bin/env python3
"""Install dependencies for Automated Processing Studio"""

import subprocess
import sys

REQUIRED_PACKAGES = {
    'opencv-python': 'cv2',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'pillow': 'PIL',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy'
}

print("🔍 Installing dependencies for Automated Processing Studio...")

for package_name in REQUIRED_PACKAGES:
    print(f"\n📦 Installing {package_name}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ])
        print(f"✓ Successfully installed {package_name}")
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        sys.exit(1)

print("\n✅ All dependencies installed successfully!")
print("You can now run: python automated_processing_studio.py")