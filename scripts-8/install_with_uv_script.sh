#!/bin/bash
# install_with_uv.sh - Installation script for UV users

echo "=================================="
echo "Geometry Detection System Installer"
echo "UV Package Manager Version"
echo "=================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found! Please install from: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✓ UV found: $(uv --version)"

# Install packages
echo ""
echo "Installing required packages..."
echo "=============================="

# Core packages
echo "📦 Installing opencv-python..."
uv pip install opencv-python

echo "📦 Installing opencv-contrib-python..."
uv pip install opencv-contrib-python

echo "📦 Installing numpy..."
uv pip install numpy

echo "📦 Installing psutil..."
uv pip install psutil

# Optional packages
echo ""
echo "Installing optional packages..."
echo "=============================="

echo "📦 Installing scipy..."
uv pip install scipy

echo "📦 Installing matplotlib..."
uv pip install matplotlib

echo "📦 Installing pandas..."
uv pip install pandas

echo "📦 Installing pillow..."
uv pip install pillow

# Test installation
echo ""
echo "Testing installation..."
echo "======================"

uv run python -c "
import cv2
import numpy as np
print(f'✓ OpenCV {cv2.__version__} installed')
print(f'✓ NumPy {np.__version__} installed')
print('✓ Basic imports successful!')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run the main program:"
echo "   uv run python integrated_geometry_system.py"
echo ""
echo "2. Or try the example:"
echo "   uv run python shape_analysis_dashboard.py"
