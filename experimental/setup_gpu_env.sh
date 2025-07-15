#!/bin/bash
# Setup script for GPU-accelerated fiber optic analysis pipeline

echo "Setting up GPU-accelerated fiber optic analysis environment..."

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU acceleration may not be available."
    echo "Continuing with CPU-only setup..."
else
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "ERROR: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_gpu
source venv_gpu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install base requirements
echo "Installing base requirements..."
pip install numpy opencv-python matplotlib scipy scikit-learn Pillow tqdm pyyaml pytest pytest-cov

# Detect CUDA version and install appropriate CuPy
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep release | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "CUDA version detected: $cuda_version"
    
    # Install CuPy based on CUDA version
    if [[ $cuda_version == "11."* ]]; then
        echo "Installing CuPy for CUDA 11.x..."
        pip install cupy-cuda11x
    elif [[ $cuda_version == "12."* ]]; then
        echo "Installing CuPy for CUDA 12.x..."
        pip install cupy-cuda12x
    else
        echo "WARNING: Unsupported CUDA version. Please install CuPy manually."
    fi
else
    echo "CUDA not detected. Skipping CuPy installation."
    echo "The pipeline will run in CPU-only mode."
fi

# Create test script to verify installation
cat > test_gpu_setup.py << 'EOF'
#!/usr/bin/env python3
import sys

print("Testing GPU setup...")

# Test numpy
try:
    import numpy as np
    print("✓ NumPy installed:", np.__version__)
except ImportError:
    print("✗ NumPy not installed")

# Test OpenCV
try:
    import cv2
    print("✓ OpenCV installed:", cv2.__version__)
    
    # Check for CUDA support
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("  ✓ OpenCV CUDA support detected")
    else:
        print("  - OpenCV CUDA support not available")
except ImportError:
    print("✗ OpenCV not installed")

# Test CuPy
try:
    import cupy as cp
    print("✓ CuPy installed:", cp.__version__)
    
    # Test GPU
    try:
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        result = cp.asnumpy(c)
        print("  ✓ GPU computation test passed")
    except Exception as e:
        print(f"  ✗ GPU computation failed: {e}")
except ImportError:
    print("- CuPy not installed (CPU mode only)")

# Test RAPIDS
try:
    from cuml.cluster import DBSCAN
    print("✓ RAPIDS cuML installed")
except ImportError:
    print("- RAPIDS cuML not installed (CPU clustering will be used)")

print("\nSetup verification complete!")
EOF

# Run test script
echo -e "\n\nVerifying installation..."
python test_gpu_setup.py

# Create run script
cat > run_gpu_pipeline.sh << 'EOF'
#!/bin/bash
# Run the GPU-accelerated pipeline

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    source venv_gpu/bin/activate
fi

# Run the pipeline
python app_gpu.py "$@"
EOF

chmod +x run_gpu_pipeline.sh

# Instructions
echo -e "\n\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the GPU-accelerated pipeline:"
echo "1. Activate the virtual environment:"
echo "   source venv_gpu/bin/activate"
echo ""
echo "2. Run the pipeline:"
echo "   python app_gpu.py <input_image> <output_dir>"
echo "   or"
echo "   ./run_gpu_pipeline.sh <input_image> <output_dir>"
echo ""
echo "3. To force CPU mode (for testing):"
echo "   python app_gpu.py <input_image> <output_dir> --cpu"
echo ""
echo "4. To run tests:"
echo "   python test_gpu_pipeline.py"
echo ""
echo "Note: For full GPU acceleration with OpenCV, you may need to build"
echo "OpenCV from source with CUDA support."