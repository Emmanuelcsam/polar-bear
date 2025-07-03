# Complete Installation and Usage Guide for Geometry Detection System

## System Requirements

### Minimum Requirements:
- Python 3.7 or higher
- 4GB RAM
- Webcam or video file
- Windows 10/11, Ubuntu 18.04+, or macOS 10.14+

### Recommended Requirements:
- Python 3.8-3.10
- 8GB+ RAM
- NVIDIA GPU with CUDA support (optional)
- USB webcam or built-in camera

## Installation Instructions

### Step 1: Install Python
If you don't have Python installed:

**Windows:**
1. Download Python from https://www.python.org/downloads/
2. Run installer and CHECK "Add Python to PATH"
3. Verify: Open Command Prompt and type `python --version`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
# Install Homebrew first if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv geometry_env

# Activate virtual environment
# Windows:
geometry_env\Scripts\activate

# Linux/macOS:
source geometry_env/bin/activate
```

### Step 3: Install Required Libraries

```bash
# Basic installation (CPU only)
pip install opencv-python opencv-contrib-python numpy scipy matplotlib pandas

# For GPU support (optional)
pip install opencv-python-headless opencv-contrib-python numpy scipy matplotlib pandas

# Additional helpful packages
pip install pillow scikit-learn
```

### Step 4: Install GPU Support (Optional)

**For NVIDIA GPU acceleration:**

1. Check if you have NVIDIA GPU:
   ```bash
   # Windows
   wmic path win32_VideoController get name
   
   # Linux
   lspci | grep -i nvidia
   ```

2. Install CUDA Toolkit:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose your OS and follow instructions
   - Verify: `nvcc --version`

3. Build OpenCV with CUDA (Advanced):
   ```bash
   # This is complex - consider using pre-built binaries
   # See: https://pypi.org/project/opencv-python-rolling/
   ```

### Step 5: Fix Common Installation Issues

**Issue: "No module named cv2"**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install --upgrade opencv-python opencv-contrib-python
```

**Issue: "ImportError: libGL.so.1"** (Linux)
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

**Issue: Camera permissions (Linux)**
```bash
sudo usermod -a -G video $USER
# Log out and back in
```

**Issue: Camera access (macOS)**
- Go to System Preferences → Security & Privacy → Camera
- Allow Terminal/Python access

## Running the Scripts

### Basic Usage

1. **Using default webcam (camera 0):**
   ```bash
   python advanced_geometry_detector.py
   ```

2. **Using external USB camera:**
   ```bash
   # Try different indices if camera 1 doesn't work
   python advanced_geometry_detector.py -s 1
   ```

3. **Using video file:**
   ```bash
   python advanced_geometry_detector.py -f path/to/your/video.mp4
   ```

4. **Disable GPU (if having issues):**
   ```bash
   python advanced_geometry_detector.py --gpu False
   ```

5. **Adjust thread count:**
   ```bash
   python advanced_geometry_detector.py -t 8
   ```

### Controls During Runtime

- **'q'** - Quit the program
- **'g'** - Toggle GPU acceleration on/off
- **'p'** - Pause/Resume detection
- **'s'** - Save screenshot
- **'+'** - Increase detection sensitivity (more shapes)
- **'-'** - Decrease detection sensitivity (fewer shapes)
- **'r'** - Reset detector

## Troubleshooting Camera Issues

### 1. Camera Not Found

**Windows:**
```bash
# Check available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Try different indices
python advanced_geometry_detector.py -s 0  # Built-in camera
python advanced_geometry_detector.py -s 1  # USB camera
python advanced_geometry_detector.py -s 2  # Second USB camera
```

**Linux:**
```bash
# List video devices
ls /dev/video*

# Check camera with v4l2
v4l2-ctl --list-devices

# Test camera
cheese  # or 'guvcview'

# Fix permissions
sudo chmod 666 /dev/video0
```

**macOS:**
```bash
# List cameras
system_profiler SPCameraDataType

# Test with Photo Booth first
# Grant camera permissions in System Preferences
```

### 2. Camera Opens But Shows Black Screen

```python
# Test script to debug camera
import cv2
import numpy as np

# Try different backends
backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
for i, backend in enumerate(backends):
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Backend {backend} works!")
            cv2.imshow('Test', frame)
            cv2.waitKey(1000)
        cap.release()

cv2.destroyAllWindows()
```

### 3. Multiple Cameras Setup

```python
# Find all available cameras
import cv2

def find_cameras():
    index = 0
    cameras = []
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cameras.append(index)
                print(f"Camera found at index {index}")
            cap.release()
        index += 1
    return cameras

print("Available cameras:", find_cameras())
```

## Testing the Installation

Create `test_setup.py`:

```python
#!/usr/bin/env python3
import sys
import cv2
import numpy as np

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Test CUDA
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}")

# Test camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera opened successfully")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
    cap.release()
else:
    print("Failed to open camera")

print("\nSetup test complete!")
```

Run: `python test_setup.py`

## Common Error Fixes

### Error: "select timeout" or "VIDIOC_DQBUF" (Linux)
```bash
# Install v4l-utils
sudo apt-get install v4l-utils

# Reset USB camera
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo
```

### Error: "[WARN:0] global... cap_msmf.cpp" (Windows)
This is just a warning, not an error. The camera should still work.

### Error: "(-215:Assertion failed)" 
This usually means the frame is empty. Check:
1. Camera is not being used by another application
2. Camera drivers are installed
3. Try a different camera index

### Performance Issues
1. Reduce camera resolution
2. Decrease thread count: `python script.py -t 2`
3. Disable GPU: `python script.py --gpu False`
4. Close other applications

## Using IP Cameras

For IP cameras (security cameras, phone cameras):

```python
# Modify the source in the script
# Examples:
source = "http://192.168.1.100:8080/video"  # IP Webcam app
source = "rtsp://username:password@192.168.1.100:554/stream"  # RTSP stream
```

## Quick Start Examples

1. **Basic shape detection:**
   ```bash
   python advanced_geometry_detector.py
   ```

2. **High-performance mode:**
   ```bash
   python advanced_geometry_detector.py -t 8 -g
   ```

3. **Process video file:**
   ```bash
   python advanced_geometry_detector.py -f sample_video.mp4
   ```

4. **Use specific camera:**
   ```bash
   python advanced_geometry_detector.py -s 1
   ```

## Still Having Issues?

1. **Create a debug log:**
   ```bash
   python advanced_geometry_detector.py > debug.log 2>&1
   ```

2. **Check dependencies:**
   ```bash
   pip list | grep opencv
   ```

3. **Reinstall everything:**
   ```bash
   pip uninstall opencv-python opencv-contrib-python numpy
   pip install --upgrade opencv-python opencv-contrib-python numpy
   ```

4. **Try minimal test:**
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print(f"Success: {ret}")
   cap.release()
   ```

Remember: The most common issue is camera index. Try 0, 1, 2, etc. until one works!