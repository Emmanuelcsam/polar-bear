# How to Use Live Feed with Geometry Detection Scripts

## Quick Start (5 Minutes)

### 1. First Time Setup
```bash
# Run the automatic setup script
python quick_start.py
```
This will:
- Check your Python version
- Install required packages
- Find your camera
- Test everything works
- Show you next steps

### 2. Test Your Camera
```bash
# Find which camera index works
python camera_test.py --scan

# Test specific camera
python camera_test.py --index 0
```

### 3. Run Simple Version First
```bash
# This is the easiest to get working
python simple_geometry_detector.py
```

### 4. Run Full Advanced Version
```bash
# Use default camera (usually built-in webcam)
python advanced_geometry_detector.py

# Use external USB camera
python advanced_geometry_detector.py -s 1

# Use specific camera if default doesn't work
python advanced_geometry_detector.py -s 2
```

## Camera Indices Explained

Different cameras use different index numbers:
- **0** = Default camera (usually built-in webcam)
- **1** = First external USB camera
- **2** = Second external USB camera
- **3+** = Additional cameras

## Common Camera Issues and Fixes

### Issue: "Could not open video source"

**Solution 1: Find the right camera index**
```bash
python camera_test.py --scan
```
Then use the working index:
```bash
python advanced_geometry_detector.py -s [working_index]
```

**Solution 2: Check camera permissions**

Windows:
- Close all apps using camera (Zoom, Skype, etc.)
- Check Device Manager for camera drivers
- Run as Administrator

Linux:
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login again

# Or quick fix
sudo chmod 666 /dev/video0
```

macOS:
- System Preferences → Security & Privacy → Camera
- Allow Terminal/Python access

**Solution 3: Install camera drivers**

Windows:
- Update drivers in Device Manager

Linux:
```bash
sudo apt-get update
sudo apt-get install v4l-utils
sudo apt-get install cheese  # Test camera

# List cameras
v4l2-ctl --list-devices
```

### Issue: "Black screen" or "No frames"

**Fix:**
```python
# Create test_fix.py and run it
import cv2

# Try different backends
backends = [
    cv2.CAP_DSHOW,     # Windows DirectShow
    cv2.CAP_MSMF,      # Windows Media Foundation  
    cv2.CAP_V4L2,      # Linux V4L2
    cv2.CAP_AVFOUNDATION,  # macOS
    cv2.CAP_ANY        # Auto-detect
]

for backend in backends:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Backend {backend} works!")
            break
    cap.release()
```

### Issue: Camera used by another app

**Fix:**
- Close: Zoom, Teams, Skype, Discord, OBS
- Windows: Open Task Manager, end camera processes
- Linux: `sudo fuser /dev/video0` to find process using camera
- Restart computer if needed

## Using Different Camera Types

### 1. Built-in Webcam
```bash
python advanced_geometry_detector.py
# or explicitly
python advanced_geometry_detector.py -s 0
```

### 2. USB Webcam
```bash
# Usually index 1 or 2
python advanced_geometry_detector.py -s 1
```

### 3. Phone as Webcam (IP Camera)

**Using DroidCam (Android/iOS):**
1. Install DroidCam on phone
2. Connect phone and PC to same WiFi
3. Note the IP address shown in app
4. Use URL as source:
```bash
python advanced_geometry_detector.py -s "http://192.168.1.100:4747/video"
```

**Using IP Webcam (Android):**
```bash
python advanced_geometry_detector.py -s "http://192.168.1.100:8080/video"
```

### 4. Network/Security Camera (RTSP)
```bash
python advanced_geometry_detector.py -s "rtsp://username:password@192.168.1.100:554/stream"
```

### 5. Virtual Camera (OBS)
1. Install OBS Studio
2. Add your sources
3. Start Virtual Camera
4. Use index (usually 1 or 2):
```bash
python advanced_geometry_detector.py -s 1
```

## Running Each Script

### 1. Advanced Geometry Detector
```bash
# Basic usage
python advanced_geometry_detector.py

# With options
python advanced_geometry_detector.py -s 1 -t 8 --gpu

# Help
python advanced_geometry_detector.py --help
```

**Live Controls:**
- `q` - Quit
- `g` - Toggle GPU on/off
- `p` - Pause/Resume
- `s` - Save screenshot
- `+` - More sensitive (detect more shapes)
- `-` - Less sensitive (detect fewer shapes)
- `r` - Reset detector

### 2. Tube Angle Detector
```bash
# Basic usage
python tube_angle_detector.py

# Set tube diameter
python tube_angle_detector.py -d 15.5

# Calibrate camera first
python tube_angle_detector.py --calibrate
```

**Live Controls:**
- `q` - Quit
- `c` - Calibrate camera
- `p` - Pause/Resume
- `s` - Save screenshot
- `d` - Set tube diameter

### 3. Performance Benchmark
```bash
# Compare detection methods
python geometry_benchmark.py

# This will:
# - Generate test images
# - Run all detectors
# - Show performance graphs
# - Test on live camera
```

## Optimizing Performance

### For Slow/Laggy Detection:

1. **Reduce camera resolution:**
```python
# In the script, find these lines and change values:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # was 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # was 720
```

2. **Reduce threads:**
```bash
python advanced_geometry_detector.py -t 2
```

3. **Disable GPU (if causing issues):**
```bash
python advanced_geometry_detector.py --gpu False
```

4. **Adjust sensitivity:**
- Press `-` while running to reduce sensitivity
- This processes fewer shapes = faster

## Saving and Recording

### Save Screenshots:
- Press `s` while running
- Images saved as: `geometry_capture_[number].png`

### Record Video:
```python
# Add this to the script after cap = cv2.VideoCapture(0):
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

# In the main loop after drawing:
out.write(output)

# At the end:
out.release()
```

## Testing Without Camera

### Use Video File:
```bash
# Download sample video or use your own
python advanced_geometry_detector.py -f sample_shapes.mp4

# For tube detector
python tube_angle_detector.py -s tube_video.mp4
```

### Create Test Video:
```python
# save as create_test_video.py
import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_shapes.avi', fourcc, 30, (640, 480))

for i in range(300):  # 10 seconds at 30fps
    frame = np.ones((480, 640, 3), np.uint8) * 255
    
    # Animated circle
    cv2.circle(frame, (320 + int(100*np.sin(i*0.1)), 240), 
              50, (0, 0, 255), -1)
    
    # Static rectangle
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
    
    out.write(frame)

out.release()
print("Created test_shapes.avi")
```

## Emergency Fixes

### Nothing Works?

1. **Test OpenCV is installed:**
```python
python -c "import cv2; print(cv2.__version__)"
```

2. **Reinstall everything:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python numpy
```

3. **Use Docker (Linux/Mac):**
```bash
docker run -it --device=/dev/video0 python:3.8 bash
# Then install packages and run
```

4. **Use Google Colab (no local camera):**
- Upload scripts to Colab
- Use video files instead of camera

### Still Not Working?

Run diagnostic:
```bash
python camera_test.py --diagnose > diagnostic.txt
```

Check the diagnostic.txt file for specific issues.

## Success Checklist

✅ Python 3.7+ installed  
✅ OpenCV installed (`pip install opencv-python`)  
✅ Camera connected and working  
✅ No other apps using camera  
✅ Correct camera index found  
✅ Scripts in same directory  
✅ Proper permissions set  

## Contact and Support

If you've tried everything and still have issues:

1. Check you have latest script versions
2. Update all packages: `pip install --upgrade opencv-python numpy`
3. Try on different computer to isolate issue
4. Create minimal test case that shows the problem

Remember: The most common issue is just using the wrong camera index. Try 0, 1, 2, 3 until one works!