# 🔍 Unified Core Detection System

A **comprehensive, unified core detection system** that combines **manual circle overlay**, **automatic detection**, and **visual display** of your actual camera feed.

## ✅ **FEATURES**

- **🎯 Manual Circle Overlay** - Move and resize a circle with WASD/QE controls
- **🤖 Automatic Detection** - Real-time circle detection using Hough transform
- **📹 Live Camera Display** - See exactly what your camera is seeing
- **🌐 Web Interface** - Beautiful browser-based interface with live video feed
- **📊 Real-time Statistics** - FPS, frame count, detection performance
- **🔄 Auto Camera Detection** - Works with any camera (USB, Basler, etc.)
- **🎮 Demo Mode** - Works even without a camera for testing
- **⚡ High Performance** - Optimized for real-time processing

## 🚀 **QUICK START**

### **Option 1: Web Interface (Recommended)**

```bash
# Windows (double-click)
start_unified_web.bat

# Or run directly
py start_unified_web.py
```

This will:
1. Install dependencies automatically
2. Start the web server
3. Open your browser to `http://localhost:5000`
4. Show live camera feed with manual overlay and automatic detection

### **Option 2: Direct Python**

```bash
# Run the unified detector directly
py unified_web_detector.py
```

## 🎮 **CONTROLS**

### **Manual Circle Controls**
- **WASD** - Move circle (W=up, S=down, A=left, D=right)
- **Q/E** - Resize circle (Q=smaller, E=larger)
- **L** - Lock/unlock circle position
- **R** - Reset circle to center

### **System Controls**
- **M** - Toggle manual override mode
- **A** - Toggle automatic detection
- **ESC** - Exit application

## 📁 **PROJECT STRUCTURE**

```
version12/
├── unified_web_detector.py      # Main unified detector
├── start_unified_web.py         # Startup script
├── start_unified_web.bat        # Windows batch file
├── camera_manager.py            # Camera detection and management
├── circle_overlay.py            # Manual circle overlay system
├── config.json                  # Configuration file
├── templates/
│   └── unified_index.html       # Web interface template
└── trash/duplicate_files/       # Old files moved here
```

## 🔧 **SYSTEM COMPONENTS**

### **1. Manual Circle Overlay**
- **UltraFastCircleOverlay** class from `circle_overlay.py`
- Real-time circle positioning and resizing
- Lock/unlock functionality
- Smooth movement with keyboard controls

### **2. Automatic Detection**
- **Hough Circle Detection** using OpenCV
- Confidence scoring based on contrast
- Real-time processing at 10 FPS
- Configurable detection parameters

### **3. Camera Management**
- **CameraManager** class from `camera_manager.py`
- Multi-backend camera detection (DirectShow, Media Foundation, etc.)
- Automatic fallback to demo mode
- Support for Pylon cameras (if available)

### **4. Web Interface**
- **Flask-based** web server
- Real-time video streaming
- Live statistics and detection data
- Responsive design with dark theme

## 📊 **DETECTION TYPES**

### **Manual Detection (Red/Yellow Circle)**
- **Red Circle**: Locked position (confidence = 1.0)
- **Yellow Circle**: Unlocked position (confidence = 0.5)
- **Controls**: WASD to move, Q/E to resize, L to lock

### **Automatic Detection (Green Circle)**
- **Green Circle**: Automatically detected circles
- **Confidence**: Displayed next to circle (0.0-1.0)
- **Threshold**: Only shows detections above confidence threshold

## ⚙️ **CONFIGURATION**

The system uses `config.json` for configuration:

```json
{
  "detection": {
    "min_radius": 10,
    "max_radius": 200,
    "confidence_threshold": 0.3
  },
  "web": {
    "host": "localhost",
    "port": 5000
  },
  "display": {
    "show_fps": true,
    "show_info": true,
    "show_manual": true,
    "show_automatic": true
  }
}
```

## 🎯 **USE CASES**

### **Quality Control**
- Position manual circle over expected core location
- Lock circle in place for consistent measurement
- Compare with automatic detection results

### **Research & Development**
- Test automatic detection algorithms
- Collect training data with manual annotations
- Analyze detection performance in real-time

### **Educational**
- Learn computer vision concepts
- Understand circle detection algorithms
- Practice manual annotation techniques

## 🔍 **DETECTION ALGORITHM**

### **Hough Circle Detection**
1. **Preprocessing**: Convert to grayscale, apply Gaussian blur
2. **Edge Detection**: Use gradient-based edge detection
3. **Circle Detection**: Apply Hough transform for circles
4. **Confidence Scoring**: Calculate contrast-based confidence
5. **Filtering**: Only show detections above threshold

### **Confidence Calculation**
- Create mask for detected circle
- Calculate mean intensity inside and outside circle
- Compute contrast ratio as confidence score
- Normalize to 0.0-1.0 range

## 🚨 **TROUBLESHOOTING**

### **Camera Not Detected**
- Check camera drivers
- Close other applications using camera
- Try different USB ports
- System will automatically use demo mode

### **Web Interface Not Loading**
- Check if port 5000 is available
- Try accessing `http://localhost:5000` manually
- Check firewall settings

### **Performance Issues**
- Reduce frame rate in configuration
- Lower detection parameters
- Close other applications

## 📈 **PERFORMANCE**

- **Frame Rate**: 10 FPS (configurable)
- **Detection Speed**: Real-time processing
- **Memory Usage**: Minimal overhead
- **CPU Usage**: Optimized for efficiency

## 🔄 **DEMO MODE**

When no camera is detected, the system automatically switches to demo mode:
- Generates synthetic camera feed
- Shows moving circles for testing
- All features work normally
- Perfect for testing and development

## 🎉 **SUCCESS!**

Your unified core detection system is now ready! It provides:

✅ **Manual circle overlay** with full control  
✅ **Automatic detection** with confidence scoring  
✅ **Live camera display** in web browser  
✅ **Real-time statistics** and performance metrics  
✅ **Clean, organized codebase** with no duplicates  

**Start the system with `start_unified_web.bat` and see your camera feed with both manual and automatic detections!** 