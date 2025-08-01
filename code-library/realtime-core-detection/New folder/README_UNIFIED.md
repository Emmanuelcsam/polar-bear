# Unified Core Detector with Interactive Circle Overlay

A single-process application that combines live core detection and interactive circle overlay functionality, fixing all OpenCV window errors and providing maximum functionality with minimal code.

## Features

### ✅ **Fixed Issues**
- **No more OpenCV window errors** - All window management issues resolved
- **Single process** - No more separate processes causing synchronization issues
- **Error handling** - Robust error handling for all OpenCV operations
- **Clean shutdown** - Proper resource cleanup and graceful exit

### 🚀 **Core Functionality**
- **Live core detection** - Real-time geometric circle detection
- **Interactive circle overlay** - Fully controllable circle with WASD movement
- **Performance tracking** - FPS monitoring and performance statistics
- **Configuration support** - Uses existing config.json file
- **Camera support** - Pylon camera and webcam fallback

### 🎮 **Interactive Controls**
- **WASD** - Move circle (W=up, S=down, A=left, D=right)
- **Q/E** - Resize circle (Q=smaller, E=larger)
- **L** - Lock/Unlock circle position
- **R** - Reset circle to center
- **ESC** - Exit application

### 📊 **Display Features**
- **Real-time FPS** - Current frame rate display
- **Detection results** - Live core detection visualization
- **Circle information** - Current circle position and radius
- **Performance stats** - Circle overlay performance metrics
- **Lock indicator** - Visual indication when circle is locked

## Quick Start

### Method 1: Direct Python Execution
```bash
# Using the full Python path
/c/Users/Saem1001/AppData/Local/Programs/Python/Python313/python.exe unified_core_detector.py
```

### Method 2: Windows Batch File
```cmd
# Double-click or run from command line
run_unified.bat
```

### Method 3: Python Launcher
```bash
# Using the Python launcher script
/c/Users/Saem1001/AppData/Local/Programs/Python/Python313/python.exe run_unified.py
```

## Configuration

The application uses the existing `config.json` file with these key sections:

### Camera Settings
```json
{
  "camera": {
    "camera_index": 0,
    "use_pylon": true,
    "auto_exposure": true,
    "exposure_time": 10000,
    "gain": 0
  }
}
```

### Circle Overlay Settings
```json
{
  "circle_overlay": {
    "initial_center_x": 320,
    "initial_center_y": 240,
    "initial_radius": 50,
    "move_step": 8,
    "resize_step": 5,
    "color_red": 255,
    "color_green": 0,
    "color_blue": 0,
    "thickness": 2,
    "center_point_size": 3
  }
}
```

### Display Settings
```json
{
  "display": {
    "window_name": "Unified Core Detector",
    "show_fps": true,
    "show_detections": true,
    "show_info": true,
    "show_circle_info": true,
    "show_performance_stats": true
  }
}
```

## Architecture

### Single Process Design
- **UnifiedCoreDetector** - Main application class
- **PylonCamera** - Camera interface with error handling
- **InteractiveCircleOverlay** - Circle overlay functionality
- **ConfigManager** - Configuration management
- **CoreDetectionResult** - Detection result container

### Key Improvements
1. **No window positioning errors** - Eliminated problematic window positioning code
2. **Robust error handling** - All OpenCV operations wrapped in try-catch
3. **Single window** - One window instead of multiple overlay windows
4. **Clean shutdown** - Proper resource cleanup on exit
5. **Performance optimized** - Minimal code with maximum functionality

## Error Resolution

### Previous Issues Fixed
- ❌ `NULL window: 'Live Core Detector'` - **FIXED**
- ❌ `NULL window: 'Circle Overlay'` - **FIXED**
- ❌ `WND_PROP_TRANSPARENT not available` - **FIXED**
- ❌ Process synchronization issues - **FIXED**
- ❌ Window positioning errors - **FIXED**

### Current Status
- ✅ Single process architecture
- ✅ Robust error handling
- ✅ Clean window management
- ✅ Proper resource cleanup
- ✅ Full functionality preserved

## Performance

### Optimizations
- **Minimal code** - Reduced from 1000+ lines to ~400 lines
- **Single process** - No inter-process communication overhead
- **Efficient rendering** - Direct frame modification
- **Memory efficient** - Proper resource management

### Features Retained
- ✅ Live core detection
- ✅ Interactive circle control
- ✅ Performance tracking
- ✅ Configuration support
- ✅ Camera support (Pylon + webcam)
- ✅ Real-time display
- ✅ Keyboard controls

## Usage Examples

### Basic Usage
```bash
# Start with default configuration
python unified_core_detector.py
```

### Custom Configuration
```bash
# Use custom config file
python unified_core_detector.py --config my_config.json
```

### Windows Users
```cmd
# Double-click the batch file
run_unified.bat
```

## Troubleshooting

### Common Issues
1. **Python not found** - Update the Python path in launcher scripts
2. **Camera not detected** - Check camera connections and drivers
3. **OpenCV errors** - Ensure OpenCV is properly installed

### Solutions
1. **Update Python path** in `run_unified.py` and `run_unified.bat`
2. **Check camera** - Verify camera is connected and accessible
3. **Install dependencies** - Ensure all required packages are installed

## Dependencies

### Required Packages
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `pypylon` - Pylon camera support (optional)

### Installation
```bash
pip install opencv-python numpy
pip install pypylon  # Optional for Pylon cameras
```

## Comparison

### Before (Separate Processes)
- ❌ Multiple processes
- ❌ Window positioning errors
- ❌ Synchronization issues
- ❌ Complex error handling
- ❌ Resource cleanup problems

### After (Unified Process)
- ✅ Single process
- ✅ No window errors
- ✅ Synchronized operation
- ✅ Robust error handling
- ✅ Clean resource management

## Conclusion

The unified core detector provides all the functionality of the original separate processes while eliminating all the OpenCV window errors and process synchronization issues. It's more reliable, easier to use, and provides a better user experience with minimal code complexity. 