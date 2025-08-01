# Integrated Core Detection System

A modular, production-level system that combines interactive circle overlay, live video feed, and core detection functionality. Each component can work independently or be integrated together.

## 🎯 **System Overview**

This system consists of three main modular components:

1. **`circle_overlay.py`** - Interactive circle overlay with keyboard controls
2. **`live_feed.py`** - Camera interface and live video stream handler
3. **`main.py`** - Integrated application combining all functionality

## 📁 **File Structure**

```
version12/
├── circle_overlay.py    # Standalone circle overlay module
├── live_feed.py         # Standalone live feed module
├── main.py             # Integrated application
├── config.json         # Configuration file
├── README.md           # This documentation
└── old/                # Legacy files (moved from previous version)
```

## 🚀 **Quick Start**

### Run the Integrated Application
```bash
python main.py --camera 0
```

### Test Individual Components
```bash
# Test circle overlay
python circle_overlay.py --test

# Test live feed
python live_feed.py --camera 0

# Show system information
python main.py --info
```

## 🎮 **Keyboard Controls**

| Key | Action |
|-----|--------|
| **W** | Move circle up |
| **S** | Move circle down |
| **A** | Move circle left |
| **D** | Move circle right |
| **Q** | Decrease circle radius (make smaller) |
| **E** | Increase circle radius (make larger) |
| **L** | Lock/Unlock circle position |
| **R** | Reset circle to center |
| **ESC** | Exit application |

## 📋 **Component Details**

### 1. Circle Overlay (`circle_overlay.py`)

**Standalone Features:**
- Interactive blue circle overlay
- Keyboard controls for movement and resizing
- Lock/unlock functionality
- Boundary checking
- Visual feedback and instructions

**Integration Features:**
- Can be imported and used in other scripts
- Provides mask creation for region-based detection
- Configurable parameters (position, radius, color, etc.)

**Usage:**
```python
from circle_overlay import CircleOverlay

# Create overlay
circle = CircleOverlay(initial_center=(320, 240), initial_radius=50)

# Draw on frame
frame_with_circle = circle.draw_circle(frame)

# Handle keyboard input
should_continue = circle.handle_keyboard_input(key, frame.shape)

# Get circle information
info = circle.get_circle_info()
```

### 2. Live Feed (`live_feed.py`)

**Standalone Features:**
- Camera interface (webcam and Pylon support)
- Real-time video stream
- FPS tracking and performance monitoring
- Frame callback system for processing

**Integration Features:**
- Can be used with custom frame processing callbacks
- Automatic fallback between camera types
- Performance tracking and information overlay

**Usage:**
```python
from live_feed import LiveFeed

# Create live feed with custom processing
def process_frame(frame):
    # Your custom processing here
    return processed_frame

live_feed = LiveFeed(
    camera_index=0,
    use_pylon=False,
    frame_callback=process_frame
)

# Run live feed
live_feed.run(window_name="Custom Feed", show_info=True)
```

### 3. Main Application (`main.py`)

**Integrated Features:**
- Combines circle overlay with core detection
- Region-based detection within circle area
- Real-time core detection with confidence scoring
- Comprehensive information overlay
- Modular architecture for easy extension

**Core Detection:**
- Geometric approach using Hough circle detection
- Confidence calculation based on contrast
- Masked region processing for focused detection
- Multiple detection method support

## 🔧 **Installation**

### Prerequisites
- Python 3.7 or higher
- OpenCV (cv2)
- NumPy

### Dependencies
```bash
pip install opencv-python numpy
```

### Optional: Pylon Camera Support
```bash
pip install pypylon
```

## 🎯 **Usage Examples**

### Basic Usage
```bash
# Run with webcam
python main.py --camera 0

# Run with Pylon camera
python main.py --camera 0 --pylon

# Show system information
python main.py --info
```

### Component Testing
```bash
# Test circle overlay without camera
python circle_overlay.py --test

# Test live feed
python live_feed.py --camera 0 --no-info

# Test live feed with Pylon
python live_feed.py --camera 0 --pylon
```

## 🏗️ **Architecture**

### Modular Design
Each component is designed to work independently:

1. **Circle Overlay**: Handles interactive circle drawing and keyboard input
2. **Live Feed**: Manages camera interface and video stream
3. **Main App**: Orchestrates components and provides core detection

### Integration Points
- Circle overlay provides mask for region-based detection
- Live feed provides frame callback system for processing
- Main app coordinates all components with unified interface

### Extensibility
- Easy to add new detection methods
- Simple to modify circle overlay behavior
- Straightforward to integrate with other systems

## 🔍 **Core Detection Features**

### Detection Methods
- **Geometric Approach**: Uses Hough circle detection
- **Confidence Scoring**: Based on contrast analysis
- **Region Masking**: Focuses detection within circle area
- **Real-time Processing**: Optimized for live video

### Performance
- Frame-rate optimized processing
- Configurable processing intervals
- Efficient memory usage
- Responsive keyboard input

## 🛠️ **Troubleshooting**

### Camera Issues
```bash
# Try different camera indices
python main.py --camera 1
python main.py --camera 2

# Use webcam fallback
python main.py --camera 0  # (no --pylon flag)
```

### Performance Issues
- Reduce processing interval in main.py
- Lower camera resolution if needed
- Close other applications using camera

### Display Issues
- Ensure OpenCV is properly installed
- Check camera permissions
- Verify keyboard input is working

## 🔧 **Customization**

### Modifying Circle Properties
```python
# In circle_overlay.py
circle = CircleOverlay(
    initial_center=(320, 240),  # Starting position
    initial_radius=50,           # Starting radius
    move_step=10,               # Movement step size
    resize_step=5,              # Resize step size
    color=(255, 0, 0)          # Blue color (BGR)
)
```

### Adding Detection Methods
```python
# In main.py, add to CoreDetectionMethods class
@staticmethod
def your_custom_method(frame: np.ndarray, method_name: str = "custom") -> CoreDetectionResult:
    # Your detection logic here
    pass

# Add to detection_methods list
self.detection_methods.append(CoreDetectionMethods.your_custom_method)
```

### Custom Frame Processing
```python
# Create custom frame callback
def custom_frame_processor(frame):
    # Your processing logic
    return processed_frame

# Use with live feed
live_feed = LiveFeed(frame_callback=custom_frame_processor)
```

## 📊 **System Information**

The system provides comprehensive information about:
- Camera status and performance
- Circle position and state
- Detection results and confidence
- Frame rate and processing statistics

## 🔄 **Migration from Old System**

All previous files have been moved to the `old/` directory:
- `unified_live_core_detector.py` → `old/`
- `interactive_circle_overlay.py` → `old/`
- All other legacy files → `old/`

The new system maintains compatibility while providing:
- Better modularity
- Improved performance
- Enhanced debugging capabilities
- Cleaner architecture

## 🚀 **Future Enhancements**

- Mouse control for circle positioning
- Multiple circles support
- Advanced detection algorithms
- Network streaming capabilities
- Configuration file support
- Logging and analytics
- Plugin system for custom detectors

## 📝 **API Reference**

### CircleOverlay Class
```python
class CircleOverlay:
    def __init__(self, initial_center, initial_radius, move_step, resize_step, color)
    def draw_circle(self, frame) -> np.ndarray
    def handle_keyboard_input(self, key, frame_shape) -> bool
    def get_circle_info(self) -> dict
    def set_circle_info(self, center, radius, is_locked) -> None
    def create_mask(self, frame_shape) -> np.ndarray
```

### LiveFeed Class
```python
class LiveFeed:
    def __init__(self, camera_index, use_pylon, frame_callback)
    def run(self, window_name, show_info)
    def get_fps(self) -> float
    def get_camera_info(self) -> dict
    def cleanup(self)
```

### IntegratedCoreDetector Class
```python
class IntegratedCoreDetector:
    def __init__(self, camera_index, use_pylon)
    def run(self)
    def get_system_info(self) -> dict
```

## 📞 **Support**

For issues or questions:
1. Check the troubleshooting section
2. Test individual components
3. Review system information with `--info` flag
4. Check camera compatibility and permissions

---

**Version**: 2.0  
**Last Updated**: July 31, 2024  
**Status**: Production Ready 