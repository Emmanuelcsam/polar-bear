# Interactive Circle Overlay for Live Video Stream

This project provides an interactive blue circle overlay that can be added to live video streams. The circle can be moved, resized, and locked in position using keyboard controls.

## Features

- **Interactive Blue Circle**: A blue circle overlay that appears on the video stream
- **Keyboard Controls**: Full keyboard control for moving, resizing, and locking the circle
- **Real-time Display**: Live video stream with real-time circle overlay
- **Camera Support**: Supports both webcam and Pylon cameras
- **Visual Feedback**: Lock status and circle information displayed on screen
- **Boundary Checking**: Circle stays within frame boundaries

## Files

- `interactive_circle_overlay.py` - Main application with live video stream
- `test_circle_overlay.py` - Test script that simulates video stream
- `README_Circle_Overlay.md` - This documentation file

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV (cv2)
- NumPy
- Optional: Pylon SDK for industrial cameras

### Dependencies

```bash
pip install opencv-python numpy
```

For Pylon camera support:
```bash
pip install pypylon
```

## Usage

### Live Video Stream

Run the main application with your camera:

```bash
python interactive_circle_overlay.py --camera 0
```

Options:
- `--camera CAMERA`: Camera index (default: 0)
- `--pylon`: Use Pylon SDK if available
- `--help`: Show help message

### Test Mode

Run the test script to try the functionality without a camera:

```bash
python test_circle_overlay.py
```

## Keyboard Controls

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

## Features in Detail

### Circle Movement
- Use WASD keys to move the circle around the screen
- Movement is constrained to keep the circle within frame boundaries
- Movement step size is configurable (default: 10 pixels)

### Circle Resizing
- Use Q to make the circle smaller
- Use E to make the circle larger
- Resize step size is configurable (default: 5 pixels)
- Minimum radius: 5 pixels
- Maximum radius: Half the smaller dimension of the frame

### Circle Locking
- Press L to lock/unlock the circle position
- When locked, the circle cannot be moved or resized
- Lock status is displayed on screen (LOCKED/UNLOCKED)
- Lock status is indicated by color (Red for locked, Green for unlocked)

### Reset Function
- Press R to reset the circle to the center of the frame
- Resets both position and radius to default values

## Visual Elements

### Circle Display
- Blue circle with configurable thickness
- Center point marked with a small filled circle
- Lock status text displayed above the circle

### Information Overlay
- Semi-transparent overlay at the bottom of the screen
- Displays all keyboard controls
- Shows current circle position and radius
- Real-time updates as you interact with the circle

## Technical Details

### Class Structure

#### InteractiveCircleOverlay
- Handles circle drawing and keyboard input
- Manages circle position, radius, and lock state
- Provides visual feedback and instructions

#### InteractiveVideoStream
- Manages camera interface (webcam or Pylon)
- Handles frame reading and display
- Coordinates the main application loop

### Camera Support

#### Webcam
- Standard OpenCV VideoCapture
- Supports any camera accessible via OpenCV
- Fallback option if Pylon is not available

#### Pylon Camera
- Industrial camera support via Pylon SDK
- Automatic fallback to webcam if Pylon unavailable
- Optimized for Basler cameras

### Performance
- Real-time processing with minimal latency
- Efficient frame handling
- Responsive keyboard input processing

## Customization

### Modifying Circle Properties

Edit the `InteractiveCircleOverlay` class initialization:

```python
circle_overlay = InteractiveCircleOverlay(
    initial_center=(320, 240),  # Starting position
    initial_radius=50,           # Starting radius
    move_step=10,               # Movement step size
    resize_step=5               # Resize step size
)
```

### Changing Colors

Modify the color property in the `InteractiveCircleOverlay` class:

```python
self.color = (255, 0, 0)  # Blue in BGR format
```

### Adjusting Display

Modify the instruction overlay in `add_instructions_overlay()` method to change:
- Text size and font
- Overlay position and transparency
- Information displayed

## Troubleshooting

### Camera Issues
- Ensure camera is connected and accessible
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions
- For Pylon cameras, ensure SDK is properly installed

### Display Issues
- Ensure OpenCV is properly installed
- Check if display window appears
- Verify keyboard input is working

### Performance Issues
- Reduce frame processing frequency
- Lower resolution if needed
- Close other applications using the camera

## Integration with Existing Code

The circle overlay can be easily integrated into existing video processing pipelines:

```python
from interactive_circle_overlay import InteractiveCircleOverlay

# Create overlay
circle_overlay = InteractiveCircleOverlay()

# In your video processing loop
frame_with_circle = circle_overlay.draw_circle(frame)
# Handle keyboard input
circle_overlay.handle_keyboard_input(key, frame.shape)
```

## Example Use Cases

1. **Quality Control**: Mark areas of interest on production line
2. **Measurement**: Use circle as a reference for size measurements
3. **Alignment**: Position circle over targets for alignment
4. **Testing**: Verify camera positioning and focus
5. **Demonstration**: Show interactive features in presentations

## Future Enhancements

- Mouse control for circle positioning
- Multiple circles support
- Circle templates and presets
- Save/load circle configurations
- Network streaming support
- Integration with machine learning models 