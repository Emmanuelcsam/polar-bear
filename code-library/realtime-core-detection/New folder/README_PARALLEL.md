# Parallel Live Core Detector with Circle Overlay

This system runs the live core detector and circle overlay as separate parallel processes, allowing you to freely move, resize, and manipulate a circle overlay on top of the live core detector screen without any parameter limits.

## Features

- **Parallel Processing**: Live core detector and circle overlay run as separate processes
- **Free Movement**: Circle can be moved anywhere on screen without boundary restrictions
- **Unlimited Resizing**: Circle can be enlarged or shrunk without size limits
- **Interactive Controls**: Real-time keyboard controls for circle manipulation
- **Overlay Window**: Circle appears as a separate overlay window on top of the detector

## Quick Start

### Method 1: Use the Launcher Script (Recommended)

```bash
python run_parallel_detector.py
```

This will automatically start both the live core detector and circle overlay as parallel processes.

### Method 2: Run Processes Separately

1. Start the live core detector:
```bash
python live_core_detector.py
```

2. In a separate terminal, start the circle overlay:
```bash
python circle_overlay.py
```

## Circle Overlay Controls

- **WASD**: Move circle (W=up, S=down, A=left, D=right)
- **Q/E**: Resize circle (Q=smaller, E=larger)
- **L**: Lock/Unlock circle position
- **R**: Reset circle to center
- **ESC**: Exit overlay

## Command Line Options

### Launcher Script Options

```bash
python run_parallel_detector.py [OPTIONS]

Options:
  --camera INT           Camera index (default: 0)
  --no-pylon            Disable Pylon SDK and use webcam only
  --overlay-width INT    Circle overlay window width (default: 800)
  --overlay-height INT   Circle overlay window height (default: 600)
```

### Circle Overlay Options

```bash
python circle_overlay.py [OPTIONS]

Options:
  --window-name TEXT     Name of the overlay window (default: "Circle Overlay")
  --overlay-on TEXT      Name of the window to overlay on (default: "Live Core Detector")
  --width INT            Overlay window width (default: 800)
  --height INT           Overlay window height (default: 600)
```

## Examples

### Basic Usage
```bash
# Start with default settings
python run_parallel_detector.py
```

### Custom Camera and Window Size
```bash
# Use camera 1 with custom overlay size
python run_parallel_detector.py --camera 1 --overlay-width 1024 --overlay-height 768
```

### Webcam Only (No Pylon)
```bash
# Use webcam instead of Pylon camera
python run_parallel_detector.py --no-pylon
```

## How It Works

1. **Live Core Detector**: Runs the main detection algorithm and displays the camera feed
2. **Circle Overlay**: Creates a separate transparent window that overlays on top of the detector window
3. **Parallel Processing**: Both processes run independently, allowing smooth interaction
4. **No Limits**: The circle can move freely without boundary restrictions or size limits

## Technical Details

- The circle overlay creates a transparent window using OpenCV
- Window positioning attempts to align with the main detector window
- Keyboard input is handled independently in each process
- No inter-process communication is required - the overlay is purely visual

## Troubleshooting

### Circle Overlay Not Visible
- Make sure the live core detector window is open first
- Check that the overlay window name matches the detector window name
- Try adjusting the overlay window size

### Performance Issues
- Reduce the overlay window size if needed
- Close other applications to free up system resources
- Check that your graphics drivers are up to date

### Camera Issues
- Use `--no-pylon` if Pylon camera is not available
- Try different camera indices with `--camera`
- Ensure camera is not being used by another application

## Stopping the Application

- Press **Ctrl+C** in the launcher terminal to stop all processes
- Or close the circle overlay window and then the detector window
- The launcher will automatically clean up all processes

## Configuration

The circle overlay uses configuration settings that can be modified in the `circle_overlay.py` file:

- Movement speed and step sizes
- Circle colors and thickness
- Initial position and size
- Performance settings

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pylon SDK (optional, for industrial cameras)

## Files

- `live_core_detector.py`: Main core detection application
- `circle_overlay.py`: Circle overlay process
- `run_parallel_detector.py`: Launcher script for both processes
- `README_PARALLEL.md`: This documentation file 