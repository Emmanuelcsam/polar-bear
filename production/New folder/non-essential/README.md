# BMP Video Emulator

A comprehensive system for emulating real-time video feed by looping a BMP image, with full integration to the Pylon camera grabber system.

## Overview

This system provides a complete solution for emulating real-time video when actual camera hardware is unavailable or for testing purposes. It seamlessly integrates with the existing `pylon_grabber.py` module and provides both programmatic and GUI interfaces.

## Features

- **Real-time Video Emulation**: Loops BMP images at configurable frame rates
- **Full Pylon Integration**: Compatible with existing `pylon_grabber.py` interface
- **GUI Control Interface**: User-friendly Tkinter-based control panel
- **Thread-Safe Operation**: Multi-threaded design with proper synchronization
- **Comprehensive Testing**: Complete unit test suite for all components
- **Error Handling**: Robust error handling for various edge cases

## Components

### 1. BMPVideoEmulator
The core emulation engine that:
- Loads and validates BMP images
- Maintains frame rate timing
- Provides thread-safe frame access
- Tracks frame count and statistics

### 2. EmulatedPylonGrabber
Extended version of `PylonFrameGrabber` that:
- Inherits from the original PylonFrameGrabber
- Automatically falls back to emulation when real camera unavailable
- Maintains the same interface as the original grabber
- Seamlessly switches between real camera and emulation

### 3. VideoEmulatorGUI
User interface that provides:
- Configuration controls (image path, frame rate, emulation mode)
- Start/stop controls
- Real-time status monitoring
- Frame count display
- Logging interface

## Installation

1. Install required dependencies:
```bash
pip install -r non-essential/requirements.txt
```

2. Ensure `good.bmp` is in the project directory

## Usage

### Command Line
```python
from bmp_video_emulator import BMPVideoEmulator

# Create emulator
emulator = BMPVideoEmulator("good.bmp", frame_rate=30)

# Start emulation
emulator.start()

# Read frames
frame = emulator.read()

# Stop emulation
emulator.stop()
```

### With Pylon Integration
```python
from bmp_video_emulator import EmulatedPylonGrabber

# Create grabber with emulation
grabber = EmulatedPylonGrabber(
    use_emulation=True,
    image_path="good.bmp",
    frame_rate=30
)

# Start grabber
grabber.start()

# Read frames (same interface as real camera)
frame = grabber.read()

# Stop grabber
grabber.stop()
```

### GUI Application
```python
from bmp_video_emulator import main
main()
```

## Testing

Run the comprehensive test suite:

```bash
python non-essential/test_bmp_video_emulator.py
```

The test suite covers:
- Unit tests for all classes and methods
- Integration tests for complete workflows
- Error handling and edge cases
- Thread safety under stress conditions
- GUI functionality

## Architecture

```
BMPVideoEmulator
├── Image loading and validation
├── Frame rate timing control
├── Thread-safe frame provision
└── Statistics tracking

EmulatedPylonGrabber (inherits PylonFrameGrabber)
├── Automatic fallback to emulation
├── Same interface as real camera
└── Seamless integration

VideoEmulatorGUI
├── Configuration interface
├── Real-time controls
├── Status monitoring
└── Logging display
```

## Configuration

### Frame Rate
- Default: 30 FPS
- Range: 1-60 FPS
- Configurable via constructor or GUI

### Image Path
- Default: "good.bmp"
- Supports any OpenCV-readable image format
- Validates file existence and format

### Emulation Mode
- Automatic: Uses emulation when Pylon unavailable
- Manual: Force emulation or real camera
- GUI toggle available

## Error Handling

The system handles various error conditions:
- Missing or corrupted image files
- Invalid frame rates
- Thread synchronization issues
- GUI interaction errors
- Pylon SDK availability issues

## Performance

- **Frame Rate Accuracy**: Maintains target frame rate within ±1 FPS
- **Memory Usage**: Minimal memory footprint with frame copying
- **CPU Usage**: Efficient polling with small sleep intervals
- **Thread Safety**: Lock-based synchronization prevents race conditions

## Integration with Existing Code

The system is designed to be a drop-in replacement for real camera usage:

```python
# Original code with real camera
from pylon_grabber import PylonFrameGrabber

grabber = PylonFrameGrabber()
grabber.start()
frame = grabber.read()
grabber.stop()

# Same code with emulation
from bmp_video_emulator import EmulatedPylonGrabber

grabber = EmulatedPylonGrabber(use_emulation=True)
grabber.start()
frame = grabber.read()  # Same interface!
grabber.stop()
```

## Troubleshooting

### Common Issues

1. **Image not found**: Ensure `good.bmp` exists in the project directory
2. **OpenCV import error**: Install opencv-python: `pip install opencv-python`
3. **GUI not responding**: Check if Tkinter is available on your system
4. **Frame rate issues**: Verify frame rate is between 1-60 FPS

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Adding New Features
1. Extend the appropriate class
2. Add corresponding unit tests
3. Update documentation
4. Test integration with existing components

### Testing Guidelines
- All new features must have unit tests
- Integration tests for cross-component functionality
- Error handling tests for edge cases
- Performance tests for critical paths

## License

This project is part of the polar-bear production system. 