# BMP Video Emulator with Scratch Detection (Hough Lines)

This project extends the original BMP video emulator with Hough circle detection to include **scratch detection** using Hough line transform. It provides real-time line detection capabilities with manual parameter adjustment through a user-friendly GUI.

## Features

### Original Circle Detection

- Real-time Hough circle detection
- Manual parameter adjustment (DP, min distance, thresholds, radii, blur parameters)
- Multiple presets (Sensitive, Balanced, Conservative)
- Integration with Pylon camera or BMP emulation

### New Scratch Detection (Hough Lines)

- Real-time Hough line detection for scratch identification
- Two detection methods:
  - **Probabilistic Hough Transform** (recommended for line segments)
  - **Standard Hough Transform** (for infinite lines)
- Comprehensive parameter control:
  - **Rho**: Distance resolution (1-10 pixels)
  - **Theta**: Angle resolution (0.1-5.0 degrees)
  - **Threshold**: Accumulator threshold (10-300)
  - **Min Line Length**: Minimum detectable line length (5-200 pixels)
  - **Max Line Gap**: Maximum gap between line segments (1-50 pixels)
  - **Gaussian Blur**: Kernel size (1-15, odd) and sigma (0.1-5.0)
  - **Canny Edge Detection**: Low (10-200) and high (50-400) thresholds
- Multiple presets optimized for different line types:
  - **Fine Lines**: Detects thin, subtle scratches
  - **Balanced**: General-purpose detection
  - **Thick Lines**: Detects thick, prominent scratches

## Files Structure

```
├── hough_circles.py              # Original circle detection module
├── hough_lines.py                # NEW: Line detection module for scratch detection
├── bmp_video_emulator.py         # Original emulator with circle detection GUI
├── scratch_detection_emulator.py # NEW: Emulator with scratch detection GUI
├── pylon_grabber.py             # Camera interface (Pylon SDK optional)
├── run_emulator.py              # Launcher for circle detection
├── run_scratch_detection.py     # NEW: Launcher for scratch detection
├── requirements.txt             # Python dependencies
└── good.bmp                     # Sample image file (required)
```

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install opencv-python numpy pillow tkinter
   ```

2. **Optional - Install Pylon SDK for real camera:**

   ```bash
   pip install pypylon
   ```

3. **Ensure you have a sample image:**
   - Place a BMP file named `good.bmp` in the project directory
   - This will be used as the video feed source for testing

## Usage

### Running Scratch Detection

```bash
# Using the launcher script
python3 run_scratch_detection.py

# Or directly
python3 scratch_detection_emulator.py
```

### Running Original Circle Detection

```bash
# Using the launcher script
python3 run_emulator.py

# Or directly
python3 bmp_video_emulator.py
```

## GUI Controls - Scratch Detection

### Configuration Panel

- **Image Path**: BMP file to use for emulation
- **Frame Rate**: Video playback speed (1-120 FPS)
- **Use Emulation**: Toggle between emulated and real camera

### Hough Lines Detection Panel

- **Enable Line Detection**: Toggle scratch detection on/off
- **Use Probabilistic Method**: Switch between probabilistic and standard Hough transform

### Parameter Controls

All parameters can be adjusted in real-time:

1. **Rho (Distance Resolution)**: How precisely distances are measured
2. **Theta (Angle Resolution)**: How precisely angles are measured
3. **Threshold**: Minimum votes needed to detect a line
4. **Min Line Length**: Shortest line to detect (probabilistic only)
5. **Max Line Gap**: Largest gap to bridge in a line (probabilistic only)
6. **Blur Parameters**: Gaussian blur to reduce noise
7. **Canny Thresholds**: Edge detection sensitivity

### Preset Buttons

- **Fine Lines**: Optimized for detecting subtle, thin scratches
- **Balanced**: Good general-purpose settings
- **Thick Lines**: Better for prominent, thick scratches

### Real-time Feedback

- **Video Display**: Shows original image with detected lines overlaid
- **Line Counter**: Shows number of lines detected in current frame
- **Method Indicator**: Shows whether Probabilistic or Standard method is active
- **Log Panel**: Real-time parameter updates and status messages

## Line Detection Visualization

The scratch detection system provides clear visual feedback:

- **Green Lines**: Detected line segments or infinite lines
- **Blue/Red Dots**: Line endpoints (probabilistic method only)
- **Yellow Text**: Detection statistics and method information

## Technical Details

### Hough Line Transform Methods

1. **Probabilistic Hough Transform (`cv2.HoughLinesP`)**:
   - Returns line segments with start/end points
   - More efficient for practical applications
   - Better for detecting actual scratches and defects
   - Recommended for most use cases

2. **Standard Hough Transform (`cv2.HoughLines`)**:
   - Returns infinite lines in polar coordinates
   - More theoretical approach
   - Useful for detecting extended linear features

### Detection Pipeline

1. **Color Conversion**: BGR → Grayscale
2. **Gaussian Blur**: Noise reduction
3. **Canny Edge Detection**: Edge enhancement
4. **Hough Transform**: Line detection
5. **Visualization**: Overlay results on original image

### Parameter Optimization Tips

- **For Fine Scratches**: Lower threshold, smaller rho/theta, tighter Canny settings
- **For Thick Scratches**: Higher threshold, larger parameters, relaxed Canny settings
- **Noisy Images**: Increase blur parameters, adjust Canny thresholds
- **Clean Images**: Reduce blur, use more sensitive settings

## Comparison with Circle Detection

| Feature | Circle Detection | Scratch Detection |
|---------|-----------------|-------------------|
| **Primary Use** | Detect circular objects | Detect linear defects/scratches |
| **Algorithm** | HoughCircles | HoughLines/HoughLinesP |
| **Key Parameters** | DP, param1/param2, radii | Rho, theta, threshold, line length |
| **Preprocessing** | Gaussian blur only | Gaussian blur + Canny edges |
| **Output** | Circle centers & radii | Line endpoints or equations |
| **Visualization** | Circles with centers | Lines with endpoints |

## Applications

- **Quality Control**: Detect scratches on manufactured surfaces
- **Material Inspection**: Identify linear defects in materials
- **Surface Analysis**: Analyze wear patterns and damage
- **Automated Inspection**: Real-time defect detection in production lines

## Troubleshooting

1. **No lines detected**: Lower threshold, adjust Canny parameters
2. **Too many false positives**: Increase threshold, refine blur settings
3. **Performance issues**: Reduce frame rate, optimize parameters
4. **Import errors**: Ensure all dependencies are installed

## Development Notes

The scratch detection emulator follows the same architecture as the circle detection version:

- **Modular Design**: Separate detection logic from GUI
- **Real-time Parameter Updates**: All settings adjustable during operation
- **Preset System**: Quick access to optimized configurations
- **Comprehensive Logging**: Track parameter changes and performance
- **Thread-safe Operations**: Smooth video playback with parameter updates

This makes it easy to extend with additional detection algorithms or integrate into larger inspection systems.
