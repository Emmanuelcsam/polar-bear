# VideoDisplayFrequency Widget Documentation

## Overview
The `VideoDisplayFrequency` widget is a comprehensive Tkinter-based component for real-time frequency domain analysis and visualization of images. It provides side-by-side display of original images and their frequency spectra with interactive filtering capabilities.

## Features Implemented

### 1. Display Components
- **Dual Canvas Display**: Side-by-side visualization of:
  - Original image (left panel)
  - Frequency spectrum with colormap (right panel)
- **Adaptive Sizing**: Automatically scales images to fit display areas
- **Clear Visual Separation**: Labeled panels with borders

### 2. Frequency Processing
- **FFT Analysis**: Real-time 2D Fast Fourier Transform computation
- **Spectrum Visualization**: Log-scale magnitude spectrum with JET colormap
- **Frequency Filtering**:
  - Lowpass filter (removes high frequencies)
  - Highpass filter (removes low frequencies)  
  - Bandpass filter (keeps frequencies in a range)
- **Filter Visualization**: Overlay showing filter cutoff on spectrum

### 3. Feature Extraction
- **FFT Features**:
  - DC Component
  - Mean Magnitude
  - Standard Deviation
  - Spectral Centroid
  - Spectral Spread
  - High Frequency Ratio
  - Total Power
  - Phase Statistics

- **Pattern Detection**:
  - Automatic detection of periodic patterns
  - Identification of frequency peaks
  - Pattern counting and localization

### 4. Information Display
- **Real-time Feature Labels**: 
  - DC Component value
  - Mean Magnitude
  - Spectral Centroid
  - High Frequency Ratio
  - Periodic Pattern count
  - Filter Status

## Widget API

### Initialization
```python
widget = VideoDisplayFrequency(parent, width=640, height=480)
```

### Main Methods

#### `update_frame(image, apply_filter, filter_type, cutoff_freq)`
Updates the display with a new image and applies frequency processing.

**Parameters:**
- `image`: Input image (numpy array, grayscale or color)
- `apply_filter`: Boolean, whether to apply frequency filter
- `filter_type`: String, 'lowpass', 'highpass', or 'bandpass'
- `cutoff_freq`: Float (0-1), normalized cutoff frequency

**Returns:**
- Processed image (numpy array)

#### `get_frequency_features()`
Returns dictionary of extracted frequency domain features.

#### `get_periodic_patterns()`
Returns list of detected periodic pattern frequencies as (x, y) tuples.

#### `get_processed_image()`
Returns the current processed image.

#### `clear_display()`
Clears both display canvases and resets all data.

## Integration Example

```python
import tkinter as tk
from video_display_frequency import VideoDisplayFrequency

root = tk.Tk()
freq_widget = VideoDisplayFrequency(root)
freq_widget.pack()

# Process an image
import cv2
img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
processed = freq_widget.update_frame(
    img, 
    apply_filter=True,
    filter_type='lowpass',
    cutoff_freq=0.3
)

# Get features
features = freq_widget.get_frequency_features()
print(f"DC Component: {features['fft_dc_component']}")

root.mainloop()
```

## Test Applications

### 1. `video_display_frequency.py`
- Standalone widget with built-in demo
- Generates test patterns with various frequency components
- Interactive filter controls
- Run with: `python video_display_frequency.py`

### 2. `test_video_display_frequency.py`
- Comprehensive test application
- Multiple test pattern generators:
  - Sinusoidal (multiple frequencies)
  - Checkerboard (regular pattern)
  - Gradient (smooth transitions)
  - Noise (random frequencies)
  - Circles (concentric patterns)
- Feature report generation
- Image load/save capabilities
- Run with: `python test_video_display_frequency.py`

## Technical Details

### Frequency Spectrum Visualization
- Uses log transformation for better dynamic range
- JET colormap for intuitive frequency magnitude representation
- Real-time filter overlay showing active filter region

### Filter Implementation
- **Lowpass**: Circular mask in frequency domain (keeps low frequencies)
- **Highpass**: Inverted circular mask (keeps high frequencies)
- **Bandpass**: Ring-shaped mask (keeps mid-range frequencies)

### Performance Optimizations
- Efficient FFT computation using NumPy
- Image caching to avoid redundant processing
- Lazy evaluation of features

## Dependencies
- `numpy`: FFT and array operations
- `opencv-python`: Image processing and filters
- `PIL/Pillow`: Image display in Tkinter
- `tkinter`: GUI framework (included with Python)

## Files Created
1. `video_display_frequency.py` - Main widget implementation
2. `test_video_display_frequency.py` - Comprehensive test suite
3. `VIDEO_DISPLAY_FREQUENCY_README.md` - This documentation

## Future Enhancements
- Add support for color images (process each channel)
- Implement additional filter types (Gaussian, Butterworth)
- Add frequency domain editing capabilities
- Export frequency analysis reports
- Real-time video stream processing
- 3D frequency spectrum visualization
