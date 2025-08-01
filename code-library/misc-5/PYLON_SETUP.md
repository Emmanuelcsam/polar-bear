# Pylon Realtime Video Visualizer Setup Guide

This guide explains how to set up and use the realtime video visualizer for fiber optic end-face analysis using Pylon SDK.

## Overview

The realtime visualizer (`src/realtime_visualizer.py`) provides:
- **Live camera feed** from Pylon cameras or webcams
- **Real-time processing** of video frames through the CNN model
- **Interactive visualization** showing region detection and defect probabilities
- **Frame capture** and result saving capabilities

## Prerequisites

### 1. Pylon SDK Installation

#### Option A: Automatic Installation
```bash
# Run the setup script
python setup_pylon.py
```

#### Option B: Manual Installation
1. Download Pylon SDK from [Basler's website](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/pylon-6-3-0/)
2. Install the SDK for your platform (Windows/Linux)
3. Install Python wrapper:
```bash
pip install pypylon
```

### 2. Model Requirements
- Trained CNN model checkpoint (`.pth` file)
- CUDA-compatible GPU (recommended for real-time processing)

## Installation Steps

### Step 1: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install Pylon separately
pip install pypylon
```

### Step 2: Run Setup Script
```bash
python setup_pylon.py
```

This script will:
- Check Pylon SDK installation
- Detect available cameras
- Test camera connections
- Create configuration files

### Step 3: Verify Installation
```bash
# Test Pylon installation
python -c "from pypylon import pylon; print('Pylon SDK installed successfully')"
```

## Usage

### Basic Usage
```bash
python src/realtime_visualizer.py --weights checkpoints/best_model.pth
```

### Advanced Usage
```bash
python src/realtime_visualizer.py \
    --weights checkpoints/best_model.pth \
    --device cuda \
    --camera 0
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--weights` | Path to trained model weights | Required |
| `--device` | Device to use (cuda/cpu) | cuda |
| `--camera` | Camera index | 0 |

## Features

### 1. Live Camera Feed
- Real-time video capture from Pylon cameras
- Automatic fallback to webcam if Pylon cameras unavailable
- Configurable camera settings (exposure, gain, etc.)

### 2. Real-time Processing
- Each frame is processed through the CNN model
- Region detection (core, cladding, ferrule)
- Defect classification (40 defect types)
- Statistical feature extraction

### 3. Interactive Visualization
The visualizer displays 6 panels:
- **Live Camera Feed**: Raw camera input
- **Core Region Detection**: Red overlay showing core area
- **Cladding Region Detection**: Blue overlay showing cladding area
- **Ferrule Region Detection**: Green overlay showing ferrule area
- **Defect Probabilities**: Bar chart of top 10 defect probabilities
- **Processing Info**: FPS, frame count, device info

### 4. Interactive Controls
- **'q'**: Quit the application
- **'s'**: Save current frame and results
- **Mouse**: Click on plots for detailed information

## Camera Configuration

### Pylon Camera Settings
The visualizer automatically configures:
- **Pixel Format**: RGB8
- **Exposure Time**: 10ms (adjustable)
- **Gain**: 0 (adjustable)
- **Acquisition Mode**: Continuous
- **Trigger Mode**: Off

### Custom Configuration
Edit `pylon_config.json` to customize settings:
```json
{
  "camera_settings": {
    "pixel_format": "RGB8",
    "exposure_time": 10000,
    "gain": 0,
    "acquisition_mode": "Continuous",
    "trigger_mode": "Off"
  }
}
```

## Troubleshooting

### Common Issues

#### 1. "No Pylon cameras found"
- Ensure Pylon SDK is properly installed
- Check camera connections
- Verify camera drivers are installed
- The visualizer will fallback to webcam

#### 2. "Failed to open camera"
- Check camera permissions
- Ensure no other application is using the camera
- Try different camera index: `--camera 1`

#### 3. Low FPS
- Use GPU processing: `--device cuda`
- Reduce image resolution in transforms
- Close other applications using GPU
- Check camera exposure settings

#### 4. Memory Issues
- Reduce batch size
- Use CPU processing: `--device cpu`
- Close other applications

### Performance Optimization

#### For High FPS:
1. Use CUDA GPU processing
2. Optimize camera settings
3. Reduce image resolution
4. Close unnecessary applications

#### For High Accuracy:
1. Increase exposure time
2. Adjust gain settings
3. Use higher resolution images
4. Fine-tune defect thresholds

## File Structure

```
version5/
├── src/
│   ├── realtime_visualizer.py    # Main visualizer script
│   ├── infer.py                  # Batch inference script
│   ├── model.py                  # CNN model definition
│   └── dataset.py                # Data transforms
├── setup_pylon.py               # Pylon setup script
├── pylon_config.json            # Camera configuration
├── requirements.txt              # Dependencies
└── PYLON_SETUP.md              # This guide
```

## Output Files

When you press 's' to save:
- `captured_frame_[timestamp].png`: Current camera frame
- `captured_results_[timestamp].json`: Analysis results

### Results JSON Structure
```json
{
  "region_masks": {
    "core": [...],
    "cladding": [...],
    "ferrule": [...]
  },
  "defect_probabilities": [...],
  "statistical_features": [...],
  "defects_detected": [...],
  "confidence_scores": [...]
}
```

## Integration with Existing Pipeline

The realtime visualizer integrates seamlessly with your existing inference pipeline:

1. **Same Model**: Uses the same `EndfaceNet` model
2. **Same Transforms**: Uses `build_default_transforms`
3. **Same Processing**: Identical inference logic
4. **Compatible Output**: Same result format as `infer.py`

## Advanced Usage

### Custom Camera Selection
```python
# In realtime_visualizer.py, modify setup_camera()
# Select specific camera by serial number
for device in devices:
    if device.GetSerialNumber() == "YOUR_CAMERA_SERIAL":
        camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
        break
```

### Custom Visualization
```python
# Modify update_visualization() to add custom plots
# Add new subplot for custom analysis
self.axes[1, 3] = self.fig.add_subplot(2, 4, 8)
```

### Batch Processing Mode
```python
# Process video file instead of live camera
cap = cv2.VideoCapture("video.mp4")
# Replace capture_frame() with video reading
```

## Support

For issues with:
- **Pylon SDK**: Check [Basler documentation](https://docs.baslerweb.com/)
- **Model Issues**: Check your model checkpoint and architecture
- **Performance**: Monitor GPU usage and camera settings
- **Visualization**: Adjust matplotlib backend if needed

## Examples

### Basic Real-time Analysis
```bash
# Start realtime analysis with default settings
python src/realtime_visualizer.py --weights checkpoints/best_model.pth
```

### High-Performance Analysis
```bash
# Use GPU and specific camera
python src/realtime_visualizer.py \
    --weights checkpoints/best_model.pth \
    --device cuda \
    --camera 1
```

### Development Mode
```bash
# Use CPU for debugging
python src/realtime_visualizer.py \
    --weights checkpoints/best_model.pth \
    --device cpu
```

The realtime visualizer provides a powerful interface for live fiber optic end-face analysis, combining the accuracy of your trained CNN model with the convenience of real-time video processing. 