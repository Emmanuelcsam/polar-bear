# Realtime Video Visualizer Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
# Install Pylon SDK and other dependencies
pip install -r requirements.txt
```

### 2. Run Setup Script
```bash
# Test Pylon installation and camera detection
python setup_pylon.py
```

### 3. Test Camera Functionality
```bash
# Verify camera works correctly
python test_pylon.py
```

### 4. Run Realtime Visualizer
```bash
# Start realtime analysis (replace with your model path)
python src/realtime_visualizer.py --weights checkpoints/best_model.pth
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--weights` | Path to trained model | `--weights checkpoints/best_model.pth` |
| `--device` | GPU/CPU device | `--device cuda` or `--device cpu` |
| `--camera` | Camera index | `--camera 0` |

## Interactive Controls

- **'q'**: Quit the application
- **'s'**: Save current frame and analysis results
- **Mouse**: Click on plots for detailed information

## What You'll See

The visualizer displays 6 panels:

1. **Live Camera Feed**: Real-time video from your Pylon camera
2. **Core Region Detection**: Red overlay showing fiber core area
3. **Cladding Region Detection**: Blue overlay showing cladding area  
4. **Ferrule Region Detection**: Green overlay showing ferrule area
5. **Defect Probabilities**: Bar chart of top 10 detected defects
6. **Processing Info**: FPS, frame count, and device information

## Troubleshooting

### Camera Not Detected
```bash
# Check if Pylon is installed
python -c "from pypylon import pylon; print('Pylon OK')"

# List available cameras
python test_pylon.py
```

### Low Performance
- Use GPU: `--device cuda`
- Close other applications
- Check camera exposure settings

### Model Issues
- Ensure model file exists: `ls checkpoints/`
- Check model compatibility with your architecture

## Example Workflow

1. **Setup** (one-time):
   ```bash
   pip install -r requirements.txt
   python setup_pylon.py
   ```

2. **Daily Use**:
   ```bash
   # Start analysis
   python src/realtime_visualizer.py --weights checkpoints/best_model.pth
   
   # Press 's' to save interesting frames
   # Press 'q' to quit
   ```

3. **Review Results**:
   - Check saved frames: `captured_frame_*.png`
   - Review analysis: `captured_results_*.json`

## Advanced Usage

### Custom Camera Settings
Edit `pylon_config.json`:
```json
{
  "camera_settings": {
    "exposure_time": 15000,
    "gain": 2.0
  }
}
```

### Batch Processing
For processing video files instead of live camera:
```python
# Modify src/realtime_visualizer.py
# Replace camera capture with video file reading
cap = cv2.VideoCapture("your_video.mp4")
```

### Multiple Cameras
```bash
# Use different camera index
python src/realtime_visualizer.py --weights model.pth --camera 1
```

## Performance Tips

- **High FPS**: Use CUDA GPU, optimize camera settings
- **High Accuracy**: Increase exposure time, use higher resolution
- **Memory Issues**: Use CPU mode, close other applications

## Support

- **Pylon Issues**: Check [Basler documentation](https://docs.baslerweb.com/)
- **Model Issues**: Verify model checkpoint and architecture
- **Performance**: Monitor GPU usage and camera settings

The realtime visualizer provides immediate feedback on fiber optic end-face quality, making it ideal for production inspection and quality control workflows. 