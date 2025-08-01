# Integrated Learning System for Core Detection

This system separates geometric core detection from `live_feed.py` and implements PyTorch-based learning for automatic detection alignment. The system allows manual detection (circle overlay) to train automatic detection, improving accuracy over time.

## Overview

The system consists of three main components:

1. **Geometric Core Detector** (`geometric_core_detector.py`) - Extracted from `live_feed.py` with PyTorch learning capabilities
2. **Circle Overlay** (`circle_overlay.py`) - Manual detection interface
3. **Integrated Learning System** (`integrated_learning_system.py`) - Combines both for learning

## Features

### Manual Detection (Circle Overlay)
- **WASD**: Move circle (W=up, S=down, A=left, D=right)
- **Q/E**: Resize circle (Q=smaller, E=larger)
- **L**: Lock/unlock circle position
- **R**: Reset circle to center
- **ESC**: Exit application

### Automatic Detection
- **Geometric**: Original Hough circle detection from `live_feed.py`
- **Improved**: PyTorch-learned detection that aligns with manual corrections

### Learning System
- **T**: Train from manual detection (requires locked circle)
- **M**: Toggle manual override
- **A**: Toggle automatic detection
- **I**: Toggle improved detection display

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Pylon camera support (optional):
```bash
pip install pypylon
```

## Usage

### Basic Usage

Run the integrated learning system:
```bash
python integrated_learning_system.py
```

### Advanced Usage

```bash
# Use specific camera
python integrated_learning_system.py --camera 1

# Use Pylon camera
python integrated_learning_system.py --pylon

# Export learning data
python integrated_learning_system.py --export learning_data.pkl
```

### Training Process

1. **Manual Detection**: Use WASD to position the circle over the core
2. **Lock Position**: Press 'L' to lock the manual circle
3. **Train Model**: Press 'T' to train the automatic detection from manual position
4. **Observe Improvement**: The improved detection (magenta) will align better over time

## System Architecture

### Geometric Core Detector (`geometric_core_detector.py`)

Extracts the geometric detection method from `live_feed.py` and adds:

- **IntensityProfileExtractor**: Extracts radial intensity profiles
- **CoreDetectionNetwork**: PyTorch neural network for learning
- **GeometricCoreDetector**: Main detector with learning capabilities

#### Key Methods:
- `geometric_detection()`: Original Hough circle detection
- `extract_features()`: Extract intensity profiles and image characteristics
- `learn_from_manual_detection()`: Train from manual corrections
- `improved_detection()`: Use learned model for better detection

### Circle Overlay (`circle_overlay.py`)

Provides manual detection interface with:
- Configurable movement and styling
- Lock/unlock functionality
- Performance optimizations

### Integrated Learning System (`integrated_learning_system.py`)

Combines manual and automatic detection:
- Real-time display of all detection methods
- Training interface
- Data export capabilities

## Learning Process

### Feature Extraction

The system extracts 77 features from each detection:

1. **Intensity Profile** (64 features): Radial intensity distribution
2. **Image Characteristics** (5 features):
   - Mean intensity
   - Standard deviation
   - Contrast
   - Gradient magnitude
   - Texture variance
3. **Pixel Analysis** (8 features):
   - Statistical measures (mean, std, median, min, max)
   - Percentiles (25th, 75th)
   - Edge density

### Neural Network Architecture

```
Input (77 features)
    ↓
Feature Extractor (128 → 64 → 32)
    ↓
Regression Head (32 → 16 → 3)  # x, y, radius
Confidence Head (32 → 16 → 1)  # confidence
```

### Training Process

1. **Manual Detection**: User positions circle and locks it
2. **Feature Extraction**: Extract features from manual detection region
3. **Model Training**: Train network to predict manual position from features
4. **Improved Detection**: Use trained model to enhance automatic detection

## Data Export

The system can export learning data for analysis:

```python
# Export learning data
learning_system.export_learning_data("learning_data.pkl")

# Export detector data
detector.export_learning_data("detector_data.pkl")
```

### Exported Data Structure

```python
{
    'detector_data': {
        'training_data': [...],  # Training samples
        'detection_history': [...],  # Detection history
        'model_path': 'core_detection_model.pth'
    },
    'learning_history': [...],  # Learning events
    'system_config': {...}  # System configuration
}
```

## Performance Optimization

### GPU Acceleration
- Automatic CUDA detection and usage
- GPU-accelerated feature extraction
- Optimized PyTorch model inference

### Real-time Processing
- Frame skipping for smooth operation
- Adaptive processing intervals
- Parallel feature extraction

## Configuration

### Circle Overlay Configuration
Edit `circle_config.json` or run:
```bash
python circle_config.py
```

### Model Configuration
- Model path: `core_detection_model.pth`
- Training data: `detection_data.pkl`
- Learning rate: 0.001
- Batch size: 1 (online learning)

## Analysis and Visualization

### Learning Progress
- Training loss tracking
- Confidence improvement over time
- Detection accuracy metrics

### Data Analysis
- Feature importance analysis
- Detection history visualization
- Performance metrics

## Integration with Other Projects

The learning data can be used for:

1. **Transfer Learning**: Apply learned features to other detection tasks
2. **Feature Analysis**: Analyze what characteristics make good detections
3. **Model Comparison**: Compare different learning approaches
4. **Performance Optimization**: Identify bottlenecks and optimize

## Troubleshooting

### Common Issues

1. **Camera not found**: Try different camera indices or check Pylon installation
2. **PyTorch not available**: Install PyTorch with CUDA support if needed
3. **Low FPS**: Reduce processing frequency or disable some detections
4. **Poor learning**: Ensure manual detections are accurate and consistent

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Multi-class Detection**: Detect different types of cores
2. **Temporal Learning**: Learn from detection sequences
3. **Active Learning**: Automatically select training samples
4. **Ensemble Methods**: Combine multiple detection approaches
5. **Real-time Adaptation**: Adapt to changing conditions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 