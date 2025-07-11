# Image Processing Pipeline

A modular Python image processing pipeline with 15 individual scripts, each under 100 lines of code, with comprehensive unit tests.

## Overview

This project consists of multiple small, focused modules that work together to process images, analyze data, and detect patterns. Each module is designed to be independent and reusable.

## Modules

### Core Data Management

- **`data_store.py`**: Simple JSON-lines event store for logging all operations
- **`orchestrator.py`**: Auto-import system for dynamic module loading

### Data Generation

- **`pixel_generator.py`**: Generates random pixel values (0-255)
- **`realtime_processor.py`**: Runs pixel generation in background threads

### Image Processing

- **`intensity_reader.py`**: Reads images and logs pixel intensities
- **`batch_processor.py`**: Processes multiple images in a folder
- **`image_guided_generator.py`**: Reconstructs images from logged intensities

### Analysis Tools

- **`pattern_recognizer.py`**: Counts value frequencies in logged data
- **`anomaly_detector.py`**: Detects values that deviate significantly from mean
- **`trend_recorder.py`**: Records min/max/mean statistics
- **`geometry_analyzer.py`**: Computes gradients in reconstructed images

### Machine Learning

- **`learner.py`**: Builds and saves histogram models
- **`pytorch_module.py`**: PyTorch integration for neural networks
- **`hpc_module.py`**: GPU acceleration with CuPy
- **`opencv_module.py`**: OpenCV integration for computer vision

## Installation

1. Install Python 3.7+
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Note: Some modules (PyTorch, CuPy, OpenCV) gracefully fall back to CPU or skip functionality if not available.

## Usage

### Basic Usage

```bash
# Generate some random pixel data
python pixel_generator.py

# Read an image
python intensity_reader.py path/to/image.png

# Analyze patterns
python pattern_recognizer.py

# Detect anomalies
python anomaly_detector.py

# Record trends
python trend_recorder.py
```

### Batch Processing

```bash
# Process all images in a folder
python batch_processor.py path/to/image/folder
```

### Advanced Analysis

```bash
# Reconstruct image from logged data
python image_guided_generator.py

# Analyze geometry
python geometry_analyzer.py

# Learn a model
python learner.py
```

### Real-time Processing

```bash
# Run real-time pixel generation
python realtime_processor.py
```

## Testing

Run all unit tests:

```bash
python run_tests.py
```

Run individual test files:

```bash
python -m unittest test_data_store.py
python -m unittest test_pixel_generator.py
# etc.
```

## Demo

Run the comprehensive demo:

```bash
python demo.py
```

This demonstrates all modules working together in a complete pipeline.

## Architecture

The system uses a simple event-driven architecture:

1. **Data Generation**: Modules generate pixel/intensity data
2. **Storage**: All data is logged to `events.log` via `data_store.py`
3. **Analysis**: Various analysis modules process the logged data
4. **Output**: Results are printed and can be saved to files

## Features

- **Modular Design**: Each script is independent and focused
- **Comprehensive Testing**: Unit tests for all modules
- **Error Handling**: Graceful degradation when dependencies are missing
- **Cross-platform**: Works on Linux, Windows, and macOS
- **Memory Efficient**: Uses streaming and temporary files
- **Extensible**: Easy to add new analysis modules

## File Structure

```
├── requirements.txt          # Dependencies
├── data_store.py            # Core data storage
├── pixel_generator.py       # Random pixel generation
├── intensity_reader.py      # Image reading
├── batch_processor.py       # Batch image processing
├── image_guided_generator.py # Image reconstruction
├── pattern_recognizer.py    # Pattern analysis
├── anomaly_detector.py      # Anomaly detection
├── trend_recorder.py        # Statistical trends
├── geometry_analyzer.py     # Gradient analysis
├── learner.py              # Model learning
├── realtime_processor.py    # Real-time processing
├── pytorch_module.py        # PyTorch integration
├── hpc_module.py           # GPU acceleration
├── opencv_module.py        # OpenCV integration
├── orchestrator.py         # Module orchestration
├── demo.py                 # Demonstration script
├── run_tests.py            # Test runner
└── test_*.py               # Unit tests for each module
```

## Contributing

1. Each module should be under 100 lines
2. All modules must have comprehensive unit tests
3. Follow the existing error handling patterns
4. Add new modules to the demo script
5. Update this README when adding new functionality

## License

This project is provided as-is for educational and demonstration purposes.
