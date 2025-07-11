# My Image Lab

A modular image processing laboratory with computer vision, machine learning, and distributed processing capabilities.

## Features

- **Core Infrastructure**: SQLite datastore, configurable settings, logging
- **Computer Vision**: OpenCV-based image processing, histogram analysis, anomaly detection
- **Machine Learning**: PyTorch autoencoder, scikit-learn clustering (optional)
- **Image Generation**: Random and distribution-guided pixel generation
- **Processing Modes**: Batch processing, real-time monitoring, HPC parallel processing
- **Modular Design**: Each module is independent and can be used separately

## Quick Start

```bash
# Run demo
python demo.py

# Run interactive menu
python main.py

# Run tests
python test_basic.py
```

## Module Overview

### Core Modules
- `config.py`: System configuration (device, cores, paths)
- `logger.py`: Timestamped logging utility
- `datastore.py`: SQLite-based key-value store with pickle serialization

### Processing Modules
- `cv_module.py`: OpenCV operations (grayscale loading, histograms, anomaly detection)
- `random_pixel.py`: Random and guided image generation
- `intensity_reader.py`: Learn pixel intensity distributions
- `pattern_recognizer.py`: K-means clustering for image categorization
- `anomaly_detector.py`: Statistical anomaly detection
- `batch_processor.py`: Process folders of images
- `realtime_processor.py`: Monitor folders for new images
- `hpc.py`: Parallel processing across CPU cores
- `torch_module.py`: Autoencoder for image generation (requires PyTorch)

## Dependencies

Required:
- numpy
- opencv-python
- pillow

Optional:
- torch (for autoencoder features)
- scikit-learn (for clustering)
- watchdog (for enhanced real-time monitoring)

## Architecture

- No circular dependencies between core/ and modules/
- Each module is self-contained (max 100 lines)
- Graceful handling of missing optional dependencies
- Thread-safe database operations
- Automatic data directory creation

## Testing

Unit tests are provided for all modules in the `tests/` directory. Run individual tests or use the test runner for comprehensive testing.