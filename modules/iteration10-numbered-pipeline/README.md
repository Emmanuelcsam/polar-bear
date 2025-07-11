# README.md

# Image Processing System with Comprehensive Unit Tests

This project contains a modular image processing system with 10 independent scripts and comprehensive unit tests for each module. The system is designed for high modularity where each script can run independently and communicates through files.

## Project Structure

```
.
├── 0_config.py                    # Central configuration module
├── 1_batch_processor.py           # Batch processing orchestrator
├── 2_intensity_reader.py          # Image intensity data extractor
├── 3_pattern_recognizer.py        # Statistical pattern analysis
├── 4_generative_learner.py        # Pixel distribution learning (PyTorch)
├── 5_image_generator.py           # Image generation from learned models
├── 6_deviation_detector.py        # Anomaly detection in images
├── 7_geometry_recognizer.py       # Geometric pattern detection
├── 8_real_time_anomaly.py         # Real-time webcam anomaly detection
├── 9_gpu_example.py               # GPU computation example
├── 10_hpc_parallel_cpu.py         # High-performance parallel processing
├── requirements.txt               # Python dependencies
├── run_all_tests.py               # Master test runner
├── test_0_config.py               # Tests for config module
├── test_1_batch_processor.py      # Tests for batch processor
├── test_2_intensity_reader.py     # Tests for intensity reader
├── test_3_pattern_recognizer.py   # Tests for pattern recognizer
├── test_4_generative_learner.py   # Tests for generative learner
├── test_5_image_generator.py      # Tests for image generator
├── test_6_deviation_detector.py   # Tests for deviation detector
├── test_7_geometry_recognizer.py  # Tests for geometry recognizer
├── test_8_real_time_anomaly.py    # Tests for real-time anomaly
├── test_9_gpu_example.py          # Tests for GPU example
├── test_10_hpc_parallel_cpu.py    # Tests for HPC parallel CPU
├── images_input/                  # Directory for input images
├── data/                          # Directory for inter-script data
└── output/                        # Directory for generated outputs
```

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Create sample images** (optional):
   Place some `.png` or `.jpg` images in the `images_input/` directory to test the system.

## Running the Scripts

### Individual Scripts

Each script can be run independently:

```bash
# Configuration
python 0_config.py

# Batch processing
python 1_batch_processor.py

# Extract pixel intensities
python 2_intensity_reader.py

# Analyze patterns
python 3_pattern_recognizer.py

# Learn pixel distributions
python 4_generative_learner.py

# Generate new images
python 5_image_generator.py

# Detect anomalies
python 6_deviation_detector.py

# Recognize geometric patterns
python 7_geometry_recognizer.py

# Real-time anomaly detection
python 8_real_time_anomaly.py

# GPU example
python 9_gpu_example.py

# Parallel processing
python 10_hpc_parallel_cpu.py
```

### Running All Tests

**Master Test Runner:**

```bash
python run_all_tests.py
```

**Individual Test Files:**

```bash
# Run specific test
python test_0_config.py
python test_1_batch_processor.py
# ... etc
```

## Testing Framework

### Test Coverage

Each script has comprehensive unit tests covering:

- **Import Testing**: Verifies modules can be imported correctly
- **Function Testing**: Tests all public functions with various inputs
- **Error Handling**: Tests graceful handling of errors and edge cases
- **Integration Testing**: Tests integration with the config module
- **Mocking**: Uses mocks to test functionality without external dependencies

### Test Features

1. **Dependency Handling**: Tests work whether optional dependencies (OpenCV, PyTorch) are installed or not
2. **File System Testing**: Uses temporary directories to avoid affecting the real file system
3. **Error Simulation**: Tests error conditions and edge cases
4. **Performance Testing**: Includes tests for parallel processing and GPU operations
5. **Real-world Simulation**: Creates realistic test data and scenarios

### Test Structure

Each test file follows this structure:

```python
# test_X_module_name.py
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Test functions
def test_module_import():
    """Test that module imports correctly"""

def test_function_with_valid_input():
    """Test function with valid input"""

def test_function_with_invalid_input():
    """Test function error handling"""

def test_config_integration():
    """Test config module integration"""

# Test runner
if __name__ == "__main__":
    # Run all tests and display results
```

## System Architecture

### Modular Design

- **Independent Scripts**: Each script is self-contained
- **File-based Communication**: Scripts communicate through `.npy`, `.csv`, and `.png` files
- **Configurable**: Central configuration in `0_config.py`
- **Fault Tolerant**: Missing dependencies are handled gracefully

### Data Flow

1. **Input**: Images placed in `images_input/`
2. **Processing**: Scripts process data and save intermediate results to `data/`
3. **Output**: Final results saved to `output/`

### Dependencies

- **Core**: `numpy`, `opencv-python`
- **ML**: `torch`, `torchvision` (optional)
- **Testing**: `pytest` (optional, tests work without it)
- **System**: `multiprocessing` (built-in)

## Key Features

### Script Capabilities

- **Batch Processing**: Automated image processing workflows
- **Statistical Analysis**: Pixel intensity patterns and distributions
- **Machine Learning**: Generative models for image synthesis
- **Computer Vision**: Edge detection, anomaly detection
- **Real-time Processing**: Webcam-based anomaly detection
- **High Performance**: GPU acceleration and parallel CPU processing

### Test Capabilities

- **100% Function Coverage**: Every public function is tested
- **Edge Case Testing**: Comprehensive error and boundary testing
- **Mock Testing**: Tests work without external hardware (camera, GPU)
- **Temporary File Testing**: Safe testing without affecting real files
- **Performance Testing**: Parallel processing and timing tests

## Usage Examples

### Basic Workflow

1. Place images in `images_input/`
2. Run intensity reader: `python 2_intensity_reader.py`
3. Analyze patterns: `python 3_pattern_recognizer.py`
4. Generate new images: `python 4_generative_learner.py && python 5_image_generator.py`

### Testing Workflow

1. Run all tests: `python run_all_tests.py`
2. Check individual modules: `python test_X_module.py`
3. Review test reports and fix any issues

## Contributing

When adding new functionality:

1. Update the corresponding script
2. Add comprehensive tests to the test file
3. Update this README if needed
4. Run the full test suite to ensure no regressions

## Troubleshooting

### Common Issues

- **Import Errors**: Install missing dependencies with `pip install -r requirements.txt`
- **No Images**: Place test images in `images_input/` directory
- **GPU Tests**: GPU tests gracefully handle systems without CUDA
- **Camera Tests**: Real-time tests work with mocked camera input

### Test Failures

- Check the detailed error output from `run_all_tests.py`
- Individual tests can be run for more detailed debugging
- Most tests include fallback behavior for missing dependencies

## Performance Notes

- **Parallel Processing**: Uses all available CPU cores
- **GPU Acceleration**: Automatically detects and uses CUDA if available
- **Memory Efficient**: Processes images in batches to manage memory
- **Scalable**: Works with any number of input images

## License

This project is designed for educational and research purposes. Feel free to modify and extend as needed.
