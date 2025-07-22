# Polar Bear System - Quick Start Guide

## System Overview

The Polar Bear System is a comprehensive fiber optic inspection and defect detection platform with advanced AI capabilities. It combines classical computer vision techniques with modern deep learning approaches to provide accurate, automated analysis of fiber optic components.

## Key Components

1. **Mega Connector** (`mega_connector.py`) - The ultimate integration system with complete understanding of all scripts
2. **Polar Bear Brain** (`polar_bear_brain.py`) - Intelligent orchestration system with real-time and batch processing
3. **Universal Connector** (`universal_connector.py`) - Dynamic script loading and execution
4. **Enhanced Hivemind** (`hivemind_connector.py`) - Distributed processing capabilities

## Installation

### 1. Test Your System
```bash
python system_test_and_troubleshoot.py
```

This will:
- Check Python version (3.7+ required)
- Verify all dependencies
- Test component integration
- Generate a detailed report

### 2. Install Missing Dependencies
The system will auto-install dependencies, but you can manually install:
```bash
pip install numpy opencv-python pillow scipy scikit-learn scikit-image matplotlib pandas torch torchvision tqdm
```

## Quick Start Commands

### 1. Basic System Status
```bash
python connector.py status
```

### 2. Run Complete Analysis on an Image
```bash
python mega_connector.py analyze path/to/image.jpg complete
```

### 3. Quick Defect Detection
```bash
python connector.py analyze path/to/image.jpg quick
```

### 4. Interactive Mode (Recommended for Beginners)
```bash
python mega_connector.py
```
This opens an interactive menu with all options.

### 5. Start the Brain System
```bash
python polar_bear_brain.py
```
This will start the main brain with configuration prompts.

## Processing Modes

### 1. Real-time Video Processing
- Processes live video feed from camera
- Displays results in real-time
- Suitable for continuous monitoring

### 2. Batch Image Processing
- Processes multiple images from a directory
- Generates comprehensive reports
- Ideal for quality control workflows

### 3. Hybrid Mode
- Combines real-time and batch processing
- Runs batch processing in background
- Provides maximum flexibility

## Available Pipelines

### Complete Inspection Pipeline
```python
# Steps: Preprocessing → Segmentation → Detection → Feature Extraction → Analysis → Reporting
python mega_connector.py analyze image.jpg complete
```

### Quick Defect Scan
```python
# Steps: Segmentation → Detection → Visualization
python mega_connector.py analyze image.jpg quick
```

### AI-Powered Analysis
```python
# Steps: Noise Reduction → AI Segmentation → AI Detection → Statistical Analysis
python mega_connector.py analyze image.jpg ai
```

## Key Features

### 1. Defect Detection Capabilities
- **Scratches**: Linear defects using LEI algorithm
- **Pits/Digs**: Circular defects using DO2MR algorithm
- **Contamination**: Surface anomalies using texture analysis
- **Cracks**: Edge-based detection
- **General Anomalies**: AI-powered anomaly detection

### 2. Zone-based Analysis
- Automatic fiber zone detection (core, cladding, ferrule)
- Zone-specific defect analysis
- Accurate geometric measurements

### 3. AI Models
- U-Net for semantic segmentation
- Convolutional Autoencoder for anomaly detection
- CNN for defect classification

### 4. Reporting
- JSON format for programmatic access
- CSV reports for data analysis
- Visual overlays showing detected defects
- Statistical summaries and quality metrics

## Directory Structure

```
input/          # Place images here for batch processing
output/         # Results will be saved here
reference/      # Reference data for learning mode
logs/           # System logs
models/         # Pre-trained AI models
```

## Configuration

### First Run
The system will prompt for configuration on first run:
- Processing mode (realtime/batch/hybrid)
- Input/output directories
- Detection sensitivity (1-10)
- GPU usage preference
- Learning mode enable/disable

### Manual Configuration
Edit `polar_bear_brain_config.json` or `polar_bear_config.json`

## Common Use Cases

### 1. Quality Control Inspection
```bash
# Place images in input/ directory
python polar_bear_brain.py
# Select batch mode
# Results will be in output/
```

### 2. Real-time Monitoring
```bash
python polar_bear_brain.py
# Select real-time mode
# Press 'q' to quit
```

### 3. Single Image Analysis
```bash
python mega_connector.py analyze sample.jpg complete
```

### 4. Custom Pipeline
```python
from mega_connector import get_mega_connector

connector = get_mega_connector()
result = connector.run_pipeline('custom_pipeline', image_data, config)
```

## Troubleshooting

### 1. Import Errors
Run: `python system_test_and_troubleshoot.py`

### 2. No GPU Detected
The system will automatically fall back to CPU mode.

### 3. Memory Issues
Reduce `max_workers` in configuration or process smaller batches.

### 4. Script Not Found
Ensure you're running from the polar-bear directory.

## Advanced Features

### 1. Learning Mode
```python
from mega_connector import get_mega_connector

connector = get_mega_connector()
connector.enable_learning_mode('reference_data/')
```

### 2. Custom Script Integration
All scripts are automatically discovered and can be accessed:
```python
connector.execute_script('your-script', 'function_name', *args)
```

### 3. Parameter Control
```python
# Get parameter
value = connector.get_parameter('script-name', 'parameter_name')

# Set parameter
connector.set_parameter('script-name', 'parameter_name', new_value)
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is installed for PyTorch acceleration
2. **Batch Processing**: Process multiple images together for efficiency
3. **Pipeline Selection**: Use 'quick' mode for rapid screening
4. **Parallel Processing**: Increase `max_workers` for multi-core systems

## Getting Help

1. Check logs in `logs/` directory
2. Run system test: `python system_test_and_troubleshoot.py`
3. Use interactive mode for guided operation
4. Review individual script docstrings for detailed functionality

## Example Workflow

```bash
# 1. Test system
python system_test_and_troubleshoot.py

# 2. Start interactive mode
python mega_connector.py

# 3. Select "Run Analysis Pipeline"
# 4. Choose "complete_inspection"
# 5. Enter image path
# 6. Review results in output/

# Or use command line:
python mega_connector.py analyze test_image.jpg complete
```

The system is designed to be flexible and scalable, supporting everything from simple defect detection to complex AI-powered analysis workflows.