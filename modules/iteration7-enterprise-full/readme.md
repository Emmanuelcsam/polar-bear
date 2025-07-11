# Modular Image Analysis System

A collection of minimal, independent Python scripts that work together to analyze pixel intensities, recognize patterns, and generate new images. Enhanced with **PyTorch neural networks**, **OpenCV computer vision**, **real-time processing**, **High Performance Computing (HPC)** with GPU acceleration, and **advanced analysis tools**.

## Complete System: 45+ Independent Modules

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure the system
python config_manager.py

# Apply a preset configuration
python config_manager.py preset basic              # Basic analysis
python config_manager.py preset ai_powered         # AI/ML modules
python config_manager.py preset high_performance   # GPU/HPC modules
python config_manager.py preset real_time          # Real-time processing
python config_manager.py preset full_system        # Everything

# Run demos
python create_test_image.py     # Create test data
python quick_start.py           # Basic demo
python ai_demo.py              # AI capabilities
python hpc_demo.py             # GPU/HPC demo
python realtime_demo.py        # Real-time demo

# Or run everything
python main_controller.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Scripts Overview

### Core Scripts
- **pixel_reader.py** - Reads pixel intensities from images (like intensity-reader.py)
- **random_generator.py** - Continuously generates random pixel values
- **correlator.py** - Finds correlations between pixel reader and random generator

### Analysis Scripts
- **pattern_recognizer.py** - Identifies patterns in pixel data
- **anomaly_detector.py** - Detects extreme deviations using statistical methods
- **intensity_analyzer.py** - Analyzes intensity distributions and clusters
- **geometry_analyzer.py** - Finds geometric patterns and symmetries
- **trend_analyzer.py** - Analyzes trends across multiple analyses

### Computer Vision (OpenCV)
- **vision_processor.py** - Advanced image processing with edge detection, corner detection, contours, and texture analysis
- **hybrid_analyzer.py** - Combines neural network and vision processing results for advanced insights

### Deep Learning (PyTorch)
- **neural_learner.py** - Neural network that learns pixel patterns and makes predictions
- **neural_generator.py** - Generates new images using neural networks and learned features

### High Performance Computing (HPC)
- **gpu_accelerator.py** - GPU-accelerated processing using CUDA/PyTorch, with CPU fallback
- **parallel_processor.py** - Multi-core parallel processing using Python multiprocessing
- **distributed_analyzer.py** - Simulated distributed computing across multiple nodes
- **hpc_optimizer.py** - Automatically optimizes algorithms based on hardware capabilities

### Processing Scripts
- **data_store.py** - Manages persistent storage of learned data
- **data_calculator.py** - Performs intensive calculations (FFT, entropy, etc.)
- **batch_processor.py** - Processes multiple images in a folder
- **image_categorizer.py** - Categorizes images based on learned characteristics

### Generation Scripts
- **image_generator.py** - Creates new images based on learned patterns
- **learning_engine.py** - Handles manual and automatic learning modes

### Real-Time Processing
- **realtime_processor.py** - Monitors all data streams and triggers actions based on conditions
- **live_capture.py** - Captures live frames from camera or generates simulated video stream
- **stream_analyzer.py** - Analyzes multiple data streams for patterns, trends, and correlations

### Utility Scripts
- **continuous_analyzer.py** - Runs continuous analysis for a fixed duration
- **logger.py** - Centralized logging utility
- **main_controller.py** - Example orchestration of all modules
- **visualizer.py** - Creates visualizations of analysis results
- **quick_start.py** - Quick demo of basic functionality
- **communication_demo.py** - Shows how scripts communicate
- **ai_demo.py** - Demonstrates PyTorch and OpenCV features
- **realtime_demo.py** - Shows real-time processing capabilities
- **hpc_demo.py** - Demonstrates GPU and HPC capabilities
- **test_independence.py** - Verifies modules work independently
- **test_realtime_independence.py** - Verifies real-time modules work independently
- **test_hpc_independence.py** - Verifies HPC modules adapt to hardware
- **create_test_image.py** - Creates a test image with known patterns

## Advanced Features

### Configuration Management
The system includes a powerful configuration manager:
```bash
python config_manager.py                    # View current configuration
python config_manager.py interactive        # Interactive configuration mode
python config_manager.py preset basic       # Apply preset configuration
```

Available presets:
- **basic** - Essential modules for pixel analysis
- **ai_powered** - AI and machine learning modules
- **high_performance** - GPU and HPC modules
- **real_time** - Real-time processing modules
- **full_system** - All modules enabled

### Network API
Remote processing capabilities:
```bash
python network_api.py                       # Start API server on port 8080
python network_api.py 9000                  # Start on custom port
python network_api.py client                # Test client
python network_api.py remote                # Create remote processor
```

API Endpoints:
- `GET /status` - System status
- `GET /results` - All analysis results
- `POST /analyze` - Submit data for analysis
- `POST /trigger` - Trigger specific module

### Data Export/Import
Support for multiple formats:
```bash
python data_exporter.py                     # Export all data
```

Supported formats:
- JSON (native format)
- CSV (spreadsheet compatible)
- HDF5 (efficient binary storage)
- XML (structured data)
- YAML (human-readable config)
- ZIP (complete export packages)

### Advanced Visualization
Create sophisticated visualizations:
```bash
python advanced_visualizer.py               # Generate all visualizations
```

Features:
- Interactive dashboards
- 3D feature space visualization
- Animated pixel evolution
- Correlation matrices
- Time series analysis
- PDF reports

### Machine Learning
Advanced ML capabilities:
```bash
python ml_classifier.py                     # Run ML analysis
```

Features:
- K-means and DBSCAN clustering
- Random Forest and SVM classification
- Anomaly detection (Isolation Forest, LOF)
- Feature extraction and importance
- Model persistence

## Computer Vision with OpenCV
The `vision_processor.py` module adds powerful computer vision capabilities:
- **Edge Detection**: Canny edges, Sobel gradients
- **Corner Detection**: Harris corners, FAST features
- **Contour Analysis**: Shape detection and classification
- **Texture Analysis**: Local Binary Patterns, entropy
- **Feature Detection**: ORB keypoints, Hu moments

Results integrate seamlessly with the pixel analysis system through JSON files.

### Deep Learning with PyTorch
The neural modules add machine learning capabilities:
- **neural_learner.py**: Trains a neural network on pixel sequences
- **neural_generator.py**: Generates new images using learned patterns
- **hybrid_analyzer.py**: Combines vision and neural insights

The neural network learns from:
- Pixel sequences and patterns
- Vision features (edges, textures)
- Statistical distributions
- Correlation data

### How They Work Together
1. **Vision → Neural**: Vision features become inputs for neural generation
2. **Neural → Generation**: Neural predictions guide image creation
3. **Hybrid Analysis**: Combines both for advanced pattern detection
4. **No Dependencies**: Each module works independently through JSON

Example workflow with AI modules:
```bash
# Run vision analysis
python vision_processor.py

# Train neural network
python neural_learner.py

# Combine insights
python hybrid_analyzer.py

# Generate with AI
python neural_generator.py
```

## Real-Time Processing

The system now includes powerful real-time processing capabilities:

### Real-Time Modules

1. **realtime_processor.py** - Real-time monitoring and triggering
   - Monitors multiple JSON data streams simultaneously
   - Calculates real-time statistics (variance, rates, anomalies)
   - Triggers analysis modules based on conditions
   - Live dashboard with buffer visualization
   - Saves metrics to `realtime_metrics.json`

2. **live_capture.py** - Live video/image capture
   - Captures from webcam using OpenCV (if available)
   - Falls back to screenshot capture or simulation
   - Configurable FPS and duration
   - Saves frames to `pixel_data.json` for compatibility
   - Monitor mode to watch live feed

3. **stream_analyzer.py** - Multi-stream pattern analysis
   - Analyzes patterns across multiple data streams
   - Detects bursts, trends, cycles, and anomalies
   - Cross-stream correlation analysis
   - Real-time alerts for significant events
   - Comprehensive stream statistics

### Real-Time Features

**Automatic Triggering:**
- High pixel variance → Triggers pattern recognition
- Correlation burst → Triggers neural learning
- Anomaly spike → Triggers deep calculation
- Edge density change → Triggers image generation
- Stable predictions → Triggers trend analysis

**Live Dashboard:**
```
=== REAL-TIME PROCESSING DASHBOARD ===
Uptime: 45s

STATISTICS:
  Pixels Processed: 45,230
  Correlations Found: 127
  Anomalies Detected: 23
  Triggers Fired: 8

REAL-TIME METRICS:
  Processing Rate: 1005.1 pixels/sec
  Correlation Rate: 2.82 /sec
  Anomaly Rate: 0.05%
  Current Variance: 2847.3
  Prediction Stability: 87.2%
```

### Running Real-Time Processing

```bash
# Run the complete real-time demo
python realtime_demo.py

# Or run components individually:
python live_capture.py               # Start live capture
python realtime_processor.py         # Start real-time monitor
python stream_analyzer.py            # Start stream analysis

# Configure live capture
python live_capture.py fps=30        # 30 FPS capture
python live_capture.py duration=60   # Capture for 60 seconds
python live_capture.py monitor       # Monitor mode
```

## High Performance Computing (HPC)

The system now includes powerful HPC capabilities for massive datasets:

### HPC Modules

1. **gpu_accelerator.py** - GPU-accelerated processing
   - Automatic GPU detection (CUDA, Apple Metal)
   - 10-100x speedup for parallel operations
   - Falls back to CPU if no GPU available
   - Batch processing optimization
   - Memory usage tracking

2. **parallel_processor.py** - Multi-core CPU parallelization
   - Automatic CPU core detection
   - Map-reduce operations
   - Parallel correlation analysis
   - Chunk-based processing
   - Linear scaling with CPU cores

3. **distributed_analyzer.py** - Distributed computing
   - Simulates multi-node clusters
   - Load balancing across nodes
   - Map-reduce implementation
   - Network overhead simulation
   - Scalable to thousands of nodes

4. **hpc_optimizer.py** - Intelligent optimization
   - Hardware capability detection
   - Automatic algorithm selection
   - Performance benchmarking
   - Scaling analysis
   - Pipeline optimization

### HPC Features

**Automatic Hardware Detection:**
```
[HPC] System Resources:
  CPU: 16 cores @ 3.5 GHz
  Memory: 24.3/32.0 GB available
  GPU: NVIDIA RTX 3080 (10.0 GB)
  Compute Score: 134.5
```

**Performance Scaling:**
```
Size       Serial         Parallel       GPU            Distributed    
1000       10000          160000         100000         320000         
10000      1000           16000          1000000        32000          
100000     100            1600           10000000       3200           
1000000    10             160            100000000      320            
```

**Smart Algorithm Selection:**
- Small data (<1K): Serial processing
- Medium data (1K-10K): Parallel CPU
- Large data (>10K): GPU acceleration
- Massive data (>1M): Distributed computing

### Running HPC Processing

```bash
# Run the complete HPC demo
python hpc_demo.py

# Or run components individually:
python gpu_accelerator.py        # GPU processing
python parallel_processor.py     # Parallel CPU
python distributed_analyzer.py   # Distributed computing
python hpc_optimizer.py         # Optimization analysis
```

## How Scripts Communicate

The original `intensity-reader.py` and `random-pixel-generator.py` communicate through the modular system:

1. **pixel_reader.py** reads image pixels (like intensity-reader.py) and saves to `pixel_data.json`
2. **random_generator.py** generates random values and saves to `random_value.json`
3. **correlator.py** watches both files and records matches in `correlations.json`
4. **data_store.py** learns from these correlations
5. **image_generator.py** uses learned data to generate new images

Run `python communication_demo.py` to see this in action!

### Basic Usage

1. **Analyze a single image:**
```bash
python pixel_reader.py
python pattern_recognizer.py
python intensity_analyzer.py
```

2. **Run the correlation system:**
```bash
# Terminal 1
python random_generator.py

# Terminal 2
python correlator.py
```

3. **Process multiple images:**
```bash
python batch_processor.py
```

4. **Generate new images:**
```bash
python image_generator.py
```

### Advanced Usage

**Run complete analysis pipeline:**
```bash
python main_controller.py
```

**Continuous analysis for 60 seconds:**
```bash
python continuous_analyzer.py
```

**Visualize results:**
```bash
python visualizer.py
```

**Manual learning mode:**
```python
from learning_engine import LearningEngine
engine = LearningEngine()
engine.manual_learn('myimage.jpg', 'category_name')
```

## Output Files

### Core Analysis
- `pixel_data.json` - Raw pixel data from images
- `correlations.json` - Correlation matches between reader and generator
- `patterns.json` - Detected patterns and sequences
- `anomalies.json` - Statistical anomalies
- `intensity_analysis.json` - Intensity distribution analysis
- `geometry_analysis.json` - Geometric patterns
- `learned_data.json` - Accumulated learned information
- `trends.json` - Trend analysis results
- `calculations.json` - Advanced mathematical calculations

### Computer Vision (OpenCV)
- `vision_results.json` - Complete vision analysis (edges, corners, contours, texture)
- `vision_features.jpg` - Visualization of detected features
- `edges_canny.jpg` - Edge detection result
- `vision_integration.json` - Integration with pixel data
- `vision_patterns.json` - Geometric patterns from vision

### Deep Learning (PyTorch)
- `neural_results.json` - Neural network training results and predictions
- `neural_patterns.json` - Patterns learned by the network
- `neural_prediction.json` - Current prediction based on correlations
- `pixel_model.pth` - Saved PyTorch model
- `neural_generation.json` - Neural generation metadata
- `neural_generated_*.jpg` - Images generated by neural network

### Machine Learning
- `ml_clustering.json` - Clustering analysis results
- `ml_classification.json` - Classification results and accuracy
- `ml_anomalies.json` - ML-based anomaly detection
- `ml_prediction.json` - Predictions for new images
- `ml_report.json` - Comprehensive ML analysis report
- `ml_*.pkl` - Saved ML models (scikit-learn)

### High Performance Computing (HPC)
- `gpu_results.json` - GPU processing results and performance metrics
- `gpu_batch_results.json` - GPU batch processing statistics
- `gpu_generation_benchmark.json` - GPU generation benchmarks
- `gpu_fractal_*.jpg` - GPU-generated fractal images
- `gpu_pattern_*.jpg` - GPU-generated patterns
- `parallel_results.json` - Parallel processing analysis
- `parallel_correlations.json` - Multi-threaded correlation results
- `parallel_batch.json` - Parallel batch processing results
- `distributed_results.json` - Distributed computing simulation results
- `distributed_mapreduce.json` - Map-reduce operation results
- `hpc_benchmarks.json` - Performance benchmarks across algorithms
- `hpc_pipeline.json` - Optimized processing pipeline for your hardware
- `hpc_scaling.json` - Performance scaling analysis

### Real-Time Processing
- `realtime_metrics.json` - Live processing metrics and statistics
- `realtime_triggers.json` - Triggered events and conditions
- `realtime_report.json` - Final real-time processing report
- `stream_analysis.json` - Stream patterns and correlations
- `stream_alerts.json` - Real-time alerts and warnings
- `stream_report.json` - Comprehensive stream analysis report
- `live_frame.json` - Current live frame data
- `live_buffer.json` - Recent frame buffer statistics
- `live_capture_stats.json` - Live capture performance stats
- `live_current.jpg` - Most recent captured frame

### Configuration & System
- `system_config.json` - System-wide configuration
- `config_presets.json` - Configuration presets
- `config_presets.yaml` - Human-readable presets
- `config_report.json` - Configuration analysis report
- `pipelines.json` - Saved processing pipelines
- `system_log.json` - System-wide logging
- `network_api_info.json` - Network API server information
- `remote_config.json` - Remote processing configuration

### Visualization & Export
- `advanced_dashboard.png` - Comprehensive analysis dashboard
- `pixel_evolution.gif` - Animated pixel evolution
- `visual_analysis_report.pdf` - Multi-page PDF report
- `analysis_visualization.png` - Basic visual summary chart
- `pixel_analysis_data.h5` - HDF5 binary data export
- `pixel_analysis_data.xml` - XML structured export
- `pixel_analysis_config.yaml` - YAML configuration export
- `pixel_analysis_export_*.zip` - Complete export packages

### Other Outputs
- `generated_*.jpg` - Generated images
- `batch_results.json` - Batch processing summary
- `image_categories.json` - Image categorization
- `generation_log.json` - Generation history

## Modular Design

Each script is designed to:
- Run independently
- Communicate through JSON files
- Log all operations to terminal
- Handle errors gracefully
- Be deleted without breaking other scripts

## Example Workflow

1. Place images in the script directory
2. Run `pixel_reader.py` to extract pixel data
3. Run analysis scripts to understand patterns
4. Use `learning_engine.py` to learn from the data
5. Generate new images with `image_generator.py`
6. Categorize results with `image_categorizer.py`

## Notes

- All scripts use minimal code for maximum clarity
- No argparse - scripts use default values or read from files
- Latest library versions specified in requirements.txt
- Terminal logging shows real-time progress
- Delete any script and others continue working
- Make scripts executable with: `chmod +x *.py`
- **PyTorch and OpenCV modules are optional** - system works without them
- **Real-time modules are optional** - core analysis works without them
- **HPC modules are optional** - work with or without GPU
- Each module communicates only through JSON files - no direct dependencies

## Module Independence

The system maintains complete independence:
- **If PyTorch is missing**: Neural modules skip gracefully, other scripts continue
- **If OpenCV is missing**: Vision module skips, system still functions
- **If GPU is missing**: GPU module falls back to CPU automatically
- **If multiple cores unavailable**: Parallel module adjusts to available cores
- **Real-time modules**: Work independently, don't affect core processing
- **HPC modules**: Adapt to available hardware automatically
- **No module imports another**: Communication only via JSON files

This means you can:
1. Delete `neural_learner.py` and OpenCV still works
2. Delete `vision_processor.py` and PyTorch still works
3. Delete `realtime_processor.py` and everything else works
4. Delete `gpu_accelerator.py` and parallel processing still works
5. Delete all AI/real-time/HPC modules and core system works perfectly
6. Run modules in any order or combination

## Troubleshooting

If you get "No image found" errors:
1. Run `python create_test_image.py` to create a test image
2. Or place any .jpg, .png, or .bmp file in the script directory

If PyTorch/OpenCV installation fails:
1. The core system still works without them
2. Try: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
3. For OpenCV issues: `pip install opencv-python-headless`

If GPU not detected:
1. HPC modules automatically fall back to CPU
2. Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. For NVIDIA GPUs: Install CUDA toolkit
4. For AMD GPUs: Currently CPU fallback only

Performance issues:
1. HPC optimizer automatically selects best algorithm
2. Reduce data size for testing
3. Check available memory with `python hpc_optimizer.py`

## Complete Workflow Example

Here's how all 45+ modules work together:

```bash
# 1. Setup & Configuration
python config_manager.py                 # Configure system
python create_test_image.py              # Create test data

# 2. Core Analysis
python pixel_reader.py                   # Extract pixels
python random_generator.py &             # Start generator (background)
python correlator.py &                   # Start correlator (background)
python pattern_recognizer.py             # Find patterns
python anomaly_detector.py               # Detect outliers

# 3. Advanced Analysis
python intensity_analyzer.py             # Intensity distributions
python geometry_analyzer.py              # Geometric patterns
python data_calculator.py                # FFT, entropy

# 4. Computer Vision (OpenCV)
python vision_processor.py               # Edges, corners, contours

# 5. Machine Learning
python ml_classifier.py                  # Clustering, classification
python neural_learner.py                 # Train neural network
python neural_generator.py               # Generate with AI

# 6. High Performance Computing
python gpu_accelerator.py                # GPU acceleration
python gpu_image_generator.py            # GPU image generation
python parallel_processor.py             # Multi-core processing
python distributed_analyzer.py           # Distributed computing
python hpc_optimizer.py                  # Optimize for hardware

# 7. Real-Time Processing
python live_capture.py &                 # Start live capture
python realtime_processor.py &           # Monitor and trigger
python stream_analyzer.py &              # Analyze data streams

# 8. Integration & Analysis
python hybrid_analyzer.py                # Combine all insights
python trend_analyzer.py                 # Analyze trends
python image_categorizer.py              # Categorize results

# 9. Visualization & Export
python advanced_visualizer.py            # Create visualizations
python data_exporter.py                  # Export all data

# 10. Network API (optional)
python network_api.py &                  # Start API server

# Or run preset pipelines:
python main_controller.py                # Run standard pipeline

# Or use configuration presets:
python config_manager.py preset full_system
python main_controller.py

# Or run specific demos:
python realtime_demo.py                  # Real-time demo
python hpc_demo.py                       # HPC demo
python ai_demo.py                        # AI demo
```

Each script logs progress, saves JSON results, and continues working even if others are deleted!

## System Overview

**45+ Independent Modules** covering:

1. **Core Analysis** (Original System)
   - Pixel reading and correlation
   - Pattern recognition
   - Anomaly detection
   
2. **Advanced Analysis**
   - FFT and mathematical operations
   - Geometric pattern detection
   - Statistical analysis
   - Trend analysis

3. **AI & Computer Vision**
   - PyTorch neural networks
   - OpenCV image processing
   - Machine learning (clustering, classification)
   - Hybrid AI analysis

4. **High Performance Computing**
   - GPU acceleration (10-100x speedup)
   - Multi-core parallel processing
   - Distributed computing simulation
   - Hardware-aware optimization
   - GPU-powered image generation

5. **Real-Time Processing**
   - Live video capture
   - Stream analysis
   - Automatic triggering
   - Real-time dashboards

6. **Advanced Tools**
   - Network API for remote processing
   - Advanced visualization suite
   - Multi-format data export/import
   - System configuration management

Every module:
- ✅ Works independently
- ✅ Communicates via JSON only
- ✅ Logs to terminal
- ✅ Handles missing dependencies
- ✅ Can be deleted without breaking others

The perfect modular architecture for image analysis, from basic pixel processing to advanced AI and HPC!