# Real-Time Pylon Camera Integration with Defect Detection

## Overview

This document provides comprehensive code solutions to integrate your Basler Pylon camera real-time frame grabber with the defect detection system, enabling continuous anomaly detection using a specific reference image.

## Architecture Design

### Producer-Consumer Pattern
- **Producer**: PylonFrameGrabber thread captures frames continuously
- **Consumer**: Detection thread processes frames against reference image
- **Queue System**: Thread-safe frame buffer for decoupling capture and processing

### Key Components
1. **Enhanced PylonFrameGrabber**: Modified for real-time streaming
2. **RealTimeDetector**: Wrapper for OmniFiberAnalyzer with specific reference
3. **FrameProcessor**: Main integration controller
4. **Results Handler**: Manages detection results and visualization

## Benefits of This Approach

### Performance Advantages
- **Decoupled Processing**: Frame capture runs independently of detection
- **Latest Frame Strategy**: Always processes most recent image data
- **Threaded Architecture**: Utilizes multiple CPU cores efficiently
- **Memory Optimization**: Controlled buffer sizes prevent memory overflow

### Flexibility Features
- **Specific Reference Image**: Uses your chosen reference image instead of statistical model
- **Configurable Processing Rate**: Adjustable detection frequency
- **Real-time Results**: Immediate feedback on defect detection
- **Graceful Shutdown**: Clean thread termination and resource cleanup

## Integration Solutions

The following code implementations provide:

1. **Enhanced Frame Grabber**: Modified PylonFrameGrabber with better real-time performance
2. **Real-Time Detection Wrapper**: Adapter for OmniFiberAnalyzer with specific reference support
3. **Main Integration Controller**: Complete system orchestration
4. **Example Usage Scripts**: Ready-to-use implementation examples

## Technical Requirements

### Dependencies
- pypylon (Basler Pylon SDK)
- opencv-python
- numpy
- threading
- queue
- time
- pathlib

### Hardware Considerations
- Sufficient CPU cores for parallel processing
- Adequate RAM for frame buffering
- Fast storage for reference image loading

## Performance Optimization Tips

1. **Frame Rate Management**: Balance capture speed with processing capability
2. **Buffer Size Tuning**: Optimize queue sizes for your specific use case
3. **Reference Image Preparation**: Pre-load and optimize reference image format
4. **Memory Management**: Regular cleanup of processed frames
5. **Threading Priorities**: Set appropriate thread priorities for real-time performance

## Usage Workflow

1. Load your specific reference image
2. Initialize the enhanced frame grabber
3. Start the real-time detection system
4. Monitor results in real-time
5. Handle defect alerts and visualizations
6. Graceful shutdown when complete

The complete implementation follows in the accompanying code files.