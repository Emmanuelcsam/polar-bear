# Enhanced Fiber Optic Defect Detection System

## Overview

This system performs automated defect detection on fiber optic endface images using advanced image processing, machine learning, and multi-method consensus algorithms.

## System Architecture

```
Input: test_image/img(303).jpg
         ↓
    [app.py - Main Application]
         ↓
    [process.py - Image Processing]
    - Generates 49 variations using different filters
    - ML-powered variation selection
    - In-memory caching for performance
         ↓
    [separation.py - Zone Segmentation]
    - Multi-method consensus (11 different algorithms)
    - Separates: Core, Cladding, Ferrule
    - Uses reference images from separated/ folder
         ↓
    [detection.py - Defect Detection]
    - ML + Traditional detection methods
    - Identifies: scratches, pits, contamination
    - DBSCAN clustering for defect merging
         ↓
    [Data Acquisition & Visualization]
    - Defect overlay with bounding boxes
    - Zone measurements (diameters in pixels)
    - Concentricity calculations
    - Pass/Fail determination
         ↓
Output: Results + Visualizations
```

## Key Features

### 1. **Enhanced Visualization**
- **Defect Overlay**: Shows all detected defects with colored bounding boxes
- **Zone Measurements**: Displays diameter of each zone in pixels
- **Concentricity**: Calculates offset between core and cladding centers
- **Heatmaps**: Optional defect density visualization

### 2. **Zone Segmentation**
The system separates fiber optic images into three zones:
- **Core**: The bright inner circle (fiber core)
- **Cladding**: The darker ring around the core
- **Ferrule**: The outer region

### 3. **Defect Types Detected**
- **Scratches**: Linear defects on the surface
- **Pits**: Small dark spots or holes
- **Contamination**: Bright spots or particles
- **Fiber Damage**: Structural damage to the fiber
- **Statistical Anomalies**: Unusual patterns detected by ML

### 4. **Quality Metrics**
- Total defect count
- Defect density per zone
- Average severity scores
- Zone-specific defect types
- Concentricity offset (pixels and percentage)
- Zone diameters

## Usage

### Processing Test Image

1. Run the application:
   ```bash
   python3 app.py
   ```

2. Select option 4 (Run tests)

3. Select option 2 (Process test image)

The system will:
- Load `test_image/img(303).jpg`
- Process through all stages
- Display detailed results including:
  - Pass/Fail status
  - Defect count and types
  - Zone measurements
  - Concentricity metrics
- Optionally show visual overlays

### Batch Processing

1. Run the application
2. Select option 1 (Batch processing)
3. Specify input directory
4. System processes all images and saves results

### Real-time Processing

1. Run the application
2. Select option 3 (Real-time camera)
3. System processes live camera feed

## Configuration

The system uses `config_manager.py` for configuration:
- Processing options (ML models, variations, etc.)
- Detection thresholds
- Visualization settings
- Input/Output paths

Set `FIBER_NO_INTERACTIVE=1` to skip interactive setup.

## Testing

Comprehensive test suite in `tests/` directory:
- `test_app.py`: Tests main application logic
- `test_process.py`: Tests image processing
- `test_separation.py`: Tests zone segmentation  
- `test_detection.py`: Tests defect detection
- `test_integration.py`: Tests full pipeline

Run all tests:
```bash
python3 tests/run_all_tests.py
```

## Performance Optimizations

- **Parallel Processing**: Uses multiple CPU cores
- **In-memory Caching**: Reduces redundant calculations
- **Batch Operations**: Processes multiple images efficiently
- **ML Model Optimization**: Supports both PyTorch and TensorFlow

## Output Example

When processing an image, the system provides:

```
Result: PASS
Defects found: 3

=== Detailed Metrics ===
Total area: 1048576 pixels
Defect density: 0.03
Average severity: 0.45

=== Zone Measurements ===
core:
  - Diameter: 124 pixels
  - Center: (512, 432)
  - Area: 12076 pixels
cladding:
  - Diameter: 248 pixels
  - Center: (513, 431)
  - Area: 36842 pixels

=== Concentricity ===
Offset: 1.41 pixels
Percentage: 1.14%

=== Zone Defects ===
core: 0 defects
cladding: 2 defects
  Types: scratch, pit
ferrule: 1 defect
  Types: contamination
```