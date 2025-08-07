# Morphological Features Emulator

A comprehensive real-time morphological analysis system that emulates video feeds and performs advanced morphological feature extraction and analysis.

## Features

### Morphological Analysis Types

- **Morphological Features**: Multi-scale top-hat and black-hat operations
- **Shape Complexity**: Persistence, erosion rate, roughness analysis
- **Skeleton Features**: Morphological skeleton extraction and branch analysis
- **Defect Detection**: Bright/dark defect detection at multiple scales
- **Connected Components**: Component analysis with shape properties

### Real-time Processing

- Live video emulation from BMP images
- Real-time parameter adjustment
- Multiple analysis presets (Default, Fine Detail, Coarse Features, Defect Focus)
- Visual overlay of morphological operations

### GUI Features

- Interactive parameter controls
- Live statistics display
- Analysis type selection checkboxes
- Preset loading for common use cases
- Real-time log output

## Usage

### Basic Usage

```bash
python morphological_features_emulator.py
```

### GUI Controls

#### Configuration

- **Image Path**: Select BMP image for emulation
- **Frame Rate**: Set emulation frame rate (1-120 FPS)
- **Use Emulation**: Toggle between emulation and real camera

#### Analysis Types

Toggle different analysis modes:

- ☑️ **Morphological Features**: Extract multi-scale morphological statistics
- ☑️ **Shape Complexity**: Analyze shape persistence and roughness
- ☑️ **Skeleton Features**: Extract morphological skeleton properties
- ☑️ **Defect Detection**: Detect bright/dark defects
- ☑️ **Connected Components**: Analyze connected component properties

#### Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Kernel Sizes** | 3,5,7,11 | Comma-separated kernel sizes for multi-scale analysis |
| **Min Component Area** | 10-1000 | Minimum area for connected component detection |
| **Defect Threshold** | 1-255 | Threshold for defect pixel classification |
| **Filter Operation** | opening/closing/gradient/tophat/blackhat | Morphological filter for visualization |
| **Filter Kernel Size** | 1-21 | Size of morphological filter kernel |
| **Blur Kernel Size** | 1-31 (odd) | Gaussian blur kernel size |
| **Blur Sigma** | 0.1-10.0 | Gaussian blur standard deviation |

#### Presets

**Default**: Balanced analysis for general use

- Kernel sizes: 3,5,7
- Min area: 50, Defect threshold: 30
- Filter: gradient, Blur: 5, Sigma: 1.0

**Fine Detail**: High-sensitivity analysis

- Kernel sizes: 3,5,7,9
- Min area: 25, Defect threshold: 15
- Filter: tophat, Blur: 3, Sigma: 0.5

**Coarse Features**: Large-scale feature analysis

- Kernel sizes: 7,11,15
- Min area: 100, Defect threshold: 50
- Filter: opening, Blur: 9, Sigma: 2.0

**Defect Focus**: Optimized for defect detection

- Kernel sizes: 5,7,9
- Min area: 30, Defect threshold: 10
- Filter: blackhat, Blur: 5, Sigma: 1.5

## Output Analysis

### Morphological Features

Multi-scale white/black top-hat statistics:

- `morph_wth_{size}_mean`: Mean white top-hat values
- `morph_wth_{size}_max`: Maximum white top-hat values
- `morph_wth_{size}_sum`: Sum of white top-hat values
- `morph_bth_{size}_*`: Corresponding black top-hat values
- `morph_binary_area_ratio`: Binary area ratio
- `morph_gradient_sum`: Morphological gradient sum

### Shape Complexity Features

- `shape_persistence`: Fraction surviving after erosion
- `shape_erosion_rate`: Rate of shape degradation
- `shape_roughness`: Surface roughness measure
- `shape_holes`: Internal hole measure

### Skeleton Features

- `skeleton_pixels`: Number of skeleton pixels
- `skeleton_ratio`: Skeleton to total pixel ratio
- `skeleton_branches`: Number of branch points

### Connected Components

For each component:

- `bbox`: Bounding box (x, y, width, height)
- `area`: Component area in pixels
- `centroid`: Center coordinates
- `perimeter`: Perimeter length
- `circularity`: Circularity measure (4π×area/perimeter²)
- `aspect_ratio`: Width/height ratio
- `extent`: Area/bounding box ratio

### Defect Maps

Multi-scale defect detection:

- `bright_defects_{size}`: Bright anomaly map
- `dark_defects_{size}`: Dark anomaly map
- `combined_defects_{size}`: Combined defect map

## Test Images

The system includes test images optimized for morphological analysis:

- **morphological_test.bmp**: Comprehensive test with various shapes, defects, textures
- **morphological_simple_test.bmp**: Basic shapes for quick testing
- **morphological_test_color.bmp**: Color-mapped version for visualization

## Integration

### Module Integration

```python
from morphological_features_module import MorphologicalDetector, MorphologicalProcessor

# Create detector
detector = MorphologicalDetector(
    analysis_types=['features', 'complexity', 'defects'],
    kernel_sizes=[3, 5, 7],
    min_component_area=50
)

# Analyze frame
results, processed_frame = detector.analyze_frame(frame)

# Use processor for video streams
processor = MorphologicalProcessor(detector)
processed_frame = processor.process_frame(frame)
```

### Real-time Applications

- **Quality Control**: Defect detection in manufacturing
- **Medical Imaging**: Structural analysis of medical images
- **Material Analysis**: Surface texture and defect characterization
- **Biological Imaging**: Cell and tissue morphology analysis

## Performance

- **Analysis Time**: ~0.1-0.2 seconds per frame (640x480)
- **Memory Usage**: Moderate (depends on kernel sizes)
- **CPU Usage**: Moderate to high during analysis
- **Real-time Capable**: Yes, at 30 FPS for typical images

## Troubleshooting

### Common Issues

**Slow Performance**:

- Reduce kernel sizes
- Disable unused analysis types
- Lower image resolution

**No Defects Detected**:

- Lower defect threshold
- Check kernel sizes
- Verify image has actual defects

**Too Many False Positives**:

- Increase defect threshold
- Increase blur parameters
- Use coarser kernel sizes

**Component Detection Issues**:

- Adjust min component area
- Check binary threshold
- Verify image preprocessing

## Files

### Core Files

- `morphological_features_emulator.py`: Main GUI emulator
- `morphological_features_module.py`: Core detector implementation
- `dev/morphological_features.py`: Base morphological functions

### Test Files (in non-essential/)

- `test_morphological_features.py`: Comprehensive test suite
- `create_morphological_test_image.py`: Test image generator

### Output Files

- `pictures/morphological_*_result.bmp`: Analysis results
- Various preset result images for comparison

## Dependencies

- OpenCV (cv2): Image processing and morphological operations
- NumPy: Numerical operations
- tkinter: GUI framework
- PIL/Pillow: Image handling for GUI
- Python threading: Background processing

## Related Systems

- **Blob Detection Emulator**: Circular object detection
- **Hough Lines Emulator**: Linear feature detection
- **SSIM Detection Emulator**: Structural similarity analysis
- **Statistical Features Emulator**: Statistical image analysis

The morphological features emulator provides the most comprehensive shape and structure analysis capabilities in the BMP video analysis suite.
