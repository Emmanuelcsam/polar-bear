# Modular Fiber Optic Analysis Functions

This directory contains modularized, standalone functions extracted from a legacy fiber optic analysis system. Each module is designed to work independently and can be used for neural network training, image processing pipelines, or integrated into new applications.

## 🚀 Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Test All Functions**:
```bash
python test_all_functions.py
```

3. **Test Individual Functions**:
```bash
python test_all_functions.py --specific-test adaptive_intensity
```

## 📁 Module Overview

### 🔍 **Segmentation Modules**

#### `adaptive_intensity_segmenter.py`
- **Purpose**: Segments fiber images using adaptive intensity histogram analysis
- **Key Features**: 
  - CLAHE enhancement for low-contrast images
  - Automatic peak detection in intensity histograms
  - Core and cladding identification
- **Usage**:
```bash
python adaptive_intensity_segmenter.py path/to/image.jpg --output-dir results/
```
- **Best For**: Images with distinct intensity variations, multi-modal histograms

#### `bright_core_extractor.py`
- **Purpose**: Detects bright fiber cores using local contrast validation
- **Key Features**:
  - Hough circle detection with validation
  - Local contrast analysis
  - Robust against noise and artifacts
- **Usage**:
```bash
python bright_core_extractor.py path/to/image.jpg --debug
```
- **Best For**: High-quality images with clearly visible bright cores

#### `hough_fiber_separator.py`
- **Purpose**: Separates fiber zones using Hough transform circle detection
- **Key Features**:
  - Dual-stage detection (cladding then core)
  - Adaptive parameter tuning
  - Comprehensive preprocessing pipeline
- **Usage**:
```bash
python hough_fiber_separator.py path/to/image.jpg --output-dir hough_results/
```
- **Best For**: Circular fiber cross-sections with clear boundaries

#### `gradient_fiber_segmenter.py`
- **Purpose**: Multi-method fiber segmentation using gradient analysis
- **Key Features**:
  - Multiple center detection methods
  - Gradient-based radius estimation
  - Robust fallback mechanisms
- **Usage**:
```bash
python gradient_fiber_segmenter.py path/to/image.jpg --output-dir gradient_results/
```
- **Best For**: Complex images where Hough circles fail, non-circular fibers

### 🔎 **Detection Modules**

#### `traditional_defect_detector.py`
- **Purpose**: Computer vision-based defect detection
- **Key Features**:
  - Multi-type defect detection (scratches, pits, contamination, cracks)
  - Zone-aware analysis
  - Morphological and edge-based methods
- **Usage**:
```bash
python traditional_defect_detector.py path/to/image.jpg --center 256 256 --core-radius 25 --cladding-radius 150
```
- **Detects**: Scratches, pits, contamination, cracks
- **Best For**: Quality control applications, automated inspection

### 🎨 **Enhancement Modules**

#### `image_enhancer.py`
- **Purpose**: Advanced image preprocessing and enhancement
- **Key Features**:
  - Multiple enhancement algorithms (CLAHE, bilateral filtering, unsharp masking)
  - Auto-enhancement based on image characteristics
  - Comprehensive preprocessing pipeline
- **Usage**:
```bash
python image_enhancer.py path/to/image.jpg --auto-enhance --save-intermediate
```
- **Best For**: Preprocessing for other analysis modules, improving image quality

### 🧪 **Testing Module**

#### `test_all_functions.py`
- **Purpose**: Comprehensive testing framework for all modules
- **Key Features**:
  - Synthetic test image generation
  - Automated testing of all modules
  - Performance benchmarking
- **Usage**:
```bash
python test_all_functions.py --test-image path/to/image.jpg --output-dir test_results/
```

## 🔧 Integration Examples

### Neural Network Data Preparation
```python
from image_enhancer import ImageEnhancer
from adaptive_intensity_segmenter import AdaptiveIntensitySegmenter

# Prepare training data
enhancer = ImageEnhancer()
segmenter = AdaptiveIntensitySegmenter()

for image_path in training_images:
    # Enhance image
    enhanced = enhancer.auto_enhance(cv2.imread(image_path))
    
    # Extract features
    result = segmenter.segment_image(image_path)
    
    # Use for training...
```

### Pipeline Integration
```python
from hough_fiber_separator import HoughFiberSeparator
from traditional_defect_detector import TraditionalDefectDetector

# Complete analysis pipeline
separator = HoughFiberSeparator()
detector = TraditionalDefectDetector()

# Segment fiber
seg_result = separator.separate_fiber(image_path)

if seg_result['success']:
    # Create zone masks
    zone_masks = detector.create_zone_masks(
        image.shape, 
        seg_result['center'],
        seg_result['core_radius'],
        seg_result['cladding_radius']
    )
    
    # Detect defects
    defects = detector.detect_all_defects(image, zone_masks)
```

## 📊 Performance Characteristics

| Module | Speed | Accuracy | Robustness | Use Case |
|--------|-------|----------|------------|----------|
| Adaptive Intensity | Fast | Good | Medium | Multi-modal intensity images |
| Bright Core | Fast | High | Medium | High-quality bright cores |
| Hough Separation | Medium | High | Good | Circular fiber cross-sections |
| Gradient Segmentation | Slow | Good | High | Complex/non-circular fibers |
| Defect Detection | Medium | Good | Good | Quality control |
| Image Enhancement | Fast | N/A | High | Preprocessing |

## 🔬 Technical Details

### Dependencies
- **Core**: OpenCV, NumPy, Pathlib
- **Optional**: SciPy, Scikit-learn, Scikit-image
- **ML Optional**: PyTorch, TensorFlow
- **Visualization**: Matplotlib, Pillow

### Input Requirements
- **Image Formats**: JPG, PNG, BMP, TIFF
- **Color Spaces**: RGB, BGR, Grayscale (auto-converted)
- **Typical Resolution**: 256x256 to 2048x2048 pixels
- **Fiber Types**: Single-mode, multi-mode, specialty fibers

### Output Formats
- **JSON**: Structured results with coordinates, radii, confidence scores
- **Images**: Visualizations, masks, intermediate processing steps
- **Data**: NumPy arrays for programmatic access

## 🎯 Applications

### Research & Development
- **Feature Extraction**: For neural network training datasets
- **Algorithm Benchmarking**: Comparing different segmentation approaches
- **Data Augmentation**: Generating varied training samples

### Industrial Applications
- **Quality Control**: Automated fiber inspection
- **Manufacturing**: Real-time process monitoring
- **Telecommunications**: Fiber optic component testing

### Educational Use
- **Computer Vision**: Teaching image processing concepts
- **Signal Processing**: Understanding fiber optic principles
- **Machine Learning**: Training on specialized domain data

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Segmentation Failures**:
   - Try different modules (Hough → Gradient → Adaptive)
   - Adjust preprocessing parameters
   - Check image quality and resolution

3. **Poor Detection Results**:
   - Verify correct center/radius parameters
   - Enhance image quality first
   - Adjust detection thresholds

4. **Performance Issues**:
   - Use appropriate module for your use case
   - Consider image downsampling for speed
   - Enable only necessary processing steps

### Parameter Tuning

Each module has tunable parameters:

```python
# Example: Adjust sensitivity
segmenter = AdaptiveIntensitySegmenter(
    peak_prominence=300,  # Lower = more sensitive
    enhancement_enabled=True
)

extractor = BrightCoreExtractor(
    contrast_factor=1.1,  # Lower = more permissive
    hough_param2=15      # Lower = more sensitive
)
```

## 📈 Future Enhancements

- [ ] GPU acceleration for larger images
- [ ] Deep learning integration
- [ ] Real-time video processing
- [ ] Advanced defect classification
- [ ] 3D fiber analysis capabilities

## 📝 License

These modules are extracted from a legacy system and provided for educational and research purposes. Ensure compliance with your organization's policies before commercial use.

## 🤝 Contributing

Each module is designed to be standalone and modifiable. Key extension points:

1. **Add new detection methods** to existing modules
2. **Create new modules** following the established patterns
3. **Enhance testing framework** with additional test cases
4. **Optimize performance** for specific hardware/use cases

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run the test suite to verify installation
3. Review module documentation and examples
4. Consider the performance characteristics table for module selection
