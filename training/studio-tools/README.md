# Automated Image Processing Studio

An advanced automated image processing system that uses machine learning to match input images to target images by intelligently applying a sequence of OpenCV operations.

## 📁 Directory Structure

```
studio/
├── automated_processing_studio_v2.py  # Main application (fixed version)
├── image_processor_gui.py            # GUI interface for the studio
├── scripts/                          # Legacy/original processing scripts
├── opencv_scripts/                   # Organized OpenCV processing scripts
├── guides/                           # Documentation and guides
├── tests/                            # Unit tests and test environments
├── setup/                            # Setup and dependency files
├── demos/                            # Demo scripts and output examples
├── utilities/                        # Utility scripts for management
└── deprecated/                       # Old versions and deprecated files
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   python automated_processing_studio_v2.py
   ```
   The script will automatically check and install required dependencies on first run.

2. **Run the Studio**
   ```bash
   python automated_processing_studio_v2.py
   ```

3. **Use the GUI** (Optional)
   ```bash
   python image_processor_gui.py
   ```

## 📂 Core Components

### Main Files
- `automated_processing_studio_v2.py` - Enhanced studio with intelligent matching
- `image_processor_gui.py` - Graphical user interface

### Script Directories
- `scripts/` - Contains 400+ legacy image processing scripts
- `opencv_scripts/` - Organized scripts by category:
  - `edge_detection/` - Edge detection algorithms
  - `filtering/` - Image filtering operations
  - `morphology/` - Morphological operations
  - `thresholding/` - Various thresholding methods
  - `transformations/` - Geometric transformations
  - `effects/` - Artistic and special effects
  - `features/` - Feature detection algorithms
  - `histogram/` - Histogram operations
  - `noise/` - Noise addition/removal

### Support Directories
- `guides/` - Documentation for writing custom scripts
- `tests/` - Comprehensive test suite
- `setup/` - Installation and setup scripts
- `demos/` - Example usage and demonstrations
- `utilities/` - Helper scripts for script management

## 🎯 Key Features

1. **Intelligent Image Matching**
   - Automatically finds the best sequence of operations to match a target image
   - Uses machine learning to improve over time
   - Supports up to 200 iterations for fine-tuning

2. **Comprehensive Script Library**
   - 440+ pre-built image processing scripts
   - Organized by category for easy navigation
   - Extensible with custom scripts

3. **Advanced Analysis**
   - Similarity scoring using multiple metrics (MSE, SSIM, histogram, edges)
   - Anomaly detection and visualization
   - Detailed processing reports with visualizations

4. **Learning System**
   - Records successful processing sequences
   - Improves recommendations based on past results
   - Persistent knowledge storage

## 📋 Usage Examples

### Command Line
```python
# Basic usage - will prompt for input
python automated_processing_studio_v2.py

# The script will ask for:
# - Input image path
# - Target image path
# - Maximum iterations (default: 200)
# - Similarity threshold (default: 0.05)
# - Verbose output (y/n)
```

### Python API
```python
from automated_processing_studio_v2 import EnhancedProcessingStudio
import cv2

# Create studio instance
studio = EnhancedProcessingStudio()

# Load images
input_img = cv2.imread("input.jpg")
target_img = cv2.imread("target.jpg")

# Process to match target
results = studio.process_to_match_target(
    input_img,
    target_img,
    max_iterations=100,
    similarity_threshold=0.05,
    verbose=True
)

# Access results
print(f"Success: {results['success']}")
print(f"Final similarity: {results['final_similarity']}")
print(f"Applied operations: {results['pipeline']}")
cv2.imwrite("output.jpg", results['final_image'])
```

## 🛠️ Creating Custom Scripts

Scripts should follow this template:

```python
import numpy as np
import cv2

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Process the input image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Processed image as numpy array
    """
    # Your processing code here
    processed = cv2.GaussianBlur(image, (5, 5), 1.0)
    return processed
```

See `guides/script_writing_guide.md` for detailed instructions.

## 📊 Output and Reports

The studio generates comprehensive reports including:
- Input, target, and output image comparisons
- Processing pipeline visualization
- Anomaly maps and difference visualizations
- Histogram comparisons
- Detailed statistics and metrics

Reports are saved in timestamped directories within the cache folder.

## 🧪 Testing

Run the test suite:
```bash
cd tests
python test_automated_studio_v2.py
```

## 📝 License

This project is part of the polar-bear repository. See the main repository for license information.

## 🤝 Contributing

1. Add new scripts to the appropriate category in `opencv_scripts/`
2. Follow the script template structure
3. Test your scripts with the validation tool
4. Submit a pull request

## 📞 Support

For issues or questions:
- Check the guides in the `guides/` directory
- Review test cases for usage examples
- Open an issue in the main repository