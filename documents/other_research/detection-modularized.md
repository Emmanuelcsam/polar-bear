# Modular OmniFiber Analyzer

This is a modular version of the original `detection.py` file, split into focused, reusable components that can run individually or together.

## Overview

The original monolithic `detection.py` file has been decomposed into multiple specialized modules, each handling a specific aspect of the fiber optic anomaly detection system. This modular approach provides:

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Reusability**: Individual modules can be imported and used independently
- **Maintainability**: Easier to debug, test, and modify specific functionality
- **Testability**: Each module can be tested in isolation
- **Flexibility**: Modules can be combined in different ways for different use cases

## Module Structure

### Core Modules

#### `config.py`
**Purpose**: Configuration management and data serialization
- `OmniConfig`: Dataclass containing all analysis parameters
- `NumpyEncoder`: Custom JSON encoder for numpy data types

**Usage**:
```python
from config import OmniConfig

config = OmniConfig(
    min_defect_size=10,
    max_defect_size=5000,
    confidence_threshold=0.3
)
```

#### `utils.py`
**Purpose**: Common utility functions
- `get_timestamp()`: Generate formatted timestamps
- `load_image()`: Load images from various formats (including JSON)
- `load_from_json()`: Specialized JSON image loader
- `sanitize_feature_value()`: Ensure feature values are finite and valid

**Usage**:
```python
from utils import load_image, get_timestamp

image = load_image("path/to/image.jpg")
timestamp = get_timestamp()
```

#### `statistical_functions.py`
**Purpose**: Statistical computation functions
- `compute_skewness()`: Calculate distribution skewness
- `compute_kurtosis()`: Calculate distribution kurtosis
- `compute_entropy()`: Calculate Shannon entropy
- `compute_correlation()`: Calculate Pearson correlation
- `compute_spearman_correlation()`: Calculate Spearman rank correlation
- `compute_ks_statistic()`: Calculate Kolmogorov-Smirnov statistic
- `compute_wasserstein_distance()`: Calculate 1D Wasserstein distance

**Usage**:
```python
from statistical_functions import compute_skewness, compute_entropy
import numpy as np

data = np.random.normal(0, 1, 1000)
skewness = compute_skewness(data)
entropy = compute_entropy(data)
```

### Feature Extraction

#### `feature_extraction.py`
**Purpose**: Comprehensive feature extraction from images
- `extract_ultra_comprehensive_features()`: Main feature extraction orchestrator
- Individual extractors for different feature types:
  - `extract_statistical_features()`: Basic statistical features
  - `extract_matrix_norms()`: Matrix norm features
  - `extract_lbp_features()`: Local Binary Pattern features
  - `extract_glcm_features()`: Gray-Level Co-occurrence Matrix features
  - `extract_fourier_features()`: Fourier Transform features
  - `extract_multiscale_features()`: Multi-scale analysis features
  - `extract_morphological_features()`: Morphological features
  - `extract_shape_features()`: Shape descriptors
  - `extract_svd_features()`: Singular Value Decomposition features
  - `extract_entropy_features()`: Entropy measures
  - `extract_gradient_features()`: Gradient-based features
  - `extract_topological_proxy_features()`: Topological features

**Usage**:
```python
from feature_extraction import extract_ultra_comprehensive_features
from utils import load_image

image = load_image("test_image.jpg")
features, feature_names = extract_ultra_comprehensive_features(image)
print(f"Extracted {len(features)} features")
```

### Comparison and Analysis

#### `comparison.py`
**Purpose**: Feature and image comparison functions
- `compute_exhaustive_comparison()`: Comprehensive feature comparison using multiple metrics
- `compute_image_structural_comparison()`: Structural similarity (SSIM) computation

**Usage**:
```python
from comparison import compute_exhaustive_comparison

# Compare two feature sets
comparison = compute_exhaustive_comparison(features1, features2)
print(f"Euclidean distance: {comparison['euclidean_distance']}")
print(f"Cosine distance: {comparison['cosine_distance']}")
```

#### `defect_detection.py`
**Purpose**: Specific defect detection algorithms
- `detect_specific_defects()`: Detect scratches, digs, blobs, and edge irregularities
- `compute_local_anomaly_map()`: Compute pixel-wise anomaly scores
- `find_anomaly_regions()`: Find distinct anomaly regions from anomaly map
- `create_defect_mask()`: Create binary mask of all detected defects

**Usage**:
```python
from defect_detection import detect_specific_defects
import cv2

gray_image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
defects = detect_specific_defects(gray_image)
print(f"Found {len(defects['scratches'])} scratches")
print(f"Found {len(defects['digs'])} digs")
```

### Reference Model Management

#### `reference_model.py`
**Purpose**: Reference model building, loading, and saving
- `build_comprehensive_reference_model()`: Build reference model from directory of images
- `load_knowledge_base()`: Load saved reference model from JSON
- `save_knowledge_base()`: Save reference model to JSON
- `build_minimal_reference()`: Build minimal reference from single image
- `compute_robust_statistics()`: Compute robust statistical parameters
- `get_default_thresholds()`: Get default anomaly thresholds

**Usage**:
```python
from reference_model import build_comprehensive_reference_model, load_knowledge_base
from config import OmniConfig

config = OmniConfig()
# Build reference model from directory
reference_model = build_comprehensive_reference_model("reference_images/", config)

# Or load existing model
reference_model = load_knowledge_base("fiber_anomaly_kb.json")
```

### Main Analysis Engine

#### `anomaly_detection.py`
**Purpose**: Main anomaly detection orchestration
- `detect_anomalies_comprehensive()`: Perform comprehensive anomaly detection
- `analyze_end_face()`: Main analysis method compatible with pipeline expectations
- `convert_to_pipeline_format()`: Convert internal results to pipeline format
- `confidence_to_severity()`: Convert confidence scores to severity levels

**Usage**:
```python
from anomaly_detection import detect_anomalies_comprehensive
from config import OmniConfig

config = OmniConfig()
results = detect_anomalies_comprehensive("test_image.jpg", reference_model, config)
print(f"Anomalous: {results['verdict']['is_anomalous']}")
print(f"Confidence: {results['verdict']['confidence']:.3f}")
```

### Output and Visualization

#### `visualization.py`
**Purpose**: Result visualization and image generation
- `visualize_comprehensive_results()`: Create comprehensive visualization with multiple panels
- `save_simple_anomaly_image()`: Save simple image with anomalies highlighted

**Usage**:
```python
from visualization import visualize_comprehensive_results

# Create comprehensive visualization
visualize_comprehensive_results(results, "output_visualization.png")
```

#### `report_generation.py`
**Purpose**: Text report generation
- `generate_detailed_report()`: Generate detailed text report of analysis

**Usage**:
```python
from report_generation import generate_detailed_report

# Generate detailed text report
generate_detailed_report(results, "detailed_report.txt")
```

#### `defect_mask.py`
**Purpose**: Defect mask creation
- `create_defect_mask()`: Create binary mask showing all detected defects

**Usage**:
```python
from defect_mask import create_defect_mask
import numpy as np

# Create defect mask
mask = create_defect_mask(results)
np.save("defect_mask.npy", mask)
```

### Main Orchestrator

#### `main_analyzer.py`
**Purpose**: Main orchestrator that ties all modules together
- `OmniFiberAnalyzer`: Main class that coordinates all functionality
- `main()`: Interactive testing interface

**Usage**:
```python
from main_analyzer import OmniFiberAnalyzer
from config import OmniConfig

# Create analyzer
config = OmniConfig()
analyzer = OmniFiberAnalyzer(config)

# Analyze image
result = analyzer.analyze_end_face("test_image.jpg", "output_dir")
```

## Running Individual Modules

Each module can be run independently for testing or specific functionality:

### Test Feature Extraction
```bash
python -c "
from feature_extraction import extract_ultra_comprehensive_features
from utils import load_image
image = load_image('test_image.jpg')
features, names = extract_ultra_comprehensive_features(image)
print(f'Extracted {len(features)} features')
"
```

### Test Statistical Functions
```bash
python -c "
from statistical_functions import compute_skewness, compute_entropy
import numpy as np
data = np.random.normal(0, 1, 1000)
print(f'Skewness: {compute_skewness(data):.4f}')
print(f'Entropy: {compute_entropy(data):.4f}')
"
```

### Test Defect Detection
```bash
python -c "
from defect_detection import detect_specific_defects
import cv2
image = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
defects = detect_specific_defects(image)
print(f'Found {len(defects[\"scratches\"])} scratches')
"
```

## Running the Complete System

### Interactive Mode
```bash
python main_analyzer.py
```

### Programmatic Usage
```python
from main_analyzer import OmniFiberAnalyzer
from config import OmniConfig

# Initialize
config = OmniConfig()
analyzer = OmniFiberAnalyzer(config)

# Build reference model (if needed)
analyzer.build_reference_model("reference_images/")

# Analyze image
result = analyzer.analyze_end_face("test_image.jpg", "output_dir")
```

## Module Dependencies

```
main_analyzer.py
├── config.py
├── utils.py
├── statistical_functions.py
├── feature_extraction.py
│   ├── utils.py
│   └── statistical_functions.py
├── comparison.py
│   └── statistical_functions.py
├── defect_detection.py
├── reference_model.py
│   ├── utils.py
│   ├── feature_extraction.py
│   └── comparison.py
├── anomaly_detection.py
│   ├── utils.py
│   ├── feature_extraction.py
│   ├── comparison.py
│   ├── defect_detection.py
│   └── reference_model.py
├── visualization.py
├── report_generation.py
│   └── utils.py
└── defect_mask.py
```

## Benefits of Modular Design

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual modules can be unit tested
3. **Reusability**: Modules can be imported and used independently
4. **Scalability**: New modules can be added without affecting existing ones
5. **Debugging**: Issues can be isolated to specific modules
6. **Documentation**: Each module is self-documenting with clear purpose

## Migration from Original Code

The modular version provides the exact same functionality as the original `detection.py` but with better organization:

- **Original**: Single 2500+ line file with everything mixed together
- **Modular**: 12 focused modules, each handling specific functionality
- **Compatibility**: Same API and output format as original
- **Performance**: Identical performance characteristics
- **Features**: All original features preserved

## Testing Individual Components

Each module includes built-in testing capabilities through the main analyzer's interactive interface, allowing you to test specific functionality in isolation. 