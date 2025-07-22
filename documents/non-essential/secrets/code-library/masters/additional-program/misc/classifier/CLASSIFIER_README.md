# Advanced Image Classifier - User Guide

## Overview

The Advanced Image Classifier (`image-classifier-advanced.py`) is a fully functional, machine learning-based image classification and renaming system. It analyzes images, learns from reference data, and automatically classifies and renames images based on their visual characteristics.

## Key Features

### Core Functionality
- **Multiple Feature Extraction Methods**: Extracts color histograms, texture features (Gabor filters, LBP), edge features (Canny, Sobel), shape features (contours, moments), and statistical features
- **Intelligent Similarity Search**: Uses cosine similarity with adaptive thresholds
- **Persistent Knowledge Bank**: Learns from every classification and improves over time
- **Hierarchical Classification**: Supports complex folder structures and multi-level classifications
- **Confidence Scoring**: Provides confidence levels for each classification
- **Feature Caching**: Improves performance by caching extracted features

### Modes of Operation
1. **Automated Mode**: Batch processes entire folders automatically
2. **Manual Mode**: Interactive classification with suggestions and user feedback

### Advanced Features
- Custom keyword support
- User feedback incorporation
- Classification history tracking
- Adaptive threshold adjustment
- Progress tracking with tqdm
- Comprehensive error logging
- Defect pattern recognition
- Automatic folder structure creation

## Installation

### Option 1: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install Pillow numpy scikit-learn opencv-python imagehash tqdm
```

### Option 2: Using the provided script
```bash
chmod +x run_classifier.sh
./run_classifier.sh
```

## Folder Structure

```
training/
├── image-classifier-advanced.py   # Main classifier script
├── reference/                     # Reference images (training data)
│   ├── fc/                       # Connector type folders
│   │   ├── 50/                   # Core diameter folders
│   │   │   ├── core/             # Region folders
│   │   │   │   ├── clean/        # Condition folders
│   │   │   │   └── dirty/        # Contains actual images
│   │   │   └── cladding/
│   │   └── 91/
│   └── sma/
└── dataset/                       # Images to be classified
```

## Usage

### Basic Usage

1. **Prepare Reference Images**: Organize your reference images in the `reference/` folder following the hierarchical structure that represents your classification scheme.

2. **Add Images to Classify**: Place the images you want to classify in the `dataset/` folder.

3. **Run the Classifier**:
   ```bash
   python image-classifier-advanced.py
   ```

4. **Choose Mode**:
   - Enter `1` for Automated Mode
   - Enter `2` for Manual Mode

### Automated Mode

The classifier will:
1. Analyze all reference images and build a knowledge bank
2. Process each image in the dataset folder
3. Extract visual features from each image
4. Find similar images in the knowledge bank
5. Classify based on weighted voting from similar images
6. Rename files according to classification
7. Save high-confidence classifications back to reference folder

### Manual Mode

Interactive mode where you can:
1. Review each image individually
2. See automatic classification suggestions
3. Accept, modify, or skip classifications
4. Add custom keywords
5. Provide feedback for continuous learning

### Classification Components

The classifier recognizes these components:
- **Core Diameter**: Numeric values (e.g., 50, 91)
- **Connector Type**: fc, sma, sc, lc, st
- **Region**: core, cladding, ferrule
- **Condition**: clean, dirty
- **Defect Type**: scratched, oil, blob, dig, anomaly
- **Additional Characteristics**: Any custom keywords

### Configuration

The classifier creates a `classifier_config.json` file with these settings:
```json
{
    "similarity_threshold": 0.70,
    "min_confidence": 0.50,
    "auto_create_folders": true,
    "save_learned_references": true,
    "use_adaptive_threshold": true,
    "feature_extraction_method": "combined",
    "max_features_per_method": 256
}
```

## Examples

### Example Classifications
- `50-fc-core-clean.jpg` - 50μm FC connector, core region, clean
- `91-sma-cladding-scratched.png` - 91μm SMA connector, cladding, with scratch
- `fc-ferrule-oil-blob.jpg` - FC connector, ferrule region, oil and blob defects

### Command Line Examples
```bash
# Run with default settings
python image-classifier-advanced.py

# Run verification script
python verify_classifier.py

# Run tests
python test_classifier.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **No Images Found**: Check that image files have supported extensions (.jpg, .jpeg, .png, .bmp, .gif, .tiff, .webp)
3. **Low Confidence**: Add more reference images or adjust similarity threshold
4. **Feature Dimension Mismatch**: Clear knowledge_bank.pkl and re-run

### Performance Tips

1. **First Run**: The first analysis of reference folder may take time
2. **Feature Caching**: Subsequent runs will be faster due to caching
3. **Large Datasets**: Use automated mode for batch processing
4. **Memory Usage**: Feature cache is limited to 1000 images

## Advanced Usage

### Adding Custom Keywords
In manual mode, choose option 5 to add custom keywords that will be recognized in future classifications.

### Adjusting Thresholds
Edit `classifier_config.json` to adjust:
- `similarity_threshold`: Lower = more matches, higher = stricter matching
- `min_confidence`: Minimum confidence required for classification

### Building Custom Folder Structures
The classifier will automatically create folders based on detected components. Customize this by modifying the `classification_components` order in the config.

## Files Created

- `knowledge_bank.pkl`: Persistent storage of learned features and classifications
- `classifier_config.json`: Configuration settings
- Log output with timestamps for debugging

## Notes

- The classifier improves over time as it processes more images
- High-confidence classifications are automatically added to the reference set
- Manual mode feedback helps improve future classifications
- All features are normalized and scaled for consistent comparison