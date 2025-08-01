
# Fiber Optic Endface Defect Detection Neural Network

## Overview

This system implements a state-of-the-art deep learning solution for detecting and classifying defects on fiber optic endfaces. The neural network is built using PyTorch and incorporates examples and techniques from the **Dive into Deep Learning** textbook (https://d2l.ai/).

## Key Features

### Multi-Task Learning Architecture
- **Region Classification**: Automatically identifies core, cladding, ferrule, and mixed regions
- **Defect Detection**: Binary classification for defect presence
- **Feature Extraction**: Hierarchical feature learning using CNN architecture

### Integration with Traditional Computer Vision
- **OpenCV Integration**: Edge detection for scratch identification
- **Blob Detection**: Automated detection of circular defects and debris
- **Pixel Intensity Analysis**: Statistical analysis of surface quality

### Based on D2L Examples
All neural network components are based on proven examples from Dive into Deep Learning:
- Convolutional layers (Chapter 7.2): `https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html`
- Pooling operations (Chapter 7.5): `https://d2l.ai/chapter_convolutional-neural-networks/pooling.html`
- Batch normalization (Chapter 8.5): `https://d2l.ai/chapter_convolutional-modern/batch-norm.html`
- Image augmentation (Chapter 14.1): `https://d2l.ai/chapter_computer-vision/image-augmentation.html`

## Directory Structure

```
project-directory/
├── dataset/                     # Input images organized in chunks
│   ├── chunk_1/                # Contains fiber optic endface images
│   ├── chunk_2/
│   ├── chunk_3/
│   └── ... (up to chunk_135)
├── reference/                   # Reference tensors for comparison
│   ├── core_ref/               # Core region reference .pt files
│   ├── cladding_ref/           # Cladding region reference .pt files
│   ├── ferrule_ref/            # Ferrule region reference .pt files
│   └── ... (40 total subfolders)
├── fiber_optic_defect_detection.py  # Main neural network script
└── README.md                   # This documentation
```

## Neural Network Architecture

### Feature Extraction Backbone
Based on D2L CNN examples with the following layers:

1. **Conv1**: 3→64 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool
2. **Conv2**: 64→128 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool  
3. **Conv3**: 128→256 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool
4. **Conv4**: 256→512 channels, 3×3 kernel, BatchNorm, ReLU, MaxPool

### Multi-Task Heads
- **Region Classifier**: 4-class output (core/cladding/ferrule/mixed)
- **Defect Detector**: Binary output (defect/clean)

## Key Components

### 1. FiberOpticCNN Class
The main neural network class implementing the multi-task architecture:

```python
class FiberOpticCNN(nn.Module):
    def __init__(self, num_classes=4):
        # CNN backbone for feature extraction
        # Multi-task heads for region and defect classification

    def forward(self, x):
        # Returns region predictions, defect predictions, and features
```

### 2. FiberOpticDataset Class
Custom PyTorch dataset handling the specific directory structure:

```python
class FiberOpticDataset(Dataset):
    def __init__(self, data_dir, reference_dir, transform=None):
        # Loads images from chunk directories
        # Integrates with reference tensor files
        # Applies D2L-style data augmentation
```

### 3. FiberOpticTrainer Class
Training loop implementation based on D2L patterns:

```python
class FiberOpticTrainer:
    def train_epoch(self):
        # Multi-task loss computation
        # Backpropagation and optimization

    def validate(self):
        # Accuracy calculation for both tasks
```

### 4. OpenCV Integration
Traditional computer vision techniques for enhanced detection:

- **Edge Detection**: Canny edge detection for scratch identification
- **Blob Detection**: SimpleBlobDetector for circular defects
- **Pixel Analysis**: Statistical analysis of surface quality

## Installation and Setup

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy matplotlib tqdm
pip install Pillow
```

### Dataset Preparation
1. Create the directory structure as shown above
2. Place fiber optic endface images in `dataset/chunk_*` directories
3. Add reference tensor files (.pt format) in `reference/` subdirectories

## Usage

### Basic Execution
```bash
python fiber_optic_defect_detection.py
```

### Expected Output
The system will:
1. Load and preprocess images from dataset chunks
2. Train the neural network using multi-task learning
3. Validate performance on held-out data
4. Generate comprehensive statistics and analysis report
5. Save results in JSON format

### Performance Metrics
The system tracks:
- **Region Classification Accuracy**: Percentage of correctly classified regions
- **Defect Detection Accuracy**: Binary classification performance
- **Confidence Scores**: Model certainty in predictions
- **Processing Speed**: Images processed per second
- **Loss Convergence**: Training loss reduction over epochs

## Customization Options

### Network Architecture
Modify the CNN backbone by adjusting:
- Number of convolutional layers
- Filter sizes and channel dimensions
- Activation functions and normalization

### Data Augmentation
Based on D2L Chapter 14.1, customize transforms:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Training Parameters
Adjust hyperparameters:
- Learning rate: Default 0.001
- Batch size: Default 16
- Number of epochs: Default 20
- Loss function weights for multi-task learning

## Integration with Reference Data

The system automatically integrates with reference tensor files:
1. Loads .pt files from reference subdirectories
2. Uses reference features for comparison and validation
3. Incorporates reference data into training process

## Output and Analysis

### Statistics Report
Generated JSON report includes:
```json
{
  "analysis_summary": {
    "total_images_processed": 1000,
    "average_confidence_score": "0.892",
    "average_processing_time_per_batch": "0.045s"
  },
  "region_distribution": {
    "core": 450,
    "cladding": 320,
    "ferrule": 180,
    "mixed": 50
  },
  "defect_distribution": {
    "defect": 127,
    "clean": 873
  }
}
```

### Visual Outputs
The system can generate:
- Training loss curves
- Accuracy progression plots
- Confusion matrices
- Feature visualization maps

## Performance Optimization

### GPU Acceleration
The system automatically detects and uses CUDA when available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Memory Management
- Efficient batch processing
- Gradient accumulation for large datasets
- Memory-mapped dataset loading

### Multi-Processing
- Parallel data loading with configurable workers
- Batch processing optimization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image resolution
2. **Dataset Not Found**: Ensure proper directory structure
3. **Reference Files Missing**: Check .pt file locations
4. **Poor Convergence**: Adjust learning rate or add regularization

### Debug Mode
Enable detailed logging by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This implementation follows the educational approach of Dive into Deep Learning. For improvements:
1. Base new features on D2L examples when possible
2. Maintain compatibility with the existing directory structure
3. Document any modifications to the neural network architecture

## References

- **Dive into Deep Learning**: https://d2l.ai/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **OpenCV Documentation**: https://docs.opencv.org/
- **Fiber Optic Standards**: IEC 61300-3-35 (referenced in implementation)

## License

This educational implementation is based on open-source D2L examples and is intended for research and learning purposes.
