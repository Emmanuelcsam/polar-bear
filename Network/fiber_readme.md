# Fiber Optics Neural Network System

A comprehensive deep learning system for fiber optic endface analysis, featuring simultaneous feature classification, anomaly detection, and quality assessment.

## Overview

This system implements an advanced integrated neural network that:
- **Simultaneously** performs feature extraction, classification, and anomaly detection
- Segments fiber optic images into core, cladding, and ferrule regions
- Compares images to a reference database with similarity threshold requirements (>0.7)
- Detects and locates defects while distinguishing them from normal region transitions
- Implements the equation: `I = Ax1 + Bx2 + Cx3... = S(R)` where coefficients can be adjusted

## Key Features

### 1. Integrated Processing
- Neural network performs segmentation, comparison, and anomaly detection internally
- Multi-scale feature extraction with correlation between scales
- Simultaneous analysis at each feature level

### 2. Advanced Anomaly Detection
- Combines neural network detection with structural similarity analysis
- Distinguishes between defects and normal region transitions
- Learns gradient and position trends for each region

### 3. Reference Comparison
- Maintains database of reference tensors organized by region
- Calculates multiple similarity metrics (cosine, L2, learned, SSIM)
- Enforces similarity threshold of 0.7 for quality assurance

### 4. Real-time Capabilities
- Supports batch processing and streaming analysis
- GPU acceleration for HPC environments
- Optimized for fast inference

## System Architecture

```
main.py                    # Entry point
├── config.py             # Configuration management
├── logger.py             # Verbose logging system
├── tensor_processor.py   # Image tensorization
├── feature_extractor.py  # Multi-scale feature extraction
├── reference_comparator.py # Reference database comparison
├── anomaly_detector.py   # Comprehensive anomaly detection
├── integrated_network.py # Complete integrated neural network
├── trainer.py           # Training system
└── data_loader.py       # Data loading and batching
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fiber-optics-nn

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Note: This system does NOT use argparse or command-line flags by design.**

### Training the Model

```bash
# Train for default epochs
python main.py train

# Train for specific epochs
python main.py train 100

# Resume from checkpoint
python main.py train 50 checkpoints/best_model.pth
```

### Analyzing Images

```bash
# Analyze single image
python main.py analyze path/to/image.jpg

# Batch process folder
python main.py batch path/to/folder

# Batch process with limit
python main.py batch path/to/folder 100
```

### Real-time Processing

```bash
# Start real-time processing
python main.py realtime
```

### Evaluate Performance

```bash
# Evaluate model performance
python main.py evaluate
```

### Update Parameters

```bash
# Update equation coefficient
python main.py update A 1.5
python main.py update B 0.8
```

## Configuration

Edit `fiber_optics_config.json` or use the update command to modify:
- Equation coefficients (A, B, C, D, E)
- Gradient weight factor
- Position weight factor
- Similarity threshold
- Anomaly threshold

## Data Organization

Expected folder structure for tensorized data:
```
reference/
└── tensorized-data/
    ├── core-batch-1/
    ├── core-batch-2/
    ├── cladding-batch-1/
    ├── ferrule-batch-1/
    └── defects/
```

## Key Concepts

### Equation System
The system implements: `I = Ax1 + Bx2 + Cx3 + Dx4 + Ex5 = S(R)`

Where:
- x1: Reference similarity
- x2: Trend adherence
- x3: Inverse anomaly score
- x4: Segmentation confidence
- x5: Reconstruction similarity

### Multi-scale Processing
- Features extracted at 4 different scales
- Small defects visible at fine scales
- Large defects visible at coarse scales
- Cross-scale correlation ensures consistency

### Trend Analysis
- Learns expected gradient patterns for each region
- Features following trends → classified as regions
- Features deviating from trends → classified as defects

## Output

Results are exported to the `results/` folder:
- `<image_name>_results.txt`: Anomaly map matrix
- `<image_name>_results_complete.npz`: Complete analysis results

## Logging

Verbose logs are saved to `logs/` with timestamps for every operation.
Check logs for:
- Detailed processing steps
- Similarity scores and threshold checks
- Anomaly detection results
- Performance metrics

## GPU Support

The system automatically detects and uses GPU if available.
Optimized for HPC environments with parallel processing capabilities.

## Development

To extend the system:
1. Modify individual modules (maintain modular structure)
2. Update configuration in `config.py`
3. Add new loss functions in `trainer.py`
4. Extend the integrated network in `integrated_network.py`

## Citations

Based on the advanced integrated approach from the research on simultaneous feature classification and anomaly detection in fiber optic analysis.