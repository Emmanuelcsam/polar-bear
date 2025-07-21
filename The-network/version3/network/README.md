# Fiber Optics Neural Network Framework

A comprehensive deep learning framework for fiber optics image classification and defect detection.

## Overview

This framework implements a specialized neural network for analyzing fiber optic endface images. It performs:
- Image segmentation into core, cladding, and ferrule regions
- Defect detection and anomaly mapping
- Reference-based similarity analysis
- Real-time and batch processing capabilities

## Key Features

- **Custom Weighted Neural Network**: Weights dependent on image gradient intensity and pixel positions
- **Automatic Segmentation**: Identifies core, cladding, and ferrule regions
- **Anomaly Detection**: Combines structural similarity and local anomaly heatmaps
- **HPC Support**: Distributed training across multiple GPUs
- **Comprehensive Logging**: Detailed timestamps and process tracking
- **Minimal Export Sizes**: Optimized file formats for results

## Project Structure

```
fiber_optics_nn/
├── config/          # Configuration settings
├── core/            # Core algorithms (similarity, anomaly detection)
├── data_loaders/    # Tensor and metadata loading
├── models/          # Neural network architectures
├── processors/      # Image processing and segmentation
├── trainers/        # Training loops (single and distributed)
└── utils/           # Logging and results export
```

## Usage

### Training

Single GPU training:
```bash
python main_training.py [num_epochs]
```

Distributed training (multiple GPUs):
```bash
python main_training.py distributed
```

Resume training:
```bash
python main_training.py resume
```

### Inference

Process images for defect detection:
```bash
python main_inference.py
```

## Configuration

Edit `fiber_optics_nn/config/config.py` to adjust:
- Reference data paths
- Neural network parameters
- Weight factors (gradient, pixel position)
- Similarity thresholds
- Training hyperparameters

## Requirements

- PyTorch
- OpenCV
- NumPy
- scikit-image
- tqdm
- CUDA (for GPU acceleration)

## Mathematical Model

The system implements the equation: I = Ax₁ + Bx₂ + Cx₃... = S(R)
- A: Gradient weight factor
- B: Pixel position weight factor
- S: Similarity coefficient
- R: Reference image

## Output

Results are exported as compressed matrices containing:
- Defect overlay maps
- Anomaly heatmaps
- Defect locations and types
- Confidence scores
- Summary statistics