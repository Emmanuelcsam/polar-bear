# Integrated Neural Network Architecture

## Overview

This implementation creates a truly integrated neural network where segmentation, reference comparison, and anomaly detection happen internally through the network layers, not as separate preprocessing steps.

## Key Concepts

### 1. Feature Extraction with Dynamic Weighting

The `GradientPixelWeightedConv` layers extract features (edges) while simultaneously modulating weights based on:
- **Gradient intensity** (A coefficient in I=Ax₁+Bx₂+...)
- **Pixel position** (B coefficient)
- Each layer learns how much these factors should influence feature extraction

### 2. Internal Segmentation Through Feature Comparison

The `FeatureComparisonBlock`:
- Learns reference patterns for core, cladding, and ferrule regions
- Compares extracted features to these patterns
- Classifies each pixel into regions based on similarity
- This happens at multiple scales for robustness

### 3. Integrated Anomaly Detection

Anomaly detection occurs by:
1. **Reconstruction**: The network learns to reconstruct the input
2. **Subtraction**: Taking the absolute difference between reconstruction and original
3. **Trend Filtering**: Learned parameters distinguish true anomalies from expected region transitions
4. **SSIM Integration**: Structural similarity provides additional anomaly signals

### 4. Reference Learning

Instead of loading external references:
- The network learns reference embeddings during training
- Compares current features to these learned references
- Outputs similarity scores that must exceed 0.7 threshold

## How It Works Internally

```
Input Image
    ↓
[GradientPixelWeightedConv Layers]
    - Extract features/edges
    - Weight by gradient intensity & pixel position
    ↓
[FeatureComparisonBlocks]
    - Compare to learned patterns
    - Classify into regions (core/cladding/ferrule)
    ↓
[Parallel Branches]
    ├─ Segmentation Branch → Region masks
    ├─ Reference Branch → Similarity scores
    └─ Reconstruction Branch → For anomaly detection
    ↓
[Anomaly Detection]
    - Reconstruction - Original = Difference
    - Apply trend filtering
    - Combine with SSIM
    ↓
Output: Segmentation + Anomalies + Similarity Score
```

## Training Process

1. **Multi-task Learning**: The network simultaneously learns to:
   - Segment regions correctly
   - Match to appropriate references
   - Reconstruct normal patterns
   - Identify anomalies

2. **Dynamic Parameter Adjustment**: During training, gradient and position influences are adjusted based on loss

3. **Trend Learning**: The network learns expected gradient trends for each region to distinguish defects from normal transitions

## Key Differences from Previous Approach

- **No separate segmentation step**: Segmentation emerges from feature comparison
- **No external reference loading**: References are learned as network parameters
- **Integrated anomaly detection**: Not a post-processing step but part of the forward pass
- **End-to-end differentiable**: Everything can be optimized together

## Usage

```python
# Training
python integrated_main.py train

# Inference  
python integrated_main.py inference image.jpg

# Demo
python integrated_main.py demo
```

## Equation Implementation

The network implements I=Ax₁+Bx₂+Cx₃...=S(R) where:
- A: Gradient influence (learned per layer)
- B: Position influence (learned per layer)
- C: Circle alignment (commented out as requested)
- S(R): Similarity to learned references

These parameters can be adjusted and are tracked throughout training.