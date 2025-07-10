# Key Improvements to Fiber Optic Anomaly Detection

## 1. **Siamese Network Architecture**
Based on the tutorial's explanation of Siamese networks for similarity learning, I've added:
- `FiberSiameseNet` class that processes fiber images in parallel with shared weights
- Feature extraction backbone using pretrained ResNet (configurable)
- Embedding projection to learn a similarity space where normal fibers cluster together

## 2. **Multiple Loss Functions**
Following the tutorial's discussion on loss functions:
- **Contrastive Loss**: Brings similar fiber images together and pushes anomalous ones apart
- **Triplet Loss**: Learns relative rankings (normal fiber closer to normal than to defective)
- Both losses use configurable margins as discussed in the tutorial

## 3. **Correlation Layer**
Inspired by the FlowNet architecture in the tutorial:
- Added `CorrelationLayer` to compute local matching scores between test and reference images
- Helps identify specific regions where defects occur
- Parameter-free operation that enhances defect localization

## 4. **Hierarchical Triplet Sampling**
Implemented the tutorial's hierarchical sampling strategy:
- `HierarchicalTripletSampler` builds a tree of fiber conditions
- Samples diverse classes while focusing on hard cases
- Improves training efficiency by selecting informative triplets

## 5. **Ensemble Methods**
Following the divide-and-conquer approach:
- Support for training multiple models (`ensemble_size` parameter)
- Each model can specialize in different defect types
- Ensemble predictions improve robustness

## 6. **Deep Feature Integration**
The enhanced system combines:
- Original handcrafted features (statistical, morphological, etc.)
- Deep learned features from Siamese network
- Hybrid approach leverages both domain knowledge and learned representations

## 7. **Training Capabilities**
New training functionality:
- `train_similarity_model()` method for end-to-end learning
- Support for both pair-wise and triplet training
- Validation during training to monitor performance

## 8. **Enhanced Similarity Metrics**
Multiple similarity measures as discussed in the tutorial:
- Cosine similarity in embedding space
- Euclidean distance
- Learned similarity from the neural network

## Usage Example

```python
# Initialize with deep learning enabled
config = OmniConfig(
    use_deep_features=True,
    embedding_dim=128,
    backbone='resnet50',
    ensemble_size=3,
    use_correlation_layer=True
)

analyzer = DeepFiberAnalyzer(config)

# Train on normal vs defective fibers
analyzer.train_similarity_model(
    train_dir='fiber_data/train',
    val_dir='fiber_data/val',
    epochs=100
)

# Analyze new fiber image
results = analyzer.analyze_end_face('test_fiber.png', 'output_dir')
```

## Benefits Over Original Approach

1. **Scalability**: Once trained, can detect novel defect types without retraining
2. **Robustness**: Deep features are more invariant to lighting, angle variations
3. **Precision**: Correlation layer provides pixel-level defect localization
4. **Adaptability**: Can fine-tune for specific fiber types or defect patterns
5. **Performance**: Ensemble approach reduces false positives/negatives

## Architecture Overview

```
Test Image → Siamese Network → Embedding
     ↓              ↓              ↓
Reference → Siamese Network → Embedding
     ↓              ↓              ↓
Correlation Layer → Local Matching → Defect Regions
     ↓              ↓              ↓
Classical Features + Deep Features → Decision Network → Anomaly Score
```

This enhanced system maintains backward compatibility with your existing pipeline while adding powerful deep learning capabilities for more accurate and robust fiber anomaly detection.