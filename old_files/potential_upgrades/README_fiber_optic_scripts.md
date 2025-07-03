# Fiber Optic Detection Scripts Collection

This directory contains various implementations for fiber optic end-face inspection and defect detection, ranging from classical computer vision approaches to modern deep learning solutions.

## Overview of Scripts

### 1. **AI-Powered PyTorch Models**

#### `ai_segmenter_pytorch.py`
- **Purpose**: U-Net based segmentation for fiber zones (core, cladding, ferrule) and defects
- **Key Features**:
  - ResNet-34 encoder with pre-trained ImageNet weights
  - 4-class output: background, core, cladding, ferrule (+optional defect class)
  - Drop-in replacement for classical segmentation methods
- **Usage**: 
  ```python
  segmenter = AI_Segmenter("segmenter_best.pth")
  masks = segmenter.segment(bgr_image)
  ```

#### `anomaly_detector_pytorch.py`
- **Purpose**: Unsupervised anomaly detection using convolutional autoencoder
- **Key Features**:
  - Trains only on clean images (no defect labels needed)
  - Pixel-wise reconstruction error for anomaly detection
  - Automatic defect localization and severity scoring
- **Usage**:
  ```python
  detector = AI_AnomalyDetector("cae_last.pth")
  score_map, defects = detector.detect(image, fiber_mask)
  ```

#### `train_segmenter.py` & `train_anomaly.py`
- Training scripts for the above models
- Segmenter uses synthetic Cut-Paste defects for training
- Anomaly detector uses unsupervised reconstruction loss

### 2. **Classical Computer Vision Methods**

#### `do2mr_lei_detector.py`
- **Purpose**: Implementation of DO²MR and LEI algorithms from Mei et al. paper
- **Key Features**:
  - DO²MR (Difference-of-Min-Max Ranking) for region defects
  - LEI (Linear Enhancement Inspector) for scratch detection
  - No training required - pure image processing
- **Usage**:
  ```python
  detector = UnifiedDefectDetector()
  results = detector.detect_all(image, zone_masks)
  ```

### 3. **Real-Time Processing**

#### `realtime_analyzer.py`
- **Purpose**: Continuous video frame processing wrapper
- **Key Features**:
  - Integrates with existing segmentation/detection pipeline
  - Frame throttling to maintain real-time performance
  - Thread-safe processing with minimal latency

#### `realtime_dashboard.py`
- **Purpose**: Flask-based live dashboard with MJPEG streaming
- **Key Features**:
  - Web interface for real-time inspection
  - Live defect statistics table
  - Annotated video stream
- **Usage**: `python realtime_dashboard.py --source 0`

#### `realtime_location_pipeline.py`
- **Purpose**: Advanced real-time pipeline with physical coordinate tracking
- **Key Features**:
  - Pixel → micron → mm → GPS coordinate transformation
  - WebSocket streaming for live data
  - Performance metrics and quality scoring
  - GPS integration for field deployment

### 4. **Advanced Vision-Language Models**

#### `gemma_fiber_analyzer.py`
- **Purpose**: Google Gemma 3 Vision integration
- **Key Features**:
  - Multi-modal defect detection (vision + language)
  - Structured JSON output from natural language prompts
  - Quality assessment with recommendations
  - Visual embeddings for similarity comparison

#### `llama_vision_finetuner.py`
- **Purpose**: Fine-tune Llama-3.2-Vision for fiber inspection
- **Key Features**:
  - LoRA/QLoRA for efficient fine-tuning
  - Converts existing defect annotations to training data
  - Structured output generation
  - 4-bit quantization support

### 5. **Dataset Building and Training Infrastructure**

#### `ml_dataset_builder.py`
- **Purpose**: Comprehensive dataset creation from raw images
- **Key Features**:
  - Unsupervised zone segmentation
  - Automatic defect detection and clustering
  - Feature extraction (intensity, texture, gradients)
  - SQLite database for metadata
  - Defect library generation

#### `fiber_dataset_pytorch.py`
- **Purpose**: PyTorch dataset builder integrated with existing pipeline
- **Key Features**:
  - Reuses existing pipeline outputs
  - Creates train/val/test splits
  - Includes simple CNN training example
  - SQLite-based metadata management

### 6. **Framework Integration Examples**

#### `torch_quality_classifier.py`
- **Purpose**: Minimal PyTorch classifier for pass/fail prediction
- **Key Features**:
  - Simple MLP architecture (matches "PyTorch in 100 seconds")
  - Drop-in integration with 3 lines of code
  - Training helper included

#### `tensorflow_attachment.py`
- **Purpose**: TensorFlow/Keras integration example
- **Key Features**:
  - Fashion-MNIST demo architecture
  - CLI for training and inference
  - Minimal integration footprint

### 7. **Supporting Scripts**

#### `separation_ai.py` & `detection_ai.py`
- Wrapper modules for AI models
- Maintain compatibility with existing pipeline interfaces

## Getting Started

### Installation

```bash
# PyTorch dependencies
pip install torch torchvision opencv-python numpy

# For real-time dashboard
pip install flask imutils

# For vision-language models
pip install transformers accelerate bitsandbytes

# For GPS support (optional)
pip install pynmea2 pyserial
```

### Quick Start Examples

1. **Classical Detection**:
```python
from do2mr_lei_detector import UnifiedDefectDetector
detector = UnifiedDefectDetector()
results = detector.detect_all(cv2.imread("fiber_image.jpg"))
print(f"Found {results['count']} defects")
```

2. **AI Detection**:
```python
from ai_segmenter_pytorch import AI_Segmenter
from anomaly_detector_pytorch import AI_AnomalyDetector

segmenter = AI_Segmenter("segmenter_best.pth")
detector = AI_AnomalyDetector("cae_last.pth")

image = cv2.imread("fiber_image.jpg")
masks = segmenter.segment(image)
score_map, defects = detector.detect(image, masks['core'] | masks['cladding'])
```

3. **Real-Time Dashboard**:
```bash
python realtime_dashboard.py --source 0
# Open browser to http://localhost:5000
```

4. **Build ML Dataset**:
```python
from ml_dataset_builder import DatasetBuilder
builder = DatasetBuilder("./output_dataset")
builder.build_dataset("./raw_images")
```

## Model Training

### Train Segmentation Model
```bash
python train_segmenter.py --clean_dir ./clean_images --epochs 30 --out segmenter.pth
```

### Train Anomaly Detector
```bash
python train_anomaly.py --clean_dirs ./clean_images --epochs 50 --out anomaly.pth
```

## Configuration

Most scripts support configuration through JSON files or command-line arguments. Example config structure:

```json
{
  "calibration": {
    "pixels_per_micron": 0.65,
    "physical_origin_mm": [0, 0, 0]
  },
  "realtime_settings": {
    "max_fps": 10,
    "display": true,
    "broadcast": {
      "mode": "websocket",
      "port": 8765
    }
  }
}
```

## Performance Notes

- Classical methods (DO²MR/LEI): ~30-50ms per frame on CPU
- AI segmentation: ~20-30ms on GPU, ~100ms on CPU
- Real-time pipeline: 10-15 FPS with full processing
- Vision-language models: 100-500ms depending on model size

## References

- Mei et al., "Automated Inspection of Fiber Optic Connector End Face", Sensors 2018
- IEC 61300-3-35 standard for fiber optic connector inspection