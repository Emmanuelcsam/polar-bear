# Fiber Optic Analysis System

A modular, scalable system for analyzing fiber optic images using deep learning. The system performs region classification, anomaly detection, and similarity comparison for quality control in fiber optic manufacturing.

## 🏗️ Architecture

The system is organized into multiple specialized modules that work together:

```
fiber-optic-analysis/
├── main.py              # Main entry point
├── config.py            # Configuration management
├── model.py             # Neural network architecture
├── dataset.py           # Data loading and preprocessing
├── trainer.py           # Training logic
├── evaluator.py         # Evaluation and metrics
├── losses.py            # Loss functions
├── optimizer.py         # Model optimization (pruning, distillation)
├── utils.py             # Utility functions
├── app.py               # Web interface
├── config.yaml          # Configuration file
└── requirements.txt     # Dependencies
```

## 🚀 Features

- **Multi-task Learning**: Simultaneous region classification, anomaly detection, and similarity comparison
- **Modular Design**: Each component is independent and can be used separately
- **Distributed Training**: Support for multi-GPU training
- **Model Optimization**: Pruning and knowledge distillation for deployment
- **Web Interface**: Interactive Gradio-based UI for real-time analysis
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Evaluation**: Detailed metrics and failure analysis

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fiber-optic-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data structure:
```
fiber_optic_data/
├── core/
│   ├── image1.jpg
│   └── ...
├── cladding/
│   ├── image1.jpg
│   └── ...
├── ferrule/
│   ├── image1.jpg
│   └── ...
└── defects/
    ├── image1.jpg
    └── ...
```

## 🎯 Usage

### Training

Train the model using the main script:

```bash
# Basic training
python main.py --mode train --config config.yaml

# Distributed training (multi-GPU)
torchrun --nproc_per_node=2 main.py --mode train

# Resume from checkpoint
python main.py --mode train --checkpoint checkpoints/epoch_10.pth

# Verbose logging
python main.py --mode train --verbose
```

### Evaluation

Evaluate a trained model:

```bash
# Evaluate best model
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# Evaluate specific epoch
python main.py --mode eval --checkpoint checkpoints/epoch_50.pth
```

### Model Optimization

Apply pruning and knowledge distillation:

```bash
# Optimize model
python main.py --mode optimize --checkpoint checkpoints/best_model.pth
```

### Web Interface

Launch the interactive web interface:

```bash
# Local interface
python app.py --config config.yaml --checkpoint checkpoints/best_model.pth

# Public interface (shareable link)
python app.py --share

# Custom host and port
python app.py --host 0.0.0.0 --port 8080
```

## 🔧 Configuration

The system uses a YAML configuration file (`config.yaml`) to manage all parameters:

```yaml
# Model configuration
model:
  backbone: "resnet18"
  num_classes: 4
  embedding_dim: 128

# Training configuration
training:
  num_epochs: 50
  batch_size: 32
  use_amp: true

# Loss weights
loss:
  weights:
    classification: 1.0
    anomaly: 0.7
    similarity: 0.5

# Similarity equation: S = A×conf + B×(1-anom) + C×sim
equation:
  coefficients:
    A: 0.4
    B: 0.4
    C: 0.2
```

## 🧩 Module Details

### Core Modules

- **`config.py`**: Centralized configuration management with validation
- **`model.py`**: FiberOpticNet architecture with attention mechanisms
- **`dataset.py`**: Data loading with support for images and tensor files
- **`trainer.py`**: Training loop with mixed precision and distributed support
- **`evaluator.py`**: Comprehensive evaluation and failure analysis
- **`losses.py`**: Focal loss and combined loss functions
- **`optimizer.py`**: Model compression techniques

### Utility Modules

- **`utils.py`**: Visualization, preprocessing, and utility functions
- **`app.py`**: Gradio-based web interface
- **`main.py`**: Command-line interface and orchestration

## 📊 Model Architecture

The FiberOpticNet model consists of:

1. **Backbone**: ResNet-based feature extractor
2. **Attention**: CBAM (Convolutional Block Attention Module)
3. **Multi-head Output**: 
   - Region classification head
   - Anomaly detection head
   - Embedding similarity head

## 🎨 Web Interface Features

- **Real-time Analysis**: Upload images for instant analysis
- **Interactive Controls**: Adjust equation coefficients dynamically
- **Visual Results**: Segmentation overlays and anomaly heatmaps
- **Detailed Metrics**: Classification probabilities and confidence scores

## 📈 Performance Features

- **Mixed Precision Training**: Faster training with AMP
- **Distributed Support**: Multi-GPU training
- **Model Optimization**: Pruning and knowledge distillation
- **Efficient Data Loading**: Optimized data pipeline

## 🔬 Evaluation Metrics

- Overall classification accuracy
- Per-class accuracy
- Average similarity score
- Pass/fail rate based on threshold
- Failure case analysis
- Confusion matrices

## 🚀 Deployment

### For Production

1. Train and optimize your model:
```bash
python main.py --mode train
python main.py --mode optimize --checkpoint checkpoints/best_model.pth
```

2. Deploy the web interface:
```bash
python app.py --checkpoint checkpoints/distilled_student_model.pth --host 0.0.0.0
```

### For Development

1. Use the modular components individually:
```python
from config import ConfigManager
from model import create_model
from evaluator import Evaluator

config = ConfigManager('config.yaml').config
model = create_model(config)
evaluator = Evaluator(config)
```

## 🛠️ Customization

### Adding New Loss Functions

1. Add your loss class to `losses.py`
2. Update the configuration in `config.yaml`
3. Modify the `CombinedLoss` class to include your loss

### Adding New Backbones

1. Modify the `_build_backbone` method in `model.py`
2. Add the new backbone option to your configuration

### Custom Data Formats

1. Extend the `FiberOpticsDataset` class in `dataset.py`
2. Implement custom loading methods for your data format

## 📝 Contributing

1. Follow the modular design principles
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Add tests for new features

## 📄 License

[Add your license information here]

## 🤝 Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the example configurations
