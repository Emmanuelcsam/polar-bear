# utils.py
# Utility functions for the fiber optic analysis system

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import json

def setup_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)  # Uncomment if using random module
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info():
    """Get information about available compute devices."""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_names': []
    }
    
    if torch.cuda.is_available():
        device_info['current_device'] = torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            device_info['device_names'].append(torch.cuda.get_device_name(i))
    
    return device_info

def create_segmentation_overlay(image, logits, class_names, colors=None):
    """
    Create a colored segmentation mask overlay.
    
    Args:
        image: Input image (PIL Image or numpy array)
        logits: Model prediction logits
        class_names: List of class names
        colors: Optional list of colors for each class
        
    Returns:
        PIL Image with segmentation overlay
    """
    if colors is None:
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ]
    
    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Get predictions
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    else:
        preds = logits
    
    # Resize if necessary
    if image_np.shape[:2] != preds.shape:
        preds = cv2.resize(preds.astype(np.uint8), 
                          (image_np.shape[1], image_np.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    # Create segmentation map
    seg_map = np.zeros_like(image_np)
    for i, color in enumerate(colors[:len(class_names)]):
        seg_map[preds == i] = color
    
    # Blend with original image
    overlay = cv2.addWeighted(image_np, 0.6, seg_map, 0.4, 0)
    
    return Image.fromarray(overlay.astype(np.uint8))

def create_anomaly_heatmap(anomaly_map, colormap=cv2.COLORMAP_JET):
    """
    Create a visual heatmap from anomaly detection output.
    
    Args:
        anomaly_map: Anomaly detection output tensor or array
        colormap: OpenCV colormap to use
        
    Returns:
        PIL Image of the heatmap
    """
    # Convert to numpy if tensor
    if isinstance(anomaly_map, torch.Tensor):
        heatmap = torch.sigmoid(anomaly_map).squeeze().cpu().detach().numpy()
    else:
        heatmap = anomaly_map.squeeze()
    
    # Normalize to 0-1 range
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert to 0-255 range
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Convert BGR to RGB
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(heatmap_rgb)

def preprocess_image_for_inference(image, target_size=224):
    """
    Preprocess an image for model inference.
    
    Args:
        image: Input image (PIL Image, numpy array, or file path)
        target_size: Target image size
        
    Returns:
        Preprocessed tensor ready for inference
    """
    # Load image if it's a file path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor

def save_predictions_to_json(predictions, output_path):
    """Save prediction results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    predictions_serializable = convert_numpy(predictions)
    
    with open(output_path, 'w') as f:
        json.dump(predictions_serializable, f, indent=2)

def plot_training_history(history, save_path=None):
    """
    Plot training history (loss, accuracy, etc.).
    
    Args:
        history: Dictionary containing training metrics over epochs
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot training loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # Plot validation accuracy
    if 'val_accuracy' in history:
        axes[0, 1].plot(history['val_accuracy'])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
    
    # Plot similarity scores
    if 'val_similarity' in history:
        axes[1, 0].plot(history['val_similarity'])
        axes[1, 0].set_title('Validation Similarity Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Similarity')
        axes[1, 0].grid(True)
    
    # Plot learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        dataset: Dataset object
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            label = sample['label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] += 1
        except:
            continue
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    if total_samples > 0:
        # Avoid division by zero
        class_weights = torch.ones(num_classes)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (num_classes * class_counts[i])
    else:
        class_weights = torch.ones(num_classes)
    
    return class_weights

def log_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Log a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for calculation
    """
    logger = logging.getLogger('ModelSummary')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Summary:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    logger.info(f"Estimated model size: {model_size_mb:.2f} MB")
    
    # Test forward pass to check model
    try:
        dummy_input = torch.randn(*input_size)
        with torch.no_grad():
            output = model(dummy_input)
        logger.info("Model forward pass: SUCCESS")
        
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"Output '{key}': {tuple(value.shape)}")
                    
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")

class MetricsTracker:
    """Track and manage training metrics over time."""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, metric_name):
        """Get the latest value of a metric."""
        return self.metrics.get(metric_name, [])[-1] if metric_name in self.metrics else None
    
    def get_best(self, metric_name, mode='max'):
        """Get the best value of a metric."""
        if metric_name not in self.metrics:
            return None
        
        values = self.metrics[metric_name]
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def save(self, filepath):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filepath):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
