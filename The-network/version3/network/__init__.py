"""
Fiber Optics Neural Network Framework
=====================================

A comprehensive framework for fiber optics image classification and defect detection.
"""

from .utils import get_logger, ResultsExporter
from .config import get_config
from .data_loaders import TensorDataLoader
from .processors import ImageProcessor, FiberOpticSegmentation
from .models import FiberOpticsNeuralNetwork
from .core import SimilarityCalculator, AnomalyDetector
from .trainers import FiberOpticsTrainer, launch_distributed_training

__version__ = "1.0.0"

__all__ = [
    'get_logger',
    'get_config',
    'ResultsExporter',
    'TensorDataLoader',
    'ImageProcessor',
    'FiberOpticSegmentation',
    'FiberOpticsNeuralNetwork',
    'SimilarityCalculator',
    'AnomalyDetector',
    'FiberOpticsTrainer',
    'launch_distributed_training'
]