"""
Neural Network Integration System
A comprehensive framework for unifying disparate scripts into a neural network
"""

from .neural_network import NeuralNetwork
from .core.node_base import BaseNode, AtomicNode, CompositeNode, NodeMetadata, NodeInput, NodeOutput
from .core.synapse import SynchronousSynapse, AsynchronousSynapse, StreamingSynapse, BroadcastSynapse
from .core.logger import logger, LogChannel, LogLevel
from .core.config_manager import config_manager
from .core.parameter_tuner import ParameterTuner, TuningStrategy

__version__ = "1.0.0"
__author__ = "Polar Bear Neural Network Team"

__all__ = [
    "NeuralNetwork",
    "BaseNode",
    "AtomicNode", 
    "CompositeNode",
    "NodeMetadata",
    "NodeInput",
    "NodeOutput",
    "SynchronousSynapse",
    "AsynchronousSynapse",
    "StreamingSynapse",
    "BroadcastSynapse",
    "logger",
    "LogChannel",
    "LogLevel",
    "config_manager",
    "ParameterTuner",
    "TuningStrategy"
]