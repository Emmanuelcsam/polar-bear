#!/usr/bin/env python3
"""
Neural Network Node Architecture
Base classes and interfaces for creating neural network nodes
"""

import os
import sys
import time
import json
import uuid
import inspect
import asyncio
import threading
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

from .logger import logger, LogChannel, TimedOperation


class NodeState(Enum):
    """Node operational states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    PAUSED = "paused"
    SHUTDOWN = "shutdown"


class NodeType(Enum):
    """Types of neural network nodes"""
    ATOMIC = "atomic"          # Single function wrapper
    COMPOSITE = "composite"    # Group of related functions
    META = "meta"             # Higher-order orchestrator
    GATEWAY = "gateway"       # External system interface
    TRANSFORM = "transform"   # Data transformation
    VALIDATOR = "validator"   # Input/output validation
    SYNAPSE = "synapse"       # Connection node
    NEURON = "neuron"         # Processing unit


@dataclass
class NodeMetadata:
    """Metadata for a neural network node"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: NodeType = NodeType.ATOMIC
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        result = {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'created_at': self.created_at,
            'tags': self.tags,
            'capabilities': self.capabilities,
            'requirements': self.requirements,
            'parameters': self.parameters
        }
        return result


@dataclass
class NodeInput:
    """Input data structure for nodes"""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_node: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input against schema"""
        if schema is None:
            return True
        # TODO: Implement schema validation
        return True


@dataclass
class NodeOutput:
    """Output data structure from nodes"""
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    correlation_id: str = ""


@dataclass
class NodeMetrics:
    """Performance metrics for a node"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    last_call_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    
    def update(self, processing_time: float, success: bool):
        """Update metrics with new call data"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_calls
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.last_call_time = time.time()
        self.error_rate = self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0
        
        # Calculate throughput (calls per second)
        if self.total_processing_time > 0:
            self.throughput = self.total_calls / self.total_processing_time


class BaseNode(ABC):
    """Abstract base class for all neural network nodes"""
    
    def __init__(self, metadata: Optional[NodeMetadata] = None):
        """Initialize base node"""
        self.metadata = metadata or NodeMetadata()
        self.state = NodeState.UNINITIALIZED
        self.metrics = NodeMetrics()
        self.connections: Dict[str, 'BaseNode'] = {}
        self.parameters: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._executor = None
        self._shutdown_event = threading.Event()
        
        # Tunable parameters
        self.tunable_params: Dict[str, Dict[str, Any]] = {}
        
        # Initialize logging
        logger.info(LogChannel.MODULE, f"Creating node: {self.metadata.name}",
                   node_id=self.metadata.id, node_type=self.metadata.type.value)
    
    @abstractmethod
    def process(self, input_data: NodeInput) -> NodeOutput:
        """Process input and return output - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: NodeInput) -> bool:
        """Validate input data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data: NodeOutput) -> bool:
        """Validate output data - must be implemented by subclasses"""
        pass
    
    def initialize(self) -> bool:
        """Initialize the node"""
        try:
            self.state = NodeState.INITIALIZING
            logger.info(LogChannel.MODULE, f"Initializing node: {self.metadata.name}")
            
            # Create executor based on node type
            if self.metadata.type in [NodeType.META, NodeType.COMPOSITE]:
                self._executor = ProcessPoolExecutor(max_workers=4)
            else:
                self._executor = ThreadPoolExecutor(max_workers=2)
            
            # Load parameters
            self._load_parameters()
            
            # Perform custom initialization
            if self._custom_init():
                self.state = NodeState.READY
                logger.success(LogChannel.MODULE, f"Node initialized: {self.metadata.name}")
                return True
            else:
                self.state = NodeState.ERROR
                return False
                
        except Exception as e:
            self.state = NodeState.ERROR
            logger.error(LogChannel.MODULE, f"Failed to initialize node: {self.metadata.name}",
                        error=str(e))
            return False
    
    def _custom_init(self) -> bool:
        """Custom initialization - can be overridden by subclasses"""
        return True
    
    def _load_parameters(self):
        """Load node parameters from metadata"""
        self.parameters.update(self.metadata.parameters)
    
    def execute(self, input_data: NodeInput) -> NodeOutput:
        """Execute node processing with validation and metrics"""
        start_time = time.time()
        
        with self._lock:
            # Check state
            if self.state != NodeState.READY:
                return NodeOutput(
                    data=None,
                    node_id=self.metadata.id,
                    success=False,
                    error=f"Node not ready: {self.state.value}",
                    correlation_id=input_data.correlation_id
                )
            
            self.state = NodeState.PROCESSING
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Input validation failed")
            
            # Log processing start
            logger.debug(LogChannel.DATA_FLOW, f"Processing in node: {self.metadata.name}",
                        correlation_id=input_data.correlation_id,
                        input_size=sys.getsizeof(input_data.data))
            
            # Process data
            with TimedOperation(f"node_{self.metadata.name}_process"):
                output = self.process(input_data)
            
            # Validate output
            if not self.validate_output(output):
                raise ValueError("Output validation failed")
            
            # Update output metadata
            output.node_id = self.metadata.id
            output.correlation_id = input_data.correlation_id
            output.processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update(output.processing_time, True)
            
            return output
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update(processing_time, False)
            
            logger.error(LogChannel.ERRORS, f"Node processing failed: {self.metadata.name}",
                        error=str(e), correlation_id=input_data.correlation_id)
            
            return NodeOutput(
                data=None,
                node_id=self.metadata.id,
                success=False,
                error=str(e),
                correlation_id=input_data.correlation_id,
                processing_time=processing_time
            )
        
        finally:
            with self._lock:
                self.state = NodeState.READY
    
    async def execute_async(self, input_data: NodeInput) -> NodeOutput:
        """Asynchronous execution wrapper"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.execute, input_data)
    
    def connect(self, target_node: 'BaseNode', connection_type: str = "default"):
        """Connect this node to another node"""
        self.connections[f"{connection_type}:{target_node.metadata.id}"] = target_node
        logger.info(LogChannel.SYNAPSE, f"Connected nodes: {self.metadata.name} -> {target_node.metadata.name}",
                   connection_type=connection_type)
    
    def disconnect(self, target_node: 'BaseNode', connection_type: str = "default"):
        """Disconnect from another node"""
        key = f"{connection_type}:{target_node.metadata.id}"
        if key in self.connections:
            del self.connections[key]
            logger.info(LogChannel.SYNAPSE, f"Disconnected nodes: {self.metadata.name} -X-> {target_node.metadata.name}")
    
    def get_connected_nodes(self, connection_type: Optional[str] = None) -> List['BaseNode']:
        """Get all connected nodes"""
        if connection_type:
            prefix = f"{connection_type}:"
            return [node for key, node in self.connections.items() if key.startswith(prefix)]
        return list(self.connections.values())
    
    def set_parameter(self, name: str, value: Any):
        """Set a node parameter"""
        with self._lock:
            old_value = self.parameters.get(name)
            self.parameters[name] = value
            logger.info(LogChannel.MODULE, f"Parameter updated: {self.metadata.name}.{name}",
                       old_value=old_value, new_value=value)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a node parameter"""
        return self.parameters.get(name, default)
    
    def register_tunable_parameter(self, name: str, param_type: type, 
                                 min_value: Any = None, max_value: Any = None,
                                 description: str = ""):
        """Register a parameter that can be tuned by the neural network"""
        self.tunable_params[name] = {
            'type': param_type,
            'min': min_value,
            'max': max_value,
            'description': description,
            'current_value': self.parameters.get(name)
        }
    
    def tune_parameter(self, name: str, value: Any) -> bool:
        """Tune a registered parameter"""
        if name not in self.tunable_params:
            logger.warning(LogChannel.MODULE, f"Attempted to tune unregistered parameter: {name}")
            return False
        
        param_info = self.tunable_params[name]
        
        # Validate type
        if not isinstance(value, param_info['type']):
            logger.error(LogChannel.MODULE, f"Invalid type for parameter {name}",
                        expected=param_info['type'].__name__, 
                        got=type(value).__name__)
            return False
        
        # Validate range
        if param_info['min'] is not None and value < param_info['min']:
            value = param_info['min']
        if param_info['max'] is not None and value > param_info['max']:
            value = param_info['max']
        
        # Update parameter
        self.set_parameter(name, value)
        self.tunable_params[name]['current_value'] = value
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get node performance metrics"""
        return {
            'node_id': self.metadata.id,
            'node_name': self.metadata.name,
            'state': self.state.value,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'average_processing_time': self.metrics.average_processing_time,
                'max_processing_time': self.metrics.max_processing_time,
                'min_processing_time': self.metrics.min_processing_time,
                'error_rate': self.metrics.error_rate,
                'throughput': self.metrics.throughput
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self._lock:
            self.metrics = NodeMetrics()
            logger.info(LogChannel.MODULE, f"Metrics reset for node: {self.metadata.name}")
    
    def pause(self):
        """Pause node processing"""
        with self._lock:
            if self.state == NodeState.READY:
                self.state = NodeState.PAUSED
                logger.info(LogChannel.MODULE, f"Node paused: {self.metadata.name}")
    
    def resume(self):
        """Resume node processing"""
        with self._lock:
            if self.state == NodeState.PAUSED:
                self.state = NodeState.READY
                logger.info(LogChannel.MODULE, f"Node resumed: {self.metadata.name}")
    
    def shutdown(self):
        """Shutdown the node gracefully"""
        logger.info(LogChannel.MODULE, f"Shutting down node: {self.metadata.name}")
        
        with self._lock:
            self.state = NodeState.SHUTDOWN
            self._shutdown_event.set()
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True)
        
        # Custom cleanup
        self._custom_cleanup()
        
        logger.info(LogChannel.MODULE, f"Node shutdown complete: {self.metadata.name}")
    
    def _custom_cleanup(self):
        """Custom cleanup - can be overridden by subclasses"""
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.metadata.name}, id={self.metadata.id}, state={self.state.value})>"


class AtomicNode(BaseNode):
    """Node that wraps a single function"""
    
    def __init__(self, function: Callable, metadata: Optional[NodeMetadata] = None):
        """Initialize atomic node with a function"""
        if metadata is None:
            metadata = NodeMetadata(
                name=function.__name__,
                type=NodeType.ATOMIC,
                description=function.__doc__ or ""
            )
        
        super().__init__(metadata)
        self.function = function
        self._analyze_function()
    
    def _analyze_function(self):
        """Analyze the wrapped function for metadata"""
        sig = inspect.signature(self.function)
        self.metadata.parameters['function_signature'] = str(sig)
        self.metadata.parameters['function_params'] = list(sig.parameters.keys())
    
    def process(self, input_data: NodeInput) -> NodeOutput:
        """Process by calling the wrapped function"""
        try:
            # Extract function arguments from input data
            if isinstance(input_data.data, dict):
                result = self.function(**input_data.data)
            elif isinstance(input_data.data, (list, tuple)):
                result = self.function(*input_data.data)
            else:
                result = self.function(input_data.data)
            
            return NodeOutput(
                data=result,
                success=True
            )
        except Exception as e:
            return NodeOutput(
                data=None,
                success=False,
                error=str(e)
            )
    
    def validate_input(self, input_data: NodeInput) -> bool:
        """Basic input validation"""
        return input_data.data is not None
    
    def validate_output(self, output_data: NodeOutput) -> bool:
        """Basic output validation"""
        return True


class CompositeNode(BaseNode):
    """Node that contains multiple sub-nodes"""
    
    def __init__(self, metadata: Optional[NodeMetadata] = None):
        """Initialize composite node"""
        if metadata is None:
            metadata = NodeMetadata(type=NodeType.COMPOSITE)
        
        super().__init__(metadata)
        self.sub_nodes: Dict[str, BaseNode] = {}
        self.execution_order: List[str] = []
    
    def add_node(self, node: BaseNode, name: Optional[str] = None):
        """Add a sub-node"""
        name = name or node.metadata.name
        self.sub_nodes[name] = node
        self.execution_order.append(name)
        logger.info(LogChannel.MODULE, f"Added sub-node '{name}' to composite node '{self.metadata.name}'")
    
    def remove_node(self, name: str):
        """Remove a sub-node"""
        if name in self.sub_nodes:
            del self.sub_nodes[name]
            self.execution_order.remove(name)
            logger.info(LogChannel.MODULE, f"Removed sub-node '{name}' from composite node '{self.metadata.name}'")
    
    def set_execution_order(self, order: List[str]):
        """Set the execution order of sub-nodes"""
        # Validate that all names exist
        for name in order:
            if name not in self.sub_nodes:
                raise ValueError(f"Unknown sub-node: {name}")
        self.execution_order = order
    
    def process(self, input_data: NodeInput) -> NodeOutput:
        """Process by executing sub-nodes in order"""
        results = {}
        current_data = input_data.data
        
        for node_name in self.execution_order:
            node = self.sub_nodes[node_name]
            
            # Create input for sub-node
            sub_input = NodeInput(
                data=current_data,
                metadata=input_data.metadata.copy(),
                source_node=self.metadata.id,
                correlation_id=input_data.correlation_id
            )
            
            # Execute sub-node
            sub_output = node.execute(sub_input)
            
            if not sub_output.success:
                return NodeOutput(
                    data=results,
                    success=False,
                    error=f"Sub-node '{node_name}' failed: {sub_output.error}"
                )
            
            results[node_name] = sub_output.data
            current_data = sub_output.data
        
        return NodeOutput(
            data=results,
            success=True
        )
    
    def validate_input(self, input_data: NodeInput) -> bool:
        """Validate input for first sub-node"""
        if self.execution_order and self.execution_order[0] in self.sub_nodes:
            return self.sub_nodes[self.execution_order[0]].validate_input(input_data)
        return True
    
    def validate_output(self, output_data: NodeOutput) -> bool:
        """Validate output from last sub-node"""
        return output_data.success


if __name__ == "__main__":
    # Example usage
    print("🧠 Neural Network Node Architecture Demo")
    print("=" * 50)
    
    # Create a simple atomic node
    def square(x):
        """Square a number"""
        return x ** 2
    
    square_node = AtomicNode(square, NodeMetadata(name="square", description="Squares input"))
    square_node.initialize()
    
    # Test the node
    input_data = NodeInput(data=5)
    output = square_node.execute(input_data)
    
    print(f"\nSquare node result: {output.data}")
    print(f"Processing time: {output.processing_time:.4f}s")
    print(f"\nNode metrics: {json.dumps(square_node.get_metrics(), indent=2)}")