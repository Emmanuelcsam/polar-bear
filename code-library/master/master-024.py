#!/usr/bin/env python3
"""
Synaptic Connection Framework
Advanced inter-node communication system for the neural network
"""

import asyncio
import threading
import queue
import time
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    import zmq
except ImportError:
    zmq = None
import pickle
import weakref
from concurrent.futures import Future, ThreadPoolExecutor

from .logger import logger, LogChannel, TimedOperation
from .node_base import BaseNode, NodeInput, NodeOutput


class ConnectionType(Enum):
    """Types of synaptic connections"""
    SYNCHRONOUS = "synchronous"        # Direct, blocking connection
    ASYNCHRONOUS = "asynchronous"      # Non-blocking, event-driven
    STREAMING = "streaming"            # Continuous data flow
    BROADCAST = "broadcast"            # One-to-many
    AGGREGATION = "aggregation"        # Many-to-one
    BIDIRECTIONAL = "bidirectional"    # Full duplex
    PUBSUB = "pubsub"                 # Publish-subscribe pattern


class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CLOSING = "closing"


@dataclass
class Message:
    """Message structure for inter-node communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    destination: str = ""
    payload: Any = None
    headers: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    ttl: Optional[float] = None  # Time to live
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        return pickle.loads(data)


@dataclass
class ConnectionMetrics:
    """Metrics for a synaptic connection"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    last_activity: float = field(default_factory=time.time)
    
    def update_sent(self, message_size: int, latency: float):
        """Update metrics for sent message"""
        self.messages_sent += 1
        self.bytes_sent += message_size
        self.latency_ms = (self.latency_ms * 0.9) + (latency * 1000 * 0.1)  # Exponential moving average
        self.last_activity = time.time()
    
    def update_received(self, message_size: int):
        """Update metrics for received message"""
        self.messages_received += 1
        self.bytes_received += message_size
        self.last_activity = time.time()


class BaseSynapse(ABC):
    """Abstract base class for synaptic connections"""
    
    def __init__(self, source_node: BaseNode, connection_type: ConnectionType):
        """Initialize synapse"""
        self.source_node = weakref.ref(source_node)
        self.connection_type = connection_type
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self.filters: List[Callable[[Message], bool]] = []
        self.transformers: List[Callable[[Message], Message]] = []
        self._lock = threading.RLock()
        
        logger.info(LogChannel.SYNAPSE, f"Creating {connection_type.value} synapse from {source_node.metadata.name}")
    
    @abstractmethod
    def connect(self, target: Union[BaseNode, str, List[BaseNode]]):
        """Establish connection"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection"""
        pass
    
    @abstractmethod
    def send(self, data: Any, **kwargs) -> Optional[Any]:
        """Send data through the synapse"""
        pass
    
    @abstractmethod
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive data from the synapse"""
        pass
    
    def add_filter(self, filter_func: Callable[[Message], bool]):
        """Add a message filter"""
        self.filters.append(filter_func)
    
    def add_transformer(self, transformer_func: Callable[[Message], Message]):
        """Add a message transformer"""
        self.transformers.append(transformer_func)
    
    def _apply_filters(self, message: Message) -> bool:
        """Apply all filters to a message"""
        for filter_func in self.filters:
            if not filter_func(message):
                return False
        return True
    
    def _apply_transformers(self, message: Message) -> Message:
        """Apply all transformers to a message"""
        for transformer in self.transformers:
            message = transformer(message)
        return message
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics"""
        return {
            'type': self.connection_type.value,
            'state': self.state.value,
            'messages_sent': self.metrics.messages_sent,
            'messages_received': self.metrics.messages_received,
            'bytes_sent': self.metrics.bytes_sent,
            'bytes_received': self.metrics.bytes_received,
            'errors': self.metrics.errors,
            'latency_ms': self.metrics.latency_ms,
            'throughput_mbps': self.metrics.throughput_mbps
        }


class SynchronousSynapse(BaseSynapse):
    """Direct, blocking connection between nodes"""
    
    def __init__(self, source_node: BaseNode):
        super().__init__(source_node, ConnectionType.SYNCHRONOUS)
        self.target_node: Optional[weakref.ref] = None
    
    def connect(self, target: BaseNode):
        """Connect to target node"""
        with self._lock:
            self.target_node = weakref.ref(target)
            self.state = ConnectionState.CONNECTED
            
            # Register connection in source node
            source = self.source_node()
            if source:
                source.connect(target, self.connection_type.value)
            
            logger.info(LogChannel.SYNAPSE, 
                       f"Synchronous connection established: {source.metadata.name} -> {target.metadata.name}")
    
    def disconnect(self):
        """Disconnect from target"""
        with self._lock:
            if self.target_node:
                target = self.target_node()
                source = self.source_node()
                
                if source and target:
                    source.disconnect(target, self.connection_type.value)
                
                self.target_node = None
            
            self.state = ConnectionState.DISCONNECTED
    
    def send(self, data: Any, timeout: Optional[float] = None, **kwargs) -> Optional[NodeOutput]:
        """Send data and wait for response"""
        if self.state != ConnectionState.CONNECTED:
            logger.error(LogChannel.SYNAPSE, "Cannot send: synapse not connected")
            return None
        
        target = self.target_node()
        if not target:
            logger.error(LogChannel.SYNAPSE, "Target node no longer exists")
            return None
        
        # Create message
        source = self.source_node()
        message = Message(
            source=source.metadata.id if source else "unknown",
            destination=target.metadata.id,
            payload=data,
            headers=kwargs
        )
        
        # Apply transformers
        message = self._apply_transformers(message)
        
        # Apply filters
        if not self._apply_filters(message):
            logger.debug(LogChannel.SYNAPSE, "Message filtered out")
            return None
        
        start_time = time.time()
        
        try:
            # Create node input
            node_input = NodeInput(
                data=message.payload,
                metadata=message.headers,
                source_node=message.source,
                correlation_id=message.correlation_id or message.id
            )
            
            # Execute on target node
            with TimedOperation(f"synapse_send_{target.metadata.name}"):
                result = target.execute(node_input)
            
            # Update metrics
            latency = time.time() - start_time
            message_size = len(message.to_bytes())
            self.metrics.update_sent(message_size, latency)
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(LogChannel.SYNAPSE, f"Error sending through synapse: {str(e)}")
            return None
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Not applicable for synchronous connections"""
        raise NotImplementedError("Synchronous synapses don't support receive()")


class AsynchronousSynapse(BaseSynapse):
    """Non-blocking, event-driven connection"""
    
    def __init__(self, source_node: BaseNode, max_queue_size: int = 1000):
        super().__init__(source_node, ConnectionType.ASYNCHRONOUS)
        self.target_node: Optional[weakref.ref] = None
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.response_futures: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._worker_thread = None
        self._stop_event = threading.Event()
    
    def connect(self, target: BaseNode):
        """Connect to target node"""
        with self._lock:
            self.target_node = weakref.ref(target)
            self.state = ConnectionState.CONNECTED
            
            # Start worker thread
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._process_messages, daemon=True)
            self._worker_thread.start()
            
            # Register connection
            source = self.source_node()
            if source:
                source.connect(target, self.connection_type.value)
            
            logger.info(LogChannel.SYNAPSE, 
                       f"Asynchronous connection established: {source.metadata.name} -> {target.metadata.name}")
    
    def disconnect(self):
        """Disconnect from target"""
        with self._lock:
            self._stop_event.set()
            
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)
            
            if self.target_node:
                target = self.target_node()
                source = self.source_node()
                
                if source and target:
                    source.disconnect(target, self.connection_type.value)
                
                self.target_node = None
            
            self.executor.shutdown(wait=True)
            self.state = ConnectionState.DISCONNECTED
    
    def send(self, data: Any, callback: Optional[Callable[[NodeOutput], None]] = None, **kwargs) -> str:
        """Send data asynchronously"""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Synapse not connected")
        
        source = self.source_node()
        target = self.target_node()
        
        if not target:
            raise RuntimeError("Target node no longer exists")
        
        # Create message
        message = Message(
            source=source.metadata.id if source else "unknown",
            destination=target.metadata.id,
            payload=data,
            headers=kwargs
        )
        
        # Create future for response
        future = Future()
        self.response_futures[message.id] = future
        
        # Add callback if provided
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        
        # Queue message
        try:
            self.message_queue.put_nowait((message, future))
        except queue.Full:
            del self.response_futures[message.id]
            raise RuntimeError("Message queue full")
        
        return message.id
    
    def _process_messages(self):
        """Worker thread to process messages"""
        while not self._stop_event.is_set():
            try:
                message, future = self.message_queue.get(timeout=1)
                
                # Apply transformers and filters
                message = self._apply_transformers(message)
                if not self._apply_filters(message):
                    future.set_result(None)
                    continue
                
                target = self.target_node()
                if not target:
                    future.set_exception(RuntimeError("Target node no longer exists"))
                    continue
                
                # Process message
                start_time = time.time()
                
                try:
                    node_input = NodeInput(
                        data=message.payload,
                        metadata=message.headers,
                        source_node=message.source,
                        correlation_id=message.id
                    )
                    
                    result = target.execute(node_input)
                    
                    # Update metrics
                    latency = time.time() - start_time
                    message_size = len(message.to_bytes())
                    self.metrics.update_sent(message_size, latency)
                    
                    future.set_result(result)
                    
                except Exception as e:
                    self.metrics.errors += 1
                    future.set_exception(e)
                    logger.error(LogChannel.SYNAPSE, f"Error processing message: {str(e)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(LogChannel.SYNAPSE, f"Worker thread error: {str(e)}")
    
    def get_result(self, message_id: str, timeout: Optional[float] = None) -> NodeOutput:
        """Get result for a sent message"""
        if message_id not in self.response_futures:
            raise ValueError(f"Unknown message ID: {message_id}")
        
        future = self.response_futures[message_id]
        result = future.result(timeout=timeout)
        
        # Clean up
        del self.response_futures[message_id]
        
        return result
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Not applicable for this implementation"""
        raise NotImplementedError("Use get_result() to retrieve responses")


class StreamingSynapse(BaseSynapse):
    """Continuous data flow connection with backpressure handling"""
    
    def __init__(self, source_node: BaseNode, buffer_size: int = 100):
        super().__init__(source_node, ConnectionType.STREAMING)
        self.target_node: Optional[weakref.ref] = None
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.stream_active = False
        self._consumer_task = None
        self._producer_task = None
    
    def connect(self, target: BaseNode):
        """Connect to target node"""
        with self._lock:
            self.target_node = weakref.ref(target)
            self.state = ConnectionState.CONNECTED
            self.stream_active = True
            
            # Register connection
            source = self.source_node()
            if source:
                source.connect(target, self.connection_type.value)
            
            logger.info(LogChannel.SYNAPSE, 
                       f"Streaming connection established: {source.metadata.name} -> {target.metadata.name}")
    
    def disconnect(self):
        """Disconnect stream"""
        with self._lock:
            self.stream_active = False
            
            if self._consumer_task:
                self._consumer_task.cancel()
            if self._producer_task:
                self._producer_task.cancel()
            
            if self.target_node:
                target = self.target_node()
                source = self.source_node()
                
                if source and target:
                    source.disconnect(target, self.connection_type.value)
                
                self.target_node = None
            
            self.state = ConnectionState.DISCONNECTED
    
    async def send_stream(self, data_generator: Callable[[], Any]):
        """Send data stream"""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Synapse not connected")
        
        async def producer():
            while self.stream_active:
                try:
                    data = await asyncio.get_event_loop().run_in_executor(None, data_generator)
                    if data is None:
                        break
                    
                    message = Message(
                        source=self.source_node().metadata.id,
                        destination=self.target_node().metadata.id,
                        payload=data
                    )
                    
                    await self.buffer.put(message)
                    
                except Exception as e:
                    logger.error(LogChannel.SYNAPSE, f"Stream producer error: {str(e)}")
                    break
        
        self._producer_task = asyncio.create_task(producer())
    
    async def consume_stream(self, handler: Callable[[Any], None]):
        """Consume data stream"""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Synapse not connected")
        
        async def consumer():
            while self.stream_active:
                try:
                    message = await self.buffer.get()
                    
                    # Apply transformers and filters
                    message = self._apply_transformers(message)
                    if not self._apply_filters(message):
                        continue
                    
                    target = self.target_node()
                    if not target:
                        break
                    
                    # Process message
                    node_input = NodeInput(
                        data=message.payload,
                        metadata=message.headers,
                        source_node=message.source,
                        correlation_id=message.id
                    )
                    
                    result = await target.execute_async(node_input)
                    
                    # Call handler with result
                    await asyncio.get_event_loop().run_in_executor(None, handler, result)
                    
                    # Update metrics
                    self.metrics.update_received(len(message.to_bytes()))
                    
                except Exception as e:
                    logger.error(LogChannel.SYNAPSE, f"Stream consumer error: {str(e)}")
        
        self._consumer_task = asyncio.create_task(consumer())
    
    def send(self, data: Any, **kwargs) -> None:
        """Add data to stream buffer"""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Synapse not connected")
        
        message = Message(
            source=self.source_node().metadata.id,
            destination=self.target_node().metadata.id,
            payload=data,
            headers=kwargs
        )
        
        try:
            self.buffer.put_nowait(message)
        except asyncio.QueueFull:
            raise RuntimeError("Stream buffer full - backpressure activated")
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Not implemented for streaming - use consume_stream instead"""
        raise NotImplementedError("Use consume_stream() for streaming connections")


class BroadcastSynapse(BaseSynapse):
    """One-to-many broadcast connection"""
    
    def __init__(self, source_node: BaseNode):
        super().__init__(source_node, ConnectionType.BROADCAST)
        self.subscribers: Set[weakref.ref] = set()
        self.topic = f"broadcast_{source_node.metadata.id}"
        if zmq:
            self.context = zmq.Context()
            self.publisher = None
            self._bind_address = None
        else:
            self.context = None
            self.publisher = None
            self._bind_address = None
    
    def connect(self, targets: Union[BaseNode, List[BaseNode]]):
        """Connect to multiple target nodes"""
        if isinstance(targets, BaseNode):
            targets = [targets]
        
        with self._lock:
            # Create publisher socket if not exists (only if zmq available)
            if zmq and self.publisher is None:
                self.publisher = self.context.socket(zmq.PUB)
                self._bind_address = f"inproc://{self.topic}"
                self.publisher.bind(self._bind_address)
                time.sleep(0.1)  # Allow socket to bind
            
            # Add subscribers
            source = self.source_node()
            for target in targets:
                self.subscribers.add(weakref.ref(target))
                if source:
                    source.connect(target, self.connection_type.value)
                
                logger.info(LogChannel.SYNAPSE, 
                           f"Broadcast subscriber added: {source.metadata.name} -> {target.metadata.name}")
            
            self.state = ConnectionState.CONNECTED
    
    def disconnect(self):
        """Disconnect all subscribers"""
        with self._lock:
            source = self.source_node()
            
            # Disconnect all subscribers
            for sub_ref in self.subscribers:
                target = sub_ref()
                if source and target:
                    source.disconnect(target, self.connection_type.value)
            
            self.subscribers.clear()
            
            # Close publisher (if zmq available)
            if zmq and self.publisher:
                self.publisher.close()
                self.publisher = None
            
            self.state = ConnectionState.DISCONNECTED
    
    def send(self, data: Any, topic_filter: Optional[str] = None, **kwargs) -> int:
        """Broadcast data to all subscribers"""
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError("Synapse not connected")
        
        source = self.source_node()
        message = Message(
            source=source.metadata.id if source else "unknown",
            destination="*",  # Broadcast
            payload=data,
            headers=kwargs
        )
        
        if topic_filter:
            message.headers['topic'] = topic_filter
        
        # Apply transformers
        message = self._apply_transformers(message)
        
        # Serialize and send (only if zmq available)
        if zmq and self.publisher:
            topic = topic_filter.encode() if topic_filter else b""
            self.publisher.send_multipart([topic, message.to_bytes()])
        
        # Update metrics
        message_size = len(message.to_bytes())
        self.metrics.update_sent(message_size, 0)
        
        # Process locally for connected nodes
        sent_count = 0
        for sub_ref in list(self.subscribers):
            target = sub_ref()
            if not target:
                self.subscribers.remove(sub_ref)
                continue
            
            # Apply filters
            if not self._apply_filters(message):
                continue
            
            # Create input and execute
            node_input = NodeInput(
                data=message.payload,
                metadata=message.headers,
                source_node=message.source,
                correlation_id=message.id
            )
            
            # Execute asynchronously
            threading.Thread(target=target.execute, args=(node_input,), daemon=True).start()
            sent_count += 1
        
        return sent_count
    
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Not applicable for broadcast publisher"""
        raise NotImplementedError("Broadcast synapses are send-only")


class SynapticRouter:
    """Advanced routing system for synaptic connections"""
    
    def __init__(self):
        """Initialize router"""
        self.routes: Dict[str, List[BaseSynapse]] = {}
        self.load_balancers: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Tuple[int, float]] = {}
        self._lock = threading.RLock()
        
        logger.info(LogChannel.SYNAPSE, "Synaptic router initialized")
    
    def add_route(self, pattern: str, synapse: BaseSynapse):
        """Add a routing rule"""
        with self._lock:
            if pattern not in self.routes:
                self.routes[pattern] = []
                self.load_balancers[pattern] = 0
            
            self.routes[pattern].append(synapse)
            logger.info(LogChannel.SYNAPSE, f"Route added: {pattern}")
    
    def remove_route(self, pattern: str, synapse: BaseSynapse):
        """Remove a routing rule"""
        with self._lock:
            if pattern in self.routes and synapse in self.routes[pattern]:
                self.routes[pattern].remove(synapse)
                if not self.routes[pattern]:
                    del self.routes[pattern]
                    del self.load_balancers[pattern]
    
    def route(self, message: Message) -> Optional[BaseSynapse]:
        """Route a message to appropriate synapse"""
        with self._lock:
            # Find matching route
            for pattern, synapses in self.routes.items():
                if self._match_pattern(pattern, message):
                    # Check circuit breaker
                    if self._is_circuit_open(pattern):
                        logger.warning(LogChannel.SYNAPSE, f"Circuit breaker open for route: {pattern}")
                        continue
                    
                    # Load balance
                    idx = self.load_balancers[pattern] % len(synapses)
                    self.load_balancers[pattern] = (idx + 1) % len(synapses)
                    
                    return synapses[idx]
            
            return None
    
    def _match_pattern(self, pattern: str, message: Message) -> bool:
        """Match message against pattern"""
        # Simple pattern matching - can be enhanced
        if pattern == "*":
            return True
        
        if pattern.startswith("dest:"):
            return pattern[5:] == message.destination
        
        if pattern.startswith("source:"):
            return pattern[7:] == message.source
        
        if pattern.startswith("header:"):
            key, value = pattern[7:].split("=", 1)
            return message.headers.get(key) == value
        
        return False
    
    def _is_circuit_open(self, pattern: str) -> bool:
        """Check if circuit breaker is open"""
        if pattern not in self.circuit_breakers:
            return False
        
        failures, last_check = self.circuit_breakers[pattern]
        
        # Reset after 60 seconds
        if time.time() - last_check > 60:
            del self.circuit_breakers[pattern]
            return False
        
        # Open circuit after 5 failures
        return failures >= 5
    
    def report_failure(self, pattern: str):
        """Report a routing failure"""
        with self._lock:
            if pattern not in self.circuit_breakers:
                self.circuit_breakers[pattern] = (0, time.time())
            
            failures, _ = self.circuit_breakers[pattern]
            self.circuit_breakers[pattern] = (failures + 1, time.time())


if __name__ == "__main__":
    # Demo synaptic connections
    print("🧠 Synaptic Connection Framework Demo")
    print("=" * 50)
    
    # Create test nodes
    from .node_base import AtomicNode, NodeMetadata
    
    def process_data(x):
        return x * 2
    
    source = AtomicNode(lambda x: x, NodeMetadata(name="source"))
    target = AtomicNode(process_data, NodeMetadata(name="target"))
    
    # Initialize nodes
    source.initialize()
    target.initialize()
    
    # Test synchronous connection
    print("\n1. Testing Synchronous Connection:")
    sync_synapse = SynchronousSynapse(source)
    sync_synapse.connect(target)
    
    result = sync_synapse.send(5)
    print(f"   Result: {result.data if result else 'None'}")
    print(f"   Metrics: {sync_synapse.get_metrics()}")
    
    # Test asynchronous connection
    print("\n2. Testing Asynchronous Connection:")
    async_synapse = AsynchronousSynapse(source)
    async_synapse.connect(target)
    
    message_id = async_synapse.send(10)
    result = async_synapse.get_result(message_id, timeout=5)
    print(f"   Result: {result.data if result else 'None'}")
    
    # Cleanup
    sync_synapse.disconnect()
    async_synapse.disconnect()
    
    print("\n✅ Synaptic connection demo complete!")