# 🧠 Neural Network Integration System

A sophisticated Python framework that transforms disparate scripts and modules into a unified neural network architecture. This system automatically analyzes, wraps, and connects existing Python code into neural network nodes with intelligent parameter tuning and real-time processing capabilities.

## 🌟 Features

### Core Capabilities
- **Automatic Module Analysis**: Scans and analyzes Python files to extract functions, classes, and dependencies
- **Dynamic Node Creation**: Wraps functions and modules into neural network nodes automatically
- **Intelligent Dependency Management**: Auto-detects and installs required packages
- **Synaptic Connections**: Multiple connection types (synchronous, asynchronous, streaming, broadcast)
- **Parameter Tuning**: Multiple optimization strategies (Bayesian, genetic, gradient-based)
- **Comprehensive Logging**: Multi-channel logging with file, console, and database outputs
- **Interactive Configuration**: User-friendly setup wizard with auto-detection

### Advanced Features
- **Real-time Processing**: Streaming data through connected nodes
- **Distributed Computing**: Process and thread pool executors for parallel execution
- **Fault Tolerance**: Circuit breakers, automatic retries, and graceful degradation
- **Performance Monitoring**: Detailed metrics for each node and connection
- **Hot-swappable Nodes**: Add/remove nodes without system restart
- **State Persistence**: Save and restore network configurations

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
cd /home/jarvis/Documents/GitHub/polar-bear/modules/neural-network-integration
```

2. Run the dependency installer:
```bash
python core/dependency_manager.py
```

3. Run the main neural network:
```bash
python neural_network.py
```

### Basic Usage

```python
from neural_network import NeuralNetwork
from core.node_base import AtomicNode, NodeMetadata

# Create a neural network
network = NeuralNetwork("MyNetwork")

# Define a simple function
def process_data(x):
    return x * 2

# Create a node from the function
node = AtomicNode(process_data, NodeMetadata(name="processor"))
node.initialize()

# Add to network
network.add_node(node)

# Process data
result = network.process(5, "processor")
print(result.data)  # Output: 10
```

## 📁 Project Structure

```
neural-network-integration/
├── core/                           # Core framework modules
│   ├── node_base.py               # Base node classes and interfaces
│   ├── synapse.py                 # Connection framework
│   ├── logger.py                  # Advanced logging system
│   ├── config_manager.py          # Configuration management
│   ├── dependency_manager.py      # Automatic dependency handling
│   ├── module_analyzer.py         # Code analysis and wrapping
│   └── parameter_tuner.py         # Parameter optimization
├── nodes/                         # Custom node implementations
├── synapses/                      # Custom connection types
├── configs/                       # Configuration profiles
├── logs/                          # Log files and databases
├── neural_network.py              # Main orchestration system
├── demo.py                        # Demonstration scripts
└── README.md                      # This file
```

## 🔧 Configuration

### Interactive Setup
Run the configuration wizard:
```python
from core.config_manager import config_manager, ConfigLevel

# Run interactive setup (beginner-friendly)
config_manager.interactive_setup(ConfigLevel.BEGINNER)

# Or for advanced users
config_manager.interactive_setup(ConfigLevel.EXPERT)
```

### Configuration Options

#### System Settings
- `system.cpu_cores`: Number of CPU cores to use
- `system.memory_limit`: Maximum memory usage (MB)
- `system.use_gpu`: Enable GPU acceleration

#### Neural Network Settings
- `neural.batch_size`: Default batch size
- `neural.learning_rate`: Default learning rate
- `neural.auto_tune`: Enable automatic parameter tuning

#### Logging Settings
- `logging.level`: Verbosity (TRACE, DEBUG, INFO, WARNING, ERROR)
- `logging.file_enabled`: Enable file logging
- `logging.max_file_size`: Maximum log file size (MB)

## 🏗️ Architecture

### Node Types

1. **Atomic Nodes**: Wrap single functions
2. **Composite Nodes**: Groups of related nodes
3. **Meta Nodes**: Higher-order orchestrators
4. **Gateway Nodes**: External system interfaces
5. **Transform Nodes**: Data transformation
6. **Validator Nodes**: Input/output validation

### Connection Types

1. **Synchronous**: Direct, blocking connections
2. **Asynchronous**: Non-blocking with futures
3. **Streaming**: Continuous data flow with backpressure
4. **Broadcast**: One-to-many distribution
5. **Aggregation**: Many-to-one collection
6. **Bidirectional**: Full-duplex communication

### Parameter Tuning Strategies

1. **Grid Search**: Exhaustive parameter search
2. **Random Search**: Randomized exploration
3. **Bayesian Optimization**: Smart sampling with Optuna
4. **Genetic Algorithm**: Evolutionary optimization
5. **Gradient-Based**: Numerical optimization
6. **Adaptive**: Automatic strategy switching

## 📊 Monitoring & Analytics

### Real-time Metrics
- Node processing times
- Connection latencies
- Error rates and types
- Throughput measurements
- Resource utilization

### Performance Analysis
```python
# Get network statistics
stats = network.get_network_stats()
print(f"Total calls: {stats['performance']['total_calls']}")
print(f"Average latency: {stats['performance']['average_latency']}ms")

# Get node-specific metrics
node_metrics = network.get_node("processor").get_metrics()
```

## 🔌 Integration Examples

### Analyzing Existing Modules

```python
from core.module_analyzer import ModuleAnalyzer, NodeFactory

# Analyze a directory
analyzer = ModuleAnalyzer()
modules = analyzer.analyze_directory("/path/to/modules")

# Create nodes automatically
factory = NodeFactory(analyzer)
nodes = factory.create_all_nodes()

# Connect based on dependencies
factory.connect_nodes_by_dependencies()
```

### Creating Custom Nodes

```python
from core.node_base import BaseNode, NodeInput, NodeOutput

class CustomNode(BaseNode):
    def process(self, input_data: NodeInput) -> NodeOutput:
        # Your processing logic
        result = custom_processing(input_data.data)
        return NodeOutput(data=result, success=True)
    
    def validate_input(self, input_data: NodeInput) -> bool:
        # Input validation
        return True
    
    def validate_output(self, output_data: NodeOutput) -> bool:
        # Output validation
        return True
```

### Parameter Tuning

```python
from core.parameter_tuner import ParameterTuner, TuningStrategy

# Register tunable parameters
node.register_tunable_parameter("threshold", float, 0.0, 1.0)
node.register_tunable_parameter("scale", float, 0.1, 10.0)

# Run tuning
tuner = ParameterTuner()
tuner.register_node(node)
best_result = tuner.tune(TuningStrategy.BAYESIAN, max_iterations=100)

# Apply best parameters
tuner.apply_best_parameters()
```

## 🛠️ Advanced Usage

### Streaming Pipeline

```python
# Create streaming synapse
from core.synapse import StreamingSynapse

stream = StreamingSynapse(source_node)
stream.connect(target_node)

# Stream data
async def data_generator():
    while True:
        yield generate_data()

await stream.send_stream(data_generator)
```

### Broadcast System

```python
from core.synapse import BroadcastSynapse

# Create broadcast
broadcast = BroadcastSynapse(source_node)
broadcast.connect([node1, node2, node3])

# Send to all subscribers
broadcast.send(data, topic_filter="important")
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Run `python core/dependency_manager.py` to auto-install dependencies
2. **Node Initialization Failed**: Check logs in `logs/neural_network.log`
3. **Connection Errors**: Verify nodes are initialized before connecting
4. **Performance Issues**: Adjust `system.cpu_cores` and `system.memory_limit`

### Debug Mode

Enable detailed logging:
```python
from core.logger import logger, LogLevel
logger.min_level = LogLevel.TRACE
```

## 📈 Performance Tips

1. **Use appropriate node types**: Atomic for simple functions, Composite for pipelines
2. **Enable caching**: Set `performance.cache_enabled = True`
3. **Tune batch sizes**: Adjust `neural.batch_size` based on your hardware
4. **Monitor metrics**: Regular check `get_network_stats()` for bottlenecks
5. **Parallel processing**: Use Meta nodes for CPU-intensive operations

## 🤝 Contributing

When adding new features:

1. Follow the existing node interface patterns
2. Add comprehensive logging using the logger
3. Include parameter tuning support where applicable
4. Write unit tests for new functionality
5. Update documentation

## 📄 License

This project is part of the Polar Bear system for image analysis and defect detection.

## 🙏 Acknowledgments

Built with:
- Python 3.7+
- NumPy for numerical operations
- OpenCV for image processing
- PyTorch for deep learning
- Optuna for hyperparameter optimization
- ZMQ for high-performance messaging

---

For more information, run the demo:
```bash
python demo.py
```

Or start the interactive neural network:
```bash
python neural_network.py
```