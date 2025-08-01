# Pylance and Real-time Integration for Fiber CNN Project

This document provides a comprehensive guide for setting up Pylance integration and real-time monitoring for the Fiber CNN Quality Assurance system.

## 🚀 Quick Start

### 1. Setup Development Environment

```bash
# Run the complete setup script
python setup_dev_environment.py

# Or run individual components
python pylance_integration.py --analyze --generate-stubs --create-vscode-config --run-checks
```

### 2. Activate Virtual Environment

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Start Real-time Monitoring

```bash
# Start monitoring with visualization
python realtime_monitor.py --visualize

# Start monitoring without visualization
python realtime_monitor.py
```

## 📁 Project Structure

```
sauber/
├── .vscode/                    # VS Code configurations
│   ├── settings.json          # Pylance settings
│   ├── launch.json            # Debug configurations
│   ├── tasks.json             # Build tasks
│   └── extensions.json        # Recommended extensions
├── typings/                   # Type stubs
├── logs/                      # Monitoring logs
│   ├── monitoring/            # Real-time metrics
│   └── tensorboard/           # TensorBoard logs
├── scripts/                   # Development scripts
├── pylance_integration.py     # Pylance setup script
├── realtime_monitor.py        # Real-time monitoring
├── setup_dev_environment.py   # Environment setup
├── requirements.txt           # Main dependencies
├── requirements-dev.txt       # Development dependencies
└── pyproject.toml            # Project configuration
```

## 🔧 Pylance Integration

### Features

- **Type Checking**: Full type hint support with MyPy integration
- **Code Analysis**: Real-time error detection and suggestions
- **Auto-completion**: Intelligent code completion and import suggestions
- **Refactoring**: Safe code refactoring with type safety
- **Debugging**: Enhanced debugging with type information

### Configuration

The Pylance integration is automatically configured with:

- **Type Checking Mode**: Basic (can be set to strict for more rigorous checking)
- **Auto Import Completions**: Enabled for faster development
- **Inlay Hints**: Show function return types and variable types
- **Diagnostic Mode**: Workspace-wide analysis
- **Stub Path**: `./typings` for type stubs

### Usage

```bash
# Analyze project files
python pylance_integration.py --analyze

# Generate type stubs
python pylance_integration.py --generate-stubs

# Create VS Code configurations
python pylance_integration.py --create-vscode-config

# Run code quality checks
python pylance_integration.py --run-checks

# Complete setup
python pylance_integration.py
```

## 📊 Real-time Monitoring

### Features

- **Live Metrics**: Real-time training loss, accuracy, and system metrics
- **GPU Monitoring**: GPU utilization, memory usage, and temperature
- **System Resources**: CPU, memory, and disk usage monitoring
- **Visualization**: Interactive plots with matplotlib and tkinter
- **TensorBoard Integration**: Automatic logging to TensorBoard
- **Metrics Export**: JSON export of training metrics

### Usage

```bash
# Start monitoring with GUI visualization
python realtime_monitor.py --visualize

# Start monitoring without GUI
python realtime_monitor.py

# Specify custom log directory
python realtime_monitor.py --log-dir logs/custom_monitoring

# Start TensorBoard
tensorboard --logdir=logs/monitoring/tensorboard --port=6006
```

### Integration with Training

The monitoring system can be integrated with your training loops:

```python
from realtime_monitor import create_monitoring_hook, MetricsCollector

# Create metrics collector
metrics_collector = MetricsCollector("logs/monitoring")

# Create monitoring hook
monitoring_hook = create_monitoring_hook(metrics_collector)

# Use in training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... training code ...
        
        # Call monitoring hook
        monitoring_hook(
            epoch=epoch,
            batch=batch_idx,
            total_batches=len(train_loader),
            losses={'total': total_loss.item(), 'zone': zone_loss.item()},
            lr=optimizer.param_groups[0]['lr'],
            batch_time=batch_time,
            data_time=data_time
        )
```

## 🛠️ Development Tools

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **MyPy**: Type checking
- **pre-commit**: Git hooks

### VS Code Tasks

Available tasks in VS Code (Ctrl+Shift+P → "Tasks: Run Task"):

- **Install Dependencies**: Install all requirements
- **Run Tests**: Execute pytest tests
- **Lint Code**: Run flake8 linting
- **Type Check**: Run MyPy type checking
- **Format Code**: Run Black formatting
- **Sort Imports**: Run isort import sorting
- **Train Pure CNN**: Start training with pure CNN
- **Train Distributed CNN**: Start distributed training
- **Run Inference**: Execute inference script

### Debug Configurations

Pre-configured debug configurations:

1. **Debug Fiber CNN Pure**: Single GPU training
2. **Debug Fiber CNN Distributed**: Multi-GPU training
3. **Debug Inference**: Model inference
4. **Debug Version1 Main**: Version1 main script
5. **Debug Version2 Main**: Version2 main script
6. **Attach to Process**: Remote debugging

## 📈 Monitoring Dashboard

The real-time monitoring provides a comprehensive dashboard with:

### Training Metrics
- **Loss Plots**: Total, zone, defect, and quality losses
- **Learning Rate**: Current learning rate over time
- **Progress**: Training progress percentage
- **Batch Timing**: Batch and data loading times

### System Resources
- **CPU Utilization**: Real-time CPU usage
- **Memory Usage**: RAM utilization
- **GPU Utilization**: GPU compute usage
- **GPU Memory**: GPU memory usage and total

### Interactive Features
- **Live Updates**: Real-time plot updates
- **Save Metrics**: Export metrics to JSON
- **Status Display**: Current training status
- **Control Panel**: Manual metric saving

## 🔍 Code Analysis

### Type Coverage Analysis

The Pylance integration provides detailed type coverage analysis:

```bash
# Generate type coverage report
python pylance_integration.py --analyze --report type_coverage_report.json
```

### Code Quality Metrics

- **Type Coverage**: Percentage of code with type hints
- **Function Analysis**: Function signatures and return types
- **Class Analysis**: Class structure and inheritance
- **Import Analysis**: Import statements and dependencies
- **Issue Detection**: TODO, FIXME, and potential issues

## 🚀 Performance Optimization

### GPU Monitoring

Real-time GPU monitoring helps optimize training:

```python
# Monitor GPU memory usage
gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print(f"GPU Memory: {gpu_memory_used:.2f}GB / {gpu_memory_total:.2f}GB")
```

### Memory Profiling

```bash
# Profile memory usage
python -m memory_profiler your_script.py

# Profile line-by-line
python -m line_profiler your_script.py
```

## 📋 Development Workflow

### 1. Initial Setup

```bash
# Complete environment setup
python setup_dev_environment.py

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 2. Daily Development

```bash
# Start VS Code with project
code .

# Run code quality checks
python -m black .
python -m isort .
python -m flake8 .
python -m mypy .

# Run tests
python -m pytest test_*.py -v
```

### 3. Training with Monitoring

```bash
# Start monitoring
python realtime_monitor.py --visualize

# In another terminal, start training
python fiber_cnn_pure.py --epochs 10 --batch-size 8
```

### 4. Debugging

```bash
# Use VS Code debugger
# Set breakpoints and use F5 to start debugging

# Or use remote debugging
python -m debugpy --listen 5678 your_script.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Pylance not working**
   - Check Python interpreter path in VS Code
   - Verify virtual environment activation
   - Run `python pylance_integration.py --create-vscode-config`

2. **Monitoring not starting**
   - Check if tkinter is available: `python -c "import tkinter"`
   - Install missing dependencies: `pip install -r requirements-dev.txt`

3. **Type checking errors**
   - Run `python pylance_integration.py --generate-stubs`
   - Check MyPy configuration in `pyproject.toml`

4. **GPU monitoring issues**
   - Install GPUtil: `pip install GPUtil`
   - Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Performance Tips

1. **VS Code Performance**
   - Exclude large directories in settings
   - Use workspace-specific settings
   - Disable unnecessary extensions

2. **Monitoring Performance**
   - Reduce update frequency for large datasets
   - Use smaller window sizes for history
   - Export metrics periodically

3. **Training Performance**
   - Monitor GPU memory usage
   - Use mixed precision training
   - Optimize data loading with proper num_workers

## 📚 Additional Resources

### Documentation
- [Pylance Documentation](https://github.com/microsoft/pylance-release)
- [VS Code Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

### Tools
- [Black Code Formatter](https://black.readthedocs.io/)
- [isort Import Sorter](https://pycqa.github.io/isort/)
- [flake8 Linter](https://flake8.pycqa.org/)
- [pre-commit Hooks](https://pre-commit.com/)

### Monitoring
- [GPUtil GPU Monitoring](https://github.com/anderskm/gputil)
- [psutil System Monitoring](https://psutil.readthedocs.io/)
- [matplotlib Visualization](https://matplotlib.org/)

## 🤝 Contributing

When contributing to the project:

1. **Code Quality**: Ensure all code passes linting and type checking
2. **Type Hints**: Add type hints to new functions and classes
3. **Documentation**: Update documentation for new features
4. **Tests**: Add tests for new functionality
5. **Monitoring**: Integrate monitoring hooks for new training scripts

### Pre-commit Hooks

The project includes pre-commit hooks that automatically:

- Format code with Black
- Sort imports with isort
- Check code quality with flake8
- Verify types with MyPy

Install hooks:
```bash
pre-commit install
```

## 📄 License

This integration setup is part of the Fiber CNN Quality Assurance project. See the main project license for details.

---

**Happy coding with Pylance and real-time monitoring! 🚀** 