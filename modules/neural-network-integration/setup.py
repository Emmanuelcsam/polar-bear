#!/usr/bin/env python3
"""
Setup script for the Neural Network Integration System
Handles initial setup, dependency installation, and system verification
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║       🧠 Neural Network Integration System Setup 🧠           ║
║                                                               ║
║  Transforming Scripts into Intelligent Neural Networks        ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_python_version():
    """Check Python version compatibility"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    directories = [
        "nodes",
        "synapses", 
        "configs",
        "logs",
        "tests",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ {directory}/")
    
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "numpy",
        "opencv-python",
        "scikit-learn",
        "matplotlib",
        "Pillow",
        "PyYAML",
        "colorama",
        "psutil",
        "GPUtil",
        "optuna",
        "pyzmq",
        "toml"
    ]
    
    # Optional dependencies
    optional_deps = {
        "torch": "PyTorch for deep learning",
        "torchvision": "PyTorch vision utilities",
        "tensorflow": "TensorFlow for deep learning",
        "pandas": "Data analysis",
        "jupyter": "Interactive notebooks",
        "plotly": "Interactive visualizations"
    }
    
    # Install core dependencies
    print("\n Installing core dependencies...")
    for dep in core_deps:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                check=True,
                capture_output=True
            )
            print(f"  ✓ {dep}")
        except subprocess.CalledProcessError:
            print(f"  ✗ {dep} (failed)")
    
    # Ask about optional dependencies
    print("\n📋 Optional dependencies:")
    for dep, desc in optional_deps.items():
        response = input(f"  Install {dep} ({desc})? [y/N]: ").strip().lower()
        if response == 'y':
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep],
                    check=True,
                    capture_output=True
                )
                print(f"    ✓ {dep} installed")
            except subprocess.CalledProcessError:
                print(f"    ✗ {dep} installation failed")
    
    return True


def create_default_config():
    """Create default configuration file"""
    print("\n⚙️  Creating default configuration...")
    
    config = {
        "system": {
            "name": "PolarBearNeuralNetwork",
            "version": "1.0.0",
            "auto_start": True
        },
        "modules": {
            "paths": [
                "../iteration6-lab-framework",
                "../iteration4-modular-start",
                "../experimental-features",
                "../artificial-intelligence"
            ],
            "recursive_scan": True
        },
        "performance": {
            "max_workers": 4,
            "cache_enabled": True,
            "cache_size_mb": 1024
        },
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "console_enabled": True
        }
    }
    
    with open("configs/default_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("  ✓ Created configs/default_config.json")
    return True


def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ("Core modules", "from core import *"),
        ("Neural network", "from neural_network import NeuralNetwork"),
        ("Demo", "import demo")
    ]
    
    all_good = True
    for name, import_str in test_imports:
        try:
            exec(import_str)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {str(e)}")
            all_good = False
    
    return all_good


def create_launcher_scripts():
    """Create launcher scripts for different platforms"""
    print("\n🚀 Creating launcher scripts...")
    
    # Unix launcher (Linux/Mac)
    unix_launcher = """#!/bin/bash
# Neural Network Integration System Launcher

echo "🧠 Starting Neural Network Integration System..."
python3 neural_network.py "$@"
"""
    
    with open("run.sh", "w") as f:
        f.write(unix_launcher)
    os.chmod("run.sh", 0o755)
    print("  ✓ Created run.sh")
    
    # Windows launcher
    windows_launcher = """@echo off
REM Neural Network Integration System Launcher

echo 🧠 Starting Neural Network Integration System...
python neural_network.py %*
"""
    
    with open("run.bat", "w") as f:
        f.write(windows_launcher)
    print("  ✓ Created run.bat")
    
    # Demo launcher
    demo_launcher = """#!/usr/bin/env python3
import demo
demo.main()
"""
    
    with open("run_demo.py", "w") as f:
        f.write(demo_launcher)
    os.chmod("run_demo.py", 0o755)
    print("  ✓ Created run_demo.py")
    
    return True


def quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick test...")
    
    try:
        # Import and create a simple network
        from neural_network import NeuralNetwork
        from core.node_base import AtomicNode, NodeMetadata
        
        # Create network
        network = NeuralNetwork("TestNetwork")
        
        # Create simple node
        def test_func(x):
            return x * 2
        
        node = AtomicNode(test_func, NodeMetadata(name="test"))
        node.initialize()
        network.add_node(node)
        
        # Test processing
        result = network.process(5, "test")
        
        if result and result.success and result.data == 10:
            print("  ✓ Basic functionality test passed!")
            network.shutdown()
            return True
        else:
            print("  ✗ Basic functionality test failed")
            network.shutdown()
            return False
            
    except Exception as e:
        print(f"  ✗ Test failed: {str(e)}")
        return False


def main():
    """Main setup function"""
    print_banner()
    
    steps = [
        ("Python version check", check_python_version),
        ("Create directories", create_directories),
        ("Install dependencies", install_dependencies),
        ("Create configuration", create_default_config),
        ("Verify installation", verify_installation),
        ("Create launchers", create_launcher_scripts),
        ("Quick test", quick_test)
    ]
    
    total_steps = len(steps)
    completed = 0
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n[{i}/{total_steps}] {name}")
        print("-" * 50)
        
        if func():
            completed += 1
        else:
            print(f"\n⚠️  Step failed: {name}")
            response = input("Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("\n❌ Setup aborted")
                return False
    
    print("\n" + "=" * 60)
    print(f"✅ Setup completed! ({completed}/{total_steps} steps successful)")
    print("\nTo start the system:")
    
    if platform.system() == "Windows":
        print("  > run.bat")
    else:
        print("  $ ./run.sh")
    
    print("\nTo run the demo:")
    print("  $ python run_demo.py")
    
    print("\nTo use in your code:")
    print("  from neural_network import NeuralNetwork")
    print("  network = NeuralNetwork()")
    
    print("\n📚 See README.md for detailed documentation")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)