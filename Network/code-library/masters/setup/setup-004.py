#!/usr/bin/env python3
"""
Unified Setup System for Neural Framework
Combines all setup functionality into a single comprehensive script
"""

import os
import sys
import subprocess
import importlib
import platform
import json
import pathlib
import time
import datetime
import csv
import warnings
import argparse
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

# Attempt to import dependencies that might not be installed yet
try:
    from shared_config import DATA_DIR, IMAGE_INPUT_DIR
    SHARED_CONFIG_AVAILABLE = True
except ImportError:
    SHARED_CONFIG_AVAILABLE = False
    DATA_DIR = "./data"
    IMAGE_INPUT_DIR = "./data/input_images"

try:
    from connector_interface import setup_connector, get_hivemind_parameter, send_hivemind_status
    CONNECTOR_AVAILABLE = True
except ImportError:
    CONNECTOR_AVAILABLE = False

# Package definitions for setuptools
PACKAGES = {
    "visualization": {
        "name": "visualization",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "tutorials": {
        "name": "tutorials",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "real-time-monitoring": {
        "name": "real-time-monitoring",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "preprocessing": {
        "name": "preprocessing",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "optimization": {
        "name": "optimization",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "mock-libraries": {
        "name": "mock-libraries",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "ml-models": {
        "name": "ml-models",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration11-hybrid-1": {
        "name": "iteration11-hybrid-1",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration10-scaling": {
        "name": "iteration10-scaling",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration9-refactor": {
        "name": "iteration9-refactor",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration8-test-suite": {
        "name": "iteration8-test-suite",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration7-analytics": {
        "name": "iteration7-analytics",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration6-optimization": {
        "name": "iteration6-optimization",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration5-data-processing": {
        "name": "iteration5-data-processing",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration4-modular-start": {
        "name": "iteration4-modular-start",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration3-minimal-core": {
        "name": "iteration3-minimal-core",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "iteration2-basic-stats": {
        "name": "iteration2-basic-stats",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "image-quality-check": {
        "name": "image-quality-check",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "helper-utilities": {
        "name": "helper-utilities",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "deployment-configs": {
        "name": "deployment-configs",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "data-logger": {
        "name": "data-logger",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "configuration": {
        "name": "configuration",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "core": {
        "name": "core",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "analysis-reporting": {
        "name": "analysis-reporting",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "pytorch-production": {
        "name": "pytorch-production",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "pytorch-img-gen": {
        "name": "pytorch-img-gen",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "pytorch-cpu-gen": {
        "name": "pytorch-cpu-gen",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "unit-tests": {
        "name": "unit-tests",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "test-utils": {
        "name": "test-utils",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "test-app": {
        "name": "test-app",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    },
    "test": {
        "name": "test",
        "version": "0.1.0",
        "description": "A project dynamically packaged by the Neural Framework."
    }
}

# Fiber Optic Configuration Classes
@dataclass
class FiberSpecifications:
    """Fiber optic cable specifications"""
    core_diameter: float
    cladding_diameter: float
    coating_diameter: float
    fiber_type: str
    connector_type: str
    
@dataclass
class ZoneDefinition:
    """Zone definition for inspection"""
    zone_name: str
    inner_radius: float
    outer_radius: float
    max_defect_size: float
    max_scratch_count: int
    
@dataclass 
class DefectInfo:
    """Information about a detected defect"""
    defect_type: str
    location: Tuple[float, float]
    size: float
    zone: str
    severity: str
    
@dataclass
class InspectorConfig:
    """Configuration for the fiber optic inspector"""
    inspection_standard: str = "IEC-61300-3-35"
    detection_method: str = "DO2MR"
    magnification: float = 400.0
    image_resolution: Tuple[int, int] = (1920, 1080)
    calibration_factor: float = 0.5
    fiber_specs: Optional[FiberSpecifications] = None
    zones: List[ZoneDefinition] = field(default_factory=list)
    output_format: str = "csv"
    save_annotated_images: bool = True
    report_template: str = "standard"
    
class FiberInspector:
    """Advanced Fiber Optic End Face Defect Detection System"""
    
    def __init__(self, config: InspectorConfig):
        self.config = config
        self.calibration_data = None
        self.inspection_results = []
        
    def calibrate_system(self):
        """Calibrate the inspection system"""
        print("Calibrating fiber optic inspection system...")
        self.calibration_data = {
            "timestamp": datetime.datetime.now(),
            "magnification": self.config.magnification,
            "resolution": self.config.image_resolution,
            "factor": self.config.calibration_factor
        }
        
    def define_inspection_zones(self):
        """Define inspection zones based on standard"""
        if self.config.inspection_standard == "IEC-61300-3-35":
            if self.config.fiber_specs:
                core = self.config.fiber_specs.core_diameter
                cladding = self.config.fiber_specs.cladding_diameter
                
                self.config.zones = [
                    ZoneDefinition("Core", 0, core/2, 3.0, 0),
                    ZoneDefinition("Cladding", core/2, cladding/2, 5.0, 5),
                    ZoneDefinition("Adhesive", cladding/2, cladding/2 + 5, 10.0, -1),
                    ZoneDefinition("Contact", cladding/2 + 5, 250, -1, -1)
                ]
                
    def analyze_end_face(self, image_path: str) -> List[DefectInfo]:
        """Analyze fiber end face for defects"""
        print(f"Analyzing fiber end face: {image_path}")
        # Placeholder for actual analysis
        return []
        
    def generate_report(self, output_path: str):
        """Generate inspection report"""
        print(f"Generating inspection report: {output_path}")
        
    def setup_inspection_environment(self):
        """Setup the inspection environment"""
        print("Setting up fiber optic inspection environment...")
        self.calibrate_system()
        self.define_inspection_zones()

# Main Setup Class
class UnifiedSetup:
    """Unified setup system for Neural Framework"""
    
    def __init__(self):
        self.python_min_version = (3, 7)
        self.core_dependencies = [
            "numpy",
            "opencv-python",
            "scikit-learn",
            "matplotlib",
            "Pillow",
            "PyYAML",
            "pandas",
            "colorama",
            "psutil",
            "GPUtil",
            "optuna",
            "pyzmq",
            "toml"
        ]
        self.optional_dependencies = [
            "torch",
            "torchvision",
            "tensorflow"
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements"""
        current_version = sys.version_info[:2]
        if current_version < self.python_min_version:
            print(f"Error: Python {self.python_min_version[0]}.{self.python_min_version[1]}+ required")
            print(f"Current version: {current_version[0]}.{current_version[1]}")
            return False
        return True
        
    def create_directories(self):
        """Create all required directories"""
        directories = [
            # From setup_directories.py
            DATA_DIR,
            IMAGE_INPUT_DIR,
            # From setup-s.py
            "nodes",
            "synapses", 
            "configs",
            "logs",
            "tests",
            "data",
            "data/input",
            "data/output",
            "data/models",
            "data/cache"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            
        # Create placeholder files
        placeholder_content = "# Placeholder file\n"
        placeholders = [
            os.path.join(DATA_DIR, "placeholder.txt"),
            os.path.join(IMAGE_INPUT_DIR, "placeholder.txt")
        ]
        
        for placeholder in placeholders:
            with open(placeholder, 'w') as f:
                f.write(placeholder_content)
                
    def check_library_installed(self, library_name: str) -> bool:
        """Check if a library is installed"""
        try:
            importlib.import_module(library_name.replace('-', '_'))
            return True
        except ImportError:
            return False
            
    def install_library(self, library_name: str) -> bool:
        """Install a library using pip"""
        try:
            print(f"Installing {library_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {library_name}: {e}")
            return False
            
    def install_dependencies(self, include_optional: bool = True):
        """Install all dependencies"""
        print("\nInstalling core dependencies...")
        for dep in self.core_dependencies:
            if not self.check_library_installed(dep):
                self.install_library(dep)
            else:
                print(f"{dep} already installed")
                
        if include_optional:
            print("\nInstalling optional dependencies...")
            for dep in self.optional_dependencies:
                if not self.check_library_installed(dep):
                    try:
                        self.install_library(dep)
                    except Exception as e:
                        print(f"Optional dependency {dep} failed to install: {e}")
                else:
                    print(f"{dep} already installed")
                    
    def auto_install_dependencies(self):
        """Auto install dependencies with hivemind integration if available"""
        if CONNECTOR_AVAILABLE:
            connector = setup_connector()
            total_libs = len(self.core_dependencies) + len(self.optional_dependencies)
            
            for i, lib in enumerate(self.core_dependencies + self.optional_dependencies):
                if not self.check_library_installed(lib):
                    send_hivemind_status(f"Installing: {lib}")
                    self.install_library(lib)
                    
                progress = int((i + 1) / total_libs * 100)
                send_hivemind_status(f"Progress: {progress}%")
        else:
            self.install_dependencies()
            
    def create_default_config(self):
        """Create default configuration file"""
        config = {
            "neural_network": {
                "input_size": 784,
                "hidden_layers": [256, 128, 64],
                "output_size": 10,
                "activation": "relu",
                "learning_rate": 0.001
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2
            },
            "system": {
                "gpu_enabled": True,
                "num_workers": 4,
                "log_level": "INFO"
            }
        }
        
        with open("configs/default_config.json", "w") as f:
            json.dump(config, f, indent=4)
        print("Created default configuration file")
        
    def verify_installation(self):
        """Verify that all components are properly installed"""
        print("\nVerifying installation...")
        
        # Check core imports
        failed_imports = []
        for module in ["numpy", "cv2", "sklearn", "matplotlib"]:
            try:
                importlib.import_module(module)
                print(f"✓ {module} imported successfully")
            except ImportError:
                failed_imports.append(module)
                print(f"✗ Failed to import {module}")
                
        if failed_imports:
            print(f"\nWarning: Some modules failed to import: {failed_imports}")
            return False
            
        # Check directories
        required_dirs = ["nodes", "synapses", "configs", "logs", "data"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✓ Directory '{dir_name}' exists")
            else:
                print(f"✗ Directory '{dir_name}' missing")
                return False
                
        print("\nInstallation verified successfully!")
        return True
        
    def create_launcher_scripts(self):
        """Create platform-specific launcher scripts"""
        # Windows batch file
        if platform.system() == "Windows":
            with open("launch.bat", "w") as f:
                f.write("@echo off\n")
                f.write("python main.py %*\n")
            print("Created launch.bat for Windows")
            
        # Unix shell script  
        else:
            with open("launch.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("python3 main.py \"$@\"\n")
            os.chmod("launch.sh", 0o755)
            print("Created launch.sh for Unix/Linux")
            
    def quick_test(self):
        """Run a quick test to ensure basic functionality"""
        print("\nRunning quick test...")
        
        try:
            import numpy as np
            import cv2
            
            # Create a simple test array
            test_array = np.random.rand(100, 100, 3)
            print("✓ NumPy array created")
            
            # Simple OpenCV operation
            gray = cv2.cvtColor((test_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            print("✓ OpenCV conversion successful")
            
            print("\nQuick test passed!")
            return True
            
        except Exception as e:
            print(f"\nQuick test failed: {e}")
            return False
            
    def create_setup_file(self, package_name: str, output_dir: str = "."):
        """Create a setup.py file for a specific package"""
        if package_name not in PACKAGES:
            print(f"Error: Package '{package_name}' not found")
            return False
            
        package_info = PACKAGES[package_name]
        setup_content = f'''from setuptools import setup, find_packages

NAME = "{package_info['name']}"
VERSION = "{package_info['version']}"
DESCRIPTION = "{package_info['description']}"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    python_requires='>=3.6',
)
'''
        
        output_path = os.path.join(output_dir, f"setup-{package_name}.py")
        with open(output_path, 'w') as f:
            f.write(setup_content)
            
        print(f"Created setup file: {output_path}")
        return True
        
    def run_full_setup(self):
        """Run the complete setup process"""
        print("=" * 50)
        print("Neural Network Integration System - Full Setup")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return
            
        # Create directories
        print("\n1. Creating directories...")
        self.create_directories()
        
        # Install dependencies
        print("\n2. Installing dependencies...")
        self.auto_install_dependencies()
        
        # Create configuration
        print("\n3. Creating configuration...")
        self.create_default_config()
        
        # Create launcher scripts
        print("\n4. Creating launcher scripts...")
        self.create_launcher_scripts()
        
        # Verify installation
        print("\n5. Verifying installation...")
        if self.verify_installation():
            # Run quick test
            print("\n6. Running quick test...")
            self.quick_test()
            
        print("\n" + "=" * 50)
        print("Setup completed!")
        print("=" * 50)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Setup System for Neural Framework")
    parser.add_argument('--mode', choices=['full', 'directories', 'dependencies', 'package', 'fiber-optic'],
                       default='full', help='Setup mode to run')
    parser.add_argument('--package', type=str, help='Package name for package mode')
    parser.add_argument('--no-optional', action='store_true', help='Skip optional dependencies')
    
    args = parser.parse_args()
    
    setup = UnifiedSetup()
    
    if args.mode == 'full':
        setup.run_full_setup()
    elif args.mode == 'directories':
        setup.create_directories()
    elif args.mode == 'dependencies':
        setup.install_dependencies(include_optional=not args.no_optional)
    elif args.mode == 'package':
        if args.package:
            setup.create_setup_file(args.package)
        else:
            print("Error: --package argument required for package mode")
            print("\nAvailable packages:")
            for pkg in sorted(PACKAGES.keys()):
                print(f"  - {pkg}")
    elif args.mode == 'fiber-optic':
        # Setup fiber optic inspection system
        fiber_specs = FiberSpecifications(
            core_diameter=50.0,
            cladding_diameter=125.0,
            coating_diameter=250.0,
            fiber_type="MM OM3",
            connector_type="LC/PC"
        )
        
        config = InspectorConfig(fiber_specs=fiber_specs)
        inspector = FiberInspector(config)
        inspector.setup_inspection_environment()
        print("Fiber optic inspection system configured")

if __name__ == "__main__":
    main()