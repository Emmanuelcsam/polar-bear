#!/usr/bin/env python3
"""
Setup script for Fiber Optics Neural Network System
Creates necessary directories and validates environment
"""

import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from datetime import datetime

def print_header():
    """Print setup header"""
    print("\n" + "="*60)
    print("FIBER OPTICS NEURAL NETWORK SYSTEM - SETUP")
    print("="*60 + "\n")

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python {version.major}.{version.minor} detected")
        print("  ⚠️  Python 3.8 or higher is required")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check required dependencies"""
    print("\nChecking dependencies...")
    
    dependencies = {
        'PyTorch': ('torch', torch.__version__ if 'torch' in locals() else None),
        'NumPy': ('numpy', np.__version__ if 'np' in locals() else None),
        'OpenCV': ('cv2', cv2.__version__ if 'cv2' in locals() else None),
    }
    
    all_good = True
    for name, (module, version) in dependencies.items():
        if version:
            print(f"  ✓ {name}: {version}")
        else:
            print(f"  ❌ {name}: Not installed")
            all_good = False
    
    # Check GPU
    print("\nChecking GPU availability...")
    if torch.cuda.is_available():
        print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠️  No GPU detected - will use CPU")
    
    return all_good

def create_directory_structure():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    
    directories = [
        'checkpoints',
        'results',
        'logs',
        'data',
        'data/raw_images',
        'data/tensorized',
        'reference',
        'reference/core',
        'reference/cladding', 
        'reference/ferrule',
        'reference/defects'
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {dir_path}")
        else:
            print(f"  • Exists: {dir_path}")

def create_sample_config():
    """Create sample configuration file"""
    print("\nCreating sample configuration...")
    
    config_content = {
        "equation_coefficients": {
            "A": 1.0,
            "B": 1.0,
            "C": 0.0,
            "D": 1.0,
            "E": 1.0
        },
        "gradient_weight_factor": 1.0,
        "position_weight_factor": 1.0,
        "similarity_threshold": 0.7,
        "anomaly_threshold": 0.3,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 100
    }
    
    import json
    config_path = Path("fiber_optics_config.json")
    
    if not config_path.exists():
        with open(config_path, 'w') as f:
            json.dump(config_content, f, indent=4)
        print("  ✓ Created fiber_optics_config.json")
    else:
        print("  • fiber_optics_config.json already exists")

def create_test_data():
    """Create test tensor data"""
    print("\nCreating test data...")
    
    test_data_dir = Path("data/tensorized/test_samples")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a few test tensors
    for i in range(5):
        # Create synthetic fiber optic image tensor
        tensor = torch.randn(3, 256, 256)
        
        # Add circular patterns to simulate fiber structure
        center = 128
        y, x = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        
        # Core (bright center)
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        core_mask = dist < 30
        tensor[:, core_mask] += 0.5
        
        # Cladding
        cladding_mask = (dist >= 30) & (dist < 80)
        tensor[:, cladding_mask] += 0.2
        
        # Save
        save_path = test_data_dir / f"test_sample_{i:04d}.pt"
        torch.save({
            'tensor': tensor,
            'metadata': {
                'type': 'test',
                'created': datetime.now().isoformat()
            }
        }, save_path)
    
    print(f"  ✓ Created 5 test samples in {test_data_dir}")

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your tensorized .pt files in the appropriate folders")
    print("2. Update paths in config.py if needed")
    print("3. Run training: python main.py train")
    print("4. Analyze images: python main.py analyze <image_path>")
    print("\nFor more information, see README.md")

def main():
    """Run setup"""
    print_header()
    
    # Check environment
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Create structure
    create_directory_structure()
    create_sample_config()
    create_test_data()
    
    # Done
    print_next_steps()

if __name__ == "__main__":
    main()
