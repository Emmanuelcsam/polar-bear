#!/usr/bin/env python3
"""
Setup script for Fiber Optic End-Face CNN Pipeline
Initializes project structure and generates statistical priors
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = ['torch', 'torchvision', 'cv2', 'albumentations', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {missing_modules}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary project directories."""
    directories = [
        'checkpoints',
        'logs',
        'results',
        'cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def generate_statistical_priors():
    """Generate reference statistics from analysis reports."""
    try:
        # Import and run the statistical utilities
        sys.path.append('src')
        from utils import load_statistical_reports, create_reference_statistics
        
        print("Loading statistical reports...")
        reports = load_statistical_reports()
        
        print("Generating reference statistics...")
        create_reference_statistics(reports)
        
        print("Statistical priors generated successfully!")
        return True
        
    except Exception as e:
        print(f"Error generating statistical priors: {e}")
        return False

def validate_project_structure():
    """Validate that the project structure is correct."""
    required_files = [
        'src/train.py',
        'src/model.py',
        'src/dataset.py',
        'src/utils.py',
        'src/infer.py',
        'configs/bora.yaml',
        'fiber-cnn-bora.slurm',
        'requirements.txt'
    ]
    
    required_dirs = [
        'dataset',
        'statistics',
        'reference'
    ]
    
    missing_items = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_items.append(file_path)
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_items.append(dir_path)
    
    if missing_items:
        print(f"Missing required files/directories: {missing_items}")
        return False
    
    print("Project structure validation passed!")
    return True

def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        sys.path.append('src')
        
        # Test imports
        from model import EndfaceNet, CompositeLoss
        from dataset import EndfaceChunkDataset, make_dataloader
        from utils import load_statistical_reports
        
        print("All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Fiber Optic End-Face CNN Pipeline Setup")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("Setup failed: Missing dependencies")
        return False
    
    # Validate project structure
    print("\n2. Validating project structure...")
    if not validate_project_structure():
        print("Setup failed: Invalid project structure")
        return False
    
    # Create directories
    print("\n3. Creating directories...")
    create_directories()
    
    # Generate statistical priors
    print("\n4. Generating statistical priors...")
    if not generate_statistical_priors():
        print("Warning: Could not generate statistical priors")
        print("You may need to run 'python src/utils.py' manually")
    
    # Test imports
    print("\n5. Testing imports...")
    if not test_imports():
        print("Warning: Import test failed")
        print("Check that all dependencies are installed correctly")
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Activate conda environment: conda activate fiber-ai")
    print("2. Submit SLURM job: sbatch fiber-cnn-bora.slurm")
    print("3. Or run locally: torchrun --nproc_per_node=1 src/train.py --config configs/bora.yaml")
    print("4. For inference: python src/infer.py --weights checkpoints/epoch_49.pt --input dataset/sample.jpg")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 