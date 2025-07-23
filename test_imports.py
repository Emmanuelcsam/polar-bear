#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all data module imports"""
    print("Testing imports...")
    
    try:
        print("1. Testing core imports...")
        from core.config_loader import get_config
        from core.logger import get_logger
        from core.statistical_config import get_statistical_config
        print("   ✓ Core imports successful")
    except ImportError as e:
        print(f"   ✗ Core imports failed: {e}")
        return False
    
    try:
        print("2. Testing data.tensor_processor...")
        from data.tensor_processor import TensorProcessor
        print("   ✓ tensor_processor imports successful")
    except ImportError as e:
        print(f"   ✗ tensor_processor imports failed: {e}")
        return False
    
    try:
        print("3. Testing data.data_loader...")
        from data.data_loader import FiberOpticsDataLoader, ReferenceDataLoader, FiberOpticsDataset
        print("   ✓ data_loader imports successful")
    except ImportError as e:
        print(f"   ✗ data_loader imports failed: {e}")
        return False
    
    try:
        print("4. Testing data.augmentation...")
        from data.augmentation import FiberOpticsAugmentation, get_augmentation_pipeline
        print("   ✓ augmentation imports successful")
    except ImportError as e:
        print(f"   ✗ augmentation imports failed: {e}")
        return False
    
    try:
        print("5. Testing data.feature_extractor...")
        from data.feature_extractor import FeatureExtractionPipeline
        print("   ✓ feature_extractor imports successful")
    except ImportError as e:
        print(f"   ✗ feature_extractor imports failed: {e}")
        return False
    
    try:
        print("6. Testing data.reference_comparator...")
        from data.reference_comparator import ReferenceComparator, ReferenceDatabase
        print("   ✓ reference_comparator imports successful")
    except ImportError as e:
        print(f"   ✗ reference_comparator imports failed: {e}")
        return False
    
    try:
        print("7. Testing utilities imports...")
        from utilities.distributed_utils import get_rank, get_world_size, is_main_process
        print("   ✓ utilities imports successful")
    except ImportError as e:
        print(f"   ✗ utilities imports failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True

if __name__ == "__main__":
    print("Import Test for Fiber Optics Neural Network")
    print("=" * 50)
    print(f"Working directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n🎉 All modules can be imported successfully!")
        print("The import paths have been fixed correctly.")
    else:
        print("\n❌ Some imports failed. Please check the error messages above.")
        sys.exit(1)