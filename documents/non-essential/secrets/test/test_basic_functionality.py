#!/usr/bin/env python3
"""
Basic functionality test to ensure the system works end-to-end
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from fiber_config import get_config, FiberOpticsConfig
        print("✓ fiber_config imported successfully")
        
        from fiber_logger import get_logger, FiberOpticsLogger
        print("✓ fiber_logger imported successfully")
        
        from fiber_tensor_processor import TensorProcessor
        print("✓ fiber_tensor_processor imported successfully")
        
        from fiber_feature_extractor import FeatureExtractionPipeline
        print("✓ fiber_feature_extractor imported successfully")
        
        from fiber_reference_comparator import ReferenceComparator
        print("✓ fiber_reference_comparator imported successfully")
        
        from fiber_anomaly_detector import ComprehensiveAnomalyDetector
        print("✓ fiber_anomaly_detector imported successfully")
        
        from fiber_integrated_network import IntegratedAnalysisPipeline
        print("✓ fiber_integrated_network imported successfully")
        
        from fiber_trainer import FiberOpticsTrainer
        print("✓ fiber_trainer imported successfully")
        
        from fiber_data_loader import FiberOpticsDataLoader, ReferenceDataLoader
        print("✓ fiber_data_loader imported successfully")
        
        from fiber_main import FiberOpticsSystem
        print("✓ fiber_main imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_initialization():
    """Test basic initialization of main components"""
    print("\nTesting basic initialization...")
    
    try:
        # Test config
        from fiber_config import get_config
        config = get_config()
        print(f"✓ Config initialized - Device: {config.get_device()}")
        
        # Test logger
        from fiber_logger import get_logger
        logger = get_logger("Test")
        logger.info("Test message")
        print("✓ Logger initialized and working")
        
        # Test tensor processor
        from fiber_tensor_processor import TensorProcessor
        processor = TensorProcessor()
        print("✓ TensorProcessor initialized")
        
        # Create a test tensor
        test_tensor = torch.randn(1, 3, 224, 224)
        stats = processor.get_tensor_statistics(test_tensor)
        print(f"✓ TensorProcessor can process tensors - Shape: {stats['shape']}")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_forward():
    """Test network forward pass"""
    print("\nTesting network forward pass...")
    
    try:
        from fiber_integrated_network import FiberOpticsIntegratedNetwork
        
        # Create network
        network = FiberOpticsIntegratedNetwork()
        network.eval()
        print("✓ Network created successfully")
        
        # Create test input
        x = torch.randn(1, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = network(x)
        
        print("✓ Forward pass completed")
        print(f"  - Final similarity: {output['final_similarity'].item():.4f}")
        print(f"  - Meets threshold: {bool(output['meets_threshold'].item())}")
        print(f"  - Output keys: {list(output.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Network forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_system():
    """Test main system initialization"""
    print("\nTesting main system...")
    
    try:
        from fiber_main import FiberOpticsSystem
        
        # Initialize system
        system = FiberOpticsSystem()
        print("✓ FiberOpticsSystem initialized")
        
        # Check components
        assert hasattr(system, 'tensor_processor')
        assert hasattr(system, 'integrated_pipeline')
        assert hasattr(system, 'trainer')
        assert hasattr(system, 'data_loader')
        print("✓ All system components present")
        
        return True
    except Exception as e:
        print(f"✗ Main system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image(path):
    """Create a test image file"""
    import cv2
    # Create a simple test pattern
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    # Add some circles to simulate fiber optic patterns
    cv2.circle(image, (112, 112), 50, (255, 255, 255), -1)
    cv2.circle(image, (112, 112), 30, (128, 128, 128), -1)
    cv2.imwrite(str(path), image)
    return path

def test_image_analysis():
    """Test image analysis functionality"""
    print("\nTesting image analysis...")
    
    try:
        from fiber_main import FiberOpticsSystem
        import tempfile
        
        # Create temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image_path = create_test_image(tmp.name)
        
        print(f"✓ Created test image: {test_image_path}")
        
        # Initialize system
        system = FiberOpticsSystem()
        
        # Analyze image
        results = system.analyze_single_image(test_image_path)
        
        print("✓ Image analysis completed")
        print(f"  - Similarity score: {results['summary']['final_similarity_score']:.4f}")
        print(f"  - Meets threshold: {results['summary']['meets_threshold']}")
        print(f"  - Primary region: {results['summary']['primary_region']}")
        print(f"  - Anomaly score: {results['summary']['anomaly_score']:.4f}")
        
        # Cleanup
        os.unlink(test_image_path)
        
        return True
    except Exception as e:
        print(f"✗ Image analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic functionality tests"""
    print("=" * 60)
    print("FIBER OPTICS NEURAL NETWORK - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Initialization", test_basic_initialization),
        ("Network Forward Pass", test_network_forward),
        ("Main System", test_main_system),
        ("Image Analysis", test_image_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The system is fully functional.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())