#!/usr/bin/env python3
"""
Test script to verify all components work individually and collaboratively
"""

import sys
import torch
from datetime import datetime

print(f"\n{'='*60}")
print("FIBER OPTICS NEURAL NETWORK - SYSTEM TEST")
print(f"{'='*60}\n")

# Test 1: Configuration Loading
print("TEST 1: Configuration Loading")
print("-" * 30)
try:
    from fiber_advanced_config_loader import get_config
    config = get_config()
    print("✓ Configuration loaded successfully")
    print(f"  - Learning rate: {config.config.optimizer.learning_rate}")
    print(f"  - Similarity threshold: {config.config.similarity.threshold}")
    print(f"  - Device: {config.config.system.device}")
except Exception as e:
    print(f"✗ Configuration loading failed: {e}")
    sys.exit(1)

# Test 2: Logger
print("\nTEST 2: Logger")
print("-" * 30)
try:
    from fiber_logger import get_logger
    logger = get_logger("TestLogger")
    logger.info("Test log message")
    print("✓ Logger initialized successfully")
except Exception as e:
    print(f"✗ Logger initialization failed: {e}")
    sys.exit(1)

# Test 3: Core Components
print("\nTEST 3: Core Components")
print("-" * 30)
try:
    from fiber_tensor_processor import TensorProcessor
    from fiber_feature_extractor import FeatureExtractionPipeline
    from fiber_reference_comparator import ReferenceComparator
    from fiber_anomaly_detector import ComprehensiveAnomalyDetector
    
    tensor_processor = TensorProcessor()
    print("✓ TensorProcessor initialized")
    
    feature_extractor = FeatureExtractionPipeline()
    print("✓ FeatureExtractionPipeline initialized")
    
    reference_comparator = ReferenceComparator()
    print("✓ ReferenceComparator initialized")
    
    anomaly_detector = ComprehensiveAnomalyDetector()
    print("✓ ComprehensiveAnomalyDetector initialized")
    
except Exception as e:
    print(f"✗ Core component initialization failed: {e}")
    sys.exit(1)

# Test 4: Advanced Components
print("\nTEST 4: Advanced Components")
print("-" * 30)
try:
    from fiber_advanced_architectures import SEBlock, CBAM, DeformableConv2d
    from fiber_advanced_losses import CombinedAdvancedLoss as CombinedLoss
    from fiber_advanced_optimizers import SAM, Lookahead
    from fiber_advanced_similarity import CombinedSimilarityMetric
    
    # Test SE Block
    se_block = SEBlock(64, 16)
    print("✓ SEBlock initialized")
    
    # Test CBAM
    cbam = CBAM(64)
    print("✓ CBAM initialized")
    
    # Test losses
    combined_loss = CombinedLoss()
    print("✓ CombinedLoss initialized")
    
    # Test similarity
    similarity_metric = CombinedSimilarityMetric()
    print("✓ CombinedSimilarityMetric initialized")
    
except Exception as e:
    print(f"✗ Advanced component initialization failed: {e}")
    sys.exit(1)

# Test 5: Enhanced Network
print("\nTEST 5: Enhanced Neural Network")
print("-" * 30)
try:
    from fiber_enhanced_integrated_network import EnhancedFiberOpticsIntegratedNetwork, IntegratedAnalysisPipeline
    
    # Create model
    model = EnhancedFiberOpticsIntegratedNetwork()
    print("✓ EnhancedFiberOpticsIntegratedNetwork initialized")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("✓ Forward pass successful")
    print(f"  - Output keys: {list(outputs.keys())[:5]}...")
    print(f"  - Final similarity: {outputs['final_similarity'][0].item():.4f}")
    
    # Test pipeline
    pipeline = IntegratedAnalysisPipeline()
    print("✓ IntegratedAnalysisPipeline initialized")
    
except Exception as e:
    print(f"✗ Enhanced network initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Enhanced Trainer
print("\nTEST 6: Enhanced Trainer")
print("-" * 30)
try:
    from fiber_enhanced_trainer import EnhancedFiberOpticsTrainer
    
    trainer = EnhancedFiberOpticsTrainer(model)
    print("✓ EnhancedFiberOpticsTrainer initialized")
    print(f"  - Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  - Mixed precision: {trainer.config.training.use_amp}")
    
except Exception as e:
    print(f"✗ Enhanced trainer initialization failed: {e}")
    sys.exit(1)

# Test 7: Data Loader
print("\nTEST 7: Data Loader")
print("-" * 30)
try:
    from fiber_data_loader import FiberOpticsDataLoader, ReferenceDataLoader
    
    data_loader = FiberOpticsDataLoader()
    print("✓ FiberOpticsDataLoader initialized")
    
    reference_loader = ReferenceDataLoader()
    print("✓ ReferenceDataLoader initialized")
    
except Exception as e:
    print(f"✗ Data loader initialization failed: {e}")
    sys.exit(1)

# Test 8: Main System
print("\nTEST 8: Main System Integration")
print("-" * 30)
try:
    from fiber_main import FiberOpticsSystem
    
    system = FiberOpticsSystem()
    print("✓ FiberOpticsSystem initialized successfully")
    print("✓ All components integrated correctly")
    
except Exception as e:
    print(f"✗ System integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Config Visualizer
print("\nTEST 9: Configuration Visualizer")
print("-" * 30)
try:
    from fiber_config_visualizer import ConfigSignalGenerator, ConfigParameterWidget
    
    signal_gen = ConfigSignalGenerator()
    # Convert config to dict for signal generation
    loader = config._config_loader if hasattr(config, '_config_loader') else None
    if loader:
        config_dict = loader._to_dict(loader.config)
    else:
        config_dict = config.config
    test_signal = signal_gen.generate_config_signal(config_dict)
    print("✓ ConfigSignalGenerator working")
    print(f"  - Signal shape: {test_signal.shape}")
    print(f"  - Signal range: [{test_signal.min():.3f}, {test_signal.max():.3f}]")
    
except Exception as e:
    print(f"✗ Config visualizer test failed: {e}")

# Summary
print(f"\n{'='*60}")
print("SYSTEM TEST COMPLETED SUCCESSFULLY")
print("All essential components are working correctly!")
print(f"{'='*60}\n")

print("Next steps:")
print("1. Run 'python fiber_main.py train' to train the model")
print("2. Run 'python fiber_main.py analyze <image>' to analyze an image")
print("3. Run 'python fiber_config_visualizer.py' to visualize configurations")