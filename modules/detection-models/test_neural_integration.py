#!/usr/bin/env python3
"""
Test the Neural Network Integration System
"""

import sys
import os

# Add the neural network integration path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neural-network-integration'))

try:
    from core.node_base import AtomicNode, NodeMetadata, NodeInput
    from core.logger import logger, LogChannel
    
    print("✅ Successfully imported core modules")
    
    # Create a simple test node
    def square(x):
        """Square a number"""
        return x ** 2
    
    # Create node
    node = AtomicNode(square, NodeMetadata(name='test_square', description='Squares input'))
    
    # Initialize node
    if node.initialize():
        print("✅ Node initialized successfully")
    else:
        print("❌ Node initialization failed")
        sys.exit(1)
    
    # Test the node
    test_values = [2, 5, 10, -3]
    
    print("\n🧪 Testing node processing:")
    for value in test_values:
        result = node.execute(NodeInput(data=value))
        if result.success:
            print(f"  square({value}) = {result.data} ✓ (time: {result.processing_time*1000:.2f}ms)")
        else:
            print(f"  square({value}) = ERROR: {result.error}")
    
    # Get node metrics
    metrics = node.get_metrics()
    print(f"\n📊 Node Metrics:")
    print(f"  Total calls: {metrics['metrics']['total_calls']}")
    print(f"  Successful: {metrics['metrics']['successful_calls']}")
    print(f"  Failed: {metrics['metrics']['failed_calls']}")
    print(f"  Avg time: {metrics['metrics']['average_processing_time']*1000:.2f}ms")
    
    # Test parameter tuning
    print("\n🎯 Testing parameter tuning:")
    node.register_tunable_parameter("multiplier", float, 0.1, 10.0, "Output multiplier")
    
    # Test tuning
    if node.tune_parameter("multiplier", 2.5):
        print("  ✓ Parameter tuned successfully")
        print(f"  Current value: {node.get_parameter('multiplier')}")
    else:
        print("  ✗ Parameter tuning failed")
    
    print("\n✅ All tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure the neural-network-integration directory exists and contains the required modules.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)