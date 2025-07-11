#!/usr/bin/env python3
"""
Neural Network Integration Demo
Demonstrates the capabilities of the unified neural network system
"""

import sys
import os
import numpy as np
import json
import time

# Add core modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network import NeuralNetwork
from core.node_base import AtomicNode, CompositeNode, NodeMetadata, NodeInput
from core.logger import logger, LogChannel
from core.parameter_tuner import TuningStrategy


def create_demo_nodes():
    """Create some demo nodes for testing"""
    nodes = []
    
    # Image processing node
    def process_image(image_array):
        """Simulate image processing"""
        if isinstance(image_array, np.ndarray):
            return {
                'mean': float(np.mean(image_array)),
                'std': float(np.std(image_array)),
                'shape': image_array.shape
            }
        return {'error': 'Invalid input'}
    
    image_node = AtomicNode(
        process_image,
        NodeMetadata(name="demo_image_processor", description="Demo image processor")
    )
    image_node.register_tunable_parameter("threshold", float, 0.0, 1.0, "Processing threshold")
    nodes.append(image_node)
    
    # Anomaly detection node
    def detect_anomalies(data, sensitivity=0.5):
        """Simulate anomaly detection"""
        if isinstance(data, dict) and 'mean' in data:
            threshold = 128 * sensitivity
            is_anomaly = data['mean'] > threshold
            return {
                'is_anomaly': is_anomaly,
                'confidence': min(abs(data['mean'] - threshold) / threshold, 1.0),
                'sensitivity': sensitivity
            }
        return {'error': 'Invalid input'}
    
    anomaly_node = AtomicNode(
        detect_anomalies,
        NodeMetadata(name="demo_anomaly_detector", description="Demo anomaly detector")
    )
    anomaly_node.register_tunable_parameter("sensitivity", float, 0.1, 1.0, "Detection sensitivity")
    nodes.append(anomaly_node)
    
    # Pattern recognition node
    def recognize_patterns(data):
        """Simulate pattern recognition"""
        patterns = []
        if isinstance(data, dict):
            if data.get('is_anomaly', False):
                patterns.append('anomaly_pattern')
            if data.get('mean', 0) > 100:
                patterns.append('bright_pattern')
            if data.get('std', 0) > 50:
                patterns.append('noisy_pattern')
        
        return {
            'patterns': patterns,
            'pattern_count': len(patterns),
            'timestamp': time.time()
        }
    
    pattern_node = AtomicNode(
        recognize_patterns,
        NodeMetadata(name="demo_pattern_recognizer", description="Demo pattern recognizer")
    )
    nodes.append(pattern_node)
    
    return nodes


def run_simple_demo():
    """Run a simple demonstration"""
    print("\n🎯 Running Simple Neural Network Demo")
    print("=" * 50)
    
    # Create network
    network = NeuralNetwork("DemoNetwork")
    
    # Add demo nodes
    demo_nodes = create_demo_nodes()
    for node in demo_nodes:
        node.initialize()
        network.add_node(node)
    
    # Connect nodes in sequence
    image_node = network.get_node("demo_image_processor")
    anomaly_node = network.get_node("demo_anomaly_detector")
    pattern_node = network.get_node("demo_pattern_recognizer")
    
    # Create connections
    image_node.connect(anomaly_node)
    anomaly_node.connect(pattern_node)
    
    print(f"\n✅ Created {len(demo_nodes)} demo nodes")
    
    # Test with sample data
    print("\n📊 Processing sample data:")
    
    # Create test images
    test_images = [
        np.random.normal(50, 10, (32, 32)),   # Dark image
        np.random.normal(200, 20, (32, 32)),  # Bright image
        np.random.normal(128, 60, (32, 32))   # Noisy image
    ]
    
    for i, image in enumerate(test_images):
        print(f"\n🖼️  Test Image {i+1}:")
        
        # Process through image node
        result1 = network.process(image, "demo_image_processor")
        if result1 and result1.success:
            print(f"  Image stats: mean={result1.data['mean']:.2f}, std={result1.data['std']:.2f}")
            
            # Process through anomaly detector
            result2 = network.process(result1.data, "demo_anomaly_detector")
            if result2 and result2.success:
                print(f"  Anomaly: {result2.data['is_anomaly']} (confidence: {result2.data['confidence']:.2f})")
                
                # Process through pattern recognizer
                result3 = network.process(result2.data, "demo_pattern_recognizer")
                if result3 and result3.success:
                    print(f"  Patterns: {', '.join(result3.data['patterns']) or 'None'}")
    
    # Show network statistics
    print("\n📈 Network Statistics:")
    stats = network.get_network_stats()
    print(f"  Total calls: {stats['performance']['total_calls']}")
    print(f"  Average latency: {stats['performance']['average_latency']*1000:.2f}ms")
    
    # Cleanup
    network.shutdown()


def run_pipeline_demo():
    """Run a pipeline demonstration"""
    print("\n🔄 Running Pipeline Demo")
    print("=" * 50)
    
    # Create network
    network = NeuralNetwork("PipelineNetwork")
    
    # Create a composite pipeline node
    pipeline = CompositeNode(NodeMetadata(
        name="image_analysis_pipeline",
        description="Complete image analysis pipeline"
    ))
    
    # Add demo nodes to pipeline
    demo_nodes = create_demo_nodes()
    for i, node in enumerate(demo_nodes):
        node.initialize()
        pipeline.add_node(node, f"stage_{i}")
    
    # Set execution order
    pipeline.set_execution_order(["stage_0", "stage_1", "stage_2"])
    pipeline.initialize()
    
    # Add pipeline to network
    network.add_node(pipeline)
    
    print("✅ Created image analysis pipeline")
    
    # Process test data
    test_image = np.random.normal(150, 30, (32, 32))
    
    print("\n🚀 Processing through pipeline:")
    result = network.process(test_image, "image_analysis_pipeline")
    
    if result and result.success:
        print("\n📋 Pipeline Results:")
        for stage, data in result.data.items():
            print(f"\n  {stage}:")
            print(f"    {json.dumps(data, indent=4)}")
    
    # Cleanup
    network.shutdown()


def run_tuning_demo():
    """Run parameter tuning demonstration"""
    print("\n🎯 Running Parameter Tuning Demo")
    print("=" * 50)
    
    # Create network
    network = NeuralNetwork("TuningNetwork")
    
    # Add demo nodes
    demo_nodes = create_demo_nodes()
    for node in demo_nodes:
        node.initialize()
        network.add_node(node)
    
    # Register nodes with tuner
    for node in demo_nodes:
        if node.tunable_params:
            network.parameter_tuner.register_node(node)
    
    print(f"✅ Registered {len(network.parameter_tuner.parameters)} tunable parameters")
    
    # Define a simple objective function
    def demo_objective(nodes, params):
        """Demo objective: minimize processing time"""
        total_time = 0
        test_data = np.random.normal(128, 30, (32, 32))
        
        # Process through nodes
        for node_name in ["demo_image_processor", "demo_anomaly_detector"]:
            if node_name in nodes:
                input_data = NodeInput(data=test_data)
                result = nodes[node_name].execute(input_data)
                if result.success:
                    total_time += result.processing_time
        
        return total_time
    
    from core.parameter_tuner import TuningObjective
    network.parameter_tuner.add_objective(TuningObjective(
        name="processing_time",
        function=demo_objective,
        minimize=True
    ))
    
    print("\n🔧 Running parameter tuning (10 iterations)...")
    best_result = network.tune_parameters(
        strategy=TuningStrategy.RANDOM_SEARCH,
        max_iterations=10
    )
    
    if best_result:
        print("\n✨ Best parameters found:")
        for param, value in best_result.parameters.items():
            print(f"  {param}: {value:.4f}")
        print(f"  Objective value: {best_result.objective_value:.6f}")
    
    # Cleanup
    network.shutdown()


def run_full_system_demo():
    """Run a full system demonstration with real modules"""
    print("\n🌟 Running Full System Demo")
    print("=" * 50)
    
    # Define a subset of module paths for demo
    module_paths = [
        "/home/jarvis/Documents/GitHub/polar-bear/modules/iteration6-lab-framework/modules"
    ]
    
    # Check if path exists
    if not os.path.exists(module_paths[0]):
        print("❌ Demo module path not found. Using simple demo instead.")
        run_simple_demo()
        return
    
    print("📂 Analyzing real modules...")
    
    # Create network
    network = NeuralNetwork("FullSystemDemo")
    
    try:
        # Initialize with real modules
        network.initialize(module_paths)
        
        # Show what was loaded
        stats = network.get_network_stats()
        print(f"\n📊 Loaded Network:")
        print(f"  Nodes: {stats['nodes']['total']}")
        print(f"  Types: {stats['nodes']['by_type']}")
        
        # Test with some real nodes if available
        if "random_pixel.gen" in network.nodes:
            print("\n🎲 Testing random pixel generation:")
            result = network.process({}, "random_pixel.gen")
            if result and result.success:
                print(f"  Generated image shape: {result.data.shape if hasattr(result.data, 'shape') else 'unknown'}")
        
        if "cv_module.hist" in network.nodes:
            print("\n📊 Testing histogram calculation:")
            test_image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            result = network.process(test_image, "cv_module.hist")
            if result and result.success:
                print(f"  Histogram computed: {len(result.data) if result.data is not None else 0} bins")
        
        # Save network state
        network.save_state("demo_network_state.json")
        print("\n💾 Network state saved to demo_network_state.json")
        
    except Exception as e:
        print(f"\n❌ Error during full system demo: {str(e)}")
        print("Falling back to simple demo...")
        run_simple_demo()
    
    finally:
        network.shutdown()


def main():
    """Main demo entry point"""
    print("\n🧠 Neural Network Integration System Demo 🧠")
    print("=" * 60)
    print("\nSelect demo to run:")
    print("1. Simple Demo - Basic node processing")
    print("2. Pipeline Demo - Composite node pipeline")
    print("3. Tuning Demo - Parameter optimization")
    print("4. Full System Demo - Real module integration")
    print("5. Run All Demos")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    demos = {
        '1': run_simple_demo,
        '2': run_pipeline_demo,
        '3': run_tuning_demo,
        '4': run_full_system_demo,
        '5': lambda: [run_simple_demo(), run_pipeline_demo(), run_tuning_demo(), run_full_system_demo()]
    }
    
    if choice in demos:
        demos[choice]()
        print("\n✅ Demo completed!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()