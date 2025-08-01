#!/usr/bin/env python3
"""
Unified Demo Script - Merges all demonstration scripts into one comprehensive module
This script combines functionality from all individual demo scripts while eliminating duplicates.
"""

import json
import socket
import time
import subprocess
import sys
import os
import numpy as np
import tempfile
import glob
from pathlib import Path
from PIL import Image

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - AI demos will be limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - Computer vision demos will be limited")

# Import custom modules with error handling
MODULE_IMPORTS = {
    'connector': ['ConnectorSystem', 'ConnectorClient', 'Connector', 'get_connector'],
    'hivemind_connector': ['HivemindConnector'],
    'connector_interface': ['setup_connector', 'send_hivemind_status'],
    'data_store': [],
    'pixel_generator': [],
    'intensity_reader': ['learn'],
    'image_guided_generator': [],
    'pattern_recognizer': ['cluster'],
    'anomaly_detector': ['detect'],
    'trend_recorder': [],
    'learner': [],
    'geometry_analyzer': [],
    'neural_network': ['NeuralNetwork'],
    'modules.random_pixel': ['gen', 'guided'],
    'modules.cv_module': ['batch'],
    'core.logger': ['log', 'logger', 'LogChannel'],
    'core.datastore': ['scan'],
    'core.node_base': ['AtomicNode', 'CompositeNode', 'NodeMetadata', 'NodeInput'],
    'core.parameter_tuner': ['TuningStrategy', 'TuningObjective'],
    'pixel_sampler_refactored': [],
    'correlation_analyzer_refactored': [],
    'batch_processor_refactored': [],
    'self_reviewer_refactored': []
}

# Dynamic import with fallback
imported_modules = {}
for module_name, items in MODULE_IMPORTS.items():
    try:
        if items:
            module = __import__(module_name, fromlist=items)
            for item in items:
                imported_modules[f"{module_name}.{item}"] = getattr(module, item)
        else:
            imported_modules[module_name] = __import__(module_name)
    except ImportError:
        print(f"Warning: Could not import {module_name}")
        imported_modules[module_name] = None

# ============================================================================
# UTILITY FUNCTIONS (Consolidated from duplicates)
# ============================================================================

def send_command(command, data=None, host='localhost', port=10113):
    """Unified socket command sender (consolidated from multiple scripts)"""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        
        message = {"command": command}
        if data:
            message["data"] = data
            
        client.send(json.dumps(message).encode())
        
        response = client.recv(4096).decode()
        client.close()
        
        return json.loads(response) if response else None
    except Exception as e:
        print(f"Socket communication error: {e}")
        return None

def cleanup_files(patterns=None):
    """Unified cleanup function for demo files"""
    if patterns is None:
        patterns = ['demo_*.png', 'test_*.png', '*.log', 'demo_*.json']
    
    removed_files = []
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                removed_files.append(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    return removed_files

def create_demo_image(size=(100, 100), mode='RGB', name='demo_image.png'):
    """Create a demo image for testing"""
    img = Image.new(mode, size)
    pixels = img.load()
    
    # Create a gradient pattern
    for i in range(size[0]):
        for j in range(size[1]):
            if mode == 'RGB':
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
            else:
                pixels[i, j] = (i + j) % 256
    
    img.save(name)
    return name

# ============================================================================
# DEMO FUNCTIONS (Organized by category)
# ============================================================================

# --- CONNECTOR DEMOS ---

def demo_connector_basics():
    """Basic connector functionality demo (from usage22.py)"""
    print("\n=== Connector Basics Demo ===")
    
    if 'connector.ConnectorSystem' not in imported_modules:
        print("Connector module not available")
        return
    
    ConnectorSystem = imported_modules['connector.ConnectorSystem']
    conn = ConnectorSystem()
    
    # Initialize and load scripts
    print("Initializing connector system...")
    conn.initialize()
    
    # Load some demo scripts
    scripts = ['pixel_generator.py', 'intensity_reader.py']
    for script in scripts:
        if os.path.exists(script):
            result = conn.load_script(script)
            print(f"Loaded {script}: {result}")
    
    # Execute correlation analysis
    print("\nExecuting correlation analysis...")
    result = conn.execute_correlation_analysis('test_image.png')
    print(f"Analysis result: {result}")
    
    # Test shared memory
    print("\nTesting shared memory...")
    conn.set_shared_data('test_key', {'value': 42})
    data = conn.get_shared_data('test_key')
    print(f"Shared data: {data}")
    
    conn.cleanup()

def demo_hivemind_client():
    """Hivemind client demo (from hivemindclient22.py)"""
    print("\n=== Hivemind Client Demo ===")
    
    # Check status
    response = send_command("status")
    if response:
        print(f"Status: {response}")
    
    # Get script info
    response = send_command("script_info")
    if response:
        print(f"Scripts: {response}")
    
    # Execute correlation analysis
    response = send_command("execute_correlation", {"image_path": "test.png"})
    if response:
        print(f"Analysis: {response}")
    
    # Variable management
    send_command("set_variable", {"name": "test_var", "value": 123})
    response = send_command("get_variable", {"name": "test_var"})
    if response:
        print(f"Variable: {response}")

def demo_script_collaboration():
    """Script collaboration demo (from integration21.py and integration19.py)"""
    print("\n=== Script Collaboration Demo ===")
    
    if 'connector.ConnectorClient' not in imported_modules:
        print("Connector client not available")
        return
    
    ConnectorClient = imported_modules['connector.ConnectorClient']
    
    # Create connector clients for different scripts
    print("Creating connector clients...")
    pixel_gen = ConnectorClient("pixel_generator", port=10089)
    intensity_reader = ConnectorClient("intensity_reader", port=10089)
    
    # Collaborative task
    print("\nExecuting collaborative task...")
    
    # Generate pixels
    pixel_gen.send_command("generate", {"count": 100})
    
    # Read intensities
    intensity_reader.send_command("analyze", {"source": "generated_pixels"})
    
    # Get results
    results = pixel_gen.send_command("get_results")
    print(f"Collaboration results: {results}")

# --- IMAGE PROCESSING DEMOS ---

def demo_image_processing_pipeline():
    """Complete image processing pipeline (from 18.py)"""
    print("\n=== Image Processing Pipeline Demo ===")
    
    # Create demo images
    print("Creating demo images...")
    for i in range(3):
        create_demo_image(name=f"demo_pipeline_{i}.png")
    
    # Check available modules
    modules = ['pixel_generator', 'intensity_reader', 'pattern_recognizer', 
               'anomaly_detector', 'trend_recorder', 'geometry_analyzer']
    
    available_modules = [m for m in modules if imported_modules.get(m) is not None]
    print(f"Available modules: {available_modules}")
    
    if not available_modules:
        print("No image processing modules available")
        return
    
    # Run pipeline with available modules
    for module_name in available_modules:
        module = imported_modules[module_name]
        print(f"\nRunning {module_name}...")
        
        try:
            if hasattr(module, 'process'):
                result = module.process('demo_pipeline_0.png')
                print(f"Result: {result}")
            elif hasattr(module, 'main'):
                module.main()
            else:
                print(f"Module {module_name} has no process() or main() function")
        except Exception as e:
            print(f"Error in {module_name}: {e}")

def demo_my_image_lab():
    """My Image Lab functionality (from 26.py)"""
    print("\n=== My Image Lab Demo ===")
    
    # Check required modules
    required = ['modules.random_pixel.gen', 'modules.intensity_reader.learn', 
                'modules.anomaly_detector.detect', 'modules.pattern_recognizer.cluster']
    
    missing = [r for r in required if imported_modules.get(r) is None]
    if missing:
        print(f"Missing required modules: {missing}")
        return
    
    # Generate random images
    print("Generating random images...")
    for i in range(5):
        if imported_modules.get('modules.random_pixel.gen'):
            imported_modules['modules.random_pixel.gen'](10, 10, f"random_{i}.png")
    
    # Learn intensity distribution
    print("\nLearning intensity distribution...")
    if imported_modules.get('modules.intensity_reader.learn'):
        distribution = imported_modules['modules.intensity_reader.learn']()
        print(f"Learned distribution: {distribution}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    if imported_modules.get('modules.anomaly_detector.detect'):
        for i in range(5):
            anomaly = imported_modules['modules.anomaly_detector.detect'](f"random_{i}.png")
            print(f"Image {i} anomaly score: {anomaly}")
    
    # Cluster images
    print("\nClustering images...")
    if imported_modules.get('modules.pattern_recognizer.cluster'):
        clusters = imported_modules['modules.pattern_recognizer.cluster'](5)
        print(f"Clusters: {clusters}")

# --- AI/ML DEMOS ---

def demo_ai_capabilities():
    """AI capabilities demo (from ai.py)"""
    print("\n=== AI Capabilities Demo ===")
    
    # Check for required libraries
    if not TORCH_AVAILABLE and not CV2_AVAILABLE:
        print("Neither PyTorch nor OpenCV available - skipping AI demo")
        return
    
    # Create test image
    test_img = create_demo_image(size=(200, 200), name='ai_test.png')
    
    # Computer Vision Analysis
    if CV2_AVAILABLE:
        print("\n--- Computer Vision Analysis ---")
        img = cv2.imread(test_img)
        
        # Edge detection
        edges = cv2.Canny(img, 100, 200)
        print(f"Detected {np.sum(edges > 0)} edge pixels")
        
        # Corner detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        print(f"Detected {len(corners) if corners is not None else 0} corners")
    
    # Neural Network Learning
    if TORCH_AVAILABLE:
        print("\n--- Neural Network Learning ---")
        
        # Simple neural network
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)
        
        net = SimpleNet()
        print(f"Created neural network with {sum(p.numel() for p in net.parameters())} parameters")
        
        # Generate sample data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        # Simple training loop
        optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(10):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X), y)
            loss.backward()
            optimizer.step()
        
        print(f"Final loss: {loss.item():.4f}")

def demo_neural_network_integration():
    """Neural network integration demo (from network.py)"""
    print("\n=== Neural Network Integration Demo ===")
    
    if 'neural_network.NeuralNetwork' not in imported_modules:
        print("Neural network module not available")
        return
    
    NeuralNetwork = imported_modules['neural_network.NeuralNetwork']
    
    # Initialize network
    print("Initializing neural network system...")
    nn_system = NeuralNetwork()
    
    # Create demo nodes
    print("\nCreating demo nodes...")
    nodes = [
        {"name": "image_processor", "type": "atomic", "function": "process_image"},
        {"name": "anomaly_detector", "type": "atomic", "function": "detect_anomalies"},
        {"name": "pattern_recognizer", "type": "atomic", "function": "recognize_patterns"}
    ]
    
    for node in nodes:
        nn_system.add_node(node)
        print(f"Added node: {node['name']}")
    
    # Create pipeline
    print("\nCreating processing pipeline...")
    pipeline = nn_system.create_pipeline(["image_processor", "anomaly_detector", "pattern_recognizer"])
    print(f"Pipeline created: {pipeline}")
    
    # Run demo
    print("\nRunning pipeline demo...")
    test_data = {"image": "test.png", "threshold": 0.5}
    result = nn_system.run_pipeline(pipeline, test_data)
    print(f"Pipeline result: {result}")

# --- SYSTEM INTEGRATION DEMOS ---

def demo_image_categorization_system():
    """Image categorization system demo (from system.py)"""
    print("\n=== Image Categorization System Demo ===")
    
    # Check required modules
    required_modules = ['pixel_sampler_refactored', 'correlation_analyzer_refactored', 
                       'batch_processor_refactored', 'self_reviewer_refactored']
    
    available = [m for m in required_modules if imported_modules.get(m) is not None]
    
    if not available:
        print("No categorization modules available")
        return
    
    print(f"Available modules: {available}")
    
    # Create sample data
    print("\nCreating sample images...")
    categories = ['circles', 'squares', 'triangles']
    for cat in categories:
        os.makedirs(cat, exist_ok=True)
        for i in range(3):
            create_demo_image(name=f"{cat}/sample_{i}.png")
    
    # Build pixel database
    if 'pixel_sampler_refactored' in imported_modules:
        ps = imported_modules['pixel_sampler_refactored']
        print("\nBuilding pixel database...")
        ps.build_database(categories)
    
    # Analyze correlations
    if 'correlation_analyzer_refactored' in imported_modules:
        ca = imported_modules['correlation_analyzer_refactored']
        print("\nAnalyzing correlations...")
        correlations = ca.analyze_all()
        print(f"Found {len(correlations)} correlations")
    
    # Process batch
    if 'batch_processor_refactored' in imported_modules:
        bp = imported_modules['batch_processor_refactored']
        print("\nProcessing batch...")
        results = bp.process_directory(".")
        print(f"Processed {len(results)} images")
    
    # Self review
    if 'self_reviewer_refactored' in imported_modules:
        sr = imported_modules['self_reviewer_refactored']
        print("\nPerforming self-review...")
        review = sr.review_accuracy()
        print(f"Accuracy: {review}")

# ============================================================================
# MAIN MENU SYSTEM
# ============================================================================

def print_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("UNIFIED DEMO SYSTEM")
    print("="*60)
    print("\nCONNECTOR DEMOS:")
    print("1. Connector Basics")
    print("2. Hivemind Client")
    print("3. Script Collaboration")
    print("\nIMAGE PROCESSING DEMOS:")
    print("4. Image Processing Pipeline")
    print("5. My Image Lab")
    print("\nAI/ML DEMOS:")
    print("6. AI Capabilities")
    print("7. Neural Network Integration")
    print("\nSYSTEM DEMOS:")
    print("8. Image Categorization System")
    print("\nUTILITIES:")
    print("9. Cleanup Demo Files")
    print("0. Exit")
    print("-"*60)

def main():
    """Main entry point with interactive menu"""
    print("Welcome to the Unified Demo System!")
    print("This combines all demo scripts into one comprehensive module.")
    
    # Command line argument support
    if len(sys.argv) > 1:
        # Run specific demo from command line
        demo_map = {
            'connector': demo_connector_basics,
            'hivemind': demo_hivemind_client,
            'collaboration': demo_script_collaboration,
            'pipeline': demo_image_processing_pipeline,
            'imagelab': demo_my_image_lab,
            'ai': demo_ai_capabilities,
            'neural': demo_neural_network_integration,
            'categorization': demo_image_categorization_system
        }
        
        demo_name = sys.argv[1].lower()
        if demo_name in demo_map:
            demo_map[demo_name]()
            return
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {list(demo_map.keys())}")
            return
    
    # Interactive menu
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '0':
                print("\nExiting... Thank you for using the Unified Demo System!")
                break
            elif choice == '1':
                demo_connector_basics()
            elif choice == '2':
                demo_hivemind_client()
            elif choice == '3':
                demo_script_collaboration()
            elif choice == '4':
                demo_image_processing_pipeline()
            elif choice == '5':
                demo_my_image_lab()
            elif choice == '6':
                demo_ai_capabilities()
            elif choice == '7':
                demo_neural_network_integration()
            elif choice == '8':
                demo_image_categorization_system()
            elif choice == '9':
                removed = cleanup_files()
                print(f"\nCleaned up {len(removed)} files")
            else:
                print("\nInvalid choice! Please try again.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted! Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()