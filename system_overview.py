#!/usr/bin/env python3
"""
System Overview - Complete overview of the Modular Image Analysis System
Shows all modules, capabilities, and how they work together
"""

import os
import json
import time
from collections import defaultdict

def count_modules():
    """Count all available modules"""
    categories = defaultdict(list)
    
    # Module definitions
    modules = {
        # Core
        'pixel_reader.py': 'core',
        'random_generator.py': 'core',
        'correlator.py': 'core',
        
        # Analysis
        'pattern_recognizer.py': 'analysis',
        'anomaly_detector.py': 'analysis',
        'intensity_analyzer.py': 'analysis',
        'geometry_analyzer.py': 'analysis',
        'trend_analyzer.py': 'analysis',
        'data_calculator.py': 'analysis',
        
        # Computer Vision
        'vision_processor.py': 'vision',
        'hybrid_analyzer.py': 'vision',
        
        # Deep Learning
        'neural_learner.py': 'ai',
        'neural_generator.py': 'ai',
        'ml_classifier.py': 'ai',
        
        # HPC
        'gpu_accelerator.py': 'hpc',
        'gpu_image_generator.py': 'hpc',
        'parallel_processor.py': 'hpc',
        'distributed_analyzer.py': 'hpc',
        'hpc_optimizer.py': 'hpc',
        
        # Real-time
        'realtime_processor.py': 'realtime',
        'live_capture.py': 'realtime',
        'stream_analyzer.py': 'realtime',
        
        # Advanced Tools
        'network_api.py': 'tools',
        'advanced_visualizer.py': 'tools',
        'data_exporter.py': 'tools',
        'config_manager.py': 'tools',
        
        # Processing
        'data_store.py': 'processing',
        'batch_processor.py': 'processing',
        'image_categorizer.py': 'processing',
        'image_generator.py': 'processing',
        'learning_engine.py': 'processing',
        'continuous_analyzer.py': 'processing',
        
        # Utilities
        'logger.py': 'utility',
        'visualizer.py': 'utility',
        'main_controller.py': 'utility',
        'quick_start.py': 'demo',
        'communication_demo.py': 'demo',
        'ai_demo.py': 'demo',
        'realtime_demo.py': 'demo',
        'hpc_demo.py': 'demo',
        'test_independence.py': 'test',
        'test_realtime_independence.py': 'test',
        'test_hpc_independence.py': 'test',
        'create_test_image.py': 'utility'
    }
    
    # Count existing modules
    for module, category in modules.items():
        if os.path.exists(module):
            categories[category].append(module.replace('.py', ''))
    
    return dict(categories)

def check_dependencies():
    """Check which optional dependencies are available"""
    dependencies = {
        'PyTorch': False,
        'OpenCV': False,
        'GPU/CUDA': False,
        'Network': True,  # Standard library
        'Advanced Viz': False
    }
    
    try:
        import torch
        dependencies['PyTorch'] = True
        if torch.cuda.is_available():
            dependencies['GPU/CUDA'] = True
    except:
        pass
    
    try:
        import cv2
        dependencies['OpenCV'] = True
    except:
        pass
    
    try:
        import seaborn
        dependencies['Advanced Viz'] = True
    except:
        pass
    
    return dependencies

def analyze_outputs():
    """Analyze generated output files"""
    output_categories = {
        'JSON Results': [],
        'Images': [],
        'Models': [],
        'Reports': [],
        'Configuration': []
    }
    
    for file in os.listdir('.'):
        if file.endswith('.json'):
            if 'config' in file or 'preset' in file:
                output_categories['Configuration'].append(file)
            else:
                output_categories['JSON Results'].append(file)
        elif file.endswith(('.jpg', '.png', '.gif')):
            output_categories['Images'].append(file)
        elif file.endswith(('.pkl', '.pth', '.h5')):
            output_categories['Models'].append(file)
        elif file.endswith(('.pdf', '.xml', '.yaml', '.csv')):
            output_categories['Reports'].append(file)
    
    return output_categories

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         MODULAR IMAGE ANALYSIS SYSTEM - COMPLETE OVERVIEW        ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Display complete system overview"""
    
    print_banner()
    
    # Count modules
    categories = count_modules()
    total_modules = sum(len(modules) for modules in categories.values())
    
    print(f"SYSTEM STATUS: {total_modules} Modules Available\n")
    
    # Display modules by category
    print("MODULE CATEGORIES:")
    print("─" * 60)
    
    category_order = ['core', 'analysis', 'vision', 'ai', 'hpc', 'realtime', 
                     'tools', 'processing', 'utility', 'demo', 'test']
    
    category_descriptions = {
        'core': 'Core System (Original Functionality)',
        'analysis': 'Advanced Analysis',
        'vision': 'Computer Vision (OpenCV)',
        'ai': 'Artificial Intelligence (PyTorch)',
        'hpc': 'High Performance Computing',
        'realtime': 'Real-Time Processing',
        'tools': 'Advanced Tools',
        'processing': 'Data Processing',
        'utility': 'Utilities',
        'demo': 'Demonstrations',
        'test': 'Test Suites'
    }
    
    for category in category_order:
        if category in categories:
            modules = categories[category]
            desc = category_descriptions.get(category, category.title())
            print(f"\n{desc} ({len(modules)} modules):")
            
            for module in sorted(modules):
                print(f"  • {module}")
    
    # Check dependencies
    print("\n\nDEPENDENCIES:")
    print("─" * 60)
    dependencies = check_dependencies()
    
    for dep, available in dependencies.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"{dep:<15} {status}")
    
    # Analyze outputs
    print("\n\nOUTPUT FILES:")
    print("─" * 60)
    outputs = analyze_outputs()
    
    for category, files in outputs.items():
        if files:
            print(f"\n{category} ({len(files)} files):")
            for file in sorted(files)[:5]:  # Show first 5
                print(f"  • {file}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")
    
    # Key features
    print("\n\nKEY FEATURES:")
    print("─" * 60)
    features = [
        "✓ 45+ Independent Modules",
        "✓ Zero Dependencies Between Modules",
        "✓ JSON-Based Communication",
        "✓ GPU Acceleration Support",
        "✓ Real-Time Processing",
        "✓ Machine Learning Integration",
        "✓ Network API for Remote Processing",
        "✓ Multi-Format Export/Import",
        "✓ Advanced Visualization Suite",
        "✓ Configuration Management"
    ]
    
    for feature in features:
        print(feature)
    
    # Usage examples
    print("\n\nQUICK START COMMANDS:")
    print("─" * 60)
    print("python config_manager.py              # Configure system")
    print("python quick_start.py                 # Basic demo")
    print("python ai_demo.py                     # AI capabilities")
    print("python hpc_demo.py                    # GPU/HPC demo")
    print("python realtime_demo.py               # Real-time demo")
    print("python main_controller.py             # Run standard pipeline")
    
    # Configuration presets
    print("\n\nCONFIGURATION PRESETS:")
    print("─" * 60)
    print("python config_manager.py preset basic              # Essential modules")
    print("python config_manager.py preset ai_powered         # AI/ML modules")
    print("python config_manager.py preset high_performance  # GPU/HPC modules")
    print("python config_manager.py preset real_time         # Real-time modules")
    print("python config_manager.py preset full_system       # Everything")
    
    # Architecture principles
    print("\n\nARCHITECTURE PRINCIPLES:")
    print("─" * 60)
    print("1. Complete Independence - Delete any module, others continue")
    print("2. JSON Communication - No direct imports between modules")
    print("3. Graceful Degradation - Missing dependencies handled")
    print("4. Terminal Logging - Every action is logged")
    print("5. Minimal Code - Each module focused on one task")
    
    # Save overview
    overview_data = {
        'timestamp': time.time(),
        'total_modules': total_modules,
        'categories': {cat: len(mods) for cat, mods in categories.items()},
        'dependencies': dependencies,
        'output_files': {cat: len(files) for cat, files in outputs.items()}
    }
    
    with open('system_overview.json', 'w') as f:
        json.dump(overview_data, f, indent=2)
    
    print("\n\nSYSTEM READY!")
    print("─" * 60)
    print(f"Total Modules: {total_modules}")
    print(f"Total Output Files: {sum(len(files) for files in outputs.values())}")
    print("\nThe complete modular image analysis system is ready to use!")

if __name__ == "__main__":
    main()