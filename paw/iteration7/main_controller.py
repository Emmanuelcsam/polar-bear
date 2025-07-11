#!/usr/bin/env python3
"""
Main Controller - Example of how to use all modules together
Run individual scripts or this controller to orchestrate them
"""

import subprocess
import sys
import time
import os

def run_script(script_name):
    """Run a Python script and return success status"""
    try:
        print(f"\n[CONTROLLER] Running {script_name}...")
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"[CONTROLLER] {script_name} completed successfully")
            return True
        else:
            print(f"[CONTROLLER] {script_name} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[CONTROLLER] Error running {script_name}: {e}")
        return False

def main():
    print("[CONTROLLER] Image Analysis System Starting...")
    
    # Check if we have an image to process
    test_image = None
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_image = file
            break
    
    if not test_image:
        print("[CONTROLLER] No image found. Please add an image to the directory.")
        return
    
    print(f"[CONTROLLER] Found image: {test_image}")
    
    # Phase 1: Initial data extraction
    print("\n=== Phase 1: Data Extraction ===")
    run_script('pixel_reader.py')
    time.sleep(1)
    
    # Phase 2: Start random generator in background (optional)
    # This would normally run continuously in a separate terminal
    # subprocess.Popen([sys.executable, 'random_generator.py'])
    
    # Phase 3: Analysis
    print("\n=== Phase 2: Analysis ===")
    analysis_scripts = [
        'pattern_recognizer.py',
        'anomaly_detector.py',
        'intensity_analyzer.py',
        'geometry_analyzer.py'
    ]
    
    for script in analysis_scripts:
        if os.path.exists(script):
            run_script(script)
            time.sleep(0.5)
    
    # Phase 4: Learning
    print("\n=== Phase 3: Learning ===")
    if os.path.exists('learning_engine.py'):
        # Manual learning example
        print("[CONTROLLER] Starting manual learning...")
        subprocess.run([
            sys.executable, '-c',
            f"from learning_engine import LearningEngine; "
            f"engine = LearningEngine(); "
            f"engine.manual_learn('{test_image}', 'test')"
        ])
    
    # Phase 5: Advanced calculations
    print("\n=== Phase 4: Advanced Calculations ===")
    run_script('data_calculator.py')
    
    # Phase 6: Trend analysis
    print("\n=== Phase 5: Trend Analysis ===")
    run_script('trend_analyzer.py')
    
    # Phase 7: Image generation
    print("\n=== Phase 6: Image Generation ===")
    run_script('image_generator.py')
    
    # Phase 8: Categorization
    print("\n=== Phase 7: Categorization ===")
    run_script('image_categorizer.py')
    
    # Phase 9: Computer Vision Analysis (OpenCV)
    print("\n=== Phase 8: Computer Vision Analysis ===")
    if os.path.exists('vision_processor.py'):
        run_script('vision_processor.py')
    
    # Phase 10: Neural Network Learning (PyTorch)
    print("\n=== Phase 9: Neural Network Learning ===")
    if os.path.exists('neural_learner.py'):
        run_script('neural_learner.py')
    
    # Phase 11: Hybrid Analysis
    print("\n=== Phase 10: Hybrid Analysis ===")
    if os.path.exists('hybrid_analyzer.py'):
        run_script('hybrid_analyzer.py')
    
    # Phase 12: Neural Generation
    print("\n=== Phase 11: Neural Generation ===")
    if os.path.exists('neural_generator.py'):
        run_script('neural_generator.py')
    
    # Phase 13: Visualization
    print("\n=== Phase 12: Visualization ===")
    if os.path.exists('visualizer.py'):
        print("[CONTROLLER] Creating visualizations...")
        run_script('visualizer.py')
    
    print("\n[CONTROLLER] Analysis complete!")
    print("[CONTROLLER] Note: Additional modules can be run separately:")
    print("\n  Real-time processing:")
    print("    - python realtime_processor.py  # Real-time monitoring")
    print("    - python live_capture.py        # Live video capture")
    print("    - python stream_analyzer.py     # Stream analysis")
    print("    - python realtime_demo.py       # Complete real-time demo")
    print("\n  High Performance Computing:")
    print("    - python gpu_accelerator.py     # GPU acceleration")
    print("    - python gpu_image_generator.py # GPU image generation")
    print("    - python parallel_processor.py  # Multi-core processing")
    print("    - python distributed_analyzer.py # Distributed computing")
    print("    - python hpc_optimizer.py       # HPC optimization")
    print("    - python hpc_demo.py            # Complete HPC demo")
    print("\n  Advanced Tools:")
    print("    - python ml_classifier.py       # Machine learning analysis")
    print("    - python network_api.py         # Network API server")
    print("    - python advanced_visualizer.py # Advanced visualizations")
    print("    - python data_exporter.py       # Export/import data")
    print("    - python config_manager.py      # System configuration")
    
    print("\n[CONTROLLER] Check generated files for results:")
    print("  Core Analysis:")
    print("    - pixel_data.json: Raw pixel data")
    print("    - patterns.json: Detected patterns")
    print("    - anomalies.json: Anomaly detection results")
    print("    - intensity_analysis.json: Intensity analysis")
    print("    - geometry_analysis.json: Geometric patterns")
    print("    - calculations.json: Advanced calculations")
    print("    - trends.json: Trend analysis")
    print("  AI/Vision:")
    print("    - vision_results.json: OpenCV analysis")
    print("    - neural_results.json: PyTorch neural network results")
    print("    - ml_report.json: Machine learning analysis")
    print("    - hybrid_analysis.json: Combined AI analysis")
    print("  Generated:")
    print("    - generated_*.jpg: Generated images")
    print("    - neural_generated_*.jpg: Neural network generated images")
    print("    - gpu_*.jpg: GPU generated images")
    print("    - vision_features.jpg: Detected visual features")
    print("    - analysis_visualization.png: Visual summary")

if __name__ == "__main__":
    main()