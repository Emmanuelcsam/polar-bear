#!/usr/bin/env python3
"""
AI Demo - Showcases PyTorch and OpenCV integration
Demonstrates how computer vision and neural networks enhance the system
"""

import subprocess
import time
import json
import os
import sys

def run_and_wait(script, message):
    """Run a script and show results"""
    print(f"\n{message}")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {script} completed successfully")
    else:
        print(f"✗ {script} failed: {result.stderr}")
    time.sleep(1)

def main():
    print("=== AI-Enhanced Image Analysis Demo ===")
    print("This demonstrates PyTorch neural networks and OpenCV vision processing\n")
    
    # Check for required libraries
    try:
        import torch
        import cv2
        print("✓ PyTorch and OpenCV are installed")
    except ImportError as e:
        print(f"ERROR: Missing required library: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Check for image
    image_file = None
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_file = file
            break
    
    if not image_file:
        print("\nNo image found. Creating test image...")
        subprocess.run([sys.executable, 'create_test_image.py'], capture_output=True)
        image_file = 'test_pattern.jpg'
    
    print(f"\nUsing image: {image_file}")
    
    # Phase 1: Basic Analysis
    print("\n=== Phase 1: Basic Pixel Analysis ===")
    run_and_wait('pixel_reader.py', "Reading pixel intensities...")
    run_and_wait('pattern_recognizer.py', "Recognizing patterns...")
    run_and_wait('intensity_analyzer.py', "Analyzing intensity distribution...")
    
    # Phase 2: Computer Vision
    print("\n=== Phase 2: Computer Vision Analysis (OpenCV) ===")
    run_and_wait('vision_processor.py', "Detecting edges, corners, and contours...")
    
    # Show vision results
    if os.path.exists('vision_results.json'):
        with open('vision_results.json', 'r') as f:
            vision = json.load(f)
            print(f"\nVision Results:")
            print(f"  - Edges found: {vision['edges']['canny_edge_count']}")
            print(f"  - Corners detected: {vision['corners']['harris_count']}")
            print(f"  - Contours found: {vision['contours']['total_count']}")
            print(f"  - Texture entropy: {vision['texture']['lbp_entropy']:.2f}")
    
    # Phase 3: Neural Network Learning
    print("\n=== Phase 3: Neural Network Learning (PyTorch) ===")
    run_and_wait('neural_learner.py', "Training neural network on pixel patterns...")
    
    # Show neural results
    if os.path.exists('neural_results.json'):
        with open('neural_results.json', 'r') as f:
            neural = json.load(f)
            print(f"\nNeural Network Results:")
            print(f"  - Sequences trained: {neural['sequences_trained']}")
            print(f"  - Training loss: {neural['training_loss']:.4f}")
            print(f"  - Predictions generated: {len(neural['predictions'])}")
            print(f"  - First 5 predictions: {neural['predictions'][:5]}")
    
    # Phase 4: Hybrid Analysis
    print("\n=== Phase 4: Hybrid AI Analysis ===")
    run_and_wait('hybrid_analyzer.py', "Combining vision and neural insights...")
    
    # Show hybrid results
    if os.path.exists('hybrid_analysis.json'):
        with open('hybrid_analysis.json', 'r') as f:
            hybrid = json.load(f)
            print(f"\nHybrid Analysis:")
            if 'synthesis' in hybrid:
                synth = hybrid['synthesis']
                print(f"  - Complexity level: {synth['complexity_level']}")
                print(f"  - Quality score: {synth['quality_score']:.1f}/100")
                print(f"  - Characteristics: {', '.join(synth['dominant_characteristics'])}")
    
    # Phase 5: AI-Enhanced Generation
    print("\n=== Phase 5: AI-Enhanced Image Generation ===")
    run_and_wait('neural_generator.py', "Generating images with neural network...")
    
    # Summary
    print("\n=== Demo Complete! ===")
    print("\nGenerated Files:")
    
    files_to_check = [
        ('vision_features.jpg', 'Visualization of detected features'),
        ('edges_canny.jpg', 'Edge detection result'),
        ('neural_generated_*.jpg', 'Neural network generated images'),
        ('neural_trained_generated.jpg', 'Trained generator output'),
        ('neural_styled.jpg', 'Style transfer result')
    ]
    
    for pattern, description in files_to_check:
        if '*' in pattern:
            # Handle wildcard patterns
            import glob
            matches = glob.glob(pattern)
            if matches:
                print(f"  ✓ {matches[0]} - {description}")
        elif os.path.exists(pattern):
            print(f"  ✓ {pattern} - {description}")
    
    print("\nJSON Results:")
    json_files = [
        'vision_results.json',
        'neural_results.json',
        'hybrid_analysis.json',
        'hybrid_recommendations.json'
    ]
    
    for f in json_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
    
    # Show recommendations
    if os.path.exists('hybrid_recommendations.json'):
        with open('hybrid_recommendations.json', 'r') as f:
            recs = json.load(f)
            print("\nAI Recommendations:")
            for action in recs['immediate_actions'][:3]:
                print(f"  • {action}")
    
    print("\nTry these next steps:")
    print("  1. View 'vision_features.jpg' to see detected features")
    print("  2. Compare original image with 'neural_generated_*.jpg'")
    print("  3. Run 'python visualizer.py' for charts and graphs")
    print("  4. Add more images and run 'python batch_processor.py'")

if __name__ == "__main__":
    main()