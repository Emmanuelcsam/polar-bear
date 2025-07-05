#!/usr/bin/env python3
"""
Demo script for Automated Processing Studio
Shows how to use the system programmatically
"""

import numpy as np
import cv2
import os
from pathlib import Path

# First ensure dependencies are installed
try:
    from automated_processing_studio import AutomatedProcessingStudio, DependencyManager
except ImportError:
    print("Installing dependencies first...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "setup_dependencies.py"])
    from automated_processing_studio import AutomatedProcessingStudio, DependencyManager


def create_demo_images():
    """Create simple demo images for testing"""
    print("Creating demo images...")
    
    # Create input image - simple gradient
    input_img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(200):
        input_img[i, :] = int(i * 255 / 200)
    
    # Create target image - gradient with circle
    target_img = input_img.copy()
    cv2.circle(target_img, (100, 100), 50, 255, -1)
    
    # Add some noise to make it more interesting
    noise = np.random.randint(-20, 20, (200, 200))
    target_img = np.clip(target_img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Save demo images
    cv2.imwrite("demo_input.png", input_img)
    cv2.imwrite("demo_target.png", target_img)
    
    print("✓ Created demo_input.png and demo_target.png")
    return input_img, target_img


def run_demo():
    """Run a demonstration of the automated processing studio"""
    print("\n" + "="*60)
    print("AUTOMATED PROCESSING STUDIO DEMO")
    print("="*60 + "\n")
    
    # Check dependencies
    DependencyManager.check_and_install_dependencies()
    
    # Create demo images if they don't exist
    if not os.path.exists("demo_input.png") or not os.path.exists("demo_target.png"):
        input_img, target_img = create_demo_images()
    else:
        print("Using existing demo images...")
        input_img = cv2.imread("demo_input.png", cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread("demo_target.png", cv2.IMREAD_GRAYSCALE)
    
    # Create studio instance
    print("\nInitializing Automated Processing Studio...")
    studio = AutomatedProcessingStudio(
        scripts_dir="scripts",
        cache_dir=".demo_cache"
    )
    
    print(f"✓ Loaded {len(studio.script_manager.functions)} processing scripts")
    
    # Process to match
    print("\n🎯 Starting automated processing...")
    print("Goal: Transform input image to match target image")
    
    results = studio.process_to_match(
        input_image=input_img,
        target_image=target_img,
        max_iterations=50,  # Reduced for demo
        similarity_threshold=0.15
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success: {results['success']}")
    print(f"Iterations used: {results['iterations']}")
    print(f"Final similarity: {results['final_similarity']:.4f}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    
    if results['pipeline']:
        print(f"\nOptimal pipeline found ({len(results['pipeline'])} steps):")
        for i, script in enumerate(results['pipeline'][-10:], 1):  # Show last 10
            print(f"  {i}. {script}")
    
    # Anomaly detection
    print("\n🔍 Running anomaly detection...")
    anomaly_results = studio.find_anomalies_and_similarities(results['final_image'])
    
    print(f"Anomaly detected: {anomaly_results['anomaly_analysis']['is_anomaly']}")
    print(f"Anomaly score: {anomaly_results['anomaly_analysis']['anomaly_score']:.3f}")
    print(f"Similar images in library: {len(anomaly_results['similar_images'])}")
    
    # Create visualization
    if 'final_image' in results:
        print("\n📊 Creating comparison visualization...")
        
        # Create side-by-side comparison
        h, w = input_img.shape
        comparison = np.zeros((h, w * 3 + 20), dtype=np.uint8)
        
        # Input
        comparison[:, :w] = input_img
        # Target
        comparison[:, w+10:2*w+10] = target_img
        # Result
        comparison[:, 2*w+20:] = results['final_image']
        
        # Save comparison
        cv2.imwrite("demo_comparison.png", comparison)
        print("✓ Saved comparison to demo_comparison.png")
    
    print("\n✨ Demo complete! Check the output files:")
    print("  - demo_comparison.png: Visual comparison")
    print("  - .demo_cache/report_*/: Detailed reports")
    
    # Learning statistics
    print(f"\n📈 Learning Statistics:")
    print(f"  - Exploration rate: {studio.learner.epsilon:.4f}")
    print(f"  - Q-table entries: {len(studio.learner.q_table)}")
    print(f"  - Successful combinations: {len(studio.successful_combinations)}")
    print(f"  - Failed combinations: {len(studio.failed_combinations)}")


if __name__ == "__main__":
    run_demo()