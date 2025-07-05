#!/usr/bin/env python3
"""
Demo script to test the fixed automated processing studio
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Import the fixed module
from automated_processing_studio_v2_fixed import (
    EnhancedProcessingStudio,
    check_dependencies
)


def create_test_images():
    """Create simple test images for demo"""
    # Create a dark square image
    input_img = np.zeros((200, 200, 3), dtype=np.uint8)
    input_img[50:150, 50:150] = 50  # Dark gray square
    
    # Create a bright circle target
    target_img = np.zeros((200, 200, 3), dtype=np.uint8)
    center = (100, 100)
    radius = 50
    cv2.circle(target_img, center, radius, (200, 200, 200), -1)
    
    return input_img, target_img


def main():
    """Run the demo"""
    print("Fixed Automated Processing Studio Demo")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install dependencies")
        return
    
    # Create test images
    print("\nCreating test images...")
    input_img, target_img = create_test_images()
    
    # Save test images
    demo_dir = Path("demo_output")
    demo_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(demo_dir / "input.png"), input_img)
    cv2.imwrite(str(demo_dir / "target.png"), target_img)
    print(f"✓ Test images saved to {demo_dir}")
    
    # Create studio instance
    print("\nInitializing studio...")
    try:
        studio = EnhancedProcessingStudio(
            scripts_dirs=["scripts"],
            cache_dir=str(demo_dir / "cache")
        )
        print(f"✓ Loaded {len(studio.script_manager.functions)} scripts")
        print(f"✓ Categories: {list(studio.script_manager.category_map.keys())[:5]}...")
    except Exception as e:
        print(f"❌ Failed to initialize studio: {e}")
        return
    
    # Test processing
    print("\nTesting image processing...")
    try:
        results = studio.process_to_match_target(
            input_img,
            target_img,
            max_iterations=10,  # Limited for demo
            similarity_threshold=0.1,
            verbose=False
        )
        
        print("\nResults:")
        print(f"- Success: {results['success']}")
        print(f"- Final similarity: {results['final_similarity']:.4f}")
        print(f"- Iterations: {results['iterations']}")
        print(f"- Pipeline length: {len(results['pipeline'])}")
        
        # Save result
        cv2.imwrite(str(demo_dir / "result.png"), results['final_image'])
        print(f"\n✓ Result saved to {demo_dir / 'result.png'}")
        
        # Test anomaly visualization
        print("\nGenerating anomaly visualizations...")
        vis = studio.generate_anomaly_visualization(input_img, results['final_image'])
        
        for name, img in vis.items():
            if isinstance(img, np.ndarray) and img.size > 0:
                cv2.imwrite(str(demo_dir / f"vis_{name}.png"), img)
        
        print(f"✓ Visualizations saved to {demo_dir}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Demo completed successfully!")
    print(f"Check the {demo_dir} directory for outputs.")


if __name__ == "__main__":
    main()