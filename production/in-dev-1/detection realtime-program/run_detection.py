#!/usr/bin/env python3
"""
Simple Startup Script for Real-Time Defect Detection

This script provides a user-friendly way to start the real-time defect detection system.
It handles command line arguments, validates inputs, and provides clear feedback.

Usage:
    python run_detection.py <reference_image_path> [options]
    
Example:
    python run_detection.py reference.jpg
    python run_detection.py reference.jpg --fast --no-visualization
"""

import sys
import os
import argparse
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from realtime_defect_detection import (
        DetectionConfig, RealTimeController, defect_alert_handler
    )
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import system components: {e}")
    print("Please ensure all required files are in the current directory.")
    SYSTEM_AVAILABLE = False


def create_sample_reference():
    """Create a sample reference image for testing."""
    import numpy as np
    import cv2
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some structure to make it more realistic
    cv2.rectangle(test_image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(test_image, (320, 240), 50, (128, 128, 128), -1)
    
    filename = "sample_reference.jpg"
    cv2.imwrite(filename, test_image)
    
    print(f"✅ Created sample reference image: {filename}")
    return filename


def validate_reference_image(image_path):
    """Validate that the reference image exists and is readable."""
    if not os.path.exists(image_path):
        print(f"❌ Reference image not found: {image_path}")
        return False
    
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image file: {image_path}")
            return False
        
        print(f"✅ Reference image loaded: {img.shape}")
        return True
    
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Defect Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py reference.jpg
  python run_detection.py reference.jpg --fast --no-visualization
  python run_detection.py reference.jpg --sensitive --save-results
  python run_detection.py --create-sample
        """
    )
    
    parser.add_argument(
        "reference_image",
        nargs="?",
        help="Path to reference image for comparison"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample reference image for testing"
    )
    
    # Detection settings
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="Use more sensitive detection settings"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast detection mode (default)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full detection algorithm (slower but more accurate)"
    )
    
    # Processing settings
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Processing FPS (default: 10.0)"
    )
    
    parser.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Image resize factor (default: 1.0)"
    )
    
    # Output settings
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable live visualization"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detection results to files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="realtime_output",
        help="Output directory for results (default: realtime_output)"
    )
    
    # Camera settings
    parser.add_argument(
        "--exposure",
        type=int,
        default=10000,
        help="Camera exposure time in microseconds (default: 10000)"
    )
    
    parser.add_argument(
        "--gain",
        type=int,
        default=0,
        help="Camera gain value (default: 0)"
    )
    
    return parser.parse_args()


def create_config(args):
    """Create detection configuration from arguments."""
    # Set detection sensitivity
    if args.sensitive:
        anomaly_threshold = 1.5
        ssim_threshold = 0.9
        confidence_threshold = 0.3
        min_defect_area = 10
    else:
        anomaly_threshold = 2.0
        ssim_threshold = 0.8
        confidence_threshold = 0.5
        min_defect_area = 25
    
    # Set detection mode
    enable_fast_mode = not args.full
    
    # Set visualization
    enable_visualization = not args.no_visualization
    
    # Set result saving
    save_results = args.save_results
    
    config = DetectionConfig(
        reference_image_path=args.reference_image,
        anomaly_threshold=anomaly_threshold,
        ssim_threshold=ssim_threshold,
        confidence_threshold=confidence_threshold,
        enable_fast_mode=enable_fast_mode,
        resize_factor=args.resize,
        min_defect_area=min_defect_area,
        max_defect_area=5000,
        enable_visualization=enable_visualization,
        save_results=save_results,
        output_dir=args.output_dir,
        exposure_time=args.exposure,
        gain=args.gain,
        buffer_size=5,
        grab_strategy="LatestImageOnly",
        processing_fps=args.fps
    )
    
    return config


def print_system_info(config):
    """Print system configuration information."""
    print("\n🎥 Real-Time Defect Detection System")
    print("="*50)
    print(f"Reference Image: {config.reference_image_path}")
    print(f"Detection Mode: {'Fast' if config.enable_fast_mode else 'Full'}")
    print(f"Anomaly Threshold: {config.anomaly_threshold}")
    print(f"SSIM Threshold: {config.ssim_threshold}")
    print(f"Confidence Threshold: {config.confidence_threshold}")
    print(f"Processing FPS: {config.processing_fps}")
    print(f"Resize Factor: {config.resize_factor}")
    print(f"Visualization: {'Enabled' if config.enable_visualization else 'Disabled'}")
    print(f"Save Results: {'Enabled' if config.save_results else 'Disabled'}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Camera Exposure: {config.exposure_time} μs")
    print(f"Camera Gain: {config.gain}")
    print("="*50)


def main():
    """Main entry point."""
    if not SYSTEM_AVAILABLE:
        print("❌ System components not available")
        print("Please ensure all required files are present:")
        print("  - realtime_defect_detection.py")
        print("  - detection.py")
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle sample creation
    if args.create_sample:
        sample_path = create_sample_reference()
        print(f"\n✅ Sample reference image created: {sample_path}")
        print("You can now run: python run_detection.py sample_reference.jpg")
        return
    
    # Check if reference image is provided
    if not args.reference_image:
        print("❌ Reference image is required")
        print("Usage: python run_detection.py <reference_image_path>")
        print("Or create a sample: python run_detection.py --create-sample")
        sys.exit(1)
    
    # Validate reference image
    if not validate_reference_image(args.reference_image):
        print("\n💡 Tip: Create a sample reference image with:")
        print("  python run_detection.py --create-sample")
        sys.exit(1)
    
    # Create configuration
    config = create_config(args)
    
    # Print system information
    print_system_info(config)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Create controller
    controller = RealTimeController(config)
    
    # Register alert handler
    controller.register_defect_alert(defect_alert_handler)
    
    print("\n🚀 Starting real-time detection system...")
    print("Press Ctrl+C to stop or 'q' in visualization window")
    print()
    
    # Start system
    try:
        start_time = time.time()
        success = controller.start()
        
        if not success:
            print("❌ Failed to start real-time detection system")
            print("\nTroubleshooting tips:")
            print("1. Check if your camera is connected and accessible")
            print("2. Verify Pylon SDK is installed: pip install pypylon")
            print("3. Try different camera settings (--exposure, --gain)")
            print("4. Check system logs in the output directory")
            sys.exit(1)
        
        runtime = time.time() - start_time
        print(f"\n✅ System ran for {runtime:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check camera connection and permissions")
        print("2. Verify reference image format and quality")
        print("3. Check system resources (CPU, memory)")
        print("4. Review logs in the output directory")
        sys.exit(1)


if __name__ == "__main__":
    main() 