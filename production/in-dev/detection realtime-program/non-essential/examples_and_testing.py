#!/usr/bin/env python3
"""
Usage Examples and Testing

This script provides complete usage examples for the real-time defect detection system,
showing different configurations and use cases.
"""

import time
import sys
from pathlib import Path
import logging
import cv2
import numpy as np

# Import our modules
try:
    from enhanced_pylon_grabber import EnhancedPylonFrameGrabber
    from realtime_detector import RealTimeDetector, RealTimeConfig
    from realtime_controller import RealTimeController
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    MODULES_AVAILABLE = False


def create_sample_reference_image():
    """Create a sample reference image for testing."""
    # Create a simple reference pattern
    reference = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Add some pattern
    cv2.circle(reference, (320, 240), 100, (255, 255, 255), -1)
    cv2.rectangle(reference, (200, 150), (440, 330), (200, 200, 200), 2)
    
    # Add some texture
    for i in range(10):
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        cv2.circle(reference, (x, y), 5, (180, 180, 180), -1)
    
    # Save reference image
    ref_path = "sample_reference.jpg"
    cv2.imwrite(ref_path, reference)
    print(f"Sample reference image created: {ref_path}")
    return ref_path


def example_1_basic_usage():
    """Example 1: Basic real-time detection with default settings."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Real-Time Detection")
    print("="*60)
    
    # Create sample reference image
    reference_path = create_sample_reference_image()
    
    # Basic configuration
    controller = RealTimeController(
        reference_image_path=reference_path,
        processing_fps=10.0,
        visualization=True,
        save_results=True,
        output_dir="example1_output"
    )
    
    print("Starting basic real-time detection system...")
    print("Press Ctrl+C to stop")
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\nStopping...")


def example_2_high_performance():
    """Example 2: High-performance configuration for fast processing."""
    print("\n" + "="*60)
    print("EXAMPLE 2: High-Performance Configuration")
    print("="*60)
    
    reference_path = create_sample_reference_image()
    
    # High-performance configuration
    controller = RealTimeController(
        reference_image_path=reference_path,
        processing_fps=30.0,  # High processing rate
        visualization=False,  # Disable visualization for speed
        save_results=False,   # Disable saving for speed
        output_dir="example2_output"
    )
    
    # Custom alert handler for defects
    def high_speed_alert(result, frame):
        if result.confidence > 0.8:
            print(f"⚠️  HIGH CONFIDENCE DEFECT: {result.confidence:.3f}")
    
    controller.register_defect_alert(high_speed_alert)
    
    print("Starting high-performance detection system...")
    print("Processing at 30 FPS with alerts only")
    print("Press Ctrl+C to stop")
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\nStopping...")


def example_3_custom_detector_config():
    """Example 3: Custom detector configuration with specific thresholds."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Detector Configuration")
    print("="*60)
    
    reference_path = create_sample_reference_image()
    
    # Create detector with custom config
    custom_config = RealTimeConfig(
        reference_image_path=reference_path,
        anomaly_threshold=1.5,      # More sensitive
        ssim_threshold=0.9,         # Higher similarity requirement
        confidence_threshold=0.3,   # Lower confidence threshold
        enable_fast_mode=False,     # Use full detection
        resize_factor=0.75,         # Slightly smaller for speed
        min_defect_area=10,         # Smaller defects
        max_defect_area=2000,       # Smaller max area
        enable_visualization=True,
        save_detections=True,
        output_dir="example3_output"
    )
    
    detector = RealTimeDetector(custom_config)
    
    # Test with frame grabber
    grabber = EnhancedPylonFrameGrabber(
        buffer_size=3,
        grab_strategy="LatestImageOnly"
    )
    
    if grabber.initialize_camera(exposure_time=8000):  # 8ms exposure
        grabber.start()
        
        if grabber.wait_for_initialization():
            print("Custom detector test - Processing 50 frames...")
            
            for i in range(50):
                frame, metadata = grabber.read_latest_frame()
                if frame is not None:
                    result = detector.detect_defects(frame, i)
                    
                    print(f"Frame {i}: Anomalous={result.is_anomalous}, "
                          f"SSIM={result.ssim_score:.3f}, "
                          f"Confidence={result.confidence:.3f}, "
                          f"Defects={result.defect_count}")
                
                time.sleep(0.1)  # 10 FPS
        
        grabber.stop()
        grabber.join()
    
    print("Custom detector test completed")


def example_4_standalone_components():
    """Example 4: Using components independently."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Standalone Component Usage")
    print("="*60)
    
    reference_path = create_sample_reference_image()
    
    print("Testing Enhanced Pylon Frame Grabber independently...")
    
    # Test frame grabber alone
    grabber = EnhancedPylonFrameGrabber(buffer_size=5)
    
    if grabber.initialize_camera():
        grabber.start()
        
        if grabber.wait_for_initialization():
            print("Frame grabber test - monitoring for 10 seconds...")
            
            for i in range(10):
                time.sleep(1)
                stats = grabber.get_statistics()
                
                print(f"Second {i+1}: FPS={stats['fps']:.1f}, "
                      f"Frames={stats.get('total_frames', 0)}, "
                      f"Errors={stats['error_count']}")
                
                frame, metadata = grabber.read_latest_frame()
                if frame is not None:
                    print(f"  Frame shape: {frame.shape}, "
                          f"Timestamp: {metadata.get('timestamp', 0):.3f}")
        
        grabber.stop()
        grabber.join()
    
    print("\nTesting Real-Time Detector independently...")
    
    # Test detector alone with sample frames
    config = RealTimeConfig(
        reference_image_path=reference_path,
        enable_fast_mode=True
    )
    
    detector = RealTimeDetector(config)
    
    # Create test frames
    reference_img = cv2.imread(reference_path)
    
    # Test with identical frame (should be OK)
    result1 = detector.detect_defects(reference_img, 1)
    print(f"Identical frame: Anomalous={result1.is_anomalous}, SSIM={result1.ssim_score:.3f}")
    
    # Test with modified frame (should detect anomaly)
    modified_frame = reference_img.copy()
    cv2.rectangle(modified_frame, (100, 100), (200, 200), (0, 0, 255), -1)  # Add red square
    
    result2 = detector.detect_defects(modified_frame, 2)
    print(f"Modified frame: Anomalous={result2.is_anomalous}, SSIM={result2.ssim_score:.3f}")
    
    print("Standalone component tests completed")


def example_5_continuous_monitoring():
    """Example 5: Continuous monitoring with data logging."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Continuous Monitoring with Logging")
    print("="*60)
    
    reference_path = create_sample_reference_image()
    
    # Setup detailed logging
    log_file = "continuous_monitoring.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ContinuousMonitoring")
    
    controller = RealTimeController(
        reference_image_path=reference_path,
        processing_fps=5.0,  # Moderate rate for stability
        visualization=True,
        save_results=True,
        output_dir="continuous_monitoring_output"
    )
    
    # Advanced alert system
    defect_history = []
    
    def monitoring_alert(result, frame):
        defect_history.append({
            'timestamp': result.timestamp,
            'confidence': result.confidence,
            'defect_count': result.defect_count,
            'ssim_score': result.ssim_score
        })
        
        # Keep only last 100 records
        if len(defect_history) > 100:
            defect_history.pop(0)
        
        # Log significant events
        if result.confidence > 0.8:
            logger.warning(f"HIGH CONFIDENCE DEFECT: {result.confidence:.3f}")
        
        # Check for trends
        if len(defect_history) >= 10:
            recent_defects = sum(1 for d in defect_history[-10:] if d['confidence'] > 0.5)
            if recent_defects >= 5:
                logger.critical("TREND ALERT: High defect rate in recent frames!")
    
    controller.register_defect_alert(monitoring_alert)
    
    print("Starting continuous monitoring system...")
    print(f"Logging to: {log_file}")
    print("Press Ctrl+C to stop")
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print(f"\nStopping... Logged {len(defect_history)} detection events")
        
        # Print summary
        if defect_history:
            avg_confidence = sum(d['confidence'] for d in defect_history) / len(defect_history)
            max_confidence = max(d['confidence'] for d in defect_history)
            total_defects = sum(d['defect_count'] for d in defect_history)
            
            print(f"Summary: Avg confidence={avg_confidence:.3f}, "
                  f"Max confidence={max_confidence:.3f}, "
                  f"Total defects={total_defects}")


def main():
    """Main function with example selection."""
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please check imports.")
        return
    
    print("🎥 Real-Time Defect Detection Examples")
    print("=====================================")
    
    examples = {
        '1': ('Basic Usage', example_1_basic_usage),
        '2': ('High Performance', example_2_high_performance),
        '3': ('Custom Configuration', example_3_custom_detector_config),
        '4': ('Standalone Components', example_4_standalone_components),
        '5': ('Continuous Monitoring', example_5_continuous_monitoring)
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        
        choice = input("\nSelect example (1-5): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        
        try:
            func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Invalid choice: {choice}")
        print("Valid options: 1, 2, 3, 4, 5")


if __name__ == "__main__":
    main()