#!/usr/bin/env python3
"""
Performance Test Script
Demonstrates the improvements in circle overlay smoothness and speed.
"""

import time
import cv2
import numpy as np
from circle_overlay import CircleOverlay
from live_feed import LiveFeed
from main import IntegratedCoreDetector


def test_circle_overlay_performance():
    """Test circle overlay performance improvements"""
    print("=== Circle Overlay Performance Test ===")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.circle(frame, (400, 200), 80, (255, 255, 255), 2)
    cv2.putText(frame, "Performance Test Frame", (200, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create optimized circle overlay
    circle_overlay = CircleOverlay()
    circle_overlay.set_performance_mode(True)  # High performance mode
    
    print("Testing optimized circle overlay...")
    print("Controls: WASD to move, Q/E to resize, L to lock/unlock, R to reset")
    print("Press ESC to exit")
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    
    while True:
        frame_start = time.time()
        
        # Draw circle overlay
        frame_with_circle = circle_overlay.draw_circle(frame)
        
        # Add instructions overlay
        frame_with_instructions = circle_overlay.add_instructions_overlay(
            frame_with_circle
        )
        
        # Convert to BGR for display
        display_frame = cv2.cvtColor(frame_with_instructions, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow("Performance Test", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        circle_overlay.update_pressed_keys(key)
        
        if key != 255:  # Key was pressed
            should_continue = circle_overlay.handle_keyboard_input(
                key, frame.shape
            )
            if not should_continue:
                break
        else:
            # Handle continuous input for smooth movement
            circle_overlay.handle_continuous_input(frame.shape)
        
        # Check if window was closed
        if cv2.getWindowProperty("Performance Test", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Performance monitoring
        frame_count += 1
        current_time = time.time()
        
        if frame_count % 60 == 0:
            elapsed = current_time - last_fps_time
            fps = 60 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}")
            last_fps_time = current_time
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    cv2.destroyAllWindows()
    print(f"\nPerformance Test Results:")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("Test completed!")


def test_live_feed_performance():
    """Test live feed performance improvements"""
    print("\n=== Live Feed Performance Test ===")
    
    try:
        # Create optimized live feed
        live_feed = LiveFeed(
            camera_index=0,
            use_pylon=False,
            auto_detect=True
        )
        
        # Set high performance mode
        live_feed.set_performance_mode(True)
        
        print("Testing optimized live feed...")
        print("Press ESC to exit")
        
        # Run for a limited time to test performance
        start_time = time.time()
        test_duration = 10  # 10 seconds
        
        def test_callback(frame):
            # Add performance info to frame
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = test_duration - elapsed
            
            cv2.putText(frame, f"Test Time: {elapsed:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Remaining: {remaining:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if elapsed >= test_duration:
                live_feed.is_running = False
            
            return frame
        
        live_feed.frame_callback = test_callback
        
        # Run live feed
        live_feed.run(
            window_name="Live Feed Performance Test",
            show_info=True
        )
        
        # Get performance info
        info = live_feed.get_camera_info()
        print(f"\nLive Feed Performance Results:")
        print(f"Total frames: {info['frame_count']}")
        print(f"Average FPS: {info['fps']:.1f}")
        print(f"Target FPS: {info['target_fps']}")
        print(f"Frame skips: {info['frame_skip_count']}")
        print(f"Camera: {info['camera_name']}")
        
    except Exception as e:
        print(f"Error in live feed test: {e}")


def test_integrated_performance():
    """Test integrated application performance"""
    print("\n=== Integrated Application Performance Test ===")
    
    try:
        # Create optimized integrated detector
        detector = IntegratedCoreDetector(
            camera_index=0,
            use_pylon=False
        )
        
        # Set high performance mode
        detector.set_performance_mode(True)
        
        print("Testing optimized integrated application...")
        print("Controls: WASD to move circle, Q/E to resize, L to lock/unlock")
        print("Press ESC to exit")
        
        # Run for a limited time
        start_time = time.time()
        test_duration = 10  # 10 seconds
        
        def test_callback(frame):
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = test_duration - elapsed
            
            cv2.putText(frame, f"Test Time: {elapsed:.1f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Remaining: {remaining:.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if elapsed >= test_duration:
                detector.live_feed.is_running = False
            
            return frame
        
        detector.live_feed.frame_callback = test_callback
        
        # Run integrated application
        detector.run()
        
        # Get performance info
        info = detector.get_system_info()
        print(f"\nIntegrated Application Performance Results:")
        print(f"Total frames: {info['performance']['frame_count']}")
        print(f"Average FPS: {info['performance']['fps']:.1f}")
        print(f"Target FPS: {info['performance']['target_fps']}")
        print(f"Camera: {info['camera']['camera_name']}")
        print(f"Circle position: {info['circle']['center']}")
        print(f"Circle radius: {info['circle']['radius']}")
        
    except Exception as e:
        print(f"Error in integrated test: {e}")


def main():
    """Main performance test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Test Suite")
    parser.add_argument("--test", choices=["circle", "live", "integrated", "all"], 
                       default="all", help="Which test to run")
    parser.add_argument("--no-camera", action="store_true", 
                       help="Skip camera-dependent tests")
    
    args = parser.parse_args()
    
    print("Performance Test Suite")
    print("Testing optimized circle overlay and live feed performance")
    print("=" * 50)
    
    if args.test == "circle" or args.test == "all":
        test_circle_overlay_performance()
    
    if args.test == "live" or args.test == "all":
        if not args.no_camera:
            test_live_feed_performance()
        else:
            print("\nSkipping live feed test (--no-camera specified)")
    
    if args.test == "integrated" or args.test == "all":
        if not args.no_camera:
            test_integrated_performance()
        else:
            print("\nSkipping integrated test (--no-camera specified)")
    
    print("\nPerformance test suite completed!")
    print("Key improvements:")
    print("- Frame rate optimization with target FPS control")
    print("- Cached overlays to reduce redundant drawing")
    print("- Optimized keyboard input with key repeat")
    print("- Reduced frame copying for better performance")
    print("- Adaptive processing intervals based on performance mode")


if __name__ == "__main__":
    main() 