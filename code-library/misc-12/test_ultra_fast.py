#!/usr/bin/env python3
"""
Ultra-Fast Circle Overlay Test
Demonstrates the maximum smoothness and speed improvements.
"""

import cv2
import numpy as np
import time
from circle_overlay import CircleOverlay


def test_ultra_fast_performance():
    """Test the ultra-fast circle overlay performance"""
    print("Ultra-Fast Circle Overlay Performance Test")
    print("=" * 50)
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.circle(frame, (400, 200), 80, (255, 255, 255), 2)
    cv2.putText(frame, "Ultra-Fast Test", (200, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create ultra-optimized circle overlay
    circle_overlay = CircleOverlay()
    circle_overlay.set_performance_mode(True)  # Enable ultra-fast mode
    
    print("Controls:")
    for control, description in circle_overlay.instructions.items():
        print(f"  {control}: {description}")
    print("\nPress any key to start...")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    
    print("Starting ultra-fast test...")
    print("Move the circle with WASD keys to test smoothness")
    print("Press ESC to exit")
    
    while True:
        frame_start = time.time()
        
        # Draw circle overlay (ultra-fast)
        frame_with_circle = circle_overlay.draw_circle(frame.copy())
        
        # Add minimal instructions overlay
        frame_with_instructions = circle_overlay.add_instructions_overlay(
            frame_with_circle
        )
        
        # Convert to BGR for display
        display_frame = cv2.cvtColor(frame_with_instructions, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow("Ultra-Fast Performance Test", display_frame)
        
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
            # Handle continuous input for maximum smoothness
            circle_overlay.handle_continuous_input(frame.shape)
        
        # Check if window was closed
        if cv2.getWindowProperty("Ultra-Fast Performance Test", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Performance monitoring
        frame_count += 1
        current_time = time.time()
        
        # Calculate and display FPS every second
        if current_time - last_fps_time >= 1.0:
            elapsed = current_time - last_fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f} | Frame: {frame_count} | "
                  f"Circle: ({circle_overlay.center[0]}, {circle_overlay.center[1]}) "
                  f"R={circle_overlay.radius}")
            frame_count = 0
            last_fps_time = current_time
    
    cv2.destroyAllWindows()
    
    # Final performance summary
    total_time = time.time() - start_time
    print(f"\nTest completed!")
    print(f"Total runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    test_ultra_fast_performance() 