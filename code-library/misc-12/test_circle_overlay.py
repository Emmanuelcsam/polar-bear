#!/usr/bin/env python3
"""
Test script for Interactive Circle Overlay
Simulates a video stream to test the circle overlay functionality.
"""

import cv2
import numpy as np
from interactive_circle_overlay import InteractiveCircleOverlay


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with some visual elements"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some visual elements
    # Background gradient
    for y in range(height):
        for x in range(width):
            frame[y, x] = [
                int(255 * x / width),  # Blue gradient
                int(255 * y / height),  # Green gradient
                128  # Red constant
            ]
    
    # Add some shapes
    cv2.rectangle(frame, (50, 50), (150, 150), (255, 255, 255), 2)
    cv2.circle(frame, (400, 200), 80, (255, 255, 255), 2)
    cv2.line(frame, (100, 300), (500, 350), (255, 255, 255), 3)
    
    # Add text
    cv2.putText(frame, "Test Frame", (200, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def test_circle_overlay():
    """Test the circle overlay functionality"""
    print("Testing Interactive Circle Overlay")
    print("Controls:")
    print("  WASD: Move circle")
    print("  Q/E: Resize circle")
    print("  L: Lock/Unlock circle")
    print("  R: Reset circle")
    print("  ESC: Exit")
    print("\nPress any key to start...")
    
    # Create circle overlay
    circle_overlay = InteractiveCircleOverlay()
    
    # Create test frame
    frame = create_test_frame()
    
    # Main loop
    while True:
        # Draw circle overlay
        frame_with_circle = circle_overlay.draw_circle(frame)
        
        # Add instructions overlay
        frame_with_instructions = circle_overlay.add_instructions_overlay(
            frame_with_circle
        )
        
        # Convert to BGR for display
        display_frame = cv2.cvtColor(frame_with_instructions, cv2.COLOR_RGB2BGR)
        
        # Display frame
        cv2.imshow("Test Circle Overlay", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Key was pressed
            should_continue = circle_overlay.handle_keyboard_input(
                key, frame.shape
            )
            if not should_continue:
                break
        
        # Check if window was closed
        if cv2.getWindowProperty("Test Circle Overlay", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cv2.destroyAllWindows()
    print("Test completed!")


if __name__ == "__main__":
    test_circle_overlay() 