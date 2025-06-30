import pyautogui
import cv2
import numpy as np
import time
import threading
from PIL import ImageGrab
import sys

class ContinueButtonClicker:
    def __init__(self):
        self.running = False
        self.click_delay = 1.0  # Delay between clicks in seconds
        self.scan_interval = 0.5  # How often to scan the screen
        self.last_click_time = 0
        
        # Blue color range for the Continue button (in BGR format for OpenCV)
        # These values target the specific blue in your screenshots
        self.lower_blue = np.array([200, 100, 0])  # Lower bound of blue
        self.upper_blue = np.array([255, 180, 100])  # Upper bound of blue
        
        # Fail-safe: Move mouse to corner to stop
        pyautogui.FAILSAFE = True
        
    def find_continue_button(self, screenshot):
        """Find the blue Continue button in the screenshot"""
        # Convert PIL image to OpenCV format
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Create mask for blue color
        mask = cv2.inRange(screenshot_bgr, self.lower_blue, self.upper_blue)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the contour has button-like dimensions
            # Continue button is typically wider than tall
            if w > 80 and h > 25 and h < 60 and w/h > 1.5 and w/h < 5:
                # Check if it contains white text (high brightness in the blue region)
                button_region = screenshot_bgr[y:y+h, x:x+w]
                
                # Convert to grayscale and check for bright pixels (white text)
                gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
                white_pixels = np.sum(gray > 200)
                total_pixels = w * h
                
                # If there's enough white (text), it's likely our button
                if white_pixels / total_pixels > 0.1:
                    return (x + w//2, y + h//2)  # Return center of button
        
        return None
    
    def click_button(self, x, y):
        """Click the button at the given coordinates"""
        current_time = time.time()
        if current_time - self.last_click_time >= self.click_delay:
            pyautogui.click(x, y)
            self.last_click_time = current_time
            print(f"Clicked Continue button at ({x}, {y})")
            return True
        return False
    
    def scan_and_click(self):
        """Main loop to scan and click"""
        while self.running:
            try:
                # Take screenshot
                screenshot = ImageGrab.grab()
                
                # Find the button
                button_pos = self.find_continue_button(screenshot)
                
                if button_pos:
                    self.click_button(button_pos[0], button_pos[1])
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the auto-clicker"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.scan_and_click)
            self.thread.daemon = True
            self.thread.start()
            print("Auto-clicker started. Move mouse to top-left corner to stop.")
    
    def stop(self):
        """Stop the auto-clicker"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        print("Auto-clicker stopped.")

def main():
    print("=== Blue Continue Button Auto-Clicker ===")
    print("\nThis script will automatically click blue 'Continue' buttons.")
    print("Move your mouse to the top-left corner of the screen to stop.\n")
    
    # Create clicker instance
    clicker = ContinueButtonClicker()
    
    # Configuration options
    print("Configuration:")
    print(f"- Click delay: {clicker.click_delay}s")
    print(f"- Scan interval: {clicker.scan_interval}s\n")
    
    try:
        # Start the clicker
        clicker.start()
        
        # Keep the script running
        print("Press Ctrl+C to stop the script.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping auto-clicker...")
        clicker.stop()
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    # Check dependencies
    try:
        import pyautogui
        import cv2
        from PIL import ImageGrab
    except ImportError as e:
        print("Missing required dependencies. Please install:")
        print("pip install pyautogui opencv-python pillow numpy")
        sys.exit(1)
    
    main()