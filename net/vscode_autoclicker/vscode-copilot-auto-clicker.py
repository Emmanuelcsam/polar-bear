#!/usr/bin/env python3
"""
VS Code GitHub Copilot Auto-Continue Script
Automatically detects and clicks "Continue" buttons in VS Code Copilot dialogs
"""

import cv2
import numpy as np
import pyautogui
import time
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import mss
import threading
from typing import Optional, Tuple, List

# Configure logging
LOG_DIR = Path.home() / ".vscode-auto-clicker"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"auto_clicker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Safety settings
    FAILSAFE = True
    PAUSE_BETWEEN_ACTIONS = 0.1
    
    # Detection settings
    SCAN_INTERVAL = 0.5  # seconds between scans
    CONFIDENCE_THRESHOLD = 0.85  # Higher threshold for more accuracy
    
    # Click safety
    CLICK_COOLDOWN = 3.0  # seconds between clicks
    
    # Debug mode
    DEBUG_MODE = False  # Set to True to save screenshots
    
    # Button characteristics for VS Code Copilot Continue button
    BUTTON_MIN_WIDTH = 70
    BUTTON_MAX_WIDTH = 150
    BUTTON_MIN_HEIGHT = 25
    BUTTON_MAX_HEIGHT = 50

# Initialize pyautogui safety
pyautogui.FAILSAFE = Config.FAILSAFE
pyautogui.PAUSE = Config.PAUSE_BETWEEN_ACTIONS

class ContinueButtonDetector:
    """Detect VS Code Copilot Continue button"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.last_click_time = 0
        self.debug_dir = LOG_DIR / "debug"
        if Config.DEBUG_MODE:
            self.debug_dir.mkdir(exist_ok=True)
    
    def capture_screen(self) -> np.ndarray:
        """Capture screenshot of all monitors"""
        try:
            screenshot = self.sct.grab(self.sct.monitors[0])
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def find_blue_buttons(self, screenshot: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find blue button regions in the screenshot"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        
        # Define range for VS Code Copilot blue button
        # These values are calibrated for the typical VS Code blue
        lower_blue = np.array([100, 130, 100])  # Adjusted for VS Code blue
        upper_blue = np.array([120, 255, 255])
        
        # Create mask for blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buttons = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size - must be button-sized
            if (Config.BUTTON_MIN_WIDTH <= w <= Config.BUTTON_MAX_WIDTH and
                Config.BUTTON_MIN_HEIGHT <= h <= Config.BUTTON_MAX_HEIGHT):
                
                # Check aspect ratio (buttons are typically wider than tall)
                aspect_ratio = w / h
                if 1.5 <= aspect_ratio <= 4.0:
                    buttons.append((x, y, w, h))
                    logger.debug(f"Found potential button: x={x}, y={y}, w={w}, h={h}")
        
        return buttons
    
    def verify_continue_button(self, screenshot: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Verify if the button region likely contains 'Continue' text"""
        # Extract the button region
        button_roi = screenshot[y:y+h, x:x+w]
        
        # Check for text-like patterns (white text on blue background)
        gray = cv2.cvtColor(button_roi, cv2.COLOR_BGR2GRAY)
        
        # Look for white/light pixels (text)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_pixel_ratio = np.sum(white_mask == 255) / (w * h * 255)
        
        # Continue button should have some white text (5-40% of area)
        if 0.05 <= white_pixel_ratio <= 0.4:
            logger.debug(f"Button verification passed: white_pixel_ratio={white_pixel_ratio:.2f}")
            return True
        
        return False
    
    def detect_continue_button(self, screenshot: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the Continue button in the screenshot"""
        buttons = self.find_blue_buttons(screenshot)
        
        if not buttons:
            return None
        
        # Look for the best candidate
        for x, y, w, h in buttons:
            if self.verify_continue_button(screenshot, x, y, w, h):
                center_x = x + w // 2
                center_y = y + h // 2
                
                if Config.DEBUG_MODE:
                    # Save debug image
                    debug_img = screenshot.copy()
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    debug_path = self.debug_dir / f"detected_{int(time.time())}.png"
                    cv2.imwrite(str(debug_path), debug_img)
                    logger.debug(f"Saved debug image: {debug_path}")
                
                return (center_x, center_y)
        
        return None
    
    def can_click(self) -> bool:
        """Check if enough time has passed since last click"""
        return time.time() - self.last_click_time >= Config.CLICK_COOLDOWN
    
    def click_button(self, x: int, y: int) -> bool:
        """Click the button at the specified coordinates"""
        try:
            if not self.can_click():
                logger.debug("Click cooldown active, skipping click")
                return False
            
            # Move and click
            pyautogui.moveTo(x, y, duration=0.1)
            time.sleep(0.05)
            pyautogui.click(x, y)
            
            self.last_click_time = time.time()
            logger.info(f"Clicked Continue button at ({x}, {y})")
            
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

class AutoClicker:
    """Main auto-clicker application"""
    
    def __init__(self):
        self.detector = ContinueButtonDetector()
        self.running = False
        self.stats = {
            'scans': 0,
            'detections': 0,
            'clicks': 0,
            'start_time': time.time()
        }
    
    def scan_loop(self):
        """Main scanning loop"""
        logger.info("Scanning for VS Code Copilot Continue button...")
        
        while self.running:
            try:
                # Capture screen
                screenshot = self.detector.capture_screen()
                if screenshot is None:
                    time.sleep(Config.SCAN_INTERVAL)
                    continue
                
                self.stats['scans'] += 1
                
                # Detect continue button
                button_pos = self.detector.detect_continue_button(screenshot)
                
                if button_pos:
                    self.stats['detections'] += 1
                    logger.info(f"Continue button detected at {button_pos}")
                    
                    # Click the button
                    if self.detector.click_button(button_pos[0], button_pos[1]):
                        self.stats['clicks'] += 1
                        # Wait after clicking
                        time.sleep(Config.CLICK_COOLDOWN)
                    else:
                        time.sleep(Config.SCAN_INTERVAL)
                else:
                    time.sleep(Config.SCAN_INTERVAL)
                
                # Print stats every 60 seconds
                if self.stats['scans'] % 120 == 0:  # ~60 seconds at 0.5s interval
                    self.print_stats()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                time.sleep(1)
    
    def print_stats(self):
        """Print statistics"""
        runtime = time.time() - self.stats['start_time']
        logger.info(
            f"Stats - Runtime: {runtime:.0f}s, "
            f"Scans: {self.stats['scans']}, "
            f"Detections: {self.stats['detections']}, "
            f"Clicks: {self.stats['clicks']}"
        )
    
    def start(self):
        """Start the auto-clicker"""
        logger.info("Starting VS Code Copilot Auto-Clicker...")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("Press Ctrl+C to stop, or move mouse to top-left corner")
        logger.info(f"Click cooldown: {Config.CLICK_COOLDOWN}s")
        
        self.running = True
        self.stats['start_time'] = time.time()
        self.scan_loop()
    
    def stop(self):
        """Stop the auto-clicker"""
        self.running = False
        self.print_stats()
        logger.info("Auto-clicker stopped")

def test_detection():
    """Test detection without clicking"""
    logger.info("Running detection test...")
    detector = ContinueButtonDetector()
    
    # Enable debug mode for test
    Config.DEBUG_MODE = True
    
    for i in range(5):
        screenshot = detector.capture_screen()
        if screenshot is None:
            logger.error("Failed to capture screen")
            return
        
        button_pos = detector.detect_continue_button(screenshot)
        if button_pos:
            logger.info(f"Test {i+1}: Button found at {button_pos}")
        else:
            logger.info(f"Test {i+1}: No button detected")
        
        time.sleep(1)
    
    logger.info("Test complete. Check debug folder for screenshots.")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VS Code Copilot Auto-Continue Script")
    parser.add_argument('--test', action='store_true', help='Run detection test without clicking')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with screenshots')
    parser.add_argument('--interval', type=float, default=0.5, help='Scan interval in seconds')
    parser.add_argument('--cooldown', type=float, default=3.0, help='Click cooldown in seconds')
    
    args = parser.parse_args()
    
    # Apply arguments
    if args.debug:
        Config.DEBUG_MODE = True
        logger.info("Debug mode enabled")
    
    Config.SCAN_INTERVAL = args.interval
    Config.CLICK_COOLDOWN = args.cooldown
    
    # Check dependencies
    try:
        import cv2
        import pyautogui
        import mss
        import numpy
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install opencv-python pyautogui mss numpy")
        sys.exit(1)
    
    if args.test:
        test_detection()
    else:
        try:
            clicker = AutoClicker()
            clicker.start()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if 'clicker' in locals():
                clicker.stop()

if __name__ == "__main__":
    main()
