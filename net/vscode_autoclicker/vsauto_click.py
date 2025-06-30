#!/usr/bin/env python3
"""
VS Code GitHub Copilot Auto-Continue Script - Linux Enhanced Edition
Automatically detects and clicks "Continue" buttons in VS Code Copilot dialogs

Enhanced features:
- Automatic X11 authorization fixing for Linux
- Multiple screen capture backends
- Better error handling and recovery
- Robust detection with fallback mechanisms
"""

import sys
import os
import time
import logging
import argparse
import threading
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import signal

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
    SCAN_INTERVAL = 0.5
    CONFIDENCE_THRESHOLD = 0.85
    
    # Click safety
    CLICK_COOLDOWN = 3.0
    
    # Debug mode
    DEBUG_MODE = False
    DRY_RUN = False
    
    # Button characteristics
    BUTTON_MIN_WIDTH = 70
    BUTTON_MAX_WIDTH = 150
    BUTTON_MIN_HEIGHT = 25
    BUTTON_MAX_HEIGHT = 50
    
    # Backend preference order
    CAPTURE_BACKENDS = ['mss', 'pyscreenshot', 'scrot', 'gnome-screenshot']

# X11 Authorization Handler
class X11AuthHandler:
    """Handle X11 authorization issues on Linux"""
    
    @staticmethod
    def fix_x11_auth():
        """Try to fix X11 authorization issues"""
        logger.info("Attempting to fix X11 authorization...")
        
        # Method 1: Try to get XAUTHORITY from the current user's session
        try:
            # Get the current user
            user = os.environ.get('USER', '')
            
            # Try to find the .Xauthority file
            xauth_locations = [
                Path.home() / '.Xauthority',
                Path(f'/run/user/{os.getuid()}/gdm/Xauthority'),
                Path('/tmp/.gdm*') if Path('/tmp').exists() else None
            ]
            
            for loc in xauth_locations:
                if loc and loc.exists():
                    os.environ['XAUTHORITY'] = str(loc)
                    logger.info(f"Set XAUTHORITY to: {loc}")
                    return True
                    
        except Exception as e:
            logger.debug(f"XAUTHORITY search failed: {e}")
        
        # Method 2: Try xhost
        try:
            subprocess.run(['xhost', '+local:'], capture_output=True, check=False)
            logger.info("Enabled local X11 access with xhost")
            return True
        except Exception as e:
            logger.debug(f"xhost failed: {e}")
        
        # Method 3: Get DISPLAY from loginctl
        try:
            result = subprocess.run(
                ['loginctl', 'show-session', '-p', 'Display'],
                capture_output=True, text=True, check=False
            )
            if result.returncode == 0 and 'Display=' in result.stdout:
                display = result.stdout.strip().split('=')[1]
                if display:
                    os.environ['DISPLAY'] = display
                    logger.info(f"Set DISPLAY to: {display}")
                    return True
        except Exception as e:
            logger.debug(f"loginctl failed: {e}")
        
        return False

# Screen capture backends
class ScreenCapture:
    """Multi-backend screen capture with fallbacks"""
    
    def __init__(self):
        self.backend = None
        self.sct = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the best available backend"""
        for backend in Config.CAPTURE_BACKENDS:
            if self._try_backend(backend):
                self.backend = backend
                logger.info(f"Using screen capture backend: {backend}")
                break
        else:
            logger.error("No screen capture backend available!")
    
    def _try_backend(self, backend: str) -> bool:
        """Try to initialize a specific backend"""
        try:
            if backend == 'mss':
                import mss
                self.sct = mss.mss()
                # Test capture
                self.sct.grab(self.sct.monitors[0])
                return True
                
            elif backend == 'pyscreenshot':
                import pyscreenshot
                # Test capture
                pyscreenshot.grab()
                return True
                
            elif backend == 'scrot':
                # Check if scrot is installed
                result = subprocess.run(['which', 'scrot'], capture_output=True)
                return result.returncode == 0
                
            elif backend == 'gnome-screenshot':
                # Check if gnome-screenshot is installed
                result = subprocess.run(['which', 'gnome-screenshot'], capture_output=True)
                return result.returncode == 0
                
        except Exception as e:
            logger.debug(f"Backend {backend} failed: {e}")
        
        return False
    
    def capture(self) -> Optional['np.ndarray']:
        """Capture screen using the available backend"""
        try:
            import numpy as np
            import cv2
            
            if self.backend == 'mss' and self.sct:
                screenshot = self.sct.grab(self.sct.monitors[0])
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            elif self.backend == 'pyscreenshot':
                import pyscreenshot
                from PIL import Image
                screenshot = pyscreenshot.grab()
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            elif self.backend == 'scrot':
                # Use scrot to capture to temp file
                temp_file = Path('/tmp/vscode_clicker_temp.png')
                subprocess.run(['scrot', '-z', str(temp_file)], check=True)
                img = cv2.imread(str(temp_file))
                temp_file.unlink(missing_ok=True)
                return img
                
            elif self.backend == 'gnome-screenshot':
                # Use gnome-screenshot to capture
                temp_file = Path('/tmp/vscode_clicker_temp.png')
                subprocess.run(
                    ['gnome-screenshot', '-f', str(temp_file)],
                    check=True, capture_output=True
                )
                img = cv2.imread(str(temp_file))
                temp_file.unlink(missing_ok=True)
                return img
                
        except Exception as e:
            logger.error(f"Screen capture failed ({self.backend}): {e}")
        
        return None

# Mouse control wrapper
class MouseController:
    """Mouse control with fallback methods"""
    
    def __init__(self):
        self.backend = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize mouse control backend"""
        try:
            import pyautogui
            pyautogui.FAILSAFE = Config.FAILSAFE
            pyautogui.PAUSE = Config.PAUSE_BETWEEN_ACTIONS
            # Test mouse functions
            pyautogui.position()
            self.backend = 'pyautogui'
            logger.info("Using pyautogui for mouse control")
        except Exception as e:
            logger.warning(f"pyautogui not available: {e}")
            # Try xdotool as fallback
            try:
                subprocess.run(['which', 'xdotool'], check=True, capture_output=True)
                self.backend = 'xdotool'
                logger.info("Using xdotool for mouse control")
            except:
                logger.warning("No mouse control available - dry run mode")
                Config.DRY_RUN = True
    
    def move_and_click(self, x: int, y: int) -> bool:
        """Move mouse and click at position"""
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] Would click at ({x}, {y})")
            return True
        
        try:
            if self.backend == 'pyautogui':
                import pyautogui
                pyautogui.moveTo(x, y, duration=0.1)
                time.sleep(0.05)
                pyautogui.click(x, y)
                return True
                
            elif self.backend == 'xdotool':
                # Move mouse
                subprocess.run(['xdotool', 'mousemove', str(x), str(y)], check=True)
                time.sleep(0.1)
                # Click
                subprocess.run(['xdotool', 'click', '1'], check=True)
                return True
                
        except Exception as e:
            logger.error(f"Mouse click failed: {e}")
        
        return False

class ContinueButtonDetector:
    """Detect VS Code Copilot Continue button"""
    
    def __init__(self):
        self.capture = ScreenCapture()
        self.debug_dir = LOG_DIR / "debug"
        if Config.DEBUG_MODE:
            self.debug_dir.mkdir(exist_ok=True)
        
        # Import cv2 and numpy here
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
        except ImportError:
            logger.error("OpenCV and NumPy required for detection")
            raise
    
    def find_blue_buttons(self, screenshot) -> List[Tuple[int, int, int, int]]:
        """Find blue button regions in the screenshot"""
        # Convert to HSV
        hsv = self.cv2.cvtColor(screenshot, self.cv2.COLOR_BGR2HSV)
        
        # Blue color range (adjusted for VS Code)
        lower_blue = self.np.array([100, 130, 100])
        upper_blue = self.np.array([120, 255, 255])
        
        # Create mask
        mask = self.cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Clean up mask
        kernel = self.np.ones((3, 3), self.np.uint8)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        
        buttons = []
        for contour in contours:
            x, y, w, h = self.cv2.boundingRect(contour)
            
            # Filter by size
            if (Config.BUTTON_MIN_WIDTH <= w <= Config.BUTTON_MAX_WIDTH and
                Config.BUTTON_MIN_HEIGHT <= h <= Config.BUTTON_MAX_HEIGHT):
                
                # Check aspect ratio
                aspect_ratio = w / h
                if 1.5 <= aspect_ratio <= 4.0:
                    buttons.append((x, y, w, h))
                    logger.debug(f"Found potential button: x={x}, y={y}, w={w}, h={h}")
        
        return buttons
    
    def verify_continue_button(self, screenshot, x: int, y: int, w: int, h: int) -> bool:
        """Verify if the button region likely contains 'Continue' text"""
        button_roi = screenshot[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = self.cv2.cvtColor(button_roi, self.cv2.COLOR_BGR2GRAY)
        
        # Look for white/light pixels (text)
        _, white_mask = self.cv2.threshold(gray, 200, 255, self.cv2.THRESH_BINARY)
        white_pixel_ratio = self.np.sum(white_mask == 255) / (w * h * 255)
        
        # Button should have white text
        if 0.05 <= white_pixel_ratio <= 0.4:
            logger.debug(f"Button verification passed: white_ratio={white_pixel_ratio:.2f}")
            return True
        
        return False
    
    def detect_continue_button(self, screenshot) -> Optional[Tuple[int, int]]:
        """Detect the Continue button in the screenshot"""
        if screenshot is None:
            return None
        
        buttons = self.find_blue_buttons(screenshot)
        
        for x, y, w, h in buttons:
            if self.verify_continue_button(screenshot, x, y, w, h):
                center_x = x + w // 2
                center_y = y + h // 2
                
                if Config.DEBUG_MODE:
                    # Save debug image
                    debug_img = screenshot.copy()
                    self.cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    self.cv2.circle(debug_img, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    debug_path = self.debug_dir / f"detected_{int(time.time())}.png"
                    self.cv2.imwrite(str(debug_path), debug_img)
                    logger.debug(f"Saved debug image: {debug_path}")
                
                return (center_x, center_y)
        
        return None

class AutoClicker:
    """Main auto-clicker application"""
    
    def __init__(self):
        self.detector = ContinueButtonDetector()
        self.mouse = MouseController()
        self.last_click_time = 0
        self.running = False
        self.stats = {
            'scans': 0,
            'detections': 0,
            'clicks': 0,
            'errors': 0,
            'start_time': time.time()
        }
    
    def can_click(self) -> bool:
        """Check if enough time has passed since last click"""
        return time.time() - self.last_click_time >= Config.CLICK_COOLDOWN
    
    def click_button(self, x: int, y: int) -> bool:
        """Click the button at the specified coordinates"""
        if not self.can_click():
            logger.debug("Click cooldown active")
            return False
        
        if self.mouse.move_and_click(x, y):
            self.last_click_time = time.time()
            self.stats['clicks'] += 1
            logger.info(f"Clicked Continue button at ({x}, {y})")
            return True
        
        return False
    
    def scan_loop(self):
        """Main scanning loop"""
        logger.info("Scanning for VS Code Copilot Continue button...")
        
        while self.running:
            try:
                # Capture screen
                screenshot = self.detector.capture.capture()
                
                if screenshot is None:
                    self.stats['errors'] += 1
                    time.sleep(Config.SCAN_INTERVAL)
                    continue
                
                self.stats['scans'] += 1
                
                # Detect button
                button_pos = self.detector.detect_continue_button(screenshot)
                
                if button_pos:
                    self.stats['detections'] += 1
                    logger.info(f"Continue button detected at {button_pos}")
                    
                    # Click the button
                    if self.click_button(button_pos[0], button_pos[1]):
                        time.sleep(Config.CLICK_COOLDOWN)
                    else:
                        time.sleep(Config.SCAN_INTERVAL)
                else:
                    time.sleep(Config.SCAN_INTERVAL)
                
                # Print stats periodically
                if self.stats['scans'] % 120 == 0:
                    self.print_stats()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                self.stats['errors'] += 1
                time.sleep(1)
    
    def print_stats(self):
        """Print statistics"""
        runtime = time.time() - self.stats['start_time']
        logger.info(
            f"Stats - Runtime: {runtime:.0f}s, "
            f"Scans: {self.stats['scans']}, "
            f"Detections: {self.stats['detections']}, "
            f"Clicks: {self.stats['clicks']}, "
            f"Errors: {self.stats['errors']}"
        )
    
    def start(self):
        """Start the auto-clicker"""
        logger.info("Starting VS Code Copilot Auto-Clicker...")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("Press Ctrl+C to stop")
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Set up signal handler
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        
        self.scan_loop()
    
    def stop(self):
        """Stop the auto-clicker"""
        self.running = False
        self.print_stats()
        logger.info("Auto-clicker stopped")

def check_dependencies() -> Dict[str, bool]:
    """Check all dependencies"""
    deps = {}
    
    # Python packages
    packages = ['cv2', 'numpy', 'mss', 'pyautogui', 'pyscreenshot']
    for pkg in packages:
        try:
            __import__(pkg)
            deps[pkg] = True
        except ImportError:
            deps[pkg] = False
    
    # System tools
    tools = ['scrot', 'xdotool', 'gnome-screenshot']
    for tool in tools:
        result = subprocess.run(['which', tool], capture_output=True)
        deps[tool] = result.returncode == 0
    
    return deps

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VS Code Copilot Auto-Continue Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     # Normal operation
  %(prog)s --fix-x11          # Fix X11 issues first
  %(prog)s --dry-run          # Test without clicking
  %(prog)s --debug            # Save debug screenshots
  %(prog)s --check-deps       # Check dependencies
        """
    )
    
    parser.add_argument('--test', action='store_true', 
                       help='Run detection test without clicking')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with screenshots')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run without actually clicking')
    parser.add_argument('--interval', type=float, default=0.5, 
                       help='Scan interval in seconds (default: 0.5)')
    parser.add_argument('--cooldown', type=float, default=3.0, 
                       help='Click cooldown in seconds (default: 3.0)')
    parser.add_argument('--fix-x11', action='store_true', 
                       help='Try to fix X11 authorization issues')
    parser.add_argument('--check-deps', action='store_true', 
                       help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        deps = check_dependencies()
        print("\nDependency Status:")
        print("-" * 40)
        for name, status in deps.items():
            status_str = "✓ Installed" if status else "✗ Missing"
            print(f"{name:20} {status_str}")
        
        print("\nRequired packages:")
        print("  pip install opencv-python numpy mss pyautogui pillow pyscreenshot")
        print("\nOptional tools:")
        print("  sudo apt install scrot xdotool gnome-screenshot")
        return
    
    # Apply configuration
    Config.DEBUG_MODE = args.debug
    Config.DRY_RUN = args.dry_run or args.test
    Config.SCAN_INTERVAL = args.interval
    Config.CLICK_COOLDOWN = args.cooldown
    
    # Fix X11 if requested
    if args.fix_x11:
        X11AuthHandler.fix_x11_auth()
    
    # Check core dependencies
    try:
        import cv2
        import numpy
    except ImportError:
        logger.error("Missing required dependencies!")
        logger.error("Install with: pip install opencv-python numpy mss pyautogui")
        sys.exit(1)
    
    # Run the clicker
    try:
        clicker = AutoClicker()
        clicker.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
