#!/usr/bin/env python3
"""
VS Code GitHub Copilot Auto-Continue Clicker - Enhanced Linux Edition
Robust script with multiple screen capture methods and X11/Wayland support
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import platform

# First, check and install dependencies
def check_and_install_dependencies():
    """Check for required packages and install if missing"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pyautogui': 'pyautogui',
        'mss': 'mss',
        'PIL': 'pillow',
        'pyscreenshot': 'pyscreenshot'
    }
    
    print("🔍 Checking dependencies...")
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'numpy':
                import numpy
            elif module == 'pyautogui':
                import pyautogui
            elif module == 'mss':
                import mss
            elif module == 'PIL':
                from PIL import Image
            elif module == 'pyscreenshot':
                import pyscreenshot
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencies installed successfully!")
            print("🔄 Please restart the script to use the newly installed packages.")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed!")

# Check dependencies before importing
check_and_install_dependencies()

# Now import everything
import cv2
import numpy as np
import pyautogui
import mss
import pyscreenshot
import logging
from PIL import Image
from typing import Optional, Tuple, List, Dict

# Setup logging
LOG_DIR = Path.home() / ".vscode-auto-clicker"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"clicker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration holder"""
    def __init__(self):
        self.scan_interval = 1.0
        self.click_cooldown = 3.0
        self.confidence_threshold = 0.8
        self.debug_mode = False
        self.save_screenshots = False
        self.color_tolerance = 20
        self.min_button_width = 60
        self.max_button_width = 200
        self.min_button_height = 20
        self.max_button_height = 60
        self.capture_method = 'auto'  # auto, mss, pyscreenshot, pyautogui
        
    def configure_interactively(self):
        """Interactive configuration"""
        print("\n🎯 VS Code Copilot Auto-Continue Clicker Configuration")
        print("=" * 60)
        print("Press Enter to use default values (shown in brackets)\n")
        
        # Scan interval
        interval = input(f"How often to scan for the button (seconds) [{self.scan_interval}]: ").strip()
        if interval:
            try:
                self.scan_interval = float(interval)
            except ValueError:
                print("Invalid input, using default.")
        
        # Click cooldown
        cooldown = input(f"Minimum time between clicks (seconds) [{self.click_cooldown}]: ").strip()
        if cooldown:
            try:
                self.click_cooldown = float(cooldown)
            except ValueError:
                print("Invalid input, using default.")
        
        # Debug mode
        debug = input("Enable debug mode? (y/n) [n]: ").strip().lower()
        self.debug_mode = debug == 'y'
        
        # Save screenshots
        screenshots = input("Save screenshots when button is found? (y/n) [n]: ").strip().lower()
        self.save_screenshots = screenshots == 'y'
        
        # Color tolerance
        tolerance = input(f"Color detection tolerance (0-50) [{self.color_tolerance}]: ").strip()
        if tolerance:
            try:
                self.color_tolerance = int(tolerance)
                self.color_tolerance = max(0, min(50, self.color_tolerance))
            except ValueError:
                print("Invalid input, using default.")
        
        # Capture method
        print("\nScreen capture methods:")
        print("  1. Auto (try all methods)")
        print("  2. MSS (fastest)")
        print("  3. PyScreenshot (most compatible)")
        print("  4. PyAutoGUI (reliable but slower)")
        method = input("Choose capture method [1]: ").strip()
        if method == '2':
            self.capture_method = 'mss'
        elif method == '3':
            self.capture_method = 'pyscreenshot'
        elif method == '4':
            self.capture_method = 'pyautogui'
        else:
            self.capture_method = 'auto'
        
        print("\n✅ Configuration complete!")
        print(f"   Scan interval: {self.scan_interval}s")
        print(f"   Click cooldown: {self.click_cooldown}s")
        print(f"   Debug mode: {'Yes' if self.debug_mode else 'No'}")
        print(f"   Save screenshots: {'Yes' if self.save_screenshots else 'No'}")
        print(f"   Color tolerance: {self.color_tolerance}")
        print(f"   Capture method: {self.capture_method}")
        print("=" * 60)

class ScreenCapture:
    """Multi-method screen capture with fallbacks"""
    
    def __init__(self, method='auto'):
        self.method = method
        self.sct = None
        self.working_method = None
        
        # Check display environment
        self._check_display_environment()
        
        # Initialize capture method
        if method == 'auto':
            self._find_working_method()
        else:
            self.working_method = method
            
    def _check_display_environment(self):
        """Check and fix display environment issues"""
        # Check if we're on Linux
        if platform.system() != 'Linux':
            return
            
        # Check DISPLAY variable
        if not os.environ.get('DISPLAY'):
            print("⚠️  DISPLAY not set. Trying to detect...")
            # Try common display values
            for display in [':0', ':1', ':0.0']:
                try:
                    os.environ['DISPLAY'] = display
                    # Test if it works
                    subprocess.check_output(['xdpyinfo'], stderr=subprocess.DEVNULL)
                    print(f"✅ Set DISPLAY to {display}")
                    break
                except:
                    continue
            else:
                print("❌ Could not detect DISPLAY. You may need to run with: DISPLAY=:0 python script.py")
        
        # Check if running under Wayland
        if os.environ.get('WAYLAND_DISPLAY'):
            print("⚠️  Running under Wayland. Some capture methods may not work.")
            print("   Consider running under X11 or enabling XWayland.")
            
        # Try to fix X11 authorization
        try:
            # Check if we can access X server
            subprocess.check_output(['xdpyinfo'], stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("⚠️  Cannot access X server. Trying to fix...")
            try:
                # Try xhost to allow local connections
                subprocess.run(['xhost', '+local:'], capture_output=True)
                print("✅ Enabled local X11 access")
            except:
                print("❌ Could not fix X11 access. You may need to run: xhost +local:")
    
    def _find_working_method(self):
        """Find a working capture method"""
        methods = ['mss', 'pyscreenshot', 'pyautogui']
        
        for method in methods:
            if self._test_method(method):
                self.working_method = method
                logger.info(f"Using screen capture method: {method}")
                break
        else:
            logger.error("No working screen capture method found!")
            
    def _test_method(self, method):
        """Test if a capture method works"""
        try:
            if method == 'mss':
                if not self.sct:
                    self.sct = mss.mss()
                screenshot = self.sct.grab(self.sct.monitors[0])
                img = np.array(screenshot)
                return img.size > 0
                
            elif method == 'pyscreenshot':
                img = pyscreenshot.grab()
                return img is not None
                
            elif method == 'pyautogui':
                img = pyautogui.screenshot()
                return img is not None
                
        except Exception as e:
            if self.method != 'auto':  # Only log if user specifically chose this method
                logger.debug(f"Method {method} failed: {e}")
            return False
            
        return False
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture screen using the working method"""
        if not self.working_method:
            return None
            
        try:
            if self.working_method == 'mss':
                if not self.sct:
                    self.sct = mss.mss()
                screenshot = self.sct.grab(self.sct.monitors[0])
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            elif self.working_method == 'pyscreenshot':
                img = pyscreenshot.grab()
                img_np = np.array(img)
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
            elif self.working_method == 'pyautogui':
                img = pyautogui.screenshot()
                img_np = np.array(img)
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            logger.error(f"Screen capture failed with {self.working_method}: {e}")
            # Try to find another working method
            if self.method == 'auto':
                logger.info("Trying to find alternative capture method...")
                self._find_working_method()
                
        return None

class ButtonDetector:
    """Detects blue Continue buttons on screen"""
    
    def __init__(self, config: Config):
        self.config = config
        self.capture = ScreenCapture(config.capture_method)
        self.screenshot_dir = LOG_DIR / "screenshots"
        if config.save_screenshots:
            self.screenshot_dir.mkdir(exist_ok=True)
        
        # Define multiple blue color ranges for robustness
        self.blue_ranges = [
            # Standard VS Code blue
            ([100, 100, 100], [130, 255, 255]),
            # Slightly darker blue
            ([95, 80, 80], [125, 255, 255]),
            # Slightly lighter blue
            ([105, 120, 120], [135, 255, 255]),
            # Alternative blue (some themes)
            ([90, 50, 50], [140, 255, 255]),
        ]
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture the entire screen"""
        return self.capture.capture()
    
    def find_blue_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find all blue regions that could be buttons"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Combine masks from all blue ranges
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.blue_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Apply color tolerance
            lower[0] = max(0, lower[0] - self.config.color_tolerance // 2)
            upper[0] = min(179, upper[0] + self.config.color_tolerance // 2)
            
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buttons = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.config.min_button_width <= w <= self.config.max_button_width and
                self.config.min_button_height <= h <= self.config.max_button_height):
                
                # Check aspect ratio (buttons are typically wider than tall)
                aspect_ratio = w / h
                if 1.2 <= aspect_ratio <= 5.0:
                    buttons.append((x, y, w, h))
                    if self.config.debug_mode:
                        logger.debug(f"Found blue region: x={x}, y={y}, w={w}, h={h}, aspect={aspect_ratio:.2f}")
        
        return buttons
    
    def is_continue_button(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Check if a blue region contains white text that could be 'Continue'"""
        # Extract the button region
        button_roi = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(button_roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold methods for robustness
        scores = []
        
        # Method 1: Simple threshold for white text
        _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_pixels1 = np.sum(thresh1 == 255)
        total_pixels = w * h
        white_ratio1 = white_pixels1 / total_pixels
        
        # Good buttons typically have 5-35% white pixels (text)
        if 0.05 <= white_ratio1 <= 0.35:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Method 2: Adaptive threshold (handles varying lighting)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        white_pixels2 = np.sum(thresh2 == 255)
        white_ratio2 = white_pixels2 / total_pixels
        
        if 0.10 <= white_ratio2 <= 0.40:
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Method 3: Check for text-like patterns
        # Text should have some horizontal structure
        horizontal_projection = np.sum(thresh1, axis=1)
        non_zero_rows = np.sum(horizontal_projection > 0)
        row_ratio = non_zero_rows / h
        
        # Text typically spans 30-80% of button height
        if 0.3 <= row_ratio <= 0.8:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # Method 4: Check button center has more white pixels (text usually centered)
        center_y = h // 3
        center_h = h // 3
        center_roi = thresh1[center_y:center_y+center_h, :]
        if center_roi.size > 0:
            center_white_ratio = np.sum(center_roi == 255) / center_roi.size
            if center_white_ratio > white_ratio1:
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Calculate final confidence
        confidence = sum(scores) / len(scores)
        
        if self.config.debug_mode:
            logger.debug(f"Button analysis: white_ratio={white_ratio1:.2f}, "
                        f"adaptive_ratio={white_ratio2:.2f}, row_ratio={row_ratio:.2f}, "
                        f"confidence={confidence:.2f}")
        
        return confidence
    
    def find_continue_button(self, image: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Find the Continue button in the image"""
        blue_regions = self.find_blue_regions(image)
        
        if not blue_regions:
            return None
        
        best_button = None
        best_confidence = 0.0
        
        for x, y, w, h in blue_regions:
            confidence = self.is_continue_button(image, x, y, w, h)
            
            if confidence > best_confidence and confidence >= self.config.confidence_threshold:
                best_confidence = confidence
                center_x = x + w // 2
                center_y = y + h // 2
                best_button = (center_x, center_y, confidence)
                
                if self.config.save_screenshots:
                    # Save screenshot with detection
                    debug_img = image.copy()
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(debug_img, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(debug_img, f"Conf: {confidence:.2f}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = self.screenshot_dir / f"detected_{timestamp}.png"
                    cv2.imwrite(str(filename), debug_img)
                    logger.info(f"Saved screenshot: {filename}")
        
        return best_button

class AutoClicker:
    """Main auto-clicker controller"""
    
    def __init__(self, config: Config):
        self.config = config
        self.detector = ButtonDetector(config)
        self.last_click_time = 0
        self.running = False
        self.stats = {
            'start_time': time.time(),
            'scans': 0,
            'detections': 0,
            'clicks': 0,
            'errors': 0,
            'capture_failures': 0
        }
        
        # Configure pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def can_click(self) -> bool:
        """Check if enough time has passed since last click"""
        return time.time() - self.last_click_time >= self.config.click_cooldown
    
    def click_button(self, x: int, y: int, confidence: float) -> bool:
        """Click the button at specified coordinates"""
        if not self.can_click():
            logger.debug("Click cooldown active, skipping click")
            return False
        
        try:
            # Smooth mouse movement
            current_x, current_y = pyautogui.position()
            distance = ((x - current_x)**2 + (y - current_y)**2)**0.5
            duration = min(0.3, distance / 1000)  # Faster for closer targets
            
            pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)
            time.sleep(0.05)  # Small pause before clicking
            pyautogui.click(x, y)
            
            self.last_click_time = time.time()
            self.stats['clicks'] += 1
            
            logger.info(f"✅ Clicked Continue button at ({x}, {y}) with confidence {confidence:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to click: {e}")
            self.stats['errors'] += 1
            return False
    
    def print_stats(self):
        """Print current statistics"""
        runtime = time.time() - self.stats['start_time']
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        
        print(f"\n📊 Statistics after {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"   Scans: {self.stats['scans']:,}")
        print(f"   Detections: {self.stats['detections']:,}")
        print(f"   Clicks: {self.stats['clicks']:,}")
        print(f"   Errors: {self.stats['errors']:,}")
        print(f"   Capture failures: {self.stats['capture_failures']:,}")
        print(f"   Detection rate: {self.stats['detections']/max(1, self.stats['scans'])*100:.1f}%")
    
    def run(self):
        """Main running loop"""
        print("\n🚀 Starting auto-clicker...")
        print(f"📁 Log file: {LOG_FILE}")
        print("🛑 Press Ctrl+C to stop\n")
        
        self.running = True
        consecutive_failures = 0
        
        try:
            while self.running:
                try:
                    # Capture screen
                    screen = self.detector.capture_screen()
                    if screen is None:
                        self.stats['capture_failures'] += 1
                        consecutive_failures += 1
                        
                        if consecutive_failures > 10:
                            print("\n⚠️  Multiple capture failures detected!")
                            print("Try these solutions:")
                            print("  1. Run with: DISPLAY=:0 python script.py")
                            print("  2. Run: xhost +local:")
                            print("  3. Switch to X11 if using Wayland")
                            print("  4. Check if VS Code is on the primary monitor")
                            consecutive_failures = 0  # Reset counter
                        
                        time.sleep(self.config.scan_interval)
                        continue
                    
                    consecutive_failures = 0  # Reset on success
                    self.stats['scans'] += 1
                    
                    # Look for button
                    result = self.detector.find_continue_button(screen)
                    
                    if result:
                        x, y, confidence = result
                        self.stats['detections'] += 1
                        logger.info(f"🎯 Found Continue button at ({x}, {y}) with confidence {confidence:.2f}")
                        
                        # Click it
                        if self.click_button(x, y, confidence):
                            # Longer wait after successful click
                            time.sleep(self.config.click_cooldown)
                        else:
                            time.sleep(self.config.scan_interval)
                    else:
                        # No button found, wait before next scan
                        time.sleep(self.config.scan_interval)
                    
                    # Print stats every 100 scans
                    if self.stats['scans'] % 100 == 0:
                        self.print_stats()
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")
                    self.stats['errors'] += 1
                    time.sleep(1)  # Brief pause on error
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping auto-clicker...")
        
        finally:
            self.running = False
            self.print_stats()
            print("\n✅ Auto-clicker stopped. Goodbye!")

def check_system_requirements():
    """Check system requirements and provide guidance"""
    print("\n🔍 Checking system requirements...")
    
    system = platform.system()
    print(f"   Operating System: {system}")
    
    if system == "Linux":
        # Check display server
        if os.environ.get('WAYLAND_DISPLAY'):
            print("   Display Server: Wayland (may have limitations)")
            print("   💡 Tip: For better compatibility, use X11")
        elif os.environ.get('DISPLAY'):
            print(f"   Display Server: X11 (DISPLAY={os.environ['DISPLAY']})")
        else:
            print("   Display Server: Unknown (DISPLAY not set)")
            print("   💡 Tip: Try running with: DISPLAY=:0 python script.py")
        
        # Check for useful tools
        tools = {
            'xdpyinfo': 'X11 display information',
            'xhost': 'X11 access control',
            'xwininfo': 'X window information'
        }
        
        print("\n   Available tools:")
        for tool, desc in tools.items():
            try:
                subprocess.check_output(['which', tool], stderr=subprocess.DEVNULL)
                print(f"     ✅ {tool} - {desc}")
            except:
                print(f"     ❌ {tool} - {desc} (install with: sudo apt install x11-utils)")
    
    print("")

def main():
    """Main entry point"""
    print("🤖 VS Code GitHub Copilot Auto-Continue Clicker")
    print("=" * 60)
    
    # Check system requirements
    check_system_requirements()
    
    # Create configuration
    config = Config()
    
    # Interactive configuration
    config.configure_interactively()
    
    # Confirm before starting
    input("\nPress Enter to start the auto-clicker...")
    
    # Create and run the clicker
    clicker = AutoClicker(config)
    clicker.run()

if __name__ == "__main__":
    main()
