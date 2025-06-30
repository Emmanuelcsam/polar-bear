#!/usr/bin/env python3
"""
ChatGPT CAPTCHA Handler
Detects and clicks "Verify you are human" checkboxes using computer vision
"""

import time
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Try different automation libraries
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("Warning: pyautogui not installed. Install with: pip install pyautogui")

try:
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class CaptchaHandler:
    def __init__(self, driver=None, debug=False):
        """
        Initialize CAPTCHA handler
        
        Args:
            driver: Selenium WebDriver instance (optional)
            debug: Save debug images
        """
        self.driver = driver
        self.debug = debug
        self.attempts = 0
        self.max_attempts = 3
        
    def find_checkbox_opencv(self, screenshot):
        """
        Find checkbox using OpenCV template matching
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Define what to look for (empty checkbox patterns)
        # These are common checkbox patterns
        checkbox_features = [
            # Square checkbox pattern
            np.array([[255, 255, 255, 255, 255],
                     [255, 0, 0, 0, 255],
                     [255, 0, 0, 0, 255],
                     [255, 0, 0, 0, 255],
                     [255, 255, 255, 255, 255]], dtype=np.uint8),
            
            # Rounded checkbox pattern
            np.array([[0, 255, 255, 255, 0],
                     [255, 0, 0, 0, 255],
                     [255, 0, 0, 0, 255],
                     [255, 0, 0, 0, 255],
                     [0, 255, 255, 255, 0]], dtype=np.uint8),
        ]
        
        # Try edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that look like checkboxes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_checkboxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's roughly square and the right size
            if 15 < w < 30 and 15 < h < 30 and 0.8 < w/h < 1.2:
                # Check if it's near text "Verify you are human"
                roi = gray[max(0, y-50):min(gray.shape[0], y+h+50), 
                          max(0, x-100):min(gray.shape[1], x+w+200)]
                
                if roi.size > 0:
                    potential_checkboxes.append((x + w//2, y + h//2))
        
        if self.debug:
            debug_img = img_cv.copy()
            for x, y in potential_checkboxes:
                cv2.circle(debug_img, (x, y), 20, (0, 255, 0), 2)
            cv2.imwrite(f'captcha_debug_{self.attempts}.png', debug_img)
        
        return potential_checkboxes
    
    def find_checkbox_by_text(self, screenshot):
        """
        Find checkbox by looking for "Verify you are human" text
        """
        # Use OCR to find text (requires pytesseract)
        try:
            import pytesseract
            
            # Convert to grayscale for better OCR
            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            
            # Get text and positions
            data = pytesseract.image_to_data(img_cv, output_type=pytesseract.Output.DICT)
            
            # Look for "Verify" or "human" text
            for i, text in enumerate(data['text']):
                if text and ('verify' in text.lower() or 'human' in text.lower()):
                    # Get position
                    x = data['left'][i]
                    y = data['top'][i]
                    
                    # Checkbox is usually to the left of the text
                    checkbox_x = x - 50
                    checkbox_y = y + data['height'][i] // 2
                    
                    return [(checkbox_x, checkbox_y)]
                    
        except ImportError:
            print("pytesseract not available for text detection")
            
        return []
    
    def click_with_selenium(self, element=None):
        """
        Click using Selenium WebDriver
        """
        if not self.driver or not SELENIUM_AVAILABLE:
            return False
            
        try:
            if element:
                element.click()
                return True
                
            # Try to find checkbox/iframe elements
            selectors = [
                "iframe[title*='recaptcha']",
                "iframe[src*='recaptcha']",
                "div.recaptcha-checkbox-border",
                "span#recaptcha-anchor",
                "div[role='checkbox']",
                "input[type='checkbox']"
            ]
            
            for selector in selectors:
                try:
                    # Check if it's in an iframe
                    if 'iframe' in selector:
                        iframe = self.driver.find_element(By.CSS_SELECTOR, selector)
                        self.driver.switch_to.frame(iframe)
                        time.sleep(1)
                        
                        # Now look for the checkbox
                        checkbox = self.driver.find_element(By.CSS_SELECTOR, "div.recaptcha-checkbox-border")
                        checkbox.click()
                        
                        # Switch back to main frame
                        self.driver.switch_to.default_content()
                        return True
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        element.click()
                        return True
                        
                except NoSuchElementException:
                    continue
                    
            # Try JavaScript click as fallback
            try:
                self.driver.execute_script("""
                    var checkboxes = document.querySelectorAll('input[type="checkbox"], div[role="checkbox"]');
                    for (var i = 0; i < checkboxes.length; i++) {
                        var rect = checkboxes[i].getBoundingClientRect();
                        if (rect.width > 10 && rect.width < 40 && rect.height > 10 && rect.height < 40) {
                            checkboxes[i].click();
                            return true;
                        }
                    }
                    return false;
                """)
                return True
            except:
                pass
                
        except Exception as e:
            print(f"Selenium click error: {e}")
            
        return False
    
    def click_with_pyautogui(self, x, y):
        """
        Click using PyAutoGUI
        """
        if not PYAUTOGUI_AVAILABLE:
            return False
            
        try:
            # Move to position and click
            pyautogui.moveTo(x, y, duration=0.5)
            time.sleep(0.5)
            pyautogui.click()
            return True
        except Exception as e:
            print(f"PyAutoGUI click error: {e}")
            return False
    
    def handle_captcha(self):
        """
        Main method to handle CAPTCHA
        """
        print("🔍 Checking for CAPTCHA...")
        
        while self.attempts < self.max_attempts:
            self.attempts += 1
            
            # Method 1: Try Selenium first (most reliable)
            if self.driver and self.click_with_selenium():
                print("✅ Clicked CAPTCHA checkbox with Selenium")
                time.sleep(3)  # Wait for verification
                return True
            
            # Method 2: Screenshot and computer vision
            screenshot = None
            if self.driver:
                # Get screenshot from Selenium
                screenshot_png = self.driver.get_screenshot_as_png()
                screenshot = Image.open(io.BytesIO(screenshot_png))
            elif PYAUTOGUI_AVAILABLE:
                # Get screenshot from PyAutoGUI
                screenshot = pyautogui.screenshot()
            
            if screenshot:
                # Find checkbox positions
                positions = self.find_checkbox_opencv(screenshot)
                if not positions:
                    positions = self.find_checkbox_by_text(screenshot)
                
                # Try clicking each potential position
                for x, y in positions:
                    print(f"🎯 Attempting to click at ({x}, {y})")
                    
                    if self.driver:
                        # Convert to Selenium coordinates and click
                        try:
                            actions = ActionChains(self.driver)
                            actions.move_by_offset(x, y).click().perform()
                            actions.reset_actions()
                            time.sleep(3)
                            return True
                        except:
                            pass
                    elif PYAUTOGUI_AVAILABLE:
                        if self.click_with_pyautogui(x, y):
                            time.sleep(3)
                            return True
            
            print(f"⚠️  Attempt {self.attempts}/{self.max_attempts} failed")
            time.sleep(2)
        
        return False
    
    def is_captcha_present(self):
        """
        Check if CAPTCHA is present on the page
        """
        if not self.driver:
            return False
            
        # Check for common CAPTCHA indicators
        indicators = [
            "verify you are human",
            "i'm not a robot",
            "recaptcha",
            "security check",
            "captcha"
        ]
        
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            return any(indicator in page_text for indicator in indicators)
        except:
            return False


def integrate_with_main_script(driver):
    """
    Integration function to be called from main script
    
    Usage in gptscraper.py:
    
    from captcha_handler import integrate_with_main_script
    
    # In your login method, after navigating to the page:
    if integrate_with_main_script(self.driver):
        print("CAPTCHA handled successfully")
    else:
        print("Could not handle CAPTCHA")
    """
    handler = CaptchaHandler(driver, debug=True)
    
    # Check if CAPTCHA is present
    if handler.is_captcha_present():
        print("🤖 CAPTCHA detected!")
        return handler.handle_captcha()
    
    return True  # No CAPTCHA present


# Standalone mode for testing
if __name__ == "__main__":
    print("ChatGPT CAPTCHA Handler - Standalone Mode")
    print("="*50)
    
    if SELENIUM_AVAILABLE:
        print("✅ Selenium available - best for integration")
    else:
        print("❌ Selenium not available - install for best results")
        
    if PYAUTOGUI_AVAILABLE:
        print("✅ PyAutoGUI available - can work standalone")
    else:
        print("❌ PyAutoGUI not available - install with: pip install pyautogui")
    
    print("\nTo use this with gptscraper.py:")
    print("1. Place this file in the same directory")
    print("2. The script will automatically import and use it")
    print("\nFor manual testing:")
    print("1. Take a screenshot when CAPTCHA appears")
    print("2. Run: python captcha_handler.py screenshot.png")
    
    import sys
    if len(sys.argv) > 1:
        # Test with a screenshot file
        from PIL import Image
        img = Image.open(sys.argv[1])
        handler = CaptchaHandler(debug=True)
        positions = handler.find_checkbox_opencv(img)
        print(f"\nFound {len(positions)} potential checkbox positions:")
        for x, y in positions:
            print(f"  - ({x}, {y})")
