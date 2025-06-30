#!/usr/bin/env python3
"""
ChatGPT CAPTCHA Handler - Production Version
Robust CAPTCHA detection and handling for ChatGPT automation
"""

import time
import logging
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    logger.info("PyAutoGUI available for GUI automation")
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not installed. GUI automation features disabled.")
    logger.info("To install on Ubuntu: sudo apt-get install python3-tk python3-dev")
    logger.info("Then: pip install pyautogui")

try:
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not installed. Browser automation features disabled.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.info("pytesseract not available. OCR features disabled.")


class CaptchaType(Enum):
    """Types of CAPTCHAs we can handle"""
    CHECKBOX = "checkbox"
    RECAPTCHA = "recaptcha"
    CLOUDFLARE = "cloudflare"
    CUSTOM = "custom"


@dataclass
class CaptchaLocation:
    """Store CAPTCHA location information"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    type: CaptchaType


class CaptchaHandler:
    """Production-ready CAPTCHA handler with multiple detection strategies"""
    
    def __init__(self, driver=None, debug: bool = False, save_screenshots: bool = True):
        """
        Initialize CAPTCHA handler
        
        Args:
            driver: Selenium WebDriver instance
            debug: Enable debug mode with detailed logging
            save_screenshots: Save screenshots for debugging
        """
        self.driver = driver
        self.debug = debug
        self.save_screenshots = save_screenshots
        self.attempts = 0
        self.max_attempts = 5
        self.screenshot_dir = "captcha_screenshots"
        
        if self.save_screenshots:
            import os
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
        # CAPTCHA patterns for different types
        self.captcha_patterns = {
            "checkbox": [
                "verify you are human",
                "i'm not a robot",
                "i am not a robot",
                "confirm you're human",
                "security check",
                "prove you're not a robot"
            ],
            "selectors": [
                # Common CAPTCHA selectors
                "iframe[title*='recaptcha']",
                "iframe[src*='recaptcha']",
                "iframe[title*='captcha']",
                "div.recaptcha-checkbox-border",
                "span#recaptcha-anchor",
                "div[role='checkbox']",
                "input[type='checkbox'][aria-label*='human']",
                "input[type='checkbox'][aria-label*='robot']",
                # Cloudflare
                "iframe[src*='challenges.cloudflare.com']",
                "div.cf-turnstile",
                # Custom checkboxes
                "label[for*='human']",
                "label[for*='robot']",
                "button[aria-label*='verify']"
            ]
        }
        
    def find_captcha_with_cv(self, screenshot: Image.Image) -> List[CaptchaLocation]:
        """
        Find CAPTCHA elements using computer vision
        
        Args:
            screenshot: PIL Image of the page
            
        Returns:
            List of potential CAPTCHA locations
        """
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        locations = []
        
        # Method 1: Template matching for common checkbox patterns
        checkbox_templates = self._get_checkbox_templates()
        for template in checkbox_templates:
            matches = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(matches >= threshold)
            
            for pt in zip(*loc[::-1]):
                h, w = template.shape
                locations.append(CaptchaLocation(
                    x=pt[0] + w//2,
                    y=pt[1] + h//2,
                    width=w,
                    height=h,
                    confidence=matches[pt[1], pt[0]],
                    type=CaptchaType.CHECKBOX
                ))
        
        # Method 2: Contour detection for checkbox-like shapes
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's roughly square and the right size
            if self._is_checkbox_shape(w, h):
                # Additional validation using color and content
                roi = gray[y:y+h, x:x+w]
                if self._validate_checkbox_content(roi):
                    locations.append(CaptchaLocation(
                        x=x + w//2,
                        y=y + h//2,
                        width=w,
                        height=h,
                        confidence=0.6,
                        type=CaptchaType.CHECKBOX
                    ))
        
        # Method 3: OCR-based detection if available
        if TESSERACT_AVAILABLE:
            text_locations = self._find_captcha_by_text(img_cv)
            locations.extend(text_locations)
        
        # Remove duplicates and sort by confidence
        locations = self._deduplicate_locations(locations)
        locations.sort(key=lambda l: l.confidence, reverse=True)
        
        if self.debug and locations:
            self._save_debug_image(img_cv, locations)
            
        return locations
    
    def _get_checkbox_templates(self) -> List[np.ndarray]:
        """Generate common checkbox templates"""
        templates = []
        
        # Empty square checkbox
        size = 20
        template = np.ones((size, size), dtype=np.uint8) * 255
        template[2:-2, 2:-2] = 0
        template[4:-4, 4:-4] = 255
        templates.append(template)
        
        # Rounded checkbox
        template = np.ones((size, size), dtype=np.uint8) * 255
        cv2.circle(template, (size//2, size//2), size//2-2, 0, 2)
        templates.append(template)
        
        return templates
    
    def _is_checkbox_shape(self, width: int, height: int) -> bool:
        """Check if dimensions match typical checkbox size"""
        if 15 <= width <= 30 and 15 <= height <= 30:
            aspect_ratio = width / height
            return 0.8 <= aspect_ratio <= 1.2
        return False
    
    def _validate_checkbox_content(self, roi: np.ndarray) -> bool:
        """Validate if ROI contains checkbox-like content"""
        if roi.size == 0:
            return False
            
        # Check for square-like edges
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = np.count_nonzero(edges) / roi.size
        
        # Checkboxes typically have clear edges
        return 0.1 <= edge_ratio <= 0.4
    
    def _find_captcha_by_text(self, image: np.ndarray) -> List[CaptchaLocation]:
        """Find CAPTCHA by detecting nearby text"""
        locations = []
        
        try:
            # Get text and positions
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i, text in enumerate(data['text']):
                if not text:
                    continue
                    
                text_lower = text.lower()
                # Check for CAPTCHA-related text
                for pattern in self.captcha_patterns["checkbox"]:
                    if any(word in text_lower for word in pattern.split()):
                        # Get text position
                        x = data['left'][i]
                        y = data['top'][i]
                        h = data['height'][i]
                        
                        # Checkbox is usually to the left of text
                        checkbox_x = max(0, x - 50)
                        checkbox_y = y + h // 2
                        
                        locations.append(CaptchaLocation(
                            x=checkbox_x,
                            y=checkbox_y,
                            width=20,
                            height=20,
                            confidence=0.7,
                            type=CaptchaType.CHECKBOX
                        ))
                        break
                        
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            
        return locations
    
    def _deduplicate_locations(self, locations: List[CaptchaLocation]) -> List[CaptchaLocation]:
        """Remove duplicate/overlapping locations"""
        if not locations:
            return []
            
        unique_locations = []
        
        for loc in locations:
            is_duplicate = False
            for unique_loc in unique_locations:
                # Check if locations overlap
                distance = ((loc.x - unique_loc.x)**2 + (loc.y - unique_loc.y)**2)**0.5
                if distance < 20:  # Within 20 pixels
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if loc.confidence > unique_loc.confidence:
                        unique_locations.remove(unique_loc)
                        unique_locations.append(loc)
                    break
                    
            if not is_duplicate:
                unique_locations.append(loc)
                
        return unique_locations
    
    def _save_debug_image(self, image: np.ndarray, locations: List[CaptchaLocation]):
        """Save debug image with detected locations marked"""
        debug_img = image.copy()
        
        for loc in locations:
            # Draw rectangle
            cv2.rectangle(
                debug_img,
                (loc.x - loc.width//2, loc.y - loc.height//2),
                (loc.x + loc.width//2, loc.y + loc.height//2),
                (0, 255, 0),
                2
            )
            # Add confidence score
            cv2.putText(
                debug_img,
                f"{loc.confidence:.2f}",
                (loc.x - 20, loc.y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/captcha_debug_{timestamp}.png"
        cv2.imwrite(filename, debug_img)
        logger.debug(f"Debug image saved to: {filename}")
    
    def click_with_selenium(self, element=None, location: CaptchaLocation = None) -> bool:
        """Click using Selenium with various strategies"""
        if not self.driver or not SELENIUM_AVAILABLE:
            return False
            
        try:
            # Strategy 1: Click provided element
            if element:
                self._safe_click(element)
                return True
                
            # Strategy 2: Try common selectors
            for selector in self.captcha_patterns["selectors"]:
                try:
                    element = self._find_and_switch_to_captcha_frame(selector)
                    if element:
                        self._safe_click(element)
                        return True
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
                    
            # Strategy 3: Click by coordinates if location provided
            if location:
                return self._click_by_coordinates(location.x, location.y)
                
            # Strategy 4: JavaScript injection
            return self._try_javascript_click()
            
        except Exception as e:
            logger.error(f"Selenium click error: {e}")
            return False
            
    def _find_and_switch_to_captcha_frame(self, selector: str):
        """Find element, switching to iframe if necessary"""
        # First try in main document
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element
        except NoSuchElementException:
            pass
            
        # Try in iframes
        if 'iframe' in selector:
            try:
                iframe = self.driver.find_element(By.CSS_SELECTOR, selector)
                self.driver.switch_to.frame(iframe)
                time.sleep(1)
                
                # Look for checkbox in iframe
                checkbox_selectors = [
                    "div.recaptcha-checkbox-border",
                    "span#recaptcha-anchor",
                    "div[role='checkbox']"
                ]
                
                for cb_selector in checkbox_selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, cb_selector)
                        return element
                    except NoSuchElementException:
                        continue
                        
            finally:
                # Always switch back to main content
                self.driver.switch_to.default_content()
                
        return None
        
    def _safe_click(self, element):
        """Click element with multiple strategies"""
        try:
            # Strategy 1: Regular click
            element.click()
        except Exception:
            try:
                # Strategy 2: JavaScript click
                self.driver.execute_script("arguments[0].click();", element)
            except Exception:
                # Strategy 3: Action chains
                actions = ActionChains(self.driver)
                actions.move_to_element(element).click().perform()
                
    def _click_by_coordinates(self, x: int, y: int) -> bool:
        """Click at specific coordinates"""
        try:
            # Account for any scroll offset
            scroll_x = self.driver.execute_script("return window.scrollX;")
            scroll_y = self.driver.execute_script("return window.scrollY;")
            
            # Adjust coordinates
            adjusted_x = x - scroll_x
            adjusted_y = y - scroll_y
            
            # Click using action chains
            actions = ActionChains(self.driver)
            actions.move_by_offset(adjusted_x, adjusted_y).click().perform()
            actions.reset_actions()
            
            return True
        except Exception as e:
            logger.debug(f"Coordinate click failed: {e}")
            return False
            
    def _try_javascript_click(self) -> bool:
        """Try to click CAPTCHA using JavaScript"""
        js_scripts = [
            # Generic checkbox click
            """
            var checkboxes = document.querySelectorAll('input[type="checkbox"], div[role="checkbox"]');
            for (var cb of checkboxes) {
                var rect = cb.getBoundingClientRect();
                if (rect.width > 10 && rect.width < 40) {
                    cb.click();
                    return true;
                }
            }
            return false;
            """,
            # reCAPTCHA specific
            """
            var frames = document.querySelectorAll('iframe[src*="recaptcha"]');
            for (var frame of frames) {
                try {
                    var doc = frame.contentDocument || frame.contentWindow.document;
                    var checkbox = doc.querySelector('.recaptcha-checkbox-border');
                    if (checkbox) {
                        checkbox.click();
                        return true;
                    }
                } catch(e) {}
            }
            return false;
            """,
            # Cloudflare specific
            """
            var cf = document.querySelector('input[type="checkbox"][name="cf-turnstile-response"]');
            if (cf) {
                cf.click();
                return true;
            }
            return false;
            """
        ]
        
        for script in js_scripts:
            try:
                result = self.driver.execute_script(script)
                if result:
                    return True
            except Exception as e:
                logger.debug(f"JS script failed: {e}")
                
        return False
        
    def wait_for_captcha(self, timeout: int = 10) -> bool:
        """Wait for CAPTCHA to appear with smart detection"""
        if not self.driver:
            return False
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_captcha_present():
                logger.info("CAPTCHA detected!")
                return True
            time.sleep(0.5)
            
        return False
        
    def is_captcha_present(self) -> bool:
        """Enhanced CAPTCHA detection"""
        if not self.driver:
            return False
            
        # Check page text
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            for pattern in self.captcha_patterns["checkbox"]:
                if pattern in page_text:
                    return True
        except Exception:
            pass
            
        # Check for CAPTCHA elements
        for selector in self.captcha_patterns["selectors"][:5]:  # Check first 5 selectors
            try:
                self.driver.find_element(By.CSS_SELECTOR, selector)
                return True
            except NoSuchElementException:
                continue
                
        # Check page title
        try:
            if "security check" in self.driver.title.lower():
                return True
        except Exception:
            pass
            
        return False
        
    def is_captcha_solved(self) -> bool:
        """Check if CAPTCHA has been solved"""
        if not self.driver:
            return True
            
        # Check if we're past the CAPTCHA page
        try:
            # Look for signs we're on the main site
            main_site_indicators = [
                "nav[aria-label='Chat history']",  # ChatGPT specific
                "main",
                "div[role='main']"
            ]
            
            for indicator in main_site_indicators:
                try:
                    self.driver.find_element(By.CSS_SELECTOR, indicator)
                    return True
                except NoSuchElementException:
                    continue
                    
            # Check if CAPTCHA elements are gone
            return not self.is_captcha_present()
            
        except Exception:
            return False
            
    def handle_captcha(self, max_wait: int = 30) -> bool:
        """Main method to handle CAPTCHA with multiple strategies"""
        logger.info("🔍 Starting CAPTCHA detection and handling...")
        
        # Wait for CAPTCHA to appear
        if not self.wait_for_captcha():
            logger.info("✅ No CAPTCHA detected")
            return True
            
        success = False
        start_time = time.time()
        
        while self.attempts < self.max_attempts and time.time() - start_time < max_wait:
            self.attempts += 1
            logger.info(f"🎯 CAPTCHA handling attempt {self.attempts}/{self.max_attempts}")
            
            # Strategy 1: Try Selenium selectors first (fastest)
            if self.driver and self.click_with_selenium():
                logger.info("✅ Clicked CAPTCHA with Selenium")
                success = True
                
            # Strategy 2: Computer vision approach
            else:
                screenshot = self._get_screenshot()
                if screenshot:
                    locations = self.find_captcha_with_cv(screenshot)
                    
                    for location in locations[:3]:  # Try top 3 locations
                        logger.info(f"📍 Trying location: ({location.x}, {location.y}) "
                                  f"confidence: {location.confidence:.2f}")
                        
                        if self.driver:
                            if self.click_with_selenium(location=location):
                                success = True
                                break
                        elif PYAUTOGUI_AVAILABLE:
                            if self._click_with_pyautogui(location.x, location.y):
                                success = True
                                break
                                
            if success:
                # Wait for CAPTCHA to process
                time.sleep(3)
                
                # Check if solved
                if self.is_captcha_solved():
                    logger.info("✅ CAPTCHA solved successfully!")
                    return True
                else:
                    logger.info("⏳ CAPTCHA clicked but not yet solved, retrying...")
                    success = False
                    
            time.sleep(2)  # Wait before retry
            
        if not success:
            logger.warning(f"❌ Could not solve CAPTCHA after {self.attempts} attempts")
            
        return success
        
    def _get_screenshot(self) -> Optional[Image.Image]:
        """Get screenshot from Selenium or PyAutoGUI"""
        screenshot = None
        
        if self.driver:
            try:
                screenshot_png = self.driver.get_screenshot_as_png()
                screenshot = Image.open(io.BytesIO(screenshot_png))
            except Exception as e:
                logger.error(f"Selenium screenshot failed: {e}")
                
        elif PYAUTOGUI_AVAILABLE:
            try:
                screenshot = pyautogui.screenshot()
            except Exception as e:
                logger.error(f"PyAutoGUI screenshot failed: {e}")
                
        return screenshot
        
    def _click_with_pyautogui(self, x: int, y: int) -> bool:
        """Click using PyAutoGUI with safety checks"""
        if not PYAUTOGUI_AVAILABLE:
            return False
            
        try:
            # Safety check - don't click outside screen
            screen_width, screen_height = pyautogui.size()
            if 0 <= x <= screen_width and 0 <= y <= screen_height:
                pyautogui.moveTo(x, y, duration=0.5)
                time.sleep(0.2)
                pyautogui.click()
                return True
            else:
                logger.warning(f"Click coordinates ({x}, {y}) outside screen bounds")
                return False
        except Exception as e:
            logger.error(f"PyAutoGUI click error: {e}")
            return False


def integrate_with_scraper(driver, max_wait: int = 30) -> bool:
    """
    Integration function for gptscraper.py
    
    Args:
        driver: Selenium WebDriver instance
        max_wait: Maximum time to wait for CAPTCHA resolution
        
    Returns:
        bool: True if no CAPTCHA or successfully handled
    """
    handler = CaptchaHandler(driver, debug=True)
    return handler.handle_captcha(max_wait)


# Standalone testing
if __name__ == "__main__":
    import sys
    
    print("🤖 ChatGPT CAPTCHA Handler - Standalone Test Mode")
    print("=" * 50)
    
    # Check dependencies
    print("\n📋 Dependency Status:")
    print(f"  Selenium: {'✅ Available' if SELENIUM_AVAILABLE else '❌ Not installed'}")
    print(f"  PyAutoGUI: {'✅ Available' if PYAUTOGUI_AVAILABLE else '❌ Not installed'}")
    print(f"  Tesseract: {'✅ Available' if TESSERACT_AVAILABLE else '❌ Not installed'}")
    
    if len(sys.argv) > 1:
        # Test with screenshot file
        screenshot_path = sys.argv[1]
        print(f"\n🔍 Analyzing screenshot: {screenshot_path}")
        
        try:
            img = Image.open(screenshot_path)
            handler = CaptchaHandler(debug=True)
            locations = handler.find_captcha_with_cv(img)
            
            print(f"\n📊 Found {len(locations)} potential CAPTCHA locations:")
            for i, loc in enumerate(locations):
                print(f"  {i+1}. Position: ({loc.x}, {loc.y})")
                print(f"     Size: {loc.width}x{loc.height}")
                print(f"     Confidence: {loc.confidence:.2%}")
                print(f"     Type: {loc.type.value}")
                
        except Exception as e:
            print(f"❌ Error analyzing screenshot: {e}")
    else:
        print("\n💡 Usage:")
        print("  Integration: Use integrate_with_scraper() in your script")
        print("  Testing: python captcha_handler.py screenshot.png")
