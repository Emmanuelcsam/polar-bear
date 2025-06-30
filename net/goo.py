import time
import numpy as np
import cv2
import pytesseract
from PIL import Image
import pyautogui

# Adjust these if needed
BLUE_LOWER = np.array([100, 150, 50])   # HSV lower bound for blue
BLUE_UPPER = np.array([140, 255, 255])  # HSV upper bound for blue
MIN_WIDTH, MIN_HEIGHT = 80, 30         # minimum size of button region
CLICK_DELAY = 1.0                      # seconds to wait after a click
SCAN_INTERVAL = 0.5                    # seconds between screen scans

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # adjust path if needed

print("Auto-Continue clicker started. Press Ctrl+C to stop.")

while True:
    # Take a screenshot and convert to OpenCV format
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for blue regions
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small regions
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        # Crop the candidate region and run OCR
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()

        # Check if it contains the word "Continue"
        if "Continue" in text:
            cx, cy = x + w // 2, y + h // 2
            print(f"Found Continue button at ({cx}, {cy}), clicking...")
            pyautogui.click(cx, cy)
            time.sleep(CLICK_DELAY)
            break  # skip other detections until next scan

    time.sleep(SCAN_INTERVAL)
