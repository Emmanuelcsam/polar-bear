import time
import sys
import signal
from pathlib import Path

import pyautogui

# ----------------------------------------- Configuration
TEMPLATE = Path(__file__).with_name("continue_btn.png")
CONFIDENCE = 0.90        # 0.80–0.95 is typical; raise if you get false positives
COOLDOWN   = 0.5         # seconds between scans
CLICK_DELAY = 0.10       # pause after a click before scanning again
# -----------------------------------------

if not TEMPLATE.exists():
    sys.exit(f"[ERROR] Cannot find reference image: {TEMPLATE}")

# Global PyAutoGUI settings
pyautogui.PAUSE = 0.01      # small pause between PyAutoGUI calls
pyautogui.FAILSAFE = True   # move mouse to upper‑left to abort

def clean_exit(signum=None, frame=None):
    print("\n[INFO] Exiting gracefully…")
    sys.exit(0)

# handle Ctrl+C and SIGTERM
signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

print("[INFO] Auto‑Continue started. Press Ctrl+C or move mouse to top‑left corner to stop.")

while True:
    try:
        # locate reference image on the current screen
        box = pyautogui.locateOnScreen(TEMPLATE, confidence=CONFIDENCE)

        if box:
            x, y = pyautogui.center(box)
            print(f"[INFO] Found button at ({x}, {y}); clicking…")
            pyautogui.click(x, y)
            time.sleep(CLICK_DELAY)

        time.sleep(COOLDOWN)

    except pyautogui.FailSafeException:
        clean_exit()
