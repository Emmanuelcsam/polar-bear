#!/usr/bin/env python3
"""
Randomly emit 0–255 pixel values forever.
Logs each value via data_store.
"""
import random
import time
from data_store import save_event

def generate_pixels(interval=0.01, max_iterations=None):
    """Loop: pick a value, log it, wait."""
    count = 0
    while max_iterations is None or count < max_iterations:
        v = random.randint(0, 255)
        print(f"[PixelGen] {v}")
        save_event({"pixel": v})
        time.sleep(interval)
        count += 1
        if max_iterations is not None and count >= max_iterations:
            break

if __name__ == "__main__":
    generate_pixels()
