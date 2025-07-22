#!/usr/bin/env python3
"""
Rebuild an image array from logged intensities.
Prints the NumPy array to stdout.
"""
import numpy as np
from data_store import load_events

def generate_image(width=64, height=64):
    """Generate image array from logged intensities."""
    events = load_events()
    vals = [e["intensity"] for e in events if "intensity" in e]

    if not vals:
        print("No intensity data found")
        return np.zeros((height, width))

    # Pad or truncate to fit desired dimensions
    total_pixels = width * height
    if len(vals) < total_pixels:
        vals.extend([0] * (total_pixels - len(vals)))

    arr = np.array(vals[:total_pixels]).reshape((height, width))
    print(arr)
    return arr

if __name__ == "__main__":
    generate_image()
