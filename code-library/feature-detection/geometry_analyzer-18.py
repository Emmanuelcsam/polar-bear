#!/usr/bin/env python3
"""
Compute gradients in X/Y of the intensity map.
"""
import numpy as np
from data_store import load_events

def analyze_geometry(width=64, height=64):
    """Analyze geometry by computing gradients."""
    events = load_events()
    vals = [e["intensity"] for e in events if "intensity" in e]

    if not vals:
        print("[Geom] No intensity data found")
        return None, None

    # Pad or truncate to fit desired dimensions
    total_pixels = width * height
    if len(vals) < total_pixels:
        vals.extend([0] * (total_pixels - len(vals)))

    arr = np.array(vals[:total_pixels]).reshape((height, width))
    gx, gy = np.gradient(arr)

    print("[Geom] ∂x:\n", gx)
    print("[Geom] ∂y:\n", gy)

    return gx, gy

if __name__ == "__main__":
    analyze_geometry()
