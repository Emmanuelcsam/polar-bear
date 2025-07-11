#!/usr/bin/env python3
"""
Count how often each value appears in the log.
Works on both pixel and intensity events.
"""
import numpy as np
from data_store import load_events

def find_patterns():
    """Find patterns in logged values."""
    events = load_events()
    vals = [e.get("pixel", e.get("intensity")) for e in events]
    vals = [v for v in vals if v is not None]  # Filter out None values

    if not vals:
        print("[Pattern] No data found")
        return {}

    unique, counts = np.unique(vals, return_counts=True)
    patterns = dict(zip(unique, counts))

    for v, c in zip(unique, counts):
        print(f"[Pattern] Value {v}: {c} times")

    return patterns

if __name__ == "__main__":
    find_patterns()
