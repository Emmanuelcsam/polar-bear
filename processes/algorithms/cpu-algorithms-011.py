#!/usr/bin/env python3
"""
Report any events whose value deviates from the mean by >50.
"""
import numpy as np
from data_store import load_events

def detect_anomalies(threshold=50):
    """Detect anomalies in logged values."""
    events = load_events()
    vals = np.array([e.get("pixel", e.get("intensity")) for e in events])
    vals = vals[vals != None]  # Filter out None values

    if vals.size == 0:
        print("[Anomaly] No data")
        return []

    mean = vals.mean()
    anomalies = []

    for e in events:
        val = e.get("pixel", e.get("intensity"))
        if val is not None and abs(val - mean) > threshold:
            anomalies.append(e)
            print(f"[Anomaly] {e}")

    return anomalies

if __name__ == "__main__":
    detect_anomalies()
