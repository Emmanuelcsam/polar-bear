#!/usr/bin/env python3
"""
Print min, max, mean of all logged values.
"""
import json
from data_store import load_events

def record_trends():
    """Record trends in logged values."""
    events = load_events()
    vals = [e.get("pixel", e.get("intensity")) for e in events]
    vals = [v for v in vals if v is not None]  # Filter out None values

    if not vals:
        print("[Trend] No data")
        return None

    stats = {
        "min": min(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
        "count": len(vals)
    }

    print("[Trend]", json.dumps(stats))
    return stats

if __name__ == "__main__":
    record_trends()
