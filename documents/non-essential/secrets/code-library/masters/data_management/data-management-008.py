#!/usr/bin/env python3
"""
Simple JSON‑lines event store.
Any script can save or load events here.
"""
import json
import time
import os

LOG_FILE = "events.log"

def save_event(event):
    """Append event dict + timestamp to the log."""
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def load_events():
    """Read all events (or return empty list)."""
    try:
        with open(LOG_FILE) as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []

def clear_events():
    """Clear all events from the log file."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
