#!/usr/bin/env python3
"""
Build a simple histogram "model" and pickle it.
"""
import pickle
import os
from data_store import load_events

def learn_model(model_file="model.pkl"):
    """Learn a histogram model from intensity data."""
    events = load_events()
    data = [e["intensity"] for e in events if "intensity" in e]

    if not data:
        print("[Learn] No intensity data found")
        return None

    # Build histogram
    hist = {}
    for d in data:
        hist[d] = hist.get(d, 0) + 1

    # Save model
    with open(model_file, "wb") as f:
        pickle.dump(hist, f)

    print(f"[Learn] Model saved to {model_file}")
    return hist

def load_model(model_file="model.pkl"):
    """Load a pickled histogram model."""
    try:
        with open(model_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[Learn] Model file {model_file} not found")
        return None

if __name__ == "__main__":
    learn_model()
