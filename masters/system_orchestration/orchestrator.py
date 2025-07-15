#!/usr/bin/env python3
"""
Auto‑import every .py in this folder (except itself).
Just to prove you can drop modules in/out.
"""
import os
import importlib
import sys

def load_all():
    """Load all Python modules in the current directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f[:-3] for f in os.listdir(current_dir)
             if f.endswith(".py") and f != "orchestrator.py" and not f.startswith("test_")]

    loaded_modules = []
    for name in files:
        try:
            # Handle relative imports
            if name in sys.modules:
                module = importlib.reload(sys.modules[name])
            else:
                module = importlib.import_module(name)
            loaded_modules.append(name)
            print(f"[Orch] Loaded {name}")
        except Exception as e:
            print(f"[Orch] Failed to load {name}: {e}")

    return loaded_modules

if __name__ == "__main__":
    load_all()
