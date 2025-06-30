#!/usr/bin/env python3
"""Neural Nexus IDE Launcher"""
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).parent
main_script = script_dir / "neural_nexus_ide.py"

if main_script.exists():
    subprocess.run([sys.executable, str(main_script)])
else:
    print(f"Error: Main script not found at {main_script}")
    input("Press Enter to exit...")
