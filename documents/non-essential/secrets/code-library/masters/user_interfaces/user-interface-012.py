#!/usr/bin/env python3
"""
Simple launcher for the interactive fiber optic analysis pipeline
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the interactive pipeline
from app_gpu_interactive import main

if __name__ == "__main__":
    sys.exit(main())