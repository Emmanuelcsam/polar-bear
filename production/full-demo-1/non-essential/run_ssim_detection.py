#!/usr/bin/env python3
"""
Runner script for SSIM Detection Emulator.
"""

import logging
from ssim_detection_emulator import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the SSIM detection emulator
    print("Starting SSIM Detection Emulator...")
    main()
