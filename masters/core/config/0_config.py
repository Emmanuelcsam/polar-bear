# 0_config.py
import os
import time

# --- Main Settings ---
# 'auto': Process all images in INPUT_DIR.
# 'manual': Process only the first image found.
LEARNING_MODE = 'auto'

# Set the duration in seconds for continuous analysis.
# The batch processor will loop and re-analyze for this long.
ANALYSIS_DURATION_SECONDS = 60 # 1 minute

# --- Directory Paths ---
INPUT_DIR = 'images_input'
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# --- File Paths ---
# Ensures directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data files used for inter-script communication
INTENSITY_DATA_PATH = os.path.join(DATA_DIR, 'intensities.npy')
STATS_DATA_PATH = os.path.join(DATA_DIR, 'image_stats.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'pixel_model.pth')
AVG_IMAGE_PATH = os.path.join(DATA_DIR, 'average_image.npy')

# --- Script Timers ---
# Used by the batch processor to run for a fixed time
START_TIME = time.time()
END_TIME = START_TIME + ANALYSIS_DURATION_SECONDS
