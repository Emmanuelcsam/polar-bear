# shared_config.py
# Central configuration values used by multiple scripts.
import torch
import os

# --- Device Configuration ---
# This makes the system GPU-capable. It automatically uses a CUDA-enabled GPU
# if PyTorch detects one, otherwise it falls back to the CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Directory Paths ---
# All data exchange between scripts happens through these directories.
DATA_DIR = "data"
IMAGE_INPUT_DIR = "input_images"

# --- File Paths for Data Exchange ---
# Path to the file storing the learned parameters for the generator
GENERATOR_PARAMS_PATH = os.path.join(DATA_DIR, "generator_params.json")
# Path to the file storing analysis results for each image
ANALYSIS_RESULTS_PATH = os.path.join(DATA_DIR, "analysis_results.json")
# Path to the file storing detected anomalies
ANOMALIES_PATH = os.path.join(DATA_DIR, "anomalies.json")

# --- Generation Parameters ---
# The dimensions (height, width) of the new images to generate
GENERATED_IMAGE_SIZE = (64, 64)