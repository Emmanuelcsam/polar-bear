#!/bin/bash

# This script runs the entire pipeline in sequence, simulating an
# "automatic learning mode".

echo "===== STARTING AUTOMATIC PIPELINE ====="

# Step 1: Ensure directories are set up
echo -e "\n[STEP 1] Setting up directories..."
python3 1_setup_directories.py

# Add a check to see if user has added images
if [ -z "$(ls -A input_images/*.png 2>/dev/null || ls -A input_images/*.jpg 2>/dev/null)" ]; then
    echo -e "\n[HALT] The 'input_images' directory is empty. Please add images and run again."
    exit 1
fi


# Step 2: Analyze images in the input folder
echo -e "\n[STEP 2] Analyzing all images..."
python3 2_analyze_images.py

# Step 3: Learn from the analysis
echo -e "\n[STEP 3] Learning from analysis data..."
python3 3_learn_from_analysis.py

# Step 4: Generate a new image based on what was learned
echo -e "\n[STEP 4] Generating a new image..."
python3 4_generate_new_image.py

# Step 5: Detect anomalies in the input set
echo -e "\n[STEP 5] Detecting anomalies..."
python3 5_detect_anomalies.py

echo -e "\n===== PIPELINE FINISHED ======"
echo "Check the 'data' directory for all output files."