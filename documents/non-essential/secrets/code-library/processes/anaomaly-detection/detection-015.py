# 5_detect_anomalies.py
# This module reads the full analysis results and identifies images
# that are statistically different from the rest of the dataset.
import json
import os
import numpy as np
from shared_config import ANALYSIS_RESULTS_PATH, ANOMALIES_PATH, DATA_DIR

print("--- Module: Detecting Anomalies ---")
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(ANALYSIS_RESULTS_PATH):
    print("Analysis file not found. Cannot detect anomalies. Run analysis first.")
    exit()

with open(ANALYSIS_RESULTS_PATH, 'r') as f:
    analysis_data = json.load(f)

if len(analysis_data) < 2:
    print("Need at least two images to compare for anomalies. Aborting.")
    exit()

# Calculate the overall average and standard deviation of the means
all_means = [data['mean_intensity'] for data in analysis_data.values()]
global_mean = np.mean(all_means)
# This is the key: we measure how much the means themselves vary
global_std_of_means = np.std(all_means)

anomalies = {}
# Anomaly is defined as a mean intensity > 2 standard deviations from the global mean
DEVIATION_THRESHOLD = 2.0

print(f"Global Mean Intensity of all images: {global_mean:.2f}")
print(f"Standard Deviation of image means: {global_std_of_means:.2f}")
print(f"Using anomaly threshold: {DEVIATION_THRESHOLD} standard deviations.")

for filename, data in analysis_data.items():
    # Calculate how many standard deviations away this image's mean is
    deviation_score = abs(data['mean_intensity'] - global_mean) / global_std_of_means
    if deviation_score > DEVIATION_THRESHOLD:
        anomalies[filename] = {
            'mean_intensity': data['mean_intensity'],
            'deviation_score': deviation_score
        }
        print(f"  -> ANOMALY DETECTED: '{filename}' (Score: {deviation_score:.2f})")

# Save the detected anomalies to a file
with open(ANOMALIES_PATH, 'w') as f:
    json.dump(anomalies, f, indent=4)

if anomalies:
    print(f"\nAnomaly detection complete. Results saved to '{ANOMALIES_PATH}'")
else:
    print("\nNo anomalies detected matching the criteria.")