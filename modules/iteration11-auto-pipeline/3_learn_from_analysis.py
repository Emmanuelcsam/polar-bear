# 3_learn_from_analysis.py
# This module reads the analysis data and "learns" the central tendencies
# of the image set. It saves these learned parameters for the generator.
import json
import os
import numpy as np
from shared_config import ANALYSIS_RESULTS_PATH, GENERATOR_PARAMS_PATH, DATA_DIR

print("--- Module: Learning from Analysis Data ---")
os.makedirs(DATA_DIR, exist_ok=True)

# Define default parameters in case no analysis data is found
learned_params = {'target_mean': 128.0, 'target_std': 50.0}

if not os.path.exists(ANALYSIS_RESULTS_PATH):
    print("Analysis file not found. Using default generator parameters.")
else:
    with open(ANALYSIS_RESULTS_PATH, 'r') as f:
        analysis_data = json.load(f)

    if not analysis_data:
        print("No data in analysis file. Using default parameters.")
    else:
        # "Learn" by averaging the stats from all analyzed images
        all_means = [data['mean_intensity'] for data in analysis_data.values()]
        all_stds = [data['std_dev_intensity'] for data in analysis_data.values()]

        learned_params['target_mean'] = np.mean(all_means)
        learned_params['target_std'] = np.mean(all_stds)
        print("Learning complete. Calculated new target parameters from dataset.")

# Save the learned parameters for the generator module to use
with open(GENERATOR_PARAMS_PATH, 'w') as f:
    json.dump(learned_params, f, indent=4)

print(f"Learned parameters saved to '{GENERATOR_PARAMS_PATH}'")
print(f"  - Target Mean Intensity: {learned_params['target_mean']:.2f}")
print(f"  - Target Std Deviation: {learned_params['target_std']:.2f}")