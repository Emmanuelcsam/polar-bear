

import json
from typing import Dict

def load_calibration(filepath: str) -> Dict:
    """Load calibration data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Calibration file not found at {filepath}. Using default values.")
        return {"um_per_px": 0.7}

if __name__ == '__main__':
    # Create a dummy calibration file for demonstration
    dummy_cal_data = {"um_per_px": 0.65}
    dummy_filepath = "dummy_calibration.json"
    with open(dummy_filepath, 'w') as f:
        json.dump(dummy_cal_data, f)
        
    # Test loading the existing file
    print(f"--- Loading existing file '{dummy_filepath}' ---")
    calibration_data = load_calibration(dummy_filepath)
    print("Loaded data:", calibration_data)
    assert calibration_data["um_per_px"] == 0.65

    # Test loading a non-existent file
    print(f"--- Loading non-existent file 'non_existent.json' ---")
    calibration_data_default = load_calibration("non_existent.json")
    print("Loaded data:", calibration_data_default)
    assert calibration_data_default["um_per_px"] == 0.7
    
    print("\nScript finished successfully.")

