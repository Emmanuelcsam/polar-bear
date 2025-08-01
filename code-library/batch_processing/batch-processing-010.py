

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def save_batch_summary_report_csv(
    batch_summary_list: List[Dict[str, Any]],
    output_dir: Path,
    filename: str
):
    """Saves a summary CSV report for the entire batch."""
    log_message("Saving batch summary report...")
    
    if not batch_summary_list:
        log_message("No batch results to save.", level="WARNING")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / filename
    
    try:
        summary_df = pd.DataFrame(batch_summary_list)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        log_message(f"Batch summary report saved to {summary_path}")
    except Exception as e:
        log_message(f"Error saving batch summary report: {e}", level="ERROR")

# Dummy log_message for standalone execution
def log_message(message, level="INFO"):
    print(f"[{level}] {message}")

if __name__ == '__main__':
    # Example of how to use save_batch_summary_report_csv

    # 1. Setup: Create a mock list of summary data
    mock_batch_summary = [
        {
            "Filename": "image01.jpg", "Timestamp": "2025-07-10T10:00:00",
            "Status": "Pass", "Total_Defects": 0, "Core_Defects": 0,
            "Processing_Time_s": 0.75, "Error": ""
        },
        {
            "Filename": "image02.jpg", "Timestamp": "2025-07-10T10:01:00",
            "Status": "Review", "Total_Defects": 3, "Core_Defects": 1,
            "Processing_Time_s": 1.25, "Error": ""
        },
        {
            "Filename": "image03.jpg", "Timestamp": "2025-07-10T10:02:00",
            "Status": "Error", "Total_Defects": 0, "Core_Defects": 0,
            "Processing_Time_s": 0.15, "Error": "Could not find fiber"
        }
    ]
    
    output_directory = Path("./modularized_scripts/test_reports")
    summary_filename = "batch_summary_test.csv"

    # 2. Run the save function
    print(f"\n--- Saving mock batch summary to '{output_directory / summary_filename}' ---")
    save_batch_summary_report_csv(mock_batch_summary, output_directory, summary_filename)

    # 3. Verify the output
    expected_file = output_directory / summary_filename
    if expected_file.exists():
        print(f"Success! Batch summary file created at: {expected_file}")
        with open(expected_file, 'r') as f:
            print("\n--- File Content ---")
            print(f.read())
            print("--------------------")
    else:
        print("Error: Batch summary file was not created.")

