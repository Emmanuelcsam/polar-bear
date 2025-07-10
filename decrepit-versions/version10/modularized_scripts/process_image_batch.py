
from pathlib import Path
from typing import List

# Import modularized functions and data structures
from log_message import log_message
from inspector_config import InspectorConfig
from fiber_specifications import FiberSpecifications
from get_image_paths_from_user import get_image_paths_from_user
from process_single_image import process_single_image
from save_batch_summary_report_csv import save_batch_summary_report_csv

def process_image_batch(
    image_paths: List[Path],
    config: InspectorConfig, 
    fiber_specs: FiberSpecifications,
    operating_mode: str,
    output_dir: Path
):
    """Processes a batch of images and saves a summary report."""
    log_message(f"Starting batch processing for {len(image_paths)} images...")
    batch_summary_list = []

    for i, image_path in enumerate(image_paths):
        log_message(f"--- Processing image {i+1}/{len(image_paths)} ---")
        image_res = process_single_image(image_path, config, fiber_specs, operating_mode, output_dir)
        
        # Create summary dictionary
        summary_item = {
            "Filename": image_res.filename,
            "Timestamp": image_res.timestamp.isoformat(),
            "Operating_Mode": image_res.operating_mode,
            "Status": image_res.stats.status,
            "Total_Defects": image_res.stats.total_defects,
            "Core_Defects": image_res.stats.core_defects,
            "Cladding_Defects": image_res.stats.cladding_defects,
            "Processing_Time_s": f"{image_res.stats.processing_time_s:.2f}",
            "Error": image_res.error_message if image_res.error_message else ""
        }
        batch_summary_list.append(summary_item)

    # Save the final batch summary report
    save_batch_summary_report_csv(batch_summary_list, output_dir, config.BATCH_SUMMARY_FILENAME)
    
    log_message(f"--- Batch processing complete. ---")

if __name__ == '__main__':
    # Example of how to run a full batch process
    
    # 1. Setup
    conf = InspectorConfig()
    specs = FiberSpecifications() # Using default specs
    mode = "PIXEL_ONLY"
    output_root = Path("./modularized_scripts/test_run_batch")
    
    print("--- Starting full BATCH pipeline test ---")
    
    # 2. Get image paths (using a sub-directory for a smaller test)
    # For a real run, you might provide the root 'version10' directory to the prompt
    print("Please provide a directory of images to process for the batch test.")
    print("You can use './fiber_inspection_output/ima18' or similar for a quick test.")
    image_files = get_image_paths_from_user()

    if image_files:
        print(f"\nFound {len(image_files)} images. Starting batch processing...")
        print(f"Output will be saved in: {output_root}")
        
        # 3. Run the batch processing function
        process_image_batch(image_files, conf, specs, mode, output_root)
        
        print("\n--- BATCH PROCESSING COMPLETE ---")
        summary_file = output_root / conf.BATCH_SUMMARY_FILENAME
        if summary_file.exists():
            print(f"Batch summary report saved to: {summary_file}")
            with open(summary_file, 'r') as f:
                print("\n--- Summary Report Content ---")
                print(f.read())
                print("----------------------------")
        else:
            print("Batch summary report was not created.")
    else:
        print("No images selected. Batch processing cancelled.")
