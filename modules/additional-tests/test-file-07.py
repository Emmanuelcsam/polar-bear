
from pathlib import Path

# Import the high-level functions and data structures
from log_message import log_message
from inspector_config import InspectorConfig
from get_user_specifications import get_user_specifications
from get_image_paths_from_user import get_image_paths_from_user
from process_image_batch import process_image_batch

def main():
    """Main function to drive the fiber inspection script."""
    print("=" * 70)
    print("      Advanced Automated Optical Fiber End Face Inspector      ")
    print("=" * 70)

    try:
        # 1. Initialize configuration
        config = InspectorConfig()
        output_directory = Path(config.OUTPUT_DIR_NAME)
        
        # 2. Get user input for fiber specs to determine operating mode
        fiber_specs = get_user_specifications()
        
        # Determine operating mode based on user input
        if fiber_specs.cladding_diameter_um is not None:
            operating_mode = "MICRON_CALCULATED"
        else:
            operating_mode = "PIXEL_ONLY"
        log_message(f"Operating mode set to: {operating_mode}")

        # 3. Get image paths from the user
        image_paths = get_image_paths_from_user()
        
        if not image_paths:
            log_message("No images to process. Exiting.", level="INFO")
            return
            
        # 4. Process the batch of images
        process_image_batch(
            image_paths=image_paths,
            config=config,
            fiber_specs=fiber_specs,
            operating_mode=operating_mode,
            output_dir=output_directory
        )

    except Exception as e:
        log_message(f"An unexpected error occurred in the main script: {e}", level="CRITICAL")
        import traceback
        traceback.print_exc()
    finally:
        print("=" * 70)
        print("Inspection Run Finished.")
        print("=" * 70)

if __name__ == "__main__":
    main()
