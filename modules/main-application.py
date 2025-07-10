from utils import _log_message, _start_timer, _log_duration
from inspector import FiberInspector
from config import InspectorConfig

def main():
    """Main function to drive the fiber inspection script."""
    print("=" * 70)
    print(" Advanced Automated Optical Fiber End Face Inspector")
    print("=" * 70)
    script_start_time = _start_timer()

    try:
        config = InspectorConfig()
        inspector = FiberInspector(config)
        inspector._get_user_specifications()
        image_paths = inspector._get_image_paths_from_user()
        if not image_paths:
            _log_message("No images to process. Exiting.", level="INFO")
            return
        inspector.process_image_batch(image_paths)
    except FileNotFoundError as fnf_error:
        _log_message(f"Error: {fnf_error}", level="CRITICAL")
    except ValueError as val_error:
        _log_message(f"Input Error: {val_error}", level="CRITICAL")
    except Exception as e:
        _log_message(f"An unexpected error occurred: {e}", level="CRITICAL")
        import traceback
        traceback.print_exc()
    finally:
        _log_duration("Total Script Execution", script_start_time)
        print("=" * 70)
        print("Inspection Run Finished.")
        print("=" * 70)

if __name__ == "__main__":
    main()
