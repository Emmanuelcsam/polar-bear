from utils import _log_message, _start_timer, _log_duration
from inspector import FiberInspector
from config import InspectorConfig
from connector_interface import ConnectorInterface
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connector
connector = ConnectorInterface('main-application.py')

def main():
    """Main function to drive the fiber inspection script."""
    # Wait for connector
    if connector.wait_for_connector(timeout=10):
        logger.info("Connected to monitoring connector")
    else:
        logger.warning("Running without connector integration")
    
    # Register parameters
    connector.register_parameter('batch_size', 10, 'int', 'Number of images to process in batch', min_value=1, max_value=100)
    connector.register_parameter('enable_logging', True, 'bool', 'Enable detailed logging')
    connector.register_parameter('processing_mode', 'normal', 'str', 'Processing mode', choices=['normal', 'fast', 'accurate'])
    
    connector.set_status('running')
    
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
        # Update metrics
        connector.update_metric('total_images', len(image_paths))
        connector.update_metric('processing_mode', connector.get_parameter('processing_mode'))
        
        inspector.process_image_batch(image_paths)
        
        # Update completion metrics
        connector.update_metric('processing_complete', True)
        connector.set_status('idle')
    except FileNotFoundError as fnf_error:
        _log_message(f"Error: {fnf_error}", level="CRITICAL")
        connector.set_status('error', str(fnf_error))
    except ValueError as val_error:
        _log_message(f"Input Error: {val_error}", level="CRITICAL")
        connector.set_status('error', str(val_error))
    except Exception as e:
        _log_message(f"An unexpected error occurred: {e}", level="CRITICAL")
        import traceback
        traceback.print_exc()
        connector.set_status('error', str(e))
    finally:
        _log_duration("Total Script Execution", script_start_time)
        print("=" * 70)
        print("Inspection Run Finished.")
        print("=" * 70)
        connector.cleanup()

if __name__ == "__main__":
    main()
