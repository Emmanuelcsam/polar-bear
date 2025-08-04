"""
Main application entry point for the defect detection system.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging

# Import our modular components
from config.system_config import SystemConfig
from log_utils.async_logger import setup_logging
from camera.pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
from detection.preprocessing import preprocess_image
from processing.frame_processor import process_frame


def load_reference_image():
    """Load and process the reference 'golden template' image."""
    # Check filesystem for existence of reference template image file
    if not os.path.exists(SystemConfig.REFERENCE_IMAGE_PATH):
        # Log warning message about missing reference file with interpolated path
        logging.warning(f"Reference image not found at "
                       f"'{SystemConfig.REFERENCE_IMAGE_PATH}'. "
                       f"Creating a test reference image.")
        # Create 3-channel BGR image array filled with zeros (black pixels)
        ref_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Fill blue channel with 128 base value plus random noise (-10 to +10)
        ref_image[:, :, 0] = 128 + np.random.randint(-10, 10, (480, 640))
        # Fill green channel with 128 base value plus random noise (-10 to +10)
        ref_image[:, :, 1] = 128 + np.random.randint(-10, 10, (480, 640))
        # Fill red channel with 128 base value plus random noise (-10 to +10)
        ref_image[:, :, 2] = 128 + np.random.randint(-10, 10, (480, 640))
        # Write the generated image array to disk as image file
        cv2.imwrite(SystemConfig.REFERENCE_IMAGE_PATH, ref_image)
        # Log successful creation of synthetic reference image
        logging.info("Test reference image created and saved.")
    else:
        # Load existing reference image from disk into memory as BGR array
        ref_image = cv2.imread(SystemConfig.REFERENCE_IMAGE_PATH)
        # Check if Pylon camera library is unavailable (mock mode)
        if not PYLON_AVAILABLE:
            # Resize loaded image to standard mock frame dimensions
            ref_image = cv2.resize(ref_image, (640, 480))
        
    # Apply preprocessing transformations to normalize reference image
    ref_image_processed = preprocess_image(ref_image)
    # Log successful completion of reference image preparation
    logging.info("Reference image loaded and preprocessed.")
    # Return the processed reference image for defect comparison
    return ref_image_processed


def check_gui_availability():
    """Check if OpenCV GUI is available."""
    try:
        # Attempt to create a test window to verify GUI functionality
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        # Immediately destroy the test window to clean up
        cv2.destroyWindow("test")
        # Return True indicating GUI is available and functional
        return True
    except Exception:
        # Log warning about headless environment or missing display
        logging.warning("OpenCV GUI not available. Running in headless mode.")
        # Return False indicating GUI is not available
        return False


def main():
    """Main application function."""
    # --- Initialization ---
    # Initialize asynchronous logging system and get listener/queue handles
    log_listener, log_queue = setup_logging(SystemConfig.LOG_LEVEL)
    # Record application startup in log with INFO level
    logging.info("Application starting.")

    # Load reference template and apply preprocessing transformations
    ref_image_processed = load_reference_image()

    # Create new instance of Pylon camera frame capture thread
    frame_grabber = PylonFrameGrabber()
    # Start the camera thread to begin frame acquisition
    frame_grabber.start()

    # Pause execution for 2 seconds to allow camera initialization
    time.sleep(2)
    # Check if camera thread successfully started and is running
    if not frame_grabber.is_running.is_set():
        # Log critical error with detailed troubleshooting information
        logging.critical("ERROR: Failed to start Basler camera.")
        logging.critical("Please check:")
        logging.critical("1. Basler camera is connected and powered on")
        logging.critical("2. Basler Pylon SDK is installed")
        logging.critical("3. pypylon package is installed: pip install pypylon")
        logging.critical("4. No other application is using the camera")
        # Stop the logging listener before application exit
        log_listener.stop()
        # Terminate application with error code 1
        sys.exit(1)

    # Test OpenCV GUI capabilities for display functionality
    gui_available = check_gui_availability()

    # --- Main Loop ---
    # Log entry into continuous frame processing loop
    logging.info("Entering main processing loop. Press 'q' to exit.")
    # Initialize counter to track processed frame numbers
    frame_count = 0
    try:
        # Begin infinite loop for continuous frame processing
        while True:
            # Retrieve latest frame from camera capture thread
            live_frame = frame_grabber.read()
            # Check if frame retrieval failed (camera not ready)
            if live_frame is None:
                # Log warning about frame unavailability
                logging.warning("Waiting for frame from camera...")
                # Short delay before retrying frame acquisition
                time.sleep(0.1)
                # Skip to next iteration of processing loop
                continue
            
            # Increment frame counter for current processing cycle
            frame_count += 1
            # Log debug message with current frame number
            logging.debug(f"--- Processing Frame {frame_count} ---")

            # Execute defect detection algorithm on current frame
            annotated_frame, defects, diff_mask = process_frame(
                live_frame, ref_image_processed
            )

            # Check if graphical user interface is available for display
            if gui_available:
                # Display processed frame with defect annotations
                cv2.imshow("Live Defect Detection", annotated_frame)
                # Display difference mask if processing generated one
                if diff_mask is not None:
                    cv2.imshow("Difference Mask", diff_mask)

                # Capture keyboard input with 1ms timeout
                key = cv2.waitKey(1) & 0xFF
                # Check if 'q' key was pressed for quit command
                if key == ord('q'):
                    # Log user-initiated shutdown request
                    logging.info("'q' pressed. Initiating shutdown.")
                    # Exit the main processing loop
                    break
            else:
                # Handle headless mode operation without GUI display
                if defects:
                    # Log defect detection results to console
                    logging.info(f"Frame {frame_count}: Found {len(defects)} "
                               f"defects")
                # Brief pause to prevent excessive CPU utilization
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    # Handle user interrupt signal (Ctrl+C) gracefully
    except KeyboardInterrupt:
        # Log receipt of keyboard interrupt for shutdown
        logging.info("Keyboard interrupt received. Shutting down.")
    # Catch any unexpected exceptions during main loop execution
    except Exception as e:
        # Log critical error with full exception details and stack trace
        logging.critical(f"An unhandled exception occurred in the main loop: "
                       f"{e}", exc_info=True)
    finally:
        # --- Graceful Shutdown ---
        # Log initiation of cleanup sequence
        logging.info("Starting graceful shutdown sequence.")
        
        # Check if camera thread is still executing
        if frame_grabber.is_alive():
            # Signal frame grabber thread to stop execution
            frame_grabber.stop()
            # Wait maximum 5 seconds for thread to terminate cleanly
            frame_grabber.join(timeout=5)  # Wait for thread to finish
            # Check if thread failed to terminate within timeout
            if frame_grabber.is_alive():
                # Log warning about unclean thread termination
                logging.warning("Frame grabber thread did not terminate "
                              "cleanly.")

        # Check if graphical interface was being used
        if gui_available:
            try:
                # Close all OpenCV display windows
                cv2.destroyAllWindows()
                # Log successful window cleanup
                logging.info("OpenCV windows destroyed.")
            except Exception:
                # Log warning if window cleanup encounters errors
                logging.warning("Error destroying OpenCV windows.")

        # Stop the asynchronous logging listener thread
        log_listener.stop()
        # Log successful logging system shutdown
        logging.info("Logging listener stopped.")
        
        # Log final completion of application shutdown
        logging.info("Application shutdown complete.")


# Check if script is being executed directly (not imported as module)
if __name__ == "__main__":
    # Execute main application function when run as primary script
    main() 