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
from custom_logging.async_logger import setup_logging
from camera.pylon_grabber import PylonFrameGrabber, PYLON_AVAILABLE
from detection.preprocessing import preprocess_image
from processing.frame_processor import process_frame


def load_reference_image():
    """Load and process the reference 'golden template' image."""
    if not os.path.exists(SystemConfig.REFERENCE_IMAGE_PATH):
        logging.warning(f"Reference image not found at "
                       f"'{SystemConfig.REFERENCE_IMAGE_PATH}'. "
                       f"Creating a test reference image.")
        # Create a test reference image with the same size as mock frame
        ref_image = np.zeros((480, 640, 3), dtype=np.uint8)
        ref_image[:, :, 0] = 128 + np.random.randint(-10, 10, (480, 640))
        ref_image[:, :, 1] = 128 + np.random.randint(-10, 10, (480, 640))
        ref_image[:, :, 2] = 128 + np.random.randint(-10, 10, (480, 640))
        # Save the test reference image
        cv2.imwrite(SystemConfig.REFERENCE_IMAGE_PATH, ref_image)
        logging.info("Test reference image created and saved.")
    else:
        ref_image = cv2.imread(SystemConfig.REFERENCE_IMAGE_PATH)
        # Resize reference image to match mock frame size if needed
        if not PYLON_AVAILABLE:
            ref_image = cv2.resize(ref_image, (640, 480))
        
    ref_image_processed = preprocess_image(ref_image)
    logging.info("Reference image loaded and preprocessed.")
    return ref_image_processed


def check_gui_availability():
    """Check if OpenCV GUI is available."""
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except Exception:
        logging.warning("OpenCV GUI not available. Running in headless mode.")
        return False


def main():
    """Main application function."""
    # --- Initialization ---
    log_listener, log_queue = setup_logging(SystemConfig.LOG_LEVEL)
    logging.info("Application starting.")

    # Load and process the reference image
    ref_image_processed = load_reference_image()

    # Start the camera frame grabber thread
    frame_grabber = PylonFrameGrabber()
    frame_grabber.start()

    # Wait for the camera to initialize
    time.sleep(2)
    if not frame_grabber.is_running.is_set():
        logging.critical("ERROR: Failed to start Basler camera.")
        logging.critical("Please check:")
        logging.critical("1. Basler camera is connected and powered on")
        logging.critical("2. Basler Pylon SDK is installed")
        logging.critical("3. pypylon package is installed: pip install pypylon")
        logging.critical("4. No other application is using the camera")
        log_listener.stop()
        sys.exit(1)

    # Check if OpenCV GUI is available
    gui_available = check_gui_availability()

    # --- Main Loop ---
    logging.info("Entering main processing loop. Press 'q' to exit.")
    frame_count = 0
    try:
        while True:
            live_frame = frame_grabber.read()
            if live_frame is None:
                logging.warning("Waiting for frame from camera...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            logging.debug(f"--- Processing Frame {frame_count} ---")

            annotated_frame, defects, diff_mask = process_frame(
                live_frame, ref_image_processed
            )

            # Display results if GUI is available
            if gui_available:
                cv2.imshow("Live Defect Detection", annotated_frame)
                if diff_mask is not None:
                    cv2.imshow("Difference Mask", diff_mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("'q' pressed. Initiating shutdown.")
                    break
            else:
                # Headless mode - just log the results
                if defects:
                    logging.info(f"Frame {frame_count}: Found {len(defects)} "
                               f"defects")
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logging.critical(f"An unhandled exception occurred in the main loop: "
                       f"{e}", exc_info=True)
    finally:
        # --- Graceful Shutdown ---
        logging.info("Starting graceful shutdown sequence.")
        
        # Stop the frame grabber thread
        if frame_grabber.is_alive():
            frame_grabber.stop()
            frame_grabber.join(timeout=5)  # Wait for thread to finish
            if frame_grabber.is_alive():
                logging.warning("Frame grabber thread did not terminate "
                              "cleanly.")

        # Close OpenCV windows if GUI is available
        if gui_available:
            try:
                cv2.destroyAllWindows()
                logging.info("OpenCV windows destroyed.")
            except Exception:
                logging.warning("Error destroying OpenCV windows.")

        # Stop the logging listener
        log_listener.stop()
        logging.info("Logging listener stopped.")
        
        logging.info("Application shutdown complete.")


if __name__ == "__main__":
    main() 