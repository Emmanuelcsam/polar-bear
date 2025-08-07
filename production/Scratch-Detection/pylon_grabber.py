"""
Pylon camera frame grabber module.
A dedicated thread to continuously grab frames from a Basler pylon camera.
"""

import time
import threading
import logging

# Fix the import structure to match checkpoint-5 and checkpoint-6
PYLON_AVAILABLE = False  # Global flag to track if Pylon SDK is available for use
try:
    from pypylon import pylon  # Import Basler's pylon SDK for camera control
    PYLON_AVAILABLE = True  # Set flag to True since import succeeded
    print("INFO: Pylon SDK found. Basler camera support is enabled.")
    # Try to import genicam, but don't fail if it's not available
    try:
        from genicam import GenericException  # Import GenICam exception for proper error handling
    except ImportError:
        # Define a fallback GenericException if genicam is not available
        class GenericException(Exception):  # Create dummy exception class as fallback
            pass  # Empty class body - just inherits from Exception
except ImportError:
    print("WARNING: Pylon SDK not found. Cannot use Basler camera.")
    print("Please install pypylon: pip install pypylon")


class PylonFrameGrabber(threading.Thread):
    """
    A dedicated thread to continuously grab frames from a Basler pylon camera.
    This acts as the 'producer' in our producer-consumer architecture.
    """
    
    def __init__(self):
        super().__init__(name="PylonGrabber")  # Initialize parent Thread class with descriptive name
        self.daemon = True  # Thread will exit when main program exits - prevents hanging processes
        self.camera = None  # Will hold the pylon camera instance once initialized
        self.latest_frame = None  # Stores the most recent captured frame as numpy array
        self.is_running = threading.Event()  # Thread-safe flag to control main grabbing loop
        self.lock = threading.Lock()  # Mutex to protect shared frame data from concurrent access
        
        if PYLON_AVAILABLE:  # Only setup converter if Pylon SDK is available
            # Image converter for different pixel formats
            self.converter = pylon.ImageFormatConverter()  # Creates converter for image format conversion
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed  # Set output to BGR format for OpenCV compatibility
            self.converter.OutputBitAlignment = (  # Align bits to MSB for proper image interpretation
                pylon.OutputBitAlignment_MsbAligned
            )

    def run(self):
        """The main loop of the grabbing thread."""
        logging.info("PylonFrameGrabber thread started.")  # Log thread startup for debugging
        
        if not PYLON_AVAILABLE:  # Check if Pylon SDK is available before proceeding
            logging.critical("ERROR: Pylon SDK not available. Cannot use "
                           "Basler camera.")
            logging.critical("Please install pypylon: pip install pypylon")
            logging.critical("And ensure Basler Pylon SDK is installed.")
            return  # Exit early if no Pylon SDK - cannot operate camera
            
        try:
            # Initialize and open the camera
            self.camera = pylon.InstantCamera(  # Create camera instance from first available device
                pylon.TlFactory.GetInstance().CreateFirstDevice()  # Get transport layer factory and create first device
            )
            self.camera.Open()  # Establish connection to the physical camera hardware
            logging.info(f"Using device: "  # Log which camera model is being used
                        f"{self.camera.GetDeviceInfo().GetModelName()}")

            # Configure for continuous acquisition
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # Begin frame capture, keeping only newest frame
            self.is_running.set()  # Signal that grabbing loop is active and running
            logging.info("Camera started grabbing frames.")  # Confirm successful start of frame acquisition

            while self.is_running.is_set():  # Continue grabbing while thread should remain active
                if not self.camera.IsGrabbing():  # Check if camera stopped grabbing unexpectedly
                    logging.warning("Camera stopped grabbing unexpectedly.")
                    break  # Exit loop if camera is no longer grabbing frames
                
                try:
                    grabResult = self.camera.RetrieveResult(  # Attempt to get next available frame from camera
                        5000, pylon.TimeoutHandling_ThrowException  # Wait up to 5 seconds, throw exception on timeout
                    )
                    if grabResult.GrabSucceeded():  # Check if frame capture was successful
                        # Convert the image to a format OpenCV can use (BGR)
                        image = self.converter.Convert(grabResult)  # Convert pylon image format to BGR format
                        frame = image.GetArray()  # Extract numpy array from converted image
                        with self.lock:  # Acquire mutex lock for thread-safe frame update
                            self.latest_frame = frame.copy()  # Store deep copy of frame to prevent data corruption
                    else:
                        logging.error(f"Grab failed: {grabResult.ErrorCode} "  # Log detailed error information
                                   f"{grabResult.ErrorDescription}")
                    grabResult.Release()  # Free memory used by grab result to prevent memory leaks
                except GenericException as e:  # Catch GenICam-specific exceptions during frame retrieval
                    logging.error(f"An error occurred while grabbing a frame: "
                               f"{e}")
                    time.sleep(0.1)  # Avoid tight loop on error - prevents CPU overload during repeated failures
            
        except pylon.RuntimeException as e:  # Catch Pylon SDK runtime errors (camera disconnected, etc.)
            logging.critical(f"Pylon runtime exception: {e}. "
                           f"Is a camera connected?")
        except Exception as e:  # Catch any other unexpected exceptions
            logging.critical(f"An unexpected error occurred in "
                           f"PylonFrameGrabber: {e}", exc_info=True)  # Include stack trace for debugging
        finally:  # Cleanup code that always runs regardless of how try block exits
            if self.camera and self.camera.IsGrabbing():  # Check if camera exists and is still grabbing
                self.camera.StopGrabbing()  # Stop frame acquisition to release camera resources
                logging.info("Camera stopped grabbing.")
            if self.camera and self.camera.IsOpen():  # Check if camera exists and connection is open
                self.camera.Close()  # Close camera connection to free hardware resources
                logging.info("Camera closed.")
            self.is_running.clear()  # Clear running flag to signal thread completion
            logging.info("PylonFrameGrabber thread finished.")  # Log thread termination for debugging

    def read(self):
        """Returns the most recent frame."""
        with self.lock:  # Acquire mutex lock to ensure thread-safe access to frame data
            if self.latest_frame is None:  # Check if no frame has been captured yet
                return None  # Return None to indicate no frame available
            return self.latest_frame.copy()  # Return deep copy to prevent external modification of stored frame

    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stopping PylonFrameGrabber thread.")  # Log stop request for debugging
        self.is_running.clear()  # Clear event flag to signal main loop to exit gracefully


# Make PYLON_AVAILABLE accessible as a class attribute
PylonFrameGrabber.PYLON_AVAILABLE = PYLON_AVAILABLE  # Attach global flag to class for external availability checking 