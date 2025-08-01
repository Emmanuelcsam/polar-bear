"""
Pylon camera frame grabber module.
A dedicated thread to continuously grab frames from a Basler pylon camera.
"""

import time
import threading
import logging

# Fix the import structure to match checkpoint-5 and checkpoint-6
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("INFO: Pylon SDK found. Basler camera support is enabled.")
    # Try to import genicam, but don't fail if it's not available
    try:
        from genicam import GenericException
    except ImportError:
        # Define a fallback GenericException if genicam is not available
        class GenericException(Exception):
            pass
except ImportError:
    print("WARNING: Pylon SDK not found. Cannot use Basler camera.")
    print("Please install pypylon: pip install pypylon")


class PylonFrameGrabber(threading.Thread):
    """
    A dedicated thread to continuously grab frames from a Basler pylon camera.
    This acts as the 'producer' in our producer-consumer architecture.
    """
    
    def __init__(self):
        super().__init__(name="PylonGrabber")
        self.daemon = True  # Thread will exit when main program exits
        self.camera = None
        self.latest_frame = None
        self.is_running = threading.Event()
        self.lock = threading.Lock()
        
        if PYLON_AVAILABLE:
            # Image converter for different pixel formats
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = (
                pylon.OutputBitAlignment_MsbAligned
            )

    def run(self):
        """The main loop of the grabbing thread."""
        logging.info("PylonFrameGrabber thread started.")
        
        if not PYLON_AVAILABLE:
            logging.critical("ERROR: Pylon SDK not available. Cannot use "
                           "Basler camera.")
            logging.critical("Please install pypylon: pip install pypylon")
            logging.critical("And ensure Basler Pylon SDK is installed.")
            return
            
        try:
            # Initialize and open the camera
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.Open()
            logging.info(f"Using device: "
                        f"{self.camera.GetDeviceInfo().GetModelName()}")

            # Configure for continuous acquisition
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_running.set()
            logging.info("Camera started grabbing frames.")

            while self.is_running.is_set():
                if not self.camera.IsGrabbing():
                    logging.warning("Camera stopped grabbing unexpectedly.")
                    break
                
                try:
                    grabResult = self.camera.RetrieveResult(
                        5000, pylon.TimeoutHandling_ThrowException
                    )
                    if grabResult.GrabSucceeded():
                        # Convert the image to a format OpenCV can use (BGR)
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray()
                        with self.lock:
                            self.latest_frame = frame.copy()
                    else:
                        logging.error(f"Grab failed: {grabResult.ErrorCode} "
                                   f"{grabResult.ErrorDescription}")
                    grabResult.Release()
                except GenericException as e:
                    logging.error(f"An error occurred while grabbing a frame: "
                               f"{e}")
                    time.sleep(0.1)  # Avoid tight loop on error
            
        except pylon.RuntimeException as e:
            logging.critical(f"Pylon runtime exception: {e}. "
                           f"Is a camera connected?")
        except Exception as e:
            logging.critical(f"An unexpected error occurred in "
                           f"PylonFrameGrabber: {e}", exc_info=True)
        finally:
            if self.camera and self.camera.IsGrabbing():
                self.camera.StopGrabbing()
                logging.info("Camera stopped grabbing.")
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                logging.info("Camera closed.")
            self.is_running.clear()
            logging.info("PylonFrameGrabber thread finished.")

    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        """Signals the thread to stop."""
        logging.info("Stopping PylonFrameGrabber thread.")
        self.is_running.clear()


# Make PYLON_AVAILABLE accessible as a class attribute
PylonFrameGrabber.PYLON_AVAILABLE = PYLON_AVAILABLE 