
import os
import sys
import time
import logging
import logging.handlers
import threading
from queue import Queue

import cv2
import numpy as np

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

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. SSIM functionality disabled.")

# ======================================================================================
# 1. SYSTEM CONFIGURATION
# ======================================================================================
class SystemConfig:
    """
    Centralized configuration for all tunable parameters.
    This makes the system easier to adjust for different products or environments.
    """
    # --- General ---
    REFERENCE_IMAGE_PATH = "good.bmp"
    LOG_LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Preprocessing ---
    GAUSSIAN_BLUR_KERNEL = (7, 7)

    # --- SSIM Difference Thresholding ---
    # Lower this value if minor differences should be ignored.
    SSIM_THRESHOLD = 0.85

    # --- Scratch Detection (Morphological) ---
    # Kernel size should be larger than the scratch width.
    SCRATCH_KERNEL_SIZE = (25, 25)
    SCRATCH_BINARY_THRESHOLD = 50  # Threshold for the combined Top/Black-hat result.

    # --- Blob Detection (Contours) ---
    MIN_BLOB_AREA = 200  # Minimum pixel area to be considered a blob.
    MAX_BLOB_AREA = 5000 # Maximum pixel area.
    # Circularity: 1.0 is a perfect circle. Lower values are more irregular.
    # We use this to distinguish blobs from scratches.
    MIN_BLOB_CIRCULARITY = 0.3

    # --- Circle Detection (Hough Transform) ---
    HOUGH_DP = 1.2  # Inverse ratio of accumulator resolution.
    HOUGH_MIN_DIST = 100  # Minimum distance between detected circle centers.
    HOUGH_PARAM1 = 100  # Upper threshold for internal Canny edge detector.
    HOUGH_PARAM2 = 60   # Accumulator threshold for center detection.
    HOUGH_MIN_RADIUS = 10 # Minimum circle radius in pixels.
    HOUGH_MAX_RADIUS = 100 # Maximum circle radius in pixels.

# ======================================================================================
# 2. ASYNCHRONOUS LOGGING SETUP
# ======================================================================================
def setup_logging(log_level=logging.DEBUG):
    """
    Configures an asynchronous logging system.
    This prevents logging I/O from blocking the performance-critical threads.
    """
    log_queue = Queue()
    
    # Formatter for all log messages
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(threadName)-12s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler to send logs to the console from the listener thread
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Listener thread that pulls logs from the queue and sends to handlers
    listener = logging.handlers.QueueListener(log_queue, console_handler)
    listener.start()

    # Root logger configuration to use the queue
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    logging.info("Asynchronous logging system initialized.")
    return listener, log_queue

# ======================================================================================
# 3. REAL-TIME CAMERA FRAME GRABBER (PRODUCER)
# ======================================================================================
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
            self.converter.OutputBitAlignment = \
                pylon.OutputBitAlignment_MsbAligned

    def run(self):
        """The main loop of the grabbing thread."""
        logging.info("PylonFrameGrabber thread started.")
        
        if not PYLON_AVAILABLE:
            logging.critical("ERROR: Pylon SDK not available. Cannot use Basler camera.")
            logging.critical("Please install pypylon: pip install pypylon")
            logging.critical("And ensure Basler Pylon SDK is installed.")
            return
            
        try:
            # Initialize and open the camera
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            logging.info(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")

            # Configure for continuous acquisition
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.is_running.set()
            logging.info("Camera started grabbing frames.")

            while self.is_running.is_set():
                if not self.camera.IsGrabbing():
                    logging.warning("Camera stopped grabbing unexpectedly.")
                    break
                
                try:
                    grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded():
                        # Convert the image to a format OpenCV can use (BGR)
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray()
                        with self.lock:
                            self.latest_frame = frame.copy()
                    else:
                        logging.error(f"Grab failed: {grabResult.ErrorCode} {grabResult.ErrorDescription}")
                    grabResult.Release()
                except GenericException as e:
                    logging.error(f"An error occurred while grabbing a frame: {e}")
                    time.sleep(0.1) # Avoid tight loop on error
            
        except pylon.RuntimeException as e:
            logging.critical(f"Pylon runtime exception: {e}. Is a camera connected?")
        except Exception as e:
            logging.critical(f"An unexpected error occurred in PylonFrameGrabber: {e}", exc_info=True)
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

# ======================================================================================
# 4. DEFECT DETECTION ALGORITHMS
# ======================================================================================
def preprocess_image(frame):
    """Converts frame to grayscale, applies blur and histogram equalization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, SystemConfig.GAUSSIAN_BLUR_KERNEL, 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def compute_ssim_difference(ref_img, live_img):
    """Computes the SSIM difference map and returns a binary mask of defects."""
    # Ensure both images have the same size
    if ref_img.shape != live_img.shape:
        live_img = cv2.resize(live_img, (ref_img.shape[1], ref_img.shape[0]))
        logging.debug(f"Resized live image to match reference: {ref_img.shape}")
    
    if SKIMAGE_AVAILABLE:
        (score, diff) = ssim(ref_img, live_img, full=True)
        diff = (diff * 255).astype("uint8")
        
        if score > SystemConfig.SSIM_THRESHOLD:
            # If images are very similar, there are no significant defects.
            # Return an empty mask to save processing time.
            return None, score

        # Threshold the difference image to get a binary mask of the defects
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return thresh, score
    else:
        # Fallback to simple difference when scikit-image is not available
        diff = cv2.absdiff(ref_img, live_img)
        score = 1.0 - (np.mean(diff) / 255.0)
        
        if score > SystemConfig.SSIM_THRESHOLD:
            return None, score
            
        # Threshold the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return thresh, score

def detect_scratches(gray_frame):
    """Detects scratches using morphological Top-Hat and Black-Hat transforms."""
    detections = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, SystemConfig.SCRATCH_KERNEL_SIZE)
    
    # Top-Hat for bright scratches on dark background
    tophat = cv2.morphologyEx(gray_frame, cv2.MORPH_TOPHAT, kernel)
    
    # Black-Hat for dark scratches on bright background
    blackhat = cv2.morphologyEx(gray_frame, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine and threshold
    combined = cv2.add(tophat, blackhat)
    _, thresh = cv2.threshold(combined, SystemConfig.SCRATCH_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Filter by area to remove noise
        if cv2.contourArea(c) > 50: # A small threshold for scratch segments
            x, y, w, h = cv2.boundingRect(c)
            detections.append({
                "type": "Scratch",
                "location": (x, y, w, h),
                "confidence": 1.0
            })
    return detections

def detect_blobs(diff_mask):
    """Detects blobs by analyzing contours from the SSIM difference mask."""
    detections = []
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if SystemConfig.MIN_BLOB_AREA < area < SystemConfig.MAX_BLOB_AREA:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            if circularity > SystemConfig.MIN_BLOB_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(c)
                detections.append({
                    "type": "Blob",
                    "location": (x, y, w, h),
                    "confidence": area / SystemConfig.MAX_BLOB_AREA
                })
    return detections

def detect_circles(gray_frame, diff_mask):
    """Detects circular defects using Hough Transform and cross-references with diff_mask."""
    detections = []
    circles = cv2.HoughCircles(
        gray_frame,
        cv2.HOUGH_GRADIENT,
        dp=SystemConfig.HOUGH_DP,
        minDist=SystemConfig.HOUGH_MIN_DIST,
        param1=SystemConfig.HOUGH_PARAM1,
        param2=SystemConfig.HOUGH_PARAM2,
        minRadius=SystemConfig.HOUGH_MIN_RADIUS,
        maxRadius=SystemConfig.HOUGH_MAX_RADIUS
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            # Cross-reference with the difference mask to validate defect
            if diff_mask is not None:
                # Check if the circle center is within the image bounds
                if (0 <= center_y < diff_mask.shape[0] and 
                    0 <= center_x < diff_mask.shape[1]):
                    if diff_mask[center_y, center_x] == 255:
                        detections.append({
                            "type": "Circle",
                            "location": (center_x, center_y, radius),
                            "confidence": 1.0
                        })
            else:
                # If no diff_mask, accept all circles
                detections.append({
                    "type": "Circle",
                    "location": (center_x, center_y, radius),
                    "confidence": 1.0
                })
    return detections

# ======================================================================================
# 5. MAIN PROCESSING AND VISUALIZATION
# ======================================================================================
def process_frame(live_frame, ref_img_processed):
    """
    Orchestrates the entire defect detection pipeline for a single frame.
    """
    start_time = time.perf_counter()
    
    if live_frame is None or ref_img_processed is None:
        return None, None, None

    # --- Preprocessing ---
    live_img_processed = preprocess_image(live_frame)
    
    # --- Difference Analysis ---
    diff_mask, ssim_score = compute_ssim_difference(ref_img_processed, live_img_processed)
    
    all_detections = []
    if diff_mask is not None:
        logging.debug(f"SSIM score: {ssim_score:.4f}. Potential defects found, running detectors.")
        
        # --- Run Specialized Detectors ---
        scratch_detections = detect_scratches(live_img_processed)
        all_detections.extend(scratch_detections)
        
        blob_detections = detect_blobs(diff_mask)
        all_detections.extend(blob_detections)
        
        circle_detections = detect_circles(live_img_processed, diff_mask)
        all_detections.extend(circle_detections)

    else:
        logging.debug(f"SSIM score: {ssim_score:.4f}. No significant difference.")
        
    # --- Visualization ---
    annotated_frame = live_frame.copy()
    for det in all_detections:
        det_type = det["type"]
        if det_type == "Scratch" or det_type == "Blob":
            x, y, w, h = det["location"]
            color = (0, 0, 255) if det_type == "Scratch" else (0, 255, 255) # Red for scratch, Yellow for blob
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated_frame, det_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif det_type == "Circle":
            center_x, center_y, radius = det["location"]
            color = (255, 0, 0) # Blue for circle
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, 2)
            cv2.circle(annotated_frame, (center_x, center_y), 2, color, 3) # Center dot
            # Fix overflow error by ensuring coordinates are within bounds
            text_x = max(0, center_x - radius)
            text_y = max(0, center_y - radius - 10)
            cv2.putText(annotated_frame, det_type, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add FPS and status info to the frame
    end_time = time.perf_counter()
    processing_time = (end_time - start_time) * 1000
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Defects: {len(all_detections)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    logging.debug(f"Frame processing took {processing_time:.2f} ms. Found {len(all_detections)} defects.")
    
    return annotated_frame, all_detections, diff_mask

# ======================================================================================
# 6. MAIN APPLICATION
# ======================================================================================
if __name__ == "__main__":
    # --- Initialization ---
    log_listener, log_queue = setup_logging(SystemConfig.LOG_LEVEL)
    logging.info("Application starting.")

    # Load and process the reference "golden template" image
    if not os.path.exists(SystemConfig.REFERENCE_IMAGE_PATH):
        logging.warning(f"Reference image not found at '{SystemConfig.REFERENCE_IMAGE_PATH}'. "
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
    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        GUI_AVAILABLE = True
    except:
        GUI_AVAILABLE = False
        logging.warning("OpenCV GUI not available. Running in headless mode.")

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

            annotated_frame, defects, diff_mask = process_frame(live_frame, ref_image_processed)

            # Display results if GUI is available
            if GUI_AVAILABLE:
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
                    logging.info(f"Frame {frame_count}: Found {len(defects)} defects")
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logging.critical(f"An unhandled exception occurred in the main loop: {e}", exc_info=True)
    finally:
        # --- Graceful Shutdown ---
        logging.info("Starting graceful shutdown sequence.")
        
        # Stop the frame grabber thread
        if frame_grabber.is_alive():
            frame_grabber.stop()
            frame_grabber.join(timeout=5) # Wait for thread to finish
            if frame_grabber.is_alive():
                logging.warning("Frame grabber thread did not terminate cleanly.")

        # Close OpenCV windows if GUI is available
        if GUI_AVAILABLE:
            try:
                cv2.destroyAllWindows()
                logging.info("OpenCV windows destroyed.")
            except:
                logging.warning("Error destroying OpenCV windows.")

        # Stop the logging listener
        log_listener.stop()
        logging.info("Logging listener stopped.")
        
        logging.info("Application shutdown complete.")