#!/usr/bin/env python3

import cv2
import numpy as np
import time
import os
from collections import deque
import logging

# --- Setup Pylon Integration ---
# Attempt to import the pypylon library for Basler camera support.
PYLON_AVAILABLE = False
try:
    from pypylon import pylon
    PYLON_AVAILABLE = True
    print("INFO: Pylon SDK found. Basler camera support is enabled.")
except ImportError:
    print("WARNING: Pylon SDK not found. Falling back to standard webcam support.")
    print("         For industrial camera performance, please install the pypylon package.")

# --- Setup Logging ---
# Configure a robust logging system to print detailed, timestamped information
# to the terminal for real-time debugging and monitoring.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - [%(levelname)s] - (%(threadName)-10s) - %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger()


class Camera:
    """
    A unified camera interface for handling both Pylon industrial cameras and
    standard USB webcams. It abstracts the complexities of frame grabbing.
    """
    def __init__(self, camera_index=0, use_pylon=True):
        self.camera_index = camera_index
        self.use_pylon = use_pylon and PYLON_AVAILABLE
        self.camera = None
        self.is_grabbing = False
        self._initialize_camera()

    def _initialize_camera(self):
        """Initializes the camera device based on availability and configuration."""
        log.info("Initializing camera...")
        if self.use_pylon:
            try:
                factory = pylon.TlFactory.GetInstance()
                devices = factory.EnumerateDevices()
                if not devices:
                    log.warning("No Pylon devices found. Attempting webcam fallback.")
                    self.use_pylon = False
                else:
                    self.camera = pylon.InstantCamera(factory.CreateFirstDevice())
                    self.camera.Open()
                    # Optimize for performance
                    self.camera.PixelFormat.SetValue("Mono8")
                    self.camera.ExposureAuto.SetValue("Continuous")
                    self.camera.GainAuto.SetValue("Continuous")
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    self.is_grabbing = True
                    model_name = self.camera.GetDeviceInfo().GetModelName()
                    log.info(f"Successfully initialized Pylon camera: {model_name}")
                    return
            except Exception as e:
                log.error(f"Failed to initialize Pylon camera: {e}. Falling back to webcam.")
                self.use_pylon = False

        # Fallback to OpenCV VideoCapture
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            log.error(f"Failed to open webcam at index {self.camera_index}.")
            raise RuntimeError("Could not open any camera device.")
        log.info(f"Initialized webcam at index {self.camera_index}.")

    def read_frame(self):
        """Reads a single frame from the camera, handling both Pylon and webcam."""
        try:
            if self.use_pylon:
                if not self.is_grabbing:
                    return None
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    frame = res.Array
                    res.Release()
                    return frame
                return None
            else:
                ret, frame = self.camera.read()
                if ret:
                    # Convert to grayscale for consistent processing
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return None
        except Exception as e:
            log.error(f"Failed to read frame: {e}")
            return None

    def release(self):
        """Releases the camera resources cleanly."""
        log.info("Releasing camera resources.")
        if self.camera:
            if self.use_pylon and self.is_grabbing:
                self.camera.StopGrabbing()
                self.camera.Close()
            elif hasattr(self.camera, 'release'):
                self.camera.release()
        log.info("Camera released.")


class DefectDetector:
    """
    The core analysis engine. This class contains the algorithms for detecting
    scratches, blobs, and other anomalies by comparing a live frame to a
    reference image.
    """
    def __init__(self, ref_image_path="good.bmp"):
        self.ref_image_gray = self._load_reference_image(ref_image_path)
        self.ref_circles = self._analyze_reference_circles(self.ref_image_gray)
        self.gpu_available = hasattr(cv2, 'UMat')
        if self.gpu_available:
            log.info("GPU acceleration (UMat) is available and will be used.")
        else:
            log.info("GPU acceleration (UMat) not available, processing on CPU.")

    def _load_reference_image(self, path):
        """Loads and preprocesses the reference image."""
        log.info(f"Loading reference image from: {path}")
        if not os.path.exists(path):
            log.error(f"Reference image not found at '{path}'. Please provide a 'good.bmp' file.")
            raise FileNotFoundError("Reference image 'good.bmp' not found.")
        ref_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ref_image is None:
            raise IOError(f"Could not read reference image at {path}")
        log.info("Reference image loaded successfully.")
        return cv2.GaussianBlur(ref_image, (5, 5), 0)

    def _analyze_reference_circles(self, image):
        """Analyzes the reference image to find the main circular feature."""
        log.info("Analyzing reference image for core circle...")
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=20, maxRadius=int(image.shape[0] / 2)
        )
        if circles is not None:
            log.info(f"Found {len(circles[0])} circle(s) in reference image.")
            return np.uint16(np.around(circles))
        log.warning("No circles found in the reference image. Detection may be impaired.")
        return None

    def detect(self, frame_gray):
        """
        Runs the full suite of defect detection algorithms on a given frame.
        """
        detections = {
            'core': [],
            'scratches': [],
            'blobs': []
        }

        # Use UMat for GPU acceleration if available
        if self.gpu_available:
            frame_umat = cv2.UMat(frame_gray)
            detections['core'] = self._detect_core(frame_umat)
            detections['scratches'] = self._detect_scratches(frame_umat)
            detections['blobs'] = self._detect_blobs(frame_umat)
        else:
            detections['core'] = self._detect_core(frame_gray)
            detections['scratches'] = self._detect_scratches(frame_gray)
            detections['blobs'] = self._detect_blobs(frame_gray)

        return detections

    def _detect_core(self, image):
        """Detects the main circular object in the frame."""
        log.debug("Detecting core circle...")
        
        if isinstance(image, cv2.UMat):
            image_shape = image.get().shape
        else:
            image_shape = image.shape
            
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=30, minRadius=20, maxRadius=int(image_shape[0] / 2)
        )
        detected_cores = []
        if circles is not None:
            # FIX: Convert UMat to numpy array before using numpy functions
            if isinstance(circles, cv2.UMat):
                circles_np = circles.get()
            else:
                circles_np = circles

            if circles_np is not None:
                circles_np = np.uint16(np.around(circles_np))
                for i in circles_np[0, :]:
                    detected_cores.append({'center': (i[0], i[1]), 'radius': i[2]})
        log.debug(f"Core detection found {len(detected_cores)} candidate(s).")
        return detected_cores

    def _detect_scratches(self, image):
        """
        Detects linear defects (scratches) using morphological operations.
        This method is highly sensitive to thin, dark lines on a lighter background.
        """
        log.debug("Detecting scratches...")
        scratches = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

        if isinstance(thresh, cv2.UMat):
            thresh_np = thresh.get()
        else:
            thresh_np = thresh
        contours, _ = cv2.findContours(thresh_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > h * 3 or h > w * 3:
                    scratches.append({'bbox': (x, y, w, h), 'contour': cnt})
        log.debug(f"Scratch detection found {len(scratches)} candidate(s).")
        return scratches

    def _detect_blobs(self, image):
        """
        Detects non-linear defects (blobs, contamination) using adaptive thresholding.
        """
        log.debug("Detecting blobs...")
        blobs = []
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        if isinstance(morphed, cv2.UMat):
            morphed_np = morphed.get()
        else:
            morphed_np = morphed
        contours, _ = cv2.findContours(morphed_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                blobs.append({'bbox': cv2.boundingRect(cnt), 'area': area, 'contour': cnt})
        log.debug(f"Blob detection found {len(blobs)} candidate(s).")
        return blobs


class UIController:
    """
    Manages the user interface, including the interactive circle overlay,
    keyboard inputs, and drawing results on the screen.
    """
    def __init__(self, window_name="Real-Time Defect Detection"):
        self.window_name = window_name
        self.center = [400, 300]
        self.radius = 100
        self.is_locked = False
        self.detection_mode = "HYBRID"  # HYBRID, AUTOMATIC, MANUAL
        self.move_step = 10
        self.resize_step = 5
        self.frame_times = deque(maxlen=60)
        self.last_time = time.time()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1024, 768)
        log.info("UI Controller initialized.")

    def handle_input(self, key):
        """Handles keyboard inputs for controlling the UI."""
        if key == 27:  # ESC
            return False

        if key == ord('m'):
            modes = ["HYBRID", "AUTOMATIC", "MANUAL"]
            current_index = modes.index(self.detection_mode)
            self.detection_mode = modes[(current_index + 1) % len(modes)]
            log.info(f"Detection mode changed to: {self.detection_mode}")
        elif key == ord('l'):
            self.is_locked = not self.is_locked
            log.info(f"Manual overlay lock: {'ON' if self.is_locked else 'OFF'}")
        elif key == ord('r'):
            self.center, self.radius = [400, 300], 100
            log.info("Manual overlay reset to default position.")

        if not self.is_locked:
            if key == ord('w'): self.center[1] -= self.move_step
            if key == ord('s'): self.center[1] += self.move_step
            if key == ord('a'): self.center[0] -= self.move_step
            if key == ord('d'): self.center[0] += self.move_step
            if key == ord('q'): self.radius = max(10, self.radius - self.resize_step)
            if key == ord('e'): self.radius += self.resize_step
        return True

    def draw_overlay(self, frame, detections):
        """Draws all UI elements and detection results onto the frame."""
        is_gray = False
        if isinstance(frame, cv2.UMat):
            if len(frame.get().shape) == 2:
                is_gray = True
        else:
            if len(frame.shape) == 2:
                is_gray = True

        if is_gray:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            if isinstance(frame, np.ndarray):
                 display_frame = frame.copy()
            else:
                 display_frame = frame

        if hasattr(cv2, 'UMat') and not isinstance(display_frame, cv2.UMat):
            display_umat = cv2.UMat(display_frame)
        else:
            display_umat = display_frame

        if self.detection_mode in ["HYBRID", "AUTOMATIC"]:
            for scratch in detections.get('scratches', []):
                x, y, w, h = scratch['bbox']
                cv2.rectangle(display_umat, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_umat, "Scratch", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for blob in detections.get('blobs', []):
                x, y, w, h = blob['bbox']
                cv2.rectangle(display_umat, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(display_umat, f"Blob ({blob['area']:.0f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            for core in detections.get('core', []):
                cv2.circle(display_umat, core['center'], core['radius'], (0, 255, 0), 2)
                cv2.putText(display_umat, "Core", (core['center'][0], core['center'][1] - core['radius'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.detection_mode in ["HYBRID", "MANUAL"]:
            color = (0, 0, 255) if self.is_locked else (255, 255, 0)
            cv2.circle(display_umat, tuple(self.center), self.radius, color, 2)
            cv2.circle(display_umat, tuple(self.center), 3, color, -1)

        self._draw_info_text(display_umat)
        
        if isinstance(display_umat, cv2.UMat):
            return display_umat.get()
        else:
            return display_umat


    def _draw_info_text(self, frame):
        """Draws the informational text panel on the frame."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0

        info = [
            f"FPS: {fps:.1f}",
            f"Mode: {self.detection_mode}",
            f"Lock: {'ON' if self.is_locked else 'OFF'}",
            f"Manual Circle: C=({self.center[0]},{self.center[1]}) R={self.radius}"
        ]
        
        # Draw text directly onto the UMat or numpy array
        for i, text in enumerate(info):
             cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    def display(self, frame):
        """Displays the final frame in the OpenCV window."""
        cv2.imshow(self.window_name, frame)


def main():
    """The main function to run the entire application."""
    log.info("Application starting...")
    try:
        camera = Camera(use_pylon=True)
        detector = DefectDetector(ref_image_path="good.bmp")
        ui = UIController()

        frame_count = 0
        log.info("Entering main processing loop...")
        while True:
            start_time = time.time()
            frame_count += 1
            log.debug(f"--- Frame {frame_count} ---")

            # 1. Acquire Frame
            frame_gray = camera.read_frame()
            if frame_gray is None:
                log.warning("Failed to acquire frame. Retrying...")
                time.sleep(0.1)
                continue
            log.debug("Frame acquired successfully.")

            # 2. Detect Defects
            detections = {}
            if ui.detection_mode in ["HYBRID", "AUTOMATIC"]:
                detections = detector.detect(frame_gray)
                log.debug(f"Detection complete. Scratches: {len(detections['scratches'])}, Blobs: {len(detections['blobs'])}")

            # 3. Draw UI and Detections
            display_frame = ui.draw_overlay(frame_gray, detections)
            log.debug("Overlay drawn.")

            # 4. Display Frame
            ui.display(display_frame)
            log.debug("Frame displayed.")

            # 5. Handle Input
            key = cv2.waitKey(1) & 0xFF
            if not ui.handle_input(key):
                log.info("Exit signal received. Shutting down.")
                break
            
            processing_time = (time.time() - start_time) * 1000
            log.debug(f"End of frame {frame_count}. Total processing time: {processing_time:.2f} ms")

    except Exception as e:
        log.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
        log.info("Application has been shut down cleanly.")


if __name__ == "__main__":
    main()
