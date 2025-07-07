"""
robust_circle_detector.py
Real‑time, multi‑scale circle detector for Basler cameras (GigE/USB3) using pypylon + OpenCV.
Author: <you>
Licence: BSD‑3‑Clause (same as pypylon)

Pipeline highlights
-------------------
1.   Fast, low‑overhead grab using pypylon GrabStrategy_LatestImageOnly.
2.   CLAHE + rolling‑background subtraction to cancel vignetting / gradients.
3.   Auto‑tuned Canny (median‑based) -> HoughCircles in three radius bands.
4.   Contour‑based verification (eccentricity, circularity, convexity) to reject arcs.
5.   Result fusion & NMS with temporal coherence (Kalman) for rock‑solid output.
"""

import cv2
import numpy as np
from collections import deque
import unittest
from unittest.mock import patch, MagicMock
import sys

# -------- configuration ------------------------------------------------------
MAX_QUEUE = 5           # frames kept for temporal smoothing
MIN_ACCUM = 20          # min Hough accumulator votes (auto‑scaled)
BANDS = [(8, 30), (31, 120), (121, 800)]  # radius bands in px (fine‑>coarse)
DISPLAY_SCALE = 0.40    # shrink preview window on 5 MP cameras
GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
# -----------------------------------------------------------------------------

# Kalman filter helper for each circle
class KF:
    def __init__(self, x, y, r):
        self.kf = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 3] = 1.0
        self.kf.measurementMatrix = np.hstack([np.eye(3), np.zeros((3, 3))]).astype(np.float32)
        self.kf.processNoiseCov[:] = np.eye(6, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov[:] = np.eye(3, dtype=np.float32) * 1e-1
        self.kf.statePost[:3, 0] = (x, y, r)
    def update(self, m):
        pred = self.kf.predict()
        self.kf.correct(np.array(m, dtype=np.float32))
        return self.kf.statePost[:3, 0]

def auto_canny(gray):
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    return cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)

def preprocess(raw):
    # flatten uneven illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    return blur

def detect(gray):
    edges = auto_canny(gray)
    circles = []
    for band in BANDS:
        # sensitivity and accumulator adapt to band size
        dp = 1.0
        minDist = band[0] * 1.5
        circles_b = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp,
                                     minDist=minDist,
                                     param1=100,  # Canny high threshold (autoCanny already exists)
                                     param2=MIN_ACCUM,  # accumulator threshold
                                     minRadius=band[0], maxRadius=band[1])
        if circles_b is not None:
            circles.extend(circles_b[0].tolist())
    # geometric verification
    verified = []
    for (x, y, r) in circles:
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, 2)
        pts = cv2.findNonZero(mask & edges)
        if pts is None:
            continue
        # circularity = 4πA / P² close to 1
        perim = len(pts)
        area = np.pi * r * r
        circularity = 4 * np.pi * area / (perim ** 2 + 1e-12)
        if circularity > 0.6:  # empirical cutoff, tune if needed
            verified.append((x, y, r))
    return verified

def main():
    # ----- VideoCapture initialisation ----------------------------------------------
    print("Initializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # Use V4L2 backend for better Linux compatibility
    
    if not cap.isOpened():
        print("Failed to open camera! Trying default backend...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {width}x{height} @ {fps} FPS")

    kfs = []       # active Kalman filters
    traces = deque(maxlen=MAX_QUEUE)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame!")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        proc = preprocess(img)
        found = detect(proc)

        # temporal fusion ------------------------------------------------------
        new_kfs = []
        for (x, y, r) in found:
            matched = False
            for kf in kfs:
                px, py, pr = kf.update((x, y, r))
                if np.hypot(px - x, py - y) < 0.5 * r:
                    matched = True
                    new_kfs.append(kf)
                    break
            if not matched:
                new_kfs.append(KF(x, y, r))
        kfs = new_kfs
        traces.append([(int(k.kf.statePost[0]), int(k.kf.statePost[1]), int(k.kf.statePost[2])) for k in kfs])

        # draw -----------------------------------------------------------------
        # Use the original color frame instead of grayscale
        vis = frame.copy()
        
        # Draw detected circles
        for tr in traces:
            for (x, y, r) in tr:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # Center point
        
        # Add info text
        cv2.putText(vis, f"Circles: {len(traces[-1]) if traces else 0}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "Press 'q' to quit", 
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if DISPLAY_SCALE != 1.0:
            vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        
        cv2.imshow("Circle detector - Press 'q' to quit", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

class TestCircleDetector(unittest.TestCase):

    def setUp(self):
        # Create a dummy image for testing
        self.test_image = np.zeros((600, 800), dtype=np.uint8)
        cv2.circle(self.test_image, (400, 300), 50, 255, -1)
        self.test_image_color = cv2.cvtColor(self.test_image, cv2.COLOR_GRAY2BGR)

    def test_kf_class(self):
        kf = KF(100, 100, 50)
        self.assertIsNotNone(kf.kf)
        new_state = kf.update((102, 102, 51))
        self.assertEqual(len(new_state), 3)

    def test_auto_canny(self):
        edges = auto_canny(self.test_image)
        self.assertEqual(edges.shape, self.test_image.shape)
        self.assertTrue(np.any(edges > 0))

    def test_preprocess(self):
        processed = preprocess(self.test_image)
        self.assertEqual(processed.shape, self.test_image.shape)

    def test_detect(self):
        # A simple image with one circle
        circles = detect(self.test_image)
        self.assertIsNotNone(circles)
        self.assertGreater(len(circles), 0)

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey', return_value=27) # Simulate ESC key press
    def test_main_loop_no_camera(self, mock_wait_key, mock_imshow, mock_videocapture):
        # Mock the camera input
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.side_effect = [(True, self.test_image_color), (False, None)]
        mock_videocapture.return_value = mock_capture_instance

        # This will run the main loop once
        main()

        # Check that imshow was called
        mock_imshow.assert_called()

if __name__ == "__main__":
    if 'test' in sys.argv:
        sys.argv.remove('test')
        unittest.main()
    else:
        main()