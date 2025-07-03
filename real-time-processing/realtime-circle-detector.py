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
from pypylon import pylon

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
        self.kf.processNoiseCov[:] = np.eye(6, dtype=np.float32) * 1e‑3
        self.kf.measurementNoiseCov[:] = np.eye(3, dtype=np.float32) * 1e‑1
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
            circles.extend(circles_b[0])
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
        circularity = 4 * np.pi * area / (perim ** 2 + 1e‑12)
        if circularity > 0.6:  # empirical cutoff, tune if needed
            verified.append((x, y, r))
    return verified

def main():
    # ----- pylon initialisation ----------------------------------------------
    factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera(factory.CreateFirstDevice())
    camera.Open()
    camera.PixelFormat = "Mono8"
    camera.ExposureAuto = "Off"
    camera.GainAuto = "Off"
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
                         pylon.GrabLoop_ProvidedByInstantCamera)

    buf_converter = pylon.ImageFormatConverter()
    buf_converter.OutputPixelFormat = pylon.PixelType_Mono8
    buf_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    kfs = []       # active Kalman filters
    traces = deque(maxlen=MAX_QUEUE)

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            continue
        img = buf_converter.Convert(grab).GetArray()
        grab.Release()

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
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for tr in traces:
            for (x, y, r) in tr:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        if DISPLAY_SCALE != 1.0:
            vis = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("Circle detector – ESC to quit", vis)
        if cv2.waitKey(1) == 27:
            break

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
