# realtime_analyzer.py
"""
Continuously processes video frames through the existing segmentation +
defect‑detection pipeline, returning an annotated frame and a defect list.

* Re‑uses UnifiedSegmentationSystem and OmniFiberAnalyzer exactly as they are
  implemented in separation.py and detection.py – no internal changes needed.
"""

from pathlib import Path
import cv2
import numpy as np
import time
import threading
import shared_config # Import the shared configuration module

# Note: These imports assume the original modules exist
# from separation import UnifiedSegmentationSystem        
# from detection import OmniFiberAnalyzer, OmniConfig

# For demo purposes, we'll use our AI versions
# from ai_segmenter_pytorch import AI_Segmenter
# from anomaly_detector_pytorch import AI_AnomalyDetector

class RealTimeAnalyzer:
    """Initialise once, then call .analyze(frame) for every new video frame."""
    def __init__(self,
                 config_path: str = "config.json", # This might be overridden by shared_config
                 fast_segmentation_method: str = None, # Default to None, will be set by shared_config
                 min_frame_interval: float = None): # Default to None, will be set by shared_config
        """
        Args
        ----
        fast_segmentation_method : name of ONE segmentation script to call
                                   (must exist in zones_methods).  Using a
                                   single fast method is ≈10× quicker than the
                                   11‑method consensus.  If you prefer consensus
                                   simply set this to None.
        min_frame_interval       : minimum seconds between analysed frames.
                                   Frames that arrive sooner are skipped so that
                                   latency never accumulates.
        """
        # Initialize AI models instead of original pipeline
        self.segmenter = AI_Segmenter("segmenter_best.pth") if Path("segmenter_best.pth").exists() else None
        self.detector = AI_AnomalyDetector("cae_last.pth") if Path("cae_last.pth").exists() else None
        
        # Load configuration from shared_config.py
        self.current_config = shared_config.get_config()
        self.fast_method = self.current_config.get("fast_segmentation_method", fast_segmentation_method if fast_segmentation_method is not None else "ai")
        self.min_dt = self.current_config.get("min_frame_interval", min_frame_interval if min_frame_interval is not None else 0.05)

        self.last_ts      = 0.0
        self.lock         = threading.Lock()   # protects heavy pipeline
        self.status = "initialized" # Add a status variable

    def get_script_info(self):
        """Returns information about the script, its status, and exposed parameters."""
        return {
            "name": "Live Fiber Analyzer",
            "status": self.status,
            "parameters": {
                "fast_segmentation_method": self.fast_method,
                "min_frame_interval": self.min_dt,
                "log_level": self.current_config.get("log_level"),
                "data_source": self.current_config.get("data_source"),
                "processing_enabled": self.current_config.get("processing_enabled"),
                "threshold_value": self.current_config.get("threshold_value")
            }
        }

    def set_script_parameter(self, key, value):
        """Sets a specific parameter for the script and updates shared_config."""
        if key in self.current_config:
            self.current_config[key] = value
            shared_config.set_config_value(key, value) # Update shared config
            
            # Apply changes if they affect the running script
            if key == "min_frame_interval":
                self.min_dt = value
            elif key == "fast_segmentation_method":
                self.fast_method = value
            # Add more conditions here for other parameters that need immediate effect
            
            self.status = f"parameter '{key}' updated"
            return True
        return False

    # --------------------------------------------------------------------- #
    def analyze(self, frame_bgr: np.ndarray):
        """
        Returns
        -------
        vis_frame : BGR image with core/cladding boundaries and defect overlays
        defects   : list of dicts – [{'region': 'core', 'cx': 120, 'cy': 230,
                                      'area': 58, 'severity': 'HIGH'}, ...]
        """
        now = time.time()
        if now - self.last_ts < self.min_dt:
            return None, []                   # skip – caller can display prev.

        with self.lock:
            self.last_ts = now

            # --- 1 Segmentation -------------
            if self.segmenter:
                masks = self.segmenter.segment(frame_bgr)
                
                # Calculate center and radii
                ys, xs = np.where(masks['core'] > 0)
                if len(xs) > 10:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    r_core = int(np.sqrt(((xs - cx)**2 + (ys - cy)**2).mean()))
                else:
                    cx = cy = r_core = 0
                
                ys2, xs2 = np.where(masks['cladding'] > 0)
                if len(xs2) > 10 and cx > 0:
                    r_clad = int(np.sqrt(((xs2 - cx)**2 + (ys2 - cy)**2).mean()))
                else:
                    r_clad = 0
            else:
                # Fallback: simple circle detection
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                         param1=50, param2=30, minRadius=50, maxRadius=200)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    cx, cy, r_clad = circles[0, 0]
                    r_core = r_clad // 5  # Approximate
                    # Create simple masks
                    masks = {
                        'core': np.zeros(gray.shape, dtype=np.uint8),
                        'cladding': np.zeros(gray.shape, dtype=np.uint8),
                        'ferrule': np.zeros(gray.shape, dtype=np.uint8)
                    }
                    cv2.circle(masks['core'], (cx, cy), r_core, 255, -1)
                    cv2.circle(masks['cladding'], (cx, cy), r_clad, 255, -1)
                else:
                    return frame_bgr, []

            # --- 2 Detection -------------------------------------------------
            defects = []
            if self.detector:
                fiber_mask = masks['core'] | masks['cladding']
                score_map, defect_list = self.detector.detect(frame_bgr, fiber_mask)
                
                for d in defect_list:
                    x, y, w, h = d['bbox']
                    cx_def, cy_def = x + w//2, y + h//2
                    
                    # Determine region
                    if masks['core'][cy_def, cx_def] > 0:
                        region = "core"
                    elif masks['cladding'][cy_def, cx_def] > 0:
                        region = "cladding"
                    else:
                        region = "ferrule"
                    
                    # Determine severity based on confidence and area
                    if d['confidence'] > 0.8 or d['area_px'] > 100:
                        severity = "HIGH"
                    elif d['confidence'] > 0.5 or d['area_px'] > 50:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"
                    
                    defects.append({
                        'region':   region,
                        'cx':       int(cx_def),
                        'cy':       int(cy_def),
                        'area':     int(d['area_px']),
                        'severity': severity
                    })

            # --- 3 Visual overlay -------------------------------------------
            vis = frame_bgr.copy()
            # draw core / cladding circles
            if cx > 0 and cy > 0:
                cv2.circle(vis, (cx, cy), r_core,  (0,255,255), 2)  # core - yellow
                cv2.circle(vis, (cx, cy), r_clad,  (255,0,255), 2)  # clad - magenta
            
            # draw defects
            for d in defects:
                color = (0,0,255) if d['severity'] == "HIGH" else (0,165,255) if d['severity'] == "MEDIUM" else (0,255,0)
                cv2.circle(vis, (d['cx'], d['cy']), 6, color, -1)
                cv2.putText(vis, d['severity'][0],
                            (d['cx']+8, d['cy']+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
            
            # Add status text
            status = f"Defects: {len(defects)}"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            return vis, defects