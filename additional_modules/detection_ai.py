"""
AI-powered defect detection wrapper
"""

from anomaly_detector_pytorch import AI_AnomalyDetector
import cv2
from pathlib import Path

# default model path
_DEFAULT_WEIGHTS = Path(__file__).parent / 'cae_last.pth'

_detector = None
def _lazy_load(cfg_path=_DEFAULT_WEIGHTS):
    global _detector
    if _detector is None:
        _detector = AI_AnomalyDetector(cfg_path)

def detect_defects(image_path: str,
                   masks: dict = None,
                   weights_path: str | None = None) -> dict:
    """
    Replacement for OmniFiberAnalyzer anomaly detection.
    Returns defects list in the same schema used by data_acquisition.py
    """
    _lazy_load(weights_path or _DEFAULT_WEIGHTS)
    bgr = cv2.imread(image_path)
    
    # Create fiber mask from core + cladding if provided
    fiber_mask = None
    if masks and 'core' in masks and 'cladding' in masks:
        fiber_mask = masks['core'] | masks['cladding']
    
    score_map, defects = _detector.detect(bgr, fiber_mask)
    
    # Convert to expected format
    return {
        'score_map': score_map,
        'defects': defects,
        'anomaly_detected': len(defects) > 0
    }