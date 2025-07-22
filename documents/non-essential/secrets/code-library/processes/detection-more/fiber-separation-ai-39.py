"""
Thin wrapper to satisfy the original import `from separation import ...`.
Re‑exports AI_Segmenter behaviour.
"""

from ai_segmenter_pytorch import AI_Segmenter
import cv2, numpy as np
from pathlib import Path

# default model path – can also be given in config["separation_settings"]["weights"]
_DEFAULT_WEIGHTS = Path(__file__).parent / 'segmenter_best.pth'

_segmenter = None
def _lazy_load(cfg_path=_DEFAULT_WEIGHTS):
    global _segmenter
    if _segmenter is None:
        _segmenter = AI_Segmenter(cfg_path)

# ---------------------------------------------------------------------
def segment_image(image_path: str,
                  weights_path: str | None = None) -> dict:
    """
    Replacement for the multiple segmentation methods + consensus.
    Returns:
        { 'masks': {core, cladding, ferrule, defect}, 'center': (cx,cy),
          'core_radius': r_c, 'cladding_radius': r_cl }
    """
    _lazy_load(weights_path or _DEFAULT_WEIGHTS)
    bgr = cv2.imread(image_path)
    masks = _segmenter.segment(bgr)

    # naive circle‑fit for centre/radius (core)
    ys, xs = np.where(masks['core'] > 0)
    if len(xs) > 10:
        cx, cy = xs.mean(), ys.mean()
        core_r = np.sqrt(((xs - cx)**2 + (ys - cy)**2).mean())
    else:
        cx = cy = core_r = cl_r = None

    # cladding radius
    ys2, xs2 = np.where(masks['cladding'] > 0)
    if len(xs2) > 10:
        cl_r = np.sqrt(((xs2 - cx)**2 + (ys2 - cy)**2).mean())
    else:
        cl_r = None
    return {
        'masks': masks,
        'center': (cx, cy) if cx else None,
        'core_radius': core_r,
        'cladding_radius': cl_r,
        'success': True
    }