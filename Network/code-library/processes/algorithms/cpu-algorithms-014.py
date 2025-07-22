import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.datastore import put, log
from modules.cv_module import load_gray, anomalies
def detect(path):
    img = load_gray(path)
    mask = anomalies(img)
    dev = np.where(mask)
    put(f"anom:{path}", dev)
    log("anom","deviations found", len(dev[0]))
log("anom","module ready")