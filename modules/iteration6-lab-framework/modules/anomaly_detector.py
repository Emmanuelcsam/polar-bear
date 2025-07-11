import numpy as np
from core.datastore import put, log
from .cv_module import load_gray, anomalies
def detect(path):
    img = load_gray(path)
    mask = anomalies(img)
    dev = np.where(mask)
    put(f"anom:{path}", dev)
    log("anom","deviations found", len(dev[0]))
log("anom","module ready")