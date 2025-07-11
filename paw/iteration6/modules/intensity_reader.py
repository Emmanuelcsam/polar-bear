import numpy as np
from core.datastore import scan, put
from core.logger import log
def learn():
    h = [v for _,v in scan("hist:")]
    if not h: 
        log("learn","no histograms yet"); return
    dist = np.sum(h, axis=0).astype(np.float64)
    dist /= dist.sum()
    put("dist", dist); log("learn","distribution learned")
def next_guided():
    from .random_pixel import guided
    return guided()
log("learn","module ready")