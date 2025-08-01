try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import numpy as np
from core.datastore import scan, put
from core.logger import log

def cluster(k=5):
    if not SKLEARN_AVAILABLE:
        log("pattern", "scikit-learn not available")
        return
    rows = scan("hist:")
    if not rows: log("pattern","nothing to cluster"); return
    vecs = np.stack([v for _,v in rows])
    km = KMeans(k, n_init="auto").fit(vecs)
    for (key,_), label in zip(rows, km.labels_):
        put(f"cat:{key}", int(label))
    log("pattern", "clustered into", k, "groups")

if SKLEARN_AVAILABLE:
    log("pattern","module ready")
else:
    log("pattern","module loaded (scikit-learn not available)")