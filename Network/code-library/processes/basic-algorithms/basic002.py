import cv2, numpy as np, pathlib, glob
from core.datastore import put, log
RES = (32, 32)
def load_gray(path): return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
def hist(img): return np.bincount(img.flatten(), minlength=256)
def save_hist(path):
    h = hist(load_gray(path))
    put(f"hist:{path}", h)
def batch(folder):
    for p in glob.glob(f"{folder}/*"):
        ext = pathlib.Path(p).suffix.lower()
        if ext in [".png", ".jpg", ".jpeg"]: save_hist(p)
def anomalies(img, z=3):
    m, s = img.mean(), img.std()
    return np.abs(img - m) > z * s
def show_mask(path):
    m = anomalies(load_gray(path))
    cv2.imshow("anom", m.astype(np.uint8)*255); cv2.waitKey(1)
log("cv", "module ready")