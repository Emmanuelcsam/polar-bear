import numpy as np, cv2, time, pathlib
from core.datastore import put, get, log
SIZE = 32
OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
def gen():
    img = np.random.randint(0,256,(SIZE,SIZE), np.uint8)
    name = f"rand_{int(time.time()*1000)}.png"
    cv2.imwrite(str(OUT/name), img)
    put(f"rand:{name}", img)
    return img
def guided():
    dist = get("dist")
    if dist is None: return gen()
    flat = np.random.choice(256, SIZE*SIZE, p=dist).astype(np.uint8)
    return flat.reshape(SIZE,SIZE)
log("rand","module ready")