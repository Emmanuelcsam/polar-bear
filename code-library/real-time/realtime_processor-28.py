import time, glob, pathlib, threading
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modules.cv_module import save_hist
from modules.intensity_reader import learn
from core.logger import log
def _loop(folder, poll):
    seen = set()
    while True:
        paths = [p for p in glob.glob(f"{folder}/*") if p not in seen]
        for p in paths:
            save_hist(p); seen.add(p)
        if paths: learn()
        time.sleep(poll)
def watch(folder=".", poll=2):
    t = threading.Thread(target=_loop, args=(folder,poll), daemon=True)
    t.start(); log("rt","watching", folder)
if __name__=="__main__": watch()