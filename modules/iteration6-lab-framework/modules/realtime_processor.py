import time, glob, pathlib, threading
from .cv_module import save_hist
from .intensity_reader import learn
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