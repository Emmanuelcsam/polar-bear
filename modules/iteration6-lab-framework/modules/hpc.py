try:
    import torch.multiprocessing as mp
    MP_AVAILABLE = True
except ImportError:
    import multiprocessing as mp
    MP_AVAILABLE = False
from core.config import CORES
from .random_pixel import gen
from core.logger import log

def _worker(n):
    for _ in range(n): gen()

def run(total=100):
    each = max(1,total//CORES)
    if MP_AVAILABLE:
        mp.spawn(_worker, args=(each,), nprocs=CORES, join=True)
    else:
        # Fallback to standard multiprocessing
        from multiprocessing import Process
        processes = []
        for _ in range(CORES):
            p = Process(target=_worker, args=(each,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    log("hpc","generated", total, "images across", CORES, "cores")

if __name__=="__main__": run()