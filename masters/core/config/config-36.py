import os, pathlib, multiprocessing as mp
BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
DEVICE = "cuda" if os.getenv("USE_CPU") is None else "cpu"
CORES  = max(1, mp.cpu_count() - 1)
PRINT_PREFIX = "[config]"
print(PRINT_PREFIX, "base =", BASE, "device =", DEVICE, "cores =", CORES)