import sys, pathlib
from .cv_module import batch
from .intensity_reader import learn
from core.logger import log
def run(folder="."):
    folder = pathlib.Path(folder)
    if not folder.exists(): raise SystemExit("folder not found")
    batch(folder)
    learn()
    log("batch","completed", folder)
if __name__=="__main__":
    run(sys.argv[1] if len(sys.argv)>1 else ".")