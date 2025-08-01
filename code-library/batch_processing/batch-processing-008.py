import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modules.cv_module import batch
from modules.intensity_reader import learn
from core.logger import log
def run(folder="."):
    folder = pathlib.Path(folder)
    if not folder.exists(): raise SystemExit("folder not found")
    batch(folder)
    learn()
    log("batch","completed", folder)
if __name__=="__main__":
    run(sys.argv[1] if len(sys.argv)>1 else ".")