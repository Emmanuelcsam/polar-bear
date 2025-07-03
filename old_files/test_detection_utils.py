import sys
import pathlib
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / 'current-process'))
sys.path.append(str(root))
import detection
import numpy as np

def test_basic_stats_functions():
    ana = detection.OmniFiberAnalyzer(detection.OmniConfig(enable_visualization=False))
    data = np.array([0,1,2,3,4,5], dtype=np.float32)
    assert ana._compute_skewness(data) == 0
    assert ana._compute_kurtosis(data) != 0
    assert ana._compute_entropy(np.array([0,0,0,255], dtype=np.uint8)) >= 0
