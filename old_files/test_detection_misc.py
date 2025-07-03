import sys
import pathlib
import re
import numpy as np
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / 'current-process'))
sys.path.append(str(root))

import detection


def test_confidence_to_severity():
    cfg = detection.OmniConfig(enable_visualization=False)
    ana = detection.OmniFiberAnalyzer(cfg)
    assert ana._confidence_to_severity(0.95) == 'CRITICAL'
    assert ana._confidence_to_severity(0.75) == 'HIGH'
    assert ana._confidence_to_severity(0.05) == 'NEGLIGIBLE'


def test_get_timestamp_format():
    cfg = detection.OmniConfig(enable_visualization=False)
    ana = detection.OmniFiberAnalyzer(cfg)
    ts = ana._get_timestamp()
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", ts)


def test_sanitize_feature_value():
    cfg = detection.OmniConfig(enable_visualization=False)
    ana = detection.OmniFiberAnalyzer(cfg)
    assert ana._sanitize_feature_value([1.0]) == 1.0
    assert ana._sanitize_feature_value(np.nan) == 0.0
    assert ana._sanitize_feature_value("5") == 5.0
