# anomaly_detector.py
import numpy as np
from data_store import load_events
def anomalies(th=50):
    vals=[v.get('pixel',v.get('intensity')) for v in load_events()]
    return [v for v in load_events() if abs(v.get('pixel',v.get('intensity'))-np.mean(vals))>th]
if __name__=='__main__': print(anomalies())
