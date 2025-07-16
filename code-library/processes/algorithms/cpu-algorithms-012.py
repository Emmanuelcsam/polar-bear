# anomaly_detector.py
import numpy as np
from data_store import load_events

# Control parameters
_params = {'threshold': 50, 'method': 'deviation', 'enabled': True}

def anomalies(th=None):
    if th is None: th = _params['threshold']
    if not _params.get('enabled', True): return []
    events = load_events()
    if not events: return []
    vals=[v.get('pixel',v.get('intensity',0)) for v in events]
    mean_val = np.mean(vals)
    return [v for v in events if abs(v.get('pixel',v.get('intensity',0))-mean_val)>th]

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['anomalies']}
def process(data=None): return anomalies(data.get('threshold') if isinstance(data,dict) else data)

if __name__=='__main__': print(f"Found {len(anomalies())} anomalies")
