# trend_reader.py
import json
import numpy as np
from data_store import load_events

# Control parameters
_params = {'window_size': 10, 'feature': 'intensity', 'format': 'json'}

def trends(window=None):
    if window is None: window = _params['window_size']
    events = load_events()
    if not events: return {'error': 'No data available'}
    
    feature = _params['feature']
    d = [v.get(feature, v.get('pixel', 0)) for v in events]
    
    # Calculate trends
    result = {
        'min': min(d),
        'max': max(d),
        'mean': sum(d)/len(d),
        'std': np.std(d),
        'count': len(d)
    }
    
    # Moving average if window specified
    if window > 1 and len(d) >= window:
        ma = [sum(d[i:i+window])/window for i in range(len(d)-window+1)]
        result['moving_avg'] = ma[-10:]  # Last 10 values
        result['trend'] = 'up' if ma[-1] > ma[0] else 'down' if ma[-1] < ma[0] else 'stable'
    
    if _params['format'] == 'json':
        return json.dumps(result)
    return result

def get_latest(n=10):
    events = load_events()
    if not events: return []
    feature = _params['feature']
    return [v.get(feature, v.get('pixel', 0)) for v in events[-n:]]

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['trends', 'get_latest']}

if __name__=='__main__': print(trends())
