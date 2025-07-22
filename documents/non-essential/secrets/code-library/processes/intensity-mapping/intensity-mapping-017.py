# pattern_recognition.py
import numpy as np
from data_store import load_events

# Control parameters
_params = {'min_count': 2, 'top_n': 10, 'feature': 'pixel'}

def patterns():
    events = load_events()
    if not events: return {'error': 'No data available'}
    
    feature = _params['feature']
    values = [v.get(feature, v.get('intensity', 0)) for v in events]
    
    # Find unique patterns
    unique, counts = np.unique(values, return_counts=True)
    
    # Filter by minimum count
    min_count = _params['min_count']
    mask = counts >= min_count
    unique = unique[mask]
    counts = counts[mask]
    
    # Sort by frequency
    sorted_idx = np.argsort(counts)[::-1]
    
    # Get top N
    top_n = _params['top_n']
    if len(sorted_idx) > top_n:
        sorted_idx = sorted_idx[:top_n]
    
    patterns = [
        {'value': int(unique[i]), 'count': int(counts[i]), 'frequency': counts[i]/len(values)}
        for i in sorted_idx
    ]
    
    return {
        'patterns': patterns,
        'total_unique': len(unique),
        'total_samples': len(values)
    }

def find_sequences(length=3):
    events = load_events()
    if len(events) < length: return []
    
    feature = _params['feature']
    values = [v.get(feature, v.get('intensity', 0)) for v in events]
    
    sequences = {}
    for i in range(len(values) - length + 1):
        seq = tuple(values[i:i+length])
        sequences[seq] = sequences.get(seq, 0) + 1
    
    # Return most common sequences
    return sorted([(list(k), v) for k, v in sequences.items()], key=lambda x: x[1], reverse=True)[:5]

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['patterns', 'find_sequences']}

if __name__=='__main__': 
    result = patterns()
    print(f"Found {result.get('total_unique', 0)} unique patterns")
