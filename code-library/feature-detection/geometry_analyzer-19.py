# geometry_analyzer.py
import numpy as np
from data_store import load_events

# Control parameters
_params = {'default_shape': (64,64), 'feature': 'intensity', 'analysis': 'gradient'}

def geometry(w=None, h=None):
    if w is None or h is None:
        w, h = _params['default_shape']
    
    events = load_events()
    if not events: return np.zeros((w,h))
    
    feature = _params['feature']
    values = [v.get(feature, v.get('pixel', 0)) for v in events]
    
    # Ensure we have enough data
    required = w * h
    if len(values) < required:
        values.extend([0] * (required - len(values)))
    elif len(values) > required:
        values = values[:required]
    
    a = np.array(values).reshape(w, h)
    
    # Analysis type
    analysis = _params['analysis']
    if analysis == 'gradient':
        return np.gradient(a)
    elif analysis == 'edges':
        gx, gy = np.gradient(a)
        return np.sqrt(gx**2 + gy**2)
    elif analysis == 'corners':
        from scipy import ndimage
        return ndimage.generic_gradient_magnitude(a, ndimage.sobel)
    else:
        return a

def analyze_shape(threshold=10):
    w, h = _params['default_shape']
    grad = geometry(w, h)
    if isinstance(grad, tuple):
        gx, gy = grad
        edges = np.sqrt(gx**2 + gy**2)
    else:
        edges = grad
    
    # Find significant features
    features = np.where(edges > threshold)
    return {
        'edge_count': len(features[0]),
        'mean_gradient': float(np.mean(edges)),
        'max_gradient': float(np.max(edges)),
        'shape': (w, h)
    }

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['geometry', 'analyze_shape']}

if __name__=='__main__': 
    result = analyze_shape()
    print(f"Shape analysis: {result}")
