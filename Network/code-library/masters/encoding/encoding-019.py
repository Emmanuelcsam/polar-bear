# learner.py
import pickle
import numpy as np
from data_store import load_events

# Control parameters
_params = {'model_path': 'model.pkl', 'feature': 'intensity', 'stats': True}
_model = None

def learn(mdl=None):
    if mdl is None: mdl = _params['model_path']
    events = load_events()
    if not events: return {'error': 'No data to learn from'}
    
    # Extract features
    feature = _params['feature']
    hist = [v.get(feature, v.get('pixel', 0)) for v in events]
    
    # Calculate stats if enabled
    model_data = {'history': hist}
    if _params['stats']:
        model_data['stats'] = {
            'mean': np.mean(hist),
            'std': np.std(hist),
            'min': min(hist),
            'max': max(hist),
            'count': len(hist)
        }
    
    # Save model
    with open(mdl, 'wb') as f:
        pickle.dump(model_data, f)
    
    global _model
    _model = model_data
    return {'saved': mdl, 'samples': len(hist)}

def predict(value=None):
    global _model
    if _model is None:
        try:
            with open(_params['model_path'], 'rb') as f:
                _model = pickle.load(f)
        except: return {'error': 'No model loaded'}
    
    if value is not None and 'stats' in _model:
        stats = _model['stats']
        deviation = abs(value - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
        return {'value': value, 'deviation': deviation, 'normal': deviation < 2}
    return _model

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'model_loaded': _model is not None}

if __name__=='__main__': print(learn())
