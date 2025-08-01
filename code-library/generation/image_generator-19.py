# image_generator.py
import numpy as np
from PIL import Image
from data_store import load_events

# Control parameters
_params = {'default_shape': (64,64), 'normalize': True, 'save_path': None}

def gen(shape=None):
    if shape is None: shape = _params['default_shape']
    events = load_events()
    if not events: return np.zeros(shape)
    
    # Extract intensity values
    intensities = [v.get('intensity', v.get('pixel', 0)) for v in events]
    
    # Reshape or pad/trim to fit shape
    total_pixels = shape[0] * shape[1]
    if len(intensities) < total_pixels:
        intensities.extend([0] * (total_pixels - len(intensities)))
    elif len(intensities) > total_pixels:
        intensities = intensities[:total_pixels]
    
    arr = np.array(intensities).reshape(shape)
    if _params['normalize']: arr = arr / 255.0
    return arr

def save_image(arr, path=None):
    if path is None: path = _params.get('save_path', 'generated.png')
    if arr.max() <= 1.0: arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)
    return path

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['gen', 'save_image']}

if __name__=='__main__': 
    arr = gen()
    print(f"Generated array shape: {arr.shape}, range: [{arr.min():.2f}, {arr.max():.2f}]")
