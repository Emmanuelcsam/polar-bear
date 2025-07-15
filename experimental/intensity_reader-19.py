# intensity_reader.py
import sys
from PIL import Image
from data_store import save_event

# Control parameters
_params = {'mode': 'L', 'save_enabled': True}

def run(path):
    try:
        img = Image.open(path).convert(_params['mode'])
        data = list(img.getdata())
        if _params['save_enabled']:
            for p in data: save_event({'intensity':p, 'source':path})
        return {'path': path, 'pixels': len(data), 'mode': _params['mode']}
    except Exception as e:
        return {'error': str(e), 'path': path}

def read_image(path):
    # Read without saving
    img = Image.open(path).convert(_params['mode'])
    return list(img.getdata())

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['run', 'read_image']}

if __name__=='__main__': 
    if len(sys.argv)>1: print(run(sys.argv[1]))
    else: print("Usage: intensity_reader.py <image_path>")
