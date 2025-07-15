# batch_processor.py
import glob,sys,os

# Control parameters
_params = {'enabled': True, 'pattern': '*', 'recursive': False}
_results = []

def process_file(path):
    # Process individual file
    return {'file': path, 'size': os.path.getsize(path) if os.path.exists(path) else 0}

def run_folder(d='.'):
    if not _params['enabled']: return []
    global _results
    pattern = os.path.join(d, _params['pattern'])
    files = glob.glob(pattern, recursive=_params['recursive'])
    _results = [process_file(f) for f in files]
    print(f'batch done: {len(_results)} files')
    return _results

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'last_results': len(_results)}
def get_results(): return _results

if __name__=='__main__': 
    d = sys.argv[1] if len(sys.argv)>1 else '.'
    run_folder(d)
