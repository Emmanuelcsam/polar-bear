# data_store.py
import json,time

# Control variables for connector integration
_params = {'events_file': 'events.log', 'enabled': True}

def save_event(e,f=None):
    if f is None: f = _params['events_file']
    if not _params.get('enabled', True): return e
    e.update(ts=time.time())
    with open(f,'a') as file: file.write(json.dumps(e)+'\n')
    return e

def load_events(f=None):
    if f is None: f = _params['events_file']
    try: return [json.loads(l.strip()) for l in open(f) if l.strip()]
    except: return []

# Connector interface functions
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['save_event', 'load_events']}

if __name__ == '__main__':
    # Test standalone
    print(f"Data store params: {_params}")
