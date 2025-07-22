# pixel_generator.py
import random, time, threading
from data_store import save_event

# Control parameters
_params = {'min': 0, 'max': 255, 'delay': 0.01, 'running': False}
_thread = None

def generate(): return {'pixel': random.randint(_params['min'], _params['max'])}

def run(duration=None):
    _params['running'] = True
    start = time.time()
    while _params['running']:
        if duration and (time.time()-start) > duration: break
        save_event(generate())
        time.sleep(_params['delay'])

def start_async(duration=None):
    global _thread
    if _thread and _thread.is_alive(): return False
    _thread = threading.Thread(target=run, args=(duration,))
    _thread.daemon = True
    _thread.start()
    return True

def stop(): _params['running'] = False; return True

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'functions': ['generate', 'run', 'start_async', 'stop']}

if __name__=='__main__': 
    try: run()
    except KeyboardInterrupt: print("\nStopped")
