# data_store.py
import json,time
def save_event(e,f='events.log'): e.update(ts=time.time()); open(f,'a').write(json.dumps(e)+'\n')
def load_events(f='events.log'): return [json.loads(l) for l in open(f)]
