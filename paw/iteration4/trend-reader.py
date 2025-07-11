# trend_recorder.py
import json
from data_store import load_events
def trends():
    d=[v['intensity'] for v in load_events()]
    print(json.dumps({'min':min(d),'max':max(d),'mean':sum(d)/len(d)}))
if __name__=='__main__': trends()
