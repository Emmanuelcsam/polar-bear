# geometry_analyzer.py
import numpy as np
from data_store import load_events
def geometry(w,h):
    a=np.array([v['intensity'] for v in load_events()]).reshape(w,h)
    return np.gradient(a)
if __name__=='__main__': print(geometry(64,64))
