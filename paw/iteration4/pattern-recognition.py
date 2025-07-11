# pattern_recognizer.py
import numpy as np
from data_store import load_events
def patterns(): return np.unique([v.get('pixel',v.get('intensity')) for v in load_events()],return_counts=True)
if __name__=='__main__': print(patterns())
