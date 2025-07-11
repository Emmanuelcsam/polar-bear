# image_guided_generator.py
import numpy as np
from data_store import load_events
def gen(shape): arr=np.array([v['intensity'] for v in load_events()]).reshape(shape); return arr
if __name__=='__main__': print(gen((64,64)))
