# pixel_generator.py
import random, time
from data_store import save_event
def run():
    while True: save_event({'pixel':random.randint(0,255)}); time.sleep(0.01)
if __name__=='__main__': run()
