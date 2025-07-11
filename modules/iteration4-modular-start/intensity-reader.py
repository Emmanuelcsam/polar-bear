# intensity_reader.py
import sys
from PIL import Image
from data_store import save_event
def run(path):
    for p in Image.open(path).convert('L').getdata(): save_event({'intensity':p})
if __name__=='__main__': run(sys.argv[1])
