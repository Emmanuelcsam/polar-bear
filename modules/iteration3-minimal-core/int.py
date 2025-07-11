# intensity_reader.py
import sys
from PIL import Image
def read_intensity(path, threshold=None, callback=print):
    data=Image.open(path).convert("L").getdata(); [callback(p) for p in data if threshold is None or p>=threshold]
if __name__=="__main__":
    v=sys.argv; read_intensity(v[1], int(v[2]) if len(v)>2 else None)
