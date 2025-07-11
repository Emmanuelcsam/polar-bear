# random_pixel_generator.py
import random, time, sys
def generate(min_val=0, max_val=255, delay=0, callback=print):
    while True: callback(random.randint(min_val, max_val)); time.sleep(delay)
if __name__=="__main__":
    a=sys.argv; generate(int(a[1]) if len(a)>1 else 0,
                       int(a[2]) if len(a)>2 else 255,
                       float(a[3]) if len(a)>3 else 0)
