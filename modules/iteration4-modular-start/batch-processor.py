# batch_processor.py
import glob,sys
from intensity_reader import run as read
def run_folder(d): [read(p) for p in glob.glob(d+'/*')]; print('batch done')
if __name__=='__main__': run_folder(sys.argv[1])
