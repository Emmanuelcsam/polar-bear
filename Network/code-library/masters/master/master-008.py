from modules import (cv_module, torch_module, random_pixel,
                     intensity_reader, pattern_recognizer,
                     anomaly_detector, batch_processor,
                     realtime_processor, hpc)
from core.logger import log
MENU = """
0 quit            5 train AE            10 cluster
1 rand img        6 gen imgs            11 anomalies
2 guided img      7 batch folder        12 realtime watch
3 learn dist      8 HPC                 13 show mask
4 save hist       9 list cats
"""
def main():
    while True:
        print(MENU)
        cmd = input("> ").strip()
        if cmd=="0": break
        elif cmd=="1": random_pixel.gen()
        elif cmd=="2": random_pixel.guided()
        elif cmd=="3": intensity_reader.learn()
        elif cmd=="4": cv_module.save_hist(input("path: "))
        elif cmd=="5": torch_module.train_folder(input("folder: "))
        elif cmd=="6": torch_module.generate(int(input("n: ")))
        elif cmd=="7": batch_processor.run(input("folder: "))
        elif cmd=="8": hpc.run(int(input("total imgs: ")))
        elif cmd=="9": 
            from core.datastore import scan
            print("cats:", scan("cat:"))
        elif cmd=="10": pattern_recognizer.cluster(int(input("k: ")))
        elif cmd=="11": anomaly_detector.detect(input("img: "))
        elif cmd=="12": realtime_processor.watch(input("folder: "))
        elif cmd=="13": cv_module.show_mask(input("img: "))
        else: log("main","unknown",cmd)
if __name__=="__main__": main()