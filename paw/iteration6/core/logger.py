import time
def log(tag, *msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}]", *msg, flush=True)