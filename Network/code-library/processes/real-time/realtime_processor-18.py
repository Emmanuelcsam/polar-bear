#!/usr/bin/env python3
"""
Run pixel_generator in a background thread for 10 s.
"""
import threading
import time
from pixel_generator import generate_pixels

def run_real_time(duration=10):
    """Run pixel generator in background for specified duration."""
    def background_generator():
        """Background function that generates pixels."""
        generate_pixels(interval=0.1, max_iterations=duration*10)

    t = threading.Thread(target=background_generator)
    t.daemon = True
    t.start()

    print(f"[Realtime] Running for {duration}s")
    time.sleep(duration)
    print("[Realtime] Done")

if __name__ == "__main__":
    run_real_time()
