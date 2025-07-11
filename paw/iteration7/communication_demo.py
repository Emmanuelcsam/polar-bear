#!/usr/bin/env python3
"""
Demo showing how intensity-reader and random-generator communicate
Run this to see the correlation system in action
"""

import subprocess
import time
import json
import os
import sys
import threading

def monitor_correlations():
    """Monitor and display correlations as they happen"""
    print("[DEMO] Monitoring correlations...")
    last_count = 0
    
    while True:
        try:
            if os.path.exists('correlations.json'):
                with open('correlations.json', 'r') as f:
                    correlations = json.load(f)
                    
                if len(correlations) > last_count:
                    new_corrs = correlations[last_count:]
                    for corr in new_corrs:
                        print(f"[MATCH!] Value {corr['value']} matched at pixel index {corr['pixel_index']}")
                    last_count = len(correlations)
                    
        except:
            pass
        
        time.sleep(0.1)

def main():
    print("=== Communication Demo ===")
    print("This shows how intensity-reader and random-generator communicate\n")
    
    # Check for image
    image_file = None
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_file = file
            break
    
    if not image_file:
        print("ERROR: No image found! Please add an image file.")
        return
    
    print(f"Using image: {image_file}")
    
    # Clean up old files
    for f in ['pixel_data.json', 'random_value.json', 'correlations.json']:
        if os.path.exists(f):
            os.remove(f)
    
    # Step 1: Read pixels
    print("\n1. Reading pixel intensities...")
    subprocess.run([sys.executable, 'pixel_reader.py'], capture_output=True)
    
    with open('pixel_data.json', 'r') as f:
        data = json.load(f)
        print(f"   Read {len(data['pixels'])} pixels")
        print(f"   First 10 values: {data['pixels'][:10]}")
    
    # Step 2: Start processes
    print("\n2. Starting background processes...")
    
    # Start random generator
    gen_process = subprocess.Popen(
        [sys.executable, 'random_generator.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("   ✓ Random generator started")
    
    # Start correlator
    corr_process = subprocess.Popen(
        [sys.executable, 'correlator.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("   ✓ Correlator started")
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_correlations, daemon=True)
    monitor_thread.start()
    
    print("\n3. Watching for correlations (press Ctrl+C to stop)...")
    print("   When random generator produces a value that matches a pixel,")
    print("   the correlator detects it and stores the match.\n")
    
    try:
        time.sleep(30)  # Run for 30 seconds
    except KeyboardInterrupt:
        pass
    
    # Clean up
    print("\n\nStopping processes...")
    gen_process.terminate()
    corr_process.terminate()
    
    # Show results
    if os.path.exists('correlations.json'):
        with open('correlations.json', 'r') as f:
            correlations = json.load(f)
            print(f"\nFound {len(correlations)} total correlations!")
            
            if correlations:
                # Show value distribution
                value_counts = {}
                for corr in correlations:
                    val = corr['value']
                    value_counts[val] = value_counts.get(val, 0) + 1
                
                print("\nMost matched values:")
                sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                for val, count in sorted_values[:5]:
                    print(f"   Value {val}: {count} matches")

if __name__ == "__main__":
    main()