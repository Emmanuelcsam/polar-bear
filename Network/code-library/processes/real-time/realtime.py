#!/usr/bin/env python3
"""
Real-Time Processing Demo
Shows live capture, real-time analysis, and stream processing working together
"""

import subprocess
import time
import os
import sys
import threading
import json

def start_process(script, name):
    """Start a process in the background"""
    print(f"[DEMO] Starting {name}...")
    
    process = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start thread to read output
    def read_output():
        for line in iter(process.stdout.readline, b''):
            print(f"  {line.decode().strip()}")
    
    thread = threading.Thread(target=read_output)
    thread.daemon = True
    thread.start()
    
    return process

def show_live_stats():
    """Display live statistics from all components"""
    print("\n=== LIVE STATISTICS ===")
    
    stats_files = [
        ('realtime_metrics.json', 'Real-Time Processor'),
        ('stream_analysis.json', 'Stream Analyzer'),
        ('live_capture_stats.json', 'Live Capture')
    ]
    
    for filename, component in stats_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                print(f"\n{component}:")
                
                if 'processing_rate' in data:
                    print(f"  Processing Rate: {data['processing_rate']:.1f} pixels/sec")
                
                if 'stream_rates' in data:
                    for stream, rate in data['stream_rates'].items():
                        if rate > 0:
                            print(f"  {stream}: {rate:.2f}/sec")
                
                if 'total_frames' in data:
                    print(f"  Total Frames: {data['total_frames']}")
                    print(f"  Average FPS: {data['average_fps']:.1f}")
                
            except:
                pass

def main():
    print("=== REAL-TIME PROCESSING DEMO ===")
    print("This demonstrates live capture, real-time analysis, and stream processing\n")
    
    # Check if we have an image for initial setup
    if not any(f.endswith(('.jpg', '.png', '.bmp')) for f in os.listdir('.')):
        print("Creating test image...")
        subprocess.run([sys.executable, 'create_test_image.py'], capture_output=True)
    
    processes = []
    
    try:
        print("\n=== Phase 1: Starting Core Components ===")
        
        # 1. Start pixel reader for initial data
        print("\n[DEMO] Reading initial image...")
        subprocess.run([sys.executable, 'pixel_reader.py'], capture_output=True)
        print("✓ Initial pixel data ready")
        
        # 2. Start random generator
        print("\n[DEMO] Starting random generator...")
        gen_process = start_process('random_generator.py', 'Random Generator')
        processes.append(gen_process)
        time.sleep(1)
        
        # 3. Start correlator
        print("\n[DEMO] Starting correlator...")
        corr_process = start_process('correlator.py', 'Correlator')
        processes.append(corr_process)
        time.sleep(1)
        
        print("\n=== Phase 2: Starting Real-Time Components ===")
        
        # 4. Start stream analyzer
        print("\n[DEMO] Starting stream analyzer...")
        stream_process = subprocess.Popen(
            [sys.executable, 'stream_analyzer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(stream_process)
        time.sleep(2)
        
        # 5. Start real-time processor
        print("\n[DEMO] Starting real-time processor...")
        realtime_process = subprocess.Popen(
            [sys.executable, 'realtime_processor.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(realtime_process)
        time.sleep(2)
        
        # 6. Start live capture
        print("\n[DEMO] Starting live capture (simulated camera)...")
        capture_process = subprocess.Popen(
            [sys.executable, 'live_capture.py', 'fps=5'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(capture_process)
        
        print("\n=== Phase 3: Real-Time Processing Active ===")
        print("\nAll components are running!")
        print("The system is now:")
        print("  • Capturing live frames (simulated or real camera)")
        print("  • Detecting patterns and anomalies in real-time")
        print("  • Analyzing data streams for trends and correlations")
        print("  • Triggering analysis modules based on conditions")
        print("  • Generating continuous insights")
        
        print("\nPress Ctrl+C to stop the demo...")
        print("\n" + "="*50 + "\n")
        
        # Monitor for 30 seconds
        start_time = time.time()
        duration = 30
        
        while (time.time() - start_time) < duration:
            remaining = duration - (time.time() - start_time)
            print(f"\rDemo running... {int(remaining)}s remaining", end='')
            time.sleep(1)
            
            # Show stats every 10 seconds
            if int(remaining) % 10 == 0 and remaining > 0:
                show_live_stats()
                print(f"\n\rDemo running... {int(remaining)}s remaining", end='')
        
        print("\n\n=== Demo Time Expired ===")
        
    except KeyboardInterrupt:
        print("\n\n=== Demo Interrupted ===")
    
    finally:
        print("\nStopping all processes...")
        
        # Terminate all processes
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                process.kill()
        
        print("✓ All processes stopped")
        
        # Show final statistics
        print("\n=== FINAL RESULTS ===")
        
        results_files = [
            ('realtime_report.json', 'Real-Time Processing Report'),
            ('stream_report.json', 'Stream Analysis Report'),
            ('live_capture_stats.json', 'Live Capture Statistics'),
            ('realtime_triggers.json', 'Triggered Events')
        ]
        
        for filename, title in results_files:
            if os.path.exists(filename):
                print(f"\n{title}:")
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    
                    if 'total_events' in data:
                        print(f"  Total Events: {data['total_events']}")
                    
                    if 'stats' in data:
                        stats = data['stats']
                        if 'pixels_processed' in stats:
                            print(f"  Pixels Processed: {stats['pixels_processed']:,}")
                        if 'correlations_found' in stats:
                            print(f"  Correlations Found: {stats['correlations_found']}")
                        if 'triggers_fired' in stats:
                            print(f"  Triggers Fired: {stats['triggers_fired']}")
                    
                    if 'total_frames' in data:
                        print(f"  Frames Captured: {data['total_frames']}")
                        print(f"  Average FPS: {data.get('average_fps', 0):.1f}")
                    
                    if 'type' in data:  # Trigger data
                        print(f"  Last Trigger: {data['type']}")
                        print(f"  Trigger Value: {data.get('value', 'N/A')}")
                    
                except:
                    pass
        
        print("\n=== Generated Files ===")
        output_files = [
            'realtime_metrics.json',
            'realtime_report.json',
            'realtime_triggers.json',
            'stream_analysis.json',
            'stream_alerts.json',
            'stream_report.json',
            'live_frame.json',
            'live_buffer.json',
            'live_capture_stats.json',
            'live_current.jpg'
        ]
        
        existing_files = [f for f in output_files if os.path.exists(f)]
        for f in existing_files:
            print(f"  ✓ {f}")
        
        print("\n=== What Just Happened ===")
        print("The real-time system:")
        print("1. Captured live frames continuously")
        print("2. Analyzed pixel patterns in real-time")
        print("3. Detected anomalies and correlations as they occurred")
        print("4. Triggered appropriate analysis modules automatically")
        print("5. Tracked trends across multiple data streams")
        print("6. Generated alerts for significant events")
        
        print("\nTry these next:")
        print("  • Run individual components: python realtime_processor.py")
        print("  • Monitor live feed: python live_capture.py monitor")
        print("  • View real-time dashboard: python realtime_processor.py")
        print("  • Analyze streams: python stream_analyzer.py")

if __name__ == "__main__":
    main()