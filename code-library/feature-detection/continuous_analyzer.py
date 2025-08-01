import time
import json
import os
import subprocess
import sys

def continuous_analyze(duration_seconds=60):
    start_time = time.time()
    iteration = 0
    analysis_log = []
    
    print(f"[CONTINUOUS] Starting {duration_seconds}s analysis")
    
    # Scripts to run in rotation
    scripts = [
        'pattern_recognizer.py',
        'anomaly_detector.py',
        'geometry_analyzer.py',
        'intensity_analyzer.py',
        'trend_analyzer.py',
        'data_calculator.py'
    ]
    
    while (time.time() - start_time) < duration_seconds:
        iteration += 1
        current_script = scripts[iteration % len(scripts)]
        
        try:
            # Run analysis script
            if os.path.exists(current_script):
                print(f"[CONTINUOUS] Iteration {iteration}: Running {current_script}")
                
                result = subprocess.run(
                    [sys.executable, current_script],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                analysis_log.append({
                    'iteration': iteration,
                    'script': current_script,
                    'timestamp': time.time(),
                    'success': result.returncode == 0,
                    'elapsed': time.time() - start_time
                })
            
            # Check for new data
            if os.path.exists('pixel_data.json'):
                with open('pixel_data.json', 'r') as f:
                    data = json.load(f)
                    if data.get('timestamp', 0) > start_time:
                        print(f"[CONTINUOUS] New pixel data detected")
            
            # Brief pause
            time.sleep(1)
            
        except Exception as e:
            print(f"[CONTINUOUS] Error in iteration {iteration}: {e}")
    
    # Summary
    summary = {
        'total_iterations': iteration,
        'duration': duration_seconds,
        'analyses_per_second': iteration / duration_seconds,
        'scripts_run': len(analysis_log),
        'success_rate': sum(1 for a in analysis_log if a['success']) / len(analysis_log) if analysis_log else 0
    }
    
    with open('continuous_analysis_log.json', 'w') as f:
        json.dump({
            'summary': summary,
            'log': analysis_log[-100:]  # Last 100 entries
        }, f)
    
    print(f"[CONTINUOUS] Analysis complete")
    print(f"[CONTINUOUS] Ran {iteration} iterations in {duration_seconds}s")
    print(f"[CONTINUOUS] Success rate: {summary['success_rate']:.2%}")

if __name__ == "__main__":
    # Run for 60 seconds by default
    continuous_analyze(60)