import json
import time
import os
import numpy as np
from collections import deque
import subprocess
import sys
import threading

class RealtimeProcessor:
    def __init__(self):
        self.running = True
        self.data_buffer = {
            'pixels': deque(maxlen=1000),
            'correlations': deque(maxlen=100),
            'predictions': deque(maxlen=50),
            'anomalies': deque(maxlen=20)
        }
        self.file_states = {}
        self.stats = {
            'pixels_processed': 0,
            'correlations_found': 0,
            'anomalies_detected': 0,
            'triggers_fired': 0
        }
        self.thresholds = self.load_thresholds()
        
    def load_thresholds(self):
        """Load or create threshold configuration"""
        if os.path.exists('realtime_config.json'):
            with open('realtime_config.json', 'r') as f:
                return json.load(f)
        else:
            # Default thresholds
            return {
                'anomaly_rate': 0.1,  # Trigger if >10% anomalies
                'correlation_burst': 10,  # Trigger if >10 correlations/second
                'pixel_variance': 5000,  # Trigger if variance exceeds this
                'edge_density': 0.2,  # Trigger vision if edge density high
                'pattern_confidence': 0.8  # Trigger learning if pattern strong
            }
    
    def monitor_file(self, filepath, data_type):
        """Monitor a JSON file for changes"""
        last_modified = 0
        
        while self.running:
            try:
                if os.path.exists(filepath):
                    current_modified = os.path.getmtime(filepath)
                    
                    if current_modified > last_modified:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        self.process_new_data(data, data_type)
                        last_modified = current_modified
                        
            except Exception as e:
                pass
            
            time.sleep(0.1)
    
    def process_new_data(self, data, data_type):
        """Process incoming data based on type"""
        
        if data_type == 'pixels':
            if 'pixels' in data:
                new_pixels = data['pixels']
                self.data_buffer['pixels'].extend(new_pixels[-100:])  # Last 100
                self.stats['pixels_processed'] += len(new_pixels)
                
                # Real-time statistics
                if len(self.data_buffer['pixels']) > 10:
                    recent_pixels = list(self.data_buffer['pixels'])
                    variance = np.var(recent_pixels)
                    mean_val = np.mean(recent_pixels)
                    
                    print(f"[REALTIME] Pixel stats - Mean: {mean_val:.1f}, Var: {variance:.1f}")
                    
                    # Check for triggers
                    if variance > self.thresholds['pixel_variance']:
                        self.trigger_action('high_variance', variance)
        
        elif data_type == 'correlations':
            if isinstance(data, list):
                new_correlations = data[-10:]  # Last 10
                self.data_buffer['correlations'].extend(new_correlations)
                self.stats['correlations_found'] += len(new_correlations)
                
                # Check correlation rate
                if len(new_correlations) > self.thresholds['correlation_burst']:
                    self.trigger_action('correlation_burst', len(new_correlations))
        
        elif data_type == 'anomalies':
            if 'z_score_anomalies' in data:
                anomalies = data['z_score_anomalies']
                self.data_buffer['anomalies'].extend(anomalies[-5:])
                self.stats['anomalies_detected'] += len(anomalies)
                
                # Check anomaly rate
                if self.stats['pixels_processed'] > 0:
                    anomaly_rate = self.stats['anomalies_detected'] / self.stats['pixels_processed']
                    if anomaly_rate > self.thresholds['anomaly_rate']:
                        self.trigger_action('high_anomaly_rate', anomaly_rate)
        
        elif data_type == 'vision':
            if 'edges' in data:
                edge_ratio = data['edges'].get('canny_edge_ratio', 0)
                if edge_ratio > self.thresholds['edge_density']:
                    self.trigger_action('high_edge_density', edge_ratio)
        
        elif data_type == 'neural':
            if 'predictions' in data:
                self.data_buffer['predictions'].extend(data['predictions'][-10:])
                
                # Check prediction confidence
                if len(self.data_buffer['predictions']) > 5:
                    recent_preds = list(self.data_buffer['predictions'])
                    pred_std = np.std(recent_preds)
                    if pred_std < 10:  # Low variance = high confidence
                        self.trigger_action('stable_predictions', pred_std)
    
    def trigger_action(self, trigger_type, value):
        """Trigger actions based on conditions"""
        self.stats['triggers_fired'] += 1
        
        print(f"[REALTIME] TRIGGER: {trigger_type} (value: {value:.2f})")
        
        # Save trigger event
        trigger_data = {
            'timestamp': time.time(),
            'type': trigger_type,
            'value': float(value),
            'stats': self.stats.copy()
        }
        
        with open('realtime_triggers.json', 'w') as f:
            json.dump(trigger_data, f)
        
        # Trigger appropriate modules
        if trigger_type == 'high_variance':
            print("[REALTIME] High variance detected - triggering pattern analysis")
            self.run_module_async('pattern_recognizer.py')
            
        elif trigger_type == 'correlation_burst':
            print("[REALTIME] Correlation burst - triggering neural learning")
            self.run_module_async('neural_learner.py')
            
        elif trigger_type == 'high_anomaly_rate':
            print("[REALTIME] High anomaly rate - triggering deep analysis")
            self.run_module_async('data_calculator.py')
            
        elif trigger_type == 'high_edge_density':
            print("[REALTIME] High edge density - triggering image generation")
            self.run_module_async('neural_generator.py')
            
        elif trigger_type == 'stable_predictions':
            print("[REALTIME] Stable predictions - triggering trend analysis")
            self.run_module_async('trend_analyzer.py')
    
    def run_module_async(self, module_name):
        """Run a module asynchronously"""
        if os.path.exists(module_name):
            thread = threading.Thread(
                target=lambda: subprocess.run([sys.executable, module_name], capture_output=True)
            )
            thread.daemon = True
            thread.start()
    
    def calculate_realtime_metrics(self):
        """Calculate real-time performance metrics"""
        metrics = {
            'buffer_sizes': {k: len(v) for k, v in self.data_buffer.items()},
            'processing_rate': 0,
            'correlation_rate': 0,
            'anomaly_percentage': 0,
            'current_variance': 0,
            'prediction_stability': 0
        }
        
        # Processing rate (pixels per second)
        if hasattr(self, 'start_time'):
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                metrics['processing_rate'] = self.stats['pixels_processed'] / elapsed
                metrics['correlation_rate'] = self.stats['correlations_found'] / elapsed
        
        # Current variance
        if len(self.data_buffer['pixels']) > 10:
            metrics['current_variance'] = float(np.var(list(self.data_buffer['pixels'])))
        
        # Anomaly percentage
        if self.stats['pixels_processed'] > 0:
            metrics['anomaly_percentage'] = (
                self.stats['anomalies_detected'] / self.stats['pixels_processed'] * 100
            )
        
        # Prediction stability
        if len(self.data_buffer['predictions']) > 5:
            pred_std = np.std(list(self.data_buffer['predictions']))
            metrics['prediction_stability'] = 100 - min(100, pred_std)
        
        return metrics
    
    def display_dashboard(self):
        """Display real-time statistics dashboard"""
        while self.running:
            metrics = self.calculate_realtime_metrics()
            
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end='')
            
            print("=== REAL-TIME PROCESSING DASHBOARD ===")
            print(f"Uptime: {int(time.time() - self.start_time)}s")
            print("\nSTATISTICS:")
            print(f"  Pixels Processed: {self.stats['pixels_processed']:,}")
            print(f"  Correlations Found: {self.stats['correlations_found']}")
            print(f"  Anomalies Detected: {self.stats['anomalies_detected']}")
            print(f"  Triggers Fired: {self.stats['triggers_fired']}")
            
            print("\nREAL-TIME METRICS:")
            print(f"  Processing Rate: {metrics['processing_rate']:.1f} pixels/sec")
            print(f"  Correlation Rate: {metrics['correlation_rate']:.2f} /sec")
            print(f"  Anomaly Rate: {metrics['anomaly_percentage']:.2f}%")
            print(f"  Current Variance: {metrics['current_variance']:.1f}")
            print(f"  Prediction Stability: {metrics['prediction_stability']:.1f}%")
            
            print("\nBUFFER STATUS:")
            for buffer_name, size in metrics['buffer_sizes'].items():
                bar = '█' * min(20, size // 5)
                print(f"  {buffer_name:12} [{bar:<20}] {size}")
            
            print("\nPress Ctrl+C to stop...")
            
            # Save metrics
            with open('realtime_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            time.sleep(1)
    
    def start(self):
        """Start real-time processing"""
        self.start_time = time.time()
        print("[REALTIME] Starting real-time processor...")
        
        # Start monitoring threads
        monitors = [
            ('pixel_data.json', 'pixels'),
            ('correlations.json', 'correlations'),
            ('anomalies.json', 'anomalies'),
            ('vision_results.json', 'vision'),
            ('neural_results.json', 'neural')
        ]
        
        threads = []
        for filepath, data_type in monitors:
            thread = threading.Thread(
                target=self.monitor_file,
                args=(filepath, data_type)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            print(f"[REALTIME] Monitoring {filepath}")
        
        # Start random generator if not running
        if not os.path.exists('random_value.json'):
            print("[REALTIME] Starting random generator...")
            gen_thread = threading.Thread(
                target=lambda: subprocess.run(
                    [sys.executable, 'random_generator.py'],
                    capture_output=True
                )
            )
            gen_thread.daemon = True
            gen_thread.start()
        
        # Display dashboard
        try:
            self.display_dashboard()
        except KeyboardInterrupt:
            print("\n[REALTIME] Stopping...")
            self.running = False
        
        # Save final stats
        final_report = {
            'duration': time.time() - self.start_time,
            'stats': self.stats,
            'final_metrics': self.calculate_realtime_metrics()
        }
        
        with open('realtime_report.json', 'w') as f:
            json.dump(final_report, f)
        
        print(f"[REALTIME] Processed {self.stats['pixels_processed']:,} pixels")
        print(f"[REALTIME] Found {self.stats['correlations_found']} correlations")
        print(f"[REALTIME] Fired {self.stats['triggers_fired']} triggers")
        print("[REALTIME] Report saved to realtime_report.json")

if __name__ == "__main__":
    processor = RealtimeProcessor()
    processor.start()