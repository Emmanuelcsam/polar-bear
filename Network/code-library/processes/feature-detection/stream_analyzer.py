import json
import time
import os
import numpy as np
from collections import deque, defaultdict
import threading

class StreamAnalyzer:
    def __init__(self):
        self.streams = {
            'pixel_stream': deque(maxlen=10000),
            'correlation_stream': deque(maxlen=1000),
            'anomaly_stream': deque(maxlen=500),
            'prediction_stream': deque(maxlen=500)
        }
        
        self.stream_stats = defaultdict(lambda: {
            'count': 0,
            'rate': 0,
            'last_update': 0,
            'moving_avg': deque(maxlen=100),
            'alerts': []
        })
        
        self.patterns = {
            'bursts': [],
            'trends': [],
            'cycles': [],
            'anomalous_periods': []
        }
        
        self.running = True
        self.analysis_interval = 1.0  # Analyze every second
        
    def add_to_stream(self, stream_name, data, timestamp=None):
        """Add data point to a stream"""
        if timestamp is None:
            timestamp = time.time()
        
        if stream_name in self.streams:
            self.streams[stream_name].append({
                'value': data,
                'timestamp': timestamp
            })
            
            # Update statistics
            stats = self.stream_stats[stream_name]
            stats['count'] += 1
            stats['last_update'] = timestamp
            
            # Calculate moving average
            if isinstance(data, (int, float)):
                stats['moving_avg'].append(data)
    
    def analyze_stream_patterns(self, stream_name):
        """Analyze patterns in a specific stream"""
        if stream_name not in self.streams:
            return
        
        stream = list(self.streams[stream_name])
        if len(stream) < 10:
            return
        
        # Extract values and timestamps
        values = [item['value'] for item in stream if isinstance(item['value'], (int, float))]
        timestamps = [item['timestamp'] for item in stream]
        
        if not values:
            return
        
        patterns = {}
        
        # 1. Burst Detection
        time_diffs = np.diff(timestamps)
        if len(time_diffs) > 0:
            mean_interval = np.mean(time_diffs)
            burst_threshold = mean_interval * 0.1  # 10x faster than average
            
            bursts = []
            burst_start = None
            
            for i, diff in enumerate(time_diffs):
                if diff < burst_threshold:
                    if burst_start is None:
                        burst_start = i
                else:
                    if burst_start is not None:
                        burst_end = i
                        burst_size = burst_end - burst_start + 1
                        if burst_size > 5:  # Significant burst
                            bursts.append({
                                'start_idx': burst_start,
                                'end_idx': burst_end,
                                'size': burst_size,
                                'timestamp': timestamps[burst_start]
                            })
                        burst_start = None
            
            if bursts:
                patterns['bursts'] = bursts
                print(f"[STREAM] Detected {len(bursts)} bursts in {stream_name}")
        
        # 2. Trend Detection
        if len(values) > 20:
            # Simple linear regression for trend
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Determine trend strength
            y_pred = np.polyval(coeffs, x)
            r_squared = 1 - (np.sum((values - y_pred)**2) / np.sum((values - np.mean(values))**2))
            
            if abs(r_squared) > 0.3:  # Significant trend
                trend_type = 'increasing' if slope > 0 else 'decreasing'
                patterns['trend'] = {
                    'type': trend_type,
                    'slope': float(slope),
                    'strength': float(abs(r_squared)),
                    'start_value': float(values[0]),
                    'end_value': float(values[-1])
                }
                print(f"[STREAM] {stream_name} shows {trend_type} trend (strength: {abs(r_squared):.2f})")
        
        # 3. Cycle Detection (using autocorrelation)
        if len(values) > 50:
            # Compute autocorrelation
            mean_val = np.mean(values)
            autocorr = []
            
            for lag in range(1, min(len(values)//2, 50)):
                c0 = np.sum((values[:-lag] - mean_val) * (values[lag:] - mean_val))
                c0 /= len(values) - lag
                autocorr.append(c0)
            
            # Find peaks in autocorrelation
            if len(autocorr) > 10:
                autocorr_array = np.array(autocorr)
                
                # Find local maxima
                peaks = []
                for i in range(1, len(autocorr_array) - 1):
                    if autocorr_array[i] > autocorr_array[i-1] and autocorr_array[i] > autocorr_array[i+1]:
                        if autocorr_array[i] > np.std(autocorr_array):  # Significant peak
                            peaks.append({
                                'lag': i + 1,
                                'strength': float(autocorr_array[i])
                            })
                
                if peaks:
                    patterns['cycles'] = peaks[:3]  # Top 3 cycles
                    print(f"[STREAM] Found cyclic pattern in {stream_name} with period {peaks[0]['lag']}")
        
        # 4. Anomaly Periods
        if len(values) > 30:
            # Use rolling statistics to find anomalous periods
            window_size = min(20, len(values) // 3)
            rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            rolling_std = []
            
            for i in range(len(values) - window_size + 1):
                window_std = np.std(values[i:i+window_size])
                rolling_std.append(window_std)
            
            rolling_std = np.array(rolling_std)
            
            # Find periods with high variance
            high_var_threshold = np.mean(rolling_std) + 2 * np.std(rolling_std)
            anomalous_periods = []
            
            in_anomaly = False
            anomaly_start = None
            
            for i, std_val in enumerate(rolling_std):
                if std_val > high_var_threshold:
                    if not in_anomaly:
                        in_anomaly = True
                        anomaly_start = i
                else:
                    if in_anomaly:
                        anomalous_periods.append({
                            'start_idx': anomaly_start,
                            'end_idx': i,
                            'duration': i - anomaly_start,
                            'max_variance': float(np.max(rolling_std[anomaly_start:i]))
                        })
                        in_anomaly = False
            
            if anomalous_periods:
                patterns['anomalous_periods'] = anomalous_periods
                print(f"[STREAM] Found {len(anomalous_periods)} anomalous periods in {stream_name}")
        
        return patterns
    
    def cross_stream_analysis(self):
        """Analyze relationships between different streams"""
        correlations = {}
        
        # Get recent data from each stream
        stream_data = {}
        for stream_name, stream in self.streams.items():
            if len(stream) > 10:
                recent = list(stream)[-100:]  # Last 100 items
                values = [item['value'] for item in recent if isinstance(item['value'], (int, float))]
                if values:
                    stream_data[stream_name] = values
        
        # Calculate correlations between streams
        stream_names = list(stream_data.keys())
        for i in range(len(stream_names)):
            for j in range(i + 1, len(stream_names)):
                stream1 = stream_data[stream_names[i]]
                stream2 = stream_data[stream_names[j]]
                
                # Make streams same length
                min_len = min(len(stream1), len(stream2))
                if min_len > 10:
                    s1 = stream1[-min_len:]
                    s2 = stream2[-min_len:]
                    
                    # Calculate correlation
                    corr = np.corrcoef(s1, s2)[0, 1]
                    
                    if abs(corr) > 0.5:  # Significant correlation
                        correlations[f"{stream_names[i]}_vs_{stream_names[j]}"] = {
                            'correlation': float(corr),
                            'strength': 'strong' if abs(corr) > 0.7 else 'moderate',
                            'direction': 'positive' if corr > 0 else 'negative'
                        }
                        
                        print(f"[STREAM] {stream_names[i]} and {stream_names[j]} show "
                              f"{correlations[f'{stream_names[i]}_vs_{stream_names[j]}']['strength']} "
                              f"{correlations[f'{stream_names[i]}_vs_{stream_names[j]}']['direction']} correlation")
        
        return correlations
    
    def detect_alerts(self):
        """Detect conditions that warrant alerts"""
        alerts = []
        current_time = time.time()
        
        for stream_name, stats in self.stream_stats.items():
            # Check for stream stalls
            if stats['last_update'] > 0:
                time_since_update = current_time - stats['last_update']
                if time_since_update > 5.0:  # No update for 5 seconds
                    alerts.append({
                        'type': 'stream_stall',
                        'stream': stream_name,
                        'duration': time_since_update,
                        'severity': 'warning'
                    })
            
            # Check for extreme values
            if len(stats['moving_avg']) > 10:
                recent_values = list(stats['moving_avg'])
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                
                if std_val > 0:
                    last_val = recent_values[-1]
                    z_score = abs((last_val - mean_val) / std_val)
                    
                    if z_score > 3:  # Extreme value
                        alerts.append({
                            'type': 'extreme_value',
                            'stream': stream_name,
                            'value': last_val,
                            'z_score': z_score,
                            'severity': 'high'
                        })
        
        # Save alerts
        if alerts:
            with open('stream_alerts.json', 'w') as f:
                json.dump({
                    'timestamp': current_time,
                    'alerts': alerts
                }, f)
            
            for alert in alerts:
                print(f"[STREAM] ALERT: {alert['type']} in {alert['stream']} "
                      f"(severity: {alert['severity']})")
        
        return alerts
    
    def continuous_analysis(self):
        """Run continuous analysis on all streams"""
        while self.running:
            try:
                # Analyze each stream
                all_patterns = {}
                
                for stream_name in self.streams:
                    patterns = self.analyze_stream_patterns(stream_name)
                    if patterns:
                        all_patterns[stream_name] = patterns
                
                # Cross-stream analysis
                correlations = self.cross_stream_analysis()
                
                # Detect alerts
                alerts = self.detect_alerts()
                
                # Calculate stream rates
                for stream_name, stats in self.stream_stats.items():
                    if stats['count'] > 0:
                        elapsed = time.time() - self.analysis_start_time
                        stats['rate'] = stats['count'] / elapsed if elapsed > 0 else 0
                
                # Save analysis results
                analysis_results = {
                    'timestamp': time.time(),
                    'patterns': all_patterns,
                    'correlations': correlations,
                    'stream_rates': {
                        name: stats['rate'] 
                        for name, stats in self.stream_stats.items()
                    },
                    'alerts': alerts
                }
                
                with open('stream_analysis.json', 'w') as f:
                    json.dump(analysis_results, f)
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                print(f"[STREAM] Analysis error: {e}")
                time.sleep(self.analysis_interval)
    
    def monitor_files(self):
        """Monitor JSON files and add data to streams"""
        file_monitors = {
            'pixel_data.json': ('pixel_stream', lambda d: d.get('pixels', [])[0] if d.get('pixels') else None),
            'correlations.json': ('correlation_stream', lambda d: len(d) if isinstance(d, list) else None),
            'anomalies.json': ('anomaly_stream', lambda d: len(d.get('z_score_anomalies', []))),
            'neural_results.json': ('prediction_stream', lambda d: np.mean(d.get('predictions', [])) if d.get('predictions') else None)
        }
        
        last_modified = {}
        
        while self.running:
            for filename, (stream_name, extractor) in file_monitors.items():
                try:
                    if os.path.exists(filename):
                        current_mod = os.path.getmtime(filename)
                        
                        if filename not in last_modified or current_mod > last_modified[filename]:
                            with open(filename, 'r') as f:
                                data = json.load(f)
                            
                            value = extractor(data)
                            if value is not None:
                                self.add_to_stream(stream_name, value)
                            
                            last_modified[filename] = current_mod
                            
                except Exception as e:
                    pass
            
            time.sleep(0.1)
    
    def start(self):
        """Start stream analyzer"""
        print("[STREAM] Starting stream analyzer...")
        self.analysis_start_time = time.time()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_files)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.continuous_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        print("[STREAM] Monitoring active streams...")
        print("[STREAM] Press Ctrl+C to stop")
        
        # Display real-time statistics
        try:
            while self.running:
                # Clear screen
                print("\033[2J\033[H", end='')
                
                print("=== STREAM ANALYZER ===")
                print(f"Uptime: {int(time.time() - self.analysis_start_time)}s\n")
                
                print("STREAM STATISTICS:")
                for stream_name, stats in self.stream_stats.items():
                    if stats['count'] > 0:
                        avg_val = np.mean(list(stats['moving_avg'])) if stats['moving_avg'] else 0
                        print(f"  {stream_name:20} Count: {stats['count']:6} "
                              f"Rate: {stats['rate']:6.2f}/s "
                              f"Avg: {avg_val:8.2f}")
                
                print("\nSTREAM BUFFERS:")
                for stream_name, stream in self.streams.items():
                    bar = '█' * min(20, len(stream) // 50)
                    print(f"  {stream_name:20} [{bar:<20}] {len(stream)}")
                
                # Show recent patterns
                if os.path.exists('stream_analysis.json'):
                    with open('stream_analysis.json', 'r') as f:
                        analysis = json.load(f)
                    
                    if analysis.get('patterns'):
                        print("\nRECENT PATTERNS:")
                        for stream, patterns in list(analysis['patterns'].items())[:3]:
                            pattern_types = list(patterns.keys())
                            print(f"  {stream}: {', '.join(pattern_types)}")
                    
                    if analysis.get('correlations'):
                        print("\nSTREAM CORRELATIONS:")
                        for pair, corr in list(analysis['correlations'].items())[:3]:
                            print(f"  {pair}: {corr['strength']} {corr['direction']}")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[STREAM] Stopping analyzer...")
            self.running = False
        
        # Save final report
        self.save_final_report()
    
    def save_final_report(self):
        """Save comprehensive analysis report"""
        
        report = {
            'duration': time.time() - self.analysis_start_time,
            'total_events': sum(stats['count'] for stats in self.stream_stats.values()),
            'stream_summaries': {},
            'detected_patterns': self.patterns
        }
        
        for stream_name, stats in self.stream_stats.items():
            if stats['count'] > 0:
                values = list(stats['moving_avg'])
                if values:
                    report['stream_summaries'][stream_name] = {
                        'total_count': stats['count'],
                        'average_rate': stats['rate'],
                        'mean_value': float(np.mean(values)),
                        'std_value': float(np.std(values)),
                        'min_value': float(np.min(values)),
                        'max_value': float(np.max(values))
                    }
        
        with open('stream_report.json', 'w') as f:
            json.dump(report, f)
        
        print(f"[STREAM] Processed {report['total_events']} total events")
        print("[STREAM] Report saved to stream_report.json")

if __name__ == "__main__":
    analyzer = StreamAnalyzer()
    analyzer.start()