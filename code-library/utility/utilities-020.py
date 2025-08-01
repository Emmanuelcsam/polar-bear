#!/usr/bin/env python3
"""
Enhanced script implementations with connector integration
Maintains original functionality while adding control capabilities
"""

import numpy as np
import random
import time
import json
import glob
import sys
import os
from typing import Any, Dict, List, Optional
from script_wrapper import ScriptWrapper

# Global control variables for connector access
CONTROL_VARS = {
    'anomaly_threshold': 50,
    'pixel_min': 0,
    'pixel_max': 255,
    'pixel_delay': 0.01,
    'events_file': 'events.log',
    'batch_enabled': True,
    'realtime_enabled': True
}

class DataStore(ScriptWrapper):
    """Enhanced data store with connector control"""
    
    def __init__(self):
        super().__init__('data-store')
        self.events_file = CONTROL_VARS['events_file']
        
    def initialize(self, **kwargs):
        self.events_file = kwargs.get('events_file', CONTROL_VARS['events_file'])
        self.parameters = {
            'events_file': self.events_file,
            'auto_backup': kwargs.get('auto_backup', False),
            'max_events': kwargs.get('max_events', 10000)
        }
        
    def save_event(self, e, f=None):
        """Save event with timestamp"""
        if f is None:
            f = self.events_file
        e.update(ts=time.time())
        with open(f, 'a') as file:
            file.write(json.dumps(e) + '\n')
        return e
        
    def load_events(self, f=None):
        """Load all events from file"""
        if f is None:
            f = self.events_file
        try:
            with open(f, 'r') as file:
                return [json.loads(line.strip()) for line in file if line.strip()]
        except FileNotFoundError:
            return []
            
    def process(self, data: Any) -> Any:
        """Process data based on action"""
        if isinstance(data, dict):
            action = data.get('action')
            if action == 'save':
                return self.save_event(data.get('event', {}))
            elif action == 'load':
                return self.load_events()
            elif action == 'clear':
                open(self.events_file, 'w').close()
                return {'status': 'cleared'}
        return data
        
    def get_stats(self):
        """Get statistics about stored events"""
        events = self.load_events()
        if not events:
            return {'count': 0}
        
        pixels = [e.get('pixel', e.get('intensity', 0)) for e in events]
        return {
            'count': len(events),
            'mean': np.mean(pixels) if pixels else 0,
            'std': np.std(pixels) if pixels else 0,
            'min': min(pixels) if pixels else 0,
            'max': max(pixels) if pixels else 0
        }

class AnomalyDetector(ScriptWrapper):
    """Enhanced anomaly detector with connector control"""
    
    def __init__(self):
        super().__init__('anomaly-detector')
        self.data_store = DataStore()
        self.data_store.initialize()
        
    def initialize(self, **kwargs):
        self.parameters = {
            'threshold': kwargs.get('threshold', CONTROL_VARS['anomaly_threshold']),
            'method': kwargs.get('method', 'deviation'),
            'window_size': kwargs.get('window_size', 100)
        }
        
    def anomalies(self, th=None):
        """Detect anomalies in the data"""
        if th is None:
            th = self.parameters['threshold']
            
        events = self.data_store.load_events()
        if not events:
            return []
            
        vals = [v.get('pixel', v.get('intensity', 0)) for v in events]
        mean_val = np.mean(vals)
        
        anomalies = []
        for v in events:
            value = v.get('pixel', v.get('intensity', 0))
            if abs(value - mean_val) > th:
                v['anomaly_score'] = abs(value - mean_val)
                anomalies.append(v)
                
        return anomalies
        
    def process(self, data: Any) -> Any:
        """Process data to find anomalies"""
        if isinstance(data, dict):
            threshold = data.get('threshold', self.parameters['threshold'])
            return self.anomalies(threshold)
        return self.anomalies()
        
    def main(self):
        """Standalone execution"""
        anomalies = self.anomalies()
        self.logger.info(f"Found {len(anomalies)} anomalies")
        return anomalies

class PixelGenerator(ScriptWrapper):
    """Enhanced pixel generator with connector control"""
    
    def __init__(self):
        super().__init__('pixel-generator')
        self.data_store = DataStore()
        self.data_store.initialize()
        self.running = False
        
    def initialize(self, **kwargs):
        self.parameters = {
            'min_value': kwargs.get('min_value', CONTROL_VARS['pixel_min']),
            'max_value': kwargs.get('max_value', CONTROL_VARS['pixel_max']),
            'delay': kwargs.get('delay', CONTROL_VARS['pixel_delay']),
            'batch_size': kwargs.get('batch_size', 1)
        }
        
    def generate_pixel(self):
        """Generate a single pixel value"""
        return {
            'pixel': random.randint(
                self.parameters['min_value'],
                self.parameters['max_value']
            ),
            'generator': self.script_name
        }
        
    def run(self, duration=None):
        """Run pixel generation"""
        self.running = True
        start_time = time.time()
        
        while self.running:
            if duration and (time.time() - start_time) > duration:
                break
                
            pixel_data = self.generate_pixel()
            self.data_store.save_event(pixel_data)
            time.sleep(self.parameters['delay'])
            
    def stop(self):
        """Stop pixel generation"""
        self.running = False
        
    def process(self, data: Any) -> Any:
        """Process commands"""
        if isinstance(data, dict):
            command = data.get('command')
            if command == 'start':
                duration = data.get('duration')
                self.run(duration)
                return {'status': 'started'}
            elif command == 'stop':
                self.stop()
                return {'status': 'stopped'}
            elif command == 'generate':
                count = data.get('count', 1)
                pixels = []
                for _ in range(count):
                    pixel = self.generate_pixel()
                    self.data_store.save_event(pixel)
                    pixels.append(pixel)
                return {'pixels': pixels}
        return data
        
    def main(self):
        """Standalone execution"""
        self.logger.info("Starting pixel generation...")
        try:
            self.run()
        except KeyboardInterrupt:
            self.logger.info("Pixel generation stopped")

class BatchProcessor(ScriptWrapper):
    """Enhanced batch processor with connector control"""
    
    def __init__(self):
        super().__init__('batch-processor')
        
    def initialize(self, **kwargs):
        self.parameters = {
            'enabled': kwargs.get('enabled', CONTROL_VARS['batch_enabled']),
            'batch_size': kwargs.get('batch_size', 10),
            'process_subdirs': kwargs.get('process_subdirs', True)
        }
        
    def run_folder(self, directory):
        """Process all files in a folder"""
        pattern = os.path.join(directory, '*')
        files = glob.glob(pattern)
        
        results = []
        for filepath in files:
            try:
                # Process each file
                result = self.process_file(filepath)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {filepath}: {e}")
                
        self.logger.info(f"Batch processing completed: {len(results)} files")
        return results
        
    def process_file(self, filepath):
        """Process a single file"""
        # Example processing
        return {
            'file': filepath,
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            'processed': True
        }
        
    def process(self, data: Any) -> Any:
        """Process batch commands"""
        if isinstance(data, dict):
            directory = data.get('directory', '.')
            return self.run_folder(directory)
        elif isinstance(data, str):
            return self.run_folder(data)
        return data
        
    def main(self):
        """Standalone execution"""
        if len(sys.argv) > 1:
            directory = sys.argv[1]
        else:
            directory = '.'
        return self.run_folder(directory)

# Factory functions for creating enhanced scripts
def create_data_store():
    return DataStore()

def create_anomaly_detector():
    return AnomalyDetector()

def create_pixel_generator():
    return PixelGenerator()

def create_batch_processor():
    return BatchProcessor()

# Make scripts accessible for import
data_store = create_data_store()
anomaly_detector = create_anomaly_detector()
pixel_generator = create_pixel_generator()
batch_processor = create_batch_processor()

# Export functions for backward compatibility
save_event = data_store.save_event
load_events = data_store.load_events
anomalies = anomaly_detector.anomalies