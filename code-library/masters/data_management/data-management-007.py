import json
import os
from collections import defaultdict

class DataStore:
    def __init__(self):
        self.data_file = 'learned_data.json'
        self.load()
    
    def load(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                loaded_data = json.load(f)
                self.data = {
                    'pixel_frequencies': defaultdict(int, loaded_data.get('pixel_frequencies', {})),
                    'patterns': loaded_data.get('patterns', []),
                    'anomalies': loaded_data.get('anomalies', []),
                    'image_profiles': loaded_data.get('image_profiles', {})
                }
        else:
            self.data = {
                'pixel_frequencies': defaultdict(int),
                'patterns': [],
                'anomalies': [],
                'image_profiles': {}
            }
    
    def save(self):
        # Convert defaultdict to dict for JSON
        save_data = self.data.copy()
        save_data['pixel_frequencies'] = dict(self.data['pixel_frequencies'])
        
        with open(self.data_file, 'w') as f:
            json.dump(save_data, f)
        print(f"[DATA_STORE] Saved data")
    
    def update_frequency(self, value):
        self.data['pixel_frequencies'][str(value)] += 1
        print(f"[DATA_STORE] Updated frequency for {value}")
        self.save()
    
    def add_pattern(self, pattern):
        self.data['patterns'].append(pattern)
        # Keep only last 100 patterns to prevent memory issues
        if len(self.data['patterns']) > 100:
            self.data['patterns'] = self.data['patterns'][-100:]
        print(f"[DATA_STORE] Added pattern")
        self.save()
    
    def add_anomaly(self, anomaly):
        self.data['anomalies'].append(anomaly)
        # Keep only last 100 anomalies to prevent memory issues
        if len(self.data['anomalies']) > 100:
            self.data['anomalies'] = self.data['anomalies'][-100:]
        print(f"[DATA_STORE] Added anomaly")
        self.save()

if __name__ == "__main__":
    store = DataStore()
    # Example: update based on correlations
    if os.path.exists('correlations.json'):
        with open('correlations.json', 'r') as f:
            correlations = json.load(f)
            for corr in correlations:
                store.update_frequency(corr['value'])