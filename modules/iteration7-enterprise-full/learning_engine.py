import json
import time
import os
from data_store import DataStore

class LearningEngine:
    def __init__(self):
        self.store = DataStore()
        self.mode = 'auto'  # 'auto' or 'manual'
        
    def auto_learn(self):
        print("[LEARNING] Auto-learning mode active")
        
        while self.mode == 'auto':
            try:
                # Learn from correlations
                if os.path.exists('correlations.json'):
                    with open('correlations.json', 'r') as f:
                        correlations = json.load(f)
                        for corr in correlations[-10:]:  # Last 10
                            self.store.update_frequency(corr['value'])
                
                # Learn from patterns
                if os.path.exists('patterns.json'):
                    with open('patterns.json', 'r') as f:
                        patterns = json.load(f)
                        self.store.data['patterns'] = patterns
                
                # Learn from anomalies
                if os.path.exists('anomalies.json'):
                    with open('anomalies.json', 'r') as f:
                        anomalies = json.load(f)
                        for anom in anomalies.get('z_score_anomalies', [])[:5]:
                            self.store.add_anomaly(anom)
                
                self.store.save()
                time.sleep(1)
                
            except Exception as e:
                pass
    
    def manual_learn(self, image_path, label=None):
        print(f"[LEARNING] Manual learning from {image_path}")
        
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(image_path).convert('L')
            pixels = np.array(img.getdata())
            
            # Learn pixel distribution
            unique, counts = np.unique(pixels, return_counts=True)
            for val, count in zip(unique, counts):
                for _ in range(min(count, 100)):  # Limit learning rate
                    self.store.update_frequency(int(val))
            
            # Store image profile
            profile = {
                'path': image_path,
                'label': label,
                'mean': float(np.mean(pixels)),
                'std': float(np.std(pixels)),
                'dominant_values': [int(v) for v in unique[np.argsort(counts)[-5:]]]
            }
            
            self.store.data['image_profiles'][image_path] = profile
            self.store.save()
            
            print(f"[LEARNING] Learned from {image_path}")
            
        except Exception as e:
            print(f"[LEARNING] Error: {e}")
    
    def set_mode(self, mode):
        self.mode = mode
        print(f"[LEARNING] Mode set to {mode}")

if __name__ == "__main__":
    engine = LearningEngine()
    
    # Example: manual learning
    if os.path.exists('test.jpg'):
        engine.manual_learn('test.jpg', 'test_image')
    
    # Auto learning
    engine.auto_learn()