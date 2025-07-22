import json
import time
import os

def correlate():
    correlations = []
    pixel_data = {}
    last_random = None
    
    # Load existing correlations if any
    if os.path.exists('correlations.json'):
        with open('correlations.json', 'r') as f:
            correlations = json.load(f)
    
    while True:
        try:
            # Read pixel data once if not loaded
            if not pixel_data and os.path.exists('pixel_data.json'):
                with open('pixel_data.json', 'r') as f:
                    pixel_data = json.load(f)
                    print(f"[CORRELATOR] Loaded {len(pixel_data.get('pixels', []))} pixels")
                    
            # Read random value
            if os.path.exists('random_value.json'):
                with open('random_value.json', 'r') as f:
                    random_data = json.load(f)
                
                # Only process if it's a new value
                if random_data != last_random:
                    last_random = random_data
                    
                    # Check for correlations
                    if 'pixels' in pixel_data:
                        matches = []
                        for i, pixel in enumerate(pixel_data['pixels']):
                            if pixel == random_data['value']:
                                matches.append(i)
                        
                        if matches:
                            correlation = {
                                'value': random_data['value'],
                                'pixel_indices': matches[:10],  # Limit to first 10
                                'match_count': len(matches),
                                'timestamp': time.time()
                            }
                            
                            # Store individual correlations for detailed tracking
                            for idx in matches[:10]:
                                correlations.append({
                                    'pixel_index': idx,
                                    'value': random_data['value'],
                                    'timestamp': time.time()
                                })
                            
                            print(f"[CORRELATOR] Match! Value: {random_data['value']}, Found in {len(matches)} positions")
                            
                            # Save correlations
                            with open('correlations.json', 'w') as f:
                                json.dump(correlations[-1000:], f)  # Keep last 1000
                                
        except Exception as e:
            pass
        
        time.sleep(0.05)

if __name__ == "__main__":
    correlate()