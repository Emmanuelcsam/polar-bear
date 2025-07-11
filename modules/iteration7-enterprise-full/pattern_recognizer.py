import json
import numpy as np
from collections import Counter

def recognize_patterns():
    patterns = []
    
    # Load pixel data
    try:
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = data['pixels']
        
        # Frequency patterns
        freq_counter = Counter(pixels)
        most_common = freq_counter.most_common(10)
        print(f"[PATTERN_REC] Most common values: {most_common}")
        
        # Sequential patterns
        sequences = []
        for i in range(len(pixels) - 3):
            seq = pixels[i:i+4]
            if len(set(seq)) == 1:  # All same
                sequences.append(('repeat', seq[0], i))
            elif seq == sorted(seq):  # Ascending
                sequences.append(('ascending', seq, i))
            elif seq == sorted(seq, reverse=True):  # Descending
                sequences.append(('descending', seq, i))
        
        print(f"[PATTERN_REC] Found {len(sequences)} sequential patterns")
        
        # Statistical patterns
        pixel_array = np.array(pixels)
        if len(pixels) > 0:
            patterns.append({
                'mean': float(np.mean(pixel_array)),
                'std': float(np.std(pixel_array)),
                'min': int(np.min(pixel_array)),
                'max': int(np.max(pixel_array))
            })
        else:
            patterns.append({
                'mean': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0
            })
        
        # Save patterns
        with open('patterns.json', 'w') as f:
            json.dump({
                'frequency': dict(freq_counter),
                'sequences': sequences[:100],  # Limit to first 100
                'statistics': patterns
            }, f)
        
        print(f"[PATTERN_REC] Pattern analysis complete")
        
    except Exception as e:
        print(f"[PATTERN_REC] Error: {e}")

if __name__ == "__main__":
    recognize_patterns()