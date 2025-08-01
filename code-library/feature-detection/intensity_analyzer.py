import json
import numpy as np
import os

def analyze_intensity():
    try:
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
            size = data['size']
        
        # Intensity histogram
        hist, bins = np.histogram(pixels, bins=256, range=(0, 256))
        
        # Find intensity clusters
        clusters = []
        threshold = len(pixels) * 0.01  # 1% threshold
        
        in_cluster = False
        cluster_start = 0
        
        for i, count in enumerate(hist):
            if count > threshold and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif count <= threshold and in_cluster:
                in_cluster = False
                clusters.append({
                    'start': int(cluster_start),
                    'end': int(i-1),
                    'peak': int(cluster_start + np.argmax(hist[cluster_start:i])),
                    'peak_count': int(np.max(hist[cluster_start:i]))
                })
        
        print(f"[INTENSITY] Found {len(clusters)} intensity clusters")
        
        # Analyze intensity gradients
        img_2d = pixels.reshape(size[1], size[0])
        
        # Local intensity variations
        local_vars = []
        window_size = 10
        
        for i in range(0, size[1] - window_size, window_size):
            for j in range(0, size[0] - window_size, window_size):
                window = img_2d[i:i+window_size, j:j+window_size]
                local_vars.append({
                    'position': (i, j),
                    'variance': float(np.var(window)),
                    'mean': float(np.mean(window))
                })
        
        # Sort by variance to find most/least varying regions
        local_vars.sort(key=lambda x: x['variance'], reverse=True)
        
        # Intensity transitions
        transitions = []
        for i in range(len(pixels) - 1):
            diff = abs(int(pixels[i+1]) - int(pixels[i]))
            if diff > 100:  # Large transition
                transitions.append({
                    'index': i,
                    'from': int(pixels[i]),
                    'to': int(pixels[i+1]),
                    'difference': diff
                })
        
        intensity_analysis = {
            'histogram': {
                'bins': [int(b) for b in bins[:-1]],
                'counts': [int(c) for c in hist]
            },
            'clusters': clusters,
            'high_variance_regions': local_vars[:10],
            'low_variance_regions': local_vars[-10:],
            'sharp_transitions': transitions[:20],
            'statistics': {
                'mean_intensity': float(np.mean(pixels)),
                'median_intensity': float(np.median(pixels)),
                'intensity_range': {
                    'min': int(np.min(pixels)),
                    'max': int(np.max(pixels))
                },
                'quartiles': {
                    'q1': float(np.percentile(pixels, 25)),
                    'q2': float(np.percentile(pixels, 50)),
                    'q3': float(np.percentile(pixels, 75))
                }
            }
        }
        
        with open('intensity_analysis.json', 'w') as f:
            json.dump(intensity_analysis, f)
        
        print(f"[INTENSITY] Analysis complete")
        print(f"[INTENSITY] Mean: {intensity_analysis['statistics']['mean_intensity']:.2f}")
        
    except Exception as e:
        print(f"[INTENSITY] Error: {e}")

if __name__ == "__main__":
    analyze_intensity()