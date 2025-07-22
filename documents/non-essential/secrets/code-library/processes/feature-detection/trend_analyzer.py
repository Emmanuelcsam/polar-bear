import json
import numpy as np
from collections import defaultdict
import os

def analyze_trends():
    trends = {
        'pixel_trends': {},
        'pattern_evolution': [],
        'anomaly_frequency': {},
        'image_similarities': []
    }
    
    try:
        # Analyze pixel frequency trends
        if os.path.exists('learned_data.json'):
            with open('learned_data.json', 'r') as f:
                learned = json.load(f)
                frequencies = learned.get('pixel_frequencies', {})
                
                if frequencies:
                    # Find most/least common values
                    sorted_freq = sorted(frequencies.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
                    
                    trends['pixel_trends'] = {
                        'most_common': sorted_freq[:10],
                        'least_common': sorted_freq[-10:],
                        'total_unique': len(frequencies),
                        'distribution_skew': calculate_skew(frequencies)
                    }
                    
                    print(f"[TRENDS] Top pixel values: {sorted_freq[:5]}")
        
        # Analyze pattern evolution
        if os.path.exists('patterns.json'):
            with open('patterns.json', 'r') as f:
                patterns = json.load(f)
                
                if 'statistics' in patterns:
                    for stat in patterns['statistics']:
                        trends['pattern_evolution'].append({
                            'mean_trend': stat['mean'],
                            'variance_trend': stat['std'] ** 2
                        })
        
        # Analyze anomaly trends
        if os.path.exists('anomalies.json'):
            with open('anomalies.json', 'r') as f:
                anomalies = json.load(f)
                
                anomaly_values = defaultdict(int)
                for anom in anomalies.get('z_score_anomalies', []):
                    anomaly_values[anom['value']] += 1
                
                trends['anomaly_frequency'] = dict(anomaly_values)
        
        # Analyze image similarities
        profiles = []
        for file in os.listdir('.'):
            if file.startswith('batch_') and file.endswith('.json'):
                with open(file, 'r') as f:
                    batch_data = json.load(f)
                    if 'stats' in batch_data:
                        profiles.append(batch_data['stats'])
        
        if len(profiles) > 1:
            # Find similar images based on stats
            for i, p1 in enumerate(profiles):
                for j, p2 in enumerate(profiles[i+1:], i+1):
                    similarity = calculate_similarity(p1, p2)
                    if similarity > 0.8:
                        trends['image_similarities'].append({
                            'image1': p1['filename'],
                            'image2': p2['filename'],
                            'similarity': similarity
                        })
        
        # Save trends
        with open('trends.json', 'w') as f:
            json.dump(trends, f)
        
        print(f"[TRENDS] Analysis complete")
        print(f"[TRENDS] Found {len(trends['image_similarities'])} similar images")
        
    except Exception as e:
        print(f"[TRENDS] Error: {e}")

def calculate_skew(frequencies):
    values = []
    for val, count in frequencies.items():
        values.extend([int(val)] * count)
    
    if values:
        return float(np.mean(values) - np.median(values))
    return 0

def calculate_similarity(stats1, stats2):
    # Simple similarity based on statistics
    mean_diff = abs(stats1['mean'] - stats2['mean']) / 255
    std_diff = abs(stats1['std'] - stats2['std']) / 128
    
    similarity = 1 - (mean_diff + std_diff) / 2
    return max(0, min(1, similarity))

if __name__ == "__main__":
    analyze_trends()