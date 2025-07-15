import pickle
import json
import os
from datetime import datetime
import numpy as np
from connector_interface import setup_connector, send_hivemind_status

print("Stats Viewer loading...")

# Setup hivemind connector
connector = setup_connector('stats-viewer.py')

# Load all available data
stats = {
    'categories': 0,
    'total_pixels': 0,
    'processed_images': 0,
    'learning_iterations': 0,
    'category_distribution': {},
    'confidence_history': []
}

if os.path.exists('pixel_db.pkl'):
    with open('pixel_db.pkl', 'rb') as f:
        pixel_db = pickle.load(f)
    stats['categories'] = len(pixel_db)
    stats['total_pixels'] = sum(len(pixels) for pixels in pixel_db.values())
    print(f"✓ Loaded {stats['categories']} categories with {stats['total_pixels']} pixels")

if os.path.exists('learning_history.pkl'):
    with open('learning_history.pkl', 'rb') as f:
        history = pickle.load(f)
    stats['learning_iterations'] = len(history)
    print(f"✓ Found {stats['learning_iterations']} learning iterations")

# Analyze all result files
result_files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.json')]
print(f"✓ Found {len(result_files)} result files")

for rf in result_files:
    with open(rf, 'r') as f:
        results = json.load(f)
    
    stats['processed_images'] += len(results)
    
    for img, data in results.items():
        cat = data['category']
        conf = data['confidence']
        
        if cat not in stats['category_distribution']:
            stats['category_distribution'][cat] = 0
        stats['category_distribution'][cat] += 1
        
        stats['confidence_history'].append(conf)

print("\n" + "="*60)
print("SYSTEM STATISTICS")
print("="*60)

# Send statistics to hivemind
send_hivemind_status({
    'status': 'statistics_generated',
    'stats': {
        'categories': stats['categories'],
        'total_pixels': stats['total_pixels'],
        'processed_images': stats['processed_images'],
        'learning_iterations': stats['learning_iterations'],
        'result_files': len(result_files)
    }
}, connector)

print(f"\nDatabase Stats:")
print(f"  Categories: {stats['categories']}")
print(f"  Total reference pixels: {stats['total_pixels']:,}")
print(f"  Avg pixels per category: {stats['total_pixels']/max(1,stats['categories']):.0f}")

print(f"\nProcessing Stats:")
print(f"  Total images processed: {stats['processed_images']}")
print(f"  Learning iterations: {stats['learning_iterations']}")
print(f"  Result files: {len(result_files)}")

if stats['confidence_history']:
    print(f"\nConfidence Stats:")
    print(f"  Average: {np.mean(stats['confidence_history']):.3f}")
    print(f"  Std Dev: {np.std(stats['confidence_history']):.3f}")
    print(f"  Min: {np.min(stats['confidence_history']):.3f}")
    print(f"  Max: {np.max(stats['confidence_history']):.3f}")

if stats['category_distribution']:
    print(f"\nTop 10 Categories:")
    sorted_cats = sorted(stats['category_distribution'].items(), 
                        key=lambda x: x[1], reverse=True)[:10]
    for i, (cat, count) in enumerate(sorted_cats):
        pct = count / stats['processed_images'] * 100
        print(f"  {i+1}. {cat}: {count} images ({pct:.1f}%)")

# Performance over time
if os.path.exists('live_log.txt') and os.path.getsize('live_log.txt') > 0:
    print(f"\nLive Processing Stats:")
    with open('live_log.txt', 'r') as f:
        lines = f.readlines()
    
    if lines:
        first_time = datetime.fromisoformat(lines[0].split(',')[0])
        last_time = datetime.fromisoformat(lines[-1].split(',')[0])
        duration = (last_time - first_time).total_seconds()
        
        if duration > 0:
            rate = len(lines) / duration * 60  # images per minute
            print(f"  Processing rate: {rate:.1f} images/minute")
            print(f"  Total live processed: {len(lines)}")

print("\n" + "="*60)

# Send completion status
send_hivemind_status({'status': 'stats_displayed'}, connector)