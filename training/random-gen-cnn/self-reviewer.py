import json
import numpy as np
from PIL import Image
import os
import pickle

print("Self-Reviewer starting...")
results_file = input("Results JSON file to review: ")

with open(results_file, 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} categorized images")

# Group by category
categorized = {}
for img, data in results.items():
    cat = data['category']
    if cat not in categorized:
        categorized[cat] = []
    categorized[cat].append((img, data))

print("\nReviewing consistency within categories...")

inconsistencies = []
for category, images in categorized.items():
    if len(images) < 2:
        continue
        
    print(f"\nChecking {category} ({len(images)} images)...")
    
    # Compare each image to others in same category
    for i, (img1, data1) in enumerate(images):
        avg_similarity = 0
        for j, (img2, data2) in enumerate(images):
            if i == j:
                continue
            
            # Compare confidence levels
            conf_diff = abs(data1['confidence'] - data2['confidence'])
            if conf_diff > 0.3:
                print(f"  ⚠ Large confidence difference: {img1} ({data1['confidence']:.2f}) vs {img2} ({data2['confidence']:.2f})")
                inconsistencies.append((img1, img2, conf_diff))
    
    # Check for outliers based on scores
    confidences = [d['confidence'] for _, d in images]
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    for img, data in images:
        if abs(data['confidence'] - mean_conf) > 2 * std_conf:
            print(f"  ⚠ Outlier detected: {img} (confidence: {data['confidence']:.2f}, mean: {mean_conf:.2f})")
            inconsistencies.append((img, 'outlier', data['confidence']))

print(f"\n{'='*50}")
print(f"Review complete. Found {len(inconsistencies)} potential issues.")

if inconsistencies:
    fix = input("\nAttempt to re-categorize suspicious images? (y/n): ")
    if fix == 'y':
        import sys
        sys.path.append('.')
        import correlation_analyzer as ca
        
        for issue in inconsistencies:
            if issue[1] == 'outlier':
                img = issue[0]
                print(f"\nRe-analyzing {img}...")
                # Re-analyze with updated weights
                new_cat, new_scores, new_conf = ca.analyze_image(img)
                
                if new_cat != results[img]['category']:
                    print(f"  Changed: {results[img]['category']} → {new_cat}")
                    results[img]['category'] = new_cat
                    results[img]['confidence'] = new_conf
                    results[img]['scores'] = new_scores
        
        # Save updated results
        output_file = results_file.replace('.json', '_reviewed.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Updated results saved to {output_file}")

print("\nCategory distribution:")
cats = {}
for r in results.values():
    cats[r['category']] = cats.get(r['category'], 0) + 1
for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count} images")