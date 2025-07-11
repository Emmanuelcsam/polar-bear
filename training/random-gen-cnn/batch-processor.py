import os
import json
from datetime import datetime

# Import correlation analyzer module
import sys
sys.path.append('.')
import correlation_analyzer as ca

print("Batch Processor starting...")
batch_dir = input("Directory with images to process: ")
output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

results = {}
all_images = [f for f in os.listdir(batch_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Found {len(all_images)} images to process")

for i, img_file in enumerate(all_images):
    img_path = os.path.join(batch_dir, img_file)
    print(f"\n[{i+1}/{len(all_images)}] Processing {img_file}")
    
    category, scores, confidence = ca.analyze_image(img_path)
    
    results[img_file] = {
        'category': category,
        'confidence': confidence,
        'scores': scores,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved to {output_file}")
    
    if (i + 1) % 10 == 0:
        print(f"\nProgress: {i+1}/{len(all_images)} ({(i+1)/len(all_images)*100:.1f}%)")
        print("Categories so far:")
        cats = {}
        for r in results.values():
            cats[r['category']] = cats.get(r['category'], 0) + 1
        for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")

print(f"\n✓ Batch processing complete! Results in {output_file}")