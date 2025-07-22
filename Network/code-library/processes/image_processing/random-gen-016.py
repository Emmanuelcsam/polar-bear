import json
import numpy as np
from PIL import Image
import os

def categorize_images():
    categories = {
        'dark': {'min': 0, 'max': 85, 'images': []},
        'medium': {'min': 86, 'max': 170, 'images': []},
        'bright': {'min': 171, 'max': 255, 'images': []},
        'high_contrast': {'images': []},
        'low_contrast': {'images': []},
        'uniform': {'images': []}
    }
    
    # Load learned data for advanced categorization
    learned_categories = {}
    if os.path.exists('learned_data.json'):
        with open('learned_data.json', 'r') as f:
            learned = json.load(f)
            if 'image_profiles' in learned:
                # Create categories based on learned profiles
                for path, profile in learned['image_profiles'].items():
                    mean = profile['mean']
                    category = f"learned_cluster_{int(mean // 32)}"
                    if category not in learned_categories:
                        learned_categories[category] = []
                    learned_categories[category].append(path)
    
    # Process all images in current directory
    for filename in os.listdir('.'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                img = Image.open(filename).convert('L')
                pixels = np.array(img.getdata())
                
                # Basic categorization
                mean_val = np.mean(pixels)
                std_val = np.std(pixels)
                unique_count = len(np.unique(pixels))
                
                # Brightness category
                if mean_val <= 85:
                    categories['dark']['images'].append(filename)
                elif mean_val <= 170:
                    categories['medium']['images'].append(filename)
                else:
                    categories['bright']['images'].append(filename)
                
                # Contrast category
                if std_val > 60:
                    categories['high_contrast']['images'].append(filename)
                elif std_val < 30:
                    categories['low_contrast']['images'].append(filename)
                
                # Uniformity
                if unique_count < 50:
                    categories['uniform']['images'].append(filename)
                
                print(f"[CATEGORIZER] {filename}: mean={mean_val:.1f}, std={std_val:.1f}")
                
            except Exception as e:
                print(f"[CATEGORIZER] Error with {filename}: {e}")
    
    # Advanced categorization based on learned patterns
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
            
            # Create pattern-based categories
            if 'frequency' in patterns:
                freq = patterns['frequency']
                # Find images with similar frequency distributions
                # This would require comparing with individual image data
    
    # Save categorization results
    results = {
        'basic_categories': {
            name: {
                'count': len(info['images']),
                'images': info['images'][:10]  # Limit to 10 per category
            }
            for name, info in categories.items()
        },
        'learned_categories': {
            name: {
                'count': len(images),
                'images': images[:10]
            }
            for name, images in learned_categories.items()
        },
        'total_images_processed': sum(len(cat['images']) for cat in categories.values())
    }
    
    with open('image_categories.json', 'w') as f:
        json.dump(results, f)
    
    print(f"[CATEGORIZER] Categorization complete")
    print(f"[CATEGORIZER] Processed {results['total_images_processed']} images")
    
    # Summary
    for name, info in categories.items():
        if info['images']:
            print(f"[CATEGORIZER] {name}: {len(info['images'])} images")

if __name__ == "__main__":
    categorize_images()