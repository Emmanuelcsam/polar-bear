import numpy as np
from PIL import Image
import pickle
import random
import os

print("Correlation Analyzer loading...")
with open('pixel_db.pkl', 'rb') as f:
    pixel_db = pickle.load(f)
print(f"Loaded {len(pixel_db)} categories")

weights = {cat: 1.0 for cat in pixel_db}
if os.path.exists('weights.pkl'):
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    print("✓ Loaded learned weights")

def analyze_image(img_path):
    print(f"\nAnalyzing {img_path}...")
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w = img.shape[:2]
    
    scores = {}
    for category, ref_pixels in pixel_db.items():
        print(f"  Comparing with {category}...")
        total_score = 0
        comparisons = min(100, len(ref_pixels))
        
        for _ in range(comparisons):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            img_pixel = img[y, x]
            ref_pixel = random.choice(ref_pixels)
            
            diff = np.abs(img_pixel - ref_pixel).sum()
            similarity = 1 / (1 + diff/100)
            total_score += similarity
            
        avg_score = total_score / comparisons * weights[category]
        scores[category] = avg_score
        print(f"    Score: {avg_score:.4f}")
    
    best_cat = max(scores, key=scores.get)
    confidence = scores[best_cat] / sum(scores.values())
    print(f"  → Classification: {best_cat} (confidence: {confidence:.2%})")
    
    return best_cat, scores, confidence

if __name__ == "__main__":
    while True:
        img_path = input("\nImage to analyze (or 'batch' for batch mode): ")
        if img_path == 'batch':
            import batch_processor
            break
        result, scores, conf = analyze_image(img_path)
        
        feedback = input("Correct? (y/n/skip): ")
        if feedback == 'n':
            correct = input("Correct category: ")
            weights[correct] *= 1.1
            weights[result] *= 0.9
            with open('weights.pkl', 'wb') as f:
                pickle.dump(weights, f)
            print("✓ Weights updated")