import cv2
import numpy as np
import pickle
import time
from datetime import datetime
import os

print("Live Monitor starting...")
with open('pixel_db.pkl', 'rb') as f:
    pixel_db = pickle.load(f)

weights = {cat: 1.0 for cat in pixel_db}
if os.path.exists('weights.pkl'):
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)

# Setup display
cv2.namedWindow('Live Categorizer', cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_SIMPLEX

watch_dir = input("Directory to monitor: ")
processed = set()

print(f"Monitoring {watch_dir}...")
print("Press 'q' to quit, 's' to save snapshot")

while True:
    # Check for new images
    for img_file in os.listdir(watch_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')) and img_file not in processed:
            img_path = os.path.join(watch_dir, img_file)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New image: {img_file}")
            
            # Load and analyze
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            display_img = img.copy()
            
            # Calculate scores
            scores = {}
            sample_points = []
            
            for category, ref_pixels in pixel_db.items():
                total_score = 0
                comparisons = min(50, len(ref_pixels))
                
                for _ in range(comparisons):
                    y, x = np.random.randint(0, h-1), np.random.randint(0, w-1)
                    sample_points.append((x, y))
                    
                    img_pixel = img[y, x]
                    ref_pixel = ref_pixels[np.random.randint(len(ref_pixels))]
                    
                    diff = np.abs(img_pixel - ref_pixel).sum()
                    similarity = 1 / (1 + diff/100)
                    total_score += similarity
                
                avg_score = total_score / comparisons * weights[category]
                scores[category] = avg_score
            
            # Get top categories
            sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_cat = sorted_cats[0][0]
            confidence = sorted_cats[0][1] / sum(scores.values())
            
            # Draw sample points
            for x, y in sample_points[-100:]:  # Show last 100 points
                cv2.circle(display_img, (x, y), 2, (0, 255, 0), -1)
            
            # Add text overlay
            text_y = 30
            cv2.putText(display_img, f"File: {img_file}", (10, text_y), 
                       font, 0.7, (255, 255, 255), 2)
            text_y += 30
            cv2.putText(display_img, f"Category: {best_cat} ({confidence:.1%})", 
                       (10, text_y), font, 0.7, (0, 255, 0), 2)
            
            # Show top 5 categories
            text_y += 40
            for i, (cat, score) in enumerate(sorted_cats[:5]):
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(display_img, f"{i+1}. {cat}: {score:.3f}", 
                           (10, text_y), font, 0.5, color, 1)
                text_y += 25
            
            # Show stats
            cv2.putText(display_img, f"Processed: {len(processed)+1}", 
                       (10, h-30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, f"Categories: {len(pixel_db)}", 
                       (10, h-10), font, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Live Categorizer', display_img)
            
            processed.add(img_file)
            print(f"  → {best_cat} ({confidence:.1%})")
            
            # Log result
            with open('live_log.txt', 'a') as f:
                f.write(f"{datetime.now().isoformat()},{img_file},{best_cat},{confidence:.3f}\n")
    
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        if 'display_img' in locals():
            cv2.imwrite(snapshot_name, display_img)
            print(f"✓ Saved {snapshot_name}")
    
    time.sleep(0.1)

cv2.destroyAllWindows()
print(f"\n✓ Processed {len(processed)} images")