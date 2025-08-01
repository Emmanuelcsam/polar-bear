import os
import json
from PIL import Image
import numpy as np

def process_batch(folder_path):
    results = []
    
    print(f"[BATCH_PROC] Processing folder: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # Process image
                img = Image.open(filepath).convert('L')
                pixels = list(img.getdata())
                
                # Calculate stats
                pixel_array = np.array(pixels)
                stats = {
                    'filename': filename,
                    'size': img.size,
                    'mean': float(np.mean(pixel_array)),
                    'std': float(np.std(pixel_array)),
                    'min': int(np.min(pixel_array)),
                    'max': int(np.max(pixel_array)),
                    'unique_values': len(np.unique(pixel_array))
                }
                
                results.append(stats)
                print(f"[BATCH_PROC] Processed {filename}")
                
                # Save individual pixel data
                with open(f'batch_{filename}.json', 'w') as f:
                    json.dump({
                        'filename': filename,
                        'pixels': pixels[:1000],  # Save first 1000 pixels
                        'stats': stats
                    }, f)
                    
            except Exception as e:
                print(f"[BATCH_PROC] Error with {filename}: {e}")
    
    # Save batch results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f)
    
    print(f"[BATCH_PROC] Processed {len(results)} images")
    return results

if __name__ == "__main__":
    # Process current directory by default
    process_batch('.')