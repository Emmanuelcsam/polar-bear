import json
import numpy as np
from PIL import Image
import os

def generate_image():
    try:
        # Load learned data
        if os.path.exists('learned_data.json'):
            with open('learned_data.json', 'r') as f:
                learned = json.load(f)
                frequencies = learned.get('pixel_frequencies', {})
        
        # Load original image info
        if os.path.exists('pixel_data.json'):
            with open('pixel_data.json', 'r') as f:
                original = json.load(f)
                size = original.get('size', [100, 100])
        
        # Generate based on learned frequencies
        if frequencies:
            values = list(frequencies.keys())
            weights = list(frequencies.values())
            total_pixels = size[0] * size[1]
            
            # Weighted random selection
            generated_pixels = np.random.choice(
                [int(v) for v in values],
                size=total_pixels,
                p=np.array(weights)/sum(weights)
            )
        else:
            # Random if no learned data
            generated_pixels = np.random.randint(0, 256, size[0] * size[1])
        
        # Create image
        img = Image.new('L', size)
        img.putdata(generated_pixels.tolist())
        
        filename = f'generated_{int(np.random.rand()*10000)}.jpg'
        img.save(filename)
        
        print(f"[IMAGE_GEN] Generated {filename}")
        print(f"[IMAGE_GEN] Size: {size}, Mean: {np.mean(generated_pixels):.2f}")
        
        # Save generation info
        with open('generation_log.json', 'w') as f:
            json.dump({
                'filename': filename,
                'size': size,
                'mean': float(np.mean(generated_pixels)),
                'std': float(np.std(generated_pixels))
            }, f)
            
    except Exception as e:
        print(f"[IMAGE_GEN] Error: {e}")

if __name__ == "__main__":
    generate_image()