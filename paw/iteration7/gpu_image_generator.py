import json
import time
import os
import numpy as np
from PIL import Image

# Check for GPU availability
gpu_available = False
device = None

try:
    import torch
    if torch.cuda.is_available():
        gpu_available = True
        device = torch.device('cuda')
        print(f"[GPU_GEN] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[GPU_GEN] No GPU detected, using CPU")
except ImportError:
    print("[GPU_GEN] PyTorch not available, using NumPy")

def gpu_generate_fractal(size=(512, 512), iterations=100):
    """Generate fractal patterns using GPU acceleration"""
    
    print(f"[GPU_GEN] Generating {size[0]}x{size[1]} fractal with {iterations} iterations")
    start_time = time.time()
    
    if gpu_available and device:
        import torch
        
        # Create coordinate grids on GPU
        y, x = torch.meshgrid(
            torch.linspace(-2, 2, size[0], device=device),
            torch.linspace(-2, 2, size[1], device=device),
            indexing='ij'
        )
        
        # Mandelbrot set computation on GPU
        c = x + 1j * y
        z = torch.zeros_like(c, dtype=torch.complex64)
        output = torch.zeros(size, device=device)
        
        for i in range(iterations):
            mask = torch.abs(z) < 2
            z[mask] = z[mask]**2 + c[mask]
            output[mask] = i
        
        # Convert to image
        output = output.cpu().numpy()
        
    else:
        # CPU fallback
        x = np.linspace(-2, 2, size[1])
        y = np.linspace(-2, 2, size[0])
        X, Y = np.meshgrid(x, y)
        
        c = X + 1j * Y
        z = np.zeros_like(c)
        output = np.zeros(size)
        
        for i in range(iterations):
            mask = np.abs(z) < 2
            z[mask] = z[mask]**2 + c[mask]
            output[mask] = i
    
    # Normalize and save
    output = (output / iterations * 255).astype(np.uint8)
    img = Image.fromarray(output, mode='L')
    
    filename = f'gpu_fractal_{int(time.time())}.jpg'
    img.save(filename)
    
    gen_time = time.time() - start_time
    print(f"[GPU_GEN] Generated in {gen_time:.3f}s")
    print(f"[GPU_GEN] Saved as {filename}")
    
    return filename, gen_time

def gpu_generate_noise_patterns(size=(256, 256), num_patterns=5):
    """Generate various noise patterns using GPU"""
    
    print(f"[GPU_GEN] Generating {num_patterns} noise patterns")
    patterns = []
    
    if gpu_available and device:
        import torch
        
        for i in range(num_patterns):
            # Different noise types
            if i == 0:  # Perlin-like noise
                freq = 0.1
                x = torch.arange(size[0], device=device).float()
                y = torch.arange(size[1], device=device).float()
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                noise = torch.sin(X * freq) * torch.cos(Y * freq)
                noise = (noise + 1) * 127.5
                
            elif i == 1:  # Gaussian noise
                noise = torch.randn(size, device=device) * 50 + 128
                noise = torch.clamp(noise, 0, 255)
                
            elif i == 2:  # Voronoi-like pattern
                num_points = 20
                points = torch.rand(num_points, 2, device=device) * torch.tensor(size, device=device)
                
                x = torch.arange(size[0], device=device).float()
                y = torch.arange(size[1], device=device).float()
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                distances = []
                for p in points:
                    dist = torch.sqrt((X - p[0])**2 + (Y - p[1])**2)
                    distances.append(dist)
                
                distances = torch.stack(distances)
                noise = torch.min(distances, dim=0)[0]
                noise = (noise / noise.max() * 255)
                
            elif i == 3:  # Checkerboard with noise
                x = torch.arange(size[0], device=device)
                y = torch.arange(size[1], device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                checker = ((X // 20 + Y // 20) % 2) * 200
                noise = checker + torch.randn(size, device=device) * 20
                noise = torch.clamp(noise, 0, 255)
                
            else:  # Random walk pattern
                noise = torch.zeros(size, device=device)
                x, y = size[0] // 2, size[1] // 2
                
                for _ in range(10000):
                    if 0 <= x < size[0] and 0 <= y < size[1]:
                        noise[x, y] = 255
                        dx = torch.randint(-1, 2, (1,), device=device).item()
                        dy = torch.randint(-1, 2, (1,), device=device).item()
                        x, y = x + dx, y + dy
            
            # Convert to CPU and save
            pattern = noise.cpu().numpy().astype(np.uint8)
            
    else:
        # CPU fallback with NumPy
        for i in range(num_patterns):
            if i == 0:  # Simple gradient
                x = np.linspace(0, 255, size[0])
                y = np.linspace(0, 255, size[1])
                X, Y = np.meshgrid(x, y)
                pattern = ((X + Y) / 2).astype(np.uint8)
                
            elif i == 1:  # Random noise
                pattern = np.random.randint(0, 256, size, dtype=np.uint8)
                
            elif i == 2:  # Circles
                center = (size[0] // 2, size[1] // 2)
                y, x = np.ogrid[:size[0], :size[1]]
                dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
                pattern = (np.sin(dist * 0.1) * 127 + 128).astype(np.uint8)
                
            elif i == 3:  # Stripes
                x = np.arange(size[1])
                stripes = np.sin(x * 0.1) * 127 + 128
                pattern = np.tile(stripes, (size[0], 1)).astype(np.uint8)
                
            else:  # Gradient noise
                pattern = np.random.normal(128, 30, size).astype(np.uint8)
    
    img = Image.fromarray(pattern, mode='L')
    filename = f'gpu_pattern_{i}.jpg'
    img.save(filename)
    patterns.append(filename)
    
    print(f"[GPU_GEN] Generated {len(patterns)} patterns")
    return patterns

def gpu_generate_from_learned():
    """Generate images based on learned patterns using GPU"""
    
    # Load learned data if available
    learned_stats = None
    
    if os.path.exists('gpu_results.json'):
        with open('gpu_results.json', 'r') as f:
            gpu_data = json.load(f)
            learned_stats = gpu_data.get('results', {}).get('statistics', {})
    
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns_data = json.load(f)
            if 'statistics' in patterns_data and patterns_data['statistics']:
                learned_stats = patterns_data['statistics'][0]
    
    if not learned_stats:
        print("[GPU_GEN] No learned statistics found, using defaults")
        learned_stats = {'mean': 128, 'std': 50}
    
    print(f"[GPU_GEN] Generating based on learned stats: mean={learned_stats['mean']:.1f}, std={learned_stats['std']:.1f}")
    
    size = (256, 256)
    
    if gpu_available and device:
        import torch
        
        # Generate base pattern
        base = torch.randn(size, device=device) * learned_stats['std'] + learned_stats['mean']
        
        # Apply learned transformations
        if 'quartiles' in learned_stats:
            q1, q2, q3 = learned_stats['quartiles']
            
            # Simulate distribution matching
            base = torch.where(base < q1, q1, base)
            base = torch.where(base > q3, q3, base)
        
        # Add texture based on variance
        texture = torch.randn(size, device=device) * 10
        result = base + texture
        result = torch.clamp(result, 0, 255)
        
        # Convert to CPU
        img_array = result.cpu().numpy().astype(np.uint8)
        
    else:
        # CPU fallback
        base = np.random.normal(learned_stats['mean'], learned_stats['std'], size)
        base = np.clip(base, 0, 255).astype(np.uint8)
        img_array = base
    
    img = Image.fromarray(img_array, mode='L')
    filename = 'gpu_learned_generated.jpg'
    img.save(filename)
    
    print(f"[GPU_GEN] Generated learned pattern: {filename}")
    
    return filename

def benchmark_generation():
    """Benchmark GPU vs CPU generation speed"""
    
    print("\n[GPU_GEN] Benchmarking generation speed...")
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    results = []
    
    for size in sizes:
        start = time.time()
        
        # Generate simple pattern
        if gpu_available and device:
            import torch
            data = torch.randn(size, device=device) * 50 + 128
            data = torch.clamp(data, 0, 255)
            result = data.cpu().numpy().astype(np.uint8)
        else:
            result = np.random.normal(128, 50, size).astype(np.uint8)
            result = np.clip(result, 0, 255)
        
        gen_time = time.time() - start
        pixels_per_sec = (size[0] * size[1]) / gen_time
        
        results.append({
            'size': size,
            'time': gen_time,
            'pixels_per_second': pixels_per_sec
        })
        
        print(f"  {size}: {gen_time:.4f}s ({pixels_per_sec:.0f} pixels/s)")
    
    # Save benchmark results
    with open('gpu_generation_benchmark.json', 'w') as f:
        json.dump({
            'device': 'cuda' if gpu_available else 'cpu',
            'benchmarks': results
        }, f)

if __name__ == "__main__":
    print("=== GPU IMAGE GENERATION ===\n")
    
    # 1. Generate fractal
    print("--- Fractal Generation ---")
    fractal_file, fractal_time = gpu_generate_fractal((512, 512), 50)
    
    # 2. Generate noise patterns
    print("\n--- Noise Pattern Generation ---")
    patterns = gpu_generate_noise_patterns()
    
    # 3. Generate from learned data
    print("\n--- Learned Pattern Generation ---")
    learned_file = gpu_generate_from_learned()
    
    # 4. Benchmark
    benchmark_generation()
    
    print("\n[GPU_GEN] Generation complete!")
    print(f"[GPU_GEN] Generated files:")
    print(f"  - {fractal_file} (fractal)")
    for p in patterns:
        print(f"  - {p} (pattern)")
    print(f"  - {learned_file} (learned)")
    print("  - gpu_generation_benchmark.json")