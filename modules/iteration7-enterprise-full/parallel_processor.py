import json
import time
import os
import numpy as np
from multiprocessing import Pool, cpu_count, Process, Queue, Manager
import concurrent.futures
from functools import partial
import threading

def get_cpu_info():
    """Get CPU information"""
    num_cores = cpu_count()
    
    cpu_info = {
        'physical_cores': num_cores,
        'logical_cores': num_cores,
        'recommended_workers': min(num_cores - 1, 8)  # Leave one core free
    }
    
    print(f"[PARALLEL] CPU detected: {num_cores} cores")
    print(f"[PARALLEL] Using {cpu_info['recommended_workers']} worker processes")
    
    return cpu_info

def process_chunk(chunk_data):
    """Process a chunk of pixels in parallel"""
    chunk_id, pixels = chunk_data
    
    # Perform various operations on the chunk
    results = {
        'chunk_id': chunk_id,
        'size': len(pixels),
        'mean': float(np.mean(pixels)),
        'std': float(np.std(pixels)),
        'min': float(np.min(pixels)),
        'max': float(np.max(pixels))
    }
    
    # Pattern detection in chunk
    patterns = []
    
    # Find runs of same value
    if len(pixels) > 3:
        for i in range(len(pixels) - 3):
            if pixels[i] == pixels[i+1] == pixels[i+2] == pixels[i+3]:
                patterns.append({
                    'type': 'repeat',
                    'value': int(pixels[i]),
                    'position': i,
                    'length': 4
                })
    
    # Find ascending sequences
    for i in range(len(pixels) - 4):
        seq = pixels[i:i+5]
        if all(seq[j] < seq[j+1] for j in range(4)):
            patterns.append({
                'type': 'ascending',
                'start': int(seq[0]),
                'end': int(seq[-1]),
                'position': i
            })
    
    results['patterns'] = patterns[:10]  # Limit to 10 patterns per chunk
    
    # FFT on chunk
    if len(pixels) > 10:
        fft_result = np.fft.fft(pixels)
        magnitude = np.abs(fft_result)
        results['fft_peak'] = float(np.max(magnitude))
        results['fft_peak_freq'] = int(np.argmax(magnitude))
    
    # Entropy calculation
    if len(pixels) > 0:
        hist, _ = np.histogram(pixels, bins=50)
        hist = hist[hist > 0]
        if len(hist) > 0:
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs))
            results['entropy'] = float(entropy)
    
    return results

def parallel_analyze_pixels():
    """Analyze pixel data using parallel processing"""
    
    cpu_info = get_cpu_info()
    num_workers = cpu_info['recommended_workers']
    
    # Load pixel data
    if not os.path.exists('pixel_data.json'):
        print("[PARALLEL] No pixel data found")
        return
    
    with open('pixel_data.json', 'r') as f:
        data = json.load(f)
        pixels = np.array(data['pixels'])
    
    print(f"[PARALLEL] Processing {len(pixels):,} pixels with {num_workers} workers")
    
    start_time = time.time()
    
    # Split data into chunks
    chunk_size = max(100, len(pixels) // (num_workers * 4))  # 4 chunks per worker
    chunks = []
    
    for i in range(0, len(pixels), chunk_size):
        chunk = pixels[i:i+chunk_size]
        chunks.append((i // chunk_size, chunk))
    
    print(f"[PARALLEL] Split into {len(chunks)} chunks of ~{chunk_size} pixels")
    
    # Process chunks in parallel
    with Pool(processes=num_workers) as pool:
        # Map process
        map_start = time.time()
        chunk_results = pool.map(process_chunk, chunks)
        map_time = time.time() - map_start
        
        print(f"[PARALLEL] Map phase complete in {map_time:.2f}s")
        
        # Reduce results
        reduce_start = time.time()
        
        # Aggregate statistics
        total_mean = np.mean([r['mean'] for r in chunk_results])
        total_std = np.mean([r['std'] for r in chunk_results])
        global_min = min(r['min'] for r in chunk_results)
        global_max = max(r['max'] for r in chunk_results)
        
        # Collect all patterns
        all_patterns = []
        for r in chunk_results:
            all_patterns.extend(r.get('patterns', []))
        
        # Find most common pattern types
        pattern_counts = {}
        for p in all_patterns:
            ptype = p['type']
            pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
        
        reduce_time = time.time() - reduce_start
        
        print(f"[PARALLEL] Reduce phase complete in {reduce_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Calculate speedup
    single_thread_estimate = len(pixels) * 0.00001  # Estimated time per pixel
    speedup = single_thread_estimate / total_time if total_time > 0 else 1
    
    # Save results
    parallel_results = {
        'timestamp': time.time(),
        'cpu_cores': num_workers,
        'chunk_count': len(chunks),
        'chunk_size': chunk_size,
        'total_pixels': len(pixels),
        'statistics': {
            'mean': float(total_mean),
            'std': float(total_std),
            'min': float(global_min),
            'max': float(global_max)
        },
        'pattern_summary': pattern_counts,
        'total_patterns_found': len(all_patterns),
        'performance': {
            'total_time': total_time,
            'map_time': map_time,
            'reduce_time': reduce_time,
            'pixels_per_second': len(pixels) / total_time,
            'speedup': speedup
        }
    }
    
    with open('parallel_results.json', 'w') as f:
        json.dump(parallel_results, f)
    
    print(f"[PARALLEL] Analysis complete")
    print(f"[PARALLEL] Total time: {total_time:.2f}s")
    print(f"[PARALLEL] Throughput: {parallel_results['performance']['pixels_per_second']:.0f} pixels/s")
    print(f"[PARALLEL] Speedup: {speedup:.1f}x")

def parallel_correlation_search():
    """Parallel search for correlations across data"""
    
    print("\n[PARALLEL] Starting parallel correlation search...")
    
    # Load multiple data sources
    data_sources = {}
    
    files = ['pixel_data.json', 'patterns.json', 'anomalies.json', 'neural_results.json']
    
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                data_sources[file] = json.load(f)
    
    if len(data_sources) < 2:
        print("[PARALLEL] Not enough data sources for correlation")
        return
    
    # Define correlation tasks
    def correlate_pair(pair_data):
        source1, data1, source2, data2 = pair_data
        
        # Extract numeric arrays from data
        arr1 = extract_numeric_array(data1)
        arr2 = extract_numeric_array(data2)
        
        if arr1 is None or arr2 is None:
            return None
        
        # Make same length
        min_len = min(len(arr1), len(arr2))
        if min_len < 10:
            return None
        
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]
        
        # Calculate correlation
        corr = np.corrcoef(arr1, arr2)[0, 1]
        
        return {
            'source1': source1,
            'source2': source2,
            'correlation': float(corr),
            'samples': min_len
        }
    
    # Create pairs for correlation
    source_names = list(data_sources.keys())
    pairs = []
    
    for i in range(len(source_names)):
        for j in range(i + 1, len(source_names)):
            pairs.append((
                source_names[i],
                data_sources[source_names[i]],
                source_names[j],
                data_sources[source_names[j]]
            ))
    
    # Process correlations in parallel
    num_workers = min(cpu_count() - 1, len(pairs))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        correlation_results = list(executor.map(correlate_pair, pairs))
    
    # Filter out None results
    correlations = [r for r in correlation_results if r is not None]
    
    # Sort by correlation strength
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Save correlation results
    correlation_output = {
        'timestamp': time.time(),
        'sources_analyzed': len(data_sources),
        'pairs_tested': len(pairs),
        'valid_correlations': len(correlations),
        'top_correlations': correlations[:5]
    }
    
    with open('parallel_correlations.json', 'w') as f:
        json.dump(correlation_output, f)
    
    print(f"[PARALLEL] Found {len(correlations)} correlations")
    for corr in correlations[:3]:
        print(f"[PARALLEL] {corr['source1']} <-> {corr['source2']}: {corr['correlation']:.3f}")

def extract_numeric_array(data):
    """Extract numeric array from various data formats"""
    
    if isinstance(data, list):
        # Try to extract numbers from list
        nums = []
        for item in data:
            if isinstance(item, (int, float)):
                nums.append(item)
            elif isinstance(item, dict) and 'value' in item:
                nums.append(item['value'])
        return np.array(nums) if nums else None
    
    elif isinstance(data, dict):
        # Look for common numeric fields
        if 'pixels' in data:
            return np.array(data['pixels'])
        elif 'predictions' in data:
            return np.array(data['predictions'])
        elif 'values' in data:
            return np.array(data['values'])
    
    return None

def parallel_image_batch():
    """Process multiple images in parallel"""
    
    print("\n[PARALLEL] Starting parallel image batch processing...")
    
    # Find all images
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    
    if not image_files:
        print("[PARALLEL] No images found")
        return
    
    print(f"[PARALLEL] Found {len(image_files)} images")
    
    def process_image(filename):
        """Process single image"""
        try:
            from PIL import Image
            
            img = Image.open(filename).convert('L')
            pixels = np.array(img.getdata())
            
            # Compute various metrics
            result = {
                'filename': filename,
                'size': img.size,
                'pixel_count': len(pixels),
                'statistics': {
                    'mean': float(np.mean(pixels)),
                    'std': float(np.std(pixels)),
                    'median': float(np.median(pixels)),
                    'entropy': calculate_entropy(pixels)
                },
                'histogram': np.histogram(pixels, bins=10)[0].tolist()
            }
            
            # Edge detection (simple gradient)
            if img.size[0] > 10 and img.size[1] > 10:
                img_array = pixels.reshape(img.size[1], img.size[0])
                gradx = np.abs(np.diff(img_array, axis=1))
                grady = np.abs(np.diff(img_array, axis=0))
                edge_strength = np.mean(gradx) + np.mean(grady)
                result['edge_strength'] = float(edge_strength)
            
            return result
            
        except Exception as e:
            return {'filename': filename, 'error': str(e)}
    
    # Process images in parallel
    num_workers = min(cpu_count(), len(image_files))
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image, image_files))
    
    process_time = time.time() - start_time
    
    # Save batch results
    batch_output = {
        'timestamp': time.time(),
        'total_images': len(image_files),
        'workers_used': num_workers,
        'total_time': process_time,
        'images_per_second': len(image_files) / process_time,
        'results': results
    }
    
    with open('parallel_batch.json', 'w') as f:
        json.dump(batch_output, f)
    
    print(f"[PARALLEL] Batch processing complete")
    print(f"[PARALLEL] Processed {len(image_files)} images in {process_time:.2f}s")
    print(f"[PARALLEL] Throughput: {batch_output['images_per_second']:.1f} images/s")

def calculate_entropy(pixels):
    """Calculate Shannon entropy"""
    hist, _ = np.histogram(pixels, bins=50)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0
    probs = hist / np.sum(hist)
    return float(-np.sum(probs * np.log2(probs)))

def parallel_pipeline():
    """Run complete parallel processing pipeline"""
    
    print("=== PARALLEL PROCESSING PIPELINE ===\n")
    
    # 1. Parallel pixel analysis
    parallel_analyze_pixels()
    
    # 2. Parallel correlation search
    parallel_correlation_search()
    
    # 3. Parallel image batch
    parallel_image_batch()
    
    print("\n[PARALLEL] Pipeline complete!")
    print("[PARALLEL] Results saved to:")
    print("  - parallel_results.json")
    print("  - parallel_correlations.json")
    print("  - parallel_batch.json")

if __name__ == "__main__":
    parallel_pipeline()