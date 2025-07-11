"""
Refactored Batch Processor Module
Separates functions from main execution for better testability
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import correlation_analyzer_refactored as ca


def get_image_files(directory: str) -> List[str]:
    """Get all image files from a directory"""
    if not os.path.exists(directory):
        return []
    
    image_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]


def process_batch(batch_dir: str, pixel_db: Dict, weights: Dict, 
                 progress_callback=None) -> Dict[str, Dict]:
    """Process a batch of images and return results"""
    if not os.path.exists(batch_dir):
        raise ValueError(f"Batch directory does not exist: {batch_dir}")
    
    all_images = get_image_files(batch_dir)
    if not all_images:
        raise ValueError(f"No image files found in directory: {batch_dir}")
    
    results = {}
    
    for i, img_file in enumerate(all_images):
        img_path = os.path.join(batch_dir, img_file)
        
        try:
            category, scores, confidence = ca.analyze_image(img_path, pixel_db, weights)
            
            results[img_file] = {
                'category': category,
                'confidence': confidence,
                'scores': scores,
                'timestamp': datetime.now().isoformat(),
                'path': img_path
            }
            
            if progress_callback:
                progress_callback(i + 1, len(all_images), img_file, category, confidence)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            results[img_file] = {
                'category': 'error',
                'confidence': 0.0,
                'scores': {},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    return results


def save_results(results: Dict, output_file: Optional[str] = None) -> str:
    """Save results to JSON file"""
    if output_file is None:
        output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return output_file
    except Exception as e:
        raise ValueError(f"Error saving results: {e}")


def load_results(results_file: str) -> Dict:
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading results: {e}")


def get_category_distribution(results: Dict) -> Dict[str, int]:
    """Get category distribution from results"""
    distribution = {}
    for result in results.values():
        category = result.get('category', 'unknown')
        distribution[category] = distribution.get(category, 0) + 1
    return distribution


def print_progress_stats(processed: int, total: int, category_dist: Dict[str, int]):
    """Print progress statistics"""
    print(f"\nProgress: {processed}/{total} ({processed/total*100:.1f}%)")
    print("Categories so far:")
    for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")


def default_progress_callback(current: int, total: int, filename: str, category: str, confidence: float):
    """Default progress callback function"""
    print(f"\n[{current}/{total}] Processing {filename}")
    print(f"  → {category} ({confidence:.1%})")
    
    # Print stats every 10 images
    if current % 10 == 0:
        print(f"\nProgress: {current}/{total} ({current/total*100:.1f}%)")


if __name__ == "__main__":
    print("Batch Processor starting...")
    
    # Load dependencies
    pixel_db = ca.load_pixel_database()
    if pixel_db is None:
        print("✗ Failed to load pixel database")
        exit(1)
    
    weights = ca.load_weights()
    if not weights:
        weights = {cat: 1.0 for cat in pixel_db}
    
    # Get input
    batch_dir = input("Directory with images to process: ")
    
    try:
        # Process batch
        results = process_batch(batch_dir, pixel_db, weights, default_progress_callback)
        
        # Save results
        output_file = save_results(results)
        print(f"\n✓ Batch processing complete! Results saved to {output_file}")
        
        # Print final statistics
        distribution = get_category_distribution(results)
        print("\nFinal category distribution:")
        for cat, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
            
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
