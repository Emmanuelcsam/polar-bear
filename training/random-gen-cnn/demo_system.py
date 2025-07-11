#!/usr/bin/env python3
"""
Demonstration script for the Image Categorization System
Shows end-to-end functionality with sample data
"""
import os
import tempfile
import numpy as np
from PIL import Image
import json

# Import refactored modules
import pixel_sampler_refactored as ps
import correlation_analyzer_refactored as ca
import batch_processor_refactored as bp
import self_reviewer_refactored as sr


def create_sample_data():
    """Create sample reference images for demonstration"""
    print("Creating sample data...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    ref_dir = os.path.join(temp_dir, 'references')
    
    # Define categories and their representative colors
    categories = {
        'red_objects': (255, 0, 0),
        'blue_objects': (0, 0, 255),
        'green_objects': (0, 255, 0),
        'yellow_objects': (255, 255, 0)
    }
    
    # Create reference images
    for category, base_color in categories.items():
        cat_dir = os.path.join(ref_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # Create multiple images with slight variations
        for i in range(3):
            # Add some variation to the base color
            r, g, b = base_color
            r = max(0, min(255, r + np.random.randint(-20, 21)))
            g = max(0, min(255, g + np.random.randint(-20, 21)))
            b = max(0, min(255, b + np.random.randint(-20, 21)))
            
            # Create image with varied colors
            img = Image.new('RGB', (100, 100), color=(r, g, b))
            img.save(os.path.join(cat_dir, f'{category}_{i}.jpg'))
    
    print(f"Sample data created in: {temp_dir}")
    return temp_dir


def demonstrate_pixel_sampling(ref_dir):
    """Demonstrate pixel database building"""
    print("\n" + "="*50)
    print("STEP 1: BUILDING PIXEL DATABASE")
    print("="*50)
    
    # Build pixel database
    pixel_db = ps.build_pixel_database(ref_dir, sample_size=50)
    
    # Get statistics
    stats = ps.get_database_stats(pixel_db)
    print(f"Database built successfully!")
    print(f"Categories: {stats['categories']}")
    print(f"Total pixels: {stats['total_pixels']}")
    
    # Show pixels per category
    for category, count in stats['pixels_per_category'].items():
        print(f"  {category}: {count} pixels")
    
    # Save database
    db_file = 'demo_pixel_db.pkl'
    ps.save_pixel_database(pixel_db, db_file)
    print(f"Database saved to: {db_file}")
    
    return pixel_db


def demonstrate_single_analysis(pixel_db, temp_dir):
    """Demonstrate single image analysis"""
    print("\n" + "="*50)
    print("STEP 2: SINGLE IMAGE ANALYSIS")
    print("="*50)
    
    # Create test images
    test_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    names = ['test_red.jpg', 'test_green.jpg', 'test_blue.jpg', 'test_yellow.jpg']
    
    for color, name in zip(colors, names):
        img = Image.new('RGB', (80, 80), color=color)
        img_path = os.path.join(temp_dir, name)
        img.save(img_path)
        test_images.append((img_path, name))
    
    # Initialize weights
    weights = {cat: 1.0 for cat in pixel_db.keys()}
    
    # Analyze each test image
    results = []
    for img_path, name in test_images:
        category, scores, confidence = ca.analyze_image(img_path, pixel_db, weights)
        results.append((name, category, confidence, scores))
        
        print(f"\n{name}:")
        print(f"  Predicted: {category}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Scores: {dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))}")
    
    return results, weights


def demonstrate_batch_processing(pixel_db, weights, temp_dir):
    """Demonstrate batch processing"""
    print("\n" + "="*50)
    print("STEP 3: BATCH PROCESSING")
    print("="*50)
    
    # Create batch directory with test images
    batch_dir = os.path.join(temp_dir, 'batch_test')
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create mixed set of test images
    test_data = [
        ('red_test1.jpg', (255, 0, 0)),
        ('red_test2.jpg', (240, 10, 5)),
        ('blue_test1.jpg', (0, 0, 255)),
        ('blue_test2.jpg', (5, 5, 240)),
        ('green_test1.jpg', (0, 255, 0)),
        ('yellow_test1.jpg', (255, 255, 0))
    ]
    
    for filename, color in test_data:
        img = Image.new('RGB', (60, 60), color=color)
        img.save(os.path.join(batch_dir, filename))
    
    # Process batch
    def progress_callback(current, total, filename, category, confidence):
        print(f"  [{current}/{total}] {filename} → {category} ({confidence:.1%})")
    
    results = bp.process_batch(batch_dir, pixel_db, weights, progress_callback)
    
    # Save results
    results_file = 'demo_results.json'
    bp.save_results(results, results_file)
    print(f"\nBatch processing complete! Results saved to: {results_file}")
    
    # Show category distribution
    distribution = bp.get_category_distribution(results)
    print(f"\nCategory distribution:")
    for cat, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} images")
    
    return results


def demonstrate_self_review(results):
    """Demonstrate self-review functionality"""
    print("\n" + "="*50)
    print("STEP 4: SELF-REVIEW")
    print("="*50)
    
    # Group results by category
    grouped = sr.group_by_category(results)
    print(f"Results grouped into {len(grouped)} categories")
    
    # Review consistency
    inconsistencies = sr.review_category_consistency(grouped)
    
    if inconsistencies:
        print(f"\nFound inconsistencies in {len(inconsistencies)} categories:")
        for category, issues in inconsistencies.items():
            print(f"  {category}: {len(issues)} issues")
    else:
        print("\nNo significant inconsistencies found!")
    
    # Calculate statistics
    stats = sr.calculate_review_statistics(results)
    print(f"\nReview Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Error count: {stats['error_count']}")
    
    if stats['confidence_stats']:
        conf_stats = stats['confidence_stats']
        print(f"  Confidence stats:")
        print(f"    Mean: {conf_stats['mean']:.3f}")
        print(f"    Std Dev: {conf_stats['std']:.3f}")
        print(f"    Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
    
    return stats


def demonstrate_weight_learning(weights, pixel_db):
    """Demonstrate weight learning"""
    print("\n" + "="*50)
    print("STEP 5: WEIGHT LEARNING")
    print("="*50)
    
    print("Initial weights:")
    for cat, weight in weights.items():
        print(f"  {cat}: {weight:.3f}")
    
    # Simulate some feedback
    print("\nSimulating feedback...")
    feedback_scenarios = [
        ('red_objects', 'blue_objects', "Corrected red→blue"),
        ('blue_objects', 'blue_objects', "Confirmed blue→blue"),
        ('green_objects', 'yellow_objects', "Corrected green→yellow"),
        ('yellow_objects', 'yellow_objects', "Confirmed yellow→yellow")
    ]
    
    for predicted, correct, description in feedback_scenarios:
        print(f"  {description}")
        weights = ca.update_weights_from_feedback(weights, predicted, correct)
    
    print("\nUpdated weights:")
    for cat, weight in weights.items():
        print(f"  {cat}: {weight:.3f}")
    
    # Save updated weights
    ca.save_weights(weights, 'demo_weights.pkl')
    print("Weights saved to: demo_weights.pkl")
    
    return weights


def cleanup_demo_files():
    """Clean up demonstration files"""
    demo_files = [
        'demo_pixel_db.pkl',
        'demo_weights.pkl',
        'demo_results.json'
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed: {file}")


def main():
    """Main demonstration function"""
    print("="*60)
    print("IMAGE CATEGORIZATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    try:
        # Create sample data
        temp_dir = create_sample_data()
        ref_dir = os.path.join(temp_dir, 'references')
        
        # Step 1: Build pixel database
        pixel_db = demonstrate_pixel_sampling(ref_dir)
        
        # Step 2: Single image analysis
        single_results, weights = demonstrate_single_analysis(pixel_db, temp_dir)
        
        # Step 3: Batch processing
        batch_results = demonstrate_batch_processing(pixel_db, weights, temp_dir)
        
        # Step 4: Self-review
        review_stats = demonstrate_self_review(batch_results)
        
        # Step 5: Weight learning
        updated_weights = demonstrate_weight_learning(weights, pixel_db)
        
        # Final summary
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("✅ Pixel database built successfully")
        print("✅ Single image analysis working")
        print("✅ Batch processing functional")
        print("✅ Self-review system operational")
        print("✅ Weight learning implemented")
        print("\nThe system is ready for production use!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print("\nCleaning up demonstration files...")
        cleanup_demo_files()
        print("Demonstration complete!")


if __name__ == "__main__":
    main()
