"""
Refactored Self Reviewer Module
Separates functions from main execution for better testability
"""
import json
import numpy as np
from PIL import Image
import os
import pickle
from typing import Dict, List, Tuple, Optional
import correlation_analyzer_refactored as ca


def load_results(results_file: str) -> Dict:
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading results file: {e}")


def group_by_category(results: Dict) -> Dict[str, List[Tuple[str, Dict]]]:
    """Group results by category"""
    categorized = {}
    for img, data in results.items():
        category = data.get('category', 'unknown')
        if category not in categorized:
            categorized[category] = []
        categorized[category].append((img, data))
    return categorized


def find_confidence_inconsistencies(images: List[Tuple[str, Dict]], 
                                  threshold: float = 0.3) -> List[Tuple[str, str, float]]:
    """Find confidence inconsistencies within a category"""
    inconsistencies = []
    
    for i, (img1, data1) in enumerate(images):
        for j, (img2, data2) in enumerate(images):
            if i >= j:  # Avoid duplicate comparisons
                continue
                
            conf_diff = abs(data1.get('confidence', 0) - data2.get('confidence', 0))
            if conf_diff > threshold:
                inconsistencies.append((img1, img2, conf_diff))
    
    return inconsistencies


def find_statistical_outliers(images: List[Tuple[str, Dict]], 
                            std_threshold: float = 2.0) -> List[Tuple[str, str, float]]:
    """Find statistical outliers based on confidence scores"""
    if len(images) < 3:  # Need at least 3 samples for meaningful stats
        return []
    
    confidences = [data.get('confidence', 0) for _, data in images]
    mean_conf = np.mean(confidences)
    std_conf = np.std(confidences)
    
    outliers = []
    for img, data in images:
        conf = data.get('confidence', 0)
        if abs(conf - mean_conf) > std_threshold * std_conf:
            outliers.append((img, 'outlier', conf))
    
    return outliers


def review_category_consistency(categorized: Dict[str, List[Tuple[str, Dict]]],
                              conf_threshold: float = 0.3,
                              std_threshold: float = 2.0) -> Dict[str, List]:
    """Review consistency within each category"""
    all_inconsistencies = {}
    
    for category, images in categorized.items():
        if len(images) < 2:
            continue
        
        inconsistencies = []
        
        # Find confidence inconsistencies
        conf_issues = find_confidence_inconsistencies(images, conf_threshold)
        inconsistencies.extend(conf_issues)
        
        # Find statistical outliers
        outliers = find_statistical_outliers(images, std_threshold)
        inconsistencies.extend(outliers)
        
        if inconsistencies:
            all_inconsistencies[category] = inconsistencies
    
    return all_inconsistencies


def re_analyze_suspicious_images(inconsistencies: Dict[str, List], 
                                results: Dict, pixel_db: Dict, weights: Dict) -> Dict:
    """Re-analyze suspicious images and update results"""
    updated_results = results.copy()
    changes_made = []
    
    for category, issues in inconsistencies.items():
        for issue in issues:
            if issue[1] == 'outlier':  # Only re-analyze outliers
                img = issue[0]
                
                # Get the full path if available
                img_path = img
                if img in updated_results and 'path' in updated_results[img]:
                    img_path = updated_results[img]['path']
                
                try:
                    new_cat, new_scores, new_conf = ca.analyze_image(img_path, pixel_db, weights)
                    
                    old_cat = updated_results[img]['category']
                    if new_cat != old_cat:
                        updated_results[img]['category'] = new_cat
                        updated_results[img]['confidence'] = new_conf
                        updated_results[img]['scores'] = new_scores
                        changes_made.append((img, old_cat, new_cat))
                        
                except Exception as e:
                    print(f"Error re-analyzing {img}: {e}")
    
    return updated_results, changes_made


def calculate_review_statistics(results: Dict) -> Dict:
    """Calculate statistics from review results"""
    stats = {
        'total_images': len(results),
        'categories': {},
        'confidence_stats': {},
        'error_count': 0
    }
    
    confidences = []
    for img, data in results.items():
        category = data.get('category', 'unknown')
        confidence = data.get('confidence', 0)
        
        # Count categories
        if category not in stats['categories']:
            stats['categories'][category] = 0
        stats['categories'][category] += 1
        
        # Collect confidences
        if category != 'error':
            confidences.append(confidence)
        else:
            stats['error_count'] += 1
    
    # Calculate confidence statistics
    if confidences:
        stats['confidence_stats'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences)
        }
    
    return stats


def save_reviewed_results(results: Dict, original_file: str) -> str:
    """Save reviewed results to new file"""
    output_file = original_file.replace('.json', '_reviewed.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        return output_file
    except Exception as e:
        raise ValueError(f"Error saving reviewed results: {e}")


def print_review_summary(inconsistencies: Dict[str, List], changes_made: List, stats: Dict):
    """Print a summary of the review process"""
    print(f"\n{'='*50}")
    print(f"Review Summary:")
    print(f"  Total inconsistencies found: {sum(len(issues) for issues in inconsistencies.values())}")
    print(f"  Categories with issues: {len(inconsistencies)}")
    print(f"  Changes made: {len(changes_made)}")
    print(f"  Total images reviewed: {stats['total_images']}")
    print(f"  Error images: {stats['error_count']}")
    
    if stats['confidence_stats']:
        conf_stats = stats['confidence_stats']
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {conf_stats['mean']:.3f}")
        print(f"  Std Dev: {conf_stats['std']:.3f}")
        print(f"  Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
    
    print(f"\nCategory Distribution:")
    sorted_cats = sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats:
        pct = count / stats['total_images'] * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    print("Self-Reviewer starting...")
    
    results_file = input("Results JSON file to review: ")
    
    try:
        # Load results
        results = load_results(results_file)
        print(f"✓ Loaded {len(results)} categorized images")
        
        # Group by category
        categorized = group_by_category(results)
        print(f"✓ Found {len(categorized)} categories")
        
        # Review consistency
        inconsistencies = review_category_consistency(categorized)
        
        if inconsistencies:
            print(f"\n⚠ Found inconsistencies in {len(inconsistencies)} categories")
            
            # Ask if user wants to re-analyze
            fix = input("\nAttempt to re-categorize suspicious images? (y/n): ")
            if fix.lower() == 'y':
                # Load analysis dependencies
                pixel_db = ca.load_pixel_database()
                weights = ca.load_weights()
                
                if pixel_db is None:
                    print("✗ Could not load pixel database for re-analysis")
                else:
                    if not weights:
                        weights = {cat: 1.0 for cat in pixel_db}
                    
                    # Re-analyze suspicious images
                    updated_results, changes_made = re_analyze_suspicious_images(
                        inconsistencies, results, pixel_db, weights
                    )
                    
                    if changes_made:
                        print(f"\n✓ Made {len(changes_made)} changes:")
                        for img, old_cat, new_cat in changes_made:
                            print(f"  {img}: {old_cat} → {new_cat}")
                        
                        # Save updated results
                        output_file = save_reviewed_results(updated_results, results_file)
                        print(f"✓ Updated results saved to {output_file}")
                        results = updated_results
                    else:
                        print("No changes were made during re-analysis")
        else:
            print("✓ No significant inconsistencies found")
        
        # Calculate and display final statistics
        stats = calculate_review_statistics(results)
        print_review_summary(inconsistencies, [] if 'changes_made' not in locals() else changes_made, stats)
        
    except Exception as e:
        print(f"✗ Review failed: {e}")
