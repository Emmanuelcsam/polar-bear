"""
Refactored Correlation Analyzer Module
Separates functions from main execution for better testability
"""
import numpy as np
from PIL import Image
import pickle
import random
import os
from typing import Dict, List, Tuple, Optional
from connector_interface import setup_connector, send_hivemind_status


def load_pixel_database(filename: str = 'pixel_db.pkl') -> Optional[Dict[str, List[np.ndarray]]]:
    """Load pixel database from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pixel database: {e}")
        return None


def load_weights(filename: str = 'weights.pkl') -> Dict[str, float]:
    """Load category weights from file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return {}


def save_weights(weights: Dict[str, float], filename: str = 'weights.pkl') -> bool:
    """Save category weights to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(weights, f)
        return True
    except Exception as e:
        print(f"Error saving weights: {e}")
        return False


def calculate_pixel_similarity(pixel1: np.ndarray, pixel2: np.ndarray) -> float:
    """Calculate similarity between two pixels"""
    diff = np.abs(pixel1 - pixel2).sum()
    similarity = 1 / (1 + diff/100)
    return similarity


def analyze_image(img_path: str, pixel_db: Dict[str, List[np.ndarray]],
                 weights: Dict[str, float], comparisons: int = 100) -> Tuple[str, Dict[str, float], float]:
    """Analyze an image and return category, scores, and confidence"""
    if not os.path.exists(img_path):
        raise ValueError(f"Image file does not exist: {img_path}")

    try:
        img = np.array(Image.open(img_path).convert('RGB'))
    except Exception as e:
        raise ValueError(f"Error loading image {img_path}: {e}")

    h, w = img.shape[:2]
    scores = {}

    for category, ref_pixels in pixel_db.items():
        if not ref_pixels:
            continue

        total_score = 0
        actual_comparisons = min(comparisons, len(ref_pixels))

        for _ in range(actual_comparisons):
            # Sample random pixel from image
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            img_pixel = img[y, x]

            # Sample random reference pixel
            ref_pixel = random.choice(ref_pixels)

            # Calculate similarity
            similarity = calculate_pixel_similarity(img_pixel, ref_pixel)
            total_score += similarity

        # Calculate weighted average score
        avg_score = total_score / actual_comparisons
        weighted_score = avg_score * weights.get(category, 1.0)
        scores[category] = weighted_score

    if not scores:
        raise ValueError("No valid categories found for analysis")

    # Find best category and calculate confidence
    best_category = max(scores, key=scores.get)
    total_scores = sum(scores.values())
    confidence = scores[best_category] / total_scores if total_scores > 0 else 0

    return best_category, scores, confidence


def update_weights_from_feedback(weights: Dict[str, float], predicted_category: str,
                                correct_category: str, learning_rate: float = 0.1) -> Dict[str, float]:
    """Update weights based on user feedback"""
    new_weights = weights.copy()

    if correct_category in new_weights:
        new_weights[correct_category] *= (1 + learning_rate)
    else:
        new_weights[correct_category] = 1.0 + learning_rate

    if predicted_category in new_weights:
        new_weights[predicted_category] *= (1 - learning_rate)

    return new_weights


def batch_analyze_images(image_paths: List[str], pixel_db: Dict[str, List[np.ndarray]],
                        weights: Dict[str, float], comparisons: int = 100) -> Dict[str, Dict]:
    """Analyze multiple images in batch"""
    results = {}

    for img_path in image_paths:
        try:
            category, scores, confidence = analyze_image(img_path, pixel_db, weights, comparisons)
            results[img_path] = {
                'category': category,
                'scores': scores,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
            results[img_path] = {
                'category': 'error',
                'scores': {},
                'confidence': 0.0,
                'error': str(e)
            }

    return results


def get_image_files_from_directory(directory: str) -> List[str]:
    """Get all image files from a directory"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    if not os.path.exists(directory):
        return image_files

    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_files.append(os.path.join(directory, filename))

    return image_files


if __name__ == "__main__":
    print("Correlation Analyzer loading...")

    # Setup hivemind connector
    connector = setup_connector('correlation_analyzer_refactored.py')
    connector.register_parameter('comparisons', 100, 'Number of comparisons per category')
    connector.register_parameter('learning_rate', 0.1, 'Learning rate for weight updates')

    # Load pixel database
    pixel_db = load_pixel_database()
    if pixel_db is None:
        print("✗ Failed to load pixel database")
        send_hivemind_status({'status': 'error', 'message': 'Failed to load pixel database'}, connector)
        exit(1)

    print(f"✓ Loaded {len(pixel_db)} categories")
    send_hivemind_status({'status': 'loaded', 'categories': len(pixel_db)}, connector)

    # Load weights
    weights = load_weights()
    if not weights:
        weights = {cat: 1.0 for cat in pixel_db}
    else:
        print("✓ Loaded learned weights")

    # Interactive analysis loop
    while True:
        img_path = input("\nImage to analyze (or 'batch' for batch mode): ")
        if img_path.lower() == 'batch':
            batch_dir = input("Directory to process: ")
            image_files = get_image_files_from_directory(batch_dir)
            if image_files:
                # Get comparisons from hivemind
                comparisons = connector.get_parameter('comparisons', 100)
                results = batch_analyze_images(image_files, pixel_db, weights, comparisons)
                print(f"Processed {len(results)} images")
                send_hivemind_status({
                    'status': 'batch_complete',
                    'images_processed': len(results),
                    'directory': batch_dir
                }, connector)
                for path, result in results.items():
                    print(f"{os.path.basename(path)}: {result['category']} ({result['confidence']:.2%})")
            else:
                print("No image files found in directory")
            break

        try:
            # Get comparisons from hivemind
            comparisons = connector.get_parameter('comparisons', 100)
            result, scores, conf = analyze_image(img_path, pixel_db, weights, comparisons)
            print(f"→ Classification: {result} (confidence: {conf:.2%})")

            send_hivemind_status({
                'status': 'analyzed',
                'image': img_path,
                'category': result,
                'confidence': conf
            }, connector)

            feedback = input("Correct? (y/n/skip): ")
            if feedback.lower() == 'n':
                correct = input("Correct category: ")
                # Get learning rate from hivemind
                learning_rate = connector.get_parameter('learning_rate', 0.1)
                weights = update_weights_from_feedback(weights, result, correct, learning_rate)
                if save_weights(weights):
                    print("✓ Weights updated")
                    send_hivemind_status({
                        'status': 'weights_updated',
                        'predicted': result,
                        'correct': correct
                    }, connector)
                else:
                    print("✗ Failed to save weights")

        except Exception as e:
            print(f"Error: {e}")
