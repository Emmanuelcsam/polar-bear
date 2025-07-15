import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from sklearn.covariance import MinCovDet
import pickle

from .load_image import load_image
from .extract_ultra_comprehensive_features import extract_ultra_comprehensive_features
from .compute_exhaustive_comparison import compute_exhaustive_comparison

def _get_default_thresholds():
    return {'anomaly_mean': 1.0, 'anomaly_std': 0.5, 'anomaly_p95': 2.0, 'anomaly_threshold': 2.5}

def build_comprehensive_reference_model(ref_dir: str, knowledge_base_path: str):
    """Build an exhaustive reference model from a directory of JSON/image files."""
    print(f"\nBuilding Comprehensive Reference Model from: {ref_dir}")
    
    all_files = list(Path(ref_dir).glob('*.[jp][np]g')) + list(Path(ref_dir).glob('*.json'))
    if not all_files:
        print(f"✗ No valid image or JSON files found in {ref_dir}"); return False
    
    all_features, all_images, feature_names = [], [], None
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Processing {file_path.name}")
        image, _ = load_image(str(file_path))
        if image is None: continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        features, f_names = extract_ultra_comprehensive_features(image)
        
        if feature_names is None: feature_names = f_names
        all_features.append(features); all_images.append(gray)
        print(f"  ✓ Processed: {len(features)} features extracted")

    if len(all_features) < 2:
        print(f"✗ Error: At least 2 valid reference files are required."); return False
        
    print("\nBuilding Statistical Model...")
    feature_matrix = np.array([[f.get(name, 0) for name in feature_names] for f in all_features])
    
    try:
        mcd = MinCovDet(support_fraction=0.75).fit(feature_matrix)
        robust_mean, robust_cov = mcd.location_, mcd.covariance_
    except Exception as e:
        print(f"  ⚠ MCD failed ({e}), using standard covariance")
        robust_mean = np.mean(feature_matrix, axis=0)
        robust_cov = np.cov(feature_matrix, rowvar=False)
    
    print("Creating archetype image...")
    archetype_image = np.median(all_images, axis=0).astype(np.uint8)
    
    print("Computing pairwise comparisons for threshold learning...")
    comparison_scores = [compute_exhaustive_comparison(all_features[i], all_features[j])['euclidean_distance'] 
                         for i in range(len(all_features)) for j in range(i + 1, len(all_features))]
    
    scores = np.array(comparison_scores)
    thresholds = {
        'anomaly_mean': float(np.mean(scores)), 'anomaly_std': float(np.std(scores)),
        'anomaly_p95': float(np.percentile(scores, 95)),
        'anomaly_threshold': float(np.mean(scores) + 2.5 * np.std(scores)),
    } if len(scores) > 0 else _get_default_thresholds()
    
    reference_model = {
        'features': all_features, 'feature_names': feature_names,
        'statistical_model': {
            'mean': np.mean(feature_matrix, axis=0), 'std': np.std(feature_matrix, axis=0),
            'robust_mean': robust_mean, 'robust_inv_cov': np.linalg.pinv(robust_cov),
        },
        'archetype_image': archetype_image, 'learned_thresholds': thresholds,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(knowledge_base_path, 'wb') as f: pickle.dump(reference_model, f)
    print(f"\n✓ Reference Model Built Successfully and saved to {knowledge_base_path}")
    return True

if __name__ == '__main__':
    # This is a complex function to run standalone. We'll create a dummy environment.
    print("This script contains the 'build_comprehensive_reference_model' function.")
    
    # Create a dummy reference directory
    ref_dir = Path("dummy_reference_images")
    ref_dir.mkdir(exist_ok=True)
    for i in range(3):
        path = ref_dir / f"ref_{i}.png"
        img = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)
        
    kb_path = "dummy_kb.pkl"
    print(f"\nAttempting to build a reference model from dummy images in '{ref_dir}'...")
    success = build_comprehensive_reference_model(str(ref_dir), kb_path)
    
    if success:
        print(f"\nDummy knowledge base created at '{kb_path}'.")
