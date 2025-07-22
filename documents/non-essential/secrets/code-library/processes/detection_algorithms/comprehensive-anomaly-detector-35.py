import numpy as np
import cv2
import pickle
from typing import Optional, Dict, Any
from .defect_scanning_engine import load_image
from .extract_ultra_comprehensive_features import extract_ultra_comprehensive_features
from .compute_exhaustive_comparison import compute_exhaustive_comparison
from .compute_image_structural_comparison import compute_image_structural_comparison
from .anomaly_region_finder import find_anomaly_regions
from .specific_defect_detector import detect_specific_defects
from skimage.metrics import structural_similarity


def _compute_local_anomaly_map(test_img: np.ndarray, reference_img: np.ndarray
    ) ->np.ndarray:
    """Compute local anomaly map using sliding window and SSIM."""
    h, w = test_img.shape
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    for win_size in [16, 32, 64]:
        if h < win_size or w < win_size:
            continue
        stride = win_size // 2
        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                test_win = test_img[y:y + win_size, x:x + win_size]
                ref_win = reference_img[y:y + win_size, x:x + win_size]
                win_ssim = structural_similarity(test_win, ref_win)
                local_score = 1 - win_ssim
                anomaly_map[y:y + win_size, x:x + win_size] = np.maximum(
                    anomaly_map[y:y + win_size, x:x + win_size], local_score)
    return cv2.GaussianBlur(anomaly_map, (15, 15), 0)


def detect_anomalies_comprehensive(test_path: str, knowledge_base_path: str
    ) ->Optional[Dict[str, Any]]:
    """Perform exhaustive anomaly detection on a test image."""
    print(f'\nAnalyzing: {test_path}')
    try:
        with open(knowledge_base_path, 'rb') as f:
            ref_model = pickle.load(f)
    except FileNotFoundError:
        print(
            f'✗ Knowledge base not found at {knowledge_base_path}. Build one first.'
            )
        return None
    test_image, metadata = load_image(test_path)
    if test_image is None:
        return None
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) if len(test_image
        .shape) == 3 else test_image.copy()
    print('Extracting features from test image...')
    test_features, _ = extract_ultra_comprehensive_features(test_image)
    stat_model = ref_model['statistical_model']
    test_vector = np.array([test_features.get(fname, 0) for fname in
        ref_model['feature_names']])
    diff = test_vector - stat_model['robust_mean']
    mahalanobis_dist = np.sqrt(diff.T @ stat_model['robust_inv_cov'] @ diff)
    archetype = ref_model['archetype_image']
    test_gray_resized = cv2.resize(test_gray, (archetype.shape[1],
        archetype.shape[0]))
    structural_comp = compute_image_structural_comparison(test_gray_resized,
        archetype)
    anomaly_map = _compute_local_anomaly_map(test_gray_resized, archetype)
    anomaly_regions = find_anomaly_regions(anomaly_map, test_gray.shape)
    thresholds = ref_model['learned_thresholds']
    is_anomalous = mahalanobis_dist > thresholds.get('anomaly_threshold', 2.5
        ) or structural_comp['ssim'] < 0.7
    return {'test_image': test_image, 'metadata': metadata,
        'global_analysis': {'mahalanobis_distance': float(mahalanobis_dist)
        }, 'structural_analysis': structural_comp, 'local_analysis': {
        'anomaly_map': anomaly_map, 'anomaly_regions': anomaly_regions},
        'specific_defects': detect_specific_defects(test_gray), 'verdict':
        {'is_anomalous': is_anomalous, 'confidence': min(1.0, 
        mahalanobis_dist / thresholds.get('anomaly_threshold', 2.5))}}


if __name__ == '__main__':
    print(
        "This script contains the main 'detect_anomalies_comprehensive' function."
        )
    kb_path = 'dummy_kb.pkl'
    if not Path(kb_path).exists():
        print(
            """
'dummy_kb.pkl' not found. Please run 'build_comprehensive_reference_model.py' first."""
            )
    else:
        test_img_path = 'dummy_test_image.png'
        test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(test_img_path, test_img)
        print(f"\nRunning anomaly detection on '{test_img_path}'...")
        results = detect_anomalies_comprehensive(test_img_path, kb_path)
        if results:
            print('\nAnalysis Complete!')
            print(
                f"  Verdict: {'ANOMALOUS' if results['verdict']['is_anomalous'] else 'NORMAL'}"
                )
            print(f"  Confidence: {results['verdict']['confidence']:.1%}")
            print(
                f"  Mahalanobis Distance: {results['global_analysis']['mahalanobis_distance']:.2f}"
                )
            print(f"  SSIM Score: {results['structural_analysis']['ssim']:.3f}"
                )
            print(
                f"  Anomaly Regions Found: {len(results['local_analysis']['anomaly_regions'])}"
                )
