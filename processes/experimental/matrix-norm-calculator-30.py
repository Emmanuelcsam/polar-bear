import numpy as np

def extract_matrix_norms(gray: np.ndarray) -> dict:
    """Extract various matrix norms."""
    return {
        'norm_frobenius': float(np.linalg.norm(gray, 'fro')),
        'norm_l1': float(np.linalg.norm(gray, 1)),
        'norm_l2': float(np.linalg.norm(gray, 2)),
        'norm_linf': float(np.linalg.norm(gray, np.inf)),
        'norm_nuclear': float(np.linalg.norm(gray, 'nuc')),
        'norm_trace': float(np.trace(gray)),
    }

if __name__ == '__main__':
    sample_image = np.random.rand(100, 100) * 255
    
    print("Running matrix norm extraction on a sample random image...")
    features = extract_matrix_norms(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

