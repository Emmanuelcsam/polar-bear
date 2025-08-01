import numpy as np
from scipy import ndimage

def extract_fourier_features(gray: np.ndarray) -> dict:
    """Extract 2D Fourier Transform features."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    power = magnitude**2
    phase = np.angle(fshift)
    
    center = np.array(power.shape) // 2
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]
    r = np.hypot(x - center[1], y - center[0]).astype(int)
    
    if r.max() > 0:
        radial_prof = ndimage.mean(power, labels=r, index=np.arange(1, r.max()))
        if len(radial_prof) > 0:
            freqs = np.arange(len(radial_prof))
            spectral_centroid = float(np.sum(freqs * radial_prof) / (np.sum(radial_prof) + 1e-10))
            spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * radial_prof) / (np.sum(radial_prof) + 1e-10)))
        else:
            spectral_centroid, spectral_spread = 0.0, 0.0
    else:
        spectral_centroid, spectral_spread = 0.0, 0.0
    
    return {
        'fft_mean_magnitude': float(np.mean(magnitude)),
        'fft_std_magnitude': float(np.std(magnitude)),
        'fft_max_magnitude': float(np.max(magnitude)),
        'fft_total_power': float(np.sum(power)),
        'fft_dc_component': float(magnitude[center[0], center[1]]),
        'fft_mean_phase': float(np.mean(phase)),
        'fft_std_phase': float(np.std(phase)),
        'fft_spectral_centroid': spectral_centroid,
        'fft_spectral_spread': spectral_spread,
    }

if __name__ == '__main__':
    sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    print("Running Fourier feature extraction on a sample random image...")
    features = extract_fourier_features(sample_image)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
