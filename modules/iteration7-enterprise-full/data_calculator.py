import json
import numpy as np
from scipy import stats, signal
import os

def calculate_intensive():
    results = {}
    
    try:
        # Load all available data
        data_files = {
            'pixel_data.json': None,
            'patterns.json': None,
            'intensity_analysis.json': None,
            'geometry_analysis.json': None
        }
        
        for file in data_files:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data_files[file] = json.load(f)
        
        # Fourier Transform on pixel data
        if data_files['pixel_data.json']:
            pixels = np.array(data_files['pixel_data.json']['pixels'])
            
            # FFT
            fft_result = np.fft.fft(pixels)
            frequencies = np.fft.fftfreq(len(pixels))
            
            # Find dominant frequencies
            magnitude = np.abs(fft_result)
            dominant_freq_indices = np.argsort(magnitude)[-10:]
            
            results['fourier'] = {
                'dominant_frequencies': [
                    {
                        'frequency': float(frequencies[i]),
                        'magnitude': float(magnitude[i])
                    }
                    for i in dominant_freq_indices
                ],
                'dc_component': float(magnitude[0]),
                'spectral_energy': float(np.sum(magnitude**2))
            }
            
            print(f"[CALCULATOR] FFT complete, spectral energy: {results['fourier']['spectral_energy']:.2f}")
        
        # Entropy calculations
        if data_files['intensity_analysis.json']:
            hist_data = data_files['intensity_analysis.json']['histogram']
            counts = np.array(hist_data['counts'])
            
            # Shannon entropy
            probabilities = counts / np.sum(counts)
            probabilities = probabilities[probabilities > 0]
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            results['entropy'] = {
                'shannon': float(entropy),
                'normalized': float(entropy / np.log2(256)),  # Normalized to [0,1]
                'effective_bits': float(entropy)
            }
            
            print(f"[CALCULATOR] Entropy: {entropy:.4f} bits")
        
        # Autocorrelation
        if data_files['pixel_data.json']:
            pixels = np.array(data_files['pixel_data.json']['pixels'])
            
            # Calculate autocorrelation for different lags
            autocorr = []
            max_lag = min(100, len(pixels) // 4)
            
            for lag in range(1, max_lag):
                if lag < len(pixels):
                    corr = np.corrcoef(pixels[:-lag], pixels[lag:])[0, 1]
                    autocorr.append({
                        'lag': lag,
                        'correlation': float(corr) if not np.isnan(corr) else 0.0
                    })
            
            results['autocorrelation'] = {
                'values': autocorr[:20],  # First 20 lags
                'first_zero_crossing': next(
                    (a['lag'] for a in autocorr if a['correlation'] < 0), 
                    None
                )
            }
        
        # Texture measures
        if data_files['pixel_data.json']:
            pixels = np.array(data_files['pixel_data.json']['pixels'])
            size = data_files['pixel_data.json']['size']
            img_2d = pixels.reshape(size[1], size[0])
            
            # Co-occurrence matrix (simplified)
            glcm = np.zeros((256, 256))
            for i in range(size[1] - 1):
                for j in range(size[0] - 1):
                    glcm[img_2d[i, j], img_2d[i, j+1]] += 1
            
            glcm = glcm / np.sum(glcm)
            
            # Texture features
            i, j = np.ogrid[0:256, 0:256]
            contrast = np.sum(glcm * (i - j)**2)
            homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
            
            results['texture'] = {
                'contrast': float(contrast),
                'homogeneity': float(homogeneity),
                'energy': float(np.sum(glcm**2))
            }
        
        # Statistical tests
        if data_files['pixel_data.json']:
            pixels = np.array(data_files['pixel_data.json']['pixels'])
            
            # Normality test
            _, p_value = stats.normaltest(pixels)
            
            results['statistical_tests'] = {
                'normality': {
                    'p_value': float(p_value),
                    'is_normal': bool(p_value > 0.05)
                },
                'skewness': float(stats.skew(pixels)),
                'kurtosis': float(stats.kurtosis(pixels))
            }
        
        # Save results
        with open('calculations.json', 'w') as f:
            json.dump(results, f)
        
        print(f"[CALCULATOR] Intensive calculations complete")
        
    except Exception as e:
        print(f"[CALCULATOR] Error: {e}")

if __name__ == "__main__":
    calculate_intensive()