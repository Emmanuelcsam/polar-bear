import json
import numpy as np
from scipy import stats

def detect_anomalies():
    try:
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels'])
        
        # Z-score method
        z_scores = np.abs(stats.zscore(pixels))
        threshold = 3
        anomaly_indices = np.where(z_scores > threshold)[0]
        
        # IQR method
        Q1 = np.percentile(pixels, 25)
        Q3 = np.percentile(pixels, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = np.where((pixels < lower_bound) | (pixels > upper_bound))[0]
        
        anomalies = {
            'z_score_anomalies': [
                {'index': int(i), 'value': int(pixels[i]), 'z_score': float(z_scores[i])}
                for i in anomaly_indices[:50]  # Limit to 50
            ],
            'iqr_anomalies': [
                {'index': int(i), 'value': int(pixels[i])}
                for i in iqr_anomalies[:50]
            ],
            'bounds': {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'mean': float(np.mean(pixels)),
                'std': float(np.std(pixels))
            }
        }
        
        print(f"[ANOMALY_DET] Found {len(anomaly_indices)} z-score anomalies")
        print(f"[ANOMALY_DET] Found {len(iqr_anomalies)} IQR anomalies")
        
        with open('anomalies.json', 'w') as f:
            json.dump(anomalies, f)
            
    except Exception as e:
        print(f"[ANOMALY_DET] Error: {e}")

if __name__ == "__main__":
    detect_anomalies()