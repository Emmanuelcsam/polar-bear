import cv2
import numpy as np

def find_anomaly_regions(anomaly_map: np.ndarray, original_shape: tuple) -> list:
    """Find distinct anomaly regions from the anomaly map."""
    if np.all(anomaly_map <= 0): return []
    
    threshold = np.percentile(anomaly_map[anomaly_map > 0], 80)
    binary_map = (anomaly_map > threshold).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    
    regions = []
    h_scale = original_shape[0] / anomaly_map.shape[0]
    w_scale = original_shape[1] / anomaly_map.shape[1]
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 20:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            regions.append({
                'bbox': (int(x*w_scale), int(y*h_scale), int(w*w_scale), int(h*h_scale)),
                'area': int(stats[i, cv2.CC_STAT_AREA] * h_scale * w_scale),
                'confidence': float(np.mean(anomaly_map[labels == i])),
                'centroid': (int(centroids[i][0]*w_scale), int(centroids[i][1]*h_scale)),
            })
            
    return sorted(regions, key=lambda x: x['confidence'], reverse=True)

if __name__ == '__main__':
    # Create a sample anomaly map
    anomaly_map = np.zeros((50, 50), dtype=np.float32)
    anomaly_map[10:20, 10:20] = 0.8
    anomaly_map[30:45, 30:45] = 0.6
    original_shape = (100, 100)

    print("Finding anomaly regions in a sample map...")
    regions = find_anomaly_regions(anomaly_map, original_shape)
    
    print(f"\nFound {len(regions)} regions:")
    for region in regions:
        print(f"  - BBox: {region['bbox']}, Confidence: {region['confidence']:.2f}")
