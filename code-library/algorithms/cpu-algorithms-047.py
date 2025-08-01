import cv2
import numpy as np
from skimage.morphology import disk, black_tophat

def detect_specific_defects(gray: np.ndarray) -> dict:
    """Detect specific types of defects like scratches, digs, and blobs."""
    defects = {'scratches': [], 'digs': [], 'blobs': []}
    
    # Scratch detection (linear features)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=20, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            defects['scratches'].append({'line': (x1, y1, x2, y2)})
            
    # Dig detection (small dark spots) using black tophat
    bth = black_tophat(gray, disk(7))
    _, dig_mask = cv2.threshold(bth, np.percentile(bth[bth>0], 95) if np.any(bth>0) else 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(dig_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if 10 < cv2.contourArea(c) < 500:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
                defects['digs'].append({'center': (cx, cy), 'area': cv2.contourArea(c)})

    # Blob detection (larger irregular features) via adaptive threshold
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 100:
            defects['blobs'].append({'contour': c, 'area': cv2.contourArea(c)})
            
    return defects

if __name__ == '__main__':
    # Create a sample image with different defects
    image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.line(image, (20, 20), (180, 40), 64, 2) # Scratch
    cv2.circle(image, (50, 150), 5, 32, -1) # Dig
    cv2.ellipse(image, (150, 150), (20, 10), 0, 0, 360, 192, -1) # Blob

    print("Detecting specific defects in a sample image...")
    detected_defects = detect_specific_defects(image)
    
    print("\nDetection Results:")
    print(f"  Scratches: {len(detected_defects['scratches'])}")
    print(f"  Digs: {len(detected_defects['digs'])}")
    print(f"  Blobs: {len(detected_defects['blobs'])}")
    
    # Create a visualization
    viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for s in detected_defects['scratches']:
        cv2.line(viz_img, s['line'][:2], s['line'][2:], (0, 255, 255), 2)
    for d in detected_defects['digs']:
        cv2.circle(viz_img, d['center'], 5, (255, 0, 255), -1)
    cv2.drawContours(viz_img, [b['contour'] for b in detected_defects['blobs']], -1, (255, 255, 0), 2)
    cv2.imwrite("specific_defects_viz.png", viz_img)
    print("\nSaved 'specific_defects_viz.png' with highlighted defects.")
