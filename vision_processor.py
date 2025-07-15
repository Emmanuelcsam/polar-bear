import cv2
import numpy as np
import json
import os

def process_with_opencv():
    """Advanced image processing using OpenCV"""
    
    # Find an image to process
    image_file = None
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_file = file
            break
    
    if not image_file:
        print("[VISION] No image found")
        return
    
    print(f"[VISION] Processing {image_file}")
    
    # Read image
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = {
        'image': image_file,
        'dimensions': list(img.shape),
        'edges': {},
        'corners': {},
        'contours': {},
        'features': {},
        'transforms': {}
    }
    
    # 1. Edge Detection
    print("[VISION] Detecting edges...")
    
    # Canny edge detection
    edges_canny = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges_canny > 0)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    results['edges'] = {
        'canny_edge_count': int(edge_pixels),
        'canny_edge_ratio': float(edge_pixels / gray.size),
        'gradient_mean': float(np.mean(gradient_magnitude)),
        'gradient_max': float(np.max(gradient_magnitude)),
        'strong_edges': int(np.sum(gradient_magnitude > 100))
    }
    
    # Save edge map
    cv2.imwrite('edges_canny.jpg', edges_canny)
    print(f"[VISION] Found {edge_pixels} edge pixels")
    
    # 2. Corner Detection
    print("[VISION] Detecting corners...")
    
    # Harris corner detection
    corners_harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners_harris = cv2.dilate(corners_harris, None)
    
    # Find corner coordinates
    threshold = 0.01 * corners_harris.max()
    corner_coords = np.where(corners_harris > threshold)
    corner_points = list(zip(corner_coords[1], corner_coords[0]))[:100]  # Limit to 100
    
    # FAST corner detection
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    
    results['corners'] = {
        'harris_count': len(corner_points),
        'harris_points': [[int(x), int(y)] for x, y in corner_points[:20]],  # First 20
        'fast_count': len(keypoints),
        'corner_density': float(len(corner_points) / gray.size)
    }
    
    print(f"[VISION] Found {len(corner_points)} Harris corners")
    
    # 3. Contour Detection
    print("[VISION] Finding contours...")
    
    # Threshold image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze contours
    contour_info = []
    for i, contour in enumerate(contours[:10]):  # First 10 contours
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            contour_info.append({
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'points': len(contour)
            })
    
    results['contours'] = {
        'total_count': len(contours),
        'details': contour_info,
        'largest_area': float(max([cv2.contourArea(c) for c in contours])) if contours else 0
    }
    
    print(f"[VISION] Found {len(contours)} contours")
    
    # 4. Feature Detection
    print("[VISION] Extracting features...")
    
    # ORB features
    orb = cv2.ORB_create()
    keypoints_orb, descriptors = orb.detectAndCompute(gray, None)
    
    # Image moments
    moments = cv2.moments(gray)
    
    # Hu moments (shape descriptors)
    hu_moments = cv2.HuMoments(moments)
    
    results['features'] = {
        'orb_keypoints': len(keypoints_orb),
        'moments': {
            'center_x': float(moments['m10'] / moments['m00']) if moments['m00'] != 0 else 0,
            'center_y': float(moments['m01'] / moments['m00']) if moments['m00'] != 0 else 0,
            'area': float(moments['m00'])
        },
        'hu_moments': [float(h[0]) for h in hu_moments[:4]]  # First 4 Hu moments
    }
    
    # 5. Image Transforms
    print("[VISION] Computing transforms...")
    
    # Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    results['transforms'] = {
        'fourier': {
            'dominant_frequency': float(np.max(magnitude_spectrum)),
            'mean_frequency': float(np.mean(magnitude_spectrum))
        },
        'histogram': {
            'peak_value': int(np.argmax(hist)),
            'peak_count': int(np.max(hist)),
            'entropy': float(-np.sum(hist[hist>0]/np.sum(hist) * np.log2(hist[hist>0]/np.sum(hist))))
        }
    }
    
    # 6. Texture Analysis
    print("[VISION] Analyzing texture...")
    
    # Local Binary Patterns (simplified)
    def simple_lbp(img):
        h, w = img.shape
        lbp = np.zeros_like(img)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = img[i, j]
                code = 0
                
                # 8 neighbors
                neighbors = [
                    img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                    img[i, j+1], img[i+1, j+1], img[i+1, j],
                    img[i+1, j-1], img[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    lbp = simple_lbp(gray)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    
    results['texture'] = {
        'lbp_entropy': float(-np.sum(lbp_hist[lbp_hist>0]/np.sum(lbp_hist) * 
                                    np.log2(lbp_hist[lbp_hist>0]/np.sum(lbp_hist)))),
        'lbp_uniformity': float(np.sum(lbp_hist**2) / np.sum(lbp_hist)**2)
    }
    
    # Save results
    with open('vision_results.json', 'w') as f:
        json.dump(results, f)
    
    print("[VISION] Vision processing complete")
    
    # Create feature visualization
    create_feature_image(img, gray, edges_canny, corner_points, contours)
    
    # Integrate with existing system
    integrate_with_system(results)

def create_feature_image(img, gray, edges, corners, contours):
    """Create visualization of detected features"""
    
    # Create output image
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Draw edges in blue
    output[edges > 0] = [255, 0, 0]
    
    # Draw corners in red
    for x, y in corners[:50]:  # Limit to 50 corners
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)
    
    # Draw contours in green
    cv2.drawContours(output, contours[:5], -1, (0, 255, 0), 2)
    
    cv2.imwrite('vision_features.jpg', output)
    print("[VISION] Saved feature visualization")

def integrate_with_system(vision_results):
    """Integrate vision results with existing system"""
    
    # Check if we have pixel data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            pixel_data = json.load(f)
        
        # Add vision insights to learned data
        vision_insights = {
            'edge_regions': [],
            'texture_complexity': vision_results['texture']['lbp_entropy'],
            'structural_features': vision_results['edges']['canny_edge_ratio'],
            'shape_descriptors': vision_results['features']['hu_moments']
        }
        
        # Find edge regions in pixel data
        if 'size' in pixel_data:
            width, height = pixel_data['size']
            edge_indices = []
            
            # Sample edge locations
            edge_img = cv2.imread('edges_canny.jpg', 0)
            if edge_img is not None:
                edge_coords = np.where(edge_img > 0)
                for i in range(min(100, len(edge_coords[0]))):
                    y, x = edge_coords[0][i], edge_coords[1][i]
                    pixel_idx = y * width + x
                    if pixel_idx < len(pixel_data['pixels']):
                        edge_indices.append({
                            'index': int(pixel_idx),
                            'value': pixel_data['pixels'][pixel_idx],
                            'position': [int(x), int(y)]
                        })
                
                vision_insights['edge_regions'] = edge_indices[:20]
        
        # Save integrated results
        with open('vision_integration.json', 'w') as f:
            json.dump(vision_insights, f)
        
        print("[VISION] Integrated with pixel data system")
    
    # Check if we have patterns
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
        
        # Add vision-based patterns
        vision_patterns = {
            'geometric': {
                'edge_density': vision_results['edges']['canny_edge_ratio'],
                'corner_density': vision_results['corners']['corner_density'],
                'contour_complexity': len(vision_results['contours']['details'])
            }
        }
        
        with open('vision_patterns.json', 'w') as f:
            json.dump(vision_patterns, f)
        
        print("[VISION] Added geometric patterns")

if __name__ == "__main__":
    process_with_opencv()