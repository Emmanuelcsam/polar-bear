"""
Complete OpenCV Blob Detection Scripts Collection
=================================================

This comprehensive collection includes all scripts for blob detection using OpenCV,
covering basic usage, parameter tuning, filtering methods, and advanced techniques.
"""

# ===================================================================
# SCRIPT 1: Basic Blob Detection
# ===================================================================

import cv2
import numpy as np

def basic_blob_detection():
    """
    Basic blob detection using SimpleBlobDetector with default parameters
    """
    # Read image
    img = cv2.imread('blobs.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Set up the detector with default parameters
    detector = cv2.SimpleBlobDetector_create()
    
    # Detect blobs
    keypoints = detector.detect(img)
    
    # Draw detected blobs as red circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of 
    # the circle corresponds to the size of blob
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Show blobs
    cv2.imshow("Keypoints", img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 2: Blob Detection with Full Parameter Control
# ===================================================================

def blob_detection_with_params():
    """
    Blob detection with complete parameter customization
    """
    # Read image
    img = cv2.imread('blobs.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 50000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 1.0
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.maxConvexity = 1.0
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1.0
    
    # Filter by Color
    params.filterByColor = True
    params.blobColor = 0  # 0 for dark blobs, 255 for light blobs
    
    # Distance between blobs
    params.minDistBetweenBlobs = 10
    
    # Repeatability
    params.minRepeatability = 2
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(img)
    
    # Draw detected blobs
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Add text showing number of blobs detected
    text = f"Total Blobs Detected: {len(keypoints)}"
    cv2.putText(img_with_keypoints, text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show blobs
    cv2.imshow("Blob Detection with Parameters", img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 3: Filter by Area Example
# ===================================================================

def filter_by_area_example():
    """
    Demonstrate blob detection filtering by area
    """
    # Create synthetic image with different sized blobs
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add different sized circles (blobs)
    cv2.circle(img, (100, 100), 20, 0, -1)   # Small blob
    cv2.circle(img, (300, 100), 40, 0, -1)   # Medium blob
    cv2.circle(img, (500, 100), 60, 0, -1)   # Large blob
    cv2.circle(img, (100, 300), 80, 0, -1)   # Extra large blob
    cv2.circle(img, (300, 300), 10, 0, -1)   # Tiny blob
    
    # Different area filters
    area_ranges = [
        (100, 2000, "Small blobs (100-2000 pixels)"),
        (2000, 8000, "Medium blobs (2000-8000 pixels)"),
        (8000, 50000, "Large blobs (8000-50000 pixels)")
    ]
    
    for min_area, max_area, title in area_ranges:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        
        # Disable other filters
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        
        img_with_keypoints = cv2.drawKeypoints(
            img.copy(), keypoints, np.array([]), (0,0,255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        cv2.putText(img_with_keypoints, title, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(title, img_with_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 4: Filter by Circularity Example
# ===================================================================

def filter_by_circularity_example():
    """
    Demonstrate blob detection filtering by circularity
    """
    # Create image with different shapes
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add circle (high circularity ~1.0)
    cv2.circle(img, (100, 200), 50, 0, -1)
    
    # Add square (circularity ~0.785)
    cv2.rectangle(img, (200, 150), (300, 250), 0, -1)
    
    # Add ellipse (circularity between circle and rectangle)
    cv2.ellipse(img, (450, 200), (80, 40), 0, 0, 360, 0, -1)
    
    # Different circularity filters
    circularity_ranges = [
        (0.9, 1.0, "High circularity (0.9-1.0) - Circles"),
        (0.7, 0.9, "Medium circularity (0.7-0.9) - Squares"),
        (0.3, 0.7, "Low circularity (0.3-0.7) - Ellipses")
    ]
    
    for min_circ, max_circ, title in circularity_ranges:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = min_circ
        params.maxCircularity = max_circ
        
        # Disable other filters
        params.filterByArea = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        
        img_with_keypoints = cv2.drawKeypoints(
            img.copy(), keypoints, np.array([]), (0,0,255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        cv2.putText(img_with_keypoints, title, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, img_with_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 5: Filter by Inertia Ratio Example
# ===================================================================

def filter_by_inertia_example():
    """
    Demonstrate blob detection filtering by inertia ratio
    Inertia ratio measures how elongated a shape is:
    - Circle: 1.0
    - Ellipse: between 0 and 1
    - Line: 0
    """
    # Create image with different elongated shapes
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add circle (inertia ratio = 1.0)
    cv2.circle(img, (100, 200), 50, 0, -1)
    
    # Add horizontal ellipse (low inertia ratio)
    cv2.ellipse(img, (300, 200), (80, 30), 0, 0, 360, 0, -1)
    
    # Add vertical ellipse (medium inertia ratio)
    cv2.ellipse(img, (500, 200), (30, 80), 0, 0, 360, 0, -1)
    
    # Different inertia ratio filters
    inertia_ranges = [
        (0.8, 1.0, "High inertia ratio (0.8-1.0) - Circles"),
        (0.3, 0.8, "Medium inertia ratio (0.3-0.8) - Moderate ellipses"),
        (0.01, 0.3, "Low inertia ratio (0.01-0.3) - Elongated shapes")
    ]
    
    for min_inertia, max_inertia, title in inertia_ranges:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByInertia = True
        params.minInertiaRatio = min_inertia
        params.maxInertiaRatio = max_inertia
        
        # Disable other filters
        params.filterByArea = False
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        
        img_with_keypoints = cv2.drawKeypoints(
            img.copy(), keypoints, np.array([]), (0,0,255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        cv2.putText(img_with_keypoints, title, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, img_with_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 6: Filter by Color Example
# ===================================================================

def filter_by_color_example():
    """
    Demonstrate blob detection filtering by color (intensity)
    """
    # Create image with both dark and light blobs
    img = np.ones((400, 600), dtype=np.uint8) * 128  # Gray background
    
    # Add dark blobs (intensity = 0)
    cv2.circle(img, (100, 150), 40, 0, -1)
    cv2.circle(img, (250, 150), 40, 0, -1)
    cv2.circle(img, (400, 150), 40, 0, -1)
    
    # Add light blobs (intensity = 255)
    cv2.circle(img, (100, 300), 40, 255, -1)
    cv2.circle(img, (250, 300), 40, 255, -1)
    cv2.circle(img, (400, 300), 40, 255, -1)
    
    # Filter for dark blobs
    params_dark = cv2.SimpleBlobDetector_Params()
    params_dark.filterByColor = True
    params_dark.blobColor = 0  # Detect dark blobs
    params_dark.filterByArea = False
    params_dark.filterByCircularity = False
    params_dark.filterByConvexity = False
    params_dark.filterByInertia = False
    
    detector_dark = cv2.SimpleBlobDetector_create(params_dark)
    keypoints_dark = detector_dark.detect(img)
    
    # Filter for light blobs
    params_light = cv2.SimpleBlobDetector_Params()
    params_light.filterByColor = True
    params_light.blobColor = 255  # Detect light blobs
    params_light.filterByArea = False
    params_light.filterByCircularity = False
    params_light.filterByConvexity = False
    params_light.filterByInertia = False
    
    detector_light = cv2.SimpleBlobDetector_create(params_light)
    keypoints_light = detector_light.detect(img)
    
    # Draw results
    img_dark = cv2.drawKeypoints(
        img.copy(), keypoints_dark, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.putText(img_dark, "Dark Blobs (blobColor = 0)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    img_light = cv2.drawKeypoints(
        img.copy(), keypoints_light, np.array([]), (0,255,0), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.putText(img_light, "Light Blobs (blobColor = 255)", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Dark Blobs", img_dark)
    cv2.imshow("Light Blobs", img_light)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 7: Filter by Convexity Example
# ===================================================================

def filter_by_convexity_example():
    """
    Demonstrate blob detection filtering by convexity
    Convexity = Area / Area of convex hull
    """
    # Create image with different convexity shapes
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add circle (high convexity ~1.0)
    cv2.circle(img, (100, 200), 50, 0, -1)
    
    # Add star shape (low convexity)
    star_points = np.array([
        [300, 150], [320, 200], [370, 200], [330, 230], 
        [350, 280], [300, 250], [250, 280], [270, 230], 
        [230, 200], [280, 200]
    ], np.int32)
    cv2.fillPoly(img, [star_points], 0)
    
    # Add L-shape (medium convexity)
    l_shape = np.array([
        [450, 150], [500, 150], [500, 200], 
        [550, 200], [550, 250], [450, 250]
    ], np.int32)
    cv2.fillPoly(img, [l_shape], 0)
    
    # Different convexity filters
    convexity_ranges = [
        (0.95, 1.0, "High convexity (0.95-1.0) - Convex shapes"),
        (0.7, 0.95, "Medium convexity (0.7-0.95) - Slightly concave"),
        (0.3, 0.7, "Low convexity (0.3-0.7) - Very concave shapes")
    ]
    
    for min_conv, max_conv, title in convexity_ranges:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByConvexity = True
        params.minConvexity = min_conv
        params.maxConvexity = max_conv
        
        # Disable other filters
        params.filterByArea = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByColor = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)
        
        img_with_keypoints = cv2.drawKeypoints(
            img.copy(), keypoints, np.array([]), (0,0,255), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        cv2.putText(img_with_keypoints, title, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, img_with_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 8: Real-World Application - Cell Detection
# ===================================================================

def cell_detection_example():
    """
    Detect cells in a microscopy image using blob detection
    """
    # Read microscopy image (you would use your actual cell image)
    # For demo, we create a synthetic cell-like image
    img = np.ones((600, 800), dtype=np.uint8) * 255
    
    # Add cell-like structures
    np.random.seed(42)
    for _ in range(30):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        radius = np.random.randint(15, 35)
        intensity = np.random.randint(30, 100)
        cv2.circle(img, (x, y), radius, intensity, -1)
    
    # Apply Gaussian blur to make it more realistic
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Setup parameters for cell detection
    params = cv2.SimpleBlobDetector_Params()
    
    # Thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Area filter (cells have certain size range)
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 5000
    
    # Circularity filter (cells are roughly circular)
    params.filterByCircularity = True
    params.minCircularity = 0.7
    
    # Convexity filter (cells are convex)
    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    # Inertia filter (cells are not too elongated)
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    # Create detector and detect cells
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    # Draw detected cells
    img_with_cells = cv2.drawKeypoints(
        img, keypoints, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Add cell count
    text = f"Cells Detected: {len(keypoints)}"
    cv2.putText(img_with_cells, text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Calculate statistics
    if keypoints:
        sizes = [kp.size for kp in keypoints]
        avg_size = np.mean(sizes)
        cv2.putText(img_with_cells, f"Avg Cell Size: {avg_size:.1f} pixels", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Cell Detection", img_with_cells)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 9: Video Blob Tracking
# ===================================================================

def video_blob_tracking():
    """
    Track blobs in video using SimpleBlobDetector
    """
    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)  # or 'video.mp4'
    
    # Setup blob detector for tracking
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 50000
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Store previous positions for tracking
    prev_keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Draw current blobs
        frame_with_blobs = cv2.drawKeypoints(
            frame, keypoints, np.array([]), (0,255,0), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Draw tracking lines (connect to nearest previous blob)
        for kp in keypoints:
            min_dist = float('inf')
            nearest_prev = None
            
            for prev_kp in prev_keypoints:
                dist = np.sqrt((kp.pt[0] - prev_kp.pt[0])**2 + 
                              (kp.pt[1] - prev_kp.pt[1])**2)
                if dist < min_dist and dist < 50:  # Max tracking distance
                    min_dist = dist
                    nearest_prev = prev_kp
            
            if nearest_prev is not None:
                cv2.line(frame_with_blobs, 
                        (int(nearest_prev.pt[0]), int(nearest_prev.pt[1])),
                        (int(kp.pt[0]), int(kp.pt[1])), 
                        (255, 0, 0), 2)
        
        # Update previous keypoints
        prev_keypoints = keypoints
        
        # Display info
        cv2.putText(frame_with_blobs, f"Blobs: {len(keypoints)}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Blob Tracking', frame_with_blobs)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 10: Compare Blob Detection with Other Methods
# ===================================================================

def compare_detection_methods():
    """
    Compare SimpleBlobDetector with contour detection and 
    connected components
    """
    # Create test image
    img = np.ones((400, 600), dtype=np.uint8) * 255
    
    # Add various shapes
    cv2.circle(img, (100, 100), 30, 0, -1)
    cv2.rectangle(img, (200, 70), (280, 130), 0, -1)
    cv2.ellipse(img, (400, 100), (40, 20), 45, 0, 360, 0, -1)
    
    # Add overlapping circles
    cv2.circle(img, (100, 300), 40, 0, -1)
    cv2.circle(img, (140, 300), 40, 0, -1)
    
    # Method 1: SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    img_blob = cv2.drawKeypoints(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), keypoints, np.array([]), 
        (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.putText(img_blob, f"SimpleBlobDetector: {len(keypoints)} blobs", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Method 2: Contour Detection
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    img_contour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)
    
    # Draw centers of contours
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img_contour, (cx, cy), 5, (0, 255, 0), -1)
    
    cv2.putText(img_contour, f"Contour Detection: {len(contours)} contours", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Method 3: Connected Components
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    img_cc = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes and centroids (skip background label 0)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        cv2.rectangle(img_cc, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(img_cc, (int(cx), int(cy)), 5, (255, 0, 0), -1)
    
    cv2.putText(img_cc, f"Connected Components: {num_labels-1} components", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Display all methods
    cv2.imshow("SimpleBlobDetector", img_blob)
    cv2.imshow("Contour Detection", img_contour)
    cv2.imshow("Connected Components", img_cc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================================================================
# SCRIPT 11: Advanced Blob Analysis
# ===================================================================

def advanced_blob_analysis():
    """
    Perform advanced analysis on detected blobs including:
    - Size distribution
    - Spatial distribution
    - Shape analysis
    """
    # Read image
    img = cv2.imread('blobs.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Detect blobs with specific parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    if not keypoints:
        print("No blobs detected!")
        return
    
    # Extract blob properties
    sizes = [kp.size for kp in keypoints]
    positions = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Original image with blobs
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_with_blobs = cv2.drawKeypoints(
        img_color, keypoints, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    axes[0, 0].imshow(cv2.cvtColor(img_with_blobs, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Detected Blobs: {len(keypoints)}')
    axes[0, 0].axis('off')
    
    # 2. Size distribution histogram
    axes[0, 1].hist(sizes, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Blob Size (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Blob Size Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Spatial distribution scatter plot
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    scatter = axes[1, 0].scatter(x_coords, y_coords, c=sizes, 
                                 cmap='viridis', s=100, alpha=0.6)
    axes[1, 0].set_xlabel('X Coordinate')
    axes[1, 0].set_ylabel('Y Coordinate')
    axes[1, 0].set_title('Spatial Distribution of Blobs')
    axes[1, 0].invert_yaxis()  # Match image coordinates
    plt.colorbar(scatter, ax=axes[1, 0], label='Blob Size')
    
    # 4. Statistics text
    stats_text = f"""Blob Statistics:
    
Total Blobs: {len(keypoints)}
Average Size: {np.mean(sizes):.2f} pixels
Std Dev Size: {np.std(sizes):.2f} pixels
Min Size: {np.min(sizes):.2f} pixels
Max Size: {np.max(sizes):.2f} pixels

Spatial Coverage:
X Range: {np.min(x_coords):.0f} - {np.max(x_coords):.0f}
Y Range: {np.min(y_coords):.0f} - {np.max(y_coords):.0f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ===================================================================
# SCRIPT 12: Utility Functions
# ===================================================================

def create_test_images():
    """
    Create various test images for blob detection experiments
    """
    # 1. Random blobs
    img_random = np.ones((600, 800), dtype=np.uint8) * 255
    np.random.seed(42)
    for _ in range(50):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        radius = np.random.randint(10, 50)
        intensity = np.random.randint(0, 200)
        cv2.circle(img_random, (x, y), radius, intensity, -1)
    cv2.imwrite('test_random_blobs.jpg', img_random)
    
    # 2. Grid pattern
    img_grid = np.ones((600, 800), dtype=np.uint8) * 255
    for i in range(5):
        for j in range(6):
            x = 100 + j * 120
            y = 100 + i * 100
            cv2.circle(img_grid, (x, y), 30, 0, -1)
    cv2.imwrite('test_grid_blobs.jpg', img_grid)
    
    # 3. Mixed shapes
    img_mixed = np.ones((600, 800), dtype=np.uint8) * 255
    # Circles
    cv2.circle(img_mixed, (100, 100), 40, 0, -1)
    cv2.circle(img_mixed, (300, 100), 40, 0, -1)
    # Rectangles
    cv2.rectangle(img_mixed, (50, 250), (150, 350), 0, -1)
    cv2.rectangle(img_mixed, (250, 250), (350, 350), 0, -1)
    # Ellipses
    cv2.ellipse(img_mixed, (100, 500), (60, 30), 0, 0, 360, 0, -1)
    cv2.ellipse(img_mixed, (300, 500), (30, 60), 0, 0, 360, 0, -1)
    cv2.imwrite('test_mixed_shapes.jpg', img_mixed)
    
    print("Test images created successfully!")

def save_detection_results(img, keypoints, filename):
    """
    Save blob detection results with annotations
    """
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, np.array([]), (0,0,255), 
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Add blob numbering
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.putText(img_with_keypoints, str(i+1), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Add summary text
    summary = f"Total Blobs: {len(keypoints)}"
    cv2.putText(img_with_keypoints, summary, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite(filename, img_with_keypoints)
    print(f"Results saved to {filename}")

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("OpenCV Blob Detection Demo")
    print("==========================")
    print("1. Basic Blob Detection")
    print("2. Blob Detection with Parameters")
    print("3. Filter by Area")
    print("4. Filter by Circularity")
    print("5. Filter by Inertia")
    print("6. Filter by Color")
    print("7. Filter by Convexity")
    print("8. Cell Detection Example")
    print("9. Video Blob Tracking")
    print("10. Compare Detection Methods")
    print("11. Advanced Blob Analysis")
    print("12. Create Test Images")
    
    choice = input("\nSelect demo (1-12): ")
    
    if choice == '1':
        basic_blob_detection()
    elif choice == '2':
        blob_detection_with_params()
    elif choice == '3':
        filter_by_area_example()
    elif choice == '4':
        filter_by_circularity_example()
    elif choice == '5':
        filter_by_inertia_example()
    elif choice == '6':
        filter_by_color_example()
    elif choice == '7':
        filter_by_convexity_example()
    elif choice == '8':
        cell_detection_example()
    elif choice == '9':
        video_blob_tracking()
    elif choice == '10':
        compare_detection_methods()
    elif choice == '11':
        advanced_blob_analysis()
    elif choice == '12':
        create_test_images()
    else:
        print("Invalid choice!")
