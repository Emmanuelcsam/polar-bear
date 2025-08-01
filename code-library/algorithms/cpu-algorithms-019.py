#!/usr/bin/env python3
"""
Defect Characterization and Classification Module
================================================
Advanced defect analysis including geometric characterization,
classification, and confidence scoring for fiber optic inspection.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

def characterize_defect_geometry(contour: np.ndarray, 
                                mask: np.ndarray,
                                um_per_px: Optional[float] = None) -> Dict[str, Any]:
    """
    Comprehensive geometric characterization of a single defect.
    
    Args:
        contour: Contour points of the defect
        mask: Binary mask of the defect
        um_per_px: Microns per pixel conversion factor
        
    Returns:
        Dictionary with geometric properties
    """
    # Basic measurements
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    
    # Moments for centroid
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return {}
    
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Rotated rectangle for better measurements
    if len(contour) >= 5:
        rotated_rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width_px, height_px), angle = rotated_rect
        
        # Ensure width is the smaller dimension
        if width_px > height_px:
            width_px, height_px = height_px, width_px
            angle = (angle + 90) % 180
    else:
        # Fallback for small contours
        width_px = float(w)
        height_px = float(h)
        angle = 0.0
        center_x, center_y = cx, cy
    
    # Calculate derived properties
    aspect_ratio = height_px / (width_px + 1e-6)
    circularity = 4 * np.pi * area_px / (perimeter_px * perimeter_px + 1e-6)
    solidity = area_px / cv2.contourArea(cv2.convexHull(contour)) if len(contour) > 3 else 1.0
    
    # Equivalent circle radius
    equivalent_radius = np.sqrt(area_px / np.pi)
    
    # Eccentricity (from fitted ellipse if possible)
    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (major_axis, minor_axis), _ = ellipse
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except:
            pass
    
    # Hu moments for shape description
    hu_moments = cv2.HuMoments(M).flatten()
    
    # Build result dictionary
    result = {
        "centroid_x_px": float(cx),
        "centroid_y_px": float(cy),
        "area_px": int(area_px),
        "perimeter_px": float(perimeter_px),
        "length_px": float(height_px),  # Length is the larger dimension
        "width_px": float(width_px),    # Width is the smaller dimension
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "eccentricity": float(eccentricity),
        "equivalent_radius_px": float(equivalent_radius),
        "bbox_x_px": int(x),
        "bbox_y_px": int(y),
        "bbox_w_px": int(w),
        "bbox_h_px": int(h),
        "rotated_rect_center_px": (float(center_x), float(center_y)),
        "rotated_rect_angle_deg": float(angle),
        "hu_moments": hu_moments.tolist(),
        "contour_points_px": contour.reshape(-1, 2).tolist()
    }
    
    # Add measurements in microns if scale is available
    if um_per_px and um_per_px > 0:
        result.update({
            "length_um": float(height_px * um_per_px),
            "width_um": float(width_px * um_per_px),
            "area_um2": float(area_px * um_per_px * um_per_px),
            "perimeter_um": float(perimeter_px * um_per_px),
            "equivalent_radius_um": float(equivalent_radius * um_per_px)
        })
    
    return result

def classify_defect_type(geometric_props: Dict[str, Any],
                        scratch_aspect_threshold: float = 3.0,
                        scratch_circularity_threshold: float = 0.3) -> str:
    """
    Classify defect based on geometric properties.
    
    Args:
        geometric_props: Dictionary of geometric properties
        scratch_aspect_threshold: Minimum aspect ratio for scratch classification
        scratch_circularity_threshold: Maximum circularity for scratch classification
        
    Returns:
        Defect classification string
    """
    aspect_ratio = geometric_props.get("aspect_ratio", 1.0)
    circularity = geometric_props.get("circularity", 1.0)
    eccentricity = geometric_props.get("eccentricity", 0.0)
    solidity = geometric_props.get("solidity", 1.0)
    
    # Enhanced classification logic
    if (aspect_ratio >= scratch_aspect_threshold and 
        circularity <= scratch_circularity_threshold and
        eccentricity > 0.8):
        return "Scratch"
    elif circularity > 0.7 and aspect_ratio < 2.0:
        return "Pit"
    elif circularity <= 0.7 and aspect_ratio < scratch_aspect_threshold:
        return "Dig"
    elif solidity < 0.8:
        return "Complex_Defect"
    else:
        return "Unknown_Defect"

def calculate_defect_confidence(defect_pixels: np.ndarray,
                               surrounding_pixels: np.ndarray,
                               geometric_props: Dict[str, Any]) -> float:
    """
    Calculate confidence score for defect detection.
    
    Args:
        defect_pixels: Pixel values within the defect
        surrounding_pixels: Pixel values in surrounding region
        geometric_props: Geometric properties of the defect
        
    Returns:
        Confidence score between 0 and 1
    """
    if len(defect_pixels) == 0 or len(surrounding_pixels) == 0:
        return 0.0
    
    # Intensity contrast
    defect_mean = np.mean(defect_pixels)
    surrounding_mean = np.mean(surrounding_pixels)
    contrast = abs(defect_mean - surrounding_mean)
    contrast_score = min(1.0, float(contrast / 50.0))
    
    # Statistical significance
    defect_std = np.std(defect_pixels)
    surrounding_std = np.std(surrounding_pixels)
    if defect_std > 0 and surrounding_std > 0:
        pooled_std = np.sqrt((defect_std**2 + surrounding_std**2) / 2)
        t_statistic = contrast / (pooled_std + 1e-6)
        significance_score = min(1.0, t_statistic / 3.0)  # Normalize by 3-sigma
    else:
        significance_score = 0.5
    
    # Geometric consistency score
    circularity = geometric_props.get("circularity", 0.0)
    solidity = geometric_props.get("solidity", 1.0)
    area = geometric_props.get("area_px", 0)
    
    # Penalize very small or very irregular shapes
    size_score = min(1.0, area / 10.0)  # Normalize by minimum expected size
    shape_score = (circularity + solidity) / 2.0
    geometry_score = (size_score + shape_score) / 2.0
    
    # Combined confidence score
    confidence = (contrast_score * 0.4 + 
                 significance_score * 0.4 + 
                 geometry_score * 0.2)
    
    return max(0.0, min(1.0, confidence))

def characterize_defects_comprehensive(detected_defects: List[Dict[str, Any]],
                                     original_image: np.ndarray,
                                     um_per_px: Optional[float] = None,
                                     min_defect_area: int = 5) -> List[Dict[str, Any]]:
    """
    Comprehensive defect characterization from detected defect regions.
    
    Args:
        detected_defects: List of detected defect dictionaries
        original_image: Original grayscale image
        um_per_px: Microns per pixel conversion factor
        min_defect_area: Minimum defect area in pixels
        
    Returns:
        List of fully characterized defect dictionaries
    """
    if not detected_defects:
        logging.info("No defects to characterize.")
        return []
    
    characterized_defects = []
    defect_id_counter = 0
    
    for defect_info in detected_defects:
        defect_mask = defect_info.get('defect_mask')
        zone_name = defect_info.get('zone', 'Unknown')
        
        if defect_mask is None:
            continue
        
        # Find contours of the defect
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Process each contour as a separate defect
        for contour in contours:
            # Check minimum area
            area_px = cv2.contourArea(contour)
            if area_px < min_defect_area:
                continue
            
            defect_id_counter += 1
            
            # Create single defect mask
            single_defect_mask = np.zeros_like(defect_mask)
            cv2.fillPoly(single_defect_mask, [contour], (255,))
            
            # Geometric characterization
            geometric_props = characterize_defect_geometry(contour, single_defect_mask, um_per_px)
            if not geometric_props:
                continue
            
            # Classify defect type
            classification = classify_defect_type(geometric_props)
            
            # Calculate confidence score
            defect_pixels = original_image[single_defect_mask > 0]
            
            # Get surrounding pixels for confidence calculation
            kernel_size = max(7, int(np.sqrt(area_px)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(single_defect_mask, kernel)
            surrounding_mask = dilated - single_defect_mask
            surrounding_pixels = original_image[surrounding_mask > 0]
            
            confidence_score = calculate_defect_confidence(defect_pixels, surrounding_pixels, geometric_props)
            
            # Build complete defect dictionary
            defect_dict = {
                "defect_id": f"D{defect_id_counter}",
                "zone": zone_name,
                "classification": classification,
                "confidence_score": float(confidence_score),
                **geometric_props  # Include all geometric properties
            }
            
            # Add effective diameter for circular defects
            if classification in ["Pit", "Dig"]:
                effective_diameter_px = 2 * geometric_props["equivalent_radius_px"]
                defect_dict["effective_diameter_px"] = float(effective_diameter_px)
                
                if um_per_px and um_per_px > 0:
                    defect_dict["effective_diameter_um"] = float(effective_diameter_px * um_per_px)
            
            characterized_defects.append(defect_dict)
    
    logging.info(f"Characterized {len(characterized_defects)} defects from {len(detected_defects)} detected regions")
    
    # Sort defects by zone priority and then by size
    zone_priority = {"Core": 0, "Cladding": 1, "Unknown": 2}
    characterized_defects.sort(
        key=lambda d: (
            zone_priority.get(d["zone"], 2),
            -d.get("length_um", d.get("length_px", 0))
        )
    )
    
    return characterized_defects

def analyze_defect_distribution(defects: List[Dict[str, Any]],
                               fiber_center: Tuple[float, float],
                               fiber_radius: float) -> Dict[str, Any]:
    """
    Analyze spatial distribution of defects relative to fiber geometry.
    
    Args:
        defects: List of characterized defects
        fiber_center: Center of fiber (x, y)
        fiber_radius: Radius of fiber
        
    Returns:
        Dictionary with distribution analysis
    """
    if not defects:
        return {}
    
    # Calculate radial positions
    radial_positions = []
    angular_positions = []
    
    for defect in defects:
        cx = defect.get("centroid_x_px", 0)
        cy = defect.get("centroid_y_px", 0)
        
        # Calculate distance from fiber center
        dx = cx - fiber_center[0]
        dy = cy - fiber_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        radial_position = distance / fiber_radius  # Normalized
        
        # Calculate angular position
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle) % 360
        
        radial_positions.append(radial_position)
        angular_positions.append(angle_deg)
    
    # Analyze distributions
    analysis = {
        "total_defects": len(defects),
        "radial_distribution": {
            "mean": float(np.mean(radial_positions)),
            "std": float(np.std(radial_positions)),
            "min": float(np.min(radial_positions)),
            "max": float(np.max(radial_positions))
        },
        "angular_distribution": {
            "mean": float(np.mean(angular_positions)),
            "std": float(np.std(angular_positions)),
            "angles": angular_positions
        }
    }
    
    # Classify by defect type
    type_counts = {}
    for defect in defects:
        defect_type = defect.get("classification", "Unknown")
        type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
    
    analysis["defect_types"] = type_counts
    
    # Size statistics
    sizes = [d.get("area_um2", d.get("area_px", 0)) for d in defects]
    if sizes:
        analysis["size_statistics"] = {
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
            "min": float(np.min(sizes)),
            "max": float(np.max(sizes)),
            "median": float(np.median(sizes))
        }
    
    return analysis

def filter_defects_by_criteria(defects: List[Dict[str, Any]],
                              criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter defects based on specified criteria.
    
    Args:
        defects: List of characterized defects
        criteria: Dictionary of filtering criteria
        
    Returns:
        Filtered list of defects
    """
    filtered = []
    
    for defect in defects:
        # Check minimum confidence
        min_confidence = criteria.get("min_confidence", 0.0)
        if defect.get("confidence_score", 0.0) < min_confidence:
            continue
        
        # Check minimum size
        min_area = criteria.get("min_area_px", 0)
        if defect.get("area_px", 0) < min_area:
            continue
        
        # Check maximum size
        max_area = criteria.get("max_area_px", float('inf'))
        if defect.get("area_px", 0) > max_area:
            continue
        
        # Check defect types
        allowed_types = criteria.get("allowed_types", [])
        if allowed_types and defect.get("classification") not in allowed_types:
            continue
        
        # Check zones
        allowed_zones = criteria.get("allowed_zones", [])
        if allowed_zones and defect.get("zone") not in allowed_zones:
            continue
        
        filtered.append(defect)
    
    return filtered

if __name__ == "__main__":
    """Test the defect characterization functions"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create test data
    test_image = np.random.randint(100, 150, (200, 200), dtype=np.uint8)
    
    # Create test defects
    defect_mask1 = np.zeros_like(test_image)
    cv2.rectangle(defect_mask1, (50, 50), (55, 70), 255, -1)  # Scratch-like
    
    defect_mask2 = np.zeros_like(test_image)
    cv2.circle(defect_mask2, (100, 100), 5, (255,), -1)  # Pit-like
    
    # Create test contours
    contours1, _ = cv2.findContours(defect_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(defect_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("Testing defect characterization...")
    
    if contours1:
        # Test geometric characterization
        geom_props = characterize_defect_geometry(contours1[0], defect_mask1, 0.5)
        print(f"Geometric properties: {list(geom_props.keys())}")
        
        # Test classification
        classification = classify_defect_type(geom_props)
        print(f"Classification: {classification}")
        
        # Test confidence calculation
        defect_pixels = test_image[defect_mask1 > 0]
        surrounding_mask = cv2.dilate(defect_mask1, np.ones((7,7), np.uint8)) - defect_mask1
        surrounding_pixels = test_image[surrounding_mask > 0]
        
        confidence = calculate_defect_confidence(defect_pixels, surrounding_pixels, geom_props)
        print(f"Confidence: {confidence:.3f}")
    
    # Test comprehensive characterization
    test_defects = [
        {"defect_mask": defect_mask1, "zone": "Core"},
        {"defect_mask": defect_mask2, "zone": "Cladding"}
    ]
    
    characterized = characterize_defects_comprehensive(test_defects, test_image, 0.5)
    print(f"Characterized {len(characterized)} defects")
    
    if characterized:
        # Test distribution analysis
        distribution = analyze_defect_distribution(characterized, (100, 100), 80)
        print(f"Distribution analysis: {list(distribution.keys())}")
        
        # Test filtering
        criteria = {"min_confidence": 0.3, "allowed_types": ["Scratch", "Pit"]}
        filtered = filter_defects_by_criteria(characterized, criteria)
        print(f"Filtered to {len(filtered)} defects")
    
    print("Defect characterization tests completed!")
