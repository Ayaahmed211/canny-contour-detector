import numpy as np
import cv2


def detect_ellipses(
    edges: np.ndarray,
    min_contour_points: int = 5,
    min_area: float = 100.0,
    max_area: float = None,
    min_aspect_ratio: float = 0.2,  # Minimum aspect ratio (minor/major)
    max_aspect_ratio: float = 0.9,  # Maximum aspect ratio (to avoid near-circles if you want only ellipses)
    min_solidity: float = 0.7,  # How solid/filled the ellipse should be
    max_center_distance: float = 10.0,  # Max distance from contour center to ellipse center
):
    """
    Detect ellipses by fitting them to contours found in an edge image.
    Includes extensive filtering to remove false positives.

    Parameters
    ----------
    edges              : Binary edge image (uint8)
    min_contour_points : Minimum number of contour points required to fit an ellipse (≥5)
    min_area           : Minimum ellipse area to keep
    max_area           : Maximum ellipse area to keep (None = unlimited)
    min_aspect_ratio   : Minimum aspect ratio (minor_axis/major_axis) to keep
    max_aspect_ratio   : Maximum aspect ratio to keep (set to 1.0 to include circles)
    min_solidity       : Minimum solidity (contour area / convex hull area)
    max_center_distance: Maximum distance between contour centroid and ellipse center
    
    Returns
    -------
    List of cv2.fitEllipse return values — each is
      ((cx, cy), (major_axis, minor_axis), angle_deg)
    suitable for passing directly to cv2.ellipse().
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Filter contours by size first (too small contours can't be good ellipses)
    min_contour_area = min_area * 0.5  # Contour should be at least half the ellipse area
    filtered_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_contour_area:
            filtered_contours.append(cnt)
    
    print(f"Found {len(filtered_contours)} contours after area filtering")
    
    ellipses = []
    h, w = edges.shape[:2]
    img_area = h * w

    for cnt in filtered_contours:
        if len(cnt) < max(5, min_contour_points):
            continue
        
        # Get contour properties
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Skip if contour is too small or too large
        if area < min_area * 0.5:
            continue
        if max_area is not None and area > max_area:
            continue
            
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        (cx, cy), (ma, mi), angle = ellipse
        
        # Ensure major axis is the longer one
        if ma < mi:
            ma, mi = mi, ma
            angle += 90  # Adjust angle if we swapped axes
        
        # Calculate ellipse area
        ellipse_area = np.pi * (ma / 2) * (mi / 2)
        
        # Skip if ellipse area is unrealistic compared to contour area
        area_ratio = ellipse_area / area if area > 0 else 0
        if area_ratio > 5.0 or area_ratio < 0.5:
            continue
        
        # Filter degenerate ellipses
        if ma <= 0 or mi <= 0 or ma > img_area * 2:
            continue
        
        # Filter by aspect ratio
        aspect_ratio = mi / ma
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        # Filter by size
        if ellipse_area < min_area:
            continue
        if max_area is not None and ellipse_area > max_area:
            continue
        
        # Check if ellipse center is within image bounds (with margin)
        margin = max(ma, mi) * 0.1
        if not (margin <= cx <= w - margin and margin <= cy <= h - margin):
            continue
        
        # Calculate contour centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            # Check if ellipse center is close to contour centroid
            center_dist = np.sqrt((cx - centroid_x)**2 + (cy - centroid_y)**2)
            if center_dist > max_center_distance:
                continue
        
        # Calculate solidity (how solid/filled the shape is)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < min_solidity:
            continue
        
        # Calculate circularity (4π * area / perimeter²) - 1.0 is perfect circle
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # For ellipses, circularity should be less than for circles
        # but not too low (which would indicate a very irregular shape)
        if circularity < 0.3:  # Too irregular
            continue
        
        # Calculate how well the contour fits the ellipse
        # Create a mask of the ellipse
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)
        
        # Count pixels that are both in the contour and in the ellipse
        contour_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        
        intersection = cv2.bitwise_and(mask, contour_mask)
        intersection_area = np.count_nonzero(intersection)
        
        # Calculate overlap ratio (should be high)
        overlap_ratio = intersection_area / area if area > 0 else 0
        if overlap_ratio < 0.6:  # Less than 60% overlap
            continue
        
        # Calculate edge support (how many edge points lie on the ellipse)
        # Create thin ellipse mask
        edge_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.ellipse(edge_mask, ellipse, 255, 2)  # 2-pixel thick ellipse
        
        # Count edge points on the ellipse
        edge_support = cv2.bitwise_and(edges, edge_mask)
        edge_count = np.count_nonzero(edge_support)
        
        # Expected edge points based on ellipse perimeter
        perimeter_approx = np.pi * (3 * (ma + mi) - np.sqrt((3*ma + mi) * (ma + 3*mi)))
        expected_edges = int(perimeter_approx * 0.3)  # Expect 30% of perimeter to have edges
        
        if edge_count < expected_edges:
            continue
        
        # Additional check: look for concentric edges
        # Create a slightly smaller and larger ellipse to check for consistent edges
        smaller = ((cx, cy), (ma*0.8, mi*0.8), angle)
        larger = ((cx, cy), (ma*1.2, mi*1.2), angle)
        
        smaller_mask = np.zeros(edges.shape, dtype=np.uint8)
        larger_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.ellipse(smaller_mask, smaller, 255, 1)
        cv2.ellipse(larger_mask, larger, 255, 1)
        
        smaller_support = cv2.bitwise_and(edges, smaller_mask)
        larger_support = cv2.bitwise_and(edges, larger_mask)
        
        smaller_count = np.count_nonzero(smaller_support)
        larger_count = np.count_nonzero(larger_support)
        
        # If there are more edges on nearby ellipses than on the target, it's suspicious
        if smaller_count > edge_count * 1.5 or larger_count > edge_count * 1.5:
            continue
        
        # All checks passed - this is a good ellipse
        # Store with additional metadata for debugging/filtering
        ellipse_with_meta = {
            'ellipse': ellipse,
            'area': ellipse_area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity,
            'edge_support': edge_count / expected_edges if expected_edges > 0 else 0,
            'overlap': overlap_ratio
        }
        ellipses.append(ellipse_with_meta)

    # Sort by confidence (combination of metrics)
    if ellipses:
        # Calculate confidence score for each ellipse
        for e in ellipses:
            score = (
                e['overlap'] * 0.3 +
                min(e['edge_support'], 1.0) * 0.3 +
                e['solidity'] * 0.2 +
                (1 - abs(e['circularity'] - 0.7) / 0.7) * 0.2  # Prefer moderate circularity
            )
            e['confidence'] = score
        
        # Sort by confidence
        ellipses.sort(key=lambda x: x['confidence'], reverse=True)
        
        # De-duplicate very similar ellipses
        unique = []
        for e in ellipses:
            (cx, cy), (ma, mi), angle = e['ellipse']
            
            # Ensure major axis is longer
            if ma < mi:
                ma, mi = mi, ma
            
            duplicate = False
            for u in unique:
                (ux, uy), (uma, umi), uangle = u['ellipse']
                
                # Ensure major axis is longer for comparison
                if uma < umi:
                    uma, umi = umi, uma
                
                # Check center distance
                center_dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
                
                # Check size difference
                size_diff = abs(ma - uma) / max(ma, uma)
                
                # If centers are close and sizes are similar, consider duplicate
                if center_dist < 20 and size_diff < 0.3:
                    duplicate = True
                    break
            
            if not duplicate:
                unique.append(e)
        
        # Return just the ellipse parameters (without metadata)
        print(f"Found {len(unique)} good ellipses after filtering")
        return [e['ellipse'] for e in unique]

    print("No ellipses found")
    return []


def draw_ellipses_debug(img: np.ndarray, ellipses: list) -> np.ndarray:
    """Draw ellipses with confidence visualization (for debugging)."""
    result = img.copy()
    
    for i, ellipse in enumerate(ellipses):
        (cx, cy), (ma, mi), angle = ellipse
        
        # Ensure major axis is longer for consistent coloring
        if ma < mi:
            ma, mi = mi, ma
            angle += 90
        
        # Color based on aspect ratio
        aspect_ratio = mi / ma
        if aspect_ratio > 0.8:  # Almost circular
            color = (0, 255, 0)  # Green
        elif aspect_ratio > 0.5:  # Moderately elliptical
            color = (0, 255, 255)  # Yellow
        else:  # Very elongated
            color = (255, 0, 0)  # Blue
        
        cv2.ellipse(result, ellipse, color, 2)
        cv2.circle(result, (int(cx), int(cy)), 3, (255, 255, 255), -1)
        
        # Add label with aspect ratio
        label = f"{i+1}: {aspect_ratio:.2f}"
        cv2.putText(result, label, (int(cx)-30, int(cy)-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result


def filter_ellipses_by_contour_quality(ellipses_with_meta: list, min_confidence: float = 0.5):
    """Filter ellipses by confidence score."""
    return [e for e in ellipses_with_meta if e.get('confidence', 0) >= min_confidence]