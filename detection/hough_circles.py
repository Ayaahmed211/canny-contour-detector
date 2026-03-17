import numpy as np
import cv2


def preprocess_for_circles(gray: np.ndarray) -> np.ndarray:
    """
    Preprocess grayscale image for better circle detection.
    
    Steps:
    1. Apply Gaussian blur to reduce noise
    2. Keep edges sharp for better circle detection
    """
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    return blurred


def detect_circles(
    gray: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 30,
    param1: int = 80,  # Higher = fewer edges, less circles
    param2: int = 40,  # Higher = stricter circle detection
    min_radius: int = 10,
    max_radius: int = 0,
    use_adaptive_params: bool = True,
) -> list:
    """
    Detect circles using OpenCV's Hough Circle Transform.
    
    Parameters
    ----------
    gray       : Grayscale image (uint8)
    dp         : Inverse ratio of accumulator resolution to image resolution
    min_dist   : Minimum distance between circle centers
    param1     : Higher threshold for the Canny internal detector
    param2     : Accumulator threshold for circle centers (higher = fewer circles)
    min_radius : Minimum circle radius
    max_radius : Maximum circle radius (0 = any)
    use_adaptive_params : Whether to adapt parameters based on image
    
    Returns
    -------
    List of (cx, cy, radius) tuples, or empty list.
    """
    # Make a copy to avoid modifying original
    img = gray.copy()
    
    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Preprocess the image
    processed = preprocess_for_circles(img)
    
    # Get image dimensions
    h, w = processed.shape
    
    # Set reasonable min_dist based on image size if not specified
    if min_dist == 0:
        min_dist = min(h, w) // 10
    
    # Set max_radius based on image size if not specified
    if max_radius == 0:
        max_radius = min(h, w) // 2
    
    # Adaptive parameters
    if use_adaptive_params:
        # Adjust param1 based on image contrast
        mean_intensity = np.mean(processed)
        std_intensity = np.std(processed)
        
        # Higher contrast needs higher param1
        if std_intensity > 60:
            param1 = min(150, param1 + 20)
        elif std_intensity < 30:
            param1 = max(50, param1 - 10)
        
        # Adjust param2 - higher for cleaner images, lower for noisy ones
        noise_level = cv2.Laplacian(processed, cv2.CV_64F).var()
        if noise_level < 500:  # Clean image
            param2 = max(35, param2)
        else:  # Noisy image
            param2 = min(35, param2)
    
    print(f"Circle detection params - dp: {dp}, min_dist: {min_dist}, param1: {param1}, param2: {param2}")
    
    # Only try one set of parameters with higher thresholds
    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    
    circles_list = []
    
    if circles is not None:
        # Convert to integer coordinates
        circles = np.round(circles[0, :]).astype(int)
        
        for c in circles:
            x, y, r = c[0], c[1], c[2]
            
            # Basic validation
            if r < min_radius or r > max_radius:
                continue
            
            # Check if circle is within image bounds (with margin)
            margin = 5
            if (x - r < margin or x + r >= w - margin or 
                y - r < margin or y + r >= h - margin):
                continue
            
            # Additional validation: check edge consistency
            # Create a circular mask
            mask = np.zeros_like(processed, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 2)
            
            # Get edges
            edges = cv2.Canny(processed, 50, 150)
            
            # Count edge pixels on the circle
            circle_edges = cv2.bitwise_and(edges, mask)
            edge_count = np.count_nonzero(circle_edges)
            
            # Calculate expected edge pixels (40% of circumference)
            expected_edges = int(2 * np.pi * r * 0.4)
            
            # Also check internal consistency: the circle should have some edges
            # and shouldn't be too empty
            if edge_count < expected_edges:
                # Try with a thinner mask (1 pixel)
                mask_thin = np.zeros_like(processed, dtype=np.uint8)
                cv2.circle(mask_thin, (x, y), r, 255, 1)
                circle_edges_thin = cv2.bitwise_and(edges, mask_thin)
                edge_count_thin = np.count_nonzero(circle_edges_thin)
                
                if edge_count_thin < int(2 * np.pi * r * 0.25):  # 25% of circumference
                    continue
            
            # Check if the interior is relatively uniform (not too many edges)
            interior_mask = np.zeros_like(processed, dtype=np.uint8)
            cv2.circle(interior_mask, (x, y), r-2, 255, -1)  # Fill interior
            interior_edges = cv2.bitwise_and(edges, interior_mask)
            interior_edge_count = np.count_nonzero(interior_edges)
            
            # If there are too many edges inside, it might be a textured region, not a circle
            interior_area = np.pi * (r-2) ** 2
            if interior_edge_count > interior_area * 0.3:  # More than 30% of interior is edges
                continue
            
            circles_list.append((x, y, r))
    
    # Remove duplicates (circles that are too close)
    if len(circles_list) > 1:
        filtered_circles = []
        # Sort by radius (largest first) - larger circles are usually more reliable
        circles_list.sort(key=lambda c: c[2], reverse=True)
        
        for c in circles_list:
            x, y, r = c
            duplicate = False
            
            for fc in filtered_circles:
                fx, fy, fr = fc
                # Calculate distance between centers
                dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                
                # If centers are very close, consider it duplicate
                if dist < min(r, fr) * 0.3:  # Stricter threshold
                    duplicate = True
                    break
            
            if not duplicate:
                filtered_circles.append(c)
        
        circles_list = filtered_circles
    
    # Additional filtering: if we still have too many circles, keep only the most confident ones
    if len(circles_list) > 10:
        # Sort by radius consistency (circles with good edge support)
        scored_circles = []
        edges = cv2.Canny(processed, 50, 150)
        
        for c in circles_list:
            x, y, r = c
            # Create thin mask
            mask = np.zeros_like(processed, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 1)
            
            # Count edge pixels
            circle_edges = cv2.bitwise_and(edges, mask)
            edge_count = np.count_nonzero(circle_edges)
            
            # Score based on edge coverage
            expected = int(2 * np.pi * r)
            score = edge_count / expected if expected > 0 else 0
            
            scored_circles.append((c, score))
        
        # Sort by score and keep top 5
        scored_circles.sort(key=lambda x: x[1], reverse=True)
        circles_list = [c[0] for c in scored_circles[:5]]
    
    print(f"Found {len(circles_list)} circles after filtering")
    return circles_list


def detect_circles_multiscale(
    gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 100,
    step: int = 20,
) -> list:
    """
    Detect circles using multi-scale approach.
    Only use this when you expect circles of very different sizes.
    """
    all_circles = []
    
    # Try different radius ranges with higher thresholds
    for r_min in range(min_radius, max_radius, step):
        r_max = min(r_min + step, max_radius)
        
        circles = detect_circles(
            gray,
            min_radius=r_min,
            max_radius=r_max,
            param1=80,
            param2=35,  # Higher threshold
            use_adaptive_params=True
        )
        
        all_circles.extend(circles)
    
    # Remove duplicates
    unique_circles = []
    for c in all_circles:
        x, y, r = c
        duplicate = False
        
        for uc in unique_circles:
            ux, uy, ur = uc
            dist = np.sqrt((x - ux)**2 + (y - uy)**2)
            
            if dist < min(r, ur) * 0.25:  # Stricter duplicate removal
                duplicate = True
                break
        
        if not duplicate:
            unique_circles.append(c)
    
    return unique_circles


def draw_circles_debug(img: np.ndarray, circles: list) -> np.ndarray:
    """Draw circles with different colors based on confidence (for debugging)."""
    result = img.copy()
    
    for i, (x, y, r) in enumerate(circles):
        # Color based on circle size
        if r > 50:
            color = (0, 0, 255)  # Red for large circles
        elif r > 20:
            color = (0, 255, 0)  # Green for medium circles
        else:
            color = (255, 0, 0)  # Blue for small circles
        
        cv2.circle(result, (x, y), r, color, 2)
        cv2.circle(result, (x, y), 2, (255, 255, 255), -1)
        
        # Add label
        cv2.putText(result, f"{i+1}", (x-20, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result