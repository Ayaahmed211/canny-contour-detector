import numpy as np
import cv2

def detect_lines(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 80,
    min_line_length: int = 30,
    max_line_gap: int = 10,
    use_probabilistic: bool = True,
):
    """
    Detect lines in an edge image using Hough Transform.
    
    Args:
        edges: Binary edge image
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Accumulator threshold
        min_line_length: Minimum line length (for probabilistic)
        max_line_gap: Maximum gap between line segments (for probabilistic)
        use_probabilistic: Use Probabilistic Hough Transform (True) or Standard (False)
    
    Returns:
        List of (x1, y1, x2, y2) tuples
    """
    lines = []
    
    # Ensure edges is uint8 binary image
    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    
    if use_probabilistic:
        # Probabilistic Hough Line Transform (faster, returns line segments)
        lines_raw = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines_raw is not None:
            # Flatten from (N,1,4) → list of (x1,y1,x2,y2)
            lines = [tuple(l[0]) for l in lines_raw]
    else:
        # Standard Hough Line Transform (returns infinite lines in polar coordinates)
        lines_raw = cv2.HoughLines(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
        )
        if lines_raw is not None:
            # Convert polar coordinates to line segments that span the image
            height, width = edges.shape
            for line in lines_raw:
                rho_val, theta_val = line[0]
                
                # Calculate two points on the line
                a = np.cos(theta_val)
                b = np.sin(theta_val)
                x0 = a * rho_val
                y0 = b * rho_val
                
                # Extend the line to image boundaries
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Clip to image boundaries
                x1 = max(0, min(width, x1))
                y1 = max(0, min(height, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))
                
                lines.append((x1, y1, x2, y2))
    
    return lines


def filter_lines_by_slope(lines, min_slope=0.1, max_slope=10.0):
    """
    Filter lines by slope to remove near-vertical or near-horizontal noise.
    
    Args:
        lines: List of (x1,y1,x2,y2) line segments
        min_slope: Minimum absolute slope (0 = horizontal)
        max_slope: Maximum absolute slope (inf = vertical)
    
    Returns:
        Filtered lines
    """
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line
        
        # Avoid division by zero
        if x2 - x1 == 0:
            slope = float('inf')
        else:
            slope = abs((y2 - y1) / (x2 - x1))
        
        if min_slope <= slope <= max_slope:
            filtered.append(line)
    
    return filtered


def merge_nearby_lines(lines, distance_threshold=20, angle_threshold=np.pi/36):
    """
    Merge lines that are close in space and angle.
    
    Args:
        lines: List of (x1,y1,x2,y2) line segments
        distance_threshold: Max distance between line centers to consider merging
        angle_threshold: Max angle difference to consider merging (radians)
    
    Returns:
        Merged lines
    """
    if not lines:
        return []
    
    # Calculate line parameters
    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate angle
        if x2 - x1 == 0:
            angle = np.pi / 2
        else:
            angle = np.arctan2(y2 - y1, x2 - x1)
        
        line_params.append({
            'coords': line,
            'center': (center_x, center_y),
            'angle': angle,
            'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
        })
    
    # Group lines
    merged = []
    used = [False] * len(line_params)
    
    for i, params in enumerate(line_params):
        if used[i]:
            continue
        
        # Start a new group
        group = [params]
        used[i] = True
        
        for j in range(i + 1, len(line_params)):
            if used[j]:
                continue
            
            other = line_params[j]
            
            # Check distance between centers
            dist = np.sqrt((params['center'][0] - other['center'][0])**2 +
                          (params['center'][1] - other['center'][1])**2)
            
            # Check angle difference
            angle_diff = abs(params['angle'] - other['angle'])
            angle_diff = min(angle_diff, np.pi - angle_diff)
            
            if dist < distance_threshold and angle_diff < angle_threshold:
                group.append(other)
                used[j] = True
        
        # Merge group into a single line
        if len(group) == 1:
            merged.append(group[0]['coords'])
        else:
            # Average endpoints of the longest line in group
            longest = max(group, key=lambda x: x['length'])
            merged.append(longest['coords'])
    
    return merged