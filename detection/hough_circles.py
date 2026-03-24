import numpy as np
import cv2


def preprocess_for_circles(gray: np.ndarray) -> np.ndarray:
    """Preprocess grayscale image for better circle detection."""
    return cv2.GaussianBlur(gray, (5, 5), 1.0)


# ---------------------------------------------------------------------------
# Hough Circle Transform — from scratch
# ---------------------------------------------------------------------------

def hough_circles_from_scratch(
    gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 100,
    threshold: int = 40,
    min_dist: int = 30,
) -> list:
    """
    Hough Circle Transform implemented from scratch.

    How it works
    ------------
    For a circle with center (cx, cy) and radius r, every edge point (x, y)
    on its circumference satisfies:

        (x - cx)² + (y - cy)² = r²

    So for each candidate radius r and each edge point (x, y), the set of
    possible centers forms a circle of radius r around (x, y).  We vote for
    every (cx, cy) on that locus.  Centers that collect many votes (≥ threshold)
    across many edge points are real circles.

    To make this tractable we use the gradient direction at each edge point to
    restrict the center locus to two candidate points (one on each side along
    the gradient), instead of voting for a full circle of candidates.  This is
    the same trick OpenCV's HOUGH_GRADIENT uses and keeps complexity O(N·R)
    instead of O(N·R·2πR).

    Parameters
    ----------
    gray       : Grayscale uint8 image (already preprocessed / blurred).
    min_radius : Smallest circle radius to search for (pixels).
    max_radius : Largest circle radius to search for (pixels).
    threshold  : Minimum accumulator votes for a circle center to be accepted.
    min_dist   : Minimum pixel distance between accepted circle centers.

    Returns
    -------
    List of (cx, cy, radius) tuples sorted by descending vote count.
    """
    if gray is None or gray.size == 0:
        return []
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape

    # -- Step 1: Canny edges + gradient direction ----------------------------
    edges = cv2.Canny(gray, 50, 150)

    # Sobel gradients for direction
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Edge pixel coordinates (y, x order from np.where)
    edge_ys, edge_xs = np.where(edges > 0)
    if len(edge_xs) == 0:
        return []

    # Gradient direction at each edge point (unit vector)
    gx_e = gx[edge_ys, edge_xs]
    gy_e = gy[edge_ys, edge_xs]
    mag = np.sqrt(gx_e ** 2 + gy_e ** 2)
    # Avoid division by zero for zero-gradient pixels
    valid = mag > 1e-6
    gx_e = np.where(valid, gx_e / mag, 0.0)
    gy_e = np.where(valid, gy_e / mag, 0.0)

    # -- Step 2: 2-D center accumulator (one per candidate radius) -----------
    # We maintain a single (h, w) accumulator and sweep over radii, keeping
    # a running best-radius map so we don't need a full 3-D array in memory.
    best_votes  = np.zeros((h, w), dtype=np.int32)
    best_radius = np.zeros((h, w), dtype=np.int32)

    radius_range = range(min_radius, max_radius + 1)

    for r in radius_range:
        accum = np.zeros((h, w), dtype=np.int32)

        # For each edge point, vote for two candidate centers:
        #   center = edge_point ± r * gradient_direction
        # (one center on each side of the edge along the gradient)
        for sign in (+1, -1):
            cx_f = edge_xs + sign * r * gx_e
            cy_f = edge_ys + sign * r * gy_e

            # Round to nearest pixel and clip to image bounds
            cx_i = np.round(cx_f).astype(np.int32)
            cy_i = np.round(cy_f).astype(np.int32)

            in_bounds = (cx_i >= 0) & (cx_i < w) & (cy_i >= 0) & (cy_i < h)
            cx_i = cx_i[in_bounds]
            cy_i = cy_i[in_bounds]

            np.add.at(accum, (cy_i, cx_i), 1)

        # Update best-votes map where this radius beats the previous best
        improved = accum > best_votes
        best_votes[improved]  = accum[improved]
        best_radius[improved] = r

    # -- Step 3: Extract peaks from the center accumulator -------------------
    circles = []
    # Work on a copy so we can suppress already-found peaks
    votes_map = best_votes.copy()

    candidate_ys, candidate_xs = np.where(votes_map >= threshold)
    if len(candidate_ys) == 0:
        return []

    # Sort candidates by descending vote count (strongest first)
    scores = votes_map[candidate_ys, candidate_xs]
    order  = np.argsort(-scores)
    candidate_ys = candidate_ys[order]
    candidate_xs = candidate_xs[order]
    scores       = scores[order]

    for cy, cx, score in zip(candidate_ys, candidate_xs, scores):
        if votes_map[cy, cx] < threshold:
            # Already suppressed by a nearby stronger peak
            continue

        r = int(best_radius[cy, cx])
        circles.append((int(cx), int(cy), r, int(score)))

        # Non-maximum suppression: zero out a window around this center
        # so nearby weaker candidates don't produce duplicate detections.
        suppress_r = max(min_dist // 2, r // 2, 5)
        y0 = max(0, cy - suppress_r)
        y1 = min(h, cy + suppress_r + 1)
        x0 = max(0, cx - suppress_r)
        x1 = min(w, cx + suppress_r + 1)
        votes_map[y0:y1, x0:x1] = 0

    # Return as (cx, cy, radius) without the internal score
    return [(cx, cy, r) for cx, cy, r, _ in circles]


# ---------------------------------------------------------------------------
# Edge-consistency validation (shared by both implementations)
# ---------------------------------------------------------------------------

def _validate_circle(processed: np.ndarray, edges: np.ndarray,
                     x: int, y: int, r: int,
                     min_radius: int, max_radius: int) -> bool:
    """
    Return True if (x, y, r) passes edge-consistency checks.

    Checks
    ------
    1. Radius within [min_radius, max_radius].
    2. Circle fits inside the image with a small margin.
    3. At least 25–40 % of the expected circumference has edge pixels.
    4. Interior doesn't contain more than 30 % edge pixels (rejects
       textured regions mis-detected as circles).
    """
    h, w = processed.shape

    if r < min_radius or r > max_radius:
        return False

    margin = 5
    if (x - r < margin or x + r >= w - margin or
            y - r < margin or y + r >= h - margin):
        return False

    # Thick ring mask (2 px) for primary check
    mask = np.zeros_like(processed, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, 2)
    edge_count = np.count_nonzero(cv2.bitwise_and(edges, mask))
    expected_thick = int(2 * np.pi * r * 0.4)

    if edge_count < expected_thick:
        # Retry with 1-px mask and a more lenient threshold
        mask_thin = np.zeros_like(processed, dtype=np.uint8)
        cv2.circle(mask_thin, (x, y), r, 255, 1)
        edge_count_thin = np.count_nonzero(cv2.bitwise_and(edges, mask_thin))
        if edge_count_thin < int(2 * np.pi * r * 0.25):
            return False

    # Interior edge density check
    interior_mask = np.zeros_like(processed, dtype=np.uint8)
    cv2.circle(interior_mask, (x, y), max(r - 2, 1), 255, -1)
    interior_edge_count = np.count_nonzero(cv2.bitwise_and(edges, interior_mask))
    interior_area = np.pi * max(r - 2, 1) ** 2
    if interior_edge_count > interior_area * 0.3:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API — mirrors the original detect_circles signature
# ---------------------------------------------------------------------------

def detect_circles(
    gray: np.ndarray,
    dp: float = 1.2,
    min_dist: int = 30,
    param1: int = 80,
    param2: int = 40,
    min_radius: int = 10,
    max_radius: int = 0,
    use_adaptive_params: bool = True,
) -> list:
    """
    Detect circles using the from-scratch Hough Circle Transform.

    Parameters
    ----------
    gray       : Grayscale image (uint8).
    dp         : Kept for API compatibility; not used in from-scratch impl.
    min_dist   : Minimum distance between circle centers.
    param1     : Canny high threshold.
    param2     : Accumulator vote threshold.
    min_radius : Minimum circle radius in pixels.
    max_radius : Maximum circle radius (0 = auto: half the shorter dimension).
    use_adaptive_params : Adapt param1/param2 based on image statistics.

    Returns
    -------
    List of (cx, cy, radius) tuples.
    """
    img = gray.copy()
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    processed = preprocess_for_circles(img)
    h, w = processed.shape

    if min_dist == 0:
        min_dist = min(h, w) // 10
    if max_radius == 0:
        max_radius = min(h, w) // 2

    # -- Adaptive parameter adjustment (same logic as before) ----------------
    if use_adaptive_params:
        std_intensity = np.std(processed)
        if std_intensity > 60:
            param1 = min(150, param1 + 20)
        elif std_intensity < 30:
            param1 = max(50, param1 - 10)

        noise_level = cv2.Laplacian(processed, cv2.CV_64F).var()
        if noise_level < 500:
            param2 = min(35, param2)
        else:
            param2 = max(35, param2)

    print(
        f"Circle detection params - dp: {dp}, min_dist: {min_dist}, "
        f"param1: {param1}, param2: {param2}"
    )

    # -- Detection -----------------------------------------------------------
    raw_circles = hough_circles_from_scratch(
            processed,
            min_radius=min_radius,
            max_radius=max_radius,
            threshold=param2,
            min_dist=min_dist,
        )

    # -- Edge-consistency validation -----------------------------------------
    edges = cv2.Canny(processed, 50, 150)
    circles_list = []
    for x, y, r in raw_circles:
        if _validate_circle(processed, edges, x, y, r, min_radius, max_radius):
            circles_list.append((x, y, r))

    # -- Deduplication -------------------------------------------------------
    if len(circles_list) > 1:
        circles_list.sort(key=lambda c: c[2], reverse=True)
        filtered = []
        for c in circles_list:
            x, y, r = c
            if not any(
                np.sqrt((x - fx) ** 2 + (y - fy) ** 2) < min(r, fr) * 0.3
                for fx, fy, fr in filtered
            ):
                filtered.append(c)
        circles_list = filtered

    # -- Keep top 5 by edge-coverage score if still too many -----------------
    if len(circles_list) > 10:
        scored = []
        for x, y, r in circles_list:
            mask = np.zeros_like(processed, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, 1)
            edge_count = np.count_nonzero(cv2.bitwise_and(edges, mask))
            expected   = int(2 * np.pi * r)
            score = edge_count / expected if expected > 0 else 0
            scored.append(((x, y, r), score))
        scored.sort(key=lambda s: s[1], reverse=True)
        circles_list = [s[0] for s in scored[:5]]

    print(f"Found {len(circles_list)} circles after filtering")
    return circles_list


def detect_circles_multiscale(
    gray: np.ndarray,
    min_radius: int = 10,
    max_radius: int = 100,
    step: int = 20,
) -> list:
    """
    Detect circles using a multi-scale approach (sweep over radius bands).
    Useful when circles of very different sizes are expected.
    """
    all_circles = []

    for r_min in range(min_radius, max_radius, step):
        r_max = min(r_min + step, max_radius)
        circles = detect_circles(
            gray,
            min_radius=r_min,
            max_radius=r_max,
            param1=80,
            param2=35,
            use_adaptive_params=True,
        )
        all_circles.extend(circles)

    # Deduplicate across radius bands
    unique = []
    for c in all_circles:
        x, y, r = c
        if not any(
            np.sqrt((x - ux) ** 2 + (y - uy) ** 2) < min(r, ur) * 0.25
            for ux, uy, ur in unique
        ):
            unique.append(c)

    return unique


def draw_circles_debug(img: np.ndarray, circles: list) -> np.ndarray:
    """Draw circles with size-based colors for debugging."""
    result = img.copy()
    for i, (x, y, r) in enumerate(circles):
        color = (0, 0, 255) if r > 50 else (0, 255, 0) if r > 20 else (255, 0, 0)
        cv2.circle(result, (x, y), r, color, 2)
        cv2.circle(result, (x, y), 2, (255, 255, 255), -1)
        cv2.putText(result, f"{i+1}", (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result
