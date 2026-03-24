import numpy as np
import cv2
import math
from collections import defaultdict



def hough_lines_p_from_scratch(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 80,
    min_line_length: int = 30,
    max_line_gap: int = 10,
):
    """
    Probabilistic Hough Transform from scratch.

    Args:
        edges: Binary edge image
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Accumulator threshold
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments in pixels

    Returns:
        List of (x1, y1, x2, y2) tuples
    """
    if edges is None or edges.size == 0:
        return []

    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)

    edge_points = np.column_stack(np.where(edges > 0))
    if len(edge_points) == 0:
        return []

    height, width = edges.shape
    max_dist = int(np.sqrt(height ** 2 + width ** 2))

    theta_values = np.arange(0, np.pi, theta)
    n_thetas = len(theta_values)

    n_rhos = int(2 * max_dist / rho) + 1
    rho_offset = max_dist

    accumulator = np.zeros((n_rhos, n_thetas), dtype=np.int32)

    # --- BUG FIX 5 (was Bug 7) ---
    # Was: randomly sampling up to 50k points for the accumulator but then
    # scanning ALL edge_points when collecting segment points. This means the
    # peaks found reflect only a fraction of the image, but the segment-
    # collection pass sees the full set — the peaks rarely match the densest
    # real edges and proximity checks fail silently.
    # Fix: build the accumulator from ALL edge points (vectorised so it's
    # fast), and only use sampling as a last resort on very large images.
    # Vectorised accumulation: compute all rho values at once with matrix ops.
    ys = edge_points[:, 0].astype(np.float32)
    xs = edge_points[:, 1].astype(np.float32)
    cos_vals = np.cos(theta_values).astype(np.float32)
    sin_vals = np.sin(theta_values).astype(np.float32)

    # rho_matrix shape: (n_points, n_thetas)
    rho_matrix = np.outer(xs, cos_vals) + np.outer(ys, sin_vals)

    # --- BUG FIX 6 (was Bug 5) ---
    # Was: int(rho_val / rho) — truncation without rounding shifts every bin
    # downward by up to (rho-1) pixels, scattering votes across wrong bins.
    # Fix: round before converting to int so each vote lands in the correct bin.
    rho_indices = (np.round(rho_matrix / rho) + rho_offset).astype(np.int32)

    valid_mask = (rho_indices >= 0) & (rho_indices < n_rhos)

    for theta_idx in range(n_thetas):
        valid_rho_idxs = rho_indices[valid_mask[:, theta_idx], theta_idx]
        np.add.at(accumulator, (valid_rho_idxs, theta_idx), 1)

    # --- BUG FIX 1 ---
    # Was: iterating from `threshold` upward, finding WEAK peaks first.
    # Fix: collect ALL positions above threshold, then sort by vote count
    # descending so the strongest lines are found first and block weaker
    # duplicates — not the other way around.
    peaks = []
    min_dist_between_peaks = 10

    candidate_positions = np.argwhere(accumulator >= threshold)
    if len(candidate_positions) == 0:
        return []

    # Sort by descending accumulator value so strong peaks take priority
    candidate_scores = accumulator[candidate_positions[:, 0], candidate_positions[:, 1]]
    sorted_order = np.argsort(-candidate_scores)
    candidate_positions = candidate_positions[sorted_order]

    for pos in candidate_positions:
        rho_idx, theta_idx = pos
        rho_val = (rho_idx - rho_offset) * rho
        theta_val = theta_values[theta_idx]

        is_new_peak = True
        for existing_rho, existing_theta in peaks:
            rho_dist = abs(existing_rho - rho_val)
            theta_dist = min(
                abs(existing_theta - theta_val),
                np.pi - abs(existing_theta - theta_val),
            )
            if rho_dist < min_dist_between_peaks and theta_dist < np.pi / 18:
                is_new_peak = False
                break

        if is_new_peak:
            peaks.append((rho_val, theta_val))

        if len(peaks) > 50:
            break

    lines = []

    for rho_val, theta_val in peaks:
        points_on_line = []

        # --- BUG FIX 2 ---
        # Was: `line_dist < rho / 2` — with default rho=1 this is 0.5px,
        # meaning almost no edge point ever qualifies.
        # Fix: use a sensible pixel tolerance (1.5px covers sub-pixel
        # quantisation without pulling in noise from adjacent edges).
        proximity_threshold = max(2.0, rho)

        for y, x in edge_points:
            line_dist = abs(x * np.cos(theta_val) + y * np.sin(theta_val) - rho_val)
            if line_dist < proximity_threshold:
                points_on_line.append((x, y))

        if len(points_on_line) < 2:
            continue

        a = np.cos(theta_val)
        b = np.sin(theta_val)
        # Line direction is perpendicular to rho direction:
        # rho direction = (cos θ, sin θ), line direction = (-sin θ, cos θ).
        # Old code projected onto the rho direction, giving near-zero span
        # for diagonals (a 45° line spanning 254px measured ~0px). Fixed.
        dl_x = -b   # -sin θ
        dl_y =  a   #  cos θ

        projections = []
        for x, y in points_on_line:
            t = x * dl_x + y * dl_y
            projections.append((t, x, y))

        projections.sort(key=lambda p: p[0])

        # --- BUG FIX 3 ---
        # Was: `t - last_t <= max_line_gap` kept EXTENDING the segment
        # even when the gap exceeded max_line_gap (condition inverted).
        # Fix: start a new segment when the gap EXCEEDS max_line_gap.
        segments = []
        current_segment = []
        last_t = None

        for t, x, y in projections:
            if last_t is None:
                current_segment = [(x, y)]
            elif t - last_t > max_line_gap:   # gap too large → new segment
                if current_segment:
                    segments.append(current_segment)
                current_segment = [(x, y)]
            else:
                current_segment.append((x, y))

            last_t = t

        if current_segment:
            segments.append(current_segment)

        # --- BUG FIX 4 ---
        # Was: `len(segment) >= min_line_length` — compared point *count*
        # to a pixel-length threshold.  A dense 5-pixel cluster with 40
        # points would pass; a clean 200-pixel line with 28 points would
        # fail.
        # Fix: measure actual pixel span (max_proj − min_proj) along the
        # line direction and compare that to min_line_length.
        for segment in segments:
            if len(segment) < 2:
                continue

            min_proj = float("inf")
            max_proj = float("-inf")
            min_point = None
            max_point = None

            for x, y in segment:
                proj = x * dl_x + y * dl_y   # same line-direction projection
                if proj < min_proj:
                    min_proj = proj
                    min_point = (x, y)
                if proj > max_proj:
                    max_proj = proj
                    max_point = (x, y)

            # Pixel length of this segment along the line direction
            segment_pixel_length = max_proj - min_proj

            if segment_pixel_length >= min_line_length and min_point and max_point:
                lines.append(
                    (min_point[0], min_point[1], max_point[0], max_point[1])
                )

    return lines


def detect_lines(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 80,
    min_line_length: int = 30,
    max_line_gap: int = 10,
):
    """
    Detect lines in an edge image using the from-scratch Probabilistic
    Hough Transform.

    Args:
        edges: Binary edge image
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Accumulator threshold
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments in pixels

    Returns:
        List of (x1, y1, x2, y2) tuples
    """
    return hough_lines_p_from_scratch(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )


def merge_nearby_lines(lines, distance_threshold=60, angle_threshold=np.pi / 18):
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

    line_params = []
    for line in lines:
        x1, y1, x2, y2 = line
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if x2 - x1 == 0:
            angle = np.pi / 2
        else:
            angle = np.arctan2(y2 - y1, x2 - x1)

        line_params.append(
            {
                "coords": line,
                "center": (center_x, center_y),
                "angle": angle,
                "length": np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
            }
        )

    merged = []
    used = [False] * len(line_params)

    for i, params in enumerate(line_params):
        if used[i]:
            continue

        group = [params]
        used[i] = True

        for j in range(i + 1, len(line_params)):
            if used[j]:
                continue

            other = line_params[j]

            dist = np.sqrt(
                (params["center"][0] - other["center"][0]) ** 2
                + (params["center"][1] - other["center"][1]) ** 2
            )

            angle_diff = abs(params["angle"] - other["angle"])
            angle_diff = min(angle_diff, np.pi - angle_diff)

            if dist < distance_threshold and angle_diff < angle_threshold:
                group.append(other)
                used[j] = True

        if len(group) == 1:
            merged.append(group[0]["coords"])
        else:
            longest = max(group, key=lambda x: x["length"])
            merged.append(longest["coords"])

    return merged
