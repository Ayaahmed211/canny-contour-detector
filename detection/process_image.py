import base64
import numpy as np
import cv2

from .canny import canny_from_scratch
from .hough_lines import detect_lines, merge_nearby_lines
from .hough_circles import detect_circles, detect_circles_multiscale
from .ellipse_detection import detect_ellipses


def encode_image(img: np.ndarray) -> str:
    """Encode image to base64 data URL."""
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode image from bytes."""
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def dark_bg(img: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Create a darkened version of the image for overlay."""
    return (img.astype(np.float32) * alpha).astype(np.uint8)


def glow_line(canvas, p1, p2, color, core_t=2, glow_t=6, glow_alpha=0.3):
    """Draw a line with glow effect."""
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, color, glow_t)
    cv2.addWeighted(overlay, glow_alpha, canvas, 1 - glow_alpha, 0, canvas)
    cv2.line(canvas, p1, p2, color, core_t)


def glow_circle(canvas, center, radius, color, core_t=2, glow_t=7, glow_alpha=0.25):
    """Draw a circle with glow effect."""
    overlay = canvas.copy()
    cv2.circle(overlay, center, radius, color, glow_t)
    cv2.addWeighted(overlay, glow_alpha, canvas, 1 - glow_alpha, 0, canvas)
    cv2.circle(canvas, center, radius, color, core_t)
    cv2.circle(canvas, center, 2, (255, 255, 255), -1)


def glow_ellipse(canvas, ellipse, color, core_t=2, glow_t=7, glow_alpha=0.25):
    """Draw an ellipse with glow effect."""
    overlay = canvas.copy()
    cv2.ellipse(overlay, ellipse, color, glow_t)
    cv2.addWeighted(overlay, glow_alpha, canvas, 1 - glow_alpha, 0, canvas)
    cv2.ellipse(canvas, ellipse, color, core_t)
    cv2.circle(canvas, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (255, 255, 255), -1)


def preprocess_image(img: np.ndarray) -> tuple:
    """
    Preprocess image for better edge detection.

    Returns:
        tuple: (original, grayscale, enhanced, color_enhanced)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    color_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return img, gray, enhanced, color_enhanced


def detect_shapes_with_multiple_methods(
    edges: np.ndarray,
    gray: np.ndarray,
    enhanced: np.ndarray,
    lines_params: dict,
    circles_params: dict,
    ellipses_params: dict,
) -> tuple:
    """
    Detect shapes using multiple methods and parameters for better results.

    - Lines  : single-pass from-scratch Hough Transform.
    - Circles: multi-attempt + multiscale + deduplication, from-scratch.
    - Ellipses: single clean RHT call.
    """
    lines = []
    circles = []

    # ── Lines ────────────────────────────────────────────────────────────────
    line_attempts = [
        lines_params,
        {"threshold": 20, "min_line_length": 15, "max_line_gap": 20},
    ]

    for params in line_attempts:
        detected = detect_lines(
            edges,
            threshold=params.get("threshold", lines_params.get("threshold", 30)),
            min_line_length=params.get("min_line_length", lines_params.get("min_line_length", 20)),
            max_line_gap=params.get("max_line_gap", lines_params.get("max_line_gap", 15)),
        )
        lines.extend(detected)
        if lines:
            print(
                f"Lines detected with params: threshold={params.get('threshold')}, "
                f"min_length={params.get('min_line_length')}"
            )
            break

    # ── Circles ──────────────────────────────────────────────────────────────
    # Method 1: standard detection on gray
    circles1 = detect_circles(
        gray,
        dp=circles_params.get("dp", 1.2),
        min_dist=circles_params.get("min_dist", 20),
        param1=circles_params.get("param1", 50),
        param2=circles_params.get("param2", 25),
        min_radius=circles_params.get("min_radius", 5),
        max_radius=circles_params.get("max_radius", 0),
        use_adaptive_params=True,
    )
    circles.extend(circles1)

    # Method 2: multiscale if too few found
    if len(circles) < 3:
        circles2 = detect_circles_multiscale(
            gray,
            min_radius=5,
            max_radius=100,
            step=10,
        )
        circles.extend(circles2)

    # Method 3: retry on enhanced image if still nothing
    if not circles:
        circles3 = detect_circles(
            enhanced,
            dp=1.1,
            min_dist=15,
            param1=40,
            param2=20,
            min_radius=3,
            max_radius=0,
            use_adaptive_params=True,
        )
        circles.extend(circles3)

    # Deduplication across methods (keep largest when centers overlap)
    if len(circles) > 1:
        circles.sort(key=lambda c: c[2], reverse=True)
        unique_circles = []
        for c in circles:
            x, y, r = c
            if not any(
                np.sqrt((x - ux) ** 2 + (y - uy) ** 2) < min(r, ur) * 0.5
                for ux, uy, ur in unique_circles
            ):
                unique_circles.append(c)
        circles = unique_circles

    # ── Ellipses via RHT ─────────────────────────────────────────────────────
    h, w = edges.shape[:2]
    diag = int(np.sqrt(h ** 2 + w ** 2))

    ellipses = detect_ellipses(
        edges,
        max_iter=ellipses_params.get("max_iter", 800),
        major_bound=ellipses_params.get("major_bound", [10, diag]),
        minor_bound=ellipses_params.get("minor_bound", [10, diag]),
        flattening_bound=ellipses_params.get("flattening_bound", 0.9),
        score_threshold=ellipses_params.get("score_threshold", 5),
        min_area=ellipses_params.get("min_area", 50.0),
        max_area=ellipses_params.get("max_area", None),
        min_aspect_ratio=ellipses_params.get("min_aspect_ratio", 0.2),
        max_aspect_ratio=ellipses_params.get("max_aspect_ratio", 0.99),
    )

    print(
        f"Detection results - lines: {len(lines)}, "
        f"circles: {len(circles)}, ellipses: {len(ellipses)}"
    )

    return lines, circles, ellipses


def process_image(
    file_bytes: bytes,
    # ── Canny ─────────────────────────────────────────────────────────────
    canny_blur_size: int = 5,
    canny_sigma: float = 1.4,
    canny_low_ratio: float = 0.03,
    canny_high_ratio: float = 0.10,
    # ── Lines ─────────────────────────────────────────────────────────────
    lines_threshold: int = 30,
    lines_min_length: int = 20,
    lines_max_gap: int = 15,
    # ── Circles ───────────────────────────────────────────────────────────
    circles_dp: float = 1.1,
    circles_min_dist: int = 15,
    circles_param1: int = 40,
    circles_param2: int = 20,
    circles_min_r: int = 3,
    circles_max_r: int = 0,
    circles_use_multiscale: bool = True,
    # ── RHT Ellipse ───────────────────────────────────────────────────────
    ellipse_max_iter: int = 800,
    ellipse_major_bound_min: int = 10,
    ellipse_major_bound_max: int = 0,
    ellipse_minor_bound_min: int = 10,
    ellipse_minor_bound_max: int = 0,
    ellipse_flattening_bound: float = 0.9,
    ellipse_score_threshold: int = 5,
    ellipse_min_area: float = 50.0,
    ellipse_min_aspect_ratio: float = 0.2,
    ellipse_max_aspect_ratio: float = 0.99,
) -> dict:
    """Process image to detect edges, lines, circles, and ellipses."""

    # ── Decode & preprocess ───────────────────────────────────────────────────
    img = decode_image(file_bytes)
    if img is None:
        raise ValueError("Could not decode image.")

    original, gray, enhanced, color_enhanced = preprocess_image(img)

    # ── Canny edge detection ──────────────────────────────────────────────────
    img_max = enhanced.max() if enhanced.max() > 0 else 255
    low_threshold = max(5, min(40, img_max * canny_low_ratio))
    high_threshold = max(15, min(80, img_max * canny_high_ratio))

    print(f"Image stats - shape: {original.shape}, max: {img_max:.1f}")
    print(f"Canny thresholds - low: {low_threshold:.1f}, high: {high_threshold:.1f}")

    edges = canny_from_scratch(
        enhanced,
        blur_size=canny_blur_size,
        sigma=canny_sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    edges_cv2 = cv2.Canny(enhanced, int(low_threshold), int(high_threshold))
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ── Build param dicts ─────────────────────────────────────────────────────
    h, w = original.shape[:2]
    diag = int(np.sqrt(h ** 2 + w ** 2))

    lines_params = {
        "threshold": lines_threshold,
        "min_line_length": lines_min_length,
        "max_line_gap": lines_max_gap,
    }

    circles_params = {
        "dp": circles_dp,
        "min_dist": circles_min_dist,
        "param1": circles_param1,
        "param2": circles_param2,
        "min_radius": circles_min_r,
        "max_radius": circles_max_r,
    }

    ellipses_params = {
        "max_iter": ellipse_max_iter,
        "major_bound": [
            ellipse_major_bound_min,
            ellipse_major_bound_max if ellipse_major_bound_max > 0 else diag,
        ],
        "minor_bound": [
            ellipse_minor_bound_min,
            ellipse_minor_bound_max if ellipse_minor_bound_max > 0 else diag,
        ],
        "flattening_bound": ellipse_flattening_bound,
        "score_threshold": ellipse_score_threshold,
        "min_area": ellipse_min_area,
        "min_aspect_ratio": ellipse_min_aspect_ratio,
        "max_aspect_ratio": ellipse_max_aspect_ratio,
    }

    # ── Shape detection ───────────────────────────────────────────────────────
    lines, circles, ellipses = detect_shapes_with_multiple_methods(
        edges, gray, enhanced, lines_params, circles_params, ellipses_params,
    )

    if len(lines) > 10:
        lines = merge_nearby_lines(lines)
        print(f"After merging: {len(lines)} lines")

    # ── Visualisation ─────────────────────────────────────────────────────────
    bg = dark_bg(original)

    lines_img = bg.copy()
    for x1, y1, x2, y2 in lines:
        if (
            0 <= x1 < original.shape[1]
            and 0 <= y1 < original.shape[0]
            and 0 <= x2 < original.shape[1]
            and 0 <= y2 < original.shape[0]
        ):
            glow_line(lines_img, (x1, y1), (x2, y2), (32, 72, 255))

    circles_img = bg.copy()
    for cx, cy, r in circles:
        if (
            0 <= cx < original.shape[1]
            and 0 <= cy < original.shape[0]
            and r > 0
            and cx - r >= 0
            and cx + r < original.shape[1]
            and cy - r >= 0
            and cy + r < original.shape[0]
        ):
            glow_circle(circles_img, (cx, cy), r, (160, 229, 0))

    ellipses_img = bg.copy()
    for e in ellipses:
        cx, cy = int(e[0][0]), int(e[0][1])
        if 0 <= cx < original.shape[1] and 0 <= cy < original.shape[0]:
            glow_ellipse(ellipses_img, e, (255, 80, 200))

    all_img = original.copy()
    for x1, y1, x2, y2 in lines:
        if (
            0 <= x1 < original.shape[1]
            and 0 <= y1 < original.shape[0]
            and 0 <= x2 < original.shape[1]
            and 0 <= y2 < original.shape[0]
        ):
            glow_line(all_img, (x1, y1), (x2, y2), (32, 72, 255))
    for cx, cy, r in circles:
        if (
            0 <= cx < original.shape[1]
            and 0 <= cy < original.shape[0]
            and r > 0
            and cx - r >= 0
            and cx + r < original.shape[1]
            and cy - r >= 0
            and cy + r < original.shape[0]
        ):
            glow_circle(all_img, (cx, cy), r, (160, 229, 0))
    for e in ellipses:
        cx, cy = int(e[0][0]), int(e[0][1])
        if 0 <= cx < original.shape[1] and 0 <= cy < original.shape[0]:
            glow_ellipse(all_img, e, (255, 80, 200))

    debug_img = original.copy()
    edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_overlay = cv2.addWeighted(edge_overlay, 0.3, debug_img, 0.7, 0)
    debug_img = cv2.addWeighted(debug_img, 0.7, edge_overlay, 0.3, 0)

    return {
        "original":               encode_image(original),
        "edges":                  encode_image(edge_vis),
        "edges_cv2":              encode_image(cv2.cvtColor(edges_cv2, cv2.COLOR_GRAY2BGR)),
        "lines_img":              encode_image(lines_img),
        "circles_img":            encode_image(circles_img),
        "ellipses_img":           encode_image(ellipses_img),
        "all_img":                encode_image(all_img),
        "debug_img":              encode_image(debug_img),
        "lines":                  len(lines),
        "circles":                len(circles),
        "ellipses":               len(ellipses),
        "image_width":            original.shape[1],
        "image_height":           original.shape[0],
    }
