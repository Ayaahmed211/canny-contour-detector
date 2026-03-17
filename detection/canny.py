import numpy as np
import cv2

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a Gaussian kernel."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution (no scipy)."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(ih):
        for j in range(iw):
            output[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return output

def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Apply Gaussian blur to image."""
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(image.astype(np.float64), kernel)

def sobel_gradients(image: np.ndarray):
    """Compute gradient magnitude and direction using Sobel operators."""
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)
    
    magnitude = np.hypot(Gx, Gy)
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin edges by suppressing non-maximum gradient pixels."""
    H, W = magnitude.shape
    suppressed = np.zeros((H, W), dtype=np.float64)
    angle = np.degrees(direction) % 180  # map to [0, 180)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            ang = angle[i, j]
            mag = magnitude[i, j]

            # Check neighbors in gradient direction
            if (0 <= ang < 22.5) or (157.5 <= ang < 180):
                # Horizontal direction
                p, r = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= ang < 67.5:
                # Diagonal direction
                p, r = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= ang < 112.5:
                # Vertical direction
                p, r = magnitude[i-1, j], magnitude[i+1, j]
            else:
                # Anti-diagonal direction
                p, r = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if mag >= p and mag >= r:
                suppressed[i, j] = mag

    return suppressed

def double_threshold(image: np.ndarray, low_threshold: float, high_threshold: float):
    """Apply double thresholding to classify strong/weak edges."""
    strong, weak = 255, 75
    result = np.zeros_like(image, dtype=np.uint8)
    result[image >= high_threshold] = strong
    result[(image >= low_threshold) & (image < high_threshold)] = weak
    return result, strong, weak

def hysteresis(image: np.ndarray, strong: int = 255, weak: int = 75) -> np.ndarray:
    """Edge tracking by hysteresis — keep weak edges connected to strong ones."""
    H, W = image.shape
    output = image.copy()
    
    # Multiple passes to ensure connectivity
    for _ in range(5):  # Iterate multiple times
        changed = False
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if output[i, j] == weak:
                    # Check 8-neighborhood
                    neighborhood = output[i-1:i+2, j-1:j+2]
                    if strong in neighborhood:
                        output[i, j] = strong
                        changed = True
        
        if not changed:
            break
    
    # Remove remaining weak edges
    output[output == weak] = 0
    
    return output

def canny_from_scratch(
    image: np.ndarray,
    blur_size: int = 5,
    sigma: float = 1.4,
    low_threshold: float = 20,
    high_threshold: float = 50,
) -> np.ndarray:
    """
    Full Canny edge detector implemented from scratch.
    
    Args:
        image: Input image (grayscale or BGR)
        blur_size: Size of Gaussian kernel
        sigma: Gaussian sigma
        low_threshold: Low threshold for edge tracking
        high_threshold: High threshold for edge tracking
    
    Returns:
        Binary edge image
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure image is float64 for processing
    gray = gray.astype(np.float64)
    
    # Step 1: Gaussian blur
    blurred = gaussian_blur(gray, size=blur_size, sigma=sigma)
    
    # Step 2: Gradient calculation
    magnitude, direction = sobel_gradients(blurred)
    
    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction)
    
    # Step 4: Double threshold
    thresholded, strong, weak = double_threshold(suppressed, low_threshold, high_threshold)
    
    # Step 5: Hysteresis
    edges = hysteresis(thresholded, strong, weak)
    
    return edges.astype(np.uint8)