"""
Exercise Solutions for Lesson 02: Image Basics
Computer Vision - Image I/O and Pixel Operations

Topics covered:
- Image reading modes comparison
- PSNR quality analysis
- Color grid creation with ROI
- Image border addition
- Pixel-based gradient generation
"""

import numpy as np


# =============================================================================
# Exercise 1: Compare Image Reading Modes
# =============================================================================

def exercise_1_compare_reading_modes():
    """
    Simulate reading an image in three modes (COLOR, GRAYSCALE, UNCHANGED)
    and compare their shapes.

    Since we use numpy only, we create synthetic images to demonstrate
    the concept of different reading modes.
    """
    # Simulate a 200x300 BGRA image (like a PNG with alpha channel)
    h, w = 200, 300
    bgra_img = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)

    # MODE 1: COLOR (cv2.IMREAD_COLOR) - loads as 3-channel BGR, drops alpha
    color_img = bgra_img[:, :, :3].copy()
    print(f"COLOR mode shape:     {color_img.shape}  (3 channels, no alpha)")

    # MODE 2: GRAYSCALE (cv2.IMREAD_GRAYSCALE) - single channel
    # Simulating BGR to grayscale: Y = 0.114*B + 0.587*G + 0.299*R
    gray_img = (0.114 * bgra_img[:, :, 0].astype(np.float64) +
                0.587 * bgra_img[:, :, 1].astype(np.float64) +
                0.299 * bgra_img[:, :, 2].astype(np.float64))
    gray_img = gray_img.astype(np.uint8)
    print(f"GRAYSCALE mode shape: {gray_img.shape}  (single channel)")

    # MODE 3: UNCHANGED (cv2.IMREAD_UNCHANGED) - keeps all channels including alpha
    unchanged_img = bgra_img.copy()
    print(f"UNCHANGED mode shape: {unchanged_img.shape}  (4 channels, with alpha)")

    # For JPEG (no alpha), all modes would differ only in channel count
    jpeg_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    print(f"\nJPEG COLOR:     {jpeg_img.shape}")
    jpeg_gray = (0.114 * jpeg_img[:, :, 0].astype(np.float64) +
                 0.587 * jpeg_img[:, :, 1].astype(np.float64) +
                 0.299 * jpeg_img[:, :, 2].astype(np.float64)).astype(np.uint8)
    print(f"JPEG GRAYSCALE: {jpeg_gray.shape}")
    print(f"JPEG UNCHANGED: {jpeg_img.shape}  (same as COLOR for JPEG)")

    return color_img, gray_img, unchanged_img


# =============================================================================
# Exercise 2: Image Quality Analyzer (PSNR)
# =============================================================================

def exercise_2_image_quality_analyzer():
    """
    Save a JPEG image at various qualities and calculate PSNR.
    We simulate JPEG compression by adding quantization noise proportional
    to compression level.
    """

    def calculate_psnr(original, compressed):
        """Calculate Peak Signal-to-Noise Ratio between two images."""
        mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def simulate_jpeg_compression(img, quality):
        """
        Simulate JPEG compression by adding quantization noise.
        Lower quality = more noise = larger quantization steps.
        """
        # Quantization step inversely proportional to quality
        q_step = max(1, int((100 - quality) / 5))
        # Quantize pixel values
        compressed = (img // q_step) * q_step
        return compressed.astype(np.uint8)

    # Create a test image with smooth gradients and details
    h, w = 256, 256
    original = np.zeros((h, w, 3), dtype=np.uint8)
    # Add gradient
    for i in range(h):
        original[i, :, :] = int(255 * i / h)
    # Add some pattern
    for i in range(0, h, 20):
        original[i:i+2, :, :] = 255

    qualities = [10, 30, 50, 70, 90]

    print(f"{'Quality':>10} {'PSNR (dB)':>12} {'Est. Size Ratio':>18}")
    print("-" * 45)

    for q in qualities:
        compressed = simulate_jpeg_compression(original, q)
        psnr = calculate_psnr(original, compressed)

        # Estimate relative file size (rough approximation)
        # Higher quality -> larger file
        size_ratio = 0.05 + 0.95 * (q / 100.0)

        print(f"{q:>10} {psnr:>12.2f} {size_ratio:>18.2f}")

    return calculate_psnr


# =============================================================================
# Exercise 3: Create Color Grid
# =============================================================================

def exercise_3_color_grid():
    """
    Create a 400x400 image divided into 16 cells of 100x100,
    each filled with a different color using ROI operations.

    Returns:
        numpy array (400, 400, 3) BGR color grid
    """
    # Define 16 colors in BGR format
    colors_bgr = [
        (0, 0, 255),     # Red
        (0, 255, 255),   # Yellow
        (0, 255, 0),     # Green
        (255, 255, 0),   # Cyan
        (255, 0, 0),     # Blue
        (255, 0, 255),   # Magenta
        (255, 255, 255), # White
        (0, 0, 0),       # Black
        (128, 128, 128), # Gray
        (0, 165, 255),   # Orange
        (0, 128, 0),     # Dark Green
        (128, 0, 0),     # Dark Blue (Navy)
        (203, 192, 255), # Pink
        (0, 128, 128),   # Olive
        (128, 0, 128),   # Purple
        (19, 69, 139),   # Brown
    ]

    color_names = [
        "Red", "Yellow", "Green", "Cyan",
        "Blue", "Magenta", "White", "Black",
        "Gray", "Orange", "DarkGreen", "Navy",
        "Pink", "Olive", "Purple", "Brown"
    ]

    grid = np.zeros((400, 400, 3), dtype=np.uint8)

    for idx in range(16):
        row = idx // 4
        col = idx % 4

        # ROI coordinates
        y_start = row * 100
        y_end = (row + 1) * 100
        x_start = col * 100
        x_end = (col + 1) * 100

        # Fill ROI with color
        grid[y_start:y_end, x_start:x_end] = colors_bgr[idx]

        print(f"Cell ({row},{col}): {color_names[idx]:>10} BGR={colors_bgr[idx]}")

    print(f"\nGrid shape: {grid.shape}")
    return grid


# =============================================================================
# Exercise 4: Add Image Border
# =============================================================================

def exercise_4_add_border(img, thickness=10, color=(0, 0, 255)):
    """
    Add a border around an image. The image size increases by 2*thickness
    in each dimension.

    Parameters:
        img: numpy array (H, W, C) or (H, W)
        thickness: border width in pixels
        color: border color as BGR tuple (for color images)

    Returns:
        bordered image with increased size
    """
    if len(img.shape) == 3:
        h, w, c = img.shape
        bordered = np.zeros((h + 2 * thickness, w + 2 * thickness, c), dtype=img.dtype)
        # Fill with border color
        bordered[:, :] = color
        # Place original image in center
        bordered[thickness:thickness + h, thickness:thickness + w] = img
    else:
        h, w = img.shape
        # For grayscale, use first value of color tuple
        border_val = color[0] if isinstance(color, tuple) else color
        bordered = np.full((h + 2 * thickness, w + 2 * thickness), border_val, dtype=img.dtype)
        bordered[thickness:thickness + h, thickness:thickness + w] = img

    print(f"Original size: {img.shape}")
    print(f"Bordered size: {bordered.shape}")
    print(f"Border thickness: {thickness}px")
    return bordered


# =============================================================================
# Exercise 5: Pixel-Based Gradient
# =============================================================================

def exercise_5_pixel_gradient():
    """
    Create a 300x300 image with a horizontal gradient from black (left)
    to white (right) using NumPy broadcasting without loops.

    Returns:
        gradient image (300, 300) uint8
    """
    # Create a 1D gradient from 0 to 255 with 300 steps
    gradient_1d = np.linspace(0, 255, 300).astype(np.uint8)

    # Use broadcasting to expand to 300x300
    # gradient_1d has shape (300,), we need to tile it across 300 rows
    gradient_2d = np.tile(gradient_1d, (300, 1))

    print(f"Gradient shape: {gradient_2d.shape}")
    print(f"Left edge values:  {gradient_2d[0, :5]}")
    print(f"Right edge values: {gradient_2d[0, -5:]}")
    print(f"All rows identical: {np.all(gradient_2d[0] == gradient_2d[-1])}")

    return gradient_2d


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Compare Reading Modes")
    exercise_1_compare_reading_modes()

    print("\n>>> Exercise 2: Image Quality Analyzer (PSNR)")
    exercise_2_image_quality_analyzer()

    print("\n>>> Exercise 3: Create Color Grid")
    grid = exercise_3_color_grid()

    print("\n>>> Exercise 4: Add Image Border")
    test_img = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
    bordered = exercise_4_add_border(test_img, thickness=15, color=(0, 0, 255))

    print("\n>>> Exercise 5: Pixel-Based Gradient")
    gradient = exercise_5_pixel_gradient()

    print("\nAll exercises completed successfully.")
