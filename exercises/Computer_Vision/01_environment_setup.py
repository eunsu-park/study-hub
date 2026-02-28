"""
Exercise Solutions for Lesson 01: Environment Setup
Computer Vision - OpenCV Basics

Topics covered:
- Environment information checking
- Image property inspection
- Creating blank canvas images
- NumPy brightness/contrast operations
- Channel separation
"""

import numpy as np


# =============================================================================
# Exercise 1: Environment Check Script
# =============================================================================

def exercise_1_environment_check():
    """
    Output environment information:
    - Python version
    - NumPy version
    - OpenCV availability (simulated)
    - GPU acceleration availability (simulated)
    """
    import sys

    print("=" * 50)
    print("Environment Check Report")
    print("=" * 50)

    # Python version
    print(f"Python version: {sys.version}")

    # NumPy version
    print(f"NumPy version: {np.__version__}")

    # Simulate OpenCV version check (we use numpy-only approach)
    print(f"OpenCV: Not imported (using NumPy-only solutions)")

    # Simulate GPU check
    print(f"CUDA devices: 0 (simulated - no cv2.cuda available)")

    print("=" * 50)
    print("Environment check complete.")


# =============================================================================
# Exercise 2: Image Info Printer
# =============================================================================

def exercise_2_image_info_printer(img):
    """
    Print detailed information about an image (numpy array).

    Parameters:
        img: numpy array representing an image

    Output items:
    - Image size (width x height)
    - Number of channels
    - Data type
    - Memory usage
    - Pixel value range (min, max)
    - Average brightness
    """
    print("=" * 50)
    print("Image Information")
    print("=" * 50)

    if img is None:
        print("Error: Image is None (load failed)")
        return

    # Image dimensions
    if len(img.shape) == 3:
        h, w, c = img.shape
        print(f"Size: {w} x {h}")
        print(f"Channels: {c}")
    elif len(img.shape) == 2:
        h, w = img.shape
        c = 1
        print(f"Size: {w} x {h}")
        print(f"Channels: 1 (grayscale)")
    else:
        print(f"Unexpected shape: {img.shape}")
        return

    # Data type
    print(f"Data type: {img.dtype}")

    # Memory usage
    mem_bytes = img.nbytes
    if mem_bytes < 1024:
        print(f"Memory usage: {mem_bytes} bytes")
    elif mem_bytes < 1024 * 1024:
        print(f"Memory usage: {mem_bytes / 1024:.2f} KB")
    else:
        print(f"Memory usage: {mem_bytes / (1024 * 1024):.2f} MB")

    # Pixel value range
    print(f"Pixel range: min={img.min()}, max={img.max()}")

    # Average brightness
    if c == 1 or len(img.shape) == 2:
        avg_brightness = np.mean(img)
    else:
        # Convert BGR to grayscale for brightness: 0.114*B + 0.587*G + 0.299*R
        gray = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        avg_brightness = np.mean(gray)

    print(f"Average brightness: {avg_brightness:.2f}")
    print("=" * 50)


# =============================================================================
# Exercise 3: Creating Blank Canvas
# =============================================================================

def exercise_3_blank_canvas():
    """
    Create images with the following conditions:
    1. 800x600 black image
    2. 800x600 white image
    3. 800x600 red image (BGR: blue=0, green=0, red=255)
    4. 400x400 checkerboard pattern (50px units)

    Returns:
        dict of created images
    """
    results = {}

    # 1. Black image (800x600, 3 channels)
    black = np.zeros((600, 800, 3), dtype=np.uint8)
    results['black'] = black
    print(f"Black image: shape={black.shape}, range=[{black.min()}, {black.max()}]")

    # 2. White image
    white = np.ones((600, 800, 3), dtype=np.uint8) * 255
    results['white'] = white
    print(f"White image: shape={white.shape}, range=[{white.min()}, {white.max()}]")

    # 3. Red image (in BGR: B=0, G=0, R=255)
    red = np.zeros((600, 800, 3), dtype=np.uint8)
    red[:, :, 2] = 255  # R channel is index 2 in BGR
    results['red'] = red
    print(f"Red image: shape={red.shape}, BGR at (0,0)={red[0, 0]}")

    # 4. Checkerboard pattern (400x400, 50px cells)
    checker = np.zeros((400, 400), dtype=np.uint8)
    cell_size = 50
    for row in range(400 // cell_size):
        for col in range(400 // cell_size):
            if (row + col) % 2 == 0:
                r_start = row * cell_size
                c_start = col * cell_size
                checker[r_start:r_start + cell_size, c_start:c_start + cell_size] = 255

    results['checkerboard'] = checker
    print(f"Checkerboard: shape={checker.shape}, "
          f"white cells={(checker == 255).sum() // (cell_size * cell_size)}, "
          f"black cells={(checker == 0).sum() // (cell_size * cell_size)}")

    return results


# =============================================================================
# Exercise 4: NumPy Operations Practice
# =============================================================================

def exercise_4_numpy_operations(img):
    """
    Perform brightness and contrast operations on an image.

    Parameters:
        img: numpy array (uint8) representing an image

    Returns:
        dict of modified images
    """
    results = {}

    # 1. Increase brightness by 50 (with clipping to [0, 255])
    bright_up = np.clip(img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    results['brightness_up'] = bright_up
    print(f"Brightness +50: mean {np.mean(img):.1f} -> {np.mean(bright_up):.1f}")

    # 2. Decrease brightness by 50 (with clipping)
    bright_down = np.clip(img.astype(np.int16) - 50, 0, 255).astype(np.uint8)
    results['brightness_down'] = bright_down
    print(f"Brightness -50: mean {np.mean(img):.1f} -> {np.mean(bright_down):.1f}")

    # 3. Increase contrast by 1.5x (multiply around mean)
    contrast_up = np.clip(img.astype(np.float64) * 1.5, 0, 255).astype(np.uint8)
    results['contrast_up'] = contrast_up
    print(f"Contrast 1.5x: std {np.std(img):.1f} -> {np.std(contrast_up):.1f}")

    # 4. Invert image (255 - img)
    inverted = (255 - img).astype(np.uint8)
    results['inverted'] = inverted
    print(f"Inverted: mean {np.mean(img):.1f} -> {np.mean(inverted):.1f}")

    return results


# =============================================================================
# Exercise 5: Channel Separation Preview
# =============================================================================

def exercise_5_channel_separation(img):
    """
    Separate a BGR color image into individual channels and display each
    as a grayscale representation.

    Parameters:
        img: numpy array of shape (H, W, 3) in BGR format

    Returns:
        dict with 'blue', 'green', 'red' grayscale channel images
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        print("Error: Input must be a 3-channel BGR image")
        return None

    # Separate channels using NumPy indexing
    blue_channel = img[:, :, 0]   # B channel
    green_channel = img[:, :, 1]  # G channel
    red_channel = img[:, :, 2]    # R channel

    results = {
        'blue': blue_channel,
        'green': green_channel,
        'red': red_channel
    }

    for name, ch in results.items():
        print(f"{name.upper()} channel: shape={ch.shape}, "
              f"min={ch.min()}, max={ch.max()}, mean={ch.mean():.1f}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Environment Check")
    exercise_1_environment_check()

    print("\n>>> Exercise 2: Image Info Printer")
    # Create a synthetic test image (300x400 BGR)
    test_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    exercise_2_image_info_printer(test_img)

    print("\n>>> Exercise 3: Creating Blank Canvas")
    canvases = exercise_3_blank_canvas()

    print("\n>>> Exercise 4: NumPy Operations")
    # Create a gradient test image for meaningful brightness/contrast tests
    gradient = np.tile(np.linspace(30, 200, 400, dtype=np.uint8), (300, 1))
    gradient_bgr = np.stack([gradient, gradient, gradient], axis=-1)
    ops = exercise_4_numpy_operations(gradient_bgr)

    print("\n>>> Exercise 5: Channel Separation")
    # Create a colorful test image
    color_img = np.zeros((200, 200, 3), dtype=np.uint8)
    color_img[:100, :100] = [255, 0, 0]    # Blue (top-left)
    color_img[:100, 100:] = [0, 255, 0]    # Green (top-right)
    color_img[100:, :100] = [0, 0, 255]    # Red (bottom-left)
    color_img[100:, 100:] = [255, 255, 0]  # Cyan (bottom-right)
    channels = exercise_5_channel_separation(color_img)

    print("\nAll exercises completed successfully.")
