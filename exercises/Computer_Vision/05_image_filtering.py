"""
Exercise Solutions for Lesson 05: Image Filtering
Computer Vision - Convolution, Blur, Sharpening

Topics covered:
- Noise removal comparison (Gaussian vs salt-and-pepper)
- Real-time blur intensity control (trackbar simulation)
- Custom emboss in 8 directions
- Advanced sharpening with threshold
- Tilt-shift miniature effect
"""

import numpy as np


# =============================================================================
# Helper: 2D Convolution
# =============================================================================

def convolve2d(img, kernel):
    """Apply 2D convolution to a grayscale image using zero-padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape[:2]

    # Handle multi-channel
    if len(img.shape) == 3:
        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = convolve2d(img[:, :, c], kernel)
        return result

    padded = np.pad(img.astype(np.float64), ((ph, ph), (pw, pw)), mode='reflect')
    result = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result


def calculate_psnr(original, processed):
    """Calculate PSNR between two images."""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# =============================================================================
# Exercise 1: Noise Removal Comparison
# =============================================================================

def exercise_1_noise_removal_comparison():
    """
    Generate Gaussian noise and salt-and-pepper noise separately.
    Compare removal effects with average, Gaussian, and median filters.
    Use PSNR for quantitative comparison.
    """
    # Create a clean test image with patterns
    h, w = 128, 128
    clean = np.zeros((h, w), dtype=np.uint8)
    # Add gradient background
    clean[:] = np.tile(np.linspace(50, 200, w).astype(np.uint8), (h, 1))
    # Add a rectangle
    clean[30:90, 30:90] = 180

    def add_gaussian_noise(img, mean=0, var=400):
        noise = np.random.normal(mean, np.sqrt(var), img.shape)
        noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return noisy

    def add_salt_pepper_noise(img, amount=0.05):
        noisy = img.copy()
        n_salt = int(amount * img.size / 2)
        n_pepper = int(amount * img.size / 2)
        # Salt
        coords = [np.random.randint(0, i, n_salt) for i in img.shape]
        noisy[coords[0], coords[1]] = 255
        # Pepper
        coords = [np.random.randint(0, i, n_pepper) for i in img.shape]
        noisy[coords[0], coords[1]] = 0
        return noisy

    def average_blur(img, ksize=5):
        kernel = np.ones((ksize, ksize)) / (ksize * ksize)
        result = convolve2d(img, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def gaussian_blur(img, ksize=5, sigma=1.0):
        ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        result = convolve2d(img, kernel)
        return np.clip(result, 0, 255).astype(np.uint8)

    def median_blur(img, ksize=5):
        h, w = img.shape
        pad = ksize // 2
        padded = np.pad(img, pad, mode='reflect')
        result = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                region = padded[i:i + ksize, j:j + ksize]
                result[i, j] = np.median(region)
        return result.astype(np.uint8)

    print("=" * 70)
    print(f"{'Noise Type':>15} | {'Filter':>12} | {'PSNR (dB)':>10} | {'MSE':>8}")
    print("=" * 70)

    for noise_name, noisy in [
        ("Gaussian", add_gaussian_noise(clean)),
        ("Salt&Pepper", add_salt_pepper_noise(clean, 0.05))
    ]:
        noisy_psnr = calculate_psnr(clean, noisy)
        print(f"{noise_name:>15} | {'(noisy)':>12} | {noisy_psnr:>10.2f} | "
              f"{np.mean((clean.astype(float) - noisy.astype(float))**2):>8.1f}")

        for filter_name, filtered in [
            ("Average", average_blur(noisy, 5)),
            ("Gaussian", gaussian_blur(noisy, 5, 1.0)),
            ("Median", median_blur(noisy, 5))
        ]:
            psnr = calculate_psnr(clean, filtered)
            mse = np.mean((clean.astype(float) - filtered.astype(float))**2)
            print(f"{'':>15} | {filter_name:>12} | {psnr:>10.2f} | {mse:>8.1f}")

        print("-" * 70)


# =============================================================================
# Exercise 2: Real-Time Blur Intensity Control
# =============================================================================

def exercise_2_blur_control():
    """
    Simulate trackbar-controlled blur on a test image.
    Test different kernel sizes for Gaussian and bilateral-like filters.
    """
    img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 180  # Add a bright rectangle

    kernel_sizes = [1, 3, 5, 7, 9, 11]

    print("Gaussian Blur at different kernel sizes:")
    for ksize in kernel_sizes:
        if ksize == 1:
            blurred = img.copy()
        else:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
            ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel /= kernel.sum()
            blurred = np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)

        edge_strength = np.std(np.diff(blurred.astype(float), axis=1))
        print(f"  ksize={ksize:>2}: mean={blurred.mean():.1f}, "
              f"edge_strength={edge_strength:.2f}")


# =============================================================================
# Exercise 3: Custom Emboss Directions
# =============================================================================

def exercise_3_emboss_directions(img):
    """
    Design and apply emboss kernels for 8 directions:
    N, NE, E, SE, S, SW, W, NW.

    Parameters:
        img: grayscale image (H, W)

    Returns:
        dict of 8 embossed images
    """
    # 8-direction emboss kernels
    kernels = {
        'N':  np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]], dtype=np.float64),

        'NE': np.array([[ 0, -1, -1],
                        [ 1,  0, -1],
                        [ 1,  1,  0]], dtype=np.float64),

        'E':  np.array([[ 1,  0, -1],
                        [ 1,  0, -1],
                        [ 1,  0, -1]], dtype=np.float64),

        'SE': np.array([[ 1,  1,  0],
                        [ 1,  0, -1],
                        [ 0, -1, -1]], dtype=np.float64),

        'S':  np.array([[ 1,  1,  1],
                        [ 0,  0,  0],
                        [-1, -1, -1]], dtype=np.float64),

        'SW': np.array([[ 0,  1,  1],
                        [-1,  0,  1],
                        [-1, -1,  0]], dtype=np.float64),

        'W':  np.array([[-1,  0,  1],
                        [-1,  0,  1],
                        [-1,  0,  1]], dtype=np.float64),

        'NW': np.array([[-1, -1,  0],
                        [-1,  0,  1],
                        [ 0,  1,  1]], dtype=np.float64),
    }

    results = {}
    for direction, kernel in kernels.items():
        embossed = convolve2d(img, kernel)
        # Offset by 128 to show both positive and negative edges
        embossed = np.clip(embossed + 128, 0, 255).astype(np.uint8)
        results[direction] = embossed
        print(f"  {direction:>2}: min={embossed.min():>3}, max={embossed.max():>3}, "
              f"mean={embossed.mean():.1f}")

    return results


# =============================================================================
# Exercise 4: Advanced Sharpening
# =============================================================================

def exercise_4_advanced_sharpening(img, amount=1.5, radius=2, threshold=10):
    """
    Advanced unsharp masking with:
    - Strength control (amount)
    - Blur radius control (radius)
    - Threshold to ignore small changes
    - Separate highlight/shadow handling

    Parameters:
        img: grayscale image (H, W) uint8
        amount: sharpening strength multiplier
        radius: Gaussian blur sigma
        threshold: minimum difference to sharpen

    Returns:
        sharpened image
    """
    img_f = img.astype(np.float64)

    # Create Gaussian blur kernel
    ksize = int(6 * radius + 1)
    if ksize % 2 == 0:
        ksize += 1
    ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * radius**2))
    kernel /= kernel.sum()

    blurred = convolve2d(img, kernel)

    # Unsharp mask: detail = original - blurred
    detail = img_f - blurred

    # Apply threshold: only sharpen where detail is above threshold
    mask = np.abs(detail) >= threshold

    # Separate handling for highlights and shadows
    highlight_mask = (img_f > 128) & mask
    shadow_mask = (img_f <= 128) & mask

    sharpened = img_f.copy()
    # Highlights: apply slightly reduced amount to avoid blowout
    sharpened[highlight_mask] += detail[highlight_mask] * amount * 0.8
    # Shadows: apply full amount
    sharpened[shadow_mask] += detail[shadow_mask] * amount

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    print(f"Sharpening: amount={amount}, radius={radius}, threshold={threshold}")
    print(f"Pixels sharpened: {np.sum(mask)}/{img.size} "
          f"({100*np.sum(mask)/img.size:.1f}%)")
    print(f"  Highlights: {np.sum(highlight_mask)}, Shadows: {np.sum(shadow_mask)}")
    psnr = calculate_psnr(img, sharpened)
    print(f"  PSNR vs original: {psnr:.2f} dB")

    return sharpened


# =============================================================================
# Exercise 5: Miniature Effect (Tilt Shift)
# =============================================================================

def exercise_5_tilt_shift(img, focus_y=None, focus_height=None, blur_amount=7):
    """
    Implement tilt-shift miniature effect: keep center sharp,
    progressively blur toward top and bottom.

    Parameters:
        img: (H, W) or (H, W, C) image
        focus_y: y-center of the focus band (default: image center)
        focus_height: height of the sharp region (default: H//4)
        blur_amount: Gaussian blur sigma for maximum blur

    Returns:
        tilt-shifted image
    """
    h, w = img.shape[:2]

    if focus_y is None:
        focus_y = h // 2
    if focus_height is None:
        focus_height = h // 4

    # Create a Gaussian blur of the image
    ksize = int(6 * blur_amount + 1)
    if ksize % 2 == 0:
        ksize += 1
    ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * blur_amount**2))
    kernel /= kernel.sum()

    if len(img.shape) == 3:
        blurred = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            blurred[:, :, c] = convolve2d(img[:, :, c], kernel)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        blurred = np.clip(convolve2d(img, kernel), 0, 255).astype(np.uint8)

    # Create gradient mask: 1.0 = sharp (original), 0.0 = fully blurred
    mask = np.zeros((h, 1), dtype=np.float64)

    for row in range(h):
        dist = abs(row - focus_y)
        if dist <= focus_height // 2:
            mask[row] = 1.0  # Sharp zone
        else:
            # Smooth transition
            transition = (dist - focus_height // 2) / (h // 4)
            mask[row] = max(0.0, 1.0 - transition)

    # Broadcast mask across width (and channels if applicable)
    if len(img.shape) == 3:
        mask_full = np.broadcast_to(mask.reshape(h, 1, 1), img.shape)
    else:
        mask_full = np.broadcast_to(mask.reshape(h, 1), img.shape)

    # Blend: result = mask * original + (1-mask) * blurred
    result = (mask_full * img.astype(np.float64) +
              (1 - mask_full) * blurred.astype(np.float64))
    result = np.clip(result, 0, 255).astype(np.uint8)

    print(f"Tilt shift: focus_y={focus_y}, focus_height={focus_height}, "
          f"blur_amount={blur_amount}")
    print(f"Sharp zone: rows {focus_y - focus_height//2} to {focus_y + focus_height//2}")

    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Noise Removal Comparison")
    exercise_1_noise_removal_comparison()

    print("\n>>> Exercise 2: Blur Intensity Control")
    exercise_2_blur_control()

    print("\n>>> Exercise 3: Custom Emboss Directions")
    test_img = np.zeros((64, 64), dtype=np.uint8)
    test_img[20:44, 20:44] = 200  # Bright square
    test_img[28:36, 28:36] = 100  # Dark center
    exercise_3_emboss_directions(test_img)

    print("\n>>> Exercise 4: Advanced Sharpening")
    sharp_test = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
    sharp_test[20:44, 20:44] = 180
    exercise_4_advanced_sharpening(sharp_test, amount=2.0, radius=1.5, threshold=5)

    print("\n>>> Exercise 5: Tilt Shift Miniature Effect")
    scene = np.random.randint(80, 200, (100, 150, 3), dtype=np.uint8)
    scene[40:60, 50:100] = [0, 200, 0]  # Green "focus" area
    exercise_5_tilt_shift(scene, focus_y=50, focus_height=20, blur_amount=3)

    print("\nAll exercises completed successfully.")
