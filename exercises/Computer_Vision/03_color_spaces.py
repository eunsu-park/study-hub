"""
Exercise Solutions for Lesson 03: Color Spaces
Computer Vision - BGR, HSV, LAB Color Spaces

Topics covered:
- Color palette generation
- HSV color picking
- Channel swap effects
- Skin color detection in HSV/YCrCb
- Color transition animation (hue shift)
"""

import numpy as np


# =============================================================================
# Exercise 1: Color Palette Generator
# =============================================================================

def exercise_1_color_palette():
    """
    Define 16 main colors in BGR and create a palette image by arranging
    100x100 color chips in a 4x4 grid.

    Returns:
        (palette_image, color_definitions) tuple
    """
    color_defs = {
        'Red':        (0, 0, 255),
        'Orange':     (0, 165, 255),
        'Yellow':     (0, 255, 255),
        'Green':      (0, 255, 0),
        'Cyan':       (255, 255, 0),
        'Blue':       (255, 0, 0),
        'Magenta':    (255, 0, 255),
        'Pink':       (203, 192, 255),
        'White':      (255, 255, 255),
        'Black':      (0, 0, 0),
        'Gray':       (128, 128, 128),
        'Light Gray': (200, 200, 200),
        'Dark Red':   (0, 0, 128),
        'Dark Green': (0, 128, 0),
        'Navy':       (128, 0, 0),
        'Brown':      (19, 69, 139),
    }

    palette = np.zeros((400, 400, 3), dtype=np.uint8)

    for idx, (name, bgr) in enumerate(color_defs.items()):
        row = idx // 4
        col = idx % 4
        y0 = row * 100
        x0 = col * 100
        palette[y0:y0 + 100, x0:x0 + 100] = bgr

    print(f"Palette shape: {palette.shape}")
    print(f"Colors defined: {len(color_defs)}")
    for name, bgr in color_defs.items():
        print(f"  {name:>12}: BGR={bgr}")

    return palette, color_defs


# =============================================================================
# Exercise 2: HSV Color Picker
# =============================================================================

def exercise_2_hsv_color_picker(img_bgr, x, y):
    """
    Given a click position (x, y) on a BGR image, return the HSV value
    at that pixel and create a mask highlighting similar colors.

    Parameters:
        img_bgr: (H, W, 3) BGR image
        x, y: pixel coordinates

    Returns:
        (hsv_value, mask) where mask highlights similar colors
    """
    def bgr_to_hsv_pixel(b, g, r):
        """Convert a single BGR pixel to HSV (OpenCV scale: H 0-180, S/V 0-255)."""
        b_f, g_f, r_f = b / 255.0, g / 255.0, r / 255.0
        cmax = max(r_f, g_f, b_f)
        cmin = min(r_f, g_f, b_f)
        diff = cmax - cmin

        # Hue
        if diff == 0:
            h = 0
        elif cmax == r_f:
            h = 60 * (((g_f - b_f) / diff) % 6)
        elif cmax == g_f:
            h = 60 * (((b_f - r_f) / diff) + 2)
        else:
            h = 60 * (((r_f - g_f) / diff) + 4)

        # Saturation
        s = 0 if cmax == 0 else (diff / cmax)

        # Value
        v = cmax

        # Scale to OpenCV HSV range
        return int(h / 2), int(s * 255), int(v * 255)

    h_img, w_img = img_bgr.shape[:2]
    if not (0 <= y < h_img and 0 <= x < w_img):
        print(f"Error: ({x}, {y}) is out of bounds ({w_img}x{h_img})")
        return None, None

    # Get BGR value at clicked position
    b, g, r = int(img_bgr[y, x, 0]), int(img_bgr[y, x, 1]), int(img_bgr[y, x, 2])
    h, s, v = bgr_to_hsv_pixel(b, g, r)

    print(f"Position: ({x}, {y})")
    print(f"BGR: ({b}, {g}, {r})")
    print(f"HSV: ({h}, {s}, {v})")

    # Convert entire image to HSV for masking
    hsv_img = np.zeros_like(img_bgr)
    for iy in range(h_img):
        for ix in range(w_img):
            bi, gi, ri = int(img_bgr[iy, ix, 0]), int(img_bgr[iy, ix, 1]), int(img_bgr[iy, ix, 2])
            hsv_img[iy, ix] = bgr_to_hsv_pixel(bi, gi, ri)

    # Create mask for similar colors (tolerance: H+-10, S+-40, V+-40)
    h_tol, s_tol, v_tol = 10, 40, 40
    mask = (
        (np.abs(hsv_img[:, :, 0].astype(np.int16) - h) <= h_tol) &
        (np.abs(hsv_img[:, :, 1].astype(np.int16) - s) <= s_tol) &
        (np.abs(hsv_img[:, :, 2].astype(np.int16) - v) <= v_tol)
    ).astype(np.uint8) * 255

    similar_pixels = np.sum(mask > 0)
    total_pixels = h_img * w_img
    print(f"Similar color pixels: {similar_pixels}/{total_pixels} "
          f"({100 * similar_pixels / total_pixels:.1f}%)")

    return (h, s, v), mask


# =============================================================================
# Exercise 3: Channel Swap Effects
# =============================================================================

def exercise_3_channel_swap(img_bgr):
    """
    Create 6 different color effects by rearranging BGR channels:
    BGR, BRG, GBR, GRB, RBG, RGB.

    Parameters:
        img_bgr: (H, W, 3) BGR image

    Returns:
        dict of 6 swapped images
    """
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]

    swaps = {
        'BGR': (b, g, r),      # Original
        'BRG': (b, r, g),      # Swap G and R
        'GBR': (g, b, r),      # Swap B and G
        'GRB': (g, r, b),      # Rotate channels
        'RBG': (r, b, g),      # Rotate channels
        'RGB': (r, g, b),      # Reverse (display as-is would look "correct" in RGB viewer)
    }

    results = {}
    for name, (c0, c1, c2) in swaps.items():
        swapped = np.stack([c0, c1, c2], axis=-1)
        results[name] = swapped
        print(f"{name}: mean per channel = [{c0.mean():.1f}, {c1.mean():.1f}, {c2.mean():.1f}]")

    return results


# =============================================================================
# Exercise 4: Skin Color Detection
# =============================================================================

def exercise_4_skin_detection(img_bgr):
    """
    Detect skin-colored areas using HSV and YCrCb color spaces.
    Compare both methods.

    Parameters:
        img_bgr: (H, W, 3) BGR image

    Returns:
        (hsv_mask, ycrcb_mask) binary masks
    """
    h_img, w_img = img_bgr.shape[:2]

    # --- Method 1: HSV-based skin detection ---
    # Convert BGR to HSV manually
    hsv = np.zeros_like(img_bgr, dtype=np.float64)
    bgr_float = img_bgr.astype(np.float64) / 255.0

    r, g, b = bgr_float[:, :, 2], bgr_float[:, :, 1], bgr_float[:, :, 0]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # Hue calculation
    hue = np.zeros_like(diff)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    sat = np.where(cmax > 0, diff / cmax, 0)
    val = cmax

    # Scale to OpenCV range: H: 0-180, S: 0-255, V: 0-255
    h_scaled = (hue / 2).astype(np.uint8)
    s_scaled = (sat * 255).astype(np.uint8)
    v_scaled = (val * 255).astype(np.uint8)

    # Skin color in HSV: H: 0-25, S: 20-150, V: 70-255
    hsv_mask = (
        (h_scaled >= 0) & (h_scaled <= 25) &
        (s_scaled >= 20) & (s_scaled <= 150) &
        (v_scaled >= 70) & (v_scaled <= 255)
    ).astype(np.uint8) * 255

    # --- Method 2: YCrCb-based skin detection ---
    # Convert BGR to YCrCb
    # Y  =  0.299*R + 0.587*G + 0.114*B
    # Cr = (R - Y) * 0.713 + 128
    # Cb = (B - Y) * 0.564 + 128
    y_ch = (0.299 * r + 0.587 * g + 0.114 * b)
    cr = ((r - y_ch) * 0.713 + 0.5).astype(np.float64) * 255  # scale
    cb = ((b - y_ch) * 0.564 + 0.5).astype(np.float64) * 255

    cr = np.clip(cr, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)

    # Skin in YCrCb: Cr: 135-180, Cb: 85-135
    ycrcb_mask = (
        (cr >= 135) & (cr <= 180) &
        (cb >= 85) & (cb <= 135)
    ).astype(np.uint8) * 255

    hsv_count = np.sum(hsv_mask > 0)
    ycrcb_count = np.sum(ycrcb_mask > 0)
    total = h_img * w_img

    print(f"HSV skin pixels:   {hsv_count}/{total} ({100*hsv_count/total:.1f}%)")
    print(f"YCrCb skin pixels: {ycrcb_count}/{total} ({100*ycrcb_count/total:.1f}%)")

    # Overlap between methods
    overlap = np.sum((hsv_mask > 0) & (ycrcb_mask > 0))
    print(f"Overlap:           {overlap}/{total} ({100*overlap/total:.1f}%)")

    return hsv_mask, ycrcb_mask


# =============================================================================
# Exercise 5: Color Transition Animation (Hue Shift)
# =============================================================================

def exercise_5_color_transition(img_bgr, num_frames=10):
    """
    Create frames where colors shift through the rainbow by incrementing
    the H channel in HSV space.

    Parameters:
        img_bgr: (H, W, 3) BGR image
        num_frames: number of animation frames

    Returns:
        list of shifted BGR images
    """
    h_img, w_img = img_bgr.shape[:2]
    bgr_float = img_bgr.astype(np.float64) / 255.0
    r, g, b = bgr_float[:, :, 2], bgr_float[:, :, 1], bgr_float[:, :, 0]

    # Convert to HSV
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    hue = np.zeros_like(diff)
    mask_r = (cmax == r) & (diff > 0)
    mask_g = (cmax == g) & (diff > 0)
    mask_b = (cmax == b) & (diff > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    sat = np.where(cmax > 0, diff / cmax, 0)
    val = cmax

    def hsv_to_bgr(h_arr, s_arr, v_arr):
        """Convert HSV arrays (H: 0-360, S: 0-1, V: 0-1) to BGR uint8."""
        h_arr = h_arr % 360
        c = v_arr * s_arr
        x = c * (1 - np.abs((h_arr / 60) % 2 - 1))
        m = v_arr - c

        r_out = np.zeros_like(h_arr)
        g_out = np.zeros_like(h_arr)
        b_out = np.zeros_like(h_arr)

        for lo, hi, rv, gv, bv in [
            (0, 60, c, x, 0), (60, 120, x, c, 0), (120, 180, 0, c, x),
            (180, 240, 0, x, c), (240, 300, x, 0, c), (300, 360, c, 0, x)
        ]:
            mask = (h_arr >= lo) & (h_arr < hi)
            r_out[mask] = (rv[mask] if isinstance(rv, np.ndarray) else rv) if isinstance(rv, (int, float)) else rv[mask]
            g_out[mask] = (gv[mask] if isinstance(gv, np.ndarray) else gv) if isinstance(gv, (int, float)) else gv[mask]
            b_out[mask] = (bv[mask] if isinstance(bv, np.ndarray) else bv) if isinstance(bv, (int, float)) else bv[mask]

        for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]:
            mask = (h_arr >= lo) & (h_arr < hi)
            if lo == 0:
                r_out[mask] = c[mask]; g_out[mask] = x[mask]; b_out[mask] = 0
            elif lo == 60:
                r_out[mask] = x[mask]; g_out[mask] = c[mask]; b_out[mask] = 0
            elif lo == 120:
                r_out[mask] = 0; g_out[mask] = c[mask]; b_out[mask] = x[mask]
            elif lo == 180:
                r_out[mask] = 0; g_out[mask] = x[mask]; b_out[mask] = c[mask]
            elif lo == 240:
                r_out[mask] = x[mask]; g_out[mask] = 0; b_out[mask] = c[mask]
            else:
                r_out[mask] = c[mask]; g_out[mask] = 0; b_out[mask] = x[mask]

        r_out += m
        g_out += m
        b_out += m

        bgr = np.stack([
            np.clip(b_out * 255, 0, 255).astype(np.uint8),
            np.clip(g_out * 255, 0, 255).astype(np.uint8),
            np.clip(r_out * 255, 0, 255).astype(np.uint8)
        ], axis=-1)
        return bgr

    frames = []
    step = 360 // num_frames

    for i in range(num_frames):
        h_shift = i * step
        shifted_hue = (hue + h_shift) % 360
        frame = hsv_to_bgr(shifted_hue, sat, val)
        frames.append(frame)
        print(f"Frame {i+1}/{num_frames}: hue shift = {h_shift} degrees, "
              f"mean BGR = ({frame[:,:,0].mean():.0f}, {frame[:,:,1].mean():.0f}, "
              f"{frame[:,:,2].mean():.0f})")

    return frames


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Color Palette Generator")
    palette, defs = exercise_1_color_palette()

    print("\n>>> Exercise 2: HSV Color Picker")
    # Create a small test image with skin-like and non-skin colors
    test = np.zeros((50, 50, 3), dtype=np.uint8)
    test[:25, :25] = [100, 150, 200]  # skin-ish
    test[:25, 25:] = [255, 0, 0]      # blue
    test[25:, :25] = [0, 255, 0]      # green
    test[25:, 25:] = [0, 0, 255]      # red
    exercise_2_hsv_color_picker(test, 10, 10)

    print("\n>>> Exercise 3: Channel Swap Effects")
    swap_test = np.zeros((100, 100, 3), dtype=np.uint8)
    swap_test[:50, :, 2] = 200  # Red in top half
    swap_test[50:, :, 1] = 200  # Green in bottom half
    exercise_3_channel_swap(swap_test)

    print("\n>>> Exercise 4: Skin Color Detection")
    # Create a synthetic image with some skin-like pixels
    skin_test = np.zeros((100, 100, 3), dtype=np.uint8)
    skin_test[:50, :50] = [130, 170, 220]  # skin-ish BGR
    skin_test[:50, 50:] = [255, 0, 0]      # blue
    skin_test[50:, :] = [50, 100, 180]     # darker skin-ish
    exercise_4_skin_detection(skin_test)

    print("\n>>> Exercise 5: Color Transition Animation")
    anim_test = np.zeros((50, 50, 3), dtype=np.uint8)
    anim_test[:, :, 2] = 200  # Red image
    anim_test[:, :, 1] = 50
    frames = exercise_5_color_transition(anim_test, num_frames=6)

    print("\nAll exercises completed successfully.")
