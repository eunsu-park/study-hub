"""
Exercise Solutions for Lesson 04: Geometric Transforms
Computer Vision - Resize, Rotate, Affine, Perspective

Topics covered:
- Batch resize maintaining aspect ratio
- Image rotation animation
- Perspective transformation (ID card scanner)
- Image mosaic grid
- AR card overlay effect
"""

import numpy as np


# =============================================================================
# Exercise 1: Batch Resize
# =============================================================================

def exercise_1_batch_resize(images, max_width=800):
    """
    Resize all images to max_width maintaining aspect ratio.

    Parameters:
        images: list of (name, numpy_array) tuples
        max_width: target width

    Returns:
        list of (name, resized_array) tuples
    """
    def resize_nearest(img, new_w, new_h):
        """Resize image using nearest-neighbor interpolation."""
        old_h, old_w = img.shape[:2]
        row_indices = (np.arange(new_h) * old_h / new_h).astype(int)
        col_indices = (np.arange(new_w) * old_w / new_w).astype(int)
        row_indices = np.clip(row_indices, 0, old_h - 1)
        col_indices = np.clip(col_indices, 0, old_w - 1)
        return img[np.ix_(row_indices, col_indices)]

    results = []
    for name, img in images:
        h, w = img.shape[:2]
        if w <= max_width:
            print(f"  {name}: {w}x{h} -> skipped (already <= {max_width})")
            results.append((name, img.copy()))
            continue

        # Calculate new dimensions maintaining aspect ratio
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)

        resized = resize_nearest(img, new_w, new_h)
        results.append((name, resized))
        print(f"  {name}: {w}x{h} -> {new_w}x{new_h} (scale={scale:.3f})")

    return results


# =============================================================================
# Exercise 2: Image Rotation Animation
# =============================================================================

def exercise_2_rotation_animation(img, num_steps=72):
    """
    Generate frames rotating an image from 0 to 360 degrees in equal steps.
    Expands canvas so the image doesn't get cropped.

    Parameters:
        img: (H, W) or (H, W, C) image
        num_steps: number of rotation steps (360/num_steps degrees each)

    Returns:
        list of rotated images
    """
    h, w = img.shape[:2]
    # Diagonal of the image determines the minimum canvas size
    diagonal = int(np.ceil(np.sqrt(h * h + w * w)))
    cx, cy = w / 2.0, h / 2.0

    frames = []
    angle_step = 360 / num_steps

    for i in range(num_steps):
        angle = i * angle_step
        theta = np.radians(angle)
        cos_a = np.cos(theta)
        sin_a = np.sin(theta)

        # Create output canvas
        if len(img.shape) == 3:
            canvas = np.zeros((diagonal, diagonal, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((diagonal, diagonal), dtype=img.dtype)

        # Center offset in canvas
        ox = diagonal / 2.0
        oy = diagonal / 2.0

        # For each output pixel, find the corresponding source pixel (inverse mapping)
        yy, xx = np.mgrid[0:diagonal, 0:diagonal]
        # Translate to origin, rotate back, translate to source center
        src_x = cos_a * (xx - ox) + sin_a * (yy - oy) + cx
        src_y = -sin_a * (xx - ox) + cos_a * (yy - oy) + cy

        # Nearest-neighbor sampling
        src_xi = np.round(src_x).astype(int)
        src_yi = np.round(src_y).astype(int)

        valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
        canvas[valid] = img[src_yi[valid], src_xi[valid]]

        frames.append(canvas)

    print(f"Generated {len(frames)} frames")
    print(f"Original: {w}x{h}, Canvas: {diagonal}x{diagonal}")
    print(f"Rotation step: {angle_step:.1f} degrees")

    return frames


# =============================================================================
# Exercise 3: ID Card Scanner (Perspective Transform)
# =============================================================================

def exercise_3_id_card_scanner(img, src_points):
    """
    Apply perspective transformation to produce a front-view ID card.
    Standard ID card ratio: 85.6mm x 54mm (roughly 856x540 or scaled).

    Parameters:
        img: source image (H, W, C) or (H, W)
        src_points: 4 corner points as [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    ordered as top-left, top-right, bottom-right, bottom-left

    Returns:
        warped image of standard ID card proportions
    """
    # ID card aspect ratio: 85.6:54 ~= 1.585
    out_w = 428  # 85.6mm * 5
    out_h = 270  # 54mm * 5

    dst_points = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float64)

    src = np.array(src_points, dtype=np.float64)

    # Compute perspective transform matrix using SVD
    # For each point correspondence: src[i] -> dst[i]
    # We solve for 3x3 homography H such that dst = H * src (in homogeneous coords)
    n = 4
    A = np.zeros((2 * n, 9))
    for i in range(n):
        sx, sy = src[i]
        dx, dy = dst_points[i]
        A[2*i]   = [-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx]
        A[2*i+1] = [0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Inverse warp: for each destination pixel, find source pixel
    if len(img.shape) == 3:
        result = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    else:
        result = np.zeros((out_h, out_w), dtype=img.dtype)

    H_inv = np.linalg.inv(H)
    h_img, w_img = img.shape[:2]

    yy, xx = np.mgrid[0:out_h, 0:out_w]
    ones = np.ones_like(xx)
    dst_coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    src_coords = H_inv @ dst_coords
    src_coords = src_coords / src_coords[2:3]  # Normalize homogeneous

    src_x = np.round(src_coords[0]).astype(int)
    src_y = np.round(src_coords[1]).astype(int)

    valid = (src_x >= 0) & (src_x < w_img) & (src_y >= 0) & (src_y < h_img)
    flat_result = result.reshape(-1, *result.shape[2:]) if len(result.shape) == 3 else result.reshape(-1)

    valid_src_y = src_y[valid]
    valid_src_x = src_x[valid]

    if len(img.shape) == 3:
        flat_result[valid] = img[valid_src_y, valid_src_x]
    else:
        flat_result[valid] = img[valid_src_y, valid_src_x]

    print(f"Output size: {out_w}x{out_h} (ID card ratio 85.6:54)")
    print(f"Source points: {src_points}")
    print(f"Non-zero pixels: {np.sum(result > 0)}")

    return result


# =============================================================================
# Exercise 4: Image Mosaic
# =============================================================================

def exercise_4_image_mosaic(images, rows, cols, cell_size=(200, 200)):
    """
    Arrange images in a rows x cols grid, each resized to cell_size.

    Parameters:
        images: list of numpy arrays
        rows: number of grid rows
        cols: number of grid columns
        cell_size: (width, height) of each cell

    Returns:
        mosaic image
    """
    cell_w, cell_h = cell_size
    n_channels = 3 if any(len(img.shape) == 3 for img in images) else 1

    if n_channels == 3:
        mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    else:
        mosaic = np.zeros((rows * cell_h, cols * cell_w), dtype=np.uint8)

    def resize_nearest(img, new_w, new_h):
        old_h, old_w = img.shape[:2]
        row_idx = (np.arange(new_h) * old_h / new_h).astype(int)
        col_idx = (np.arange(new_w) * old_w / new_w).astype(int)
        row_idx = np.clip(row_idx, 0, old_h - 1)
        col_idx = np.clip(col_idx, 0, old_w - 1)
        return img[np.ix_(row_idx, col_idx)]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(images):
                break

            img = images[idx]
            resized = resize_nearest(img, cell_w, cell_h)

            y0 = r * cell_h
            x0 = c * cell_w

            if n_channels == 3 and len(resized.shape) == 2:
                resized = np.stack([resized] * 3, axis=-1)

            mosaic[y0:y0 + cell_h, x0:x0 + cell_w] = resized
            idx += 1

    print(f"Mosaic: {rows}x{cols} grid, cell={cell_w}x{cell_h}")
    print(f"Total size: {cols * cell_w}x{rows * cell_h}")
    print(f"Images placed: {min(len(images), rows * cols)}/{rows * cols}")

    return mosaic


# =============================================================================
# Exercise 5: AR Card Effect
# =============================================================================

def exercise_5_ar_card_effect(background, overlay, card_corners):
    """
    Overlay an image onto a detected card region using perspective transform.

    Parameters:
        background: (H, W, 3) background image
        overlay: (Oh, Ow, 3) image to overlay on the card
        card_corners: 4 detected card corners [(x,y), ...] in TL, TR, BR, BL order

    Returns:
        composited image
    """
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]

    src_pts = np.array([
        [0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]
    ], dtype=np.float64)
    dst_pts = np.array(card_corners, dtype=np.float64)

    # Compute homography: overlay -> background card region
    n = 4
    A = np.zeros((2 * n, 9))
    for i in range(n):
        sx, sy = src_pts[i]
        dx, dy = dst_pts[i]
        A[2*i]   = [-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx]
        A[2*i+1] = [0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy]

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    result = background.copy()

    # Forward warp: for each source pixel in overlay, find destination
    yy, xx = np.mgrid[0:oh, 0:ow]
    ones = np.ones_like(xx)
    src_coords = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3).T

    dst_coords = H @ src_coords
    dst_coords = dst_coords / dst_coords[2:3]

    dst_x = np.round(dst_coords[0]).astype(int)
    dst_y = np.round(dst_coords[1]).astype(int)

    valid = (dst_x >= 0) & (dst_x < bw) & (dst_y >= 0) & (dst_y < bh)

    src_flat = overlay.reshape(-1, overlay.shape[2]) if len(overlay.shape) == 3 else overlay.reshape(-1)
    result[dst_y[valid], dst_x[valid]] = src_flat[valid]

    placed_pixels = np.sum(valid)
    print(f"Overlay {ow}x{oh} -> card region")
    print(f"Card corners: {card_corners}")
    print(f"Pixels placed: {placed_pixels}")

    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Batch Resize")
    imgs = [
        ("large", np.random.randint(0, 255, (600, 1200, 3), dtype=np.uint8)),
        ("small", np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)),
        ("wide", np.random.randint(0, 255, (300, 2000, 3), dtype=np.uint8)),
    ]
    resized = exercise_1_batch_resize(imgs, max_width=800)

    print("\n>>> Exercise 2: Rotation Animation")
    small_img = np.zeros((50, 80, 3), dtype=np.uint8)
    small_img[10:40, 20:60] = [0, 255, 0]  # Green rectangle
    frames = exercise_2_rotation_animation(small_img, num_steps=8)

    print("\n>>> Exercise 3: ID Card Scanner")
    scene = np.random.randint(100, 200, (500, 700, 3), dtype=np.uint8)
    # Draw a "card" on the scene
    scene[100:300, 150:550] = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    card_pts = [(150, 100), (550, 100), (550, 300), (150, 300)]
    scanned = exercise_3_id_card_scanner(scene, card_pts)

    print("\n>>> Exercise 4: Image Mosaic")
    mosaic_imgs = [np.random.randint(i * 30, i * 30 + 50, (100 + i * 20, 80 + i * 15, 3),
                   dtype=np.uint8) for i in range(6)]
    mosaic = exercise_4_image_mosaic(mosaic_imgs, rows=2, cols=3, cell_size=(150, 150))

    print("\n>>> Exercise 5: AR Card Effect")
    bg = np.ones((400, 600, 3), dtype=np.uint8) * 180
    card_overlay = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    corners = [(100, 80), (300, 100), (280, 250), (80, 230)]
    composited = exercise_5_ar_card_effect(bg, card_overlay, corners)

    print("\nAll exercises completed successfully.")
