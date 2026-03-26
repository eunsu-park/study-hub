"""
Exercise Solutions for Lesson 14: Feature Matching
Computer Vision - BFMatcher, FLANN, Ratio Test, Homography

Topics covered:
- Find optimal ratio threshold
- Multiple object detection via clustering
- Real-time object tracking (simulated frame-by-frame)
"""

import numpy as np


# =============================================================================
# Helper: Simple feature descriptor (patch-based)
# =============================================================================

def extract_patch_descriptors(img, keypoints, patch_size=8):
    """
    Extract simple patch descriptors around each keypoint.
    Returns (keypoints_valid, descriptors) where descriptors is (N, D).
    """
    h, w = img.shape
    half = patch_size // 2
    valid_kps = []
    descs = []

    for x, y in keypoints:
        x, y = int(x), int(y)
        if half <= y < h - half and half <= x < w - half:
            patch = img[y-half:y+half, x-half:x+half].flatten().astype(np.float64)
            # Normalize patch
            std = patch.std()
            if std > 1:
                patch = (patch - patch.mean()) / std
            descs.append(patch)
            valid_kps.append((x, y))

    if descs:
        return valid_kps, np.array(descs)
    return [], np.array([])


def detect_keypoints(img, n=100):
    """Simple corner detection returning (x, y) list."""
    h, w = img.shape
    img_f = img.astype(np.float64)
    Ix = np.zeros_like(img_f)
    Iy = np.zeros_like(img_f)
    Ix[:, 1:-1] = (img_f[:, 2:] - img_f[:, :-2]) / 2
    Iy[1:-1, :] = (img_f[2:, :] - img_f[:-2, :]) / 2

    response = np.zeros_like(img_f)
    Ixx, Iyy, Ixy = Ix*Ix, Iy*Iy, Ix*Iy

    for i in range(1, h-1):
        for j in range(1, w-1):
            sxx = np.sum(Ixx[i-1:i+2, j-1:j+2])
            syy = np.sum(Iyy[i-1:i+2, j-1:j+2])
            sxy = np.sum(Ixy[i-1:i+2, j-1:j+2])
            det = sxx * syy - sxy**2
            trace = sxx + syy
            response[i, j] = det - 0.04 * trace**2

    thresh = response.max() * 0.01
    kps = []
    for i in range(2, h-2):
        for j in range(2, w-2):
            if response[i, j] > thresh:
                local = response[i-1:i+2, j-1:j+2]
                if response[i, j] == local.max():
                    kps.append((j, i, response[i, j]))

    kps.sort(key=lambda x: x[2], reverse=True)
    return [(x, y) for x, y, _ in kps[:n]]


def match_descriptors(desc1, desc2, k=2):
    """Brute-force kNN matching using L2 distance."""
    matches = []
    for i in range(len(desc1)):
        dists = np.sqrt(np.sum((desc2 - desc1[i])**2, axis=1))
        sorted_idx = np.argsort(dists)
        if len(sorted_idx) >= k:
            matches.append([(sorted_idx[j], dists[sorted_idx[j]]) for j in range(k)])
    return matches


# =============================================================================
# Exercise 1: Find Optimal Ratio Threshold
# =============================================================================

def exercise_1_optimal_ratio(img1, img2):
    """
    Test various ratio threshold values and find the optimal one.
    Plots (simulates) match count vs ratio threshold.

    Parameters:
        img1, img2: grayscale images

    Returns:
        optimal ratio threshold
    """
    kps1 = detect_keypoints(img1, n=80)
    kps2 = detect_keypoints(img2, n=80)

    kps1_valid, desc1 = extract_patch_descriptors(img1, kps1)
    kps2_valid, desc2 = extract_patch_descriptors(img2, kps2)

    if len(desc1) == 0 or len(desc2) == 0:
        print("Not enough descriptors for matching")
        return 0.75

    # k=2 matching
    matches = match_descriptors(desc1, desc2, k=2)

    ratios = np.arange(0.5, 1.0, 0.05)
    results = []

    print(f"Keypoints: img1={len(kps1_valid)}, img2={len(kps2_valid)}")
    print(f"\n{'Ratio':>8} | {'Matches':>8} | {'Bar':>20}")
    print("-" * 42)

    for ratio in ratios:
        good = 0
        for m in matches:
            if len(m) >= 2:
                best_dist = m[0][1]
                second_dist = m[1][1]
                if second_dist > 0 and best_dist / second_dist < ratio:
                    good += 1
        results.append(good)
        bar = "#" * (good // 2)
        print(f"{ratio:>8.2f} | {good:>8} | {bar}")

    # Find optimal: look for the "elbow" where gradient changes most
    gradients = np.diff(results)
    if len(gradients) > 0:
        optimal_idx = np.argmax(np.abs(gradients))
        optimal_ratio = ratios[optimal_idx]
    else:
        optimal_ratio = 0.75

    print(f"\nRecommended ratio threshold: {optimal_ratio:.2f}")
    return optimal_ratio


# =============================================================================
# Exercise 2: Multiple Object Detection
# =============================================================================

def exercise_2_multiple_objects(template, scene):
    """
    Detect multiple instances of a template in a scene using
    feature matching and spatial clustering.

    Parameters:
        template: grayscale template image
        scene: grayscale scene image

    Returns:
        list of detected object centers
    """
    kps_t = detect_keypoints(template, n=50)
    kps_s = detect_keypoints(scene, n=200)

    kps_t_valid, desc_t = extract_patch_descriptors(template, kps_t)
    kps_s_valid, desc_s = extract_patch_descriptors(scene, kps_s)

    if len(desc_t) == 0 or len(desc_s) == 0:
        print("Not enough descriptors")
        return []

    # Match
    matches = match_descriptors(desc_t, desc_s, k=2)

    # Ratio test
    good_matches = []
    for m in matches:
        if len(m) >= 2 and m[1][1] > 0:
            if m[0][1] / m[1][1] < 0.75:
                good_matches.append(m[0][0])  # Index in scene

    print(f"Template keypoints: {len(kps_t_valid)}")
    print(f"Scene keypoints: {len(kps_s_valid)}")
    print(f"Good matches: {len(good_matches)}")

    if len(good_matches) < 3:
        print("Insufficient matches for detection")
        return []

    # Get matched scene point locations
    match_pts = np.array([kps_s_valid[idx] for idx in good_matches], dtype=np.float64)

    # Simple spatial clustering (mean-shift like)
    cluster_radius = max(template.shape) * 0.8
    used = np.zeros(len(match_pts), dtype=bool)
    clusters = []

    for i in range(len(match_pts)):
        if used[i]:
            continue

        cluster = [i]
        used[i] = True

        for j in range(i + 1, len(match_pts)):
            if not used[j]:
                dist = np.sqrt(np.sum((match_pts[i] - match_pts[j])**2))
                if dist < cluster_radius:
                    cluster.append(j)
                    used[j] = True

        if len(cluster) >= 3:  # Minimum matches per object
            center = np.mean(match_pts[cluster], axis=0)
            clusters.append(center)

    print(f"Detected objects: {len(clusters)}")
    for i, center in enumerate(clusters):
        print(f"  Object {i+1}: center=({center[0]:.0f}, {center[1]:.0f})")

    return clusters


# =============================================================================
# Exercise 3: Object Tracking (Frame-by-Frame Simulation)
# =============================================================================

def exercise_3_object_tracking():
    """
    Simulate real-time object tracking across multiple frames
    by matching features between consecutive frames.

    Returns:
        list of tracked positions per frame
    """
    # Create template (a simple pattern)
    template = np.zeros((30, 30), dtype=np.uint8)
    template[5:25, 5:25] = 200
    template[10:20, 10:20] = 100
    template[12:18, 12:18] = 250

    # Create synthetic video frames with the object at different positions
    h, w = 100, 150
    positions = [(20, 30), (35, 40), (50, 50), (65, 55), (80, 50)]
    th, tw = template.shape

    frames = []
    for px, py in positions:
        frame = np.random.randint(40, 80, (h, w), dtype=np.uint8)
        # Place template at position
        y1 = max(0, py)
        y2 = min(h, py + th)
        x1 = max(0, px)
        x2 = min(w, px + tw)
        ty1 = y1 - py
        tx1 = x1 - px
        frame[y1:y2, x1:x2] = template[ty1:ty1+(y2-y1), tx1:tx1+(x2-x1)]
        frames.append(frame)

    # Track object across frames
    tracked_positions = []
    ref_kps = detect_keypoints(template, n=20)
    ref_kps_valid, ref_desc = extract_patch_descriptors(template, ref_kps)

    print(f"Template size: {tw}x{th}")
    print(f"Template features: {len(ref_kps_valid)}")
    print(f"Frames: {len(frames)}")
    print()

    for frame_idx, frame in enumerate(frames):
        frame_kps = detect_keypoints(frame, n=100)
        frame_kps_valid, frame_desc = extract_patch_descriptors(frame, frame_kps)

        if len(ref_desc) == 0 or len(frame_desc) == 0:
            tracked_positions.append(None)
            print(f"  Frame {frame_idx}: No features")
            continue

        matches = match_descriptors(ref_desc, frame_desc, k=2)

        good_pts = []
        for m in matches:
            if len(m) >= 2 and m[1][1] > 0:
                if m[0][1] / m[1][1] < 0.8:
                    idx = m[0][0]
                    good_pts.append(frame_kps_valid[idx])

        if good_pts:
            center = np.mean(good_pts, axis=0)
            tracked_positions.append(center)
            actual = positions[frame_idx]
            error = np.sqrt((center[0] - actual[0])**2 + (center[1] - actual[1])**2)
            print(f"  Frame {frame_idx}: tracked=({center[0]:.0f},{center[1]:.0f}), "
                  f"actual=({actual[0]},{actual[1]}), matches={len(good_pts)}, "
                  f"error={error:.1f}px")
        else:
            tracked_positions.append(None)
            print(f"  Frame {frame_idx}: Lost tracking")

    return tracked_positions


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Find Optimal Ratio Threshold")
    img1 = np.zeros((80, 100), dtype=np.uint8)
    img1[10:60, 10:80] = 180
    img1[20:50, 30:60] = 80
    img2 = img1.copy()
    # Add slight transformation
    img2[5:55, 15:85] = img1[10:60, 10:80]
    exercise_1_optimal_ratio(img1, img2)

    print("\n>>> Exercise 2: Multiple Object Detection")
    template = np.zeros((25, 25), dtype=np.uint8)
    template[5:20, 5:20] = 200
    template[8:17, 8:17] = 100

    scene = np.random.randint(40, 80, (100, 150), dtype=np.uint8)
    # Place template at two locations
    scene[10:35, 10:35] = template
    scene[50:75, 80:105] = template
    exercise_2_multiple_objects(template, scene)

    print("\n>>> Exercise 3: Object Tracking")
    exercise_3_object_tracking()

    print("\nAll exercises completed successfully.")
