"""
Exercise Solutions for Lesson 32: Synthetic Data Generation
Computer Vision - Domain Randomization, Procedural Generation, Domain Gap Analysis

Topics covered:
- Procedural scene and annotation generation
- Domain randomization (lighting, color, noise, texture)
- Run-length encoding for mask annotations
- Domain gap measurement and analysis
- Synthetic-to-real transfer simulation
- Data mixing strategy evaluation
"""

import numpy as np


# =============================================================================
# Helper: Scene and annotation utilities
# =============================================================================

def generate_background(h=64, w=80, style="gradient"):
    """Generate a synthetic background image."""
    np.random.seed(hash(style) % 2**31)
    bg = np.zeros((h, w, 3), dtype=np.float64)

    if style == "gradient":
        for i in range(h):
            for j in range(w):
                bg[i, j, 0] = 0.3 + 0.4 * (i / h)
                bg[i, j, 1] = 0.5 - 0.2 * (j / w)
                bg[i, j, 2] = 0.2 + 0.3 * ((i + j) / (h + w))

    elif style == "noise":
        bg = np.random.uniform(0.2, 0.5, (h, w, 3))

    elif style == "checkerboard":
        for i in range(h):
            for j in range(w):
                if (i // 8 + j // 8) % 2 == 0:
                    bg[i, j] = [0.4, 0.4, 0.4]
                else:
                    bg[i, j] = [0.6, 0.6, 0.6]

    return bg


def place_object(image, mask, cx, cy, radius, color, shape="circle"):
    """Place a synthetic object onto the scene, updating image and mask."""
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]

    if shape == "circle":
        obj_mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2
    elif shape == "rectangle":
        obj_mask = ((abs(xx - cx) <= radius) & (abs(yy - cy) <= radius * 0.7))
    elif shape == "triangle":
        obj_mask = np.zeros((h, w), dtype=bool)
        for i in range(h):
            for j in range(w):
                # Simple triangle test
                dy = cy + radius - i
                dx = abs(j - cx)
                if 0 <= dy <= 2 * radius and dx <= dy * 0.6:
                    obj_mask[i, j] = True
    else:
        obj_mask = ((xx - cx)**2 + (yy - cy)**2) <= radius**2

    image[obj_mask] = color
    mask[obj_mask] = True
    return image, mask


# =============================================================================
# Exercise 1: Procedural Scene Generation
# =============================================================================

def exercise_1_procedural_generation():
    """
    Generate synthetic scenes with automatic annotations.

    Steps:
    1. Random background selection
    2. Random object placement (with occlusion handling)
    3. Generate bounding box and mask annotations
    4. Produce COCO-format metadata

    Returns:
        list of scene dicts with images and annotations
    """
    np.random.seed(42)
    h, w = 64, 80
    n_scenes = 5
    shapes = ["circle", "rectangle", "triangle"]
    bg_styles = ["gradient", "noise", "checkerboard"]

    print("Procedural Scene Generation")
    print(f"  Image size: {w}x{h}")
    print(f"  Scenes: {n_scenes}")
    print("=" * 60)

    all_scenes = []

    for scene_idx in range(n_scenes):
        # Random background
        bg_style = bg_styles[scene_idx % len(bg_styles)]
        image = generate_background(h, w, bg_style)

        # Random number of objects
        n_objects = np.random.randint(2, 6)
        annotations = []
        occupied = np.zeros((h, w), dtype=bool)

        for obj_idx in range(n_objects):
            # Random object properties
            cx = np.random.randint(15, w - 15)
            cy = np.random.randint(15, h - 15)
            radius = np.random.randint(5, 15)
            color = np.random.uniform(0.3, 1.0, 3)
            shape = shapes[np.random.randint(len(shapes))]

            # Place object
            obj_mask = np.zeros((h, w), dtype=bool)
            image, obj_mask = place_object(image, obj_mask, cx, cy,
                                           radius, color, shape)

            # Compute occlusion
            visible = obj_mask & (~occupied)
            visible_ratio = visible.sum() / (obj_mask.sum() + 1e-10)
            occupied |= obj_mask

            # Bounding box
            ys, xs = np.where(obj_mask)
            if len(ys) > 0:
                bbox = [int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max())]
                area = int(obj_mask.sum())
            else:
                bbox = [0, 0, 0, 0]
                area = 0

            annotations.append({
                'id': obj_idx,
                'category': shape,
                'bbox': bbox,
                'area': area,
                'visible_ratio': float(visible_ratio),
                'color': color.tolist(),
            })

        scene = {
            'image': image,
            'annotations': annotations,
            'background': bg_style,
        }
        all_scenes.append(scene)

        print(f"\n  Scene {scene_idx} (bg={bg_style}):")
        for ann in annotations:
            bbox_str = (f"[{ann['bbox'][0]},{ann['bbox'][1]},"
                       f"{ann['bbox'][2]},{ann['bbox'][3]}]")
            print(f"    {ann['category']:>10}: bbox={bbox_str}, "
                  f"area={ann['area']}, visible={ann['visible_ratio']:.2f}")

    # Dataset statistics
    total_objects = sum(len(s['annotations']) for s in all_scenes)
    shape_counts = {}
    for s in all_scenes:
        for ann in s['annotations']:
            cat = ann['category']
            shape_counts[cat] = shape_counts.get(cat, 0) + 1

    print(f"\n  Dataset Statistics:")
    print(f"    Total scenes: {n_scenes}")
    print(f"    Total objects: {total_objects}")
    for shape, count in sorted(shape_counts.items()):
        print(f"    {shape}: {count}")

    return all_scenes


# =============================================================================
# Exercise 2: Domain Randomization
# =============================================================================

def exercise_2_domain_randomization():
    """
    Apply domain randomization augmentations to synthetic images.

    Augmentations:
    1. Random brightness/contrast
    2. Random color jitter (HSV shifts)
    3. Random Gaussian noise
    4. Random blur
    5. Random texture overlay

    Returns:
        dict of augmentation -> augmented image
    """
    np.random.seed(42)
    h, w = 64, 80

    # Generate base synthetic image
    image = generate_background(h, w, "gradient")
    obj_mask = np.zeros((h, w), dtype=bool)
    image, obj_mask = place_object(image, obj_mask, 40, 30, 12,
                                   [0.8, 0.3, 0.2], "circle")
    image, _ = place_object(image, np.zeros((h, w), dtype=bool),
                           60, 45, 8, [0.2, 0.7, 0.4], "rectangle")

    print("Domain Randomization")
    print(f"  Base image: {w}x{h}")
    print(f"  Base stats: mean={image.mean():.3f}, std={image.std():.3f}")
    print("=" * 60)

    def random_brightness(img, low=0.5, high=1.5):
        factor = np.random.uniform(low, high)
        return np.clip(img * factor, 0, 1)

    def random_contrast(img, low=0.7, high=1.3):
        factor = np.random.uniform(low, high)
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 1)

    def random_noise(img, sigma_range=(0.01, 0.08)):
        sigma = np.random.uniform(*sigma_range)
        noise = np.random.randn(*img.shape) * sigma
        return np.clip(img + noise, 0, 1)

    def random_color_jitter(img, hue_range=0.05, sat_range=0.3):
        jittered = img.copy()
        # Per-channel shift
        for c in range(3):
            shift = np.random.uniform(-hue_range, hue_range)
            scale = np.random.uniform(1 - sat_range, 1 + sat_range)
            jittered[:, :, c] = np.clip(jittered[:, :, c] * scale + shift, 0, 1)
        return jittered

    def random_blur(img, max_kernel=7):
        ksize = np.random.choice([3, 5, 7])
        sigma = ksize / 3.0
        kernel = np.zeros((ksize, ksize), dtype=np.float64)
        center = ksize // 2
        for i in range(ksize):
            for j in range(ksize):
                kernel[i, j] = np.exp(-((i-center)**2 + (j-center)**2) / (2*sigma**2))
        kernel /= kernel.sum()

        # Apply to each channel
        blurred = img.copy()
        half = ksize // 2
        for c in range(3):
            padded = np.pad(img[:, :, c], half, mode='reflect')
            for yi in range(h):
                for xi in range(w):
                    blurred[yi, xi, c] = np.sum(
                        padded[yi:yi+ksize, xi:xi+ksize] * kernel
                    )
        return np.clip(blurred, 0, 1)

    def random_texture_overlay(img, alpha_range=(0.05, 0.15)):
        alpha = np.random.uniform(*alpha_range)
        texture = np.random.uniform(0, 1, img.shape)
        return np.clip(img * (1 - alpha) + texture * alpha, 0, 1)

    augmentations = {
        'brightness': random_brightness,
        'contrast': random_contrast,
        'noise': random_noise,
        'color_jitter': random_color_jitter,
        'blur': random_blur,
        'texture': random_texture_overlay,
    }

    results = {}
    for name, aug_fn in augmentations.items():
        augmented = aug_fn(image.copy())
        diff = np.abs(augmented - image)

        results[name] = augmented

        print(f"\n  {name}:")
        print(f"    Mean: {augmented.mean():.4f} (base: {image.mean():.4f})")
        print(f"    Std:  {augmented.std():.4f} (base: {image.std():.4f})")
        print(f"    Mean diff: {diff.mean():.4f}")
        print(f"    Max diff:  {diff.max():.4f}")

    # Combined randomization (apply multiple)
    print(f"\n  Combined Randomization (3 random augmentations):")
    n_variants = 5
    for v in range(n_variants):
        np.random.seed(v)
        combined = image.copy()
        selected = np.random.choice(list(augmentations.keys()), 3, replace=False)
        for aug_name in selected:
            combined = augmentations[aug_name](combined)

        diff = np.abs(combined - image).mean()
        augs_str = "+".join(selected)
        print(f"    Variant {v}: {augs_str} -> diff={diff:.4f}")

    return results


# =============================================================================
# Exercise 3: Annotation Format Conversion
# =============================================================================

def exercise_3_annotation_formats():
    """
    Convert between annotation formats: polygon, RLE, bitmap.

    Steps:
    1. Generate a binary mask
    2. Convert to polygon (contour)
    3. Convert to RLE (run-length encoding)
    4. Convert back to bitmap and verify losslessness

    Returns:
        dict of format -> data
    """
    np.random.seed(42)
    h, w = 48, 64

    print("Annotation Format Conversion")
    print(f"  Mask size: {w}x{h}")
    print("=" * 60)

    # Generate a mask (elliptical object)
    yy, xx = np.ogrid[:h, :w]
    mask = (((xx - 30) / 15)**2 + ((yy - 24) / 10)**2) <= 1.0
    mask = mask.astype(np.uint8)
    area = int(mask.sum())

    print(f"\n  Original mask: area={area} pixels")

    # 1. Bitmap (the mask itself)
    bitmap_size = mask.nbytes
    print(f"\n  [1] Bitmap:")
    print(f"    Size: {bitmap_size} bytes")
    print(f"    Shape: {mask.shape}")

    # 2. RLE encoding
    flat = mask.flatten()
    rle_counts = []
    current = flat[0]
    count = 1

    for i in range(1, len(flat)):
        if flat[i] == current:
            count += 1
        else:
            rle_counts.append((int(current), count))
            current = flat[i]
            count = 1
    rle_counts.append((int(current), count))

    rle_size = len(rle_counts) * 2 * 4  # 2 ints per run
    compression = rle_size / bitmap_size

    print(f"\n  [2] Run-Length Encoding:")
    print(f"    Runs: {len(rle_counts)}")
    print(f"    Size: {rle_size} bytes (compression: {compression:.3f}x)")
    print(f"    First 5 runs: {rle_counts[:5]}")

    # Decode RLE back to bitmap
    decoded = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for value, count in rle_counts:
        decoded[pos:pos+count] = value
        pos += count
    decoded = decoded.reshape(h, w)

    rle_lossless = np.array_equal(mask, decoded)
    print(f"    Lossless: {rle_lossless}")

    # 3. Polygon (contour extraction)
    contour_points = []
    # Simple contour: scan for boundary pixels
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mask[i, j] == 1:
                neighbors = mask[i-1:i+2, j-1:j+2]
                if neighbors.sum() < 9:  # Not fully interior
                    contour_points.append((j, i))

    # Simplify contour (keep every nth point)
    step = max(1, len(contour_points) // 20)
    simplified = contour_points[::step]
    polygon_size = len(simplified) * 2 * 4  # 2 coords per point, 4 bytes each

    print(f"\n  [3] Polygon (contour):")
    print(f"    Full contour points: {len(contour_points)}")
    print(f"    Simplified points: {len(simplified)}")
    print(f"    Size: {polygon_size} bytes "
          f"(compression: {polygon_size/bitmap_size:.3f}x)")

    # Rasterize polygon back to mask
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    # Simple scan-line fill (approximate)
    if simplified:
        for i in range(h):
            row_pts = sorted([p[0] for p in simplified if abs(p[1] - i) <= 2])
            if len(row_pts) >= 2:
                poly_mask[i, row_pts[0]:row_pts[-1]+1] = 1

    poly_iou = np.sum(poly_mask & mask) / (np.sum(poly_mask | mask) + 1e-10)
    print(f"    Polygon-to-bitmap IoU: {poly_iou:.4f}")

    # 4. COCO-style annotation
    coco_annotation = {
        'segmentation': [coord for point in simplified for coord in point],
        'bbox': [int(xx[mask > 0].min()), int(yy[mask > 0].min()),
                 int(xx[mask > 0].max() - xx[mask > 0].min()),
                 int(yy[mask > 0].max() - yy[mask > 0].min())],
        'area': area,
        'iscrowd': 0,
    }

    print(f"\n  [4] COCO Format:")
    print(f"    BBox: {coco_annotation['bbox']}")
    print(f"    Area: {coco_annotation['area']}")
    print(f"    Polygon points: {len(simplified)}")

    # Comparison summary
    print(f"\n  Format Comparison:")
    print(f"    {'Format':>12} | {'Size (bytes)':>12} | {'Compression':>12} | {'Lossless':>8}")
    print(f"    {'-'*50}")
    print(f"    {'Bitmap':>12} | {bitmap_size:>12} | {'1.000x':>12} | {'Yes':>8}")
    print(f"    {'RLE':>12} | {rle_size:>12} | {compression:>11.3f}x | {'Yes':>8}")
    print(f"    {'Polygon':>12} | {polygon_size:>12} | {polygon_size/bitmap_size:>11.3f}x | {'~No':>8}")

    return {
        'bitmap': mask,
        'rle': rle_counts,
        'polygon': simplified,
        'coco': coco_annotation,
    }


# =============================================================================
# Exercise 4: Domain Gap Analysis
# =============================================================================

def exercise_4_domain_gap():
    """
    Measure and analyze the domain gap between synthetic and real data.

    Steps:
    1. Generate synthetic and simulated-real feature distributions
    2. Compute distribution distance metrics (MMD, KL divergence)
    3. Analyze per-feature domain shift
    4. Simulate the effect of domain adaptation

    Returns:
        dict of metrics
    """
    np.random.seed(42)
    n_synthetic = 200
    n_real = 200
    n_features = 8

    print("Domain Gap Analysis")
    print(f"  Synthetic samples: {n_synthetic}")
    print(f"  Real samples: {n_real}")
    print(f"  Feature dim: {n_features}")
    print("=" * 60)

    # Simulate feature distributions
    # Synthetic: clean, regular patterns
    synthetic_features = np.random.randn(n_synthetic, n_features) * 0.8
    synthetic_features[:, 0] += 1.0  # Brightness bias
    synthetic_features[:, 1] += 0.5  # Contrast bias

    # Real: noisier, shifted distribution
    real_features = np.random.randn(n_real, n_features) * 1.2
    real_features[:, 0] += 0.3  # Different brightness
    real_features[:, 1] -= 0.2  # Different contrast
    real_features[:, 2] += 0.8  # Texture difference
    real_features += np.random.randn(n_real, n_features) * 0.3  # More noise

    # 1. Per-feature statistics
    print(f"\n  Per-Feature Distribution:")
    print(f"    {'Feature':>8} | {'Synth mean':>10} | {'Real mean':>10} | "
          f"{'Gap':>6} | {'Synth std':>9} | {'Real std':>8}")
    print(f"    {'-'*60}")

    for f in range(n_features):
        s_mean = synthetic_features[:, f].mean()
        r_mean = real_features[:, f].mean()
        gap = abs(s_mean - r_mean)
        s_std = synthetic_features[:, f].std()
        r_std = real_features[:, f].std()
        print(f"    feat_{f:>3} | {s_mean:>10.3f} | {r_mean:>10.3f} | "
              f"{gap:>6.3f} | {s_std:>9.3f} | {r_std:>8.3f}")

    # 2. Maximum Mean Discrepancy (MMD) - simplified
    def compute_mmd(X, Y, gamma=1.0):
        """Compute MMD between two distributions using RBF kernel."""
        def rbf_kernel(a, b, gamma):
            sq_dist = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
            return np.exp(-gamma * sq_dist)

        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)

        mmd = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
        return max(0, mmd)

    mmd = compute_mmd(synthetic_features, real_features, gamma=0.5)
    print(f"\n  MMD (RBF kernel): {mmd:.6f}")

    # 3. KL Divergence (per-feature, assuming Gaussian)
    total_kl = 0
    for f in range(n_features):
        s_mu = synthetic_features[:, f].mean()
        s_var = synthetic_features[:, f].var() + 1e-10
        r_mu = real_features[:, f].mean()
        r_var = real_features[:, f].var() + 1e-10

        kl = (np.log(r_var / s_var) + (s_var + (s_mu - r_mu)**2) / r_var - 1) / 2
        total_kl += kl

    print(f"  Total KL Divergence: {total_kl:.4f}")

    # 4. Euclidean distance of means
    mean_dist = np.linalg.norm(
        synthetic_features.mean(axis=0) - real_features.mean(axis=0)
    )
    print(f"  Mean Euclidean distance: {mean_dist:.4f}")

    # 5. Domain adaptation simulation
    print(f"\n  Domain Adaptation Simulation:")

    # Simple adaptation: shift synthetic to match real statistics
    adapted = synthetic_features.copy()
    for f in range(n_features):
        s_mean = synthetic_features[:, f].mean()
        s_std = synthetic_features[:, f].std()
        r_mean = real_features[:, f].mean()
        r_std = real_features[:, f].std()

        # Normalize and re-scale
        adapted[:, f] = (adapted[:, f] - s_mean) / (s_std + 1e-10) * r_std + r_mean

    adapted_mmd = compute_mmd(adapted, real_features, gamma=0.5)
    adapted_dist = np.linalg.norm(
        adapted.mean(axis=0) - real_features.mean(axis=0)
    )

    print(f"    Before adaptation: MMD={mmd:.6f}, dist={mean_dist:.4f}")
    print(f"    After adaptation:  MMD={adapted_mmd:.6f}, dist={adapted_dist:.4f}")
    print(f"    MMD reduction: {(1 - adapted_mmd/(mmd+1e-10))*100:.1f}%")

    return {
        'mmd': mmd,
        'kl_divergence': total_kl,
        'mean_distance': mean_dist,
        'adapted_mmd': adapted_mmd,
    }


# =============================================================================
# Exercise 5: Synthetic-to-Real Transfer
# =============================================================================

def exercise_5_transfer_simulation():
    """
    Simulate synthetic-to-real transfer learning.

    Training strategies:
    1. Real only (small dataset)
    2. Synthetic only
    3. Synthetic pre-train + real fine-tune
    4. Mixed training (synthetic + real)

    Returns:
        dict of strategy -> simulated accuracy
    """
    np.random.seed(42)
    n_classes = 4
    n_real_train = 50
    n_synthetic_train = 500
    n_test = 100

    print("Synthetic-to-Real Transfer Simulation")
    print(f"  Classes: {n_classes}")
    print(f"  Real train: {n_real_train}, Synthetic: {n_synthetic_train}")
    print(f"  Test (real): {n_test}")
    print("=" * 60)

    # Generate features
    n_features = 6

    def generate_data(n, domain="real"):
        X = np.random.randn(n, n_features)
        y = np.random.randint(0, n_classes, n)
        # Class-specific patterns
        for i in range(n):
            X[i, :2] += y[i] * 0.8  # Class separation

        if domain == "synthetic":
            X += 0.5  # Domain shift
            X *= 0.8  # Different scale
        elif domain == "real":
            X += np.random.randn(n, n_features) * 0.3  # More noise

        return X, y

    real_X, real_y = generate_data(n_real_train, "real")
    synth_X, synth_y = generate_data(n_synthetic_train, "synthetic")
    test_X, test_y = generate_data(n_test, "real")

    # Simple k-NN classifier for evaluation
    def knn_accuracy(train_X, train_y, test_X, test_y, k=5):
        correct = 0
        for i in range(len(test_X)):
            dists = np.sqrt(np.sum((train_X - test_X[i])**2, axis=1))
            nn_indices = np.argsort(dists)[:k]
            nn_labels = train_y[nn_indices]
            # Majority vote
            votes = np.bincount(nn_labels, minlength=n_classes)
            pred = np.argmax(votes)
            if pred == test_y[i]:
                correct += 1
        return correct / len(test_y)

    # Strategy 1: Real only
    acc_real = knn_accuracy(real_X, real_y, test_X, test_y)

    # Strategy 2: Synthetic only
    acc_synth = knn_accuracy(synth_X, synth_y, test_X, test_y)

    # Strategy 3: Pre-train on synthetic, fine-tune on real
    # Simulate by adapting synthetic features
    adapted_synth = synth_X.copy()
    for f in range(n_features):
        s_mean = synth_X[:, f].mean()
        s_std = synth_X[:, f].std() + 1e-10
        r_mean = real_X[:, f].mean()
        r_std = real_X[:, f].std() + 1e-10
        adapted_synth[:, f] = (adapted_synth[:, f] - s_mean) / s_std * r_std + r_mean

    combined_pretrain_X = np.vstack([adapted_synth, real_X])
    combined_pretrain_y = np.concatenate([synth_y, real_y])
    acc_pretrain = knn_accuracy(combined_pretrain_X, combined_pretrain_y,
                                test_X, test_y)

    # Strategy 4: Mixed training
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    mixed_results = {}

    for ratio in ratios:
        n_synth_use = int(n_synthetic_train * ratio)
        n_real_use = n_real_train

        if n_synth_use > 0:
            idx = np.random.choice(n_synthetic_train, n_synth_use, replace=False)
            mix_X = np.vstack([real_X[:n_real_use], synth_X[idx]])
            mix_y = np.concatenate([real_y[:n_real_use], synth_y[idx]])
        else:
            mix_X = real_X[:n_real_use]
            mix_y = real_y[:n_real_use]

        acc = knn_accuracy(mix_X, mix_y, test_X, test_y)
        mixed_results[ratio] = acc

    print(f"\n  Training Strategy Comparison:")
    print(f"    {'Strategy':>30} | {'Accuracy':>8}")
    print(f"    {'-'*42}")
    print(f"    {'Real only (50 samples)':>30} | {acc_real:>8.4f}")
    print(f"    {'Synthetic only (500 samples)':>30} | {acc_synth:>8.4f}")
    print(f"    {'Synth pretrain + Real finetune':>30} | {acc_pretrain:>8.4f}")

    print(f"\n  Mixed Training (varying synthetic ratio):")
    print(f"    {'Synth Ratio':>12} | {'Total Train':>12} | {'Accuracy':>8}")
    print(f"    {'-'*38}")
    for ratio, acc in mixed_results.items():
        total = n_real_train + int(n_synthetic_train * ratio)
        print(f"    {ratio:>12.2f} | {total:>12} | {acc:>8.4f}")

    # Best strategy
    all_strats = {
        'real_only': acc_real,
        'synth_only': acc_synth,
        'pretrain_finetune': acc_pretrain,
    }
    for ratio, acc in mixed_results.items():
        all_strats[f'mixed_{ratio:.2f}'] = acc

    best_name = max(all_strats, key=all_strats.get)
    best_acc = all_strats[best_name]
    print(f"\n  Best strategy: {best_name} ({best_acc:.4f})")

    return all_strats


# =============================================================================
# Exercise 6: Data Quality Assessment
# =============================================================================

def exercise_6_data_quality():
    """
    Assess quality of synthetic data for training.

    Metrics:
    1. Annotation accuracy (mask precision)
    2. Class balance
    3. Scene diversity
    4. Domain coverage

    Returns:
        dict of quality metrics
    """
    np.random.seed(42)
    h, w = 64, 80
    n_images = 20

    print("Synthetic Data Quality Assessment")
    print(f"  Dataset: {n_images} images, {w}x{h}")
    print("=" * 60)

    # Generate dataset
    all_annotations = []
    all_areas = []
    all_positions = []
    all_categories = []
    category_names = ["circle", "rectangle", "triangle"]

    for img_idx in range(n_images):
        n_objects = np.random.randint(1, 6)
        for obj_idx in range(n_objects):
            cx = np.random.randint(10, w - 10)
            cy = np.random.randint(10, h - 10)
            radius = np.random.randint(5, 15)
            cat = np.random.randint(len(category_names))

            area = int(np.pi * radius * radius)
            all_areas.append(area)
            all_positions.append((cx, cy))
            all_categories.append(cat)
            all_annotations.append({
                'image_id': img_idx,
                'category': cat,
                'center': (cx, cy),
                'radius': radius,
                'area': area,
            })

    # 1. Class balance
    print(f"\n  [1] Class Balance:")
    cat_counts = np.bincount(all_categories, minlength=len(category_names))
    total = len(all_categories)
    max_count = cat_counts.max()
    for i, name in enumerate(category_names):
        balance = cat_counts[i] / max_count
        bar = "#" * int(balance * 30)
        print(f"    {name:>10}: {cat_counts[i]:>3} ({100*cat_counts[i]/total:.0f}%) {bar}")

    imbalance = cat_counts.max() / (cat_counts.min() + 1)
    print(f"    Imbalance ratio: {imbalance:.2f}")

    # 2. Area distribution
    areas = np.array(all_areas)
    print(f"\n  [2] Object Area Distribution:")
    print(f"    Range: [{areas.min()}, {areas.max()}]")
    print(f"    Mean: {areas.mean():.1f}, Std: {areas.std():.1f}")
    print(f"    Median: {np.median(areas):.1f}")

    # Size buckets
    size_bins = [(0, 100, "small"), (100, 300, "medium"), (300, 1000, "large")]
    for low, high, label in size_bins:
        count = np.sum((areas >= low) & (areas < high))
        print(f"    {label:>8} ({low}-{high}): {count} objects")

    # 3. Spatial diversity
    positions = np.array(all_positions, dtype=np.float64)
    print(f"\n  [3] Spatial Diversity:")

    # Divide into quadrants
    q_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    q_counts = [0, 0, 0, 0]
    for cx, cy in positions:
        qi = 0
        if cx >= w / 2:
            qi += 1
        if cy >= h / 2:
            qi += 2
        q_counts[qi] += 1

    for name, count in zip(q_names, q_counts):
        pct = 100 * count / len(positions)
        print(f"    {name:>12}: {count:>3} ({pct:.0f}%)")

    spatial_uniformity = 1 - np.std(q_counts) / (np.mean(q_counts) + 1e-10)
    print(f"    Spatial uniformity: {spatial_uniformity:.3f} (1.0 = perfect)")

    # 4. Objects per image distribution
    img_counts = np.bincount(
        [ann['image_id'] for ann in all_annotations],
        minlength=n_images
    )
    print(f"\n  [4] Objects Per Image:")
    print(f"    Range: [{img_counts.min()}, {img_counts.max()}]")
    print(f"    Mean: {img_counts.mean():.1f}")

    # 5. Overall quality score
    balance_score = 1.0 / imbalance
    diversity_score = spatial_uniformity
    size_coverage = len([a for a in areas if a < 100]) > 0 and len([a for a in areas if a > 300]) > 0
    size_score = 1.0 if size_coverage else 0.5

    overall = (balance_score + diversity_score + size_score) / 3
    print(f"\n  [5] Quality Summary:")
    print(f"    Class balance score:  {balance_score:.3f}")
    print(f"    Spatial diversity:    {diversity_score:.3f}")
    print(f"    Size coverage:        {size_score:.3f}")
    print(f"    Overall quality:      {overall:.3f}")

    return {
        'class_balance': balance_score,
        'spatial_diversity': diversity_score,
        'size_coverage': size_score,
        'overall': overall,
        'imbalance_ratio': imbalance,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Procedural Scene Generation")
    exercise_1_procedural_generation()

    print("\n>>> Exercise 2: Domain Randomization")
    exercise_2_domain_randomization()

    print("\n>>> Exercise 3: Annotation Format Conversion")
    exercise_3_annotation_formats()

    print("\n>>> Exercise 4: Domain Gap Analysis")
    exercise_4_domain_gap()

    print("\n>>> Exercise 5: Synthetic-to-Real Transfer")
    exercise_5_transfer_simulation()

    print("\n>>> Exercise 6: Data Quality Assessment")
    exercise_6_data_quality()

    print("\nAll exercises completed successfully.")
