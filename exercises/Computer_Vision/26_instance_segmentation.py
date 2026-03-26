"""
Exercise Solutions for Lesson 26: Instance Segmentation
Computer Vision - Mask R-CNN, RoIAlign, YOLACT Prototypes, COCO AP Metrics

Topics covered:
- Instance mask generation and representation
- RoI Align bilinear sampling
- Prototype mask linear combination (YOLACT-style)
- Non-maximum suppression for instance masks
- COCO-style mask AP evaluation
- Instance counting and area analysis
"""

import numpy as np


# =============================================================================
# Helper: Synthetic instance data
# =============================================================================

def generate_instance_scene(h=80, w=100, n_instances=6):
    """
    Generate a synthetic scene with multiple object instances.

    Returns:
        image: (h, w) grayscale
        instance_masks: list of (h, w) binary masks
        instance_labels: list of class IDs
        bboxes: list of (x1, y1, x2, y2) tuples
    """
    np.random.seed(42)
    image = np.random.randint(20, 60, (h, w), dtype=np.uint8)

    instance_masks = []
    instance_labels = []
    bboxes = []

    # Place elliptical objects at random positions
    specs = [
        (15, 20, 12, 8, 1),   # cy, cx, ry, rx, class
        (25, 55, 10, 15, 1),
        (50, 30, 8, 8, 2),
        (55, 70, 12, 10, 2),
        (35, 85, 7, 6, 3),
        (65, 50, 9, 11, 3),
    ]

    yy, xx = np.ogrid[:h, :w]

    for cy, cx, ry, rx, cls_id in specs[:n_instances]:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        brightness = 120 + cls_id * 30 + np.random.randint(-10, 10)
        image[mask] = brightness

        instance_masks.append(mask.astype(np.uint8))
        instance_labels.append(cls_id)

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) > 0 and len(cols) > 0:
            bboxes.append((cols[0], rows[0], cols[-1], rows[-1]))
        else:
            bboxes.append((0, 0, 0, 0))

    return image, instance_masks, instance_labels, bboxes


# =============================================================================
# Exercise 1: Instance Mask Representation
# =============================================================================

def exercise_1_instance_masks():
    """
    Explore instance mask representations and properties.

    Steps:
    1. Generate instance masks for multiple objects
    2. Convert between polygon, RLE, and bitmap formats
    3. Analyze mask overlap and occlusion
    4. Compute per-instance properties (area, centroid, bbox)

    Returns:
        list of instance property dicts
    """
    h, w = 80, 100
    image, masks, labels, bboxes = generate_instance_scene(h, w)

    print("Instance Mask Representation")
    print(f"  Scene size: {w}x{h}")
    print(f"  Instances: {len(masks)}")
    print("=" * 60)

    properties = []
    for i, (mask, label, bbox) in enumerate(zip(masks, labels, bboxes)):
        area = int(mask.sum())

        # Centroid
        ys, xs = np.where(mask > 0)
        centroid_x = xs.mean() if len(xs) > 0 else 0
        centroid_y = ys.mean() if len(ys) > 0 else 0

        # Bounding box area
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        # Simple RLE encoding
        flat = mask.flatten()
        rle_runs = []
        current = flat[0]
        count = 1
        for j in range(1, len(flat)):
            if flat[j] == current:
                count += 1
            else:
                rle_runs.append(count)
                current = flat[j]
                count = 1
        rle_runs.append(count)
        rle_compression = len(rle_runs) / len(flat)

        props = {
            'id': i,
            'class': label,
            'area': area,
            'centroid': (centroid_x, centroid_y),
            'bbox': bbox,
            'bbox_area': bbox_area,
            'fill_ratio': fill_ratio,
            'rle_runs': len(rle_runs),
            'rle_compression': rle_compression,
        }
        properties.append(props)

        print(f"\n  Instance {i} (class {label}):")
        print(f"    Area: {area} pixels")
        print(f"    Centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
        print(f"    BBox: ({x1}, {y1}, {x2}, {y2}), area={bbox_area}")
        print(f"    Fill ratio: {fill_ratio:.3f}")
        print(f"    RLE runs: {len(rle_runs)} (compression: {rle_compression:.4f})")

    # Overlap analysis
    print("\n  Overlap Matrix (IoU):")
    n = len(masks)
    for i in range(n):
        row = []
        for j in range(n):
            intersection = np.sum(masks[i] & masks[j])
            union = np.sum(masks[i] | masks[j])
            iou = intersection / union if union > 0 else 0.0
            row.append(f"{iou:.2f}")
        print(f"    [{', '.join(row)}]")

    return properties


# =============================================================================
# Exercise 2: RoI Align
# =============================================================================

def exercise_2_roi_align():
    """
    Implement RoI Align with bilinear interpolation.

    Compares:
    1. RoI Pool (quantized, lossy)
    2. RoI Align (bilinear interpolation, pixel-perfect)

    Returns:
        (roi_pool_features, roi_align_features)
    """
    np.random.seed(42)
    h, w = 80, 100

    # Create a feature map with known patterns
    feature_map = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            feature_map[i, j] = np.sin(i * 0.2) * np.cos(j * 0.15) + 0.5

    # Define RoIs: (x1, y1, x2, y2) in feature map coordinates
    rois = [
        (10.3, 5.7, 35.8, 25.2),
        (45.1, 30.4, 78.9, 55.6),
        (20.0, 40.0, 50.5, 65.3),
    ]
    output_size = (7, 7)

    print("RoI Align vs RoI Pool")
    print(f"  Feature map: {w}x{h}")
    print(f"  Output size: {output_size}")
    print(f"  RoIs: {len(rois)}")
    print("=" * 60)

    def bilinear_interp(feat, y, x):
        """Bilinear interpolation at sub-pixel location."""
        fh, fw = feat.shape
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        y1 = min(y0 + 1, fh - 1)
        x1 = min(x0 + 1, fw - 1)
        y0 = max(0, y0)
        x0 = max(0, x0)

        fy = y - np.floor(y)
        fx = x - np.floor(x)

        val = (feat[y0, x0] * (1 - fy) * (1 - fx) +
               feat[y0, x1] * (1 - fy) * fx +
               feat[y1, x0] * fy * (1 - fx) +
               feat[y1, x1] * fy * fx)
        return val

    def roi_pool(feat, roi, out_h, out_w):
        """RoI Pool: quantize coordinates then max-pool."""
        x1, y1, x2, y2 = roi
        # Quantize
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        roi_h = max(y2 - y1, 1)
        roi_w = max(x2 - x1, 1)

        out = np.zeros((out_h, out_w), dtype=np.float64)
        bin_h = roi_h / out_h
        bin_w = roi_w / out_w

        for i in range(out_h):
            for j in range(out_w):
                by1 = int(np.floor(y1 + i * bin_h))
                by2 = int(np.ceil(y1 + (i + 1) * bin_h))
                bx1 = int(np.floor(x1 + j * bin_w))
                bx2 = int(np.ceil(x1 + (j + 1) * bin_w))
                by1 = max(0, min(by1, feat.shape[0] - 1))
                by2 = max(by1 + 1, min(by2, feat.shape[0]))
                bx1 = max(0, min(bx1, feat.shape[1] - 1))
                bx2 = max(bx1 + 1, min(bx2, feat.shape[1]))
                out[i, j] = feat[by1:by2, bx1:bx2].max()

        return out

    def roi_align(feat, roi, out_h, out_w, sample_points=4):
        """RoI Align: bilinear interpolation, no quantization."""
        x1, y1, x2, y2 = roi  # Keep as float
        roi_h = y2 - y1
        roi_w = x2 - x1

        out = np.zeros((out_h, out_w), dtype=np.float64)
        bin_h = roi_h / out_h
        bin_w = roi_w / out_w

        # Sample grid within each bin
        n_sample = int(np.sqrt(sample_points))
        for i in range(out_h):
            for j in range(out_w):
                vals = []
                for sy in range(n_sample):
                    for sx in range(n_sample):
                        # Sub-pixel sampling location
                        sample_y = y1 + (i + (sy + 0.5) / n_sample) * bin_h
                        sample_x = x1 + (j + (sx + 0.5) / n_sample) * bin_w
                        vals.append(bilinear_interp(feat, sample_y, sample_x))
                out[i, j] = np.mean(vals)

        return out

    pool_results = []
    align_results = []

    for idx, roi in enumerate(rois):
        pool_feat = roi_pool(feature_map, roi, *output_size)
        align_feat = roi_align(feature_map, roi, *output_size)

        pool_results.append(pool_feat)
        align_results.append(align_feat)

        diff = np.abs(pool_feat - align_feat)
        x1, y1, x2, y2 = roi
        print(f"\n  RoI {idx}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"    Pool  - mean: {pool_feat.mean():.4f}, std: {pool_feat.std():.4f}")
        print(f"    Align - mean: {align_feat.mean():.4f}, std: {align_feat.std():.4f}")
        print(f"    Difference - mean: {diff.mean():.4f}, max: {diff.max():.4f}")

    print("\n  Key insight: RoI Align preserves sub-pixel precision")
    print("  important for accurate mask prediction in Mask R-CNN.")

    return pool_results, align_results


# =============================================================================
# Exercise 3: YOLACT-Style Prototype Masks
# =============================================================================

def exercise_3_prototype_masks():
    """
    Implement YOLACT-style mask generation via prototype linear combination.

    Steps:
    1. Generate K prototype masks (full-image, class-agnostic)
    2. Predict coefficients per instance
    3. Compute instance masks as weighted sum of prototypes
    4. Apply crop and threshold

    Returns:
        list of instance masks
    """
    np.random.seed(42)
    h, w = 80, 100
    n_prototypes = 8
    _, gt_masks, gt_labels, bboxes = generate_instance_scene(h, w)
    n_instances = len(gt_masks)

    print("YOLACT-Style Prototype Masks")
    print(f"  Scene: {w}x{h}")
    print(f"  Prototypes: {n_prototypes}")
    print(f"  Instances: {n_instances}")
    print("=" * 60)

    # Generate prototype masks (smooth spatial bases)
    prototypes = np.zeros((n_prototypes, h, w), dtype=np.float64)
    for k in range(n_prototypes):
        freq_y = (k // 2 + 1) * 0.1
        freq_x = (k % 4 + 1) * 0.08
        phase = k * np.pi / n_prototypes
        for i in range(h):
            for j in range(w):
                prototypes[k, i, j] = (
                    np.sin(freq_y * i + phase) *
                    np.cos(freq_x * j + phase * 0.5) +
                    np.random.randn() * 0.05
                )

    print("\n  Prototype statistics:")
    for k in range(n_prototypes):
        p = prototypes[k]
        print(f"    Proto {k}: range=[{p.min():.3f}, {p.max():.3f}], "
              f"mean={p.mean():.3f}")

    # Learn coefficients that reconstruct GT masks
    # Solve: coeffs @ prototypes_flat ~= gt_mask_flat for each instance
    proto_flat = prototypes.reshape(n_prototypes, -1).T  # (h*w, K)

    predicted_masks = []
    for i in range(n_instances):
        gt_flat = gt_masks[i].flatten().astype(np.float64)

        # Least-squares: find coefficients
        coeffs, residuals, _, _ = np.linalg.lstsq(proto_flat, gt_flat, rcond=None)

        # Apply tanh to bound coefficients (like YOLACT)
        coeffs_bounded = np.tanh(coeffs)

        # Generate mask from linear combination
        raw_mask = np.zeros(h * w, dtype=np.float64)
        for k in range(n_prototypes):
            raw_mask += coeffs_bounded[k] * prototypes[k].flatten()

        # Sigmoid activation
        mask_prob = 1.0 / (1.0 + np.exp(-raw_mask))
        mask_prob = mask_prob.reshape(h, w)

        # Crop to bounding box
        x1, y1, x2, y2 = bboxes[i]
        cropped = np.zeros((h, w), dtype=np.float64)
        cropped[y1:y2+1, x1:x2+1] = mask_prob[y1:y2+1, x1:x2+1]

        # Threshold
        binary_mask = (cropped > 0.5).astype(np.uint8)
        predicted_masks.append(binary_mask)

        # Compute quality
        intersection = np.sum(binary_mask & gt_masks[i])
        union = np.sum(binary_mask | gt_masks[i])
        iou = intersection / union if union > 0 else 0.0

        top_coeffs = np.argsort(np.abs(coeffs_bounded))[-3:]
        coeff_str = ", ".join(
            f"p{k}={coeffs_bounded[k]:.2f}" for k in top_coeffs
        )
        print(f"\n  Instance {i} (class {gt_labels[i]}):")
        print(f"    Top coefficients: {coeff_str}")
        print(f"    Mask area: {binary_mask.sum()} (GT: {gt_masks[i].sum()})")
        print(f"    IoU: {iou:.4f}")

    return predicted_masks


# =============================================================================
# Exercise 4: Non-Maximum Suppression for Masks
# =============================================================================

def exercise_4_mask_nms():
    """
    Implement mask-based Non-Maximum Suppression.

    Steps:
    1. Generate overlapping instance predictions with scores
    2. Compute mask IoU between all pairs
    3. Apply greedy NMS to remove redundant detections
    4. Compare box NMS vs mask NMS

    Returns:
        (kept_indices_box_nms, kept_indices_mask_nms)
    """
    np.random.seed(42)
    h, w = 80, 100

    # Generate predictions: some overlapping
    pred_masks = []
    pred_scores = []
    pred_bboxes = []
    pred_labels = []

    yy, xx = np.ogrid[:h, :w]

    detection_specs = [
        (20, 25, 12, 10, 0.95, 1),  # cy, cx, ry, rx, score, class
        (22, 27, 11, 9, 0.80, 1),   # Overlapping with first
        (50, 60, 10, 12, 0.90, 2),
        (52, 58, 9, 11, 0.70, 2),   # Overlapping with third
        (35, 85, 8, 7, 0.85, 3),
        (10, 50, 6, 6, 0.60, 1),
    ]

    for cy, cx, ry, rx, score, cls in detection_specs:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        pred_masks.append(mask.astype(np.uint8))
        pred_scores.append(score)
        pred_labels.append(cls)

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        pred_bboxes.append((cols[0], rows[0], cols[-1], rows[-1]))

    n_dets = len(pred_masks)

    print("Mask Non-Maximum Suppression")
    print(f"  Detections: {n_dets}")
    print("=" * 60)

    # Compute mask IoU matrix
    mask_iou_matrix = np.zeros((n_dets, n_dets), dtype=np.float64)
    for i in range(n_dets):
        for j in range(n_dets):
            intersection = np.sum(pred_masks[i] & pred_masks[j])
            union = np.sum(pred_masks[i] | pred_masks[j])
            mask_iou_matrix[i, j] = intersection / union if union > 0 else 0.0

    # Compute box IoU matrix
    box_iou_matrix = np.zeros((n_dets, n_dets), dtype=np.float64)
    for i in range(n_dets):
        for j in range(n_dets):
            x1i, y1i, x2i, y2i = pred_bboxes[i]
            x1j, y1j, x2j, y2j = pred_bboxes[j]
            inter_x1 = max(x1i, x1j)
            inter_y1 = max(y1i, y1j)
            inter_x2 = min(x2i, x2j)
            inter_y2 = min(y2i, y2j)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_i = (x2i - x1i) * (y2i - y1i)
            area_j = (x2j - x1j) * (y2j - y1j)
            union_area = area_i + area_j - inter_area
            box_iou_matrix[i, j] = inter_area / union_area if union_area > 0 else 0.0

    def greedy_nms(iou_matrix, scores, iou_threshold=0.5):
        """Greedy NMS using given IoU matrix."""
        order = np.argsort(scores)[::-1]
        keep = []
        suppressed = set()

        for idx in order:
            if idx in suppressed:
                continue
            keep.append(idx)
            for other in order:
                if other in suppressed or other == idx:
                    continue
                if iou_matrix[idx, other] > iou_threshold:
                    suppressed.add(other)
        return keep

    scores_arr = np.array(pred_scores)
    kept_box = greedy_nms(box_iou_matrix, scores_arr, iou_threshold=0.5)
    kept_mask = greedy_nms(mask_iou_matrix, scores_arr, iou_threshold=0.5)

    print("\n  Before NMS:")
    for i in range(n_dets):
        print(f"    Det {i}: class={pred_labels[i]}, "
              f"score={pred_scores[i]:.2f}, "
              f"area={pred_masks[i].sum()}")

    print(f"\n  Box NMS kept: {kept_box}")
    print(f"  Mask NMS kept: {kept_mask}")

    # Analyze differences
    box_only = set(kept_box) - set(kept_mask)
    mask_only = set(kept_mask) - set(kept_box)
    if box_only:
        print(f"\n  Kept by box NMS only: {box_only}")
    if mask_only:
        print(f"  Kept by mask NMS only: {mask_only}")

    # Show IoU for suppressed pairs
    print("\n  Mask IoU between detections:")
    for i in range(n_dets):
        row = [f"{mask_iou_matrix[i, j]:.2f}" for j in range(n_dets)]
        print(f"    [{', '.join(row)}]")

    return kept_box, kept_mask


# =============================================================================
# Exercise 5: COCO-Style Mask AP
# =============================================================================

def exercise_5_coco_mask_ap():
    """
    Implement COCO-style Average Precision for instance segmentation.

    Steps:
    1. Generate predictions and ground truth
    2. Match predictions to GT using mask IoU
    3. Compute precision-recall curve
    4. Calculate AP at multiple IoU thresholds (AP@50:95)

    Returns:
        dict with AP values
    """
    np.random.seed(42)
    h, w = 80, 100
    _, gt_masks, gt_labels, _ = generate_instance_scene(h, w)

    print("COCO-Style Mask AP Evaluation")
    print(f"  GT instances: {len(gt_masks)}")
    print("=" * 60)

    # Generate predictions (some correct, some wrong)
    pred_masks = []
    pred_scores = []
    pred_labels_list = []

    yy, xx = np.ogrid[:h, :w]

    # Good predictions (close to GT)
    for i, (gt_m, gt_l) in enumerate(zip(gt_masks, gt_labels)):
        ys, xs = np.where(gt_m > 0)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        ry = max((ys.max() - ys.min()) / 2, 3)
        rx = max((xs.max() - xs.min()) / 2, 3)
        # Add slight offset
        cy += np.random.randn() * 1.5
        cx += np.random.randn() * 1.5
        ry *= np.random.uniform(0.85, 1.15)
        rx *= np.random.uniform(0.85, 1.15)

        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        pred_masks.append(mask.astype(np.uint8))
        pred_scores.append(0.6 + np.random.random() * 0.35)
        pred_labels_list.append(gt_l)

    # Add false positive
    fp_mask = (((xx - 80) / 8) ** 2 + ((yy - 10) / 6) ** 2) <= 1.0
    pred_masks.append(fp_mask.astype(np.uint8))
    pred_scores.append(0.45)
    pred_labels_list.append(1)

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    print(f"  Predictions: {n_pred}")

    def compute_ap_at_iou(iou_thresh):
        """Compute AP at a single IoU threshold."""
        # Sort predictions by score
        order = np.argsort(pred_scores)[::-1]
        gt_matched = [False] * n_gt

        tps = []
        fps = []

        for pred_idx in order:
            pred_m = pred_masks[pred_idx]
            pred_l = pred_labels_list[pred_idx]

            best_iou = 0.0
            best_gt = -1

            for gt_idx in range(n_gt):
                if gt_matched[gt_idx]:
                    continue
                if gt_labels[gt_idx] != pred_l:
                    continue
                intersection = np.sum(pred_m & gt_masks[gt_idx])
                union = np.sum(pred_m | gt_masks[gt_idx])
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= iou_thresh and best_gt >= 0:
                tps.append(1)
                fps.append(0)
                gt_matched[best_gt] = True
            else:
                tps.append(0)
                fps.append(1)

        # Cumulative
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / n_gt

        # 101-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 101):
            prec_at_r = 0.0
            for p, r in zip(precision, recall):
                if r >= r_thresh:
                    prec_at_r = max(prec_at_r, p)
            ap += prec_at_r
        ap /= 101

        return ap, precision, recall

    # Compute AP at standard IoU thresholds
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps = {}

    print(f"\n  {'IoU Thresh':>12} | {'AP':>8}")
    print(f"  {'-'*25}")

    for thresh in iou_thresholds:
        ap, prec, rec = compute_ap_at_iou(thresh)
        aps[f"AP@{thresh:.2f}"] = ap
        print(f"  {thresh:>12.2f} | {ap:>8.4f}")

    # Summary metrics
    ap_50 = aps.get("AP@0.50", 0.0)
    ap_75 = aps.get("AP@0.75", 0.0)
    ap_50_95 = np.mean(list(aps.values()))

    print(f"\n  AP@50:    {ap_50:.4f}")
    print(f"  AP@75:    {ap_75:.4f}")
    print(f"  AP@50:95: {ap_50_95:.4f}")

    results = {
        'AP@50': ap_50,
        'AP@75': ap_75,
        'AP@50:95': ap_50_95,
        'per_threshold': aps,
    }

    return results


# =============================================================================
# Exercise 6: Instance Counting and Analysis
# =============================================================================

def exercise_6_instance_counting():
    """
    Instance counting and area measurement application.

    Steps:
    1. Detect instances and count per class
    2. Measure area distribution
    3. Analyze spatial distribution of instances
    4. Compute density and coverage statistics

    Returns:
        dict of class -> count and area stats
    """
    np.random.seed(42)
    h, w = 80, 100
    _, masks, labels, bboxes = generate_instance_scene(h, w)

    print("Instance Counting and Analysis")
    print(f"  Scene: {w}x{h} = {h*w} pixels")
    print(f"  Total instances: {len(masks)}")
    print("=" * 60)

    # Count per class
    class_names = {1: "car", 2: "person", 3: "bicycle"}
    class_stats = {}

    for cls_id in sorted(set(labels)):
        cls_masks = [m for m, l in zip(masks, labels) if l == cls_id]
        cls_bboxes = [b for b, l in zip(bboxes, labels) if l == cls_id]

        areas = [int(m.sum()) for m in cls_masks]
        cls_name = class_names.get(cls_id, f"class_{cls_id}")

        # Compute centroids
        centroids = []
        for m in cls_masks:
            ys, xs = np.where(m > 0)
            if len(ys) > 0:
                centroids.append((xs.mean(), ys.mean()))

        # Spatial spread
        if len(centroids) > 1:
            cx = [c[0] for c in centroids]
            cy = [c[1] for c in centroids]
            spread = np.sqrt(np.var(cx) + np.var(cy))
        else:
            spread = 0.0

        # Coverage
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in cls_masks:
            combined |= m
        coverage = combined.sum() / (h * w)

        class_stats[cls_id] = {
            'name': cls_name,
            'count': len(cls_masks),
            'areas': areas,
            'mean_area': np.mean(areas),
            'total_area': sum(areas),
            'coverage': coverage,
            'spread': spread,
        }

        print(f"\n  {cls_name} (class {cls_id}):")
        print(f"    Count: {len(cls_masks)}")
        print(f"    Areas: {areas}")
        print(f"    Mean area: {np.mean(areas):.1f}")
        print(f"    Total area: {sum(areas)}")
        print(f"    Image coverage: {coverage:.4f} ({100*coverage:.1f}%)")
        print(f"    Spatial spread: {spread:.2f}")

    # Global statistics
    all_areas = [int(m.sum()) for m in masks]
    total_coverage_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        total_coverage_mask |= m
    total_coverage = total_coverage_mask.sum() / (h * w)

    # Overlap analysis
    overlap_pixels = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            overlap_pixels += np.sum(masks[i] & masks[j])

    density = len(masks) / (h * w) * 10000  # instances per 10000 pixels

    print(f"\n  Global Statistics:")
    print(f"    Total instances: {len(masks)}")
    print(f"    Area range: [{min(all_areas)}, {max(all_areas)}]")
    print(f"    Total coverage: {total_coverage:.4f} ({100*total_coverage:.1f}%)")
    print(f"    Overlap pixels: {overlap_pixels}")
    print(f"    Instance density: {density:.2f} per 10k pixels")

    return class_stats


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Instance Mask Representation")
    exercise_1_instance_masks()

    print("\n>>> Exercise 2: RoI Align")
    exercise_2_roi_align()

    print("\n>>> Exercise 3: YOLACT-Style Prototype Masks")
    exercise_3_prototype_masks()

    print("\n>>> Exercise 4: Non-Maximum Suppression for Masks")
    exercise_4_mask_nms()

    print("\n>>> Exercise 5: COCO-Style Mask AP")
    exercise_5_coco_mask_ap()

    print("\n>>> Exercise 6: Instance Counting and Analysis")
    exercise_6_instance_counting()

    print("\nAll exercises completed successfully.")
