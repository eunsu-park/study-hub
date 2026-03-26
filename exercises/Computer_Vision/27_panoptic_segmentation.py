"""
Exercise Solutions for Lesson 27: Panoptic Segmentation
Computer Vision - Stuff vs Things, Panoptic Fusion, Panoptic Quality (PQ)

Topics covered:
- Stuff vs things classification
- Panoptic fusion of semantic and instance predictions
- Panoptic Quality (PQ) metric and its SQ/RQ decomposition
- Mask2Former-style query-based segmentation
- Multi-scale feature merging
- Segment matching and tracking
"""

import numpy as np


# =============================================================================
# Helper: Synthetic panoptic scene
# =============================================================================

def generate_panoptic_scene(h=64, w=80):
    """
    Generate a synthetic panoptic scene with stuff and thing classes.

    Stuff classes (no instances): 0=sky, 1=road, 2=grass
    Thing classes (with instances): 3=car, 4=person

    Returns:
        semantic_map: (h, w) class labels
        instance_map: (h, w) instance IDs (0 = no instance / stuff)
        segment_info: list of segment dicts
    """
    np.random.seed(42)
    semantic_map = np.zeros((h, w), dtype=np.int32)
    instance_map = np.zeros((h, w), dtype=np.int32)
    segment_info = []
    next_id = 1

    # Stuff: sky (top 20 rows)
    semantic_map[:20, :] = 0
    segment_info.append({
        'id': next_id, 'category_id': 0,
        'isthing': False, 'area': int(np.sum(semantic_map == 0))
    })
    instance_map[:20, :] = next_id
    next_id += 1

    # Stuff: road (bottom 15 rows)
    semantic_map[49:, :] = 1
    segment_info.append({
        'id': next_id, 'category_id': 1,
        'isthing': False, 'area': 15 * w
    })
    instance_map[49:, :] = next_id
    next_id += 1

    # Stuff: grass (middle band)
    semantic_map[20:49, :] = 2
    segment_info.append({
        'id': next_id, 'category_id': 2,
        'isthing': False, 'area': 29 * w
    })
    instance_map[20:49, :] = next_id
    next_id += 1

    # Things: cars
    yy, xx = np.ogrid[:h, :w]
    car_specs = [
        (38, 20, 8, 12),  # cy, cx, ry, rx
        (42, 60, 7, 10),
    ]
    for cy, cx, ry, rx in car_specs:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        semantic_map[mask] = 3
        instance_map[mask] = next_id
        segment_info.append({
            'id': next_id, 'category_id': 3,
            'isthing': True, 'area': int(mask.sum())
        })
        next_id += 1

    # Things: persons
    person_specs = [
        (30, 45, 6, 3),
        (33, 70, 5, 3),
    ]
    for cy, cx, ry, rx in person_specs:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        semantic_map[mask] = 4
        instance_map[mask] = next_id
        segment_info.append({
            'id': next_id, 'category_id': 4,
            'isthing': True, 'area': int(mask.sum())
        })
        next_id += 1

    return semantic_map, instance_map, segment_info


# =============================================================================
# Exercise 1: Stuff vs Things Analysis
# =============================================================================

def exercise_1_stuff_things():
    """
    Analyze the stuff vs things decomposition of a panoptic scene.

    Steps:
    1. Classify each segment as stuff or thing
    2. Compute pixel coverage statistics
    3. Analyze spatial distribution
    4. Compute class frequency and area distributions

    Returns:
        dict with stuff and thing statistics
    """
    h, w = 64, 80
    semantic_map, instance_map, segment_info = generate_panoptic_scene(h, w)

    class_names = {0: "sky", 1: "road", 2: "grass", 3: "car", 4: "person"}
    stuff_classes = {0, 1, 2}
    thing_classes = {3, 4}

    print("Stuff vs Things Analysis")
    print(f"  Scene: {w}x{h} = {h*w} pixels")
    print(f"  Segments: {len(segment_info)}")
    print("=" * 60)

    # Pixel coverage
    stuff_pixels = 0
    thing_pixels = 0
    for seg in segment_info:
        if seg['isthing']:
            thing_pixels += seg['area']
        else:
            stuff_pixels += seg['area']

    total_pixels = h * w
    print(f"\n  Pixel Coverage:")
    print(f"    Stuff: {stuff_pixels} pixels ({100*stuff_pixels/total_pixels:.1f}%)")
    print(f"    Things: {thing_pixels} pixels ({100*thing_pixels/total_pixels:.1f}%)")

    # Per-class statistics
    print(f"\n  Per-Class Breakdown:")
    print(f"    {'Class':>8} | {'Type':>6} | {'Segments':>8} | {'Pixels':>8} | {'Coverage':>8}")
    print(f"    {'-'*50}")

    class_stats = {}
    for cls_id in sorted(class_names.keys()):
        cls_segs = [s for s in segment_info if s['category_id'] == cls_id]
        cls_pixels = sum(s['area'] for s in cls_segs)
        cls_type = "stuff" if cls_id in stuff_classes else "thing"
        coverage = cls_pixels / total_pixels

        class_stats[cls_id] = {
            'name': class_names[cls_id],
            'type': cls_type,
            'n_segments': len(cls_segs),
            'pixels': cls_pixels,
            'coverage': coverage,
        }

        print(f"    {class_names[cls_id]:>8} | {cls_type:>6} | "
              f"{len(cls_segs):>8} | {cls_pixels:>8} | {coverage:>8.4f}")

    # Instance count for things
    print(f"\n  Thing Instance Counts:")
    for cls_id in thing_classes:
        n_inst = class_stats[cls_id]['n_segments']
        name = class_names[cls_id]
        areas = [s['area'] for s in segment_info
                 if s['category_id'] == cls_id]
        if areas:
            print(f"    {name}: {n_inst} instances, "
                  f"areas={areas}, mean={np.mean(areas):.1f}")

    # Spatial analysis: where are things located?
    print(f"\n  Spatial Distribution (vertical bands):")
    n_bands = 4
    band_h = h // n_bands
    for b in range(n_bands):
        y_start = b * band_h
        y_end = min((b + 1) * band_h, h)
        band_region = semantic_map[y_start:y_end, :]
        band_stuff = sum(1 for c in band_region.flatten() if c in stuff_classes)
        band_things = sum(1 for c in band_region.flatten() if c in thing_classes)
        total_band = band_region.size
        print(f"    Band {b} (rows {y_start}-{y_end}): "
              f"stuff={100*band_stuff/total_band:.0f}%, "
              f"things={100*band_things/total_band:.0f}%")

    return {'stuff_pixels': stuff_pixels, 'thing_pixels': thing_pixels,
            'class_stats': class_stats}


# =============================================================================
# Exercise 2: Panoptic Fusion
# =============================================================================

def exercise_2_panoptic_fusion():
    """
    Merge semantic (stuff) and instance (things) predictions into
    a unified panoptic segmentation map.

    Steps:
    1. Generate separate semantic and instance predictions
    2. Place instance masks (things) with priority
    3. Fill remaining pixels with stuff classes
    4. Handle conflicts and overlaps

    Returns:
        (panoptic_map, segment_info)
    """
    np.random.seed(42)
    h, w = 64, 80
    gt_semantic, gt_instance, gt_segments = generate_panoptic_scene(h, w)

    stuff_classes = {0, 1, 2}
    thing_classes = {3, 4}
    class_names = {0: "sky", 1: "road", 2: "grass", 3: "car", 4: "person"}

    print("Panoptic Fusion")
    print(f"  Scene: {w}x{h}")
    print("=" * 60)

    # Simulate noisy predictions
    # Semantic prediction (noisy version of GT)
    semantic_pred = gt_semantic.copy()
    noise_mask = np.random.random((h, w)) < 0.05
    semantic_pred[noise_mask] = np.random.randint(0, 5, size=noise_mask.sum())

    # Instance predictions (slightly shifted masks)
    yy, xx = np.ogrid[:h, :w]
    instance_masks = []
    instance_scores = []
    instance_labels = []

    inst_specs = [
        (39, 21, 8, 12, 0.92, 3),  # Slightly shifted car 1
        (43, 59, 7, 10, 0.88, 3),  # Slightly shifted car 2
        (31, 46, 6, 3, 0.85, 4),   # Person 1
        (34, 69, 5, 3, 0.78, 4),   # Person 2
        (25, 35, 5, 4, 0.30, 3),   # False positive (low score)
    ]

    for cy, cx, ry, rx, score, cls in inst_specs:
        mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        instance_masks.append(mask.astype(np.uint8))
        instance_scores.append(score)
        instance_labels.append(cls)

    # Panoptic fusion
    panoptic_map = np.zeros((h, w), dtype=np.int32)
    fused_segments = []
    next_id = 1
    occupied = np.zeros((h, w), dtype=bool)
    score_threshold = 0.5
    overlap_threshold = 0.5

    # Step 1: Place things (sorted by score, highest first)
    sorted_idx = np.argsort(instance_scores)[::-1]
    things_placed = 0

    for idx in sorted_idx:
        score = instance_scores[idx]
        if score < score_threshold:
            continue

        mask = instance_masks[idx]
        label = instance_labels[idx]

        if label not in thing_classes:
            continue

        # Check overlap with already placed
        overlap = (mask.astype(bool) & occupied).sum()
        mask_area = mask.sum()
        if mask_area == 0:
            continue
        overlap_ratio = overlap / mask_area
        if overlap_ratio > overlap_threshold:
            continue

        panoptic_map[mask > 0] = next_id
        occupied[mask > 0] = True
        fused_segments.append({
            'id': next_id,
            'category_id': label,
            'isthing': True,
            'score': score,
            'area': int(mask.sum()),
        })
        next_id += 1
        things_placed += 1

    print(f"\n  Step 1 - Things placement:")
    print(f"    Candidates: {len(instance_masks)}")
    print(f"    Score threshold: {score_threshold}")
    print(f"    Placed: {things_placed}")

    # Step 2: Fill stuff
    stuff_placed = 0
    for stuff_cls in sorted(stuff_classes):
        stuff_mask = (semantic_pred == stuff_cls) & (~occupied)
        if stuff_mask.sum() > 0:
            panoptic_map[stuff_mask] = next_id
            fused_segments.append({
                'id': next_id,
                'category_id': stuff_cls,
                'isthing': False,
                'area': int(stuff_mask.sum()),
            })
            next_id += 1
            stuff_placed += 1

    print(f"\n  Step 2 - Stuff filling:")
    print(f"    Stuff segments placed: {stuff_placed}")

    # Coverage analysis
    assigned = panoptic_map > 0
    coverage = assigned.sum() / (h * w)
    print(f"\n  Coverage: {coverage:.4f} ({100*coverage:.1f}%)")
    print(f"  Unassigned pixels: {(~assigned).sum()}")

    print(f"\n  Final segments:")
    for seg in fused_segments:
        cls_name = class_names.get(seg['category_id'], 'unknown')
        seg_type = "thing" if seg['isthing'] else "stuff"
        score_str = f", score={seg['score']:.2f}" if 'score' in seg else ""
        print(f"    ID={seg['id']}: {cls_name} ({seg_type}), "
              f"area={seg['area']}{score_str}")

    return panoptic_map, fused_segments


# =============================================================================
# Exercise 3: Panoptic Quality (PQ) Metric
# =============================================================================

def exercise_3_panoptic_quality():
    """
    Compute Panoptic Quality and its SQ/RQ decomposition.

    PQ = SQ * RQ
    SQ = average IoU of matched segments
    RQ = F1 score of segment matching (TP / (TP + 0.5*FP + 0.5*FN))

    Returns:
        dict with PQ, SQ, RQ, per-class breakdown
    """
    np.random.seed(42)
    h, w = 64, 80
    gt_semantic, gt_instance, gt_segments = generate_panoptic_scene(h, w)

    print("Panoptic Quality (PQ) Metric")
    print(f"  GT segments: {len(gt_segments)}")
    print("=" * 60)

    # Generate predicted panoptic map (slightly noisy)
    pred_instance = gt_instance.copy()
    # Shift some segment boundaries
    for i in range(h):
        for j in range(w):
            if np.random.random() < 0.03:
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            neighbors.append(pred_instance[ni, nj])
                if neighbors:
                    pred_instance[i, j] = np.random.choice(neighbors)

    # Build segment label maps
    gt_seg_labels = {}
    for seg in gt_segments:
        gt_seg_labels[seg['id']] = seg['category_id']

    pred_seg_labels = dict(gt_seg_labels)  # Same labels (imperfect boundaries)

    # Add a false positive segment
    fp_mask = np.zeros((h, w), dtype=bool)
    fp_mask[5:12, 60:72] = True
    fp_id = max(gt_seg_labels.keys()) + 1
    pred_instance[fp_mask] = fp_id
    pred_seg_labels[fp_id] = 3  # False car

    # Remove one GT segment from prediction (false negative)
    fn_id = gt_segments[-1]['id']  # Remove last person
    pred_instance[pred_instance == fn_id] = 3  # Merge into grass segment

    def compute_pq(gt_map, pred_map, gt_labels, pred_labels, iou_threshold=0.5):
        """Compute PQ with SQ/RQ decomposition."""
        gt_ids = set(np.unique(gt_map)) - {0}
        pred_ids = set(np.unique(pred_map)) - {0}

        matched_iou = []
        tp = 0
        fp = 0
        fn = 0
        gt_matched = set()
        pred_matched = set()

        # Match predictions to GT
        for pred_id in pred_ids:
            pred_mask = pred_map == pred_id
            pred_label = pred_labels.get(pred_id, -1)

            best_iou = 0.0
            best_gt = None

            for gt_id in gt_ids:
                if gt_id in gt_matched:
                    continue
                gt_label = gt_labels.get(gt_id, -1)
                if gt_label != pred_label:
                    continue

                gt_mask = gt_map == gt_id
                intersection = np.sum(pred_mask & gt_mask)
                union = np.sum(pred_mask | gt_mask)
                iou = intersection / (union + 1e-10)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id

            if best_iou > iou_threshold and best_gt is not None:
                tp += 1
                matched_iou.append(best_iou)
                gt_matched.add(best_gt)
                pred_matched.add(pred_id)
            else:
                fp += 1

        # Unmatched GT = FN
        for gt_id in gt_ids:
            if gt_id not in gt_matched:
                fn += 1

        sq = float(np.mean(matched_iou)) if matched_iou else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-10)
        pq = sq * rq

        return {
            'PQ': pq, 'SQ': sq, 'RQ': rq,
            'TP': tp, 'FP': fp, 'FN': fn,
            'matched_ious': matched_iou,
        }

    # Overall PQ
    overall = compute_pq(gt_instance, pred_instance,
                         gt_seg_labels, pred_seg_labels)

    print(f"\n  Overall Panoptic Quality:")
    print(f"    PQ = {overall['PQ']:.4f}")
    print(f"    SQ = {overall['SQ']:.4f} (segmentation quality)")
    print(f"    RQ = {overall['RQ']:.4f} (recognition quality)")
    print(f"    TP={overall['TP']}, FP={overall['FP']}, FN={overall['FN']}")

    if overall['matched_ious']:
        print(f"    Matched IoUs: "
              + ", ".join(f"{iou:.3f}" for iou in overall['matched_ious']))

    # Per-category PQ
    class_names = {0: "sky", 1: "road", 2: "grass", 3: "car", 4: "person"}
    print(f"\n  Per-Category PQ:")
    print(f"    {'Category':>8} | {'PQ':>6} | {'SQ':>6} | {'RQ':>6} | "
          f"{'TP':>3} {'FP':>3} {'FN':>3}")
    print(f"    {'-'*50}")

    per_class_pq = {}
    for cls_id, cls_name in class_names.items():
        # Filter segments by class
        cls_gt_ids = {s['id'] for s in gt_segments if s['category_id'] == cls_id}
        cls_pred_ids = {pid for pid, plbl in pred_seg_labels.items()
                        if plbl == cls_id}

        if not cls_gt_ids and not cls_pred_ids:
            continue

        # Build class-specific maps
        cls_gt_map = np.where(np.isin(gt_instance, list(cls_gt_ids)),
                              gt_instance, 0)
        cls_pred_map = np.where(np.isin(pred_instance, list(cls_pred_ids)),
                                pred_instance, 0)

        cls_result = compute_pq(cls_gt_map, cls_pred_map,
                                gt_seg_labels, pred_seg_labels)
        per_class_pq[cls_id] = cls_result

        print(f"    {cls_name:>8} | {cls_result['PQ']:>6.3f} | "
              f"{cls_result['SQ']:>6.3f} | {cls_result['RQ']:>6.3f} | "
              f"{cls_result['TP']:>3} {cls_result['FP']:>3} {cls_result['FN']:>3}")

    return {'overall': overall, 'per_class': per_class_pq}


# =============================================================================
# Exercise 4: Query-Based Segmentation (Mask2Former Style)
# =============================================================================

def exercise_4_query_segmentation():
    """
    Simulate query-based segmentation like Mask2Former.

    Steps:
    1. Initialize learnable queries
    2. Compute cross-attention between queries and pixel features
    3. Generate class predictions and mask predictions per query
    4. Apply Hungarian matching to assign queries to GT segments

    Returns:
        (class_preds, mask_preds, matching)
    """
    np.random.seed(42)
    h, w = 32, 40
    n_queries = 10
    d_model = 16
    n_classes = 5

    gt_semantic, gt_instance, gt_segments = generate_panoptic_scene(h, w)

    print("Query-Based Segmentation (Mask2Former Style)")
    print(f"  Scene: {w}x{h}")
    print(f"  Queries: {n_queries}")
    print(f"  Classes: {n_classes}")
    print("=" * 60)

    # Simulate pixel features (from backbone + pixel decoder)
    pixel_features = np.random.randn(h * w, d_model).astype(np.float64) * 0.5

    # Encode GT info into features for simulation
    for seg in gt_segments:
        seg_mask = (gt_instance == seg['id']).flatten()
        cls_encoding = np.zeros(d_model)
        cls_encoding[seg['category_id'] % d_model] = 2.0
        pixel_features[seg_mask] += cls_encoding

    # Initialize queries
    queries = np.random.randn(n_queries, d_model).astype(np.float64) * 0.3

    # Cross-attention: queries attend to pixel features
    # Attention = softmax(Q @ K^T / sqrt(d))
    scale = np.sqrt(d_model)
    attn_scores = queries @ pixel_features.T / scale  # (n_queries, h*w)

    # Softmax over pixel dimension
    attn_max = attn_scores.max(axis=1, keepdims=True)
    attn_exp = np.exp(attn_scores - attn_max)
    attn_weights = attn_exp / attn_exp.sum(axis=1, keepdims=True)

    # Update queries with attended features
    updated_queries = attn_weights @ pixel_features  # (n_queries, d_model)

    print(f"\n  Attention Statistics:")
    for q in range(min(5, n_queries)):
        entropy = -np.sum(attn_weights[q] * np.log(attn_weights[q] + 1e-10))
        max_attn = attn_weights[q].max()
        print(f"    Query {q}: entropy={entropy:.2f}, max_attn={max_attn:.4f}")

    # Class prediction: linear layer on query
    class_weights = np.random.randn(d_model, n_classes + 1) * 0.3  # +1 for no-object
    class_logits = updated_queries @ class_weights  # (n_queries, n_classes+1)
    class_probs = np.exp(class_logits - class_logits.max(axis=1, keepdims=True))
    class_probs /= class_probs.sum(axis=1, keepdims=True)
    class_preds = np.argmax(class_probs, axis=1)

    # Mask prediction: dot product of query embedding with pixel features
    mask_weights = np.random.randn(d_model, d_model) * 0.3
    mask_embed = updated_queries @ mask_weights
    mask_logits = mask_embed @ pixel_features.T  # (n_queries, h*w)
    mask_preds = (mask_logits > 0).reshape(n_queries, h, w)

    print(f"\n  Query Predictions:")
    for q in range(n_queries):
        cls = class_preds[q]
        cls_name = "no-obj" if cls == n_classes else str(cls)
        area = mask_preds[q].sum()
        conf = class_probs[q].max()
        print(f"    Query {q}: class={cls_name}, "
              f"mask_area={area}, conf={conf:.3f}")

    # Hungarian matching (simplified greedy)
    print(f"\n  Hungarian Matching (greedy):")
    n_gt = len(gt_segments)
    cost_matrix = np.zeros((n_queries, n_gt), dtype=np.float64)

    for q in range(n_queries):
        for g in range(n_gt):
            # Class cost
            cls_cost = 1.0 - class_probs[q, gt_segments[g]['category_id']]

            # Mask cost (dice)
            pred_mask = mask_preds[q].flatten().astype(np.float64)
            gt_mask = (gt_instance == gt_segments[g]['id']).flatten().astype(np.float64)
            intersection = np.sum(pred_mask * gt_mask)
            dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-10)
            mask_cost = 1.0 - dice

            cost_matrix[q, g] = cls_cost + mask_cost

    # Greedy assignment
    matched_queries = set()
    matched_gt = set()
    matching = []

    for _ in range(min(n_queries, n_gt)):
        min_cost = float('inf')
        best_q, best_g = -1, -1
        for q in range(n_queries):
            if q in matched_queries:
                continue
            for g in range(n_gt):
                if g in matched_gt:
                    continue
                if cost_matrix[q, g] < min_cost:
                    min_cost = cost_matrix[q, g]
                    best_q, best_g = q, g
        if best_q >= 0:
            matching.append((best_q, best_g, min_cost))
            matched_queries.add(best_q)
            matched_gt.add(best_g)

    for q, g, cost in matching:
        seg = gt_segments[g]
        print(f"    Query {q} -> GT {seg['id']} "
              f"(cat={seg['category_id']}, cost={cost:.3f})")

    unmatched_gt = set(range(n_gt)) - matched_gt
    if unmatched_gt:
        print(f"    Unmatched GT: {unmatched_gt}")

    return class_preds, mask_preds, matching


# =============================================================================
# Exercise 5: Multi-Scale Feature Merging
# =============================================================================

def exercise_5_multiscale_features():
    """
    Implement FPN-style multi-scale feature merging for panoptic segmentation.

    Steps:
    1. Generate features at multiple scales (1/4, 1/8, 1/16, 1/32)
    2. Top-down pathway with lateral connections
    3. Merge all scales to uniform resolution
    4. Compare single-scale vs multi-scale prediction quality

    Returns:
        (merged_features, predictions)
    """
    np.random.seed(42)
    h, w = 64, 80
    n_classes = 5
    gt_semantic, _, _ = generate_panoptic_scene(h, w)

    print("Multi-Scale Feature Merging (FPN-style)")
    print(f"  Input: {w}x{h}")
    print("=" * 60)

    # Generate features at different scales
    scales = {
        'P2': (h // 4, w // 4),   # 16x20
        'P3': (h // 8, w // 8),   # 8x10
        'P4': (h // 16, w // 16), # 4x5
        'P5': (h // 32, w // 32), # 2x2 (or 2x3)
    }

    features = {}
    for name, (fh, fw) in scales.items():
        # Simulate features with class-related patterns
        feat = np.random.randn(fh, fw).astype(np.float64) * 0.3
        # Encode GT info at each scale
        gt_ds = gt_semantic[::h//fh, ::w//fw][:fh, :fw]
        feat += gt_ds.astype(np.float64) * 0.5
        features[name] = feat
        print(f"  {name}: {fw}x{fh}")

    # Top-down pathway
    def upsample_2x(feat, target_h, target_w):
        fh, fw = feat.shape
        out = np.zeros((target_h, target_w), dtype=np.float64)
        for i in range(target_h):
            for j in range(target_w):
                si = min(i * fh // target_h, fh - 1)
                sj = min(j * fw // target_w, fw - 1)
                out[i, j] = feat[si, sj]
        return out

    # Start from P5 (coarsest) and go up
    print(f"\n  Top-Down Pathway:")
    td = {}
    td['P5'] = features['P5'].copy()

    for src, dst in [('P5', 'P4'), ('P4', 'P3'), ('P3', 'P2')]:
        dst_h, dst_w = scales[dst]
        upsampled = upsample_2x(td[src], dst_h, dst_w)
        lateral = features[dst] * 0.8  # Lateral connection weight
        td[dst] = upsampled + lateral
        print(f"    {src} -> {dst}: up({td[src].shape}) + lateral({features[dst].shape})")

    # Merge to P2 resolution
    target_h, target_w = scales['P2']
    merged = np.zeros((target_h, target_w), dtype=np.float64)
    for name in ['P2', 'P3', 'P4', 'P5']:
        up = upsample_2x(td[name], target_h, target_w)
        merged += up
    merged /= len(scales)

    print(f"\n  Merged features: {target_w}x{target_h}")
    print(f"    Range: [{merged.min():.3f}, {merged.max():.3f}]")

    # Classification from merged vs single scale
    def classify(feat, target_h, target_w):
        """Classify using features upsampled to target resolution."""
        fh, fw = feat.shape
        scores = np.zeros((n_classes, target_h, target_w), dtype=np.float64)
        up = upsample_2x(feat, target_h, target_w)
        for c in range(n_classes):
            scores[c] = up * np.random.randn() * 0.5
            gt_ds = gt_semantic[::h//target_h, ::w//target_w][:target_h, :target_w]
            scores[c][gt_ds == c] += 1.5
        return np.argmax(scores, axis=0)

    # Multi-scale prediction
    multi_pred = classify(merged, target_h, target_w)
    gt_ds = gt_semantic[::h//target_h, ::w//target_w][:target_h, :target_w]
    multi_acc = np.mean(multi_pred == gt_ds)

    # Single-scale predictions
    print(f"\n  Accuracy Comparison:")
    print(f"    {'Scale':>8} | {'Accuracy':>8}")
    print(f"    {'-'*20}")

    for name in ['P2', 'P3', 'P4', 'P5']:
        single_pred = classify(features[name], target_h, target_w)
        single_acc = np.mean(single_pred == gt_ds)
        print(f"    {name:>8} | {single_acc:>8.4f}")

    print(f"    {'Merged':>8} | {multi_acc:>8.4f}")

    return merged, multi_pred


# =============================================================================
# Exercise 6: Temporal Panoptic Consistency
# =============================================================================

def exercise_6_temporal_consistency():
    """
    Track panoptic segments across consecutive frames.

    Steps:
    1. Generate panoptic maps for consecutive frames
    2. Match segments using IoU between frames
    3. Maintain consistent instance IDs
    4. Measure temporal consistency

    Returns:
        list of frame-to-frame matching results
    """
    np.random.seed(42)
    h, w = 64, 80
    n_frames = 5

    print("Temporal Panoptic Consistency")
    print(f"  Frames: {n_frames}, Scene: {w}x{h}")
    print("=" * 60)

    # Generate frames with gradually moving objects
    frames = []
    yy, xx = np.ogrid[:h, :w]

    for t in range(n_frames):
        instance_map = np.zeros((h, w), dtype=np.int32)
        segment_info = []
        next_id = 1

        # Stuff (static)
        instance_map[:20, :] = next_id
        segment_info.append({'id': next_id, 'category_id': 0, 'isthing': False})
        next_id += 1

        instance_map[50:, :] = next_id
        segment_info.append({'id': next_id, 'category_id': 1, 'isthing': False})
        next_id += 1

        instance_map[20:50, :] = next_id
        segment_info.append({'id': next_id, 'category_id': 2, 'isthing': False})
        next_id += 1

        # Things (moving)
        car1_cx = 20 + t * 5
        car1_mask = (((xx - car1_cx) / 10) ** 2 + ((yy - 38) / 7) ** 2) <= 1.0
        instance_map[car1_mask] = next_id
        segment_info.append({'id': next_id, 'category_id': 3, 'isthing': True})
        next_id += 1

        car2_cx = 60 - t * 3
        car2_mask = (((xx - car2_cx) / 8) ** 2 + ((yy - 42) / 6) ** 2) <= 1.0
        instance_map[car2_mask] = next_id
        segment_info.append({'id': next_id, 'category_id': 3, 'isthing': True})
        next_id += 1

        frames.append((instance_map, segment_info))

    # Track across frames
    tracking_results = []
    # Assign global IDs
    global_ids = {}  # (frame, local_id) -> global_id
    next_global = 1

    # First frame: assign fresh IDs
    for seg in frames[0][1]:
        global_ids[(0, seg['id'])] = next_global
        next_global += 1

    print(f"\n  Frame 0: {len(frames[0][1])} segments (initial)")

    for t in range(1, n_frames):
        prev_map, prev_segs = frames[t - 1]
        curr_map, curr_segs = frames[t]

        # Compute IoU between all segment pairs
        matches = []
        for curr_seg in curr_segs:
            curr_mask = curr_map == curr_seg['id']
            best_iou = 0.0
            best_prev = None

            for prev_seg in prev_segs:
                if prev_seg['category_id'] != curr_seg['category_id']:
                    continue
                prev_mask = prev_map == prev_seg['id']
                intersection = np.sum(curr_mask & prev_mask)
                union = np.sum(curr_mask | prev_mask)
                iou = intersection / (union + 1e-10)

                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev_seg

            if best_iou > 0.3 and best_prev is not None:
                prev_global = global_ids.get((t - 1, best_prev['id']))
                if prev_global is not None:
                    global_ids[(t, curr_seg['id'])] = prev_global
                    matches.append((curr_seg['id'], best_prev['id'],
                                    best_iou, prev_global))
                    continue

            # New segment
            global_ids[(t, curr_seg['id'])] = next_global
            matches.append((curr_seg['id'], None, 0.0, next_global))
            next_global += 1

        tracking_results.append(matches)

        tracked = sum(1 for m in matches if m[1] is not None)
        new = sum(1 for m in matches if m[1] is None)
        ious = [m[2] for m in matches if m[1] is not None]
        mean_iou = np.mean(ious) if ious else 0.0

        print(f"\n  Frame {t}: {len(curr_segs)} segments")
        print(f"    Tracked: {tracked}, New: {new}")
        print(f"    Mean tracking IoU: {mean_iou:.4f}")

        for local_id, prev_id, iou, gid in matches:
            status = f"matched prev={prev_id} (IoU={iou:.3f})" if prev_id else "NEW"
            print(f"    Seg {local_id} -> Global {gid}: {status}")

    # Consistency summary
    print(f"\n  Temporal Consistency Summary:")
    all_ious = []
    for frame_matches in tracking_results:
        for _, prev_id, iou, _ in frame_matches:
            if prev_id is not None:
                all_ious.append(iou)
    if all_ious:
        print(f"    Mean tracking IoU: {np.mean(all_ious):.4f}")
        print(f"    Min tracking IoU:  {np.min(all_ious):.4f}")
        print(f"    Tracks maintained: {len(all_ious)}")

    return tracking_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Stuff vs Things Analysis")
    exercise_1_stuff_things()

    print("\n>>> Exercise 2: Panoptic Fusion")
    exercise_2_panoptic_fusion()

    print("\n>>> Exercise 3: Panoptic Quality (PQ) Metric")
    exercise_3_panoptic_quality()

    print("\n>>> Exercise 4: Query-Based Segmentation")
    exercise_4_query_segmentation()

    print("\n>>> Exercise 5: Multi-Scale Feature Merging")
    exercise_5_multiscale_features()

    print("\n>>> Exercise 6: Temporal Panoptic Consistency")
    exercise_6_temporal_consistency()

    print("\nAll exercises completed successfully.")
