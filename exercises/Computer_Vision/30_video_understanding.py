"""
Exercise Solutions for Lesson 30: Video Understanding
Computer Vision - Temporal Modeling, 3D Convolutions, SlowFast, Action Recognition

Topics covered:
- Frame sampling and temporal aggregation
- 3D convolution for spatiotemporal features
- Two-stream architecture (spatial + temporal)
- SlowFast dual-pathway processing
- Temporal action detection with sliding window
- Video classification metrics
"""

import numpy as np


# =============================================================================
# Helper: Synthetic video generation
# =============================================================================

def generate_synthetic_video(n_frames=16, h=32, w=40, action="wave"):
    """
    Generate a synthetic video clip with a specific action pattern.

    Actions:
    - "wave": oscillating bright patch (left-right)
    - "jump": bright patch moving up then down
    - "still": static scene
    - "zoom": expanding circle

    Returns:
        video: (n_frames, h, w) grayscale
    """
    np.random.seed(42)
    video = np.random.randint(30, 60, (n_frames, h, w), dtype=np.uint8)

    if action == "wave":
        for t in range(n_frames):
            cx = int(w / 2 + 12 * np.sin(2 * np.pi * t / n_frames))
            cy = h // 2
            for i in range(max(0, cy-4), min(h, cy+4)):
                for j in range(max(0, cx-5), min(w, cx+5)):
                    video[t, i, j] = 180

    elif action == "jump":
        for t in range(n_frames):
            cx = w // 2
            cy = int(h / 2 - 10 * np.sin(np.pi * t / n_frames))
            for i in range(max(0, cy-3), min(h, cy+3)):
                for j in range(max(0, cx-3), min(w, cx+3)):
                    video[t, i, j] = 200

    elif action == "still":
        # Static bright rectangle
        video[:, 10:20, 15:30] = 170

    elif action == "zoom":
        for t in range(n_frames):
            cx, cy = w // 2, h // 2
            radius = 3 + t * 0.8
            for i in range(h):
                for j in range(w):
                    if (i - cy)**2 + (j - cx)**2 <= radius**2:
                        video[t, i, j] = 190

    return video


# =============================================================================
# Exercise 1: Frame Sampling and Temporal Aggregation
# =============================================================================

def exercise_1_temporal_aggregation():
    """
    Explore different frame sampling strategies for video understanding.

    Strategies:
    1. Uniform sampling
    2. Random sampling
    3. Temporal stride (subsample)
    4. Dense sliding window

    Returns:
        dict of strategy -> sampled frame indices
    """
    np.random.seed(42)
    total_frames = 120
    clip_length = 16

    print("Frame Sampling and Temporal Aggregation")
    print(f"  Video length: {total_frames} frames")
    print(f"  Clip length: {clip_length}")
    print("=" * 60)

    strategies = {}

    # 1. Uniform sampling
    uniform_indices = np.linspace(0, total_frames - 1, clip_length, dtype=int)
    strategies['uniform'] = uniform_indices

    # 2. Random sampling (sorted)
    random_indices = np.sort(np.random.choice(total_frames, clip_length, replace=False))
    strategies['random'] = random_indices

    # 3. Temporal stride
    strides = [2, 4, 8]
    for stride in strides:
        start = np.random.randint(0, max(1, total_frames - clip_length * stride))
        strided = np.arange(start, start + clip_length * stride, stride)[:clip_length]
        strided = np.clip(strided, 0, total_frames - 1)
        strategies[f'stride_{stride}'] = strided

    # 4. Dense (consecutive)
    start = np.random.randint(0, max(1, total_frames - clip_length))
    dense = np.arange(start, start + clip_length)
    strategies['dense'] = dense

    for name, indices in strategies.items():
        span = indices[-1] - indices[0] + 1
        gaps = np.diff(indices)
        print(f"\n  {name}:")
        print(f"    Indices: {indices.tolist()}")
        print(f"    Span: {span} frames ({span/total_frames*100:.0f}% of video)")
        if len(gaps) > 0:
            print(f"    Gap: mean={gaps.mean():.1f}, "
                  f"min={gaps.min()}, max={gaps.max()}")

    # Temporal aggregation comparison
    print(f"\n  Aggregation Methods:")
    video = generate_synthetic_video(total_frames, 32, 40, "wave")

    # Mean pooling
    mean_repr = video[uniform_indices].mean(axis=0)
    # Max pooling
    max_repr = video[uniform_indices].max(axis=0)
    # Temporal diff (motion)
    diff_repr = np.zeros((32, 40), dtype=np.float64)
    for i in range(1, len(uniform_indices)):
        diff_repr += np.abs(
            video[uniform_indices[i]].astype(np.float64) -
            video[uniform_indices[i-1]].astype(np.float64)
        )
    diff_repr /= (len(uniform_indices) - 1)

    print(f"    Mean pooling range: [{mean_repr.min():.0f}, {mean_repr.max():.0f}]")
    print(f"    Max pooling range:  [{max_repr.min():.0f}, {max_repr.max():.0f}]")
    print(f"    Temporal diff mean: {diff_repr.mean():.2f}")

    return strategies


# =============================================================================
# Exercise 2: 3D Convolution
# =============================================================================

def exercise_2_3d_convolution():
    """
    Implement 3D convolution for spatiotemporal feature extraction.

    Steps:
    1. Apply 3x3x3 convolution (temporal + spatial)
    2. Compare with 1x3x3 (spatial-only) and 3x1x1 (temporal-only)
    3. Analyze what temporal convolutions capture
    4. Compare computational cost

    Returns:
        dict of conv_type -> feature_map
    """
    np.random.seed(42)
    n_frames, h, w = 16, 32, 40

    # Generate videos with different motions
    video_wave = generate_synthetic_video(n_frames, h, w, "wave")
    video_still = generate_synthetic_video(n_frames, h, w, "still")

    video_f = video_wave.astype(np.float64) / 255.0

    print("3D Convolution for Video")
    print(f"  Video: {n_frames} frames x {w}x{h}")
    print("=" * 60)

    def conv3d(video, kernel, stride=1):
        """Apply 3D convolution with zero padding."""
        t, fh, fw = video.shape
        kt, kh, kw = kernel.shape
        pt, ph, pw = kt // 2, kh // 2, kw // 2

        padded = np.pad(video, ((pt, pt), (ph, ph), (pw, pw)), mode='constant')
        ot = (t + 2*pt - kt) // stride + 1
        oh = (fh + 2*ph - kh) // stride + 1
        ow = (fw + 2*pw - kw) // stride + 1

        out = np.zeros((ot, oh, ow), dtype=np.float64)
        for ti in range(ot):
            for hi in range(oh):
                for wi in range(ow):
                    si, sj, sk = ti * stride, hi * stride, wi * stride
                    patch = padded[si:si+kt, sj:sj+kh, sk:sk+kw]
                    out[ti, hi, wi] = np.sum(patch * kernel)

        return out

    # Kernels
    # 3x3x3: full spatiotemporal
    kernel_3d = np.random.randn(3, 3, 3) / 9
    kernel_3d[1, 1, 1] = 1.0  # Center weight

    # 1x3x3: spatial only (no temporal context)
    kernel_spatial = np.zeros((1, 3, 3))
    kernel_spatial[0] = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8

    # 3x1x1: temporal only (no spatial context)
    kernel_temporal = np.zeros((3, 1, 1))
    kernel_temporal[:, 0, 0] = [-1, 2, -1]  # Temporal difference

    conv_types = {
        '3x3x3 (spatiotemporal)': kernel_3d,
        '1x3x3 (spatial only)': kernel_spatial,
        '3x1x1 (temporal only)': kernel_temporal,
    }

    results = {}

    for name, kernel in conv_types.items():
        feat_wave = conv3d(video_f, kernel)
        feat_still = conv3d(
            generate_synthetic_video(n_frames, h, w, "still").astype(np.float64) / 255.0,
            kernel
        )

        feat_wave = np.maximum(feat_wave, 0)  # ReLU
        feat_still = np.maximum(feat_still, 0)

        results[name] = feat_wave

        # Compute motion sensitivity
        wave_energy = np.sum(feat_wave ** 2)
        still_energy = np.sum(feat_still ** 2)
        motion_ratio = wave_energy / (still_energy + 1e-10)

        kt, kh, kw = kernel.shape
        flops = n_frames * h * w * kt * kh * kw

        print(f"\n  {name}:")
        print(f"    Kernel shape: {kernel.shape}")
        print(f"    Output shape: {feat_wave.shape}")
        print(f"    Wave energy: {wave_energy:.4f}")
        print(f"    Still energy: {still_energy:.4f}")
        print(f"    Motion sensitivity: {motion_ratio:.2f}x")
        print(f"    FLOPs: {flops:,}")

    # (2+1)D decomposition analysis
    print(f"\n  (2+1)D Decomposition:")
    spatial_params = 1 * 3 * 3
    temporal_params = 3 * 1 * 1
    full_params = 3 * 3 * 3
    print(f"    3x3x3 params: {full_params}")
    print(f"    1x3x3 + 3x1x1 params: {spatial_params + temporal_params}")
    print(f"    Reduction: {100*(1 - (spatial_params+temporal_params)/full_params):.0f}%")

    return results


# =============================================================================
# Exercise 3: Two-Stream Architecture
# =============================================================================

def exercise_3_two_stream():
    """
    Simulate a two-stream architecture for action recognition.

    Streams:
    1. Spatial stream: appearance from single RGB frame
    2. Temporal stream: motion from frame differences (simulated optical flow)

    Returns:
        (spatial_scores, temporal_scores, fused_scores)
    """
    np.random.seed(42)
    n_frames, h, w = 16, 32, 40
    n_classes = 4  # wave, jump, still, zoom
    actions = ["wave", "jump", "still", "zoom"]

    print("Two-Stream Architecture Simulation")
    print(f"  Actions: {actions}")
    print(f"  Video: {n_frames}x{w}x{h}")
    print("=" * 60)

    # Generate videos for each action
    videos = {}
    for action in actions:
        videos[action] = generate_synthetic_video(n_frames, h, w, action)

    def extract_spatial_features(video):
        """Extract appearance features from middle frame."""
        frame = video[n_frames // 2].astype(np.float64) / 255.0
        # Simple statistics as features
        mean_val = frame.mean()
        std_val = frame.std()
        # Spatial distribution
        top_half = frame[:h//2].mean()
        bottom_half = frame[h//2:].mean()
        left_half = frame[:, :w//2].mean()
        right_half = frame[:, w//2:].mean()
        # Brightness histogram
        hist = np.histogram(frame, bins=8, range=(0, 1))[0].astype(np.float64)
        hist /= hist.sum() + 1e-10
        return np.concatenate([[mean_val, std_val, top_half, bottom_half,
                               left_half, right_half], hist])

    def extract_temporal_features(video):
        """Extract motion features from frame differences."""
        diffs = np.abs(np.diff(video.astype(np.float64), axis=0))
        # Overall motion
        total_motion = diffs.mean()
        motion_std = diffs.std()
        # Motion location
        motion_center_y = 0.0
        motion_center_x = 0.0
        total_weight = 0.0
        for t in range(diffs.shape[0]):
            for i in range(h):
                for j in range(w):
                    motion_center_y += i * diffs[t, i, j]
                    motion_center_x += j * diffs[t, i, j]
                    total_weight += diffs[t, i, j]
        if total_weight > 0:
            motion_center_y /= total_weight
            motion_center_x /= total_weight

        # Motion over time
        motion_per_frame = diffs.mean(axis=(1, 2))
        motion_variance = motion_per_frame.var()
        motion_max = motion_per_frame.max()

        # Periodicity (FFT of motion signal)
        fft_mag = np.abs(np.fft.rfft(motion_per_frame))
        dominant_freq = np.argmax(fft_mag[1:]) + 1

        return np.array([total_motion, motion_std, motion_center_y / h,
                        motion_center_x / w, motion_variance, motion_max,
                        float(dominant_freq)])

    # Extract features for all actions
    spatial_feats = {}
    temporal_feats = {}
    for action in actions:
        spatial_feats[action] = extract_spatial_features(videos[action])
        temporal_feats[action] = extract_temporal_features(videos[action])

    # Simple nearest-neighbor classification
    def classify(query_feat, database, action_labels):
        """Classify by comparing features to database."""
        scores = np.zeros(len(action_labels))
        for i, action in enumerate(action_labels):
            dist = np.linalg.norm(query_feat - database[action])
            scores[i] = 1.0 / (1.0 + dist)
        # Normalize to probabilities
        scores /= scores.sum() + 1e-10
        return scores

    print(f"\n  Classification Results:")
    print(f"    {'Action':>8} | {'Spatial':>20} | {'Temporal':>20} | {'Fused':>20}")
    print(f"    {'-'*75}")

    all_spatial = []
    all_temporal = []
    all_fused = []

    for test_action in actions:
        s_scores = classify(spatial_feats[test_action], spatial_feats, actions)
        t_scores = classify(temporal_feats[test_action], temporal_feats, actions)
        f_scores = 0.4 * s_scores + 0.6 * t_scores  # Late fusion

        all_spatial.append(s_scores)
        all_temporal.append(t_scores)
        all_fused.append(f_scores)

        s_pred = actions[np.argmax(s_scores)]
        t_pred = actions[np.argmax(t_scores)]
        f_pred = actions[np.argmax(f_scores)]

        print(f"    {test_action:>8} | pred={s_pred:>6} ({s_scores.max():.2f}) | "
              f"pred={t_pred:>6} ({t_scores.max():.2f}) | "
              f"pred={f_pred:>6} ({f_scores.max():.2f})")

    # Stream comparison
    print(f"\n  Stream Analysis:")
    print(f"    Spatial features per action:")
    for action in actions:
        feat = spatial_feats[action]
        print(f"      {action}: mean={feat[:2].tolist()}")
    print(f"    Temporal features per action:")
    for action in actions:
        feat = temporal_feats[action]
        print(f"      {action}: motion={feat[0]:.4f}, "
              f"var={feat[4]:.6f}, freq={feat[6]:.0f}")

    return np.array(all_spatial), np.array(all_temporal), np.array(all_fused)


# =============================================================================
# Exercise 4: SlowFast Dual-Pathway
# =============================================================================

def exercise_4_slowfast():
    """
    Simulate the SlowFast dual-pathway architecture.

    Slow path: low frame rate, high channel capacity
    Fast path: high frame rate, low channel capacity

    Returns:
        (slow_features, fast_features, combined)
    """
    np.random.seed(42)
    n_frames, h, w = 32, 32, 40
    alpha = 8  # Frame rate ratio

    video = generate_synthetic_video(n_frames, h, w, "wave")
    video_f = video.astype(np.float64) / 255.0

    print("SlowFast Dual-Pathway Simulation")
    print(f"  Video: {n_frames} frames x {w}x{h}")
    print(f"  Alpha (rate ratio): {alpha}")
    print("=" * 60)

    # Slow pathway: subsample every alpha frames
    slow_indices = np.arange(0, n_frames, alpha)
    slow_input = video_f[slow_indices]
    n_slow = len(slow_indices)

    # Fast pathway: all frames
    fast_input = video_f

    print(f"\n  Pathway Input:")
    print(f"    Slow: {n_slow} frames (every {alpha}th frame)")
    print(f"    Fast: {n_frames} frames (all)")

    # Process slow pathway (high capacity - more features)
    slow_features = np.zeros((n_slow, 4), dtype=np.float64)
    for t in range(n_slow):
        frame = slow_input[t]
        slow_features[t, 0] = frame.mean()
        slow_features[t, 1] = frame.std()
        slow_features[t, 2] = frame[:h//2].mean()
        slow_features[t, 3] = frame[h//2:].mean()

    # Process fast pathway (low capacity - fewer features)
    fast_features = np.zeros((n_frames, 1), dtype=np.float64)
    for t in range(n_frames):
        if t > 0:
            fast_features[t, 0] = np.mean(np.abs(
                video_f[t] - video_f[t-1]))

    # Lateral connection: fast -> slow
    # Temporal downsampling of fast features to match slow
    lateral = np.zeros((n_slow, 1), dtype=np.float64)
    for i in range(n_slow):
        start = i * alpha
        end = min(start + alpha, n_frames)
        lateral[i, 0] = fast_features[start:end].mean()

    # Combine slow + lateral
    slow_enhanced = np.concatenate([slow_features, lateral], axis=1)

    print(f"\n  Pathway Features:")
    print(f"    Slow: {slow_features.shape} (high capacity)")
    print(f"    Fast: {fast_features.shape} (low capacity)")
    print(f"    Lateral: {lateral.shape}")
    print(f"    Slow+Lateral: {slow_enhanced.shape}")

    # Global pooling
    slow_pooled = slow_enhanced.mean(axis=0)
    fast_pooled = fast_features.mean(axis=0)
    combined = np.concatenate([slow_pooled, fast_pooled])

    print(f"\n  After Global Average Pooling:")
    print(f"    Slow: {slow_pooled}")
    print(f"    Fast: {fast_pooled}")
    print(f"    Combined: {combined}")

    # Temporal resolution analysis
    print(f"\n  Temporal Resolution Analysis:")
    print(f"    Slow pathway: captures every {alpha}th frame")
    print(f"      Good for: spatial appearance, object identity")
    print(f"      Misses: fast motions, quick gestures")

    print(f"    Fast pathway: captures every frame")
    print(f"      Good for: fine motion, action timing")
    print(f"      Efficient: only {100//alpha}% channels of slow")

    # Compute information content
    slow_entropy = -np.sum(
        slow_features * np.log(np.abs(slow_features) + 1e-10)
    ) / slow_features.size
    fast_entropy = -np.sum(
        fast_features * np.log(np.abs(fast_features) + 1e-10)
    ) / fast_features.size

    print(f"\n  Information Content:")
    print(f"    Slow path entropy: {slow_entropy:.4f}")
    print(f"    Fast path entropy: {fast_entropy:.4f}")

    return slow_features, fast_features, combined


# =============================================================================
# Exercise 5: Temporal Action Detection
# =============================================================================

def exercise_5_action_detection():
    """
    Detect actions in an untrimmed video using sliding window.

    Steps:
    1. Generate a long video with multiple actions
    2. Apply sliding window classification
    3. Apply temporal non-maximum suppression
    4. Evaluate detection accuracy

    Returns:
        list of detected action segments
    """
    np.random.seed(42)
    total_frames = 200
    h, w = 32, 40
    n_classes = 4
    actions = ["wave", "jump", "still", "zoom"]

    print("Temporal Action Detection")
    print(f"  Total frames: {total_frames}")
    print(f"  Actions: {actions}")
    print("=" * 60)

    # Ground truth: sequence of actions
    gt_segments = [
        (0, 30, 2),     # still
        (30, 70, 0),    # wave
        (70, 100, 2),   # still
        (100, 140, 1),  # jump
        (140, 170, 3),  # zoom
        (170, 200, 2),  # still
    ]

    # Generate concatenated video
    video = np.random.randint(30, 60, (total_frames, h, w), dtype=np.uint8)
    for start, end, action_id in gt_segments:
        action = actions[action_id]
        clip = generate_synthetic_video(end - start, h, w, action)
        video[start:end] = clip

    print(f"\n  Ground Truth:")
    for start, end, action_id in gt_segments:
        print(f"    [{start:>3d}-{end:>3d}] {actions[action_id]}")

    # Sliding window detection
    window_size = 16
    stride = 4
    detections = []

    def classify_clip(clip):
        """Simple motion-based classifier."""
        diffs = np.abs(np.diff(clip.astype(np.float64), axis=0))
        total_motion = diffs.mean()
        motion_var = diffs.mean(axis=(1, 2)).var()

        # Motion periodicity
        motion_signal = diffs.mean(axis=(1, 2))
        if len(motion_signal) > 1:
            fft_mag = np.abs(np.fft.rfft(motion_signal))
            periodic = fft_mag[1:].max() / (fft_mag[0] + 1e-10)
        else:
            periodic = 0

        # Simple rules
        scores = np.zeros(n_classes)
        if total_motion < 2:
            scores[2] = 0.8  # still
        elif periodic > 0.3:
            scores[0] = 0.7  # wave (periodic)
        elif motion_var > 5:
            scores[1] = 0.6  # jump (variable motion)
        else:
            scores[3] = 0.5  # zoom

        scores += np.random.uniform(0, 0.15, n_classes)
        return scores

    for start in range(0, total_frames - window_size, stride):
        end = start + window_size
        clip = video[start:end]
        scores = classify_clip(clip)
        action_id = np.argmax(scores)
        confidence = scores[action_id]

        if confidence > 0.4:
            detections.append({
                'start': start,
                'end': end,
                'action': action_id,
                'confidence': confidence,
            })

    print(f"\n  Raw Detections: {len(detections)}")

    # Temporal NMS
    def temporal_nms(dets, iou_threshold=0.3):
        """Non-maximum suppression for temporal segments."""
        if not dets:
            return []

        sorted_dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        keep = []
        suppressed = set()

        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue
            keep.append(det)

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue
                if sorted_dets[j]['action'] != det['action']:
                    continue

                # Temporal IoU
                inter_start = max(det['start'], sorted_dets[j]['start'])
                inter_end = min(det['end'], sorted_dets[j]['end'])
                intersection = max(0, inter_end - inter_start)
                union = ((det['end'] - det['start']) +
                         (sorted_dets[j]['end'] - sorted_dets[j]['start']) -
                         intersection)
                tiou = intersection / (union + 1e-10)

                if tiou > iou_threshold:
                    suppressed.add(j)

        return keep

    nms_detections = temporal_nms(detections)
    print(f"  After NMS: {len(nms_detections)}")

    # Merge overlapping detections of same class
    merged = []
    used = set()
    for i, det in enumerate(nms_detections):
        if i in used:
            continue
        merged_det = dict(det)
        for j in range(i + 1, len(nms_detections)):
            if j in used:
                continue
            if (nms_detections[j]['action'] == det['action'] and
                    nms_detections[j]['start'] <= merged_det['end'] + stride):
                merged_det['end'] = max(merged_det['end'], nms_detections[j]['end'])
                merged_det['confidence'] = max(
                    merged_det['confidence'], nms_detections[j]['confidence'])
                used.add(j)
        merged.append(merged_det)

    print(f"  After Merge: {len(merged)}")
    print(f"\n  Final Detections:")
    for det in merged:
        print(f"    [{det['start']:>3d}-{det['end']:>3d}] "
              f"{actions[det['action']]} (conf={det['confidence']:.2f})")

    return merged


# =============================================================================
# Exercise 6: Video Classification Metrics
# =============================================================================

def exercise_6_video_metrics():
    """
    Compute video classification evaluation metrics.

    Metrics:
    1. Top-1 and Top-5 accuracy
    2. Per-class precision, recall, F1
    3. Confusion matrix
    4. Mean Average Precision

    Returns:
        dict of metrics
    """
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    class_names = ["walk", "run", "sit", "wave", "jump"]

    # Simulate predictions
    gt_labels = np.random.randint(0, n_classes, n_samples)
    # Make predictions somewhat correlated with GT
    pred_logits = np.random.randn(n_samples, n_classes) * 0.5
    for i in range(n_samples):
        pred_logits[i, gt_labels[i]] += 2.0  # Boost correct class
        # Add some confusion with similar classes
        confused = (gt_labels[i] + 1) % n_classes
        pred_logits[i, confused] += np.random.uniform(0, 1.5)

    pred_labels = np.argmax(pred_logits, axis=1)

    print("Video Classification Metrics")
    print(f"  Samples: {n_samples}, Classes: {n_classes}")
    print("=" * 60)

    # Top-1 accuracy
    top1_correct = np.sum(pred_labels == gt_labels)
    top1_acc = top1_correct / n_samples

    # Top-5 (top-k where k = min(5, n_classes))
    k = min(5, n_classes)
    top_k_correct = 0
    for i in range(n_samples):
        top_k_preds = np.argsort(pred_logits[i])[-k:]
        if gt_labels[i] in top_k_preds:
            top_k_correct += 1
    top_k_acc = top_k_correct / n_samples

    print(f"\n  Top-1 Accuracy: {top1_acc:.4f} ({top1_correct}/{n_samples})")
    print(f"  Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_correct}/{n_samples})")

    # Confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(n_samples):
        conf_matrix[gt_labels[i], pred_labels[i]] += 1

    print(f"\n  Confusion Matrix:")
    header = "       " + "  ".join(f"{n:>5}" for n in class_names)
    print(f"  {header}")
    for i in range(n_classes):
        row = "  ".join(f"{conf_matrix[i, j]:>5}" for j in range(n_classes))
        print(f"    {class_names[i]:>5}  {row}")

    # Per-class metrics
    print(f"\n  Per-Class Metrics:")
    print(f"    {'Class':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7}")
    print(f"    {'-'*42}")

    precisions = []
    recalls = []
    f1s = []

    for c in range(n_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        support = conf_matrix[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(f"    {class_names[c]:>6} | {precision:>6.3f} | {recall:>6.3f} | "
              f"{f1:>6.3f} | {support:>7}")

    macro_f1 = np.mean(f1s)
    macro_prec = np.mean(precisions)
    macro_recall = np.mean(recalls)
    print(f"\n  Macro Average: Prec={macro_prec:.3f}, "
          f"Recall={macro_recall:.3f}, F1={macro_f1:.3f}")

    metrics = {
        'top1_accuracy': top1_acc,
        f'top{k}_accuracy': top_k_acc,
        'macro_f1': macro_f1,
        'confusion_matrix': conf_matrix,
        'per_class_f1': f1s,
    }

    return metrics


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Frame Sampling and Temporal Aggregation")
    exercise_1_temporal_aggregation()

    print("\n>>> Exercise 2: 3D Convolution")
    exercise_2_3d_convolution()

    print("\n>>> Exercise 3: Two-Stream Architecture")
    exercise_3_two_stream()

    print("\n>>> Exercise 4: SlowFast Dual-Pathway")
    exercise_4_slowfast()

    print("\n>>> Exercise 5: Temporal Action Detection")
    exercise_5_action_detection()

    print("\n>>> Exercise 6: Video Classification Metrics")
    exercise_6_video_metrics()

    print("\nAll exercises completed successfully.")
