"""
Exercises for Lesson 15: Practical Edge Vision
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === Exercise 1: Object Detection Pipeline ===
# Problem: Implement a complete object detection postprocessing pipeline
# including confidence filtering, class selection, and NMS.

def exercise_1():
    """Implement object detection postprocessing pipeline."""
    np.random.seed(42)

    # Simulate raw detector output: 100 candidate boxes
    n_boxes = 100
    n_classes = 5
    class_names = ["person", "car", "dog", "cat", "bicycle"]

    # Generate synthetic detections
    # Format: [x1, y1, x2, y2] normalized to [0, 1]
    boxes = np.random.rand(n_boxes, 4)
    # Ensure x2 > x1 and y2 > y1
    boxes[:, 2] = boxes[:, 0] + np.random.rand(n_boxes) * 0.3
    boxes[:, 3] = boxes[:, 1] + np.random.rand(n_boxes) * 0.3
    boxes = np.clip(boxes, 0, 1)

    # Confidence scores (most are low)
    scores = np.random.exponential(0.1, n_boxes)
    scores = np.clip(scores, 0, 1)
    # Make a few high-confidence
    scores[:5] = np.random.uniform(0.7, 0.99, 5)

    # Class predictions
    classes = np.random.randint(0, n_classes, n_boxes)

    print(f"  Raw detector output: {n_boxes} candidate boxes")
    print(f"  Score distribution: min={scores.min():.3f}, "
          f"max={scores.max():.3f}, mean={scores.mean():.3f}")

    # Step 1: Confidence filtering
    conf_threshold = 0.5
    high_conf_mask = scores >= conf_threshold
    filtered_boxes = boxes[high_conf_mask]
    filtered_scores = scores[high_conf_mask]
    filtered_classes = classes[high_conf_mask]

    print(f"\n  Step 1: Confidence filter (threshold={conf_threshold})")
    print(f"    {high_conf_mask.sum()} / {n_boxes} boxes remain")

    # Step 2: Non-Maximum Suppression (NMS)
    def compute_iou(box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def nms(boxes, scores, classes, iou_threshold=0.5):
        """Per-class Non-Maximum Suppression."""
        keep = []
        unique_classes = np.unique(classes)

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            # Sort by score (descending)
            order = cls_scores.argsort()[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            selected = []
            remaining = list(range(len(cls_boxes)))

            while remaining:
                best = remaining[0]
                selected.append(best)
                remaining.remove(best)

                to_remove = []
                for idx in remaining:
                    iou = compute_iou(cls_boxes[best], cls_boxes[idx])
                    if iou > iou_threshold:
                        to_remove.append(idx)

                for idx in to_remove:
                    remaining.remove(idx)

            for idx in selected:
                keep.append({
                    "box": cls_boxes[idx],
                    "score": cls_scores[idx],
                    "class": cls,
                })

        return keep

    iou_threshold = 0.5
    detections = nms(filtered_boxes, filtered_scores,
                     filtered_classes, iou_threshold)

    print(f"\n  Step 2: NMS (IoU threshold={iou_threshold})")
    print(f"    {len(detections)} final detections")

    # Step 3: Display results
    print(f"\n  Final Detections:")
    print(f"  {'Class':<10} {'Score':>8} {'Box (x1,y1,x2,y2)'}")
    print("  " + "-" * 50)

    detections.sort(key=lambda d: -d['score'])
    for det in detections:
        box = det['box']
        print(f"  {class_names[det['class']]:<10} {det['score']:>8.3f} "
              f"[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]")


# === Exercise 2: Edge-Optimized Detection Model ===
# Problem: Design and analyze an edge-optimized object detection model
# for deployment on Jetson Nano or Raspberry Pi.

def exercise_2():
    """Design edge-optimized detection model."""
    print("  Edge Detection Model Design Comparison:\n")

    models = [
        {
            "name": "YOLOv5n (nano)",
            "backbone": "CSPDarknet (0.33x width)",
            "input_size": 640,
            "params_m": 1.9,
            "gflops": 4.5,
            "coco_map": 28.0,
            "jetson_nano_fps": 25,
            "rpi5_fps": 5,
        },
        {
            "name": "YOLOv5s (small)",
            "backbone": "CSPDarknet (0.5x width)",
            "input_size": 640,
            "params_m": 7.2,
            "gflops": 16.5,
            "coco_map": 37.4,
            "jetson_nano_fps": 12,
            "rpi5_fps": 2,
        },
        {
            "name": "SSD-MobileNet-V2",
            "backbone": "MobileNet-V2",
            "input_size": 320,
            "params_m": 3.4,
            "gflops": 1.5,
            "coco_map": 22.0,
            "jetson_nano_fps": 45,
            "rpi5_fps": 12,
        },
        {
            "name": "EfficientDet-D0",
            "backbone": "EfficientNet-B0",
            "input_size": 512,
            "params_m": 3.9,
            "gflops": 2.5,
            "coco_map": 34.6,
            "jetson_nano_fps": 18,
            "rpi5_fps": 4,
        },
        {
            "name": "NanoDet-Plus",
            "backbone": "ShuffleNet-V2",
            "input_size": 416,
            "params_m": 1.2,
            "gflops": 1.5,
            "coco_map": 30.4,
            "jetson_nano_fps": 40,
            "rpi5_fps": 10,
        },
    ]

    print(f"  {'Model':<22} {'Params':>7} {'GFLOPs':>8} {'mAP':>6} "
          f"{'Jetson':>8} {'RPi5':>7}")
    print("  " + "-" * 62)

    for m in models:
        print(f"  {m['name']:<22} {m['params_m']:>5.1f}M {m['gflops']:>8.1f} "
              f"{m['coco_map']:>5.1f} {m['jetson_nano_fps']:>6}fps "
              f"{m['rpi5_fps']:>5}fps")

    print("\n  Optimization strategy for edge deployment:")
    print("  1. Reduce input resolution (640 -> 320): ~4x fewer FLOPs")
    print("  2. Use INT8 quantization (TensorRT): 2-3x speedup on Jetson")
    print("  3. Choose anchor-free detector (NanoDet): simpler postprocess")
    print("  4. Prune unused classes: fewer output channels")
    print("  5. Batch frames if latency budget allows: better throughput")


# === Exercise 3: Non-Maximum Suppression Implementation ===
# Problem: Implement vectorized NMS using PyTorch tensors for
# efficient batch processing.

def exercise_3():
    """Implement vectorized NMS with PyTorch."""
    torch.manual_seed(42)

    def nms_pytorch(boxes, scores, iou_threshold=0.5):
        """
        Vectorized NMS using PyTorch.

        Args:
            boxes: [N, 4] tensor of (x1, y1, x2, y2)
            scores: [N] tensor of confidence scores
            iou_threshold: IoU threshold for suppression

        Returns:
            keep: indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long)

        # Sort by score (descending)
        order = scores.argsort(descending=True)

        # Compute areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []

        while order.numel() > 0:
            # Pick the best remaining box
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            # Compute IoU of best box with all remaining
            remaining = order[1:]

            xx1 = torch.clamp(x1[remaining], min=x1[i])
            yy1 = torch.clamp(y1[remaining], min=y1[i])
            xx2 = torch.clamp(x2[remaining], max=x2[i])
            yy2 = torch.clamp(y2[remaining], max=y2[i])

            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            union = areas[i] + areas[remaining] - inter
            iou = inter / (union + 1e-6)

            # Keep boxes with IoU below threshold
            mask = iou <= iou_threshold
            order = remaining[mask]

        return torch.tensor(keep, dtype=torch.long)

    # Test with synthetic data
    n = 50
    boxes = torch.rand(n, 4)
    boxes[:, 2] = boxes[:, 0] + torch.rand(n) * 0.3
    boxes[:, 3] = boxes[:, 1] + torch.rand(n) * 0.3
    boxes.clamp_(0, 1)
    scores = torch.rand(n)

    keep = nms_pytorch(boxes, scores, iou_threshold=0.5)

    print(f"  Input:  {n} boxes")
    print(f"  Output: {len(keep)} boxes after NMS (IoU > 0.5 suppressed)")

    # Compare with different thresholds
    print(f"\n  {'IoU Threshold':>14} {'Boxes Kept':>12} {'Suppressed':>12}")
    print("  " + "-" * 40)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        kept = nms_pytorch(boxes, scores, threshold)
        print(f"  {threshold:>14.1f} {len(kept):>12} {n - len(kept):>12}")

    print("\n  Lower IoU threshold = more aggressive suppression")
    print("  Higher IoU threshold = more boxes kept (more overlap tolerated)")

    # Verify with torchvision (if available)
    try:
        from torchvision.ops import nms as tv_nms
        tv_keep = tv_nms(boxes, scores, 0.5)
        our_keep = nms_pytorch(boxes, scores, 0.5)
        match = set(our_keep.tolist()) == set(tv_keep.tolist())
        print(f"\n  Verification vs torchvision.ops.nms: {'MATCH' if match else 'DIFFER'}")
    except ImportError:
        print("\n  (torchvision not available for verification)")


# === Exercise 4: Edge Vision Pipeline Optimization ===
# Problem: Analyze and optimize an end-to-end vision pipeline for
# maximum throughput on an edge device.

def exercise_4():
    """Optimize edge vision pipeline for throughput."""
    print("  Edge Vision Pipeline Optimization:\n")

    # Baseline pipeline (sequential)
    baseline = {
        "name": "Baseline (Sequential)",
        "stages": [
            ("Capture", 2),
            ("Decode (JPEG)", 5),
            ("Preprocess (resize+normalize)", 3),
            ("Inference", 15),
            ("Postprocess (NMS)", 2),
            ("Display/Action", 1),
        ],
    }

    total_baseline = sum(t for _, t in baseline['stages'])
    baseline_fps = 1000 / total_baseline

    print(f"  {baseline['name']}:")
    for name, ms in baseline['stages']:
        bar = "#" * int(ms * 2)
        print(f"    {name:<30} {ms:>4}ms {bar}")
    print(f"    {'TOTAL':<30} {total_baseline:>4}ms ({baseline_fps:.0f} FPS)")

    # Optimized pipeline (pipelined + optimized)
    print(f"\n  Optimization techniques:\n")

    optimizations = [
        {
            "technique": "1. Pipeline parallelism",
            "description": "Overlap capture(N+1) with inference(N)",
            "saving_ms": 2,
            "reason": "Camera capture runs in parallel with GPU inference",
        },
        {
            "technique": "2. Hardware JPEG decode",
            "description": "Use NVJPEG or DSP for decoding",
            "saving_ms": 3,
            "reason": "Dedicated hardware decoder frees CPU",
        },
        {
            "technique": "3. GPU preprocessing",
            "description": "Resize + normalize on GPU (DALI or custom)",
            "saving_ms": 2,
            "reason": "Avoid CPU-GPU data transfer for preprocessing",
        },
        {
            "technique": "4. INT8 quantization (TensorRT)",
            "description": "FP32 -> INT8 with calibration",
            "saving_ms": 8,
            "reason": "4x throughput on Tensor Cores",
        },
        {
            "technique": "5. Batching (batch=4)",
            "description": "Process 4 frames together",
            "saving_ms": 0,
            "reason": "Amortizes fixed overhead, increases throughput",
        },
    ]

    cumulative_saving = 0
    for opt in optimizations:
        cumulative_saving += opt['saving_ms']
        new_total = total_baseline - cumulative_saving
        new_fps = 1000 / max(1, new_total)
        print(f"  {opt['technique']}")
        print(f"    {opt['description']}")
        print(f"    Saves ~{opt['saving_ms']}ms -> {new_total}ms "
              f"({new_fps:.0f} FPS)")
        print()

    optimized_total = total_baseline - cumulative_saving
    optimized_fps = 1000 / max(1, optimized_total)

    print(f"  Summary:")
    print(f"    Baseline:  {total_baseline}ms ({baseline_fps:.0f} FPS)")
    print(f"    Optimized: {optimized_total}ms ({optimized_fps:.0f} FPS)")
    print(f"    Speedup:   {baseline_fps / max(1, optimized_fps):.1f}x -> "
          f"{optimized_fps:.0f}x")
    print(f"\n  With batching (batch=4): throughput = {optimized_fps * 3:.0f}+ FPS")
    print(f"  (3x throughput improvement, but per-frame latency = {optimized_total * 4}ms)")


if __name__ == "__main__":
    print("=== Exercise 1: Object Detection Pipeline ===")
    exercise_1()
    print("\n=== Exercise 2: Edge-Optimized Detection Model ===")
    exercise_2()
    print("\n=== Exercise 3: NMS Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Edge Vision Pipeline Optimization ===")
    exercise_4()
    print("\nAll exercises completed!")
