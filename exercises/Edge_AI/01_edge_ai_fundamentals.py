"""
Exercises for Lesson 01: Edge AI Fundamentals
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import math


# === Exercise 1: Edge vs Cloud Latency Analysis ===
# Problem: Calculate end-to-end latency for cloud vs edge inference.
# Include network round-trip, serialization, and inference time.

def exercise_1():
    """Compare cloud and edge inference latency."""
    # Cloud inference scenario
    cloud = {
        "network_rtt_ms": 50,          # Round-trip to cloud
        "serialization_ms": 5,          # Data serialization/deserialization
        "upload_ms": 20,                # Upload image (1MB at 400 Mbps)
        "inference_ms": 10,             # GPU inference (fast)
        "download_ms": 1,               # Download result (small JSON)
    }

    # Edge inference scenario
    edge = {
        "network_rtt_ms": 0,            # No network needed
        "serialization_ms": 0,          # No serialization
        "upload_ms": 0,                 # No upload
        "inference_ms": 50,             # Edge device (slower)
        "download_ms": 0,               # No download
    }

    cloud_total = sum(cloud.values())
    edge_total = sum(edge.values())

    print("  Cloud inference latency breakdown:")
    for stage, ms in cloud.items():
        bar = "#" * int(ms)
        print(f"    {stage:<22} {ms:>5} ms  {bar}")
    print(f"    {'TOTAL':<22} {cloud_total:>5} ms")

    print("\n  Edge inference latency breakdown:")
    for stage, ms in edge.items():
        bar = "#" * int(ms)
        print(f"    {stage:<22} {ms:>5} ms  {bar}")
    print(f"    {'TOTAL':<22} {edge_total:>5} ms")

    print(f"\n  Edge is {cloud_total / edge_total:.1f}x faster end-to-end")
    print(f"  Even though cloud inference alone is {edge['inference_ms'] / cloud['inference_ms']:.0f}x faster,")
    print(f"  network overhead makes cloud {cloud_total - cloud['inference_ms']}ms slower to start.")

    # Under unreliable network conditions
    print("\n  With 10% packet loss (adds ~200ms retransmission):")
    cloud_unreliable = cloud_total + 200
    print(f"    Cloud: {cloud_unreliable} ms vs Edge: {edge_total} ms")
    print(f"    Edge is {cloud_unreliable / edge_total:.1f}x faster under poor connectivity")


# === Exercise 2: Privacy and Bandwidth Tradeoff ===
# Problem: Calculate bandwidth savings and privacy risk reduction
# for a smart camera system processing locally vs in cloud.

def exercise_2():
    """Privacy and bandwidth analysis for edge AI."""
    # Smart security camera scenario
    fps = 30
    resolution = (1920, 1080)
    bytes_per_pixel = 3  # RGB
    jpeg_compression_ratio = 10

    # Raw frame size
    raw_frame_bytes = resolution[0] * resolution[1] * bytes_per_pixel
    compressed_frame_bytes = raw_frame_bytes / jpeg_compression_ratio

    # Data per second
    data_per_second = compressed_frame_bytes * fps
    data_per_hour = data_per_second * 3600
    data_per_day = data_per_hour * 24

    print("  Smart Camera: Cloud Processing")
    print(f"    Resolution: {resolution[0]}x{resolution[1]} @ {fps} FPS")
    print(f"    Raw frame: {raw_frame_bytes / 1024:.0f} KB")
    print(f"    Compressed frame: {compressed_frame_bytes / 1024:.0f} KB")
    print(f"    Bandwidth needed: {data_per_second / 1e6:.1f} MB/s")
    print(f"    Daily data: {data_per_day / 1e9:.1f} GB")
    print(f"    Monthly data: {data_per_day * 30 / 1e9:.0f} GB")
    print(f"    Privacy risk: ALL video frames sent to cloud")

    # Edge processing: only send detections
    detections_per_hour = 50  # avg person detections
    metadata_per_detection = 256  # bytes (bbox, timestamp, confidence)
    thumbnail_per_detection = 10_000  # bytes (small crop)

    edge_per_hour = detections_per_hour * (metadata_per_detection + thumbnail_per_detection)
    edge_per_day = edge_per_hour * 24

    print(f"\n  Smart Camera: Edge Processing")
    print(f"    Only detection metadata + thumbnails sent")
    print(f"    Data per detection: {metadata_per_detection + thumbnail_per_detection:.0f} bytes")
    print(f"    Daily data: {edge_per_day / 1e6:.2f} MB")

    bandwidth_reduction = data_per_day / edge_per_day
    print(f"\n  Bandwidth reduction: {bandwidth_reduction:.0f}x")
    print(f"  Cloud: {data_per_day / 1e9:.1f} GB/day vs Edge: {edge_per_day / 1e6:.2f} MB/day")
    print(f"  Privacy: Raw video NEVER leaves the device")


# === Exercise 3: Edge Computing Spectrum ===
# Problem: Classify deployment scenarios along the edge computing spectrum
# and explain the tradeoffs for each.

def exercise_3():
    """Classify scenarios along the cloud-to-edge spectrum."""
    spectrum = [
        {
            "level": "Cloud",
            "scenario": "LLM text generation (GPT-4 class)",
            "latency": "200-500ms",
            "compute": "8x A100 GPUs",
            "justification": (
                "Model too large (hundreds of GB) for any edge device. "
                "Latency acceptable for conversational AI."
            ),
        },
        {
            "level": "Edge Server",
            "scenario": "Factory quality inspection",
            "latency": "10-50ms",
            "compute": "NVIDIA Jetson AGX / edge GPU",
            "justification": (
                "Requires real-time response for production line speed. "
                "High-accuracy model needs GPU compute. "
                "Data stays on premises (IP protection)."
            ),
        },
        {
            "level": "Mobile/Tablet",
            "scenario": "Real-time camera filters (AR effects)",
            "latency": "<16ms (60 FPS)",
            "compute": "Mobile GPU / NPU",
            "justification": (
                "Must run at 60 FPS for smooth UX. "
                "No network dependency. "
                "Model must be <50MB for app size constraints."
            ),
        },
        {
            "level": "Microcontroller",
            "scenario": "Keyword spotting ('Hey Siri')",
            "latency": "<10ms",
            "compute": "ARM Cortex-M4 (256KB RAM)",
            "justification": (
                "Always-on listening requires ultra-low power (<1mW). "
                "Model must fit in <256KB RAM. "
                "Privacy: audio never leaves device."
            ),
        },
        {
            "level": "Sensor",
            "scenario": "Vibration anomaly detection",
            "latency": "<1ms",
            "compute": "DSP / FPGA on sensor board",
            "justification": (
                "Sampling at kHz rates — no time for data transfer. "
                "Simple model (decision tree / tiny NN). "
                "Deployed in remote locations with no connectivity."
            ),
        },
    ]

    print("  Edge Computing Spectrum (Cloud -> Far Edge):\n")
    for entry in spectrum:
        print(f"  [{entry['level']}] {entry['scenario']}")
        print(f"    Latency:  {entry['latency']}")
        print(f"    Compute:  {entry['compute']}")
        print(f"    Why here: {entry['justification']}")
        print()


# === Exercise 4: Deployment Decision Framework ===
# Problem: Given requirements, decide cloud vs edge deployment.

def exercise_4():
    """Apply a decision framework for cloud vs edge deployment."""
    requirements = [
        {
            "app": "Autonomous drone obstacle avoidance",
            "latency_req": "< 20ms",
            "connectivity": "Unreliable (field)",
            "privacy": "Medium",
            "model_size": "~5MB",
            "decision": "EDGE (onboard NPU)",
            "reasoning": (
                "Safety-critical: 20ms max latency rules out cloud. "
                "Unreliable connectivity makes cloud infeasible. "
                "5MB model fits on edge NPU easily."
            ),
        },
        {
            "app": "Medical image diagnosis (X-ray analysis)",
            "latency_req": "< 30 seconds",
            "connectivity": "Reliable (hospital)",
            "privacy": "High (HIPAA)",
            "model_size": "~2GB",
            "decision": "EDGE SERVER (on-premises GPU)",
            "reasoning": (
                "Latency is relaxed (not real-time). "
                "HIPAA compliance: patient data must not leave hospital. "
                "2GB model too large for mobile but fine for edge server."
            ),
        },
        {
            "app": "Social media content moderation",
            "latency_req": "< 5 seconds",
            "connectivity": "Reliable (datacenter)",
            "privacy": "Low (public posts)",
            "model_size": "~50GB (multimodal)",
            "decision": "CLOUD",
            "reasoning": (
                "Latency budget is generous. "
                "50GB model requires datacenter GPUs. "
                "Content is already public — no privacy constraint. "
                "Centralized updates ensure consistent policy."
            ),
        },
    ]

    print("  Deployment Decision Framework:\n")
    print(f"  {'Application':<40} {'Decision':<25} {'Key Factor'}")
    print("  " + "-" * 90)

    for req in requirements:
        key_factor = "latency" if "ms" in req["latency_req"] else (
            "privacy" if req["privacy"] == "High" else "model size"
        )
        print(f"  {req['app']:<40} {req['decision']:<25} {key_factor}")

    print("\n  Detailed analysis:")
    for req in requirements:
        print(f"\n  {req['app']}:")
        print(f"    Latency: {req['latency_req']}, "
              f"Connectivity: {req['connectivity']}")
        print(f"    Privacy: {req['privacy']}, Model: {req['model_size']}")
        print(f"    -> {req['decision']}")
        print(f"    Reasoning: {req['reasoning']}")


# === Exercise 5: Cost Analysis ===
# Problem: Compare TCO (Total Cost of Ownership) for cloud vs edge
# for a fleet of 100 smart cameras over 3 years.

def exercise_5():
    """Total Cost of Ownership: cloud vs edge for 100 cameras."""
    num_cameras = 100
    years = 3

    # Cloud costs (per camera per month)
    cloud = {
        "compute_per_cam_month": 50,     # GPU instance share
        "storage_per_cam_month": 10,     # Video storage
        "bandwidth_per_cam_month": 30,   # Data transfer
        "cloud_api_per_cam_month": 5,    # API calls
        "camera_hw": 200,               # Basic camera (one-time)
    }

    cloud_monthly = sum(v for k, v in cloud.items() if "month" in k)
    cloud_total = (cloud_monthly * num_cameras * years * 12 +
                   cloud["camera_hw"] * num_cameras)

    # Edge costs (per camera)
    edge = {
        "smart_camera_hw": 800,          # Camera + edge NPU (one-time)
        "power_per_cam_month": 5,        # Electricity
        "maintenance_per_cam_month": 10, # SW updates, monitoring
        "cloud_per_cam_month": 5,        # Minimal cloud (alerts only)
    }

    edge_monthly = sum(v for k, v in edge.items() if "month" in k)
    edge_total = (edge_monthly * num_cameras * years * 12 +
                  edge["smart_camera_hw"] * num_cameras)

    print(f"  TCO Comparison: {num_cameras} cameras over {years} years\n")

    print("  Cloud Deployment:")
    print(f"    Hardware (cameras):       ${cloud['camera_hw'] * num_cameras:>10,}")
    print(f"    Monthly costs/camera:     ${cloud_monthly:>10}/month")
    print(f"    Total monthly (fleet):    ${cloud_monthly * num_cameras:>10,}/month")
    print(f"    Total {years}-year TCO:          ${cloud_total:>10,}")

    print("\n  Edge Deployment:")
    print(f"    Hardware (smart cameras): ${edge['smart_camera_hw'] * num_cameras:>10,}")
    print(f"    Monthly costs/camera:     ${edge_monthly:>10}/month")
    print(f"    Total monthly (fleet):    ${edge_monthly * num_cameras:>10,}/month")
    print(f"    Total {years}-year TCO:          ${edge_total:>10,}")

    savings = cloud_total - edge_total
    savings_pct = (savings / cloud_total) * 100
    breakeven_months = (
        (edge["smart_camera_hw"] - cloud["camera_hw"]) * num_cameras /
        ((cloud_monthly - edge_monthly) * num_cameras)
    )

    print(f"\n  Edge saves: ${savings:,} ({savings_pct:.0f}%) over {years} years")
    print(f"  Break-even point: {breakeven_months:.0f} months")
    print(f"  After break-even, edge saves ${(cloud_monthly - edge_monthly) * num_cameras:,}/month")


if __name__ == "__main__":
    print("=== Exercise 1: Edge vs Cloud Latency Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Privacy and Bandwidth Tradeoff ===")
    exercise_2()
    print("\n=== Exercise 3: Edge Computing Spectrum ===")
    exercise_3()
    print("\n=== Exercise 4: Deployment Decision Framework ===")
    exercise_4()
    print("\n=== Exercise 5: Cost Analysis ===")
    exercise_5()
    print("\nAll exercises completed!")
