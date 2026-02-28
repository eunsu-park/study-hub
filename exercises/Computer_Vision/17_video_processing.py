"""
Exercise Solutions for Lesson 17: Video Processing
Computer Vision - Background Subtraction, Optical Flow, Object Tracking

Topics covered:
- Video player simulation (frame navigation)
- Motion heatmap via background subtraction
- Speed measurement using optical flow
- Vehicle counter with counting line
- Gesture recognition (trajectory analysis)
"""

import numpy as np


# =============================================================================
# Helper: Synthetic video generation
# =============================================================================

def generate_synthetic_video(n_frames=30, h=120, w=160):
    """
    Generate synthetic video frames with a moving bright blob.
    Returns list of grayscale frames and the trajectory used.
    """
    frames = []
    trajectory = []
    np.random.seed(42)

    cx, cy = 30, 60  # Start position
    vx, vy = 3, 1    # Velocity

    for t in range(n_frames):
        frame = np.random.randint(40, 70, (h, w), dtype=np.uint8)

        # Moving object (bright blob)
        x = int(cx + vx * t)
        y = int(cy + vy * t)
        x = x % w
        y = y % h
        trajectory.append((x, y))

        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - x)**2 + (yy - y)**2) <= 15**2
        frame[mask] = np.random.randint(170, 210)

        frames.append(frame)

    return frames, trajectory


# =============================================================================
# Exercise 1: Video Player (Simulated)
# =============================================================================

def exercise_1_video_player():
    """
    Simulate a basic video player with frame navigation.

    Features:
    - Play/pause state
    - Forward/backward skip
    - Frame-by-frame navigation
    - Current time / total time display
    - Progress bar

    Returns:
        playback log (list of events)
    """
    n_frames = 60
    fps = 30.0
    total_time = n_frames / fps

    class VideoPlayer:
        def __init__(self, n_frames, fps):
            self.n_frames = n_frames
            self.fps = fps
            self.current_frame = 0
            self.paused = False
            self.log = []

        def play_pause(self):
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "PLAYING"
            self.log.append(f"  [{self.current_frame:>3}] {state}")

        def skip(self, n):
            """Skip forward (positive) or backward (negative) by n frames."""
            old = self.current_frame
            self.current_frame = max(0, min(self.n_frames - 1,
                                            self.current_frame + n))
            direction = "FWD" if n > 0 else "BWD"
            self.log.append(
                f"  [{old:>3}] SKIP {direction} {abs(n)} -> "
                f"frame {self.current_frame}")

        def step(self, direction=1):
            """Frame-by-frame step (1=forward, -1=backward)."""
            old = self.current_frame
            self.current_frame = max(0, min(self.n_frames - 1,
                                            self.current_frame + direction))
            d = ">" if direction > 0 else "<"
            self.log.append(
                f"  [{old:>3}] STEP {d} frame {self.current_frame}")

        def get_time_str(self):
            curr_sec = self.current_frame / self.fps
            total_sec = self.n_frames / self.fps
            return f"{curr_sec:.1f}s / {total_sec:.1f}s"

        def get_progress_bar(self, width=40):
            ratio = self.current_frame / max(1, self.n_frames - 1)
            filled = int(ratio * width)
            bar = "#" * filled + "-" * (width - filled)
            return f"[{bar}] {ratio*100:.0f}%"

        def advance(self):
            """Advance by one frame (auto-play)."""
            if not self.paused and self.current_frame < self.n_frames - 1:
                self.current_frame += 1

    player = VideoPlayer(n_frames, fps)

    # Simulate user actions
    actions = [
        ("play", None),
        ("advance", 10),    # Play for 10 frames
        ("pause", None),
        ("step_fwd", None),
        ("step_fwd", None),
        ("skip_fwd", 15),
        ("play", None),
        ("advance", 5),
        ("skip_bwd", 20),
        ("step_bwd", None),
    ]

    print("Video Player Simulation")
    print(f"  Total: {n_frames} frames, {fps:.0f} fps, {total_time:.1f}s")
    print("=" * 60)

    for action, param in actions:
        if action == "play":
            player.play_pause()
        elif action == "advance" and param:
            for _ in range(param):
                player.advance()
            player.log.append(
                f"  [{player.current_frame:>3}] "
                f"(advanced {param} frames)")
        elif action == "step_fwd":
            player.step(1)
        elif action == "step_bwd":
            player.step(-1)
        elif action == "skip_fwd":
            player.skip(param)
        elif action == "skip_bwd":
            player.skip(-param)
        elif action == "pause":
            player.play_pause()

    for entry in player.log:
        print(entry)

    print(f"\n  Time: {player.get_time_str()}")
    print(f"  {player.get_progress_bar()}")

    return player.log


# =============================================================================
# Exercise 2: Motion Heatmap
# =============================================================================

def exercise_2_motion_heatmap():
    """
    Visualize motion accumulation as a heatmap using background subtraction.

    Pipeline:
    1. Maintain running average background
    2. Compute foreground mask per frame
    3. Accumulate motion map
    4. Normalize and apply pseudo-colormap

    Returns:
        (heatmap, accumulator)
    """
    frames, trajectory = generate_synthetic_video(n_frames=40, h=100, w=150)
    h, w = frames[0].shape

    # Running average background model
    bg = frames[0].astype(np.float64)
    alpha = 0.05  # Learning rate
    accumulator = np.zeros((h, w), dtype=np.float64)

    print("Motion Heatmap Generation")
    print(f"  Frames: {len(frames)}, Size: {w}x{h}")
    print(f"  Background learning rate: {alpha}")

    for i, frame in enumerate(frames):
        frame_f = frame.astype(np.float64)

        # Foreground detection
        diff = np.abs(frame_f - bg)
        fg_mask = (diff > 30).astype(np.float64)

        # Accumulate
        accumulator += fg_mask

        # Update background
        bg = alpha * frame_f + (1 - alpha) * bg

    # Normalize accumulator to [0, 255]
    if accumulator.max() > 0:
        norm_acc = (accumulator / accumulator.max() * 255).astype(np.uint8)
    else:
        norm_acc = np.zeros((h, w), dtype=np.uint8)

    # Create pseudo-colormap (JET-like: blue -> green -> red)
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            val = norm_acc[i, j] / 255.0
            if val < 0.25:
                # Blue
                heatmap[i, j] = [int(255 * val * 4), 0, 0]  # BGR: B channel
            elif val < 0.5:
                t = (val - 0.25) * 4
                heatmap[i, j] = [255, int(255 * t), 0]
            elif val < 0.75:
                t = (val - 0.5) * 4
                heatmap[i, j] = [int(255 * (1 - t)), 255, 0]
            else:
                t = (val - 0.75) * 4
                heatmap[i, j] = [0, int(255 * (1 - t)), int(255 * t)]

    # Statistics
    motion_pixels = np.sum(accumulator > 0)
    motion_pct = 100.0 * motion_pixels / (h * w)
    print(f"\n  Motion pixels: {motion_pixels} ({motion_pct:.1f}%)")
    print(f"  Max accumulation: {accumulator.max():.0f}")
    print(f"  Heatmap shape: {heatmap.shape}")

    # Show trajectory overlap
    traj_overlap = 0
    for x, y in trajectory:
        if 0 <= y < h and 0 <= x < w and accumulator[y, x] > 0:
            traj_overlap += 1
    print(f"  Trajectory overlap with heatmap: "
          f"{traj_overlap}/{len(trajectory)}")

    return heatmap, accumulator


# =============================================================================
# Exercise 3: Speed Measurement
# =============================================================================

def exercise_3_speed_measurement():
    """
    Measure object movement speed using optical flow (simplified).

    Computes frame-to-frame displacement and converts to speed
    using pixel-to-physical calibration.

    Returns:
        list of (frame, speed_px, speed_real) tuples
    """
    frames, trajectory = generate_synthetic_video(n_frames=25, h=100, w=150)

    # Calibration: 1 pixel = 0.5 cm, 30 fps
    pixels_to_cm = 0.5
    fps = 30.0

    # Simplified "optical flow": compute centroid displacement
    # In real implementation, this would use Lucas-Kanade or Farneback

    def find_centroid(frame, threshold=140):
        """Find centroid of bright region (simulates tracked object)."""
        mask = frame > threshold
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return (np.mean(xs), np.mean(ys))

    speeds = []
    prev_centroid = None

    print("Speed Measurement (Optical Flow Simulation)")
    print(f"  Calibration: 1px = {pixels_to_cm}cm, fps={fps:.0f}")
    print(f"  ROI: full frame")
    print(f"\n{'Frame':>6} | {'dX':>6} | {'dY':>6} | "
          f"{'px/f':>8} | {'cm/s':>8}")
    print("-" * 48)

    for i, frame in enumerate(frames):
        centroid = find_centroid(frame)

        if centroid is not None and prev_centroid is not None:
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            speed_px = np.sqrt(dx**2 + dy**2)  # pixels/frame
            speed_real = speed_px * pixels_to_cm * fps  # cm/s

            speeds.append((i, speed_px, speed_real))
            print(f"{i:>6} | {dx:>6.1f} | {dy:>6.1f} | "
                  f"{speed_px:>8.2f} | {speed_real:>8.1f}")

        prev_centroid = centroid

    if speeds:
        avg_speed_px = np.mean([s[1] for s in speeds])
        avg_speed_real = np.mean([s[2] for s in speeds])
        print(f"\n  Average speed: {avg_speed_px:.2f} px/frame "
              f"= {avg_speed_real:.1f} cm/s")

    return speeds


# =============================================================================
# Exercise 4: Vehicle Counter
# =============================================================================

def exercise_4_vehicle_counter():
    """
    Count objects crossing a virtual counting line.

    Simulates vehicles moving up/down through a counting zone.
    Detects direction of crossing and maintains counts.

    Returns:
        dict with count_up, count_down, total, events
    """
    h, w = 120, 160
    line_y = h // 2  # Counting line at vertical midpoint
    n_frames = 40

    # Simulate objects with known trajectories
    np.random.seed(42)
    objects = [
        # (start_x, start_y, vx, vy, start_frame, label)
        (30, 10, 0, 3, 0, "Vehicle_A"),     # Moving down
        (80, 110, 0, -3, 0, "Vehicle_B"),    # Moving up
        (120, 20, -1, 2, 5, "Vehicle_C"),    # Moving down
        (50, 100, 1, -2, 10, "Vehicle_D"),   # Moving up
        (100, 30, 0, 2, 15, "Vehicle_E"),    # Moving down
    ]

    # Track previous y positions
    prev_positions = {}
    count_down = 0
    count_up = 0
    events = []

    print("Vehicle Counter Simulation")
    print(f"  Frame size: {w}x{h}")
    print(f"  Counting line: y={line_y}")
    print(f"  Objects: {len(objects)}")
    print("=" * 60)

    for frame_idx in range(n_frames):
        # Create frame (background)
        frame = np.random.randint(40, 60, (h, w), dtype=np.uint8)

        # Draw counting line region
        frame[line_y-1:line_y+2, :] = 128

        # Update each object
        for start_x, start_y, vx, vy, start_frame, label in objects:
            if frame_idx < start_frame:
                continue

            t = frame_idx - start_frame
            x = int(start_x + vx * t)
            y = int(start_y + vy * t)

            if not (0 <= x < w and 0 <= y < h):
                continue

            # Draw object on frame
            y1 = max(0, y - 8)
            y2 = min(h, y + 8)
            x1 = max(0, x - 10)
            x2 = min(w, x + 10)
            frame[y1:y2, x1:x2] = 180

            # Check line crossing
            if label in prev_positions:
                prev_y = prev_positions[label]
                if prev_y < line_y <= y:
                    count_down += 1
                    events.append((frame_idx, label, "DOWN"))
                    print(f"  Frame {frame_idx:>3}: {label} crossed DOWN "
                          f"(y: {prev_y} -> {y})")
                elif prev_y > line_y >= y:
                    count_up += 1
                    events.append((frame_idx, label, "UP"))
                    print(f"  Frame {frame_idx:>3}: {label} crossed UP "
                          f"(y: {prev_y} -> {y})")

            prev_positions[label] = y

    result = {
        'count_up': count_up,
        'count_down': count_down,
        'total': count_up + count_down,
        'events': events,
    }

    print(f"\n  Summary:")
    print(f"    Down (entry): {count_down}")
    print(f"    Up (exit):    {count_up}")
    print(f"    Total:        {count_up + count_down}")

    return result


# =============================================================================
# Exercise 5: Gesture Recognition
# =============================================================================

def exercise_5_gesture_recognition():
    """
    Analyze movement trajectories to recognize simple gestures.

    Recognizes:
    - Wave (horizontal oscillation)
    - Circle (closed loop trajectory)
    - Swipe (single directional sweep)
    - None (unrecognized)

    Returns:
        list of (gesture_name, trajectory, confidence) tuples
    """

    def classify_gesture(trajectory):
        """
        Classify a trajectory based on geometric properties.
        trajectory: list of (x, y) points
        """
        if len(trajectory) < 5:
            return "None", 0.0

        pts = np.array(trajectory, dtype=np.float64)
        n = len(pts)

        # Compute displacement vectors
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])

        # Feature 1: Total displacement vs path length
        total_disp = np.sqrt((pts[-1, 0] - pts[0, 0])**2 +
                             (pts[-1, 1] - pts[0, 1])**2)
        path_length = np.sum(np.sqrt(dx**2 + dy**2))

        # Feature 2: Number of direction changes in x
        x_sign_changes = np.sum(np.abs(np.diff(np.sign(dx))) > 0)

        # Feature 3: Closedness (distance between start and end)
        closedness = total_disp / max(path_length, 1e-6)

        # Feature 4: Bounding box aspect ratio
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()

        # Feature 5: Area swept (shoelace formula)
        area = 0.5 * np.abs(
            np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        )

        # Classification rules
        if closedness < 0.2 and area > 100 and x_range > 10 and y_range > 10:
            # Closed loop with significant area = Circle
            return "Circle", min(1.0, area / 500)
        elif x_sign_changes >= 3 and x_range > y_range * 1.5:
            # Multiple horizontal oscillations = Wave
            conf = min(1.0, x_sign_changes / 6.0)
            return "Wave", conf
        elif closedness > 0.6 and path_length > 20:
            # High displacement ratio = Swipe
            angle = np.degrees(np.arctan2(
                pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0]))
            direction = ""
            if -45 <= angle <= 45:
                direction = "Right"
            elif 45 < angle <= 135:
                direction = "Down"
            elif -135 <= angle < -45:
                direction = "Up"
            else:
                direction = "Left"
            return f"Swipe_{direction}", min(1.0, closedness)
        else:
            return "None", 0.0

    # Test gestures
    np.random.seed(42)

    test_gestures = []

    # Wave gesture: horizontal oscillation
    wave = [(20 + 30 * np.sin(t * np.pi / 5), 50 + np.random.randn() * 2)
            for t in range(20)]
    test_gestures.append(("Wave Input", wave))

    # Circle gesture
    circle = [(50 + 25 * np.cos(t * 2 * np.pi / 20),
               50 + 25 * np.sin(t * 2 * np.pi / 20))
              for t in range(21)]
    test_gestures.append(("Circle Input", circle))

    # Swipe right
    swipe_r = [(10 + 5 * t + np.random.randn(),
                50 + np.random.randn() * 2)
               for t in range(15)]
    test_gestures.append(("Swipe Right Input", swipe_r))

    # Swipe up
    swipe_u = [(50 + np.random.randn() * 2,
                80 - 4 * t + np.random.randn())
               for t in range(15)]
    test_gestures.append(("Swipe Up Input", swipe_u))

    # Random noise (no gesture)
    noise = [(50 + np.random.randn() * 5, 50 + np.random.randn() * 5)
             for t in range(10)]
    test_gestures.append(("Random Noise", noise))

    results = []

    print("Gesture Recognition")
    print(f"{'Input':>20} | {'Gesture':>15} | {'Confidence':>10} | "
          f"{'Points':>7}")
    print("-" * 62)

    for name, traj in test_gestures:
        gesture, conf = classify_gesture(traj)
        results.append((gesture, traj, conf))
        print(f"{name:>20} | {gesture:>15} | {conf:>10.2f} | "
              f"{len(traj):>7}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n>>> Exercise 1: Video Player")
    exercise_1_video_player()

    print("\n>>> Exercise 2: Motion Heatmap")
    exercise_2_motion_heatmap()

    print("\n>>> Exercise 3: Speed Measurement")
    exercise_3_speed_measurement()

    print("\n>>> Exercise 4: Vehicle Counter")
    exercise_4_vehicle_counter()

    print("\n>>> Exercise 5: Gesture Recognition")
    exercise_5_gesture_recognition()

    print("\nAll exercises completed successfully.")
