"""
17. Video Processing
- VideoCapture (camera, file)
- VideoWriter (video saving)
- Frame processing
- Background subtraction
- Optical Flow
"""

import cv2
import numpy as np


def video_capture_basics():
    """Video capture basics"""
    print("=" * 50)
    print("VideoCapture Basics")
    print("=" * 50)

    print("\nCamera capture:")
    print("  cap = cv2.VideoCapture(0)  # Default camera")
    print("  cap = cv2.VideoCapture(1)  # Second camera")

    print("\nFile capture:")
    print("  cap = cv2.VideoCapture('video.mp4')")
    print("  cap = cv2.VideoCapture('rtsp://...')  # Streaming")

    print("\nKey properties:")
    properties = [
        ('CAP_PROP_FRAME_WIDTH', 'Frame width'),
        ('CAP_PROP_FRAME_HEIGHT', 'Frame height'),
        ('CAP_PROP_FPS', 'Frames per second'),
        ('CAP_PROP_FRAME_COUNT', 'Total frame count'),
        ('CAP_PROP_POS_FRAMES', 'Current frame position'),
        ('CAP_PROP_POS_MSEC', 'Current time (ms)'),
    ]

    for prop, desc in properties:
        print(f"  cv2.{prop}: {desc}")

    # Generate simulation video
    print("\nGenerating simulation video...")
    frames = []
    for i in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = [100, 100, 100]
        cv2.circle(frame, (100 + i * 15, 240), 30, (0, 255, 0), -1)
        cv2.putText(frame, f'Frame {i}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)

    cv2.imwrite('video_frame_sample.jpg', frames[15])
    print("Sample frame saved: video_frame_sample.jpg")


def video_writer_demo():
    """Video writer demo"""
    print("\n" + "=" * 50)
    print("VideoWriter (Video Saving)")
    print("=" * 50)

    # Generate frames
    width, height = 640, 480
    fps = 30.0

    # Codec settings
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4

    print(f"Video settings: {width}x{height}, {fps}fps")

    # Create VideoWriter
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    if not out.isOpened():
        print("Cannot open VideoWriter.")
        return

    # Write frames
    for i in range(90):  # 3 seconds
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [50, 50, 50]

        # Moving circle
        x = int(100 + 5 * i)
        y = int(240 + 100 * np.sin(i * 0.1))
        cv2.circle(frame, (x, y), 40, (0, 200, 0), -1)

        # Frame number
        cv2.putText(frame, f'Frame: {i}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print("Video saved: output_video.mp4")

    print("\nSupported codecs:")
    codecs = [
        ("'XVID'", '.avi', 'MPEG-4'),
        ("'mp4v'", '.mp4', 'MPEG-4'),
        ("'avc1'", '.mp4', 'H.264 (macOS)'),
        ("'MJPG'", '.avi', 'Motion JPEG'),
    ]
    for code, ext, desc in codecs:
        print(f"  cv2.VideoWriter_fourcc(*{code}) -> {ext} ({desc})")


def frame_processing_demo():
    """Frame processing demo"""
    print("\n" + "=" * 50)
    print("Frame Processing")
    print("=" * 50)

    # Simulation frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = [150, 150, 150]
    cv2.rectangle(frame, (200, 150), (440, 330), (0, 100, 200), -1)
    cv2.circle(frame, (320, 240), 50, (200, 100, 0), -1)

    # Various processing
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # 3. Edge
    edges = cv2.Canny(gray, 50, 150)

    # 4. Color adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 50)  # Increase saturation
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    print("Frame processing examples:")
    print("  1. Grayscale conversion")
    print("  2. Gaussian blur")
    print("  3. Canny edge detection")
    print("  4. Color correction (HSV)")

    cv2.imwrite('frame_original.jpg', frame)
    cv2.imwrite('frame_gray.jpg', gray)
    cv2.imwrite('frame_blurred.jpg', blurred)
    cv2.imwrite('frame_edges.jpg', edges)
    cv2.imwrite('frame_enhanced.jpg', enhanced)

    print("\nReal-time processing template:")
    code = '''
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame processing
    processed = your_processing_function(frame)

    cv2.imshow('Video', processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''
    print(code)


def background_subtraction_demo():
    """Background subtraction demo"""
    print("\n" + "=" * 50)
    print("Background Subtraction")
    print("=" * 50)

    # Simulation frames (background + moving object)
    background = np.zeros((480, 640, 3), dtype=np.uint8)
    background[:] = [120, 120, 120]
    cv2.rectangle(background, (100, 300), (250, 400), (80, 80, 80), -1)  # Static object

    # Background subtractor creation
    # MOG2: Mixture of Gaussians
    bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    # KNN
    bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=True
    )

    print("Background subtractors:")
    print("  1. MOG2 (Mixture of Gaussians)")
    print("     - Effective for complex backgrounds")
    print("     - Shadow detection capable")
    print("")
    print("  2. KNN (K-Nearest Neighbors)")
    print("     - Effective for non-standard distributions")
    print("     - Robust to lighting changes")

    # Simulation (moving circle)
    for i in range(30):
        frame = background.copy()
        x = 100 + i * 15
        cv2.circle(frame, (x, 200), 40, (0, 200, 0), -1)

        # Apply background subtraction
        fg_mask_mog2 = bg_subtractor_mog2.apply(frame)
        fg_mask_knn = bg_subtractor_knn.apply(frame)

        if i == 15:  # Save middle frame
            cv2.imwrite('bg_frame.jpg', frame)
            cv2.imwrite('bg_mask_mog2.jpg', fg_mask_mog2)
            cv2.imwrite('bg_mask_knn.jpg', fg_mask_knn)

    print("\nParameters:")
    print("  history: Number of past frames for learning")
    print("  varThreshold (MOG2): Pixel-model distance threshold")
    print("  dist2Threshold (KNN): Distance threshold")
    print("  detectShadows: Whether to detect shadows")

    print("\nPost-processing:")
    print("  - Morphological operations (noise removal)")
    print("  - Contour detection (object extraction)")
    print("  - Bounding box drawing")


def optical_flow_demo():
    """Optical flow demo"""
    print("\n" + "=" * 50)
    print("Optical Flow")
    print("=" * 50)

    # Create two frames (motion simulation)
    frame1 = np.zeros((300, 400), dtype=np.uint8)
    frame1[:] = 100
    cv2.circle(frame1, (100, 150), 30, 200, -1)
    cv2.rectangle(frame1, (250, 100), (320, 200), 180, -1)

    frame2 = np.zeros((300, 400), dtype=np.uint8)
    frame2[:] = 100
    cv2.circle(frame2, (130, 150), 30, 200, -1)  # Moved right
    cv2.rectangle(frame2, (250, 130), (320, 230), 180, -1)  # Moved down

    # Lucas-Kanade (sparse)
    print("1. Lucas-Kanade (Sparse Optical Flow)")
    print("   - Tracks motion of specific points")
    print("   - Fast, feature-point-based")

    # Feature point detection
    p0 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.3,
                                  minDistance=7, blockSize=7)

    if p0 is not None:
        # Compute optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, p0, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Select good points only
        good_old = p0[status == 1]
        good_new = p1[status == 1]

        # Visualization
        result_lk = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        for old, new in zip(good_old, good_new):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            cv2.line(result_lk, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(result_lk, (a, b), 5, (0, 0, 255), -1)

        cv2.imwrite('optflow_lk.jpg', result_lk)
        print(f"   Tracked points: {len(good_new)}")

    # Farneback (dense)
    print("\n2. Farneback (Dense Optical Flow)")
    print("   - Computes motion of every pixel")
    print("   - Slow, captures overall motion")

    flow = cv2.calcOpticalFlowFarneback(
        frame1, frame2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Visualize flow with color
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    result_fb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite('optflow_frame1.jpg', frame1)
    cv2.imwrite('optflow_frame2.jpg', frame2)
    cv2.imwrite('optflow_farneback.jpg', result_fb)

    print(f"   Flow vector shape: {flow.shape}")

    print("\nOptical flow applications:")
    print("  - Motion detection")
    print("  - Object tracking")
    print("  - Video compression")
    print("  - Action recognition")


def video_tracking_demo():
    """Object tracking demo"""
    print("\n" + "=" * 50)
    print("Object Tracking")
    print("=" * 50)

    print("OpenCV tracker types:")
    trackers = [
        ('BOOSTING', 'Legacy method, slow'),
        ('MIL', 'Multiple Instance Learning'),
        ('KCF', 'Kernelized Correlation Filters, fast'),
        ('TLD', 'Tracking-Learning-Detection'),
        ('MEDIANFLOW', 'Good for predictable motion'),
        ('GOTURN', 'Deep Learning-based'),
        ('MOSSE', 'Very fast'),
        ('CSRT', 'Accurate, somewhat slow'),
    ]

    for name, desc in trackers:
        print(f"  cv2.Tracker{name}_create(): {desc}")

    print("\nTracker usage template:")
    code = '''
# Create tracker
tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()  # More accurate

# Set initial bounding box
bbox = (x, y, w, h)  # or cv2.selectROI()
tracker.init(first_frame, bbox)

# Tracking loop
while True:
    ret, frame = cap.read()
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
'''
    print(code)


def performance_tips():
    """Performance optimization tips"""
    print("\n" + "=" * 50)
    print("Video Processing Performance Optimization")
    print("=" * 50)

    print("""
1. Reduce frame size
   - Before processing: frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
   - Map results back to original after detection

2. Frame skipping
   - Not every frame needs processing
   - Process only when frame_count % skip_frames == 0

3. Use ROI (Region of Interest)
   - Process only the region of interest instead of full frame
   - roi = frame[y:y+h, x:x+w]

4. Multithreading/Multiprocessing
   - Separate capture and processing
   - Pass frames via Queue

5. GPU acceleration (CUDA)
   - cv2.cuda.GpuMat()
   - Use cv2.cuda module functions

6. Capture buffer settings
   - cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
   - Minimize latency
""")


def main():
    """Main function"""
    # VideoCapture basics
    video_capture_basics()

    # VideoWriter
    video_writer_demo()

    # Frame processing
    frame_processing_demo()

    # Background subtraction
    background_subtraction_demo()

    # Optical flow
    optical_flow_demo()

    # Object tracking
    video_tracking_demo()

    # Performance tips
    performance_tips()

    print("\nVideo processing demo complete!")


if __name__ == '__main__':
    main()
