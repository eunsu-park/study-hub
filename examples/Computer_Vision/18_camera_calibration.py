"""
18. Camera Calibration
- Camera intrinsic parameters
- Distortion correction
- Chessboard detection
- Stereo vision basics
"""

import cv2
import numpy as np


def camera_model_concept():
    """Camera model concept"""
    print("=" * 50)
    print("Camera Model Concept")
    print("=" * 50)

    print("\n1. Pinhole Camera Model")
    print("   - 3D point -> 2D image projection")
    print("   - Perspective Projection")

    print("\n2. Camera Intrinsic Parameters")
    print("""
   K = | fx  0  cx |
       |  0 fy  cy |
       |  0  0   1 |

   fx, fy: Focal length (in pixels)
   cx, cy: Principal point (image center)
""")

    print("3. Camera Extrinsic Parameters")
    print("   - R: Rotation matrix (3x3)")
    print("   - t: Translation vector (3x1)")
    print("   - World coordinates -> Camera coordinates transform")

    print("\n4. Projection Matrix")
    print("   P = K[R|t]")
    print("   p = P * X  (homogeneous coordinates)")

    print("\n5. Distortion Coefficients")
    print("   - k1, k2, k3: Radial distortion")
    print("   - p1, p2: Tangential distortion")
    print("   - dist_coeffs = [k1, k2, p1, p2, k3]")


def create_chessboard_image():
    """Create chessboard image"""
    # Chessboard pattern
    rows, cols = 7, 9
    square_size = 40

    img = np.zeros((rows * square_size, cols * square_size, 3), dtype=np.uint8)
    img[:] = [255, 255, 255]

    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 1:
                x1, y1 = j * square_size, i * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    return img


def chessboard_detection_demo():
    """Chessboard corner detection demo"""
    print("\n" + "=" * 50)
    print("Chessboard Corner Detection")
    print("=" * 50)

    # Generate chessboard image
    chessboard = create_chessboard_image()

    # Apply slight perspective transform (simulating real capture)
    h, w = chessboard.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[20, 30], [w-30, 20], [w-20, h-10], [10, h-30]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(chessboard, M, (w, h), borderValue=(200, 200, 200))

    # Convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Corner detection parameters
    # Number of internal corners (black-white intersections)
    pattern_size = (8, 6)  # 8 horizontal, 6 vertical corners

    # Corner detection
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"Pattern size: {pattern_size}")
    print(f"Corner detection success: {found}")

    if found:
        # Refine to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        print(f"Number of corners detected: {len(corners)}")

        # Visualize result
        result = warped.copy()
        cv2.drawChessboardCorners(result, pattern_size, corners, found)

        cv2.imwrite('chessboard_input.jpg', warped)
        cv2.imwrite('chessboard_corners.jpg', result)
        print("Images saved")

    print("\nDetection flags:")
    print("  CALIB_CB_ADAPTIVE_THRESH: Adaptive binarization")
    print("  CALIB_CB_NORMALIZE_IMAGE: Image normalization")
    print("  CALIB_CB_FAST_CHECK: Fast check (early exit on failure)")


def camera_calibration_simulation():
    """Camera calibration simulation"""
    print("\n" + "=" * 50)
    print("Camera Calibration Simulation")
    print("=" * 50)

    # Chessboard parameters
    pattern_size = (8, 6)
    square_size = 1.0  # Actual square size (units: cm, mm, etc.)

    # 3D object points (chessboard plane, z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    print(f"Object points (3D): {objp.shape}")
    print(f"First point: {objp[0]}")
    print(f"Last point: {objp[-1]}")

    # Points detected from multiple images for simulation
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    # Simulation (in practice, use images captured from multiple angles)
    chessboard = create_chessboard_image()
    gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"\nNumber of images used: {len(objpoints)}")

        # Calibration (minimum 3-5 images required)
        # This is a simulation so results may differ from actual
        h, w = gray.shape

        # Initial camera matrix estimate
        fx = fy = w  # Approximate focal length
        cx, cy = w/2, h/2

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros(5)

        print("\nEstimated camera matrix:")
        print(camera_matrix)

        print("\nCalibration process:")
        print("  1. Capture chessboard from multiple angles (10-20 images)")
        print("  2. Detect corners in each image")
        print("  3. Call cv2.calibrateCamera()")
        print("  4. Obtain camera matrix and distortion coefficients")


def calibration_workflow():
    """Calibration workflow"""
    print("\n" + "=" * 50)
    print("Actual Calibration Workflow")
    print("=" * 50)

    code = '''
import cv2
import numpy as np
import glob

# Chessboard setup
pattern_size = (9, 6)  # Number of internal corners
square_size = 25.0     # In mm

# Generate object points
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D
imgpoints = []  # 2D

# Load images and detect corners
images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

# Perform calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"RMS error: {ret}")
print(f"Camera matrix:\\n{camera_matrix}")
print(f"Distortion coefficients:\\n{dist_coeffs}")

# Save results
np.savez('calibration.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs)
'''

    print(code)

    print("\nCalibration tips:")
    print("  1. Use at least 10-20 images")
    print("  2. Capture from various angles and positions")
    print("  3. Ensure the chessboard is distributed across the entire image")
    print("  4. Lighting should be uniform")
    print("  5. Sharp images without blur")


def undistort_demo():
    """Undistortion demo"""
    print("\n" + "=" * 50)
    print("Undistortion")
    print("=" * 50)

    # Create distorted image for simulation
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = [200, 200, 200]

    # Grid pattern
    for i in range(0, 600, 50):
        cv2.line(img, (i, 0), (i, 400), (0, 0, 0), 1)
    for j in range(0, 400, 50):
        cv2.line(img, (0, j), (600, j), (0, 0, 0), 1)

    # Apply virtual distortion (barrel distortion simulation)
    h, w = img.shape[:2]
    camera_matrix = np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], dtype=np.float64)

    # Distortion coefficients (negative k1 = barrel, positive k1 = pincushion)
    dist_coeffs = np.array([-0.3, 0.1, 0, 0, 0])

    # Apply distortion (using undistort in reverse)
    distorted = cv2.undistort(img, camera_matrix, -dist_coeffs)

    # Correct distortion
    undistorted = cv2.undistort(distorted, camera_matrix, dist_coeffs)

    cv2.imwrite('undistort_original.jpg', img)
    cv2.imwrite('undistort_distorted.jpg', distorted)
    cv2.imwrite('undistort_corrected.jpg', undistorted)

    print("Undistortion methods:")
    print("  1. cv2.undistort()")
    print("     - Simple to use")
    print("     - Computes every time")

    print("\n  2. cv2.initUndistortRectifyMap() + cv2.remap()")
    print("     - Pre-computes the map")
    print("     - Efficient for video")

    code = '''
# Efficient method (for video)
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None,
    camera_matrix, (w, h), cv2.CV_32FC1
)

# Apply per frame
undistorted = cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR)
'''
    print(code)


def stereo_vision_concept():
    """Stereo vision concept"""
    print("\n" + "=" * 50)
    print("Stereo Vision Basics")
    print("=" * 50)

    print("\n1. Stereo Vision Principle")
    print("   - Capture the same scene with two cameras")
    print("   - Compute depth from disparity")
    print("   - depth = (baseline * focal_length) / disparity")

    print("\n2. Stereo Calibration")
    print("   - Calibrate each camera individually")
    print("   - Calibrate the stereo pair")
    print("   - Compute epipolar geometry")

    code = '''
# Stereo calibration
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, D1, K2, D2, image_size,
    flags=cv2.CALIB_FIX_INTRINSIC
)
'''
    print(code)

    print("\n3. Stereo Rectification")
    print("   - Align both images to the same plane")
    print("   - Search for matches only along horizontal lines")

    print("\n4. Disparity Map Computation")
    print("   - StereoBM: Block Matching (fast)")
    print("   - StereoSGBM: Semi-Global BM (accurate)")


def stereo_matching_demo():
    """Stereo matching simulation"""
    print("\n" + "=" * 50)
    print("Stereo Matching Simulation")
    print("=" * 50)

    # Create stereo image pair for simulation
    left = np.zeros((300, 400), dtype=np.uint8)
    left[:] = 150
    cv2.rectangle(left, (100, 100), (200, 200), 80, -1)  # Close object
    cv2.rectangle(left, (250, 120), (350, 180), 100, -1)  # Far object

    # Right image (with disparity applied)
    right = np.zeros((300, 400), dtype=np.uint8)
    right[:] = 150
    cv2.rectangle(right, (80, 100), (180, 200), 80, -1)   # Disparity 20 (close)
    cv2.rectangle(right, (240, 120), (340, 180), 100, -1)  # Disparity 10 (far)

    # StereoBM
    stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity_bm = stereo_bm.compute(left, right)

    # StereoSGBM
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity_sgbm = stereo_sgbm.compute(left, right)

    # Normalize
    disparity_bm_norm = cv2.normalize(disparity_bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_sgbm_norm = cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite('stereo_left.jpg', left)
    cv2.imwrite('stereo_right.jpg', right)
    cv2.imwrite('stereo_disparity_bm.jpg', disparity_bm_norm)
    cv2.imwrite('stereo_disparity_sgbm.jpg', disparity_sgbm_norm)

    print("Disparity map generation complete")
    print("\nStereoBM parameters:")
    print("  numDisparities: Disparity range (multiple of 16)")
    print("  blockSize: Matching block size (odd, 5~21)")

    print("\nStereoSGBM parameters:")
    print("  P1, P2: Smoothness control")
    print("  uniquenessRatio: Matching uniqueness")
    print("  speckleWindowSize: Speckle filter size")


def pose_estimation_concept():
    """Pose estimation concept"""
    print("\n" + "=" * 50)
    print("Pose Estimation")
    print("=" * 50)

    print("\nCamera pose estimation:")
    print("  - Estimate camera position/orientation from 3D-2D correspondences")
    print("  - cv2.solvePnP()")

    code = '''
# 3D object points (known world coordinates)
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
], dtype=np.float32)

# 2D image points (detected coordinates)
image_points = np.array([...], dtype=np.float32)

# Pose estimation
success, rvec, tvec = cv2.solvePnP(
    object_points, image_points,
    camera_matrix, dist_coeffs
)

# Rotation vector -> Rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvec)

# Draw 3D axes
axis_points = np.float32([
    [3, 0, 0], [0, 3, 0], [0, 0, -3]
]).reshape(-1, 3)
imgpts, _ = cv2.projectPoints(
    axis_points, rvec, tvec, camera_matrix, dist_coeffs
)
'''
    print(code)

    print("\nApplications:")
    print("  - AR (Augmented Reality)")
    print("  - Robot vision")
    print("  - 3D reconstruction")


def main():
    """Main function"""
    # Camera model concept
    camera_model_concept()

    # Chessboard detection
    chessboard_detection_demo()

    # Calibration simulation
    camera_calibration_simulation()

    # Calibration workflow
    calibration_workflow()

    # Undistortion
    undistort_demo()

    # Stereo vision
    stereo_vision_concept()

    # Stereo matching
    stereo_matching_demo()

    # Pose estimation
    pose_estimation_concept()

    print("\nCamera calibration demo complete!")


if __name__ == '__main__':
    main()
