"""
14. Feature Matching
- BFMatcher (Brute Force)
- FLANN matcher
- Good match selection (ratio test)
- Homography computation
"""

import cv2
import numpy as np


def create_test_images():
    """Create test image pairs for matching"""
    # Original image
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img1[:] = [200, 200, 200]

    # Patterns with features
    cv2.rectangle(img1, (50, 50), (150, 150), (50, 50, 50), -1)
    cv2.circle(img1, (250, 100), 40, (100, 100, 100), -1)
    cv2.rectangle(img1, (300, 150), (380, 250), (80, 80, 80), -1)

    # Add checkerboard pattern
    for i in range(3):
        for j in range(3):
            x, y = 100 + i * 30, 180 + j * 30
            if (i + j) % 2 == 0:
                cv2.rectangle(img1, (x, y), (x + 30, y + 30), (0, 0, 0), -1)

    cv2.putText(img1, 'MATCH', (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Transformed image (rotation + scale)
    h, w = img1.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 0.9)  # 15 degree rotation, 0.9x scale
    img2 = cv2.warpAffine(img1, M, (w, h), borderValue=(200, 200, 200))

    return img1, img2


def bf_matcher_demo():
    """Brute Force matcher demo"""
    print("=" * 50)
    print("BFMatcher (Brute Force Matcher)")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB feature detection
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # BFMatcher creation
    # NORM_HAMMING: For binary descriptors (ORB, BRIEF)
    # NORM_L2: For float descriptors (SIFT, SURF)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Matching
    matches = bf.match(des1, des2)

    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")
    print(f"Number of matches: {len(matches)}")

    # Top match info
    print("\nTop 5 matches:")
    for i, m in enumerate(matches[:5]):
        print(f"  {i}: queryIdx={m.queryIdx}, trainIdx={m.trainIdx}, distance={m.distance:.1f}")

    # Draw results
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nBFMatcher properties:")
    print("  - Compares all pairs (O(n*m))")
    print("  - crossCheck=True: Selects only mutual nearest neighbors")
    print("  - Accurate but slow")

    cv2.imwrite('bf_match_img1.jpg', img1)
    cv2.imwrite('bf_match_img2.jpg', img2)
    cv2.imwrite('bf_match_result.jpg', result)


def knn_match_demo():
    """KNN matching and Ratio Test demo"""
    print("\n" + "=" * 50)
    print("KNN Matching + Ratio Test")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # BFMatcher (crossCheck=False for knnMatch)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN matching (k=2)
    matches = bf.knnMatch(des1, des2, k=2)

    print(f"KNN matches: {len(matches)}")

    # Lowe's Ratio Test
    # nearest distance / second nearest distance < threshold
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"After Ratio Test: {len(good_matches)}")

    # Draw results
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nRatio Test (Lowe's ratio):")
    print("  - nearest / second nearest < 0.75 (typically 0.7~0.8)")
    print("  - Removes ambiguous matches")
    print("  - Reduces false matches")

    cv2.imwrite('knn_match_result.jpg', result)

    return kp1, kp2, good_matches


def flann_matcher_demo():
    """FLANN matcher demo"""
    print("\n" + "=" * 50)
    print("FLANN Matcher")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        # Use SIFT (float descriptors)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # FLANN parameters (for SIFT/SURF)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

    except AttributeError:
        # Use ORB if SIFT unavailable
        print("SIFT unavailable, using ORB")
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # FLANN parameters (for ORB)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)

    # FLANN matcher creation
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN matching
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio Test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print(f"FLANN matches: {len(matches)} -> Good matches: {len(good_matches)}")

    # Draw results
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    print("\nFLANN properties:")
    print("  - Approximate nearest neighbor search")
    print("  - Efficient for large datasets")
    print("  - KD-Tree (SIFT) or LSH (ORB)")

    cv2.imwrite('flann_match_result.jpg', result)


def homography_demo():
    """Homography computation demo"""
    print("\n" + "=" * 50)
    print("Homography")
    print("=" * 50)

    img1, img2 = create_test_images()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Number of good matches: {len(good_matches)}")

    if len(good_matches) >= 4:
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography (RANSAC)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        inliers = sum(matches_mask)
        print(f"Inliers: {inliers}/{len(good_matches)}")

        if H is not None:
            print(f"\nHomography matrix:\n{H}")

            # Project img1 boundary onto img2
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)

            # Draw boundary on img2
            result = img2.copy()
            dst = np.int32(dst)
            cv2.polylines(result, [dst], True, (0, 255, 0), 3)

            cv2.imwrite('homography_result.jpg', result)

            # Match visualization (inliers only)
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matches_mask,
                flags=2
            )
            match_result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            cv2.imwrite('homography_matches.jpg', match_result)

    print("\nHomography applications:")
    print("  - Image registration")
    print("  - Panorama stitching")
    print("  - Object recognition (pose estimation)")
    print("  - Augmented reality")


def match_object_demo():
    """Object matching demo"""
    print("\n" + "=" * 50)
    print("Object Matching Practice")
    print("=" * 50)

    # Template image (object to find)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    template[:] = [200, 200, 200]
    cv2.rectangle(template, (10, 10), (90, 90), (50, 50, 50), -1)
    cv2.circle(template, (50, 50), 20, (100, 100, 100), -1)

    # Scene image (containing the object)
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    scene[:] = [180, 180, 180]

    # Place template in scene (with rotation and scale)
    h, w = template.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 30, 0.8)
    rotated_template = cv2.warpAffine(template, M, (w, h), borderValue=(180, 180, 180))

    # Paste into scene
    scene[100:200, 150:250] = rotated_template

    # Add other objects (distractors)
    cv2.circle(scene, (80, 80), 30, (120, 120, 120), -1)
    cv2.rectangle(scene, (300, 200), (380, 280), (90, 90, 90), -1)

    # Feature matching
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_template, None)
    kp2, des2 = orb.detectAndCompute(gray_scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Template keypoints: {len(kp1)}")
    print(f"Scene keypoints: {len(kp2)}")
    print(f"Good matches: {len(good)}")

    # Visualize results
    result = cv2.drawMatches(template, kp1, scene, kp2, good, None)
    cv2.imwrite('object_template.jpg', template)
    cv2.imwrite('object_scene.jpg', scene)
    cv2.imwrite('object_match.jpg', result)


def main():
    """Main function"""
    # BF matcher
    bf_matcher_demo()

    # KNN matching
    knn_match_demo()

    # FLANN matcher
    flann_matcher_demo()

    # Homography
    homography_demo()

    # Object matching
    match_object_demo()

    print("\nFeature matching demo complete!")


if __name__ == '__main__':
    main()
