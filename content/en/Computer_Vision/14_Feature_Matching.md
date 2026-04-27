# Feature Matching

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the feature matching pipeline from descriptor extraction to correspondence establishment between images
2. Implement brute-force matching with BFMatcher and approximate matching with FLANN using OpenCV
3. Apply appropriate distance metrics (L2 norm vs. Hamming distance) based on descriptor type
4. Filter matches using Lowe's ratio test and cross-check validation to reduce false correspondences
5. Compute a homography matrix using RANSAC to robustly estimate geometric transformations between matched images
6. Design an image stitching workflow that combines feature detection, matching, and homography estimation

---

## Overview

Feature matching is the process of finding and connecting identical feature points across two images. Detection alone cannot answer "is this the same object?" or "how do these two views relate geometrically?" — matching is the step that establishes point correspondences between images, making it the bridge from individual keypoints to higher-level tasks like image stitching, 3D reconstruction, and object tracking. In this lesson, we will learn about BFMatcher, FLANN, distance metrics, Lowe's ratio test, Homography, and RANSAC.

---

## Table of Contents

1. [Feature Matching Fundamentals](#1-feature-matching-fundamentals)
2. [BFMatcher](#2-bfmatcher)
3. [FLANN-based Matcher](#3-flann-based-matcher)
4. [Distance Metrics](#4-distance-metrics)
5. [Match Filtering](#5-match-filtering)
6. [Homography and RANSAC](#6-homography-and-ransac)
7. [Image Stitching Basics](#7-image-stitching-basics)
8. [Practice Problems](#8-practice-problems)

---

## 1. Feature Matching Fundamentals

### Theory: Descriptor Space and Distance Metrics

Each detected keypoint has a descriptor: a fixed-size vector representing its local appearance. Matching two keypoints means asking whether their descriptors are "close" in the descriptor space. Different descriptor families use different spaces:

#### A.1 Float descriptors and L2 distance

SIFT, SURF, and other histogram-of-gradients descriptors produce 128- or 64-dimensional `float32` vectors. The canonical distance is **Euclidean (L2)**:

```
d(a, b) = √ Σ_i (a_i - b_i)²
```

Also commonly used: squared-L2 (skip the sqrt for speed; same ordering), L1 (Manhattan distance, less sensitive to outlier dimensions).

Why L2? SIFT descriptors are approximately Gaussian-distributed around the true location in descriptor space, and L2 is the maximum-likelihood distance under Gaussian noise. Empirically L2 works well even when assumptions are only loosely met.

#### A.2 Binary descriptors and Hamming distance

ORB, BRIEF, BRISK, AKAZE produce binary strings (typically 256 or 512 bits). Each bit compares a pair of pixels and stores which was brighter. The canonical distance is **Hamming distance**:

```
d(a, b) = popcount(a XOR b)   = number of differing bits
```

Computed by bitwise XOR then counting set bits (`__builtin_popcount` / `POPCNT` instruction). For 256-bit descriptors, Hamming distance ranges from 0 (identical) to 256 (maximally different); random pairs have expected distance 128.

Hamming distance on binary descriptors is **roughly 100× faster** than L2 on float descriptors — the reason ORB is used in real-time systems where SIFT is too slow.

The rule is: **match the distance metric to the descriptor type**. Using L2 on binary descriptors is wrong (they aren't vectors in a metric space); using Hamming on float descriptors doesn't even make sense.

### Matching Process

```
+---------------------------------------------------------------------+
|                     Feature Matching Pipeline                        |
+---------------------------------------------------------------------+
|                                                                      |
|   Image 1                        Image 2                             |
|   +---------+                     +---------+                        |
|   | *  *    |                     |   *  *  |                        |
|   |    *  * |                     | *    *  |                        |
|   |  *      |                     |   *     |                        |
|   +---------+                     +---------+                        |
|       |                               |                              |
|       v                               v                              |
|  +----------+                   +----------+                         |
|  | Feature  |                   | Feature  |                         |
|  | Detection|                   | Detection|                         |
|  +----+-----+                   +----+-----+                         |
|       |                               |                              |
|       v                               v                              |
|  +----------+                   +----------+                         |
|  |Descriptor|                   |Descriptor|                         |
|  | Compute  |                   | Compute  |                         |
|  +----+-----+                   +----+-----+                         |
|       |                               |                              |
|       +----------+-------------------+                               |
|                  v                                                   |
|           +--------------+                                           |
|           |   Matching   |                                           |
|           | (BFMatcher   |                                           |
|           |  or FLANN)   |                                           |
|           +------+-------+                                           |
|                  v                                                   |
|           +--------------+                                           |
|           | Filtering    |                                           |
|           | (Ratio Test, |                                           |
|           |  RANSAC)     |                                           |
|           +--------------+                                           |
|                                                                      |
+---------------------------------------------------------------------+
```

### DMatch Structure

```python
import cv2

# DMatch attributes
# match.queryIdx  : Descriptor index in query (first) image
# match.trainIdx  : Descriptor index in train (second) image
# match.imgIdx    : Index of train image (when matching multiple images)
# match.distance  : Distance between descriptors (similarity)
```

---

## 2. BFMatcher

### Theory: Nearest-Neighbor Search: Brute-Force vs FLANN

#### B.1 Brute-force (`BFMatcher`)

For each descriptor in set A, compute distance to every descriptor in set B, return the one with minimum distance. `O(n_A · n_B · d)` where `d` is descriptor length. Exact, simple, and the baseline.

Fine for hundreds of keypoints per image. Becomes expensive when each image has 5000+ keypoints and you're matching many image pairs.

#### B.2 FLANN: Fast Library for Approximate Nearest Neighbors

FLANN trades a small amount of accuracy for a large speedup using spatial indexes:

- For **float descriptors**, uses a forest of randomized **KD-trees**. Each tree partitions descriptor space into hyperrectangles; searching a few trees in parallel produces approximately-nearest neighbors much faster than linear scan.
- For **binary descriptors**, uses **Locality-Sensitive Hashing (LSH)**. Random hyperplane projections produce hash buckets where similar descriptors are likely to collide. Finding the nearest neighbor means checking only candidates that hash to the same bucket.

FLANN's speedup grows with dataset size — typically 10-100× faster than brute-force for large descriptor sets, at the cost of missing a small fraction of true nearest neighbors. For matching-as-a-step-in-a-bigger-pipeline (e.g. stitching), this is almost always a win; every downstream stage has its own noise that dwarfs FLANN's approximation error.

### Concept

```
BFMatcher (Brute-Force Matcher):
Computes distances between all descriptor pairs to find minimum distance

Advantages:
- Simple implementation
- Always guarantees optimal match

Disadvantages:
- O(N * M) complexity (N, M: number of descriptors)
- Slow for large feature sets

                Query Descriptors
                d1   d2   d3   d4
            +----+----+----+----+
Train   d1' | 10 | 25 | 15 | 30 |
Desc    d2' | 20 |  5 | 35 | 12 |  <- Each cell: distance
        d3' | 30 | 18 |  8 | 22 |
            +----+----+----+----+

Match: d1<->d1'(10), d2<->d2'(5), d3<->d3'(8), d4<->d2'(12)
```

### cv2.BFMatcher

```python
import cv2
import numpy as np

def bf_matching_demo(img1_path, img2_path):
    """Basic BFMatcher usage"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # ORB detector (binary descriptors)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # NORM_HAMMING: counts differing bits between binary ORB descriptors;
    # much faster than L2 for binary vectors (single XOR + popcount instruction)
    # crossCheck=True: a match is only kept if A→B and B→A agree — this
    # eliminates one-to-many false matches with no extra computation
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match (returns one best match per query descriptor)
    matches = bf.match(des1, des2)

    # Sort by distance so the first N drawn are the most confident matches
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 30 matches
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:30], None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    print(f"Total matches: {len(matches)}")
    print(f"Min distance: {matches[0].distance:.2f}")
    print(f"Max distance: {matches[-1].distance:.2f}")

    cv2.imshow('BF Matches', result)
    cv2.waitKey(0)

    return matches

matches = bf_matching_demo('query.jpg', 'train.jpg')
```

### crossCheck Option

```python
import cv2

def bf_crosscheck_comparison(img1_path, img2_path):
    """Compare crossCheck option"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # crossCheck=False: every query descriptor gets a match regardless of quality;
    # produces more matches but includes many spurious one-to-many assignments
    bf_no_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_no_cross = bf_no_cross.match(des1, des2)

    # crossCheck=True: symmetric consistency filter — descriptor A must be B's
    # nearest neighbour AND B must be A's nearest neighbour; effectively a free
    # version of the ratio test that works well when feature counts are similar
    bf_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_cross = bf_cross.match(des1, des2)

    print(f"crossCheck=False: {len(matches_no_cross)} matches")
    print(f"crossCheck=True:  {len(matches_cross)} matches")
    # crossCheck=True is always the safer default for bf.match(); use False
    # only when you plan to apply a separate ratio test via knnMatch

bf_crosscheck_comparison('query.jpg', 'train.jpg')
```

### knnMatch

```python
import cv2
import numpy as np

def bf_knn_matching(img1_path, img2_path, k=2):
    """k-nearest neighbors matching"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # SIFT detector (float descriptors)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # L2 (Euclidean) distance for float descriptors like SIFT — these represent
    # gradient histograms where Euclidean distance is geometrically meaningful
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # k=2: retrieve the 2 nearest neighbours so that the ratio test (Lowe's)
    # can compare the best match against the second-best; k=1 cannot do this
    matches = bf.knnMatch(des1, des2, k=k)

    # k matches per query descriptor
    print(f"Query descriptor count: {len(des1)}")
    print(f"{k} matches per query")

    # Check first query's matches
    if len(matches) > 0:
        print(f"\nFirst query's matches:")
        for i, m in enumerate(matches[0]):
            print(f"  Match {i+1}: trainIdx={m.trainIdx}, distance={m.distance:.2f}")

    return matches

matches = bf_knn_matching('query.jpg', 'train.jpg', k=2)
```

---

## 3. FLANN-based Matcher

### Concept

```
FLANN (Fast Library for Approximate Nearest Neighbors):
Library for approximate nearest neighbor search

Characteristics:
- Faster than BFMatcher (for large datasets)
- Approximate algorithm (not 100% accurate)
- Uses KD-Tree, K-Means Tree, etc.

Index Types:
1. FLANN_INDEX_KDTREE (0): For float descriptors
2. FLANN_INDEX_LSH (6): For binary descriptors
```

### FLANN Usage

```python
import cv2
import numpy as np

def flann_matching_sift(img1_path, img2_path):
    """FLANN matching (SIFT - float descriptors)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN (Fast Library for Approximate Nearest Neighbours) is preferred over
    # BFMatcher when descriptor counts are large (thousands): it uses tree-based
    # approximate search instead of exhaustive O(N*M) comparison
    FLANN_INDEX_KDTREE = 1
    index_params = dict(
        algorithm=FLANN_INDEX_KDTREE,  # KD-Tree partitions the 128-D SIFT space;
                                       # efficient for float descriptors up to ~128 dims
        trees=5   # Multiple randomised trees improve accuracy at little memory cost;
                  # 5 is the standard trade-off between build time and query speed
    )
    search_params = dict(
        checks=50  # How many tree nodes to visit per query; 50 gives ~95% recall
                   # vs brute-force — increase to 100+ for higher precision applications
    )

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # k=2 to enable Lowe's ratio test — we need the second-best match to judge
    # whether the best match stands out clearly enough
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test: reject matches where best and second-best distances are
    # similar (ambiguous); 0.7 is Lowe's recommended threshold from the 2004 paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Total matches: {len(matches)}")
    print(f"Good matches: {len(good_matches)}")

    # Draw results
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('FLANN Matches', result)
    cv2.waitKey(0)

    return good_matches, kp1, kp2

matches, kp1, kp2 = flann_matching_sift('query.jpg', 'train.jpg')
```

### FLANN for ORB (Binary)

```python
import cv2
import numpy as np

def flann_matching_orb(img1_path, img2_path):
    """FLANN matching (ORB - binary descriptors)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # LSH (Locality-Sensitive Hashing) for binary descriptors: KD-Trees are
    # inefficient for bit vectors because Euclidean distance is meaningless there;
    # LSH hashes similar binary strings to the same bucket using Hamming distance
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,        # More tables = higher recall but more memory;
                               # 6–12 is typical for 256-bit ORB descriptors
        key_size=12,           # Bits per hash key: smaller keys are less selective
                               # but handle descriptor noise better
        multi_probe_level=1    # Probe neighbouring buckets to boost recall;
                               # 1–2 is the standard setting
    )
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # FLANN's internal math expects float32; the cast does not change the binary
    # content but satisfies the type check before passing to FLANN's C++ backend
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches, None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow('FLANN ORB Matches', result)
    cv2.waitKey(0)

    return good_matches

flann_matching_orb('query.jpg', 'train.jpg')
```

---

## 4. Distance Metrics

### Distance Types

The choice of distance metric must match the descriptor's mathematical structure. SIFT encodes gradient histograms as float vectors — two similar patches produce nearby vectors in Euclidean space, so L2 is geometrically meaningful. ORB encodes patch comparisons as individual bits — two similar patches differ in few bits, so XOR + popcount (Hamming distance) is the natural measure and runs in a single CPU instruction on modern hardware.

```
+--------------------------------------------------------------------+
|                        Distance Metric Comparison                   |
+--------------------------------------------------------------------+
|                                                                     |
|  cv2.NORM_L1 (Manhattan Distance)                                  |
|  d = sum|a_i - b_i|                                                 |
|  -> Rarely used                                                     |
|                                                                     |
|  cv2.NORM_L2 (Euclidean Distance)                                  |
|  d = sqrt(sum(a_i - b_i)^2)                                        |
|  -> For SIFT, SURF, etc. float descriptors                         |
|                                                                     |
|  cv2.NORM_HAMMING                                                   |
|  d = sum(a_i XOR b_i)                                              |
|  -> For ORB, BRIEF, etc. binary descriptors (256 bits)             |
|                                                                     |
|  cv2.NORM_HAMMING2                                                  |
|  -> For ORB (WTA_K=3,4)                                            |
|                                                                     |
+--------------------------------------------------------------------+
```

### Recommended Metrics per Descriptor

```python
import cv2

# Recommended distance metric per descriptor type
descriptor_distance = {
    'SIFT': cv2.NORM_L2,
    'SURF': cv2.NORM_L2,
    'KAZE': cv2.NORM_L2,
    'ORB': cv2.NORM_HAMMING,
    'BRISK': cv2.NORM_HAMMING,
    'AKAZE': cv2.NORM_HAMMING,  # Binary mode
    'BRIEF': cv2.NORM_HAMMING,
    'FREAK': cv2.NORM_HAMMING,
}

def get_matcher(descriptor_type):
    """Return matcher for descriptor type"""
    norm_type = descriptor_distance.get(descriptor_type, cv2.NORM_L2)
    return cv2.BFMatcher(norm_type, crossCheck=True)
```

---

## 5. Match Filtering

### Theory: Filtering: Deciding Which Matches to Trust

Even with a perfect descriptor and exact nearest-neighbor search, not every returned match is correct. Two common filters dramatically reduce false matches:

#### C.1 Cross-check

A match `(i, j(i))` is accepted only if:

- `j(i) = argmin_j  d(desc_A[i], desc_B[j])` (B's j is A's i's best match), **and**
- `i = argmin_i  d(desc_A[i], desc_B[j(i)])` (A's i is B's j(i)'s best match).

Implemented via `BFMatcher(..., crossCheck=True)`. Roughly halves the number of matches but with much higher precision. Incompatible with `knnMatch` (which returns k-nearest-neighbors — cross-check only works for single nearest match).

#### C.2 Lowe's Ratio Test

For each descriptor in A, find the **two** nearest neighbors in B: the best at distance `d₁` and second-best at `d₂`. Accept the match only if `d₁ / d₂ < τ` (typically `τ = 0.7–0.8`).

The idea: if the best match is much closer than the second-best, it's likely correct; if both are similarly close, the descriptor is ambiguous (repeated pattern, or no true match exists) and we should reject. This is a **statistical test** — Lowe showed empirically that a ratio of 0.8 eliminates ~90% of false matches while losing only ~5% of correct ones.

The ratio test is implemented via `knnMatch(k=2)` followed by the ratio check:

```python
good_matches = [m for m, n in knn_matches if m.distance < 0.75 * n.distance]
```

This is the standard filter for SIFT/ORB matching pipelines and is one of the most practically important results in classical computer vision.

### Lowe's Ratio Test

```
Lowe's Ratio Test:
Filter by ratio of distances between nearest and second-nearest neighbor

Principle:
Good match -> Nearest neighbor is clearly closer (small ratio)
Bad match -> Multiple candidates at similar distances (large ratio)

distance(best) / distance(second_best) < threshold

Recommended threshold: 0.7 ~ 0.8
```

```python
import cv2
import numpy as np

def lowe_ratio_test(img1_path, img2_path, ratio_thresh=0.75):
    """Apply Lowe's ratio test"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # No crossCheck here: knnMatch requires crossCheck=False to return k>1 results
    bf = cv2.BFMatcher()
    # k=2: we need the runner-up to compute the ratio; k=1 would not allow filtering
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test: reject matches where the best and second-best are
    # too similar — Lowe (2004) found 0.7–0.8 eliminates ~90% of false
    # matches while retaining ~95% of correct ones.
    # Intuition: a correct match has one clearly better candidate (low ratio);
    # a false match has many similar candidates (ratio close to 1.0)
    good_matches = []
    for m, n in matches:
        ratio = m.distance / n.distance
        if ratio < ratio_thresh:
            good_matches.append(m)

    print(f"Total matches: {len(matches)}")
    print(f"Ratio test passed: {len(good_matches)}")
    print(f"Filter ratio: {len(good_matches)/len(matches)*100:.1f}%")

    # Match quality analysis
    if good_matches:
        distances = [m.distance for m in good_matches]
        print(f"Average distance: {np.mean(distances):.2f}")
        print(f"Distance std dev: {np.std(distances):.2f}")

    return good_matches, kp1, kp2

matches, kp1, kp2 = lowe_ratio_test('query.jpg', 'train.jpg')
```

### Distance-based Filtering

```python
import cv2
import numpy as np

def distance_based_filtering(matches, threshold_factor=2.0):
    """Distance-based match filtering"""
    if not matches:
        return []

    distances = [m.distance for m in matches]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    # Keep only those below mean + k*std
    threshold = mean_dist + threshold_factor * std_dist

    good_matches = [m for m in matches if m.distance < threshold]

    print(f"Distance mean: {mean_dist:.2f}")
    print(f"Distance std dev: {std_dist:.2f}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Filtering result: {len(matches)} -> {len(good_matches)}")

    return good_matches
```

### Symmetric Matching

```python
import cv2

def symmetric_matching(des1, des2, norm_type=cv2.NORM_L2):
    """Symmetric matching (verify both A->B and B->A)"""
    bf = cv2.BFMatcher(norm_type)

    # A -> B matching
    matches_ab = bf.knnMatch(des1, des2, k=1)

    # B -> A matching
    matches_ba = bf.knnMatch(des2, des1, k=1)

    # Select only bidirectionally consistent matches
    symmetric = []
    for m_ab in matches_ab:
        if len(m_ab) == 0:
            continue

        query_idx = m_ab[0].queryIdx
        train_idx = m_ab[0].trainIdx

        # Check reverse direction in B->A
        for m_ba in matches_ba:
            if len(m_ba) == 0:
                continue

            if m_ba[0].queryIdx == train_idx and m_ba[0].trainIdx == query_idx:
                symmetric.append(m_ab[0])
                break

    return symmetric
```

---

## 6. Homography and RANSAC

### Theory: Geometric Verification: RANSAC

After descriptor-based filtering you still have typically 20–50% outliers — matches that pass descriptor filters but correspond to different physical points. Outliers are caused by repeated patterns, illumination changes, near-identical descriptors from different objects, or simply a point in A having no real match in B.

A **geometric model** says: "all correct matches must be consistent with this transformation". For two images of a planar scene, the transformation is a **homography**; for two views of a rigid 3D scene, it is described by the **fundamental matrix**. Either way, an inlier is a match consistent with the model; an outlier is not.

#### D.1 The RANSAC algorithm

Fitting a model by least squares fails when 30% of the data is wrong — outliers drag the estimate away from truth. RANSAC (Random Sample Consensus, Fischler & Bolles, 1981) solves this by **assuming inliers form a majority consensus**:

1. **Randomly sample** the minimum number of points needed to fit the model (4 for a homography, 7 or 8 for a fundamental matrix).
2. **Fit the model** to just those points — minimal solver, no least squares.
3. **Count inliers**: points from the full set whose error under this model is below a threshold.
4. **Repeat** for many random samples.
5. **Keep the model with the most inliers**. Optionally re-fit that model to all its inliers using least squares for final polish.

The key insight: even with 50% outliers, randomly drawing 4 correct inliers has probability `(0.5)⁴ = 6.25%` per sample. Do 500 samples and you will almost certainly pick some clean sample, which will get a much larger inlier count than any outlier-contaminated sample.

#### D.2 Why it works as an outlier filter

After RANSAC you don't just get a model — you also get a set of **inlier matches** (those consistent with the chosen model). Discarding outliers this way is typically more effective than any descriptor-based filter, because it uses the whole image's geometric structure rather than local descriptor similarity.

### Theory: Homography and the Fundamental Matrix

OpenCV's `findHomography(pts1, pts2, method=cv2.RANSAC, threshold=5.0)` runs the RANSAC algorithm with the homography as the model (4 points per sample). Returns the 3×3 homography matrix and a mask indicating which matches were used as inliers.

Two important points about when homography is the right model:

- **Planar scene**. All matched 3D points lie on a single plane. Every pair of images of the plane (from any viewpoints) are related by a homography. Image stitching for flat things (posters, floors) and textbook examples use this.
- **Pure rotation**. If the camera only rotates (no translation), any 3D scene is related between the two views by a homography. This is the basis of panorama stitching with a tripod-mounted camera.

If the scene has **3D structure with camera translation**, homography is the wrong model — use `findFundamentalMat` instead. The fundamental matrix is 3×3 with rank 2, it relates views via epipolar geometry, and RANSAC finds it the same way (different minimal solver, different inlier test).

### Homography Concept

```
Homography:
3x3 matrix representing perspective transformation between planes

+     +   +           + +   +
| x'  |   | h11 h12 h13 | | x |
| y'  | = | h21 h22 h23 | | y |
|  1  |   | h31 h32 h33 | | 1 |
+     +   +           + +   +

x' = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
y' = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)

Applications:
- Object position estimation
- Image registration
- Panorama stitching
- AR marker detection
```

### cv2.findHomography()

```python
import cv2
import numpy as np

def find_object_homography(img1_path, img2_path, min_matches=10):
    """Find object using homography"""
    img1 = cv2.imread(img1_path)  # Query (object to find)
    img2 = cv2.imread(img2_path)  # Target (scene)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT features and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    # Homography needs at least 4 point correspondences (DLT algorithm);
    # requiring 10+ gives RANSAC enough candidates to find a consensus set
    # even if half are outliers
    if len(good_matches) >= min_matches:
        # reshape(-1, 1, 2): cv2.findHomography requires this specific shape
        # (Nx1x2) — it signals that each row is one 2D point, not a flat array
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC robustly estimates H even when 30–50% of matches are wrong
        # (outliers from background clutter or repetitive texture);
        # threshold=5.0 px: a reprojection error under 5 pixels is considered
        # an inlier — tight enough to reject bad matches, loose enough for
        # sub-pixel descriptor imprecision
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is not None:
            # Transform query image corners
            h, w = gray1.shape
            corners = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)

            transformed_corners = cv2.perspectiveTransform(corners, H)

            # Draw object location on target image
            result = img2.copy()
            cv2.polylines(
                result,
                [np.int32(transformed_corners)],
                True,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            # Match visualization
            matches_mask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 255, 0),
                singlePointColor=None,
                matchesMask=matches_mask,
                flags=2
            )

            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2,
                good_matches, None, **draw_params
            )

            cv2.imshow('Object Detection', result)
            cv2.imshow('Matches', match_img)
            cv2.waitKey(0)

            # Inlier ratio
            inliers = np.sum(mask)
            print(f"Inliers: {inliers}/{len(good_matches)}")
            print(f"Inlier ratio: {inliers/len(good_matches)*100:.1f}%")

            return H, transformed_corners
    else:
        print(f"Insufficient matches: {len(good_matches)} < {min_matches}")
        return None, None

H, corners = find_object_homography('book_cover.jpg', 'scene.jpg')
```

### Understanding RANSAC

```
RANSAC (RANdom SAmple Consensus):
Model estimation from data with outliers

Algorithm:
1. Randomly select minimum samples (homography: 4 points)
2. Compute model
3. Compute error for all points
4. Count points within threshold (inliers)
5. Repeat and select model with most inliers
6. Recompute model with inliers (optional)

+----------------------------------------+
|  *  *  *  *  *                         |
|     *  *  *        <- Inliers (near line) |
|        *  *  *                         |
|  x                                     |
|           x        <- Outliers         |
|     x          x                       |
+----------------------------------------+

findHomography parameters:
- cv2.RANSAC: Use RANSAC
- ransacReprojThreshold: Inlier threshold (pixels)
```

```python
import cv2
import numpy as np

def homography_methods_comparison(src_pts, dst_pts):
    """Compare various homography computation methods"""

    methods = [
        (0, 'Regular (LS)'),
        (cv2.RANSAC, 'RANSAC'),
        (cv2.LMEDS, 'Least-Median'),
        (cv2.RHO, 'PROSAC'),
    ]

    for method, name in methods:
        try:
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method,
                ransacReprojThreshold=5.0
            )

            if H is not None and mask is not None:
                inliers = np.sum(mask)
                print(f"{name}: {inliers}/{len(src_pts)} inliers")
            else:
                print(f"{name}: Failed")
        except Exception as e:
            print(f"{name}: Error - {e}")
```

---

## 7. Image Stitching Basics

### Simple Panorama

```python
import cv2
import numpy as np

def simple_panorama(img1_path, img2_path):
    """Simple panorama stitching"""
    img1 = cv2.imread(img1_path)  # Left image
    img2 = cv2.imread(img2_path)  # Right image

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Feature detection and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Matches: {len(good)}")

    if len(good) < 4:
        print("Not enough matches.")
        return None

    # Compute homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography computation failed")
        return None

    # Image warping
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Project the four corners of img1 through H to find where they land in
    # img2's coordinate system — this tells us the bounding box of the panorama
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners1_transformed, corners2], axis=0)

    # Bounding box of the union of both images in the target coordinate system
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Shift the entire result so the top-left is at (0,0) — negative x_min/y_min
    # mean part of img1 maps to negative coordinates which warpPerspective would clip
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float32)

    result_width = x_max - x_min
    result_height = y_max - y_min

    # translation @ H: apply H first (warp img1 to img2 coords) then translate
    # so the result sits inside the positive canvas — order matters in homogeneous coords
    warped1 = cv2.warpPerspective(
        img1,
        translation @ H,
        (result_width, result_height)
    )

    # Copy image 2
    warped1[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2

    cv2.imshow('Panorama', warped1)
    cv2.waitKey(0)

    return warped1

panorama = simple_panorama('left.jpg', 'right.jpg')
```

### Using OpenCV Stitcher

```python
import cv2
import numpy as np

def opencv_stitcher(image_paths):
    """Use OpenCV Stitcher class"""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        print("At least 2 images required.")
        return None

    # Create Stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # Or: cv2.Stitcher_SCANS (for document scans)

    # Perform stitching
    status, result = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        cv2.imshow('Stitched', result)
        cv2.waitKey(0)
        return result
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Need more images.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Homography estimation failed")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Camera parameters adjustment failed")

    return None

# Usage example
image_files = ['pano1.jpg', 'pano2.jpg', 'pano3.jpg']
result = opencv_stitcher(image_files)
```

---

## 8. Practice Problems

### Problem 1: Find Optimal Matching Parameters

Test various ratio threshold values to find the optimal value.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_ratio(img1_path, img2_path):
    """Find optimal ratio threshold"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    ratios = np.arange(0.5, 1.0, 0.05)
    results = []

    for ratio in ratios:
        good = [m for m, n in matches if m.distance < ratio * n.distance]
        results.append(len(good))

    # Graph
    plt.figure(figsize=(10, 5))
    plt.plot(ratios, results, 'b-o')
    plt.xlabel('Ratio Threshold')
    plt.ylabel('Number of Matches')
    plt.title('Ratio Threshold vs Match Count')
    plt.grid(True)
    plt.show()

    # Gradient change analysis
    gradients = np.gradient(results)
    optimal_idx = np.argmax(np.abs(gradients))
    optimal_ratio = ratios[optimal_idx]

    print(f"Recommended ratio threshold: {optimal_ratio:.2f}")

    return optimal_ratio

optimal = find_optimal_ratio('query.jpg', 'train.jpg')
```

</details>

### Problem 2: Multiple Object Detection

Detect multiple instances of the same object in a scene.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def detect_multiple_objects(template_path, scene_path, threshold=10):
    """Detect multiple identical objects"""
    template = cv2.imread(template_path)
    scene = cv2.imread(scene_path)

    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(gray_t, None)
    kp_s, des_s = sift.detectAndCompute(gray_s, None)

    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des_t, des_s, k=2)

    # Ratio test
    good_matches = []
    for m, n in all_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < threshold:
        print("Insufficient matches")
        return []

    # Clustering to find multiple instances
    scene_pts = np.array([kp_s[m.trainIdx].pt for m in good_matches])

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = min(5, len(good_matches) // threshold)  # Max 5 objects

    if k < 1:
        k = 1

    _, labels, centers = cv2.kmeans(
        np.float32(scene_pts),
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    result = scene.copy()
    detected = []

    for cluster_id in range(k):
        cluster_mask = labels.ravel() == cluster_id
        cluster_matches = [m for m, is_in in zip(good_matches, cluster_mask) if is_in]

        if len(cluster_matches) >= threshold // 2:
            # Compute homography per cluster
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_s[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                h, w = gray_t.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(corners, H)

                cv2.polylines(result, [np.int32(transformed)], True, (0, 255, 0), 3)
                detected.append(transformed)

    print(f"Objects detected: {len(detected)}")
    cv2.imshow('Multiple Objects', result)
    cv2.waitKey(0)

    return detected

detect_multiple_objects('coin.jpg', 'coins.jpg')
```

</details>

### Problem 3: Real-time Object Tracking

Track a template object in real-time from webcam.

<details>
<summary>Solution Code</summary>

```python
import cv2
import numpy as np

def realtime_object_tracking(template_path):
    """Real-time object tracking"""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template.shape

    # Use ORB (fast)
    orb = cv2.ORB_create(nfeatures=500)
    kp_t, des_t = orb.detectAndCompute(template, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = orb.detectAndCompute(gray, None)

        if des_f is not None and len(des_f) > 10:
            matches = bf.knnMatch(des_t, des_f, k=2)

            # Ratio test
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) >= 10:
                src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H)
                    cv2.polylines(frame, [np.int32(transformed)], True, (0, 255, 0), 3)

                    # Display match count
                    cv2.putText(frame, f'Matches: {len(good)}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# realtime_object_tracking('logo.jpg')
```

</details>

### Recommended Problems

| Difficulty | Topic | Description |
|------------|-------|-------------|
| * | Basic Matching | Feature matching between two images |
| ** | Filtering | Ratio test, distance filtering |
| ** | Object Detection | Find object using homography |
| *** | Panorama | Stitch 2+ images |
| *** | Real-time Tracking | Track object from webcam |

---

## Next Steps

- [15_Object_Detection_Basics.md](./15_Object_Detection_Basics.md) - Template Matching, Haar Cascade, HOG+SVM

---

## References

- [OpenCV Feature Matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
- [Homography Tutorial](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html)
- [Image Stitching](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)
