"""
Exercises for Lesson 14: GPU Computing
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Exercise 1 -- Reduction Variants (parallel max and argmax)
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Implement parallel max and parallel argmax using the GPU-style reduction
    pattern. Test with an array of 1024 random values.
    """
    np.random.seed(42)
    data = np.random.rand(1024)

    # Parallel max reduction
    def gpu_parallel_max(arr):
        buf = arr.copy()
        n = len(buf)
        steps = 0
        stride = 1
        while stride < n:
            for i in range(0, n, 2 * stride):
                if i + stride < n:
                    buf[i] = max(buf[i], buf[i + stride])
            stride *= 2
            steps += 1
        return buf[0], steps

    # Parallel argmax: carry both value and index through the reduction
    def gpu_parallel_argmax(arr):
        n = len(arr)
        vals = arr.copy()
        idxs = np.arange(n)
        steps = 0
        stride = 1
        while stride < n:
            for i in range(0, n, 2 * stride):
                if i + stride < n:
                    if vals[i + stride] > vals[i]:
                        vals[i] = vals[i + stride]
                        idxs[i] = idxs[i + stride]
            stride *= 2
            steps += 1
        return vals[0], int(idxs[0]), steps

    max_val, max_steps = gpu_parallel_max(data)
    argmax_val, argmax_idx, argmax_steps = gpu_parallel_argmax(data)

    # Verify
    expected_max = np.max(data)
    expected_argmax = np.argmax(data)

    print(f"  Array size: {len(data)}")
    print(f"  Parallel max: {max_val:.6f} in {max_steps} steps (expected: {expected_max:.6f})")
    print(f"  Parallel argmax: index={argmax_idx}, value={argmax_val:.6f} "
          f"in {argmax_steps} steps")
    print(f"  Expected argmax: index={expected_argmax}, value={data[expected_argmax]:.6f}")
    print(f"  Correct: max={'YES' if abs(max_val - expected_max) < 1e-10 else 'NO'}, "
          f"argmax={'YES' if argmax_idx == expected_argmax else 'NO'}")
    print(f"  log2(1024) = {np.log2(1024):.0f} steps (matches)")


# ---------------------------------------------------------------------------
# Exercise 2 -- Prefix Sum (Blelloch Scan)
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Implement the Blelloch scan for power-of-two and non-power-of-two arrays.
    Verify against np.cumsum.
    """

    def blelloch_scan(data):
        """Work-efficient exclusive prefix sum (Blelloch 1990)."""
        n = len(data)
        # Pad to next power of two
        orig_n = n
        pot = 1
        while pot < n:
            pot *= 2
        buf = np.zeros(pot, dtype=float)
        buf[:n] = data

        # Up-sweep (reduce)
        stride = 1
        while stride < pot:
            for i in range(2 * stride - 1, pot, 2 * stride):
                buf[i] += buf[i - stride]
            stride *= 2

        # Set last to zero (exclusive scan identity)
        buf[pot - 1] = 0

        # Down-sweep (distribute)
        stride = pot // 2
        while stride >= 1:
            for i in range(2 * stride - 1, pot, 2 * stride):
                temp = buf[i - stride]
                buf[i - stride] = buf[i]
                buf[i] += temp
            stride //= 2

        return buf[:orig_n]

    # Power-of-two test
    data_pot = np.array([3, 1, 7, 0, 4, 1, 6, 3], dtype=float)
    result_pot = blelloch_scan(data_pot)
    expected_pot = np.concatenate([[0], np.cumsum(data_pot)[:-1]])

    print(f"  Power-of-two (n=8):")
    print(f"    Input:    {data_pot.astype(int)}")
    print(f"    Scan:     {result_pot.astype(int)}")
    print(f"    Expected: {expected_pot.astype(int)}")
    print(f"    Match: {np.allclose(result_pot, expected_pot)}")

    # Non-power-of-two test
    data_npot = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    result_npot = blelloch_scan(data_npot)
    expected_npot = np.concatenate([[0], np.cumsum(data_npot)[:-1]])

    print(f"\n  Non-power-of-two (n=6):")
    print(f"    Input:    {data_npot.astype(int)}")
    print(f"    Scan:     {result_npot.astype(int)}")
    print(f"    Expected: {expected_npot.astype(int)}")
    print(f"    Match: {np.allclose(result_npot, expected_npot)}")

    # Larger test
    np.random.seed(0)
    data_large = np.random.randint(0, 100, 1000).astype(float)
    result_large = blelloch_scan(data_large)
    expected_large = np.concatenate([[0], np.cumsum(data_large)[:-1]])
    print(f"\n  Large array (n=1000):")
    print(f"    Match: {np.allclose(result_large, expected_large)}")


# ---------------------------------------------------------------------------
# Exercise 3 -- Histogram
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Design a GPU algorithm to compute a histogram of pixel intensities
    (256 bins) for a grayscale image. Explain atomic operations for bin
    increments. Implement a simulated version.
    """
    np.random.seed(42)
    # Generate a test grayscale image
    image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

    def gpu_style_histogram(image, num_bins=256):
        """
        Simulated GPU histogram computation.

        GPU algorithm:
        1. Allocate shared memory histogram per work group (256 bins)
        2. Each thread atomically increments its pixel's bin in shared memory
        3. Barrier -- wait for all threads in work group
        4. Reduce: atomically add shared histogram to global histogram

        Atomic operations are needed because multiple threads may increment
        the same bin simultaneously. Without atomics, increments would be lost
        (read-modify-write race condition).
        """
        histogram = np.zeros(num_bins, dtype=int)
        flat = image.ravel()

        # Simulate: each "thread" increments a bin
        # In a real GPU, atomicAdd would be used:
        #   atomicAdd(&histogram[pixel_value], 1);
        for pixel in flat:
            histogram[pixel] += 1

        return histogram

    hist = gpu_style_histogram(image)

    # Verify against numpy
    expected = np.bincount(image.ravel(), minlength=256)
    match = np.array_equal(hist, expected)

    print(f"  Image size: {image.shape}")
    print(f"  Histogram bins: 256")
    print(f"  Total pixels: {image.size}")
    print(f"  Match with numpy: {match}")
    print(f"  Top 5 bins: {np.argsort(hist)[-5:][::-1]} "
          f"(counts: {hist[np.argsort(hist)[-5:][::-1]]})")
    print(f"\n  GPU Algorithm Design:")
    print(f"    Work group size: 256 threads")
    print(f"    Shared memory: 256 * 4 bytes = 1 KB per work group")
    print(f"    Step 1: Each thread loads one pixel, atomicAdd to shared histogram")
    print(f"    Step 2: barrier() -- synchronize all threads")
    print(f"    Step 3: First 256 threads atomicAdd shared bins to global histogram")
    print(f"    Atomics needed: bin increments are read-modify-write operations")
    print(f"    Without atomics, concurrent threads lose increments (race condition)")


# ---------------------------------------------------------------------------
# Exercise 4 -- Tiling Analysis
# ---------------------------------------------------------------------------

def exercise_4():
    """
    For a 1920x1080 image with a 7x7 Gaussian kernel and work groups of
    16x16 threads, calculate: (a) how many tiles, (b) shared memory per tile
    (including halo), (c) total global memory reads saved compared to non-tiled.
    """
    W, H = 1920, 1080
    kernel_size = 7
    kernel_radius = kernel_size // 2  # 3
    tile_w, tile_h = 16, 16

    # (a) Number of tiles
    tiles_x = (W + tile_w - 1) // tile_w
    tiles_y = (H + tile_h - 1) // tile_h
    total_tiles = tiles_x * tiles_y

    # (b) Shared memory per tile (including halo)
    shared_w = tile_w + 2 * kernel_radius  # 16 + 6 = 22
    shared_h = tile_h + 2 * kernel_radius  # 16 + 6 = 22
    shared_pixels = shared_w * shared_h
    bytes_per_pixel = 4  # float32
    shared_bytes = shared_pixels * bytes_per_pixel

    # (c) Global memory reads comparison
    total_pixels = W * H

    # Non-tiled: each pixel reads kernel_size^2 neighbors from global memory
    reads_non_tiled = total_pixels * kernel_size * kernel_size

    # Tiled: each tile loads shared_pixels from global memory once
    # Then each pixel reads from shared memory (fast, not counted as global)
    reads_tiled = total_tiles * shared_pixels

    savings = reads_non_tiled - reads_tiled
    savings_pct = savings / reads_non_tiled * 100

    print(f"  Image: {W}x{H} = {total_pixels:,} pixels")
    print(f"  Kernel: {kernel_size}x{kernel_size} (radius={kernel_radius})")
    print(f"  Work group: {tile_w}x{tile_h}")
    print(f"\n  (a) Number of tiles:")
    print(f"      Tiles X: ceil({W}/{tile_w}) = {tiles_x}")
    print(f"      Tiles Y: ceil({H}/{tile_h}) = {tiles_y}")
    print(f"      Total tiles: {total_tiles:,}")
    print(f"\n  (b) Shared memory per tile:")
    print(f"      Shared tile size: {shared_w}x{shared_h} = {shared_pixels} pixels")
    print(f"      Shared memory: {shared_bytes} bytes ({shared_bytes/1024:.1f} KB)")
    print(f"\n  (c) Global memory reads:")
    print(f"      Non-tiled: {total_pixels:,} x {kernel_size*kernel_size} = {reads_non_tiled:,}")
    print(f"      Tiled:     {total_tiles:,} x {shared_pixels} = {reads_tiled:,}")
    print(f"      Savings:   {savings:,} reads ({savings_pct:.1f}%)")
    print(f"      Tiled reads are {reads_non_tiled / reads_tiled:.1f}x fewer global accesses")


# ---------------------------------------------------------------------------
# Exercise 5 -- SoA vs AoS Transformation
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Convert AoS particle structure to SoA layout. Measure NumPy vectorized
    update speed for both layouts.
    """
    N = 100_000
    np.random.seed(42)

    # AoS layout: array of dicts (simulating struct of 6 floats)
    particles_aos = np.zeros(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'),
    ])
    particles_aos['x'] = np.random.rand(N).astype('f4')
    particles_aos['y'] = np.random.rand(N).astype('f4')
    particles_aos['z'] = np.random.rand(N).astype('f4')
    particles_aos['vx'] = np.random.rand(N).astype('f4')
    particles_aos['vy'] = np.random.rand(N).astype('f4')
    particles_aos['vz'] = np.random.rand(N).astype('f4')

    # SoA layout: separate arrays
    x = particles_aos['x'].copy()
    y = particles_aos['y'].copy()
    z = particles_aos['z'].copy()
    vx = particles_aos['vx'].copy()
    vy = particles_aos['vy'].copy()
    vz = particles_aos['vz'].copy()

    dt = 0.016  # ~60 FPS
    gravity_y = -9.81

    # Benchmark AoS update
    t0 = time.perf_counter()
    for _ in range(100):
        particles_aos['vy'] += gravity_y * dt
        particles_aos['x'] += particles_aos['vx'] * dt
        particles_aos['y'] += particles_aos['vy'] * dt
        particles_aos['z'] += particles_aos['vz'] * dt
    t_aos = time.perf_counter() - t0

    # Benchmark SoA update
    t0 = time.perf_counter()
    for _ in range(100):
        vy += gravity_y * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
    t_soa = time.perf_counter() - t0

    print(f"  Particles: {N:,}")
    print(f"  100 update iterations:")
    print(f"    AoS (structured array): {t_aos:.4f}s")
    print(f"    SoA (separate arrays):  {t_soa:.4f}s")
    ratio = t_aos / t_soa if t_soa > 0 else float('inf')
    print(f"    SoA speedup: {ratio:.2f}x")
    print(f"\n  Why SoA is faster on GPU:")
    print(f"    - Thread k reads x[k]: consecutive threads read consecutive addresses")
    print(f"    - This enables coalesced memory access (one wide transaction)")
    print(f"    - AoS: thread k reads particles[k].x, which is 24 bytes apart")
    print(f"    - This causes strided access (many separate transactions)")


# ---------------------------------------------------------------------------
# Exercise 6 -- Sorting Comparison (Bitonic Sort)
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Implement bitonic sort and compare with Python's built-in sorted() for
    arrays of size 256, 1024, and 4096. Discuss when the parallel approach
    would be faster on a GPU.
    """

    def bitonic_sort(data):
        """Bitonic sort: GPU-friendly parallel sorting algorithm."""
        arr = list(data)
        n = len(arr)
        # Pad to power of two
        pot = 1
        while pot < n:
            pot *= 2
        arr.extend([float('inf')] * (pot - n))

        k = 2
        while k <= pot:
            j = k // 2
            while j >= 1:
                for i in range(pot):
                    partner = i ^ j
                    if partner > i:
                        ascending = ((i & k) == 0)
                        if ascending:
                            if arr[i] > arr[partner]:
                                arr[i], arr[partner] = arr[partner], arr[i]
                        else:
                            if arr[i] < arr[partner]:
                                arr[i], arr[partner] = arr[partner], arr[i]
                j //= 2
            k *= 2

        return arr[:n]

    sizes = [256, 1024, 4096]
    print(f"  {'Size':>6s}  {'Bitonic (s)':>12s}  {'sorted() (s)':>12s}  {'Ratio':>8s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>12s}  {'---':>8s}")

    for size in sizes:
        np.random.seed(42)
        data = np.random.rand(size).tolist()

        t0 = time.perf_counter()
        result_bitonic = bitonic_sort(data)
        t_bitonic = time.perf_counter() - t0

        t0 = time.perf_counter()
        result_sorted = sorted(data)
        t_sorted = time.perf_counter() - t0

        # Verify correctness
        correct = result_bitonic == result_sorted
        ratio = t_bitonic / t_sorted if t_sorted > 0 else float('inf')

        print(f"  {size:6d}  {t_bitonic:12.6f}  {t_sorted:12.6f}  {ratio:8.1f}x"
              f"  {'OK' if correct else 'FAIL'}")

    print(f"\n  Discussion:")
    print(f"    In Python, bitonic sort is much slower than sorted() because")
    print(f"    it has O(n log^2 n) comparisons vs O(n log n) for Timsort.")
    print(f"    However, on a GPU:")
    print(f"    - Each comparison step is data-independent (no divergence)")
    print(f"    - All n/2 comparisons at each step execute in parallel")
    print(f"    - Total parallel time: O(log^2 n) steps with n/2 threads each")
    print(f"    - For n=1M on a GPU with 10K+ cores, this is practical")
    print(f"    - Alternative: radix sort (O(n) work, O(n/P) parallel time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Reduction Variants (Max and Argmax) ===")
    exercise_1()

    print("\n=== Exercise 2: Prefix Sum (Blelloch Scan) ===")
    exercise_2()

    print("\n=== Exercise 3: Histogram ===")
    exercise_3()

    print("\n=== Exercise 4: Tiling Analysis ===")
    exercise_4()

    print("\n=== Exercise 5: SoA vs AoS Transformation ===")
    exercise_5()

    print("\n=== Exercise 6: Sorting Comparison ===")
    exercise_6()

    print("\nAll exercises completed!")
