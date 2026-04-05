"""
GPU Computing Patterns (CPU Simulation)
========================================

Implements classic GPU parallel computing patterns in Python, simulating
the SIMT (Single Instruction, Multiple Threads) execution model:
  1. Parallel map (per-element image brightness)
  2. Parallel reduction (sum, min, max)
  3. Prefix sum (inclusive scan) -- the workhorse of GPU algorithms
  4. Parallel histogram (atomic-style bin counting)
  5. Image convolution (simulated workgroup with shared memory)
  6. Parallel bitonic sort

Each pattern is implemented twice: a naive sequential version and a
"GPU-style" version that uses numpy vectorization to mirror how thousands
of GPU threads would process the data in parallel.  Comments explain
the mapping to real compute shader concepts (work groups, invocations,
shared memory, barriers).

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------------------
# 1. Parallel Map -- per-element transformation
# ---------------------------------------------------------------------------


def gpu_map_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness using a parallel map pattern.

    On a real GPU, each thread processes one pixel independently.
    This is the simplest GPU pattern: no communication between threads,
    perfect parallelism, zero synchronization overhead.

    GLSL compute shader equivalent:
        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            vec4 pixel = imageLoad(img, coord);
            imageStore(img, coord, pixel * factor);
        }
    """
    # Vectorized = all "threads" execute simultaneously
    return np.clip(image * factor, 0, 1)


# ---------------------------------------------------------------------------
# 2. Parallel Reduction -- aggregate a large array to a single value
# ---------------------------------------------------------------------------


def sequential_reduction(data: np.ndarray) -> float:
    """Sequential sum -- O(n) serial steps."""
    total = 0.0
    for x in data:
        total += x
    return total


def gpu_style_reduction(data: np.ndarray) -> float:
    """Parallel reduction using the binary-tree pattern.

    Why a tree?  With n elements and n/2 threads, each step halves the
    problem size.  Total work is O(n), but the number of *steps* is
    O(log n).  A 1M-element reduction takes only 20 steps on a GPU
    instead of 1M sequential additions.

    Real GPU reduction:
      - Step 0: thread i adds element[2i] + element[2i+1]  (n/2 threads)
      - Step 1: thread i adds result[2i] + result[2i+1]    (n/4 threads)
      - ...until one value remains

    Each step requires a barrier (all threads must finish before the
    next step can read the results).
    """
    buf = data.astype(float).copy()
    n = len(buf)

    step = 0
    while n > 1:
        half = n // 2
        # Each "thread" i adds buf[2i] and buf[2i+1]
        buf[:half] = buf[:2 * half:2] + buf[1:2 * half:2]
        # Handle odd element
        if n % 2 == 1:
            buf[half] = buf[n - 1]
            n = half + 1
        else:
            n = half
        step += 1

    return buf[0]


# ---------------------------------------------------------------------------
# 3. Prefix Sum (Inclusive Scan) -- Blelloch algorithm
# ---------------------------------------------------------------------------


def gpu_style_prefix_sum(data: np.ndarray) -> np.ndarray:
    """Parallel prefix sum using the up-sweep / down-sweep pattern.

    Prefix sum is the "secret weapon" of GPU programming.  It enables
    parallel versions of algorithms that seem inherently sequential:
    stream compaction, radix sort, histogram equalization, etc.

    The Blelloch scan has two phases:
      Up-sweep (reduce):  build partial sums bottom-up
      Down-sweep:         distribute prefix sums top-down

    Total steps: 2 * log2(n), each step runs n/2 threads maximum.
    """
    n = len(data)
    # Pad to next power of 2
    size = 1
    while size < n:
        size *= 2
    buf = np.zeros(size, dtype=float)
    buf[:n] = data

    # Up-sweep (reduction phase)
    d = 1
    while d < size:
        # In parallel: for each i where (i+1) is a multiple of 2*d
        indices = np.arange(2 * d - 1, size, 2 * d)
        buf[indices] += buf[indices - d]
        d *= 2

    # Set last element to zero for exclusive scan
    buf[size - 1] = 0

    # Down-sweep phase
    d = size // 2
    while d >= 1:
        indices = np.arange(2 * d - 1, size, 2 * d)
        temp = buf[indices - d].copy()
        buf[indices - d] = buf[indices]
        buf[indices] += temp
        d //= 2

    # Convert exclusive scan to inclusive scan
    result = np.zeros(n, dtype=float)
    result[0] = data[0]
    result[1:] = buf[1:n] + data[1:]
    return result


# ---------------------------------------------------------------------------
# 4. Parallel Histogram
# ---------------------------------------------------------------------------


def gpu_style_histogram(data: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute a histogram using the parallel atomic-add pattern.

    On a GPU, each thread reads one element and does an atomic increment
    on the corresponding bin.  Atomic operations serialize conflicting
    writes but are still much faster than a sequential loop for large data.

    Optimization: real GPU code uses per-workgroup shared-memory histograms
    (no conflicts within a workgroup) then merges them at the end.
    """
    hist = np.zeros(bins, dtype=int)
    # Quantize to bin indices
    indices = np.clip((data * (bins - 1)).astype(int), 0, bins - 1)
    # numpy bincount simulates the parallel atomic adds
    counts = np.bincount(indices, minlength=bins)
    hist[:len(counts)] = counts[:bins]
    return hist


# ---------------------------------------------------------------------------
# 5. Image Convolution with Simulated Shared Memory
# ---------------------------------------------------------------------------


def gpu_style_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution simulating compute shader workgroups.

    In a real compute shader, each workgroup:
      1. Cooperatively loads a tile of the image into shared memory
         (faster than global memory, shared within the workgroup)
      2. Calls barrier() to ensure all loads complete
      3. Each thread reads from shared memory to compute its output pixel

    Why shared memory?  A 3x3 convolution reads 9 pixels per output pixel.
    With 16x16 threads, that's 16x16x9 = 2304 global reads.  With shared
    memory, we load an 18x18 tile (2304 values) once, then all 256 threads
    read from fast on-chip memory.
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad input (mirrors how shared memory tiles include halo pixels)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Simulate workgroup-based processing
    output = np.zeros_like(image)
    workgroup_size = 16  # Typical GPU workgroup dimension

    for wy in range(0, h, workgroup_size):
        for wx in range(0, w, workgroup_size):
            # --- Shared memory load (cooperative, all threads in workgroup) ---
            tile_h = min(workgroup_size + 2 * pad_h, h - wy + 2 * pad_h)
            tile_w = min(workgroup_size + 2 * pad_w, w - wx + 2 * pad_w)
            shared_mem = padded[wy:wy + tile_h, wx:wx + tile_w]

            # --- Barrier: all threads have finished loading ---

            # --- Each thread computes one output pixel from shared memory ---
            for ly in range(min(workgroup_size, h - wy)):
                for lx in range(min(workgroup_size, w - wx)):
                    region = shared_mem[ly:ly + kh, lx:lx + kw]
                    if region.shape == kernel.shape:
                        output[wy + ly, wx + lx] = np.sum(region * kernel)

    return np.clip(output, 0, 1)


# ---------------------------------------------------------------------------
# 6. Parallel Bitonic Sort
# ---------------------------------------------------------------------------


def gpu_style_bitonic_sort(data: np.ndarray) -> np.ndarray:
    """Bitonic sort -- a comparison network ideal for GPU execution.

    Why bitonic sort on GPUs?  It has a fixed compare-and-swap pattern
    that doesn't depend on the data.  Every thread knows exactly which
    two elements to compare at each step -- no branching, no data-
    dependent memory access.  This predictability is perfect for SIMT.

    Complexity: O(n * log^2(n)) comparisons, O(log^2(n)) parallel steps.
    """
    n = len(data)
    # Pad to next power of 2
    size = 1
    while size < n:
        size *= 2
    arr = np.full(size, np.inf)
    arr[:n] = data

    # Bitonic sort network
    k = 2
    while k <= size:
        j = k // 2
        while j >= 1:
            # All compare-and-swap operations at this step run in parallel
            for i in range(size):
                partner = i ^ j  # XOR gives the partner index
                if partner > i:
                    # Ascending if in the first half of the bitonic sequence
                    ascending = (i & k) == 0
                    if ascending:
                        if arr[i] > arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
                    else:
                        if arr[i] < arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
            j //= 2
        k *= 2

    return arr[:n]


# ---------------------------------------------------------------------------
# 7. Demonstrations
# ---------------------------------------------------------------------------


def demo_reduction():
    """Compare sequential vs parallel reduction."""
    sizes = [100, 1000, 10000, 100000]
    print("\n  Parallel Reduction (sum):")
    print(f"  {'Size':>8s}  {'Sequential':>12s}  {'GPU-style':>12s}  {'Match':>6s}")

    for n in sizes:
        data = np.random.rand(n).astype(float)
        seq_result = sequential_reduction(data)
        par_result = gpu_style_reduction(data)
        match = abs(seq_result - par_result) < 1e-6
        print(f"  {n:>8d}  {seq_result:>12.4f}  {par_result:>12.4f}  {'OK' if match else 'FAIL':>6s}")


def demo_prefix_sum():
    """Visualize prefix sum computation."""
    data = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=float)
    prefix = gpu_style_prefix_sum(data)
    expected = np.cumsum(data)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(data))
    bar_w = 0.35
    ax.bar(x - bar_w / 2, data, bar_w, label='Input', color='steelblue')
    ax.bar(x + bar_w / 2, prefix, bar_w, label='Prefix Sum', color='coral')
    ax.set_xticks(x)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("Parallel Prefix Sum (Blelloch Scan)", fontsize=12)
    ax.legend()

    match = np.allclose(prefix, expected)
    ax.text(0.98, 0.95, f"Correct: {match}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen' if match else 'salmon'))

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_14_prefix_sum.png", dpi=100)
    plt.show()


def demo_convolution():
    """Apply GPU-style convolution filters to a procedural image."""
    # Generate a procedural test image (checkerboard + circle)
    size = 128
    y, x = np.mgrid[0:size, 0:size]
    checker = ((x // 16 + y // 16) % 2).astype(float) * 0.6 + 0.2
    circle = ((x - 64) ** 2 + (y - 64) ** 2 < 30 ** 2).astype(float) * 0.8
    image = np.clip(checker + circle, 0, 1)

    # Define kernels
    blur_kernel = np.ones((5, 5)) / 25.0
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)

    blurred = gpu_style_convolution(image, blur_kernel)
    sharpened = gpu_style_convolution(image, sharpen_kernel)
    edges = gpu_style_convolution(image, edge_kernel)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Original', 'Box Blur 5x5', 'Sharpen 3x3', 'Edge Detect 3x3']
    images = [image, blurred, sharpened, edges]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    fig.suptitle("GPU-Style Image Convolution (Workgroup + Shared Memory)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_14_convolution.png", dpi=100)
    plt.show()


def demo_bitonic_sort():
    """Visualize bitonic sort on a small dataset."""
    np.random.seed(42)
    data = np.random.rand(16) * 100

    sorted_data = gpu_style_bitonic_sort(data.copy())
    expected = np.sort(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(len(data)), data, color='steelblue')
    ax1.set_title("Before Sort", fontsize=11)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")

    ax2.bar(range(len(sorted_data)), sorted_data, color='coral')
    ax2.set_title("After Bitonic Sort", fontsize=11)
    ax2.set_xlabel("Index")

    match = np.allclose(sorted_data, expected)
    fig.suptitle(f"Parallel Bitonic Sort (Correct: {match})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_14_bitonic_sort.png", dpi=100)
    plt.show()


def demo_histogram():
    """Compute and display a parallel histogram."""
    np.random.seed(7)
    # Simulate a grayscale image
    image = np.clip(np.random.normal(0.5, 0.15, size=(256, 256)), 0, 1)
    hist = gpu_style_histogram(image.ravel(), bins=64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax1.set_title("Input Image (random noise)", fontsize=11)
    ax1.axis('off')

    ax2.bar(range(64), hist, color='steelblue', width=1.0)
    ax2.set_title("Parallel Histogram (64 bins)", fontsize=11)
    ax2.set_xlabel("Bin")
    ax2.set_ylabel("Count")

    fig.suptitle("GPU-Style Parallel Histogram", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_14_histogram.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GPU Computing Patterns (CPU Simulation)")
    print("=" * 60)

    print("\n[1/5] Parallel reduction...")
    demo_reduction()

    print("\n[2/5] Prefix sum (Blelloch scan)...")
    demo_prefix_sum()

    print("\n[3/5] Image convolution with shared memory...")
    demo_convolution()

    print("\n[4/5] Bitonic sort...")
    demo_bitonic_sort()

    print("\n[5/5] Parallel histogram...")
    demo_histogram()

    print("\nDone!")


if __name__ == "__main__":
    main()
