"""
Exercises for Lesson 15: Image Signal Processing
Topic: Signal_Processing

Solutions to practice problems from the lesson.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# === Exercise 1: 2D DFT Properties ===
# Problem: Create a 256x256 image with centered white rectangle, compute
#          2D DFT, verify shift theorem, rotation property, magnitude vs phase.

def exercise_1():
    """2D DFT properties: sinc pattern, shift theorem, rotation, phase importance."""
    N = 256

    # (a) White rectangle at center, compute 2D DFT magnitude spectrum
    img_a = np.zeros((N, N))
    rect_h, rect_w = 64, 32
    r0, c0 = N // 2 - rect_h // 2, N // 2 - rect_w // 2
    img_a[r0:r0 + rect_h, c0:c0 + rect_w] = 1.0

    F_a = fftshift(fft2(img_a))
    mag_a = np.log1p(np.abs(F_a))
    phase_a = np.angle(F_a)

    print("(a) 2D DFT of centered rectangle")
    print(f"    Rectangle size: {rect_h}x{rect_w} in {N}x{N} image")
    print("    Magnitude spectrum shows 2D sinc pattern:")
    print("    - Narrow in horizontal freq (wide rectangle in x)")
    print("    - Wide in vertical freq (tall rectangle in y)")
    print("    This is because DFT of rect = sinc, and wider spatial extent")
    print("    gives narrower frequency extent (uncertainty principle).")

    # (b) Shift rectangle 50 pixels right, verify shift theorem
    img_b = np.zeros((N, N))
    shift = 50
    img_b[r0:r0 + rect_h, c0 + shift:c0 + shift + rect_w] = 1.0

    F_b = fftshift(fft2(img_b))
    mag_b = np.log1p(np.abs(F_b))
    phase_b = np.angle(F_b)

    # Shift theorem: magnitude stays the same, phase changes linearly
    mag_diff = np.max(np.abs(np.abs(F_a) - np.abs(F_b)))
    print(f"\n(b) Shift theorem verification (shift = {shift} pixels right)")
    print(f"    Max magnitude difference: {mag_diff:.6f}")
    print(f"    Magnitude unchanged: {mag_diff < 1e-10}")
    print("    Phase changes: spatial shift adds linear phase in frequency domain")
    print(f"    Phase difference is NOT zero: {not np.allclose(phase_a, phase_b)}")

    # (c) Rotate rectangle by 45 degrees
    from scipy.ndimage import rotate as ndrotate
    img_c = ndrotate(img_a, 45, reshape=False, order=1)

    F_c = fftshift(fft2(img_c))
    mag_c = np.log1p(np.abs(F_c))

    print("\n(c) Rotation property")
    print("    Rotation in spatial domain -> same rotation in frequency domain")
    print("    The 2D sinc pattern rotates by 45 degrees in the magnitude spectrum")

    # (d) Swap magnitude and phase
    # Create a second image: a Gaussian blob
    yy, xx = np.meshgrid(np.arange(N), np.arange(N))
    img_gauss = np.exp(-((xx - N / 2) ** 2 + (yy - N / 2) ** 2) / (2 * 30 ** 2))

    F_gauss = fftshift(fft2(img_gauss))

    # Combine magnitude of rect with phase of Gaussian
    combined_mag_rect = np.abs(F_a) * np.exp(1j * np.angle(F_gauss))
    recon_mag_rect = np.real(ifft2(ifftshift(combined_mag_rect)))

    # Combine magnitude of Gaussian with phase of rect
    combined_mag_gauss = np.abs(F_gauss) * np.exp(1j * np.angle(F_a))
    recon_mag_gauss = np.real(ifft2(ifftshift(combined_mag_gauss)))

    print("\n(d) Magnitude vs Phase importance")
    print("    Swapping magnitude of rect with phase of Gaussian:")
    print("    Result looks more like the Gaussian (phase donor)")
    print("    Swapping magnitude of Gaussian with phase of rect:")
    print("    Result looks more like the rectangle (phase donor)")
    print("    Conclusion: PHASE carries more structural/visual information")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img_a, cmap='gray')
    axes[0, 0].set_title('Rectangle')
    axes[0, 1].imshow(mag_a, cmap='gray')
    axes[0, 1].set_title('Magnitude (log)')
    axes[0, 2].imshow(img_b, cmap='gray')
    axes[0, 2].set_title(f'Shifted by {shift}px')
    axes[0, 3].imshow(mag_b, cmap='gray')
    axes[0, 3].set_title('Shifted Magnitude (log)')
    axes[1, 0].imshow(img_c, cmap='gray')
    axes[1, 0].set_title('Rotated 45°')
    axes[1, 1].imshow(mag_c, cmap='gray')
    axes[1, 1].set_title('Rotated Magnitude (log)')
    axes[1, 2].imshow(recon_mag_rect, cmap='gray')
    axes[1, 2].set_title('Rect mag + Gauss phase')
    axes[1, 3].imshow(recon_mag_gauss, cmap='gray')
    axes[1, 3].set_title('Gauss mag + Rect phase')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_1_2d_dft_properties.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex15_1_2d_dft_properties.png")


# === Exercise 2: Smoothing Filter Comparison ===
# Problem: Compare box, Gaussian, and median filters on Gaussian and
#          salt-and-pepper noise. Measure PSNR and edge preservation.

def exercise_2():
    """Smoothing filter comparison: box, Gaussian, median on different noise types."""
    np.random.seed(42)
    N = 256

    # Create test image: gradient with edges
    clean = np.zeros((N, N))
    # Horizontal gradient region
    clean[:N // 2, :] = np.tile(np.linspace(0, 200, N), (N // 2, 1))
    # Vertical edges
    clean[N // 2:, :N // 3] = 50
    clean[N // 2:, N // 3:2 * N // 3] = 150
    clean[N // 2:, 2 * N // 3:] = 250
    clean = clean.astype(np.float64)

    def psnr(original, filtered):
        mse = np.mean((original - filtered) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255.0 ** 2 / mse)

    def gradient_energy(img):
        gx = np.diff(img, axis=1)
        gy = np.diff(img, axis=0)
        return np.sum(gx ** 2) + np.sum(gy ** 2)

    # (a) Additive Gaussian noise
    noisy_gauss = clean + np.random.normal(0, 30, clean.shape)
    noisy_gauss = np.clip(noisy_gauss, 0, 255)

    print("(a) Box filters on Gaussian noise (sigma=30):")
    for ks in [3, 5, 7]:
        filtered = uniform_filter(noisy_gauss, size=ks)
        p = psnr(clean, filtered)
        print(f"    Box {ks}x{ks}: PSNR = {p:.2f} dB")

    # (b) Gaussian filters
    print("\n(b) Gaussian filters on Gaussian noise:")
    for sigma in [1, 2, 3]:
        filtered = gaussian_filter(noisy_gauss, sigma=sigma)
        p = psnr(clean, filtered)
        print(f"    Gaussian sigma={sigma}: PSNR = {p:.2f} dB")

    # Salt-and-pepper noise
    noisy_sp = clean.copy()
    sp_mask = np.random.random(clean.shape)
    noisy_sp[sp_mask < 0.025] = 0
    noisy_sp[sp_mask > 0.975] = 255

    # (c) Median filters
    print("\n(c) Median filters:")
    for ks in [3, 5]:
        filt_gauss = median_filter(noisy_gauss, size=ks)
        filt_sp = median_filter(noisy_sp, size=ks)
        p_gauss = psnr(clean, filt_gauss)
        p_sp = psnr(clean, filt_sp)
        print(f"    Median {ks}x{ks} on Gaussian noise: PSNR = {p_gauss:.2f} dB")
        print(f"    Median {ks}x{ks} on S&P noise:      PSNR = {p_sp:.2f} dB")
    print("    Median filter excels at removing salt-and-pepper noise")

    # (d) Edge preservation metric
    ge_clean = gradient_energy(clean)
    print(f"\n(d) Edge preservation (gradient energy ratio, higher = better):")
    print(f"    Clean image gradient energy: {ge_clean:.0f}")

    filters = {
        'Box 3x3': uniform_filter(noisy_gauss, 3),
        'Box 5x5': uniform_filter(noisy_gauss, 5),
        'Gauss s=1': gaussian_filter(noisy_gauss, 1),
        'Gauss s=2': gaussian_filter(noisy_gauss, 2),
        'Median 3x3': median_filter(noisy_gauss, 3),
        'Median 5x5': median_filter(noisy_gauss, 5),
    }
    for name, filt in filters.items():
        ratio = gradient_energy(filt) / ge_clean
        p = psnr(clean, filt)
        print(f"    {name:12s}: edge ratio = {ratio:.3f}, PSNR = {p:.2f} dB")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(clean, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Clean')
    axes[0, 1].imshow(noisy_gauss, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Gaussian Noise')
    axes[0, 2].imshow(uniform_filter(noisy_gauss, 5), cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('Box 5x5')
    axes[0, 3].imshow(gaussian_filter(noisy_gauss, 2), cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title('Gaussian s=2')
    axes[1, 0].imshow(noisy_sp, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('S&P Noise')
    axes[1, 1].imshow(median_filter(noisy_sp, 3), cmap='gray', vmin=0, vmax=255)
    axes[1, 1].set_title('Median 3x3 (S&P)')
    axes[1, 2].imshow(median_filter(noisy_sp, 5), cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title('Median 5x5 (S&P)')
    axes[1, 3].imshow(uniform_filter(noisy_sp, 5), cmap='gray', vmin=0, vmax=255)
    axes[1, 3].set_title('Box 5x5 (S&P)')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_2_smoothing_comparison.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex15_2_smoothing_comparison.png")


# === Exercise 3: Frequency Domain Design ===
# Problem: Ideal bandpass, notch filter for periodic noise, homomorphic filter.

def exercise_3():
    """Frequency domain filter design: bandpass, notch, homomorphic."""
    np.random.seed(42)
    N = 256

    # (a) Ideal bandpass filter
    print("(a) Ideal bandpass filter (D1=20, D2=60)")

    # Create test image with various frequency content
    yy, xx = np.meshgrid(np.arange(N), np.arange(N))
    img = (np.sin(2 * np.pi * xx * 5 / N) +
           np.sin(2 * np.pi * yy * 30 / N) +
           np.sin(2 * np.pi * (xx + yy) * 60 / N))
    img = (img - img.min()) / (img.max() - img.min()) * 255

    F = fftshift(fft2(img))
    cx, cy = N // 2, N // 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    D1, D2 = 20, 60
    bandpass = ((dist >= D1) & (dist <= D2)).astype(float)
    F_bp = F * bandpass
    img_bp = np.real(ifft2(ifftshift(F_bp)))

    print(f"    Passband: {D1} to {D2} pixels from center")
    print(f"    Only mid-frequency content preserved")

    # (b) Notch filter for periodic stripes
    print("\n(b) Notch filter for periodic stripe removal")

    base_img = np.outer(
        np.linspace(0, 1, N),
        np.linspace(0, 1, N)
    ) * 200
    stripe_freq = 20  # pixels per cycle
    stripes = 30 * np.sin(2 * np.pi * np.arange(N) / stripe_freq)
    noisy_img = base_img + stripes[np.newaxis, :]

    F_noisy = fftshift(fft2(noisy_img))

    # Design notch filter: suppress the stripe frequency peaks
    notch = np.ones((N, N))
    # Stripe is vertical -> horizontal frequency peak
    freq_loc = int(N * 1.0 / stripe_freq)
    notch_radius = 3
    for dx in range(-notch_radius, notch_radius + 1):
        for dy in range(-notch_radius, notch_radius + 1):
            if dx * dx + dy * dy <= notch_radius * notch_radius:
                # Two symmetric peaks
                nr = cx + dx
                nc1 = cy + freq_loc + dy
                nc2 = cy - freq_loc + dy
                if 0 <= nr < N:
                    if 0 <= nc1 < N:
                        notch[nr, nc1] = 0
                    if 0 <= nc2 < N:
                        notch[nr, nc2] = 0

    F_clean = F_noisy * notch
    cleaned = np.real(ifft2(ifftshift(F_clean)))

    residual_stripes = np.std(cleaned - base_img)
    print(f"    Stripe frequency: 1/{stripe_freq} cycles/pixel")
    print(f"    Residual noise std after notch: {residual_stripes:.2f}")

    # (c) Homomorphic filter
    print("\n(c) Homomorphic filter for contrast enhancement")

    # Create image with non-uniform illumination
    illumination = 50 + 150 * np.outer(
        np.sin(np.linspace(0, np.pi, N)),
        np.sin(np.linspace(0, np.pi, N))
    )
    reflectance = np.zeros((N, N))
    reflectance[50:100, 50:100] = 0.8
    reflectance[100:200, 120:220] = 0.6
    reflectance[30:60, 160:200] = 0.9
    reflectance[reflectance == 0] = 0.3

    homo_img = illumination * reflectance
    homo_img = np.clip(homo_img, 1, 255)  # avoid log(0)

    # Homomorphic filtering
    log_img = np.log(homo_img + 1)
    F_log = fftshift(fft2(log_img))

    # High-pass filter (Butterworth)
    gamma_h, gamma_l = 2.0, 0.5
    D0 = 30
    H_homo = gamma_l + (gamma_h - gamma_l) * (1 - 1 / (1 + (dist / D0) ** 4))

    F_filtered = F_log * H_homo
    filtered_log = np.real(ifft2(ifftshift(F_filtered)))
    homo_result = np.exp(filtered_log) - 1

    print(f"    gamma_L={gamma_l}, gamma_H={gamma_h}, D0={D0}")
    print(f"    Homomorphic filter normalizes illumination and enhances reflectance")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Multi-frequency image')
    axes[0, 1].imshow(img_bp, cmap='gray')
    axes[0, 1].set_title(f'Bandpass D1={D1}, D2={D2}')
    axes[0, 2].imshow(np.log1p(np.abs(F)) * bandpass, cmap='gray')
    axes[0, 2].set_title('Bandpass mask in freq')
    axes[1, 0].imshow(noisy_img, cmap='gray')
    axes[1, 0].set_title('Periodic stripes')
    axes[1, 1].imshow(cleaned, cmap='gray')
    axes[1, 1].set_title('After notch filter')
    axes[1, 2].imshow(homo_result, cmap='gray')
    axes[1, 2].set_title('Homomorphic result')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_3_frequency_domain.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex15_3_frequency_domain.png")


# === Exercise 4: Edge Detection Comparison ===
# Problem: Compare Sobel, Prewitt, Scharr, Laplacian, LoG, Canny.

def exercise_4():
    """Edge detection comparison with quantitative evaluation."""
    np.random.seed(42)
    N = 256

    # Create test image with strong edges, weak edges, texture, noise
    clean = np.zeros((N, N))
    # Strong edge: bright square
    clean[50:120, 50:120] = 200
    # Weak edge: slightly different gray
    clean[50:120, 140:210] = 80
    clean[50:120, 210:230] = 100
    # Texture region
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    clean[150:230, 50:180] = 100 + 30 * np.sin(2 * np.pi * xx[150:230, 50:180] / 8)
    # Gradient region
    clean[150:230, 190:250] = np.tile(np.linspace(50, 200, 60), (80, 1))

    # Ground truth edges (boundaries of regions)
    gt_edges = np.zeros((N, N), dtype=bool)
    for region in [(50, 120, 50, 120), (50, 120, 140, 210),
                   (50, 120, 210, 230), (150, 230, 50, 180), (150, 230, 190, 250)]:
        r0, r1, c0, c1 = region
        gt_edges[r0, c0:c1] = True
        gt_edges[r1 - 1, c0:c1] = True
        gt_edges[r0:r1, c0] = True
        gt_edges[r0:r1, c1 - 1] = True
    # Dilate ground truth slightly for tolerance
    from scipy.ndimage import binary_dilation
    gt_dilated = binary_dilation(gt_edges, iterations=2)

    noisy = clean + np.random.normal(0, 10, clean.shape)

    # (a) Sobel, Prewitt, Scharr
    print("(a) Gradient operators:")
    operators = {}

    # Sobel
    sx = sig.convolve2d(noisy, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                        mode='same', boundary='symm')
    sy = sig.convolve2d(noisy, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                        mode='same', boundary='symm')
    operators['Sobel'] = np.sqrt(sx ** 2 + sy ** 2)

    # Prewitt
    px = sig.convolve2d(noisy, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                        mode='same', boundary='symm')
    py = sig.convolve2d(noisy, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
                        mode='same', boundary='symm')
    operators['Prewitt'] = np.sqrt(px ** 2 + py ** 2)

    # Scharr
    scx = sig.convolve2d(noisy, np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
                         mode='same', boundary='symm')
    scy = sig.convolve2d(noisy, np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]),
                         mode='same', boundary='symm')
    operators['Scharr'] = np.sqrt(scx ** 2 + scy ** 2)

    for name, grad_mag in operators.items():
        threshold = np.percentile(grad_mag, 90)
        edge_map = grad_mag > threshold
        # Precision-recall against ground truth
        tp = np.sum(edge_map & gt_dilated)
        fp = np.sum(edge_map & ~gt_dilated)
        fn = np.sum(~edge_map & gt_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"    {name}: max gradient = {grad_mag.max():.1f}, "
              f"precision = {precision:.3f}, recall = {recall:.3f}")

    # (b) Laplacian and LoG
    print("\n(b) Laplacian and LoG:")
    lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap = sig.convolve2d(noisy, lap_kernel, mode='same', boundary='symm')

    for sigma_val in [1, 2, 3]:
        # LoG: Gaussian smooth then Laplacian
        smoothed = gaussian_filter(noisy, sigma=sigma_val)
        log_result = sig.convolve2d(smoothed, lap_kernel, mode='same', boundary='symm')
        # Zero crossings
        zc = np.zeros_like(log_result, dtype=bool)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                neighbors = [log_result[i - 1, j], log_result[i + 1, j],
                             log_result[i, j - 1], log_result[i, j + 1]]
                if any(n * log_result[i, j] < 0 for n in neighbors):
                    zc[i, j] = True
        tp = np.sum(zc & gt_dilated)
        total_detected = np.sum(zc)
        print(f"    LoG sigma={sigma_val}: zero crossings = {total_detected}, "
              f"near true edges = {tp}")

    # (c) Canny edge detector (full implementation)
    print("\n(c) Canny edge detector implementation:")

    def canny_detector(image, sigma_c=1.0, low_thresh=20, high_thresh=50):
        """Full Canny edge detection: smooth, gradient, NMS, hysteresis."""
        # Step 1: Gaussian smoothing
        smoothed = gaussian_filter(image, sigma=sigma_c)

        # Step 2: Gradient magnitude and direction (Sobel)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = sig.convolve2d(smoothed, kx, mode='same', boundary='symm')
        gy = sig.convolve2d(smoothed, ky, mode='same', boundary='symm')
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.arctan2(gy, gx) * 180 / np.pi

        # Step 3: Non-maximum suppression
        nms = np.zeros_like(magnitude)
        direction = direction % 180  # map to [0, 180)
        rows, cols = magnitude.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = direction[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                    n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
                elif 22.5 <= angle < 67.5:
                    n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
                elif 67.5 <= angle < 112.5:
                    n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
                else:
                    n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
                if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                    nms[i, j] = magnitude[i, j]

        # Step 4-5: Double threshold + hysteresis
        strong = nms >= high_thresh
        weak = (nms >= low_thresh) & (nms < high_thresh)
        edges = strong.copy()

        # Hysteresis: connect weak edges to strong edges
        changed = True
        while changed:
            changed = False
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if weak[i, j] and not edges[i, j]:
                        if np.any(edges[i - 1:i + 2, j - 1:j + 2]):
                            edges[i, j] = True
                            changed = True
        return edges

    for sigma_c in [1, 2]:
        for low, high in [(10, 30), (20, 50), (30, 80)]:
            edges = canny_detector(noisy, sigma_c=sigma_c, low_thresh=low, high_thresh=high)
            tp = np.sum(edges & gt_dilated)
            total = np.sum(edges)
            precision = tp / total if total > 0 else 0
            print(f"    Canny(s={sigma_c}, T=[{low},{high}]): "
                  f"edges={total}, precision={precision:.3f}")

    # (d) Precision-recall curves
    print("\n(d) Precision-recall comparison:")
    print("    Varying threshold to generate precision-recall points...")
    sobel_mag = operators['Sobel']
    thresholds = np.percentile(sobel_mag, np.linspace(50, 99, 20))
    precisions, recalls = [], []
    for t in thresholds:
        edge_map = sobel_mag > t
        tp = np.sum(edge_map & gt_dilated)
        fp = np.sum(edge_map & ~gt_dilated)
        fn = np.sum(~edge_map & gt_edges)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(p)
        recalls.append(r)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[1].imshow(operators['Sobel'], cmap='hot')
    axes[1].set_title('Sobel Magnitude')
    axes[2].imshow(canny_detector(noisy, 1.5, 20, 50), cmap='gray')
    axes[2].set_title('Canny Edges')
    axes[3].plot(recalls, precisions, 'b-o', markersize=3)
    axes[3].set_xlabel('Recall')
    axes[3].set_ylabel('Precision')
    axes[3].set_title('P-R Curve (Sobel)')
    axes[3].grid(True, alpha=0.3)
    for ax in axes[:3]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_4_edge_detection.png', dpi=150)
    plt.close()
    print("    Plot saved: ex15_4_edge_detection.png")


# === Exercise 5: Histogram Processing ===
# Problem: Histogram equalization, CLAHE, histogram matching/specification.

def exercise_5():
    """Histogram processing: equalization, CLAHE, histogram matching."""
    np.random.seed(42)
    N = 256

    # Create test image with structure
    base = np.zeros((N, N))
    base[30:100, 30:100] = 0.3
    base[60:200, 120:230] = 0.6
    base[150:240, 30:120] = 0.45
    base += 0.1 * np.random.randn(N, N)
    base = np.clip(base, 0, 1)

    # (a) Dark and bright images + histogram equalization
    print("(a) Histogram equalization on dark and bright images:")
    dark = np.clip(base * 0.2, 0, 1)  # concentrated in [0, 0.2]
    bright = np.clip(0.8 + base * 0.2, 0, 1)  # concentrated in [0.8, 1.0]

    def hist_equalize(img):
        """Global histogram equalization for float [0,1] image."""
        # Quantize to 256 levels
        img_uint8 = (img * 255).astype(np.uint8)
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 255))
        cdf = np.cumsum(hist).astype(np.float64)
        cdf_min = cdf[cdf > 0].min()
        total = img_uint8.size
        # Mapping
        lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
        return lut[img_uint8] / 255.0

    dark_eq = hist_equalize(dark)
    bright_eq = hist_equalize(bright)

    print(f"    Dark image range: [{dark.min():.3f}, {dark.max():.3f}]")
    print(f"    After equalization: [{dark_eq.min():.3f}, {dark_eq.max():.3f}]")
    print(f"    Bright image range: [{bright.min():.3f}, {bright.max():.3f}]")
    print(f"    After equalization: [{bright_eq.min():.3f}, {bright_eq.max():.3f}]")

    # (b) CLAHE (Contrast Limited Adaptive Histogram Equalization)
    print("\n(b) CLAHE implementation:")

    def clahe(img, tile_size=32, clip_limit=2.0):
        """Simple CLAHE: tile-based equalization with clip limit and interpolation."""
        img_uint8 = (img * 255).astype(np.uint8)
        rows, cols = img_uint8.shape
        n_tiles_r = rows // tile_size
        n_tiles_c = cols // tile_size

        # Compute clipped CDF for each tile
        cdfs = np.zeros((n_tiles_r, n_tiles_c, 256))
        for tr in range(n_tiles_r):
            for tc in range(n_tiles_c):
                r0 = tr * tile_size
                c0 = tc * tile_size
                tile = img_uint8[r0:r0 + tile_size, c0:c0 + tile_size]
                hist, _ = np.histogram(tile, bins=256, range=(0, 255))
                # Clip histogram
                clip_val = clip_limit * tile.size / 256
                excess = np.sum(np.maximum(hist - clip_val, 0))
                hist = np.minimum(hist, clip_val)
                hist += int(excess / 256)
                cdf = np.cumsum(hist).astype(np.float64)
                cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
                cdfs[tr, tc] = cdf

        # Bilinear interpolation between tiles
        result = np.zeros_like(img_uint8, dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                # Find surrounding tile centers
                tr = min(max((i - tile_size // 2) / tile_size, 0), n_tiles_r - 1)
                tc = min(max((j - tile_size // 2) / tile_size, 0), n_tiles_c - 1)
                tr0 = int(np.floor(tr))
                tc0 = int(np.floor(tc))
                tr1 = min(tr0 + 1, n_tiles_r - 1)
                tc1 = min(tc0 + 1, n_tiles_c - 1)
                dr = tr - tr0
                dc = tc - tc0
                val = img_uint8[i, j]
                # Bilinear interpolation of CDF values
                result[i, j] = ((1 - dr) * (1 - dc) * cdfs[tr0, tc0, val] +
                                (1 - dr) * dc * cdfs[tr0, tc1, val] +
                                dr * (1 - dc) * cdfs[tr1, tc0, val] +
                                dr * dc * cdfs[tr1, tc1, val])

        return result / 255.0

    # Use on dark image (small version for speed)
    small_dark = dark[:128, :128]
    clahe_result = clahe(small_dark, tile_size=32, clip_limit=2.0)
    global_eq = hist_equalize(small_dark)

    print(f"    CLAHE tile_size=32, clip_limit=2.0")
    print(f"    CLAHE output range: [{clahe_result.min():.3f}, {clahe_result.max():.3f}]")
    print(f"    CLAHE preserves local contrast better than global equalization")

    # (c) Histogram matching (specification)
    print("\n(c) Histogram matching:")

    def hist_match(source, target_hist):
        """Match source image histogram to target histogram."""
        src_uint8 = (source * 255).astype(np.uint8)
        # Source CDF
        src_hist, _ = np.histogram(src_uint8, bins=256, range=(0, 255))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        # Target CDF
        target_cdf = np.cumsum(target_hist).astype(np.float64)
        target_cdf /= target_cdf[-1]
        # Mapping: for each source level, find closest target CDF value
        lut = np.zeros(256, dtype=np.uint8)
        for s in range(256):
            idx = np.argmin(np.abs(target_cdf - src_cdf[s]))
            lut[s] = idx
        return lut[src_uint8] / 255.0

    # Target: bimodal histogram
    target = np.zeros(256)
    target[50:100] = 1.0
    target[180:230] = 1.0
    target /= target.sum()

    matched = hist_match(dark, target)
    match_hist, _ = np.histogram((matched * 255).astype(np.uint8), bins=256, range=(0, 255))

    print(f"    Target: bimodal histogram (peaks at 75 and 205)")
    print(f"    Matched image range: [{matched.min():.3f}, {matched.max():.3f}]")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(dark, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Dark Image')
    axes[0, 1].imshow(dark_eq, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Equalized')
    axes[0, 2].imshow(matched, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Histogram Matched')
    axes[1, 0].hist(dark.ravel(), bins=64, color='gray')
    axes[1, 0].set_title('Dark Histogram')
    axes[1, 1].hist(dark_eq.ravel(), bins=64, color='gray')
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 2].hist(matched.ravel(), bins=64, color='gray')
    axes[1, 2].set_title('Matched Histogram')
    plt.tight_layout()
    plt.savefig('ex15_5_histogram_processing.png', dpi=150)
    plt.close()
    print("    Plot saved: ex15_5_histogram_processing.png")


# === Exercise 6: JPEG Compression Analysis ===
# Problem: Implement JPEG simulator (block DCT, quantization, zigzag, RLE),
#          rate-distortion, deblocking, DCT vs wavelet comparison.

def exercise_6():
    """JPEG compression analysis: block DCT, rate-distortion, deblocking."""
    np.random.seed(42)
    N = 256

    # Create test image
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    img = (128 + 60 * np.sin(2 * np.pi * xx / 40) *
           np.cos(2 * np.pi * yy / 60) +
           40 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 50 ** 2)))
    img = np.clip(img, 0, 255).astype(np.float64)

    # Standard JPEG luminance quantization matrix
    Q_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.float64)

    def quality_to_qmatrix(quality):
        """Convert quality factor (1-100) to quantization matrix."""
        if quality < 50:
            s = 5000 / quality
        else:
            s = 200 - 2 * quality
        Q = np.floor((Q_base * s + 50) / 100)
        Q[Q < 1] = 1
        return Q

    def zigzag_order(n=8):
        """Generate zigzag scan order for nxn block."""
        order = []
        for s in range(2 * n - 1):
            if s % 2 == 0:
                for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                    order.append((i, s - i))
            else:
                for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                    order.append((i, s - i))
        return order

    def jpeg_compress_decompress(image, quality):
        """Simulate JPEG compression and decompression."""
        Q = quality_to_qmatrix(quality)
        rows, cols = image.shape
        recon = np.zeros_like(image)
        total_nonzero = 0
        total_coeffs = 0
        zig = zigzag_order()

        for i in range(0, rows - 7, 8):
            for j in range(0, cols - 7, 8):
                block = image[i:i + 8, j:j + 8] - 128  # level shift
                # Forward DCT (using scipy)
                from scipy.fft import dctn, idctn
                dct_block = dctn(block, type=2, norm='ortho')
                # Quantize
                quant = np.round(dct_block / Q)
                total_nonzero += np.count_nonzero(quant)
                total_coeffs += 64
                # Dequantize
                dequant = quant * Q
                # Inverse DCT
                recon[i:i + 8, j:j + 8] = idctn(dequant, type=2, norm='ortho') + 128

        compression_ratio = total_coeffs / max(total_nonzero, 1)
        return recon, compression_ratio

    # (a) JPEG at various quality levels
    print("(a) JPEG compression at different quality factors:")
    qualities = [10, 30, 50, 70, 90]
    results = {}
    for q in qualities:
        recon, cr = jpeg_compress_decompress(img, q)
        mse = np.mean((img - recon) ** 2)
        p = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')
        results[q] = (recon, cr, p)
        print(f"    Quality {q:2d}: PSNR = {p:.2f} dB, CR ~= {cr:.1f}:1")

    # (b) Rate-distortion curve
    print("\n(b) Rate-distortion curve:")
    rd_q = list(range(5, 100, 5))
    rd_psnr = []
    rd_cr = []
    for q in rd_q:
        recon, cr = jpeg_compress_decompress(img, q)
        mse = np.mean((img - recon) ** 2)
        p = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 60
        rd_psnr.append(p)
        rd_cr.append(cr)
    print(f"    Quality range: {rd_q[0]}-{rd_q[-1]}")
    print(f"    PSNR range: {min(rd_psnr):.1f} - {max(rd_psnr):.1f} dB")
    print(f"    CR range: {min(rd_cr):.1f} - {max(rd_cr):.1f}")

    # (c) Deblocking filter
    print("\n(c) Deblocking filter for low quality:")
    low_q_recon = results[10][0]

    def deblock_filter(image, block_size=8, strength=0.5):
        """Simple deblocking: lowpass across block boundaries."""
        result = image.copy()
        kernel = np.array([1, 2, 1]) / 4.0
        # Horizontal block boundaries
        for i in range(block_size, image.shape[0], block_size):
            if i >= 1 and i < image.shape[0] - 1:
                for j in range(image.shape[1]):
                    strip = image[i - 1:i + 2, j]
                    result[i, j] = (1 - strength) * image[i, j] + strength * np.dot(kernel, strip)
        # Vertical block boundaries
        for j in range(block_size, image.shape[1], block_size):
            if j >= 1 and j < image.shape[1] - 1:
                for i in range(image.shape[0]):
                    strip = result[i, j - 1:j + 2]
                    if len(strip) == 3:
                        result[i, j] = (1 - strength) * result[i, j] + strength * np.dot(kernel, strip)
        return result

    deblocked = deblock_filter(low_q_recon)
    mse_before = np.mean((img - low_q_recon) ** 2)
    mse_after = np.mean((img - deblocked) ** 2)
    print(f"    Q=10 PSNR before deblocking: {10 * np.log10(255 ** 2 / mse_before):.2f} dB")
    print(f"    Q=10 PSNR after deblocking:  {10 * np.log10(255 ** 2 / mse_after):.2f} dB")

    # (d) DCT vs Wavelet compression (using simple Haar wavelet)
    print("\n(d) DCT vs Wavelet (Haar) compression:")

    def haar_2d(image):
        """Simple 2D Haar wavelet transform (one level)."""
        rows, cols = image.shape
        result = np.zeros_like(image)
        # Row-wise
        temp = np.zeros_like(image)
        for i in range(rows):
            for j in range(0, cols, 2):
                temp[i, j // 2] = (image[i, j] + image[i, j + 1]) / np.sqrt(2)
                temp[i, cols // 2 + j // 2] = (image[i, j] - image[i, j + 1]) / np.sqrt(2)
        # Column-wise
        for j in range(cols):
            for i in range(0, rows, 2):
                result[i // 2, j] = (temp[i, j] + temp[i + 1, j]) / np.sqrt(2)
                result[rows // 2 + i // 2, j] = (temp[i, j] - temp[i + 1, j]) / np.sqrt(2)
        return result

    def ihaar_2d(coeffs):
        """Inverse 2D Haar wavelet transform (one level)."""
        rows, cols = coeffs.shape
        temp = np.zeros_like(coeffs)
        # Column-wise inverse
        for j in range(cols):
            for i in range(rows // 2):
                temp[2 * i, j] = (coeffs[i, j] + coeffs[rows // 2 + i, j]) / np.sqrt(2)
                temp[2 * i + 1, j] = (coeffs[i, j] - coeffs[rows // 2 + i, j]) / np.sqrt(2)
        # Row-wise inverse
        result = np.zeros_like(coeffs)
        for i in range(rows):
            for j in range(cols // 2):
                result[i, 2 * j] = (temp[i, j] + temp[i, cols // 2 + j]) / np.sqrt(2)
                result[i, 2 * j + 1] = (temp[i, j] - temp[i, cols // 2 + j]) / np.sqrt(2)
        return result

    def wavelet_compress(image, keep_frac):
        """Compress by keeping only a fraction of largest wavelet coefficients."""
        coeffs = haar_2d(image)
        threshold = np.percentile(np.abs(coeffs), (1 - keep_frac) * 100)
        coeffs_thresh = coeffs * (np.abs(coeffs) >= threshold)
        nonzero = np.count_nonzero(coeffs_thresh)
        recon = ihaar_2d(coeffs_thresh)
        cr = image.size / max(nonzero, 1)
        return recon, cr

    keep_fracs = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
    print(f"    {'Keep%':>6s}  {'Wavelet PSNR':>13s}  {'Wavelet CR':>10s}  {'JPEG PSNR':>10s}")
    for kf in keep_fracs:
        w_recon, w_cr = wavelet_compress(img, kf)
        w_mse = np.mean((img - w_recon) ** 2)
        w_psnr = 10 * np.log10(255 ** 2 / w_mse) if w_mse > 0 else 60

        # Find matching JPEG quality for similar CR
        best_j_psnr = 0
        for q in range(1, 100):
            j_recon, j_cr = jpeg_compress_decompress(img, q)
            if abs(j_cr - w_cr) < w_cr * 0.3:
                j_mse = np.mean((img - j_recon) ** 2)
                j_psnr = 10 * np.log10(255 ** 2 / j_mse) if j_mse > 0 else 60
                if j_psnr > best_j_psnr:
                    best_j_psnr = j_psnr

        print(f"    {kf * 100:5.1f}%  {w_psnr:12.2f} dB  {w_cr:9.1f}:1  "
              f"{best_j_psnr:9.2f} dB" if best_j_psnr > 0 else
              f"    {kf * 100:5.1f}%  {w_psnr:12.2f} dB  {w_cr:9.1f}:1  {'N/A':>10s}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(results[10][0], cmap='gray')
    axes[0, 1].set_title(f'JPEG Q=10 ({results[10][2]:.1f} dB)')
    axes[0, 2].imshow(results[50][0], cmap='gray')
    axes[0, 2].set_title(f'JPEG Q=50 ({results[50][2]:.1f} dB)')
    axes[1, 0].imshow(deblocked, cmap='gray')
    axes[1, 0].set_title('Deblocked (Q=10)')
    axes[1, 1].plot(rd_cr, rd_psnr, 'b-o', markersize=3)
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('Rate-Distortion')
    axes[1, 1].grid(True, alpha=0.3)
    w_recon_vis, _ = wavelet_compress(img, 0.1)
    axes[1, 2].imshow(w_recon_vis, cmap='gray')
    axes[1, 2].set_title('Haar wavelet (10% coeffs)')
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_6_jpeg_compression.png', dpi=150)
    plt.close()
    print("\n    Plot saved: ex15_6_jpeg_compression.png")


# === Exercise 7: 2D Sampling and Aliasing ===
# Problem: Zone plate downsampling, interpolation methods, bandwidth estimation.

def exercise_7():
    """2D sampling and aliasing: zone plate, interpolation, bandwidth."""
    N = 512

    # (a) Zone plate pattern with downsampling
    print("(a) Zone plate downsampling (with and without anti-aliasing):")
    alpha = 0.0003
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    zone_plate = np.cos(alpha * (xx ** 2 + yy ** 2))
    zone_plate = ((zone_plate + 1) / 2 * 255).astype(np.float64)

    factors = [2, 4, 8]
    for D in factors:
        # Without anti-aliasing (naive downsampling)
        naive = zone_plate[::D, ::D]

        # With anti-aliasing (lowpass filter then downsample)
        # Cutoff at pi/D
        sigma_aa = D / (2 * np.pi) * 2  # approximate anti-aliasing
        filtered = gaussian_filter(zone_plate, sigma=sigma_aa)
        antialiased = filtered[::D, ::D]

        # Measure aliasing energy
        naive_up = np.repeat(np.repeat(naive, D, axis=0), D, axis=1)[:N, :N]
        aa_up = np.repeat(np.repeat(antialiased, D, axis=0), D, axis=1)[:N, :N]
        alias_naive = np.mean((zone_plate - naive_up) ** 2)
        alias_aa = np.mean((zone_plate - aa_up) ** 2)

        print(f"    Factor {D}x: naive MSE = {alias_naive:.1f}, "
              f"anti-aliased MSE = {alias_aa:.1f}, "
              f"improvement = {alias_naive / max(alias_aa, 1e-10):.1f}x")

    # (b) Interpolation methods from scratch
    print("\n(b) Interpolation comparison (8x upsampling of 32x32 image):")

    small = zone_plate[:32, :32]
    factor = 8
    out_size = 32 * factor

    def nearest_interp(img, scale):
        h, w = img.shape
        oh, ow = int(h * scale), int(w * scale)
        result = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                si = min(int(i / scale), h - 1)
                sj = min(int(j / scale), w - 1)
                result[i, j] = img[si, sj]
        return result

    def bilinear_interp(img, scale):
        h, w = img.shape
        oh, ow = int(h * scale), int(w * scale)
        result = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                y = i / scale
                x = j / scale
                y0, x0 = int(np.floor(y)), int(np.floor(x))
                y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
                dy, dx = y - y0, x - x0
                result[i, j] = ((1 - dy) * (1 - dx) * img[y0, x0] +
                                (1 - dy) * dx * img[y0, x1] +
                                dy * (1 - dx) * img[y1, x0] +
                                dy * dx * img[y1, x1])
        return result

    def bicubic_kernel(t):
        """Mitchell-Netravali bicubic kernel (a=-0.5)."""
        t = abs(t)
        if t <= 1:
            return 1.5 * t ** 3 - 2.5 * t ** 2 + 1
        elif t <= 2:
            return -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2
        else:
            return 0

    def bicubic_interp(img, scale):
        h, w = img.shape
        oh, ow = int(h * scale), int(w * scale)
        result = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                y = i / scale
                x = j / scale
                y0 = int(np.floor(y))
                x0 = int(np.floor(x))
                val = 0.0
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        yi = np.clip(y0 + m, 0, h - 1)
                        xi = np.clip(x0 + n, 0, w - 1)
                        val += img[yi, xi] * bicubic_kernel(y - (y0 + m)) * bicubic_kernel(x - (x0 + n))
                result[i, j] = val
        return result

    # Reference: high-res crop
    reference = zone_plate[:out_size, :out_size]

    nn_up = nearest_interp(small, factor)
    bl_up = bilinear_interp(small, factor)
    bc_up = bicubic_interp(small, factor)

    for name, upsampled in [('Nearest', nn_up), ('Bilinear', bl_up), ('Bicubic', bc_up)]:
        mse = np.mean((reference - upsampled) ** 2)
        p = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 60
        print(f"    {name:10s}: PSNR = {p:.2f} dB")

    # (c) Effective bandwidth estimation
    print("\n(c) Effective bandwidth estimation:")

    # Use a different test image with clear spectral content
    test_img = zone_plate[:256, :256]
    F_test = fftshift(fft2(test_img))
    power_spectrum = np.abs(F_test) ** 2

    # Radial power spectrum
    center = 128
    max_radius = 128
    radial_power = np.zeros(max_radius)
    count = np.zeros(max_radius)
    for i in range(256):
        for j in range(256):
            r = int(np.sqrt((i - center) ** 2 + (j - center) ** 2))
            if r < max_radius:
                radial_power[r] += power_spectrum[i, j]
                count[r] += 1
    radial_power[count > 0] /= count[count > 0]

    # Find effective bandwidth (95% energy contained)
    total_energy = np.sum(radial_power)
    cumulative = np.cumsum(radial_power)
    bw_95 = np.searchsorted(cumulative, 0.95 * total_energy)
    bw_99 = np.searchsorted(cumulative, 0.99 * total_energy)

    print(f"    95% energy bandwidth: {bw_95} cycles/image")
    print(f"    99% energy bandwidth: {bw_99} cycles/image")
    print(f"    Nyquist requirement: sample at >= {2 * bw_95} pixels")
    print(f"    Image size: {test_img.shape[0]} pixels (sufficient: {test_img.shape[0] >= 2 * bw_95})")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(zone_plate, cmap='gray')
    axes[0, 0].set_title('Zone Plate 512x512')
    axes[0, 1].imshow(zone_plate[::4, ::4], cmap='gray')
    axes[0, 1].set_title('Naive 4x downsample')
    aa_img = gaussian_filter(zone_plate, sigma=4 / (2 * np.pi) * 2)
    axes[0, 2].imshow(aa_img[::4, ::4], cmap='gray')
    axes[0, 2].set_title('Anti-aliased 4x')
    axes[1, 0].imshow(nn_up[:64, :64], cmap='gray')
    axes[1, 0].set_title('Nearest neighbor')
    axes[1, 1].imshow(bl_up[:64, :64], cmap='gray')
    axes[1, 1].set_title('Bilinear')
    axes[1, 2].imshow(bc_up[:64, :64], cmap='gray')
    axes[1, 2].set_title('Bicubic')
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('ex15_7_sampling_aliasing.png', dpi=150)
    plt.close()
    print("    Plot saved: ex15_7_sampling_aliasing.png")


# === Main ===

def main():
    exercises = [
        ("Exercise 1: 2D DFT Properties", exercise_1),
        ("Exercise 2: Smoothing Filter Comparison", exercise_2),
        ("Exercise 3: Frequency Domain Design", exercise_3),
        ("Exercise 4: Edge Detection Comparison", exercise_4),
        ("Exercise 5: Histogram Processing", exercise_5),
        ("Exercise 6: JPEG Compression Analysis", exercise_6),
        ("Exercise 7: 2D Sampling and Aliasing", exercise_7),
    ]

    for title, func in exercises:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}\n")
        func()


if __name__ == "__main__":
    main()
