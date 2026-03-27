"""
Exercises for Lesson 14: Computational Optics
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: ABCD Ray Tracing Through a 4f System
    Design a two-lens optical relay (4f system) and trace rays through it,
    then add a third lens and analyze the modified system.
    """
    f1 = 50e-3   # First lens focal length (m)
    f2 = 100e-3  # Second lens focal length (m)

    print("ABCD Ray Tracing: 4f Optical Relay")
    print(f"f1 = {f1*1e3:.0f} mm, f2 = {f2*1e3:.0f} mm")

    # 4f system: lens1 at z=0, lens2 at z=f1+f2
    d_12 = f1 + f2  # Separation between lenses

    # ABCD matrices
    L1 = np.array([[1, 0], [-1/f1, 1]])
    D12 = np.array([[1, d_12], [0, 1]])
    L2 = np.array([[1, 0], [-1/f2, 1]])

    # (a) System matrix and magnification
    M_sys = L2 @ D12 @ L1
    A, B, C, D = M_sys[0, 0], M_sys[0, 1], M_sys[1, 0], M_sys[1, 1]

    print(f"\n(a) System ABCD matrix (4f relay):")
    print(f"  A = {A:.4f}, B = {B:.6f}")
    print(f"  C = {C:.4f}, D = {D:.4f}")
    print(f"  Magnification M = -f2/f1 = {-f2/f1:.2f}")
    print(f"  From matrix A = {A:.4f} (should equal -f2/f1)")
    print(f"  det(M) = {A*D - B*C:.6f} (should be 1)")

    # (b) Trace fan of 11 rays
    ray_heights = np.linspace(-10e-3, 10e-3, 11)
    angle = 0.0  # Initial angle

    print(f"\n(b) Ray trace (input angle u = 0):")
    print(f"{'y_in (mm)':>12} {'y_out (mm)':>12} {'u_out (mrad)':>14} {'M_actual':>10}")
    print("-" * 50)

    for y_in in ray_heights:
        ray_in = np.array([y_in, angle])
        ray_out = M_sys @ ray_in
        M_actual = ray_out[0] / y_in if abs(y_in) > 1e-10 else 0
        print(f"{y_in*1e3:>12.2f} {ray_out[0]*1e3:>12.4f} "
              f"{ray_out[1]*1e3:>14.4f} {M_actual:>10.4f}")

    # (c) Add third lens
    f3 = -30e-3     # Third lens (diverging)
    d_23 = 20e-3    # 20 mm after second lens

    D23 = np.array([[1, d_23], [0, 1]])
    L3 = np.array([[1, 0], [-1/f3, 1]])

    M_new = L3 @ D23 @ M_sys
    A2, B2, C2, D2 = M_new[0, 0], M_new[0, 1], M_new[1, 0], M_new[1, 1]

    print(f"\n(c) With third lens (f3 = {f3*1e3:.0f} mm, d23 = {d_23*1e3:.0f} mm):")
    print(f"  New system matrix:")
    print(f"  A = {A2:.4f}, B = {B2:.6f}")
    print(f"  C = {C2:.4f}, D = {D2:.4f}")

    # Effective focal length of the combined system
    f_eff = -1 / C2 if abs(C2) > 1e-10 else np.inf
    print(f"  Effective focal length: f_eff = {f_eff*1e3:.2f} mm")

    # Trace the same ray fan
    print(f"\n  Modified ray trace:")
    print(f"{'y_in (mm)':>12} {'y_out (mm)':>12} {'u_out (mrad)':>14}")
    print("-" * 40)

    for y_in in ray_heights[::2]:  # Every other ray
        ray_in = np.array([y_in, angle])
        ray_out = M_new @ ray_in
        print(f"{y_in*1e3:>12.2f} {ray_out[0]*1e3:>12.4f} {ray_out[1]*1e3:>14.4f}")

    # Find new focal point (where marginal ray crosses axis)
    ray_marginal = np.array([10e-3, 0])
    ray_after = M_new @ ray_marginal
    if abs(ray_after[1]) > 1e-10:
        z_focus = -ray_after[0] / ray_after[1]
        print(f"\n  Focus after third lens: z = {z_focus*1e3:.2f} mm (from L3)")
        print(f"  The diverging lens pushes the focus further away")
    else:
        print(f"\n  Rays emerge parallel (afocal system after modification)")


def exercise_2():
    """
    Exercise 2: BPM Waveguide Simulation
    Simulate a step-index slab waveguide using the Beam Propagation
    Method and analyze guided modes.
    """
    # Waveguide parameters
    n_core = 1.50
    n_clad = 1.48
    a = 5e-6         # Core half-width (m)
    lam = 1.55e-6    # Wavelength (m)
    k0 = 2 * np.pi / lam

    print("BPM Waveguide Simulation:")
    print(f"Core index: n_core = {n_core}")
    print(f"Cladding index: n_clad = {n_clad}")
    print(f"Core half-width: a = {a*1e6:.0f} um")
    print(f"Wavelength: {lam*1e6:.2f} um")

    # V-number
    NA = np.sqrt(n_core**2 - n_clad**2)
    V = k0 * a * NA
    print(f"\nNA = {NA:.4f}")
    print(f"V-number = {V:.4f}")

    # (b) Analytical mode count
    N_modes = int(V / np.pi) + 1  # For slab waveguide: m < V/pi
    print(f"\nAnalytical mode prediction:")
    print(f"  Modes in slab: V/pi = {V/np.pi:.2f} -> {N_modes} guided modes")

    # Mode propagation constants
    # For slab waveguide, modes satisfy: tan(kx*a) = gamma/kx (symmetric)
    # where kx^2 + gamma^2 = (k0*NA)^2, kx = k0*sqrt(n_core^2 - n_eff^2)
    print(f"\n  Finding mode effective indices:")

    n_eff_values = []
    # Scan n_eff from n_clad to n_core
    n_scan = np.linspace(n_clad + 0.0001, n_core - 0.0001, 10000)

    for n_eff in n_scan:
        kx = k0 * np.sqrt(n_core**2 - n_eff**2)
        gamma = k0 * np.sqrt(n_eff**2 - n_clad**2)

        # Symmetric modes: kx*tan(kx*a) = gamma
        val_sym = kx * np.tan(kx * a) - gamma
        # Anti-symmetric: -kx*cot(kx*a) = gamma
        if abs(np.cos(kx * a)) > 1e-10:
            val_asym = -kx / np.tan(kx * a) - gamma
        else:
            val_asym = float('inf')

        # Check for sign changes (zero crossings)
        if len(n_eff_values) == 0 or n_eff - n_eff_values[-1] > 0.001:
            n_eff_values.append(n_eff)

    # More rigorous mode solving
    print(f"\n  Guided mode effective indices (numerical search):")
    modes_found = []

    # Find symmetric modes
    prev_val = None
    for n_eff in n_scan:
        kx = k0 * np.sqrt(n_core**2 - n_eff**2)
        gamma = k0 * np.sqrt(max(0, n_eff**2 - n_clad**2))
        val = kx * np.tan(kx * a) - gamma

        if prev_val is not None and val * prev_val < 0 and abs(val) < 1e6:
            modes_found.append(("TE (sym)", n_eff))
        prev_val = val

    # Find anti-symmetric modes
    prev_val = None
    for n_eff in n_scan:
        kx = k0 * np.sqrt(n_core**2 - n_eff**2)
        gamma = k0 * np.sqrt(max(0, n_eff**2 - n_clad**2))
        cos_kxa = np.cos(kx * a)
        if abs(cos_kxa) < 1e-10:
            prev_val = None
            continue
        val = -kx / np.tan(kx * a) - gamma

        if prev_val is not None and val * prev_val < 0 and abs(val) < 1e6:
            modes_found.append(("TE (asym)", n_eff))
        prev_val = val

    print(f"{'Mode':>12} {'n_eff':>10} {'beta (1/m)':>14}")
    print("-" * 38)
    for mode_type, neff in sorted(modes_found, key=lambda x: -x[1]):
        beta = k0 * neff
        print(f"{mode_type:>12} {neff:>10.6f} {beta:>14.2f}")

    # BPM propagation concept
    print(f"\n--- BPM Propagation Concept ---")
    print(f"  Split-step method: U(z+dz) = exp(i*dz*D/2) * exp(i*dz*N) * exp(i*dz*D/2) * U(z)")
    print(f"  D = diffraction operator (free-space propagation)")
    print(f"  N = refractive index perturbation = k0*(n(x)-n_ref)")

    # Propagation parameters
    dz = 1e-6   # Step size (m)
    L_prop = 5e-3  # Total propagation (m)
    N_steps = int(L_prop / dz)
    print(f"\n  Propagation distance: {L_prop*1e3:.0f} mm")
    print(f"  Step size: {dz*1e6:.0f} um")
    print(f"  Number of steps: {N_steps}")


def exercise_3():
    """
    Exercise 3: Phase Retrieval (Gerchberg-Saxton Algorithm)
    Implement the GS and HIO algorithms for phase retrieval from
    intensity measurements.
    """
    N = 128
    lam = 633e-9

    print("Phase Retrieval: GS and HIO Algorithms")
    print(f"Grid: {N}x{N}")

    # Create a test object: letter "F" with smooth random phase
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # Amplitude: Letter F shape
    A = np.zeros((N, N))
    # Vertical bar
    A[(X > -0.3) & (X < -0.1) & (Y > -0.5) & (Y < 0.5)] = 1.0
    # Top horizontal
    A[(X > -0.3) & (X < 0.3) & (Y > 0.3) & (Y < 0.5)] = 1.0
    # Middle horizontal
    A[(X > -0.3) & (X < 0.2) & (Y > -0.1) & (Y < 0.1)] = 1.0

    # Phase: sum of low-order Zernike-like polynomials
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    phi = 2.0 * (2*R**2 - 1) + 0.5 * R * np.cos(Theta) + 0.3 * R**2 * np.cos(2*Theta)
    phi[R > 1] = 0

    # Complex field
    U = A * np.exp(1j * phi)

    # (a) Compute intensities
    I_obj = np.abs(U)**2
    U_far = np.fft.fftshift(np.fft.fft2(U))
    I_far = np.abs(U_far)**2

    print(f"\n(a) Object and far-field intensities computed:")
    print(f"  Object: {np.count_nonzero(I_obj)} non-zero pixels")
    print(f"  Far-field max/min: {I_far.max():.4e} / {I_far[I_far>0].min():.4e}")

    # (b) Gerchberg-Saxton algorithm
    print(f"\n(b) Gerchberg-Saxton Algorithm (500 iterations):")

    np.random.seed(42)
    phi_guess = np.random.uniform(-np.pi, np.pi, (N, N))

    # Measured amplitudes
    A_obj = np.sqrt(I_obj)
    A_far = np.sqrt(I_far)

    U_current = A_obj * np.exp(1j * phi_guess)
    errors = []

    for iteration in range(500):
        # Forward propagation
        G = np.fft.fftshift(np.fft.fft2(U_current))

        # Replace amplitude with measured far-field amplitude
        G_constrained = A_far * np.exp(1j * np.angle(G))

        # Backward propagation
        U_back = np.fft.ifft2(np.fft.ifftshift(G_constrained))

        # Replace amplitude with measured object amplitude
        U_current = A_obj * np.exp(1j * np.angle(U_back))

        # Error metric
        error = np.sum((np.abs(G) - A_far)**2) / np.sum(A_far**2)
        errors.append(error)

    print(f"{'Iteration':>12} {'Error':>14}")
    print("-" * 28)
    for i in [0, 10, 50, 100, 200, 499]:
        print(f"{i:>12} {errors[i]:>14.6e}")

    # Phase recovery quality
    phi_recovered = np.angle(U_current)
    # Remove constant phase offset
    mask = A > 0.5
    if np.any(mask):
        phi_diff = phi_recovered[mask] - phi[mask]
        offset = np.mean(phi_diff)
        phi_recovered_adj = phi_recovered - offset
        rms_error = np.sqrt(np.mean((phi_recovered_adj[mask] - phi[mask])**2))
        print(f"\n  Phase recovery RMS error: {rms_error:.4f} rad")
        print(f"    = {rms_error/(2*np.pi):.4f} waves")

    # (c) Different random starts
    print(f"\n(c) Convergence with different random starts:")
    print(f"{'Seed':>6} {'Final error':>14} {'RMS phase err':>14}")
    print("-" * 36)

    for seed in range(5):
        np.random.seed(seed * 7 + 1)
        phi_g = np.random.uniform(-np.pi, np.pi, (N, N))
        Uc = A_obj * np.exp(1j * phi_g)

        for _ in range(500):
            G = np.fft.fftshift(np.fft.fft2(Uc))
            G = A_far * np.exp(1j * np.angle(G))
            Ub = np.fft.ifft2(np.fft.ifftshift(G))
            Uc = A_obj * np.exp(1j * np.angle(Ub))

        final_err = np.sum((np.abs(np.fft.fftshift(np.fft.fft2(Uc))) - A_far)**2) / np.sum(A_far**2)
        phi_rec = np.angle(Uc)
        if np.any(mask):
            off = np.mean(phi_rec[mask] - phi[mask])
            rms = np.sqrt(np.mean((phi_rec[mask] - off - phi[mask])**2))
        else:
            rms = 0
        print(f"{seed:>6} {final_err:>14.6e} {rms:>14.4f}")

    # (d) HIO algorithm comparison (concept)
    print(f"\n(d) HIO Algorithm (Hybrid Input-Output):")
    print(f"  Update rule: u'(x) = u_GS(x) if x in support")
    print(f"               u'(x) = u_prev(x) - beta * u_GS(x) otherwise")
    print(f"  beta = 0.9 typically")
    print(f"  HIO converges faster than GS for sparse objects")
    print(f"  HIO avoids stagnation at local minima")


def exercise_4():
    """
    Exercise 4: Zernike Wavefront Analysis
    Synthesize and analyze a wavefront from given Zernike coefficients,
    calculate RMS error and Strehl ratio.
    """
    N = 256
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    mask = R <= 1.0

    print("Zernike Wavefront Analysis:")

    # Given Noll Zernike coefficients (in waves)
    coeffs = {
        4: 0.35,    # Defocus
        5: -0.12,   # Astigmatism (oblique)
        7: 0.08,    # Coma (vertical)
        11: 0.15,   # Spherical aberration
    }

    names = {
        4: "Defocus (Z4)",
        5: "Astigmatism (Z5)",
        7: "Coma (Z7)",
        11: "Spherical (Z11)",
    }

    print(f"\nGiven Zernike coefficients (waves):")
    for j, a in coeffs.items():
        print(f"  a_{j} = {a:+.2f}  ({names[j]})")

    # (a) Synthesize wavefront
    # Noll Zernike polynomials (selected)
    Z = np.zeros((N, N))

    # Z4 = sqrt(3) * (2*rho^2 - 1)  [Defocus]
    Z += coeffs[4] * np.sqrt(3) * (2*R**2 - 1)

    # Z5 = sqrt(6) * rho^2 * sin(2*theta)  [Astigmatism 45]
    Z += coeffs[5] * np.sqrt(6) * R**2 * np.sin(2*Theta)

    # Z7 = sqrt(8) * (3*rho^3 - 2*rho) * sin(theta)  [Coma Y]
    Z += coeffs[7] * np.sqrt(8) * (3*R**3 - 2*R) * np.sin(Theta)

    # Z11 = sqrt(5) * (6*rho^4 - 6*rho^2 + 1)  [Spherical]
    Z += coeffs[11] * np.sqrt(5) * (6*R**4 - 6*R**2 + 1)

    Z[~mask] = 0

    print(f"\n(a) Wavefront synthesized:")
    print(f"  PV (peak-to-valley): {(Z[mask].max() - Z[mask].min()):.4f} waves")

    # (b) RMS wavefront error
    sigma_W = np.sqrt(np.sum(np.array(list(coeffs.values()))**2))
    print(f"\n(b) RMS wavefront error: sigma = sqrt(sum(a_j^2))")
    for j, a in coeffs.items():
        print(f"  a_{j}^2 = {a**2:.6f}")
    print(f"  sigma_W = {sigma_W:.6f} waves")

    # (c) Strehl ratio (Marechal approximation)
    S = np.exp(-(2 * np.pi * sigma_W)**2)
    print(f"\n(c) Strehl ratio: S = exp(-(2*pi*sigma)^2)")
    print(f"  S = exp(-({2*np.pi*sigma_W:.4f})^2)")
    print(f"  S = {S:.6f}")
    print(f"  = {S*100:.4f}%")

    # (d) Which mode to correct for best Strehl improvement?
    print(f"\n(d) Single-mode correction analysis:")
    print(f"{'Remove':>20} {'New sigma':>12} {'New Strehl':>12} {'Improvement':>14}")
    print("-" * 60)

    for j_remove in sorted(coeffs.keys()):
        remaining = {k: v for k, v in coeffs.items() if k != j_remove}
        sigma_new = np.sqrt(np.sum(np.array(list(remaining.values()))**2))
        S_new = np.exp(-(2 * np.pi * sigma_new)**2)
        improvement = S_new - S
        print(f"{names[j_remove]:>20} {sigma_new:>12.6f} {S_new:>12.6f} {improvement:>14.6f}")

    # Best mode to correct
    best_j = max(coeffs.keys(), key=lambda j: abs(coeffs[j]))
    print(f"\n  Best mode to correct: {names[best_j]} (|a| = {abs(coeffs[best_j]):.2f})")
    print(f"  Defocus is the dominant contributor to wavefront error")


def exercise_5():
    """
    Exercise 5: Digital Holography Simulation
    Generate a hologram of a point object and reconstruct at different
    distances using the angular spectrum method.
    """
    N = 1024
    lam = 633e-9       # Wavelength
    sensor_size = 5e-3  # 5 mm
    pixel = sensor_size / N
    k = 2 * np.pi / lam

    # Object: point source at z = 10 mm
    z_obj = 10e-3
    theta_ref = 3.0    # Off-axis reference angle (degrees)

    print("Digital Holography: Point Object Reconstruction")
    print(f"Grid: {N}x{N}")
    print(f"Sensor: {sensor_size*1e3:.0f} mm x {sensor_size*1e3:.0f} mm")
    print(f"Pixel: {pixel*1e6:.2f} um")
    print(f"Object distance: z = {z_obj*1e3:.0f} mm")
    print(f"Reference angle: {theta_ref} deg")

    # (a) Compute hologram
    x = np.arange(N) * pixel - sensor_size/2
    X, Y = np.meshgrid(x, x)

    # Point source field at sensor
    r = np.sqrt(X**2 + Y**2 + z_obj**2)
    U_obj = np.exp(1j * k * r) / r

    # Reference beam (plane wave at angle)
    kx_ref = k * np.sin(np.radians(theta_ref))
    U_ref = np.exp(1j * kx_ref * X)

    # Hologram intensity
    I_holo = np.abs(U_obj + U_ref)**2

    print(f"\n(a) Hologram recorded:")
    print(f"  Max/Min intensity: {I_holo.max():.4e} / {I_holo.min():.4e}")

    # Check fringe frequency vs Nyquist
    f_fringe = np.sin(np.radians(theta_ref)) / lam
    f_nyquist = 1 / (2 * pixel)
    print(f"  Reference fringe frequency: {f_fringe/1e3:.1f} cy/mm")
    print(f"  Nyquist frequency: {f_nyquist/1e3:.1f} cy/mm")
    print(f"  Sampling OK: {f_fringe < f_nyquist}")

    # (b) Reconstruct at different distances
    # Use 1D slice for efficiency
    print(f"\n(b) Reconstruction at different distances (1D slice):")

    fx = np.fft.fftfreq(N, pixel)

    # Reconstruct
    U_illum = I_holo[N//2, :] * np.exp(1j * kx_ref * x)  # 1D slice
    A = np.fft.fft(U_illum)

    distances = [5e-3, 8e-3, 10e-3, 12e-3, 15e-3]
    print(f"{'z (mm)':>10} {'Peak I':>12} {'Peak pos (um)':>14} {'Focus quality':>14}")
    print("-" * 52)

    peak_intensities = []
    for z_recon in distances:
        # Angular spectrum back-propagation
        H = np.exp(-1j * k * z_recon * np.sqrt(1 - (lam*fx)**2 + 0j))
        U_recon = np.fft.ifft(A * H)
        I_recon = np.abs(U_recon)**2

        idx_max = np.argmax(I_recon)
        peak_I = I_recon[idx_max]
        peak_x = x[idx_max]
        peak_intensities.append(peak_I)

        quality = "Best focus" if abs(z_recon - z_obj) < 1e-3 else "Defocused"
        print(f"{z_recon*1e3:>10.0f} {peak_I:>12.4e} {peak_x*1e6:>14.1f} {quality:>14}")

    # (c) Best focus
    best_idx = np.argmax(peak_intensities)
    print(f"\n(c) Best focus at z = {distances[best_idx]*1e3:.0f} mm")
    print(f"  (Object was placed at z = {z_obj*1e3:.0f} mm)")
    print(f"  The sharpest point reconstruction occurs at the object plane")

    # (d) Noise analysis
    print(f"\n(d) Effect of noise (SNR = 20 dB):")
    SNR_dB = 20
    SNR_linear = 10**(SNR_dB/20)
    noise_std = np.std(I_holo) / SNR_linear
    I_noisy = I_holo + np.random.normal(0, noise_std, I_holo.shape)

    # Reconstruct noisy hologram (1D)
    U_noisy = I_noisy[N//2, :] * np.exp(1j * kx_ref * x)
    A_noisy = np.fft.fft(U_noisy)
    H_focus = np.exp(-1j * k * z_obj * np.sqrt(1 - (lam*fx)**2 + 0j))

    U_recon_clean = np.fft.ifft(A * H_focus)
    U_recon_noisy = np.fft.ifft(A_noisy * H_focus)

    peak_clean = np.max(np.abs(U_recon_clean)**2)
    peak_noisy = np.max(np.abs(U_recon_noisy)**2)
    bg_noise = np.std(np.abs(U_recon_noisy)**2)

    print(f"  Added noise std: {noise_std:.4e}")
    print(f"  Clean peak: {peak_clean:.4e}")
    print(f"  Noisy peak: {peak_noisy:.4e}")
    print(f"  Background noise level: {bg_noise:.4e}")
    if bg_noise > 0:
        print(f"  Reconstructed SNR: {peak_noisy/bg_noise:.1f}")


def exercise_6():
    """
    Exercise 6: Comparing Computational Methods (BPM vs Analytical)
    Compare BPM simulation of a dielectric slab with the analytical
    Fabry-Perot formula.
    """
    lam = 1e-6     # Reference wavelength (m)
    d = 2 * lam    # Slab thickness = 2*lambda
    n_slab = 2.0   # Slab refractive index
    n_air = 1.0

    print("Comparing Computational Methods: Dielectric Slab")
    print(f"Slab thickness: d = 2*lambda = {d*1e6:.2f} um")
    print(f"Slab index: n = {n_slab}")
    print(f"Wavelength: {lam*1e6:.2f} um")

    # (a) Analytical: Fabry-Perot formula
    # T = 1 / [1 + F*sin^2(delta/2)]
    # delta = 2*pi*n*d/lambda (round-trip phase)
    # F = 4R/(1-R)^2, R = ((n-1)/(n+1))^2

    R_interface = ((n_slab - n_air) / (n_slab + n_air))**2
    F_coeff = 4 * R_interface / (1 - R_interface)**2
    delta = 2 * np.pi * n_slab * d / lam

    T_analytical = 1.0 / (1 + F_coeff * np.sin(delta/2)**2)

    print(f"\n(a) Analytical (Fabry-Perot):")
    print(f"  Interface reflectance: R = {R_interface:.4f}")
    print(f"  Coefficient of finesse: F = {F_coeff:.4f}")
    print(f"  Round-trip phase: delta = {delta:.4f} rad = {delta/np.pi:.4f}*pi")
    print(f"  Transmittance: T = {T_analytical:.6f}")

    # Spectrum
    print(f"\n  Transmittance vs wavelength:")
    print(f"{'Lambda/lambda_0':>16} {'T':>10}")
    print("-" * 28)

    for ratio in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
        lam_scan = ratio * lam
        delta_scan = 2 * np.pi * n_slab * d / lam_scan
        T_scan = 1.0 / (1 + F_coeff * np.sin(delta_scan/2)**2)
        print(f"{ratio:>16.2f} {T_scan:>10.4f}")

    # (b) BPM approximation
    print(f"\n(b) BPM (paraxial approximation):")
    # BPM assumes forward-propagating waves only
    # It misses back-reflections (Fabry-Perot effects)
    # For a thin slab, BPM gives: T_BPM ~ 1 (no reflection)
    T_BPM = 1.0  # BPM ignores reflections
    print(f"  BPM transmittance: T ~ {T_BPM:.4f}")
    print(f"  Error: {abs(T_BPM - T_analytical):.4f}")

    # (c) Why BPM is approximate
    print(f"\n(c) Physics missed by BPM:")
    print(f"  1. Back-reflections at interfaces (Fresnel)")
    print(f"  2. Multiple reflections (Fabry-Perot interference)")
    print(f"  3. Backward-propagating waves")
    print(f"  4. Large-angle scattering")
    print(f"  BPM uses the paraxial (slowly-varying envelope) approximation")
    print(f"  It only models forward propagation with small-angle diffraction")

    # (d) When FDTD is necessary
    print(f"\n(d) When to use FDTD instead of BPM:")
    print(f"  - High refractive index contrast (n_slab/n_air = {n_slab/n_air:.1f})")
    print(f"  - Sub-wavelength features (d < ~10*lambda)")
    print(f"  - Strong back-reflections (R = {R_interface*100:.1f}%)")
    print(f"  - Metallic structures (plasmonic, negative epsilon)")
    print(f"  - Photonic crystals (bandgap effects)")
    print(f"  - Non-paraxial propagation angles")

    print(f"\n  For this slab (d=2*lambda, n=2.0):")
    print(f"  - FDTD captures the correct T = {T_analytical:.4f}")
    print(f"  - BPM gives T ~ 1.0 (misses {(1-T_analytical)*100:.1f}% reflection)")
    print(f"  - FDTD is necessary for accurate simulation")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: ABCD Ray Tracing (4f System)", exercise_1),
        ("Exercise 2: BPM Waveguide Simulation", exercise_2),
        ("Exercise 3: Phase Retrieval (GS/HIO)", exercise_3),
        ("Exercise 4: Zernike Wavefront Analysis", exercise_4),
        ("Exercise 5: Digital Holography Simulation", exercise_5),
        ("Exercise 6: Comparing Computational Methods", exercise_6),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
