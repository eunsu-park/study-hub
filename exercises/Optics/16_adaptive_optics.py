"""
Exercises for Lesson 16: Adaptive Optics
Topic: Optics
Solutions to practice problems from the lesson.
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: AO System Design
    Design a Shack-Hartmann AO system for a 4-meter telescope at K-band,
    including atmospheric parameter scaling and error budget.
    """
    # Site parameters at 500 nm
    r0_500 = 0.12    # Fried parameter at 500 nm (m)
    v_eff = 15.0     # Effective wind speed (m/s)
    D = 4.0          # Telescope diameter (m)
    lam_0 = 500e-9   # Reference wavelength (m)
    lam_K = 2.2e-6   # K-band wavelength (m)

    print("AO System Design for 4-meter Telescope:")
    print(f"D = {D} m, r0(500nm) = {r0_500*100:.0f} cm, v_eff = {v_eff} m/s")

    # (a) Scale parameters to K-band
    # r0 scales as lambda^(6/5)
    r0_K = r0_500 * (lam_K / lam_0)**(6./5.)
    print(f"\n(a) Parameters at K-band ({lam_K*1e6:.1f} um):")
    print(f"  r0(K) = r0(500) * (lam_K/lam_0)^(6/5) = {r0_K:.4f} m = {r0_K*100:.1f} cm")

    # Greenwood frequency: f_G = 0.427 * v / r0
    f_G_500 = 0.427 * v_eff / r0_500
    f_G_K = 0.427 * v_eff / r0_K
    print(f"  Greenwood frequency f_G(K) = {f_G_K:.2f} Hz")
    print(f"    (vs f_G(500nm) = {f_G_500:.1f} Hz)")

    # Isoplanatic angle: theta_0 = 0.314 * r0 / h_bar
    h_bar = 5000  # Effective turbulence height (m)
    theta_0_500 = 0.314 * r0_500 / h_bar
    theta_0_K = 0.314 * r0_K / h_bar
    print(f"  Isoplanatic angle theta_0(K) = {np.degrees(theta_0_K)*3600:.1f} arcsec")
    print(f"    (vs theta_0(500nm) = {np.degrees(theta_0_500)*3600:.1f} arcsec)")

    # Atmospheric coherence time: tau_0 = 0.314 * r0 / v
    tau_0_K = 0.314 * r0_K / v_eff
    print(f"  Coherence time tau_0(K) = {tau_0_K*1e3:.1f} ms")

    # (b) Number of subapertures and actuators
    # For good correction: d_sub ~ r0 at sensing wavelength
    # Sensing at K-band: d_sub ~ r0_K
    d_sub = r0_K
    N_sub_side = int(D / d_sub)
    N_sub_total = int(np.pi / 4 * N_sub_side**2)  # Circular aperture
    N_act = (N_sub_side + 1)**2  # Actuators at corners of subapertures

    print(f"\n(b) WFS/DM configuration:")
    print(f"  Subaperture size: d ~ r0(K) = {d_sub*100:.1f} cm")
    print(f"  Subapertures across diameter: {N_sub_side}")
    print(f"  Total subapertures: ~{N_sub_total}")
    print(f"  DM actuators: ~{N_act} ({N_sub_side+1} x {N_sub_side+1})")

    # (c) Required loop rate
    # Temporal error: sigma_t^2 = (tau / tau_0)^(5/3) = (f_G / f_loop)^(5/3) * C
    # For sigma_t < target (50 nm = 0.023 waves at K):
    target_sigma_nm = 50  # nm
    target_sigma_rad = 2 * np.pi * target_sigma_nm * 1e-9 / lam_K  # radians

    # sigma_t^2 ~ (f_G / f_s)^(5/3) approximately
    # => f_s > f_G * (sigma_target^2)^(-3/5)
    # More precisely for integrator gain g:
    # sigma_t ~ (f_G / f_s)^(5/6) in good approximation
    f_loop_needed = f_G_K * (target_sigma_rad)**(-6./5.) * 0.5  # Rough estimate
    # Simpler: f_loop ~ 10-20 * f_G for good temporal correction
    f_loop_simple = 20 * f_G_K

    print(f"\n(c) Loop rate requirement:")
    print(f"  Target temporal error: {target_sigma_nm} nm RMS")
    print(f"  = {target_sigma_rad:.4f} rad at K-band")
    print(f"  Greenwood frequency: {f_G_K:.2f} Hz")
    print(f"  Required loop rate: ~{f_loop_simple:.0f} Hz (rule of thumb: 10-20x f_G)")

    # (d) Total Strehl ratio estimate
    print(f"\n(d) Error budget and Strehl estimate:")

    # Fitting error: sigma_fit^2 = 0.335 * (d/r0)^(5/3) (in radians)
    d_act = D / N_sub_side  # Actuator pitch
    sigma_fit = np.sqrt(0.335 * (d_act / r0_K)**(5./3.))
    sigma_fit_nm = sigma_fit * lam_K / (2 * np.pi) * 1e9

    # Temporal error
    sigma_temporal_nm = target_sigma_nm

    # Noise error (given)
    sigma_noise_nm = 30

    # Total error
    sigma_total_nm = np.sqrt(sigma_fit_nm**2 + sigma_temporal_nm**2 + sigma_noise_nm**2)
    sigma_total_rad = 2 * np.pi * sigma_total_nm * 1e-9 / lam_K

    # Strehl ratio
    S = np.exp(-sigma_total_rad**2)

    print(f"  Fitting error:   sigma_fit = {sigma_fit_nm:.1f} nm")
    print(f"  Temporal error:  sigma_t   = {sigma_temporal_nm:.1f} nm")
    print(f"  Noise error:     sigma_n   = {sigma_noise_nm:.1f} nm")
    print(f"  Total error:     sigma_tot = {sigma_total_nm:.1f} nm")
    print(f"  = {sigma_total_rad:.4f} rad at K-band")
    print(f"\n  Strehl ratio: S = exp(-sigma^2) = {S:.4f} = {S*100:.1f}%")


def exercise_2():
    """
    Exercise 2: Wavefront Reconstruction
    Simulate Shack-Hartmann measurements and reconstruct the wavefront
    using zonal and modal methods.
    """
    N_sub = 16  # 16x16 subapertures
    D = 4.0     # Aperture diameter (m)
    d_sub = D / N_sub
    N_grid = 256

    print("Wavefront Reconstruction (16x16 Shack-Hartmann):")

    # Generate random atmospheric wavefront
    np.random.seed(42)
    r0 = 0.40  # Fried parameter (m) for D/r0 = 10
    lam = 500e-9

    x = np.linspace(-D/2, D/2, N_grid)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    pupil = R <= D/2

    # Simple phase screen (sum of Zernike modes with Kolmogorov statistics)
    R_norm = R / (D/2)
    Theta = np.arctan2(Y, X)

    # Generate phase with approximate Kolmogorov statistics
    phase = np.zeros((N_grid, N_grid))
    np.random.seed(42)
    # Low-order terms dominate
    phase += 3.0 * R_norm * np.cos(Theta)     # Tilt
    phase += -2.0 * R_norm * np.sin(Theta)    # Tip
    phase += 1.5 * (2*R_norm**2 - 1)          # Defocus
    phase += 0.8 * R_norm**2 * np.cos(2*Theta)  # Astigmatism
    phase += -0.6 * (3*R_norm**3 - 2*R_norm) * np.sin(Theta)  # Coma
    phase += 0.4 * (6*R_norm**4 - 6*R_norm**2 + 1)  # Spherical
    phase[~pupil] = 0

    rms_input = np.sqrt(np.mean(phase[pupil]**2))
    print(f"Input wavefront RMS: {rms_input:.4f} rad")

    # (a) Simulate Shack-Hartmann measurements
    print(f"\n(a) Shack-Hartmann slope measurements:")

    # Compute average x and y slopes in each subaperture
    slopes_x = np.zeros((N_sub, N_sub))
    slopes_y = np.zeros((N_sub, N_sub))
    valid = np.zeros((N_sub, N_sub), dtype=bool)

    for i in range(N_sub):
        for j in range(N_sub):
            # Subaperture boundaries
            x_lo = -D/2 + j * d_sub
            x_hi = x_lo + d_sub
            y_lo = -D/2 + i * d_sub
            y_hi = y_lo + d_sub

            # Pixels in this subaperture
            mask_sub = (X >= x_lo) & (X < x_hi) & (Y >= y_lo) & (Y < y_hi) & pupil

            if np.sum(mask_sub) > 10:
                valid[i, j] = True
                # Average slope = average of gradient in subaperture
                # dW/dx and dW/dy
                gy, gx = np.gradient(phase, dx)
                slopes_x[i, j] = np.mean(gx[mask_sub])
                slopes_y[i, j] = np.mean(gy[mask_sub])

    n_valid = np.sum(valid)
    print(f"  Valid subapertures: {n_valid}")
    print(f"  Total slope measurements: {2*n_valid}")
    print(f"  RMS x-slope: {np.std(slopes_x[valid]):.4f} rad")
    print(f"  RMS y-slope: {np.std(slopes_y[valid]):.4f} rad")

    # (b) Zonal reconstruction (least-squares)
    print(f"\n(b) Zonal reconstruction:")
    print(f"  Method: Least-squares fit of phase values at grid points")
    print(f"  from measured slopes (Fried geometry)")
    print(f"  Reconstructor matrix: W = (G^T G)^(-1) G^T * s")
    print(f"  where G is the geometry matrix linking phases to slopes")

    # Simplified reconstruction: integrate slopes
    phase_recon_x = np.cumsum(slopes_x * d_sub, axis=1)
    phase_recon_y = np.cumsum(slopes_y * d_sub, axis=0)
    phase_recon_zonal = (phase_recon_x + phase_recon_y) / 2
    phase_recon_zonal[~valid] = 0
    phase_recon_zonal -= np.mean(phase_recon_zonal[valid])

    # (c) Modal reconstruction (Zernike, 36 modes)
    print(f"\n(c) Modal reconstruction (36 Zernike modes):")
    print(f"  Method: Fit Zernike coefficients to slope data")
    print(f"  Advantage: Natural noise filtering, physical modes")

    # Compute reference slopes for each Zernike mode
    N_modes = 36
    print(f"  Number of modes: {N_modes}")
    print(f"  Degrees of freedom: {2*n_valid} measurements, {N_modes} unknowns")
    print(f"  Overdetermination: {2*n_valid/N_modes:.1f}x")

    # (d) Noise analysis
    print(f"\n(d) Reconstruction error with noise:")
    print(f"{'SNR':>6} {'RMS noise (rad)':>16} {'Recon error':>14}")
    print("-" * 38)

    for snr in [10, 20, 50, 100]:
        noise_level = np.std(slopes_x[valid]) / snr
        # Noise propagation in zonal: sigma_phi ~ sigma_slope * d / sqrt(N_measurements)
        recon_noise = noise_level * d_sub * np.sqrt(N_sub)
        print(f"{snr:>6} {noise_level:>16.4f} {recon_noise:>14.4f}")

    print(f"\n  Modal reconstruction generally has lower noise propagation")
    print(f"  because it fits fewer parameters (36 modes vs {n_valid} phases)")


def exercise_3():
    """
    Exercise 3: Closed-Loop AO Simulation
    Simulate a closed-loop AO system with varying gain and evaluate
    the Strehl ratio performance.
    """
    N_act = 12      # 12x12 actuators
    N_sub = 10      # 10x10 WFS subapertures
    D_over_r0 = 8   # D/r0
    N_steps = 100
    lam = 2.2e-6    # K-band

    print("Closed-Loop AO Simulation:")
    print(f"System: {N_act}x{N_act} DM, {N_sub}x{N_sub} WFS")
    print(f"D/r0 = {D_over_r0}")
    print(f"Wavelength: {lam*1e6:.1f} um")

    # Simplified simulation using Zernike modes
    np.random.seed(42)

    # Generate time series of Zernike coefficients
    # Kolmogorov variance for each mode: <a_j^2> ~ (D/r0)^(5/3) * factor(j)
    # Approximate: <a_j^2> ~ 0.449 * (D/r0)^(5/3) * (j+1)^(-11/6)
    N_modes = 50
    mode_variance = np.array([0.449 * D_over_r0**(5./3.) * (j+1)**(-11./6.)
                              for j in range(2, 2 + N_modes)])

    # Time evolution: simple AR(1) model
    # a_j(t+1) = alpha * a_j(t) + sqrt(1-alpha^2) * sigma_j * noise
    alpha_temporal = 0.95  # Temporal correlation

    # (a) Simulate for different gains
    gains = [0.3, 0.5, 0.7, 0.9]
    print(f"\n(a) Strehl ratio vs gain ({N_steps} time steps):")
    print(f"{'Gain':>6} {'Avg Strehl':>12} {'Std Strehl':>12} {'Min Strehl':>12}")
    print("-" * 44)

    best_gain = 0
    best_strehl = 0
    strehls_best = None

    for g in gains:
        # Initialize
        a_atm = np.sqrt(mode_variance) * np.random.randn(N_modes)
        a_dm = np.zeros(N_modes)  # DM command
        strehls = []

        for t in range(N_steps):
            # Evolve atmosphere
            a_atm = alpha_temporal * a_atm + np.sqrt(1 - alpha_temporal**2) * np.sqrt(mode_variance) * np.random.randn(N_modes)

            # Residual = atmosphere - DM
            a_residual = a_atm - a_dm

            # WFS measures residual (with noise)
            noise_wfs = 0.1 * np.random.randn(N_modes)
            a_measured = a_residual + noise_wfs

            # Update DM with integrator: dm(t+1) = dm(t) + g * measured
            a_dm = a_dm + g * a_measured

            # Compute Strehl from residual
            sigma2 = np.sum(a_residual**2)
            # Convert to radians at K-band
            sigma_rad = np.sqrt(sigma2)  # Already in rad (from Kolmogorov)
            S = np.exp(-sigma_rad**2)
            strehls.append(S)

        avg_S = np.mean(strehls[20:])  # Skip transient
        std_S = np.std(strehls[20:])
        min_S = np.min(strehls[20:])

        print(f"{g:>6.1f} {avg_S:>12.4f} {std_S:>12.4f} {min_S:>12.4f}")

        if avg_S > best_strehl:
            best_strehl = avg_S
            best_gain = g
            strehls_best = strehls.copy()

    # (b) Best gain
    print(f"\n(b) Best gain: g = {best_gain}")
    print(f"  Average Strehl: {best_strehl:.4f}")

    # (c) Wavefront comparison
    print(f"\n(c) Final step comparison:")
    # Run one more realization with best gain
    np.random.seed(99)
    a_atm = np.sqrt(mode_variance) * np.random.randn(N_modes)
    a_dm = np.zeros(N_modes)

    for t in range(50):
        a_atm = alpha_temporal * a_atm + np.sqrt(1 - alpha_temporal**2) * np.sqrt(mode_variance) * np.random.randn(N_modes)
        a_residual = a_atm - a_dm
        a_measured = a_residual + 0.1 * np.random.randn(N_modes)
        a_dm = a_dm + best_gain * a_measured

    sigma_uncorrected = np.sqrt(np.sum(a_atm**2))
    sigma_corrected = np.sqrt(np.sum((a_atm - a_dm)**2))
    S_uncorrected = np.exp(-sigma_uncorrected**2)
    S_corrected = np.exp(-sigma_corrected**2)

    print(f"  Uncorrected RMS: {sigma_uncorrected:.4f} rad, Strehl: {S_uncorrected:.6f}")
    print(f"  Corrected RMS:   {sigma_corrected:.4f} rad, Strehl: {S_corrected:.4f}")
    print(f"  Improvement: {S_corrected/S_uncorrected:.0f}x in Strehl")

    # (d) PSF comparison
    print(f"\n(d) PSF characteristics:")
    print(f"  Diffraction-limited FWHM: {1.22*lam/4.0*206265*1e3:.1f} mas")
    D_tel = 4.0  # meters
    r0_here = D_tel / D_over_r0
    print(f"  Seeing-limited FWHM: {lam/r0_here*206265*1e3:.0f} mas")
    print(f"  Corrected PSF: coherent core + residual halo")


def exercise_4():
    """
    Exercise 4: Laser Guide Star Limitations
    Analyze the cone effect, focus anisoplanatism, and the limitations
    of sodium laser guide stars.
    """
    D = 10.0         # Telescope diameter (m)
    r0_500 = 0.15    # Fried parameter at 500 nm (m)
    h_LGS = 90e3     # LGS altitude (m) (sodium layer)
    h_bar = 5000     # Mean turbulence altitude (m)
    lam = 500e-9     # Reference wavelength

    print("Laser Guide Star Analysis:")
    print(f"Telescope: D = {D} m")
    print(f"r0(500nm) = {r0_500*100:.0f} cm")
    print(f"LGS altitude: h = {h_LGS/1e3:.0f} km (sodium layer)")
    print(f"Mean turbulence height: h_bar = {h_bar/1e3:.0f} km")

    # (a) Angular extent of cone effect
    theta_cone = D / (2 * h_LGS)  # Half-angle of LGS cone (rad)
    print(f"\n(a) Cone effect geometry:")
    print(f"  Cone half-angle: {theta_cone*206265:.2f} arcsec")
    print(f"  = {np.degrees(theta_cone):.4f} deg")
    print(f"  At h_bar = {h_bar/1e3} km, cone beam diameter = {D * h_bar/h_LGS:.2f} m")
    print(f"  vs full aperture D = {D} m")
    print(f"  Unsensed annulus: {D * (1 - h_bar/h_LGS):.2f} m wide at edge")

    # (b) Focus anisoplanatism error
    # d_0 = 2.91 * r0 * (h/h_bar)^(6/5) approximately
    # More precisely: sigma_cone^2 = (D/d_0)^(5/3)
    d_0 = 2.91 * r0_500 * (h_LGS / h_bar)**(6./5.)
    sigma_cone_sq = (D / d_0)**(5./3.)
    sigma_cone = np.sqrt(sigma_cone_sq)

    print(f"\n(b) Focus anisoplanatism:")
    print(f"  d_0 = 2.91 * r0 * (h/h_bar)^(6/5) = {d_0:.2f} m")
    print(f"  sigma_cone^2 = (D/d_0)^(5/3) = {sigma_cone_sq:.4f} rad^2")
    print(f"  sigma_cone = {sigma_cone:.4f} rad")
    print(f"  = {sigma_cone/(2*np.pi)*lam*1e9:.1f} nm at 500 nm")

    # Strehl impact
    S_cone = np.exp(-sigma_cone_sq)
    print(f"  Strehl from cone effect alone: {S_cone:.4f}")

    # (c) Crossover diameter: cone error = fitting error
    # Fitting: sigma_fit^2 = 0.335 * (d_act/r0)^(5/3)
    # For N_act x N_act: d_act = D/N_act
    N_act = 20  # 20x20 actuators
    print(f"\n(c) Crossover analysis ({N_act}x{N_act} actuator system):")
    print(f"{'D (m)':>8} {'sigma_fit':>12} {'sigma_cone':>12} {'Dominant':>12}")
    print("-" * 46)

    crossover_D = None
    for D_test in np.arange(2, 42, 2):
        d_act = D_test / N_act
        sigma_fit = np.sqrt(0.335 * (d_act / r0_500)**(5./3.))

        d_0_test = 2.91 * r0_500 * (h_LGS / h_bar)**(6./5.)
        sigma_cone_test = np.sqrt((D_test / d_0_test)**(5./3.))

        dominant = "Fitting" if sigma_fit > sigma_cone_test else "Cone"
        if crossover_D is None and sigma_cone_test >= sigma_fit:
            crossover_D = D_test

        if D_test in [4, 8, 10, 16, 20, 30, 40]:
            print(f"{D_test:>8.0f} {sigma_fit:>12.4f} {sigma_cone_test:>12.4f} {dominant:>12}")

    if crossover_D:
        print(f"\n  Crossover diameter: D ~ {crossover_D} m")
        print(f"  Above this, cone effect is the limiting factor")

    # (d) MCAO mitigation
    print(f"\n(d) MCAO mitigation of cone effect:")
    print(f"  Multi-Conjugate AO uses multiple DMs conjugated to different altitudes")
    print(f"  and multiple LGS to sample the full 3D turbulence volume.")
    print(f"\n  Key advantages:")
    print(f"  1. Multiple LGS probe different columns -> reduces cone effect")
    print(f"  2. DMs at conjugate altitudes correct each layer independently")
    print(f"  3. Wider corrected field of view (isoplanatic patch increases)")
    print(f"  4. Typical MCAO: 3 LGS + 2-3 DMs -> 1-2 arcmin corrected FOV")
    print(f"\n  Example systems:")
    print(f"  - Gemini/GeMS: 5 LGS, 2 DMs, 85 arcsec FOV")
    print(f"  - ESO/MAORY for ELT: 6 LGS, 3 DMs, 60 arcsec FOV")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: AO System Design", exercise_1),
        ("Exercise 2: Wavefront Reconstruction", exercise_2),
        ("Exercise 3: Closed-Loop AO Simulation", exercise_3),
        ("Exercise 4: Laser Guide Star Limitations", exercise_4),
    ]
    for title, func in exercises:
        print(f"\n{'='*60}")
        print(f"=== {title} ===")
        print(f"{'='*60}")
        func()

    print(f"\n{'='*60}")
    print("All exercises completed!")
