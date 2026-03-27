"""
Exercise Solutions for Lesson 16: Projects

Topics covered:
  - Burton model implementation with synthetic solar wind data
  - Statistical evaluation of Dst prediction (RMSE, MAE, correlation, HSS)
  - Radiation belt radial diffusion simulation during a real storm
  - Space weather dashboard construction concept
  - Coupling function comparison (Ey, Newell, Akasofu epsilon)
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def exercise_1():
    """
    Exercise 1: Burton Model Warm-Up with Synthetic Data

    48-hour synthetic solar wind:
    Hours 0-11: Quiet  (v=400, n=5, Bz=+2)
    Hours 12-24: Storm (v=600, n=15, Bz=-10)
    Hours 25-48: Recovery (v=450, n=8, Bz=+1)

    Implement Burton ODE with Euler method.
    """
    print("=" * 70)
    print("Exercise 1: Burton Model Warm-Up")
    print("=" * 70)

    # Parameters
    a_burton = -4.5   # nT/(mV/m * hr)
    tau = 7.7         # hr
    Ec = 0.5          # mV/m (threshold)
    b_press = 7.26    # nT/sqrt(nPa)
    c_offset = 11     # nT
    dt = 1.0          # hour

    m_p = 1.67e-27    # kg

    # Create synthetic solar wind (48 hours)
    hours = np.arange(0, 49, 1)  # 0 to 48 inclusive
    n_steps = len(hours)

    v_sw = np.zeros(n_steps)    # km/s
    n_sw = np.zeros(n_steps)    # cm^-3
    Bz = np.zeros(n_steps)      # nT

    for i, t in enumerate(hours):
        if t <= 11:
            v_sw[i], n_sw[i], Bz[i] = 400, 5, 2
        elif t <= 24:
            v_sw[i], n_sw[i], Bz[i] = 600, 15, -10
        else:
            v_sw[i], n_sw[i], Bz[i] = 450, 8, 1

    # Compute P_dyn and Ey
    # P_dyn = 0.5 * n * m_p * v^2 (in nPa)
    # n in cm^-3 = 1e6 m^-3, v in km/s = 1e3 m/s
    P_dyn = 0.5 * n_sw * 1e6 * m_p * (v_sw * 1e3)**2 * 1e9  # nPa
    Ey = -v_sw * Bz * 1e-3  # mV/m (Ey = -v * Bz, v in km/s, Bz in nT)
    # Note: Bz negative -> Ey positive (southward Bz drives injection)

    print(f"\n    Burton parameters: a={a_burton}, tau={tau} hr, Ec={Ec} mV/m")
    print(f"\n    Synthetic solar wind summary:")
    print(f"    {'Phase':<12} {'v (km/s)':>10} {'n (cm-3)':>10} {'Bz (nT)':>10} "
          f"{'Pdyn (nPa)':>12} {'Ey (mV/m)':>12}")
    print(f"    {'-'*66}")
    for phase, t_idx in [("Quiet", 5), ("Storm", 18), ("Recovery", 36)]:
        print(f"    {phase:<12} {v_sw[t_idx]:>10.0f} {n_sw[t_idx]:>10.0f} "
              f"{Bz[t_idx]:>10.0f} {P_dyn[t_idx]:>12.2f} {Ey[t_idx]:>12.2f}")

    # Integrate Burton equation: dDst*/dt = Q - Dst*/tau
    # Q = a * (Ey - Ec) for Ey > Ec, else Q = 0
    Dst_star = np.zeros(n_steps)
    Dst = np.zeros(n_steps)

    for i in range(1, n_steps):
        # Injection
        if Ey[i-1] > Ec:
            Q = a_burton * (Ey[i-1] - Ec)
        else:
            Q = 0

        # Euler step
        dDst_star = Q - Dst_star[i-1] / tau
        Dst_star[i] = Dst_star[i-1] + dDst_star * dt

        # Convert Dst* back to Dst
        Dst[i] = Dst_star[i] + b_press * np.sqrt(P_dyn[i]) - c_offset

    # Print key results
    print(f"\n    Dst time series (selected hours):")
    print(f"    {'Hour':>6} {'Ey (mV/m)':>12} {'Q (nT/hr)':>12} {'Dst* (nT)':>12} "
          f"{'Dst (nT)':>10}")
    print(f"    {'-'*54}")
    for t in [0, 6, 12, 15, 18, 21, 24, 30, 36, 42, 48]:
        Q_val = a_burton * (Ey[t] - Ec) if Ey[t] > Ec else 0
        print(f"    {t:>6} {Ey[t]:>12.2f} {Q_val:>12.2f} {Dst_star[t]:>12.1f} "
              f"{Dst[t]:>10.1f}")

    Dst_min = np.min(Dst)
    t_min = hours[np.argmin(Dst)]
    print(f"\n    Minimum Dst: {Dst_min:.1f} nT at hour {t_min:.0f}")
    print(f"    (Occurs a few hours after storm onset, consistent with")
    print(f"    the ~{tau} hour decay timescale)")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(hours, v_sw, 'b-', linewidth=1.5)
    axes[0].set_ylabel('v_sw (km/s)')
    axes[0].set_title('Burton Model: Synthetic Storm')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, Bz, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle=':')
    axes[1].set_ylabel('Bz (nT)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hours, P_dyn, 'g-', linewidth=1.5)
    axes[2].set_ylabel('P_dyn (nPa)')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(hours, Dst, 'k-', linewidth=2, label='Dst')
    axes[3].plot(hours, Dst_star, 'k--', linewidth=1, label='Dst*')
    axes[3].axhline(y=0, color='gray', linestyle=':')
    axes[3].set_ylabel('Dst (nT)')
    axes[3].set_xlabel('Time (hours)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # Mark storm onset and min Dst
    for ax in axes:
        ax.axvline(x=12, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(x=25, color='green', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig('/opt/projects/01_Personal/03_Study/exercises/Space_Weather/'
                'ex16_burton_synthetic.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n    Plot saved: ex16_burton_synthetic.png")


def exercise_2():
    """
    Exercise 2: Statistical Evaluation of Dst Prediction

    Outline the methodology for comparing Burton model vs persistence.
    Compute example metrics on synthetic data.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Statistical Evaluation Methodology")
    print("=" * 70)

    # Generate example data to demonstrate metric computation
    np.random.seed(42)
    N = 1000
    Dst_obs = np.random.normal(-20, 30, N)
    # Add some storm-like features
    storm_idx = np.random.choice(N, 50, replace=False)
    Dst_obs[storm_idx] -= 80

    # Burton-like prediction (correlated with observations + noise)
    Dst_burton = Dst_obs + np.random.normal(0, 12, N)

    # Persistence prediction: Dst(t+1) = Dst(t)
    Dst_persist = np.roll(Dst_obs, 1)
    Dst_persist[0] = Dst_obs[0]

    # (a) RMSE
    rmse_burton = np.sqrt(np.mean((Dst_burton - Dst_obs)**2))
    rmse_persist = np.sqrt(np.mean((Dst_persist - Dst_obs)**2))

    print(f"\n    Example evaluation on synthetic data (N={N}):")

    print(f"\n    RMSE:")
    print(f"      Burton model: {rmse_burton:.1f} nT")
    print(f"      Persistence:  {rmse_persist:.1f} nT")

    # (b) MAE
    mae_burton = np.mean(np.abs(Dst_burton - Dst_obs))
    mae_persist = np.mean(np.abs(Dst_persist - Dst_obs))

    print(f"\n    MAE:")
    print(f"      Burton model: {mae_burton:.1f} nT")
    print(f"      Persistence:  {mae_persist:.1f} nT")

    # (c) Correlation
    r_burton = np.corrcoef(Dst_obs, Dst_burton)[0, 1]
    r_persist = np.corrcoef(Dst_obs, Dst_persist)[0, 1]

    print(f"\n    Pearson correlation:")
    print(f"      Burton model: r = {r_burton:.3f}")
    print(f"      Persistence:  r = {r_persist:.3f}")

    # (d) HSS for storm detection (Dst < -50 nT)
    threshold = -50

    def compute_hss(obs, pred, thresh):
        a = np.sum((obs < thresh) & (pred < thresh))  # hits
        b = np.sum((obs >= thresh) & (pred < thresh))  # false alarms
        c = np.sum((obs < thresh) & (pred >= thresh))  # misses
        d = np.sum((obs >= thresh) & (pred >= thresh))  # correct rejections
        denom = (a + c) * (c + d) + (a + b) * (b + d)
        if denom == 0:
            return 0
        return 2 * (a * d - b * c) / denom

    hss_burton = compute_hss(Dst_obs, Dst_burton, threshold)
    hss_persist = compute_hss(Dst_obs, Dst_persist, threshold)

    print(f"\n    HSS (storm detection, Dst < {threshold} nT):")
    print(f"      Burton model: {hss_burton:.3f}")
    print(f"      Persistence:  {hss_persist:.3f}")

    print(f"\n    === METHODOLOGY NOTES ===")
    print(f"    1. Use OMNI hourly data 2010-2020, split: train 2010-2014, test 2015-2020")
    print(f"    2. Burton model integrates the ODE using observed solar wind at L1")
    print(f"    3. Persistence baseline: Dst(t+1) = Dst(t) for 1-hour forecast")
    print(f"    4. Report RMSE, MAE, r, and HSS on the test period")
    print(f"    5. Bin by storm intensity: weak (-30 to -50), moderate (-50 to -100),")
    print(f"       intense (< -100) to identify where Burton excels or fails")
    print(f"    6. Persistence is surprisingly competitive at short lead times")
    print(f"       because Dst changes slowly (autocorrelation > 0.95 at 1 hour)")
    print(f"    7. Burton model should outperform for longer lead times (3-6 hours)")
    print(f"       and during rapid storm development")


def exercise_3():
    """
    Exercise 3: Radiation Belt Storm Simulation

    Solve 1D radial diffusion equation for the 2003 Halloween Storm.
    Demonstrate the numerical approach.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Radiation Belt Radial Diffusion (Demo)")
    print("=" * 70)

    # Set up 1D radial diffusion: df/dt = L^2 d/dL(D_LL/L^2 df/dL) - f/tau
    # Discretize on L grid, use Crank-Nicolson

    # Grid
    L_min, L_max = 2.0, 7.0
    NL = 100
    L = np.linspace(L_min, L_max, NL)
    dL = L[1] - L[0]

    # Time: 12 days at 1-hour resolution
    T_days = 12
    dt_hr = 1
    NT = T_days * 24
    t_hours = np.arange(NT)

    # Synthetic Kp for Halloween 2003 (simplified)
    Kp = np.ones(NT) * 2  # baseline
    # Storm onset at day 4 (hour 96), peak at day 5
    for i in range(96, 144):
        Kp[i] = 2 + 7 * np.exp(-0.5 * ((i - 120) / 12)**2)  # peak Kp ~ 9

    # Brautigam & Albert diffusion coefficient: D_LL = D0 * L^10 * 10^(0.506*Kp - 9.325)
    def D_LL(L_val, Kp_val):
        return 1e-6 * L_val**10 * 10**(0.506 * Kp_val - 9.325)  # day^-1

    # Loss timescale: simplified
    def tau_loss(L_val, Kp_val):
        if L_val < 3.5:
            return 100  # days (stable inner belt)
        elif Kp_val > 5:
            return 0.5  # days (strong loss during storm)
        else:
            return 5    # days (moderate loss)

    # Initial condition: power law in L
    f = 1e-4 * np.exp(-0.5 * (L - 5)**2)  # Gaussian PSD peak near L=5
    f[0] = 1e-6  # inner boundary
    f[-1] = 1e-5  # outer boundary

    # Store snapshots
    snapshots = {0: f.copy()}
    f_at_L = {4.0: [], 5.0: [], 6.0: []}

    print(f"\n    Grid: L = [{L_min}, {L_max}], NL = {NL}, dL = {dL:.3f}")
    print(f"    Time: {T_days} days, dt = {dt_hr} hr, {NT} steps")
    print(f"    Storm peak: ~hour 120 (day 5), Kp ~ 9")

    # Simple explicit Euler (for demonstration; Crank-Nicolson preferred)
    dt_day = dt_hr / 24  # convert to days

    for n in range(NT):
        kp = Kp[n]

        # Compute diffusion term
        f_new = f.copy()
        for i in range(1, NL - 1):
            D_i = D_LL(L[i], kp)
            D_ip = D_LL(L[i] + dL / 2, kp)
            D_im = D_LL(L[i] - dL / 2, kp)

            # d/dL(D_LL/L^2 df/dL) * L^2
            diff = L[i]**2 * (D_ip / (L[i] + dL/2)**2 * (f[i+1] - f[i]) / dL
                               - D_im / (L[i] - dL/2)**2 * (f[i] - f[i-1]) / dL) / dL

            loss = f[i] / tau_loss(L[i], kp)
            f_new[i] = f[i] + dt_day * (diff - loss)

        # Enforce boundaries
        f_new[0] = 1e-6
        f_new[-1] = 1e-5
        f_new = np.maximum(f_new, 1e-10)  # prevent negative PSD
        f = f_new

        # Save snapshots
        if n in [0, 96, 120, 240]:
            snapshots[n] = f.copy()

        # Track specific L values
        for L_track in f_at_L:
            idx = np.argmin(np.abs(L - L_track))
            f_at_L[L_track].append(f[idx])

    # Results
    print(f"\n    Simulation complete.")
    print(f"    Snapshots saved at hours: {list(snapshots.keys())}")

    print(f"\n    PSD at L=5 over time:")
    print(f"    {'Hour':>6} {'Kp':>6} {'f(L=5)':>14}")
    print(f"    {'-'*28}")
    L5_idx = np.argmin(np.abs(L - 5.0))
    for hour in [0, 48, 96, 108, 120, 132, 144, 192, 240]:
        if hour < NT:
            print(f"    {hour:>6} {Kp[hour]:>6.1f} {f_at_L[5.0][hour]:>14.3e}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Kp
    axes[0, 0].plot(t_hours / 24, Kp, 'r-')
    axes[0, 0].set_ylabel('Kp')
    axes[0, 0].set_title('(a) Kp Time Series')
    axes[0, 0].set_xlim(0, T_days)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: PSD profiles at different times
    labels = {0: 'Quiet (t=0)', 96: 'Onset (day 4)', 120: 'Peak (day 5)', 240: 'Recovery (day 10)'}
    colors = {0: 'blue', 96: 'orange', 120: 'red', 240: 'green'}
    for t_snap, f_snap in snapshots.items():
        axes[0, 1].semilogy(L, f_snap, color=colors.get(t_snap, 'black'),
                            label=labels.get(t_snap, f't={t_snap}'))
    axes[0, 1].set_xlabel('L')
    axes[0, 1].set_ylabel('f (PSD)')
    axes[0, 1].set_title('(b) PSD Profiles')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: f(t) at fixed L
    for L_track, f_trace in f_at_L.items():
        axes[1, 0].semilogy(np.arange(len(f_trace)) / 24, f_trace,
                            label=f'L={L_track}')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('f (PSD)')
    axes[1, 0].set_title('(c) PSD vs Time at Fixed L')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: D_LL profiles
    for kp_val in [2, 5, 8]:
        D_vals = [D_LL(l, kp_val) for l in L]
        axes[1, 1].semilogy(L, D_vals, label=f'Kp={kp_val}')
    axes[1, 1].set_xlabel('L')
    axes[1, 1].set_ylabel('D_LL (day^-1)')
    axes[1, 1].set_title('(d) Diffusion Coefficient')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('/opt/projects/01_Personal/03_Study/exercises/Space_Weather/'
                'ex16_radiation_belt_sim.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n    Plot saved: ex16_radiation_belt_sim.png")


def exercise_4():
    """
    Exercise 4: Space Weather Dashboard Concept

    Outline the dashboard construction for the 2015 St. Patrick's Day Storm.
    Demonstrate key calculations (Shue magnetopause, auroral boundary).
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Space Weather Dashboard Components")
    print("=" * 70)

    print(f"\n    Dashboard for 2015 St. Patrick's Day Storm (March 14-22)")
    print(f"    5 panels: solar wind, magnetopause, Dst, aurora, NOAA scales")

    # Demonstrate key calculations with example storm values
    # Peak storm conditions: v~630, n~35, Bz~-25, Dst~-223, Kp=8

    print(f"\n    === Panel (b): Shue Magnetopause Model ===")
    # Shue et al. (1998): r0 = (10.22 + 1.29*tanh(0.184*(Bz+8.14))) * Dp^(-1/6.6)
    Bz_vals = [2, -5, -15, -25]
    Dp_vals = [2, 5, 15, 35]
    m_p = 1.67e-27

    print(f"    r0 = (10.22 + 1.29*tanh(0.184*(Bz+8.14))) * Dp^(-1/6.6)")
    print(f"\n    {'Bz (nT)':>10} {'Dp (nPa)':>10} {'R0 (R_E)':>10} {'GEO exposed?':>14}")
    print(f"    {'-'*46}")
    for Bz, n in zip(Bz_vals, Dp_vals):
        v = 500  # approximate
        Dp = 0.5 * n * 1e6 * m_p * (v * 1e3)**2 * 1e9
        r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Dp**(-1/6.6)
        exposed = "YES" if r0 < 6.6 else "no"
        print(f"    {Bz:>10} {Dp:>10.1f} {r0:>10.1f} {exposed:>14}")

    print(f"\n    === Panel (d): Auroral Boundary (Feldstein) ===")
    print(f"    Lambda_eq = 67 - 2 * Kp (degrees geomagnetic latitude)")
    print(f"\n    {'Kp':>6} {'Lambda_eq (deg)':>16} {'Approximate geographic':>22}")
    print(f"    {'-'*46}")
    for Kp in [2, 4, 5, 6, 7, 8, 9]:
        Lambda = 67 - 2 * Kp
        # Very rough geographic correction (depends on location)
        geo_lat = Lambda - 5  # rough offset for North America
        print(f"    {Kp:>6} {Lambda:>16.0f} {geo_lat:>18.0f} N (approx)")

    print(f"\n    === Panel (e): NOAA G-Scale from Kp ===")
    g_scale = {5: "G1-Minor", 6: "G2-Moderate", 7: "G3-Strong",
               8: "G4-Severe", 9: "G5-Extreme"}
    print(f"    {'Kp':>6} {'G Level':>18}")
    print(f"    {'-'*26}")
    for kp, g in g_scale.items():
        print(f"    {kp:>6} {g:>18}")

    print(f"\n    For the St. Patrick's Day Storm (Kp=8, Dst=-223 nT):")
    print(f"    G4 (Severe), auroral boundary at ~51 deg geomagnetic latitude")
    print(f"    Aurora visible from much of the northern US and central Europe")


def exercise_5():
    """
    Exercise 5: Coupling Function Comparison

    Compare Ey, Newell, and Akasofu epsilon as Burton model drivers.
    Demonstrate computation for sample solar wind conditions.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Coupling Function Comparison")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    R_E = 6.371e6
    l0 = 7 * R_E  # m

    # Sample solar wind conditions
    conditions = [
        {"label": "Quiet",    "v": 400, "By": 2, "Bz": 3},
        {"label": "Moderate", "v": 500, "By": 5, "Bz": -8},
        {"label": "Strong",   "v": 600, "By": 8, "Bz": -15},
        {"label": "Extreme",  "v": 800, "By": 12, "Bz": -30},
    ]

    print(f"\n    Coupling functions compared for different solar wind conditions:")
    print(f"\n    {'Condition':<12} {'v':>6} {'By':>6} {'Bz':>6} "
          f"{'Ey':>10} {'Newell':>12} {'Epsilon':>14}")
    print(f"    {'':12} {'km/s':>6} {'nT':>6} {'nT':>6} "
          f"{'mV/m':>10} {'(arb)':>12} {'(GW)':>14}")
    print(f"    {'-'*68}")

    for cond in conditions:
        v = cond["v"]       # km/s
        By = cond["By"]     # nT
        Bz = cond["Bz"]     # nT

        # 1. Ey = -v * Bz (standard Burton driver)
        Bs = abs(Bz) if Bz < 0 else 0  # southward component only
        Ey = v * Bs * 1e-3  # mV/m

        # 2. Newell: dPhi/dt = v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)
        B_T = np.sqrt(By**2 + Bz**2)  # nT
        theta_c = np.arctan2(By, Bz)
        if theta_c < 0:
            theta_c += 2 * np.pi
        sin_83 = abs(np.sin(theta_c / 2))**(8/3)
        newell = v**(4/3) * B_T**(2/3) * sin_83

        # 3. Akasofu epsilon = (4*pi/mu0) * v * B^2 * sin^4(theta_c/2) * l0^2
        v_ms = v * 1e3  # m/s
        B_T_T = B_T * 1e-9  # T
        sin4 = np.sin(theta_c / 2)**4
        epsilon = (4 * np.pi / mu0) * v_ms * B_T_T**2 * sin4 * l0**2
        epsilon_GW = epsilon * 1e-9

        print(f"    {cond['label']:<12} {v:>6} {By:>6} {Bz:>6} "
              f"{Ey:>10.2f} {newell:>12.0f} {epsilon_GW:>14.1f}")

    print(f"\n    === ANALYSIS ===")
    print(f"    Ey (= v*Bs):")
    print(f"    - Simplest coupling function, uses only southward Bz component")
    print(f"    - Zero for northward Bz (misses By contributions)")
    print(f"    - Linear in v and Bz; may underpredict for extreme conditions")

    print(f"\n    Newell dPhi/dt:")
    print(f"    - Includes clock angle dependence via sin^(8/3)(theta_c/2)")
    print(f"    - Non-zero even for northward Bz with large By")
    print(f"    - Best single-parameter predictor of magnetospheric activity")
    print(f"    - Weighted more towards clock angle than Ey")

    print(f"\n    Akasofu epsilon:")
    print(f"    - Has units of power (watts)")
    print(f"    - Includes l0^2 coupling length (empirical)")
    print(f"    - sin^4(theta_c/2) weighting: less aggressive than Newell's sin^(8/3)")
    print(f"    - Best for total energy input estimation")

    print(f"\n    For Burton model fitting:")
    print(f"    - Each coupling function requires re-fitting the injection efficiency 'a'")
    print(f"    - Newell-driven model typically gives lowest RMSE for intense storms")
    print(f"    - Ey-driven model is simplest and most robust for moderate storms")
    print(f"    - Epsilon gives good overall energy balance but less precise Dst shape")
    print(f"    - Best approach: test all three and compare RMSE by storm category")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
