"""
Exercise Solutions for Lesson 15: AI/ML for Space Weather

Topics covered:
  - LSTM model design for Dst prediction
  - Flare prediction verification metrics (class imbalance)
  - Physics-informed neural network (PINN) concept
  - CME ensemble arrival time and decision-making
  - Solar cycle generalization problem for ML models
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: LSTM Model Design for Dst Prediction

    Training data: OMNI 1995-2015. Design LSTM for 3-hour Dst forecast.
    Specify: inputs, window, architecture, loss, train/test split.
    Estimate storm events (Dst < -100) in training set.
    """
    print("=" * 70)
    print("Exercise 1: LSTM Design for 3-Hour Dst Prediction")
    print("=" * 70)

    print(f"\n    === MODEL DESIGN ===")

    print(f"\n    1. INPUT FEATURES (hourly OMNI data):")
    features = [
        ("v_sw", "Solar wind speed (km/s)"),
        ("n_sw", "Solar wind density (cm^-3)"),
        ("P_dyn", "Dynamic pressure (nPa)"),
        ("Bz_GSM", "IMF Bz in GSM (nT) - key driver"),
        ("By_GSM", "IMF By in GSM (nT) - clock angle"),
        ("|B|", "Total IMF magnitude (nT)"),
        ("E_y", "Dawn-dusk electric field v*Bz (mV/m)"),
        ("Dst(t)", "Current and recent Dst values (nT)"),
        ("SYM-H", "Minute-resolution Dst proxy if available"),
    ]
    for name, desc in features:
        print(f"       {name:<12} : {desc}")
    print(f"       Total: {len(features)} features per timestep")

    print(f"\n    2. INPUT WINDOW:")
    print(f"       Window size: 24 hours (24 timesteps at hourly resolution)")
    print(f"       Justification: Storm main phase typically lasts 6-12 hours;")
    print(f"       24 hours captures the context including pre-storm conditions")
    print(f"       and allows the LSTM to learn storm development patterns")

    print(f"\n    3. ARCHITECTURE:")
    print(f"       Input:  (batch, 24, {len(features)}) tensor")
    print(f"       Layer 1: LSTM(hidden=64, return_sequences=True)")
    print(f"       Layer 2: LSTM(hidden=32, return_sequences=False)")
    print(f"       Layer 3: Dense(16, activation=ReLU)")
    print(f"       Layer 4: Dense(1, activation=linear) -> Dst(t+3)")
    print(f"       Dropout: 0.2 between LSTM layers (prevent overfitting)")
    print(f"       Total parameters: ~25,000-50,000 (moderate complexity)")

    print(f"\n    4. LOSS FUNCTION:")
    print(f"       Primary: MSE (Mean Squared Error) on Dst")
    print(f"       Alternative: Weighted MSE with higher weight for storm hours")
    print(f"       (Dst < -50 nT weighted 3x; this prevents the model from")
    print(f"       ignoring rare but important storm periods)")
    print(f"       Optimizer: Adam(lr=1e-3) with ReduceLROnPlateau scheduler")

    print(f"\n    5. TRAIN/TEST SPLIT:")
    print(f"       Training: 1995-2009 (Solar cycles 23 ascending to 24 minimum)")
    print(f"       Validation: 2010-2012 (Solar cycle 24 ascending)")
    print(f"       Test: 2013-2015 (Solar cycle 24 maximum)")
    print(f"       IMPORTANT: Temporal split, NOT random split!")
    print(f"       Random splits would leak future information into training.")

    # Storm count estimate
    print(f"\n    6. STORM EVENT COUNT ESTIMATE:")
    print(f"       Intense storms (Dst < -100 nT): ~30 per solar cycle")
    print(f"       Training period covers ~1.8 solar cycles (1995-2015)")
    n_storms = int(30 * 1.8)
    print(f"       Estimated intense storms in training: ~{n_storms}")
    print(f"       This is borderline for deep learning — the model has")
    print(f"       only ~{n_storms} examples of extreme behavior to learn from.")
    print(f"       Data augmentation or physics-informed approaches can help.")


def exercise_2():
    """
    Exercise 2: Flare Prediction Metrics

    Random forest on SHARP features, test set of 10,000 AR-days:
    Hits=35, Misses=15, False Alarms=50, Correct Rejections=9900.
    Calculate metrics. Discuss TSS vs accuracy and threshold trade-offs.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Flare Prediction Verification Metrics")
    print("=" * 70)

    a = 35     # hits
    b = 50     # false alarms
    c = 15     # misses
    d = 9900   # correct rejections
    N = a + b + c + d

    POD = a / (a + c)
    FAR = b / (a + b)
    POFD = b / (b + d)
    TSS = POD - POFD
    HSS_num = 2 * (a * d - b * c)
    HSS_den = (a + c) * (c + d) + (a + b) * (b + d)
    HSS = HSS_num / HSS_den
    ACC = (a + d) / N

    print(f"\n    Contingency Table:")
    print(f"    {'':>20} {'Flare YES':>12} {'Flare NO':>12}")
    print(f"    {'Predicted YES':>20} {a:>12} {b:>12}")
    print(f"    {'Predicted NO':>20} {c:>12} {d:>12}")

    print(f"\n    Metrics:")
    print(f"    POD  = a/(a+c) = {a}/{a+c} = {POD:.3f} ({POD*100:.0f}%)")
    print(f"    FAR  = b/(a+b) = {b}/{a+b} = {FAR:.3f} ({FAR*100:.0f}%)")
    print(f"    POFD = b/(b+d) = {b}/{b+d} = {POFD:.5f}")
    print(f"    TSS  = POD - POFD = {TSS:.3f}")
    print(f"    HSS  = {HSS:.3f}")
    print(f"    Accuracy = (a+d)/N = {a+d}/{N} = {ACC:.4f} ({ACC*100:.2f}%)")

    print(f"\n    Why TSS > Accuracy for forecasters:")
    print(f"    A 'no flare' forecast gives accuracy = {(c+d)/N*100:.2f}% (trivially high)")
    print(f"    TSS of {TSS:.3f} indicates genuine skill in detecting flares")
    print(f"    while {ACC*100:.2f}% accuracy is misleading (dominated by {d} CR)")

    # Threshold trade-off: increase POD to 0.90
    print(f"\n    Threshold trade-off (increase POD to 0.90):")
    # If POD goes to 0.90: we detect 45 out of 50 flares (a=45, c=5)
    # But lowering threshold means more false alarms
    a_new = 45   # 0.90 * 50
    c_new = 5
    # Estimate: typically false alarms increase by 3-5x when POD increases this much
    b_new = b * 3  # rough estimate: 150 false alarms
    d_new = N - a_new - b_new - c_new

    POD_new = a_new / (a_new + c_new)
    FAR_new = b_new / (a_new + b_new)
    TSS_new = POD_new - b_new / (b_new + d_new)

    print(f"    New estimate: a={a_new}, b~{b_new}, c={c_new}, d~{d_new}")
    print(f"    POD = {POD_new:.2f}")
    print(f"    FAR = {FAR_new:.2f}")
    print(f"    TSS = {TSS_new:.3f}")
    print(f"    Trade-off: POD improves {POD:.2f} -> {POD_new:.2f}")
    print(f"               FAR worsens  {FAR:.2f} -> {FAR_new:.2f}")
    print(f"    This is the fundamental detection vs false alarm trade-off.")
    print(f"    For high-consequence events (e.g., EVA safety), high POD is worth")
    print(f"    the cost of more false alarms.")


def exercise_3():
    """
    Exercise 3: Physics-Informed Neural Network (PINN) Concept

    PINN for radiation belt radial diffusion equation.
    Define the three loss components. Discuss advantage with sparse data.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: PINN for Radiation Belt Diffusion")
    print("=" * 70)

    print(f"\n    Radiation belt radial diffusion equation:")
    print(f"    df/dt = L^2 * d/dL(D_LL/L^2 * df/dL) - f/tau")
    print(f"    f = phase space density, L = L-shell, D_LL = diffusion coeff,")
    print(f"    tau = loss timescale")

    print(f"\n    PINN Architecture:")
    print(f"    Neural network: (L, t) -> f_hat(L, t)")
    print(f"    The NN takes L-shell and time as inputs and outputs PSD.")
    print(f"    Automatic differentiation computes df/dt, df/dL, d^2f/dL^2.")

    print(f"\n    === THREE LOSS COMPONENTS ===")

    print(f"\n    1. L_data (Data fidelity loss):")
    print(f"       L_data = (1/N_d) * SUM_i [f_hat(L_i, t_i) - f_obs(L_i, t_i)]^2")
    print(f"       - Matches the NN output to observed PSD at measurement points")
    print(f"       - N_d data points from satellite observations (e.g., Van Allen Probes)")
    print(f"       - Measurements available at only 2-3 L values at any given time")

    print(f"\n    2. L_physics (Physics residual loss):")
    print(f"       R(L,t) = df_hat/dt - L^2*d/dL(D_LL/L^2*df_hat/dL) + f_hat/tau")
    print(f"       L_physics = (1/N_p) * SUM_j [R(L_j, t_j)]^2")
    print(f"       - Evaluated at N_p collocation points in the (L,t) domain")
    print(f"       - These points do NOT need observations — they enforce physics")
    print(f"       - N_p >> N_d (can use thousands of collocation points)")
    print(f"       - D_LL(L, Kp) and tau(L, Kp) are parameterized from empirical models")

    print(f"\n    3. L_BC (Boundary/initial condition loss):")
    print(f"       L_BC = (1/N_b) * SUM_k [f_hat(L_k, t_k) - f_BC(L_k, t_k)]^2")
    print(f"       Boundary conditions:")
    print(f"       - Inner boundary (L ~ 2): f = f_inner (from inner belt observations)")
    print(f"       - Outer boundary (L ~ 7): f = f_outer (from plasma sheet)")
    print(f"       - Initial condition: f(L, t=0) = f_0(L) (from observations)")

    print(f"\n    Total loss: L = w_d * L_data + w_p * L_physics + w_b * L_BC")
    print(f"    where w_d, w_p, w_b are weights balancing the three terms.")

    print(f"\n    === ADVANTAGE WITH SPARSE DATA ===")
    print(f"    - Satellites provide measurements at only 2-3 L values at any time")
    print(f"    - Pure data-driven model would need dense spatial coverage")
    print(f"    - PINN fills the gaps using physics: the diffusion equation")
    print(f"      constrains the solution between measurement points")
    print(f"    - The physics term provides 'virtual data' everywhere in the domain")
    print(f"    - This is especially valuable for radiation belts where spatial")
    print(f"      coverage is inherently sparse (few satellites crossing many L-shells)")
    print(f"    - PINN can also handle missing data gracefully: fewer data points")
    print(f"      simply reduce L_data weight, but L_physics still constrains solution")


def exercise_4():
    """
    Exercise 4: CME Ensemble Arrival Time

    Ensemble: N=500 DBM runs. Mean arrival = 48 hr, std = 8 hr (Gaussian).
    Satellite operator needs 6 hours for maneuver.
    At what time should they begin to have 90% confidence?
    """
    print("\n" + "=" * 70)
    print("Exercise 4: CME Ensemble Arrival Time Decision")
    print("=" * 70)

    mu = 48      # hours (mean arrival time)
    sigma = 8    # hours (std dev)
    t_maneuver = 6  # hours to complete maneuver
    confidence = 0.90

    print(f"\n    Ensemble: N=500 runs, mean = {mu} hr, std = {sigma} hr (Gaussian)")
    print(f"    Maneuver duration: {t_maneuver} hours")
    print(f"    Required confidence: {confidence*100:.0f}% that maneuver completes before CME")

    # Need P(T_arrival > t_begin + t_maneuver) >= 0.90
    # P(T > t_complete) >= 0.90
    # P(T <= t_complete) <= 0.10
    # t_complete such that CDF(t_complete) = 0.10
    # t_complete = mu + sigma * z_0.10
    # z_0.10 = -1.282 (10th percentile of standard normal)

    from scipy.stats import norm
    z = norm.ppf(1 - confidence)  # = norm.ppf(0.10) = -1.282
    t_complete = mu + sigma * z
    t_begin = t_complete - t_maneuver

    print(f"\n    Analysis:")
    print(f"    Need: P(T_arrival > t_begin + {t_maneuver}) >= {confidence}")
    print(f"    Equivalently: P(T_arrival <= t_complete) <= {1-confidence}")
    print(f"    10th percentile of arrival distribution:")
    print(f"    t_10% = mu + sigma * z_0.10 = {mu} + {sigma} * ({z:.3f})")
    print(f"    t_10% = {t_complete:.1f} hours after CME observation")

    print(f"\n    Begin maneuver at: t = t_10% - t_maneuver")
    print(f"    = {t_complete:.1f} - {t_maneuver} = {t_begin:.1f} hours after CME")

    print(f"\n    DECISION: Begin maneuver at t = {t_begin:.1f} hours")
    print(f"    (i.e., {t_begin:.0f} hours after CME observation)")
    print(f"    This ensures {confidence*100:.0f}% probability that the maneuver")
    print(f"    completes before CME arrival.")

    # Show confidence for different start times
    print(f"\n    Sensitivity analysis:")
    print(f"    {'Start (hr)':>12} {'Complete (hr)':>14} {'P(safe)':>10}")
    print(f"    {'-'*38}")
    for t_start in [24, 28, 32, 36, t_begin, 40]:
        t_comp = t_start + t_maneuver
        p_safe = 1 - norm.cdf(t_comp, mu, sigma)
        marker = " <-- 90%" if abs(t_start - t_begin) < 0.5 else ""
        print(f"    {t_start:>12.1f} {t_comp:>14.1f} {p_safe:>10.3f}{marker}")

    print(f"\n    Assumptions:")
    print(f"    1. Arrival time distribution is Gaussian (may have heavy tails)")
    print(f"    2. All 500 DBM runs are equally likely (unweighted ensemble)")
    print(f"    3. Maneuver takes exactly {t_maneuver} hours (no uncertainty)")
    print(f"    4. No additional information updates the forecast during the wait")


def exercise_5():
    """
    Exercise 5: Solar Cycle Generalization Problem

    Model trained on cycle 23 degrades on cycle 24. Identify physical reasons
    and propose strategies.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Solar Cycle Generalization Problem")
    print("=" * 70)

    print(f"\n    Scenario:")
    print(f"    LSTM Dst model trained on Solar Cycle 23 (1996-2008)")
    print(f"    Test on held-out portion of Cycle 23: Excellent (RMSE ~ 8 nT)")
    print(f"    Test on Solar Cycle 24 (2008-2019): Degraded (RMSE ~ 15 nT)")

    print(f"\n    === PHYSICAL REASONS FOR DEGRADATION ===")

    print(f"\n    1. Different solar activity level:")
    print(f"       Cycle 23 maximum SSN ~ 180; Cycle 24 maximum SSN ~ 120")
    print(f"       Cycle 24 was ~33% weaker than Cycle 23")
    print(f"       The model trained on stronger storms may not generalize to")
    print(f"       the weaker driving conditions of Cycle 24")
    print(f"       (Different baseline F10.7, different EUV flux, different")
    print(f"       thermospheric state)")

    print(f"\n    2. Different CME/CIR properties:")
    print(f"       Cycle 24 had fewer fast CMEs but relatively more CIR storms")
    print(f"       CME-driven storms have rapid onset and single main phase")
    print(f"       CIR-driven storms have gradual onset and multiple injections")
    print(f"       The model's learned storm morphology from Cycle 23 may not")
    print(f"       match the different storm types prevalent in Cycle 24")

    print(f"\n    3. Changed ring current composition:")
    print(f"       The O+/H+ ratio in the ring current varies with solar cycle")
    print(f"       Higher O+ during stronger solar activity changes the DPS")
    print(f"       relation and recovery timescales")
    print(f"       Different composition means different Dst decay characteristics")
    print(f"       that the model hasn't seen")

    print(f"\n    4. Different instrument calibrations:")
    print(f"       ACE (primary L1 monitor in Cycle 23) vs DSCOVR (Cycle 24)")
    print(f"       Subtle calibration differences in solar wind parameters")
    print(f"       can systematically shift model inputs")

    print(f"\n    5. Secular changes:")
    print(f"       Earth's magnetic field is slowly changing (dipole weakening)")
    print(f"       Magnetopause standoff distance shifts slightly over decades")
    print(f"       These are small but can affect edge-case predictions")

    print(f"\n    === STRATEGIES TO IMPROVE GENERALIZATION ===")

    print(f"\n    Strategy 1: Physics-informed features and architecture")
    print(f"    - Use dimensionless coupling functions (e.g., Newell's dPhi/dt)")
    print(f"      instead of raw solar wind parameters")
    print(f"    - These functions encode physical relationships that are")
    print(f"      more stable across solar cycles")
    print(f"    - Include solar cycle phase as an explicit input feature")
    print(f"    - Use a physics-guided architecture (e.g., encode the Burton")
    print(f"      equation structure as an inductive bias)")

    print(f"\n    Strategy 2: Multi-cycle training with domain adaptation")
    print(f"    - Train on data from Cycles 21-23 (spanning 3 cycles)")
    print(f"    - Use domain adaptation techniques: train on the source")
    print(f"      distribution but fine-tune on a small amount of Cycle 24 data")
    print(f"    - Apply sliding window retraining: periodically update the")
    print(f"      model as new Cycle 24 data becomes available")
    print(f"    - Ensemble models trained on different cycles and weight by")
    print(f"      similarity to current conditions")

    print(f"\n    Key insight: Pure data-driven models implicitly assume the")
    print(f"    training and test distributions are the same. The solar cycle")
    print(f"    violates this assumption. Physics-informed approaches are")
    print(f"    inherently more robust because physical laws do not change")
    print(f"    between cycles — only the driving conditions do.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
