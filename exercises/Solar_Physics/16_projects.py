"""
Exercises for Lesson 16: Projects
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
R_sun = 6.957e8        # solar radius [m]


def exercise_1():
    """
    Problem 1: TSI Model Calculation

    TSI(t) = 1360.5 + 0.008 * R(t) W/m^2
    (a) Solar minimum R = 5.
    (b) Moderate max R = 120.
    (c) Strong max R = 200.
    Peak-to-peak variation, percentage, and radiative forcing.
    """
    S_0 = 1360.5  # W/m^2 (baseline)
    coeff = 0.008  # W/m^2 per sunspot number

    sunspot_cases = [
        ("Solar minimum", 5),
        ("Moderate maximum", 120),
        ("Strong maximum", 200),
    ]

    print(f"  TSI model: TSI(t) = {S_0} + {coeff} * R(t) W/m^2")
    print()

    TSI_values = []
    for label, R in sunspot_cases:
        TSI = S_0 + coeff * R
        TSI_values.append(TSI)
        print(f"  ({chr(97 + sunspot_cases.index((label, R)))}) {label} (R = {R}):")
        print(f"      TSI = {S_0} + {coeff} * {R} = {TSI:.2f} W/m^2")

    # Peak-to-peak variation
    TSI_min = TSI_values[0]
    TSI_max = TSI_values[-1]
    variation = TSI_max - TSI_min
    variation_pct = variation / S_0 * 100

    print(f"\n  Peak-to-peak variation:")
    print(f"    TSI(max) - TSI(min) = {TSI_max:.2f} - {TSI_min:.2f} = {variation:.2f} W/m^2")
    print(f"    As percentage of S_0: {variation_pct:.3f}%")

    # Between min and moderate max
    var_mod = TSI_values[1] - TSI_values[0]
    print(f"    Between min and moderate max: {var_mod:.2f} W/m^2 ({var_mod/S_0*100:.3f}%)")

    # Radiative forcing
    # Global average forcing change = factor * delta_TSI
    factor = 0.175
    forcing_total = factor * variation
    forcing_mod = factor * var_mod

    print(f"\n  Global average radiative forcing (factor = {factor}):")
    print(f"    Min to strong max: {factor} * {variation:.2f} = {forcing_total:.3f} W/m^2")
    print(f"    Min to moderate max: {factor} * {var_mod:.2f} = {forcing_mod:.3f} W/m^2")
    print(f"    ")
    print(f"    For comparison, anthropogenic CO2 forcing since 1750: ~2.0 W/m^2")
    print(f"    Solar cycle forcing ({forcing_mod:.3f} W/m^2) is ~{forcing_mod/2.0*100:.0f}% of CO2 forcing.")
    print(f"    The solar cycle effect on climate is detectable but small compared")
    print(f"    to anthropogenic greenhouse forcing.")


def exercise_2():
    """
    Problem 2: PFSS Open Flux

    Dipole field: B_r(R_ss) = B_0 * (R_sun/R_ss)^3 * cos(theta)
    R_ss = 2.5 R_sun.
    Phi_open = integral |B_r(R_ss)| dA_ss.
    B_0 = 5 G. Compare with ~3e22 Mx.
    """
    R_ss = 2.5     # in R_sun
    B_0 = 5.0      # G (polar field strength)
    R_ss_cm = R_ss * R_sun * 100  # cm

    # B_r(R_ss, theta) = B_0 * (1/R_ss)^3 * cos(theta)
    # (B_0 is the field at the pole at the solar surface)
    B_factor = B_0 * (1.0 / R_ss)**3  # G

    print(f"  PFSS dipole model:")
    print(f"    Source surface: R_ss = {R_ss} R_sun")
    print(f"    Polar field: B_0 = {B_0} G")
    print(f"    B_r(R_ss, theta) = B_0 * (R_sun/R_ss)^3 * cos(theta)")
    print(f"                     = {B_0} * (1/{R_ss})^3 * cos(theta)")
    print(f"                     = {B_factor:.3f} * cos(theta) G")

    # Open flux: Phi_open = integral |B_r| dA over source surface
    # dA = R_ss^2 * sin(theta) * dtheta * dphi
    # Phi_open = integral_0^2pi dphi * integral_0^pi |B_factor * cos(theta)| * R_ss^2 sin(theta) dtheta
    # = 2pi * R_ss_cm^2 * B_factor * integral_0^pi |cos(theta)| sin(theta) dtheta
    # integral_0^pi |cos(theta)| sin(theta) dtheta = 2 * integral_0^(pi/2) cos(theta) sin(theta) dtheta
    # = 2 * [sin^2(theta)/2]_0^(pi/2) = 2 * 1/2 = 1

    Phi_open = 2.0 * np.pi * R_ss_cm**2 * B_factor * 1.0  # Mx (since B in G, area in cm^2)

    print(f"\n  Open flux calculation:")
    print(f"    Phi_open = 2 pi R_ss^2 * B_factor * integral |cos(theta)| sin(theta) dtheta")
    print(f"    The integral from 0 to pi of |cos(theta)| sin(theta) dtheta = 1")
    print(f"    R_ss = {R_ss} R_sun = {R_ss_cm:.3e} cm")
    print(f"    Phi_open = 2 pi * ({R_ss_cm:.3e})^2 * {B_factor:.3f}")
    print(f"             = {Phi_open:.2e} Mx")

    # Compare with observation
    Phi_obs = 3.0e22  # Mx
    ratio = Phi_open / Phi_obs
    print(f"\n  Observed total open flux: ~{Phi_obs:.0e} Mx")
    print(f"  PFSS dipole estimate: {Phi_open:.2e} Mx")
    print(f"  Ratio: {ratio:.2f}")

    if ratio < 1:
        print(f"  The simple dipole underestimates the open flux by a factor ~{1/ratio:.1f}.")
        print(f"  This is because:")
        print(f"  - The real Sun has higher-order multipoles contributing to open flux")
        print(f"  - Active regions add significant flux, especially near solar maximum")
        print(f"  - The polar field of 5 G is a quiet-Sun minimum value")
    else:
        print(f"  The estimate is consistent with (or exceeds) the observed value.")

    # Also show the formula in terms of B_0 and R_sun
    print(f"\n  In terms of B_0 and R_sun:")
    print(f"    Phi_open = 2 pi B_0 R_sun^2 / R_ss")
    Phi_formula = 2.0 * np.pi * B_0 * (R_sun * 100)**2 / R_ss
    print(f"             = {Phi_formula:.2e} Mx (same result)")


def exercise_3():
    """
    Problem 3: Flare Prediction with Logistic Regression

    SHARP parameters: Phi = 5e22 Mx, R-value = 3e21 Mx, mean shear = 45 deg.
    Logistic: log(P/(1-P)) = -5.0 + 2.0*log10(R/1e20) + 0.05*psi_bar.
    Calculate probability, compare with climatological rate 3%.
    """
    Phi = 5.0e22      # total unsigned flux [Mx]
    R_value = 3.0e21  # R-value [Mx]
    psi_bar = 45.0    # mean shear angle [degrees]
    p_climate = 0.03  # climatological M-class rate

    # Logistic regression
    # log(P/(1-P)) = -5.0 + 2.0 * log10(R/1e20) + 0.05 * psi_bar
    log_R = np.log10(R_value / 1.0e20)
    logit = -5.0 + 2.0 * log_R + 0.05 * psi_bar

    # Sigmoid: P = 1 / (1 + exp(-logit))
    P = 1.0 / (1.0 + np.exp(-logit))

    print(f"  Active region SHARP parameters:")
    print(f"    Total unsigned flux: Phi = {Phi:.0e} Mx")
    print(f"    R-value: R = {R_value:.0e} Mx")
    print(f"    Mean shear angle: psi_bar = {psi_bar:.0f} deg")

    print(f"\n  Logistic regression model:")
    print(f"    log(P/(1-P)) = -5.0 + 2.0*log10(R/1e20) + 0.05*psi_bar")
    print(f"    log10(R/1e20) = log10({R_value:.0e}/1e20) = {log_R:.3f}")
    print(f"    logit = -5.0 + 2.0*{log_R:.3f} + 0.05*{psi_bar:.0f}")
    print(f"          = -5.0 + {2.0*log_R:.3f} + {0.05*psi_bar:.2f}")
    print(f"          = {logit:.3f}")

    print(f"\n    P = 1 / (1 + exp(-{logit:.3f}))")
    print(f"      = 1 / (1 + {np.exp(-logit):.3f})")
    print(f"      = {P:.4f} = {P*100:.2f}%")

    print(f"\n  Climatological rate: {p_climate*100:.0f}%")
    print(f"  Model prediction: {P*100:.2f}%")
    ratio = P / p_climate
    print(f"  Ratio: {ratio:.1f}x climatological")

    if P > p_climate:
        print(f"  YES - this active region has ELEVATED risk ({ratio:.1f}x background).")
        if P > 0.5:
            print(f"  The probability exceeds 50% -- flare is more likely than not!")
        elif P > 0.1:
            print(f"  The probability is significantly above background.")
    else:
        print(f"  NO - this region does not show elevated risk compared to background.")


def exercise_4():
    """
    Problem 4: Flare Prediction Model Evaluation

    Test set: 1000 AR-days (30 flaring, 970 non-flaring).
    Model 1: TP=20, FN=10, FP=100, TN=870.
    Model 2: TP=25, FN=5, FP=200, TN=770.
    Calculate metrics and compare.
    """
    models = {
        "Model 1": {"TP": 20, "FN": 10, "FP": 100, "TN": 870},
        "Model 2": {"TP": 25, "FN": 5, "FP": 200, "TN": 770},
    }

    total = 1000
    pos = 30
    neg = 970

    print(f"  Test set: {total} AR-days ({pos} flaring, {neg} non-flaring)")

    for name, m in models.items():
        TP = m["TP"]
        FN = m["FN"]
        FP = m["FP"]
        TN = m["TN"]

        # Verify: TP + FN = positives, FP + TN = negatives
        assert TP + FN == pos, f"Check failed for {name}"
        assert FP + TN == neg, f"Check failed for {name}"

        # Metrics
        accuracy = (TP + TN) / total
        TPR = TP / (TP + FN)       # True Positive Rate (Recall, Hit Rate)
        FPR = FP / (FP + TN)       # False Positive Rate (False Alarm Rate)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1 = 2 * precision * TPR / (precision + TPR) if (precision + TPR) > 0 else 0
        TSS = TPR - FPR            # True Skill Statistic (HSS without bias correction)

        print(f"\n  {name}: TP={TP}, FN={FN}, FP={FP}, TN={TN}")
        print(f"    Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"    TPR (Recall): {TPR:.3f} ({TPR*100:.1f}%)")
        print(f"    FPR (FAR):    {FPR:.3f} ({FPR*100:.1f}%)")
        print(f"    Precision:    {precision:.3f} ({precision*100:.1f}%)")
        print(f"    F1 Score:     {F1:.3f}")
        print(f"    TSS:          {TSS:.3f}")

    # Comparison
    print(f"\n  Comparison:")
    m1 = models["Model 1"]
    m2 = models["Model 2"]

    TSS_1 = m1["TP"]/(m1["TP"]+m1["FN"]) - m1["FP"]/(m1["FP"]+m1["TN"])
    TSS_2 = m2["TP"]/(m2["TP"]+m2["FN"]) - m2["FP"]/(m2["FP"]+m2["TN"])

    print(f"    TSS: Model 1 = {TSS_1:.3f}, Model 2 = {TSS_2:.3f}")

    if TSS_1 > TSS_2:
        print(f"    Model 1 has HIGHER TSS (better overall skill).")
    else:
        print(f"    Model 2 has HIGHER TSS (better overall skill).")

    print(f"\n  Operational considerations:")
    print(f"    - Model 2 catches more flares (TPR: {m2['TP']/pos:.0%} vs {m1['TP']/pos:.0%})")
    print(f"      This is critical if missing flares has severe consequences")
    print(f"    - Model 1 has fewer false alarms (FPR: {m1['FP']/neg:.1%} vs {m2['FP']/neg:.1%})")
    print(f"      Important for operational fatigue and resource allocation")
    print(f"    - Model 1 has better precision ({m1['TP']/(m1['TP']+m1['FP']):.0%} vs {m2['TP']/(m2['TP']+m2['FP']):.0%})")
    print(f"      When a warning is issued, it's more likely correct")
    print(f"    - TSS (recommended metric for rare events) accounts for both")
    print(f"      hit rate and false alarm rate, is unbiased by event frequency")
    print(f"    - The choice depends on the cost ratio of misses vs false alarms:")
    print(f"      * For astronaut safety: Model 2 (maximize TPR, tolerate FP)")
    print(f"      * For satellite operations: Model 1 (balanced, fewer false alarms)")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: TSI Model Calculation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: PFSS Open Flux ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Flare Prediction (Logistic Regression) ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Model Evaluation Metrics ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
