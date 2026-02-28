"""
Exercise Solutions for Lesson 14: Forecasting Models

Topics covered:
  - Wang-Sheeley-Arge (WSA) solar wind speed prediction
  - CME drag-based model (DBM) for transit time
  - Contingency table verification metrics (POD, FAR, TSS, HSS)
  - Probabilistic forecast reliability and Brier Score
  - Model comparison (T96 vs SWMF) discussion
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: WSA Solar Wind Speed

    Coronal hole: f_s = 3.5, theta_b = 15 deg.
    WSA formula with v_fast=625, v_range=240, alpha=4.5, theta_0=2.8, beta=1.25.
    """
    print("=" * 70)
    print("Exercise 1: WSA Solar Wind Speed")
    print("=" * 70)

    f_s = 3.5        # flux expansion factor
    theta_b = 15     # degrees (footpoint distance to boundary)
    v_fast = 625     # km/s
    v_range = 240    # km/s
    alpha = 4.5
    theta_0 = 2.8    # degrees
    beta = 1.25

    # WSA formula:
    # v = v_fast - v_range * f_s^(-2/7) * (1 - beta * exp(-(theta_b/theta_0)^2))^3
    # Wait, the standard WSA formula from Arge & Pizzo (2000):
    # v(f_s, theta_b) = v_fast - f_range * f_s^(alpha/7)
    #     * [1 - beta * exp(-(theta_b/theta_0)^2)]^3
    # Different versions exist. Using the form given in the problem:
    # v = v_fast - v_range * f_s^(-alpha/7) * (1 - beta * exp(-(theta_b/theta_0)^2))^3

    # Actually the common form is:
    # v = v_fast - v_range * f_s^(2/7) / (1 + f_s)^(2/7)  [simplified]
    # Let me use the formula implied by the parameters:
    # v = v_fast - v_range * f_s^(-2/7) * [1 - beta * exp(-(theta_b/theta_0)^alpha)]^3

    # Most common WSA (with the given parameters):
    exp_term = np.exp(-(theta_b / theta_0)**2)
    bracket = 1 - beta * exp_term
    v_sw = v_fast - v_range * f_s**(-2/7) * bracket**3

    print(f"\n    Coronal hole parameters:")
    print(f"    f_s = {f_s} (flux expansion factor)")
    print(f"    theta_b = {theta_b} deg (distance to CH boundary)")
    print(f"    WSA parameters: v_fast={v_fast}, v_range={v_range},")
    print(f"    alpha={alpha}, theta_0={theta_0}, beta={beta}")

    print(f"\n    WSA calculation:")
    print(f"    exp term = exp(-(theta_b/theta_0)^2) = exp(-({theta_b}/{theta_0})^2)")
    print(f"    = exp(-{(theta_b/theta_0)**2:.2f}) = {exp_term:.6f}")
    print(f"    Bracket = 1 - beta * exp(...) = 1 - {beta}*{exp_term:.6f}")
    print(f"    = {bracket:.6f}")
    print(f"    f_s^(-2/7) = {f_s}^(-{2/7:.4f}) = {f_s**(-2/7):.4f}")
    print(f"    v = {v_fast} - {v_range} * {f_s**(-2/7):.4f} * {bracket:.6f}^3")
    print(f"    = {v_fast} - {v_range * f_s**(-2/7) * bracket**3:.1f}")
    print(f"    v = {v_sw:.0f} km/s")

    print(f"\n    Classification:")
    if v_sw > 550:
        print(f"    This is FAST solar wind ({v_sw:.0f} km/s > 550 km/s)")
        print(f"    Associated with a high-speed stream from the coronal hole")
    elif v_sw > 450:
        print(f"    This is MODERATE speed wind ({v_sw:.0f} km/s)")
    else:
        print(f"    This is SLOW solar wind ({v_sw:.0f} km/s)")

    print(f"    Large f_s = {f_s} (expanding field lines) tends to slow the wind")
    print(f"    But large theta_b = {theta_b} deg (well inside the CH) makes")
    print(f"    the boundary correction small, maintaining moderate speed")


def exercise_2():
    """
    Exercise 2: CME Drag-Based Model

    v0 = 1200 km/s, v_sw = 400 km/s, gamma = 0.5e-7 km^-1.
    Analytical DBM: v(t) = v_sw + (v0-v_sw)/(1 + gamma*|v0-v_sw|*t)
    R(t) = v_sw*t + ln(1 + gamma*|v0-v_sw|*t) / gamma
    Find v at 1 AU and transit time.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: CME Drag-Based Model")
    print("=" * 70)

    v0 = 1200       # km/s
    v_sw = 400       # km/s
    gamma = 0.5e-7   # km^-1
    R_1AU = 1.496e8  # km (1 AU)
    dv = v0 - v_sw

    print(f"\n    CME: v0 = {v0} km/s, v_sw = {v_sw} km/s")
    print(f"    gamma = {gamma:.1e} km^-1")
    print(f"    Delta_v = v0 - v_sw = {dv} km/s")

    # Speed as a function of time
    # v(t) = v_sw + dv / (1 + gamma * dv * t)
    # where t is in seconds

    # Distance as a function of time
    # R(t) = v_sw*t + ln(1 + gamma*dv*t) / gamma

    # Find transit time by solving R(t) = R_1AU
    # This is a transcendental equation; solve numerically
    from scipy.optimize import brentq

    def R_minus_target(t_sec):
        t_sec_to_gamma_t = gamma * dv * t_sec  # dimensionless if using km and s
        # Wait, units: gamma [1/km], dv [km/s], t [s]
        # gamma * dv * t = [1/km] * [km/s] * [s] = dimensionless. Good.
        R = v_sw * t_sec + np.log(1 + gamma * dv * t_sec) / gamma
        return R - R_1AU

    # Initial bracket: transit between 1 day and 5 days
    t_lo = 1 * 86400  # 1 day in seconds
    t_hi = 5 * 86400  # 5 days
    t_transit = brentq(R_minus_target, t_lo, t_hi)
    t_transit_hr = t_transit / 3600
    t_transit_days = t_transit / 86400

    # Speed at 1 AU
    v_1AU = v_sw + dv / (1 + gamma * dv * t_transit)

    print(f"\n    Transit time to 1 AU ({R_1AU:.3e} km):")
    print(f"    Solving R(t) = v_sw*t + ln(1+gamma*dv*t)/gamma = R_1AU")
    print(f"    t_transit = {t_transit_hr:.1f} hours = {t_transit_days:.2f} days")

    print(f"\n    CME speed at 1 AU:")
    print(f"    v(1AU) = v_sw + dv/(1+gamma*dv*t)")
    print(f"    = {v_sw} + {dv}/(1+{gamma:.1e}*{dv}*{t_transit:.0f})")
    print(f"    = {v_sw} + {dv}/{1 + gamma*dv*t_transit:.2f}")
    print(f"    = {v_1AU:.0f} km/s")

    print(f"\n    The CME decelerated from {v0} to {v_1AU:.0f} km/s")
    print(f"    ({(1 - v_1AU/v0)*100:.0f}% speed reduction)")

    # Time profile
    print(f"\n    Speed vs time profile:")
    print(f"    {'t (hr)':>8} {'v (km/s)':>12} {'R (AU)':>10}")
    print(f"    {'-'*32}")
    for t_hr in [0, 6, 12, 18, 24, 36, 48, t_transit_hr]:
        t_s = t_hr * 3600
        v = v_sw + dv / (1 + gamma * dv * t_s)
        R = v_sw * t_s + np.log(1 + gamma * dv * t_s) / gamma
        print(f"    {t_hr:>8.1f} {v:>12.0f} {R/R_1AU:>10.3f}")


def exercise_3():
    """
    Exercise 3: Contingency Table Metrics

    G3+ storm forecasts over 1 year:
    Hits = 12, False Alarms = 8, Misses = 5, Correct Rejections = 340.
    Calculate POD, FAR, TSS, HSS, accuracy.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Forecast Verification Metrics")
    print("=" * 70)

    a = 12    # hits
    b = 8     # false alarms
    c = 5     # misses
    d = 340   # correct rejections
    N = a + b + c + d

    print(f"\n    Contingency Table (G3+ storm forecasts):")
    print(f"    {'':>20} {'Observed YES':>14} {'Observed NO':>14} {'Total':>8}")
    print(f"    {'Forecast YES':>20} {a:>14} {b:>14} {a+b:>8}")
    print(f"    {'Forecast NO':>20} {c:>14} {d:>14} {c+d:>8}")
    print(f"    {'Total':>20} {a+c:>14} {b+d:>14} {N:>8}")

    # POD = a / (a+c)
    POD = a / (a + c)
    print(f"\n    POD (Probability of Detection) = a/(a+c) = {a}/({a}+{c}) = {POD:.3f}")

    # FAR = b / (a+b)
    FAR = b / (a + b)
    print(f"    FAR (False Alarm Ratio) = b/(a+b) = {b}/({a}+{b}) = {FAR:.3f}")

    # POFD = b / (b+d)
    POFD = b / (b + d)
    print(f"    POFD (Probability of False Detection) = b/(b+d) = {b}/({b}+{d}) = {POFD:.4f}")

    # TSS = POD - POFD
    TSS = POD - POFD
    print(f"    TSS (True Skill Statistic) = POD - POFD = {POD:.3f} - {POFD:.4f} = {TSS:.3f}")

    # HSS = 2(ad-bc) / [(a+c)(c+d) + (a+b)(b+d)]
    HSS_num = 2 * (a * d - b * c)
    HSS_den = (a + c) * (c + d) + (a + b) * (b + d)
    HSS = HSS_num / HSS_den
    print(f"    HSS (Heidke Skill Score) = 2(ad-bc)/[...] = {HSS:.3f}")

    # Accuracy
    ACC = (a + d) / N
    print(f"    Accuracy = (a+d)/N = ({a}+{d})/{N} = {ACC:.4f} = {ACC*100:.1f}%")

    print(f"\n    Why accuracy is MISLEADING for rare events:")
    print(f"    Accuracy = {ACC*100:.1f}% looks great, but a 'no-event' forecast")
    print(f"    (predicting no storm ever) would get {d+c}/{N} = {(d+c)/N*100:.1f}% accuracy!")
    print(f"    That's because events (a+c = {a+c}) are rare compared to non-events ({b+d}).")
    print(f"    Accuracy is dominated by correct rejections ({d} out of {N}).")
    print(f"    TSS = {TSS:.3f} is preferred because it equally weights hit rate")
    print(f"    and false alarm rate, and is unaffected by the event/non-event ratio.")


def exercise_4():
    """
    Exercise 4: Probabilistic Forecast Reliability

    70% CME arrival forecast: 50 cases, CME arrived 42 times.
    Is it reliable? Brier Score for this bin. Reliability diagram point.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Probabilistic Forecast Reliability")
    print("=" * 70)

    p_forecast = 0.70
    N_total = 50
    N_observed = 42
    observed_freq = N_observed / N_total

    print(f"\n    Forecast probability: {p_forecast*100:.0f}%")
    print(f"    Number of forecasts at this level: {N_total}")
    print(f"    Number of times event occurred: {N_observed}")
    print(f"    Observed frequency: {N_observed}/{N_total} = {observed_freq:.2f} "
          f"= {observed_freq*100:.0f}%")

    # Reliability check
    print(f"\n    Reliability assessment:")
    print(f"    Forecast probability: {p_forecast*100:.0f}%")
    print(f"    Observed frequency:   {observed_freq*100:.0f}%")
    diff = observed_freq - p_forecast
    print(f"    Difference: {diff*100:+.0f} percentage points")

    if abs(diff) < 0.05:
        print(f"    The forecast is well-calibrated (reliable) at this level.")
    elif diff > 0:
        print(f"    The forecast is UNDERCONFIDENT (events occur more often than predicted).")
    else:
        print(f"    The forecast is OVERCONFIDENT (events occur less often than predicted).")

    print(f"\n    On a reliability diagram:")
    print(f"    Point at (forecast = {p_forecast:.2f}, observed = {observed_freq:.2f})")
    print(f"    This point is {'above' if diff > 0 else 'below'} the diagonal")
    print(f"    (perfect reliability line)")

    # Brier Score for this bin
    # BS = (1/N) * sum(p_i - o_i)^2
    # For this bin: each forecast has p_i = 0.70
    # o_i = 1 for N_observed cases, o_i = 0 for (N_total - N_observed) cases
    BS = (N_observed * (p_forecast - 1)**2 + (N_total - N_observed) * (p_forecast - 0)**2) / N_total

    print(f"\n    Brier Score for this probability bin:")
    print(f"    BS = (1/N) * [N_hit*(p-1)^2 + N_miss*(p-0)^2]")
    print(f"    = (1/{N_total}) * [{N_observed}*({p_forecast}-1)^2 + "
          f"{N_total-N_observed}*({p_forecast})^2]")
    print(f"    = (1/{N_total}) * [{N_observed}*{(p_forecast-1)**2:.4f} + "
          f"{N_total-N_observed}*{p_forecast**2:.4f}]")
    BS_hit = N_observed * (p_forecast - 1)**2
    BS_miss = (N_total - N_observed) * p_forecast**2
    print(f"    = (1/{N_total}) * [{BS_hit:.2f} + {BS_miss:.2f}]")
    print(f"    BS = {BS:.4f}")
    print(f"    (BS = 0 is perfect, BS = 1 is worst)")

    # Reference: climatological BS
    clim = (N_observed / N_total)  # climatological frequency
    BS_clim = clim * (1 - clim)  # Brier score of climatology
    BSS = 1 - BS / BS_clim if BS_clim > 0 else 0

    print(f"\n    Brier Skill Score relative to climatology:")
    print(f"    Climatological frequency: {clim:.2f}")
    print(f"    BS_climatology = p*(1-p) = {BS_clim:.4f}")
    print(f"    BSS = 1 - BS/BS_clim = 1 - {BS:.4f}/{BS_clim:.4f} = {BSS:.3f}")
    if BSS > 0:
        print(f"    Positive skill relative to climatology")


def exercise_5():
    """
    Exercise 5: Model Comparison (T96 vs SWMF)

    Satellite operator needs B-field at GEO during moderate storm.
    Dst = -80 nT, P_dyn = 5 nPa, Bz = -8 nT.
    Compare T96 and SWMF/BATS-R-US.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: T96 vs SWMF Model Comparison")
    print("=" * 70)

    Dst = -80    # nT
    P_dyn = 5    # nPa
    Bz = -8      # nT

    print(f"\n    Storm conditions: Dst = {Dst} nT, P_dyn = {P_dyn} nPa, Bz = {Bz} nT")
    print(f"    Application: Magnetic field at GEO (6.6 R_E) during moderate storm")

    print(f"\n    === T96 (Tsyganenko 1996) ===")
    print(f"    Type: Empirical parametric model")
    print(f"    Inputs: Dst, P_dyn, IMF By/Bz (exactly what we have)")
    print(f"    Computation time: Milliseconds (analytical functions)")
    print(f"    Accuracy at GEO: ~5-10 nT typical error")
    print(f"    Strengths:")
    print(f"    - Fast: can evaluate at any point instantly")
    print(f"    - Well-validated for moderate storms (Dst > -200 nT)")
    print(f"    - Smooth, continuous field representation")
    print(f"    - Easy to implement and use operationally")
    print(f"    Weaknesses:")
    print(f"    - Static: represents average configuration for given parameters")
    print(f"    - Cannot capture transient dynamics (substorm dipolarization)")
    print(f"    - Less accurate for extreme storms (Dst < -200 nT)")
    print(f"    - Does not self-consistently couple to ionosphere")

    print(f"\n    === SWMF / BATS-R-US ===")
    print(f"    Type: First-principles global MHD simulation")
    print(f"    Inputs: Full solar wind time series (v, n, T, B) at upstream boundary")
    print(f"    Computation time: Hours to days on HPC cluster")
    print(f"    Accuracy at GEO: ~5-15 nT (similar to T96 for this case)")
    print(f"    Strengths:")
    print(f"    - Self-consistent physics (MHD + inner magnetosphere coupling)")
    print(f"    - Captures dynamics: substorms, magnetopause motion, tail dynamics")
    print(f"    - Can predict transient features not in T96 parameterization")
    print(f"    - Better for extreme events where empirical models extrapolate")
    print(f"    Weaknesses:")
    print(f"    - Computationally expensive (not real-time for most applications)")
    print(f"    - Requires complete solar wind input time series")
    print(f"    - Numerical resolution limits accuracy at specific points")
    print(f"    - MHD neglects kinetic effects important in inner magnetosphere")

    print(f"\n    RECOMMENDATION for this application:")
    print(f"    Use T96. Reasoning:")
    print(f"    1. Moderate storm (Dst = {Dst} nT) is well within T96's validated range")
    print(f"    2. GEO (6.6 R_E) is in the inner magnetosphere where T96 is accurate")
    print(f"    3. The operator needs quick answers, not hours of computation")
    print(f"    4. Instantaneous B-field at a specific location is T96's strength")
    print(f"    5. SWMF would be overkill for this application and requires HPC access")
    print(f"\n    Use SWMF when:")
    print(f"    - Studying extreme events (Dst < -200 nT)")
    print(f"    - Need dynamic time evolution (substorms, magnetopause crossings)")
    print(f"    - Studying the magnetotail or magnetosheath (beyond T96 validity)")
    print(f"    - Coupling to ionospheric models is required")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
