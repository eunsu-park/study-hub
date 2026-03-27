"""
Exercise Solutions for Lesson 01: Introduction to Space Weather

Topics covered:
  - CME transit time estimation (constant speed and constant deceleration)
  - Energy comparison (solar wind KE, ring current, hurricane)
  - Carrington-class event recurrence probability (Poisson process)
  - Infrastructure vulnerability (binomial transformer damage)
  - DSCOVR warning time analysis
"""

import numpy as np
from scipy.stats import poisson, binom


def exercise_1():
    """
    Exercise 1: CME Transit Time Estimation

    A CME is observed leaving the Sun at 1200 km/s.
    (a) Assuming constant speed, how long does it take to reach Earth at 1 AU?
    (b) If the CME decelerates to 800 km/s by Earth, estimate the actual transit
        time assuming constant deceleration.
    """
    print("=" * 70)
    print("Exercise 1: CME Transit Time Estimation")
    print("=" * 70)

    AU = 1.496e11  # m
    v0 = 1200e3    # m/s (initial speed)
    vf = 800e3     # m/s (final speed at Earth)

    # (a) Constant speed
    t_const = AU / v0
    t_const_hr = t_const / 3600
    print(f"\n(a) Constant speed at {v0/1e3:.0f} km/s:")
    print(f"    Transit time = {AU:.3e} m / {v0:.3e} m/s = {t_const:.2e} s")
    print(f"    = {t_const_hr:.1f} hours = {t_const_hr/24:.1f} days")

    # (b) Constant deceleration
    # Using v_f^2 = v_0^2 + 2*a*d  =>  a = (v_f^2 - v_0^2) / (2*d)
    a = (vf**2 - v0**2) / (2 * AU)
    print(f"\n(b) Constant deceleration from {v0/1e3:.0f} to {vf/1e3:.0f} km/s:")
    print(f"    Deceleration a = (v_f^2 - v_0^2) / (2*d)")
    print(f"    a = ({vf**2:.3e} - {v0**2:.3e}) / (2 * {AU:.3e})")
    print(f"    a = {a:.4e} m/s^2")

    # Using v_f = v_0 + a*t  =>  t = (v_f - v_0) / a
    t_decel = (vf - v0) / a
    t_decel_hr = t_decel / 3600
    print(f"    Transit time = (v_f - v_0) / a = ({vf - v0:.0f}) / ({a:.4e})")
    print(f"    = {t_decel:.2e} s = {t_decel_hr:.1f} hours = {t_decel_hr/24:.1f} days")

    # Verification using average speed
    v_avg = (v0 + vf) / 2
    t_avg = AU / v_avg
    print(f"\n    Verification: average speed = {v_avg/1e3:.0f} km/s")
    print(f"    t = d / v_avg = {t_avg/3600:.1f} hours (should match)")


def exercise_2():
    """
    Exercise 2: Energy Comparison

    Compare the ring current energy (~5e15 J) with:
    (a) Solar wind KE impacting magnetosphere cross-section for 1 hour
    (b) A typical hurricane's kinetic energy (~3e18 J)
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Energy Comparison")
    print("=" * 70)

    E_ring = 5e15  # J (ring current energy in a large storm)

    # (a) Solar wind kinetic energy
    R_mp = 10 * 6.371e6  # magnetopause standoff in meters (10 R_E)
    A = np.pi * R_mp**2  # cross-section area
    v_sw = 500e3          # m/s
    n = 10e6              # 10 cm^-3 = 10e6 m^-3
    m_p = 1.67e-27        # kg (proton mass)
    rho = n * m_p         # mass density
    dt = 3600             # 1 hour in seconds

    # KE flux = 0.5 * rho * v^3 (energy flux) * A * dt
    KE_flux = 0.5 * rho * v_sw**2  # energy per unit volume
    KE_power = KE_flux * v_sw * A  # power = energy density * speed * area
    KE_total = KE_power * dt

    print(f"\n(a) Solar wind kinetic energy through magnetosphere cross-section:")
    print(f"    Magnetopause radius: R_mp = 10 R_E = {R_mp:.3e} m")
    print(f"    Cross-section area: A = pi * R_mp^2 = {A:.3e} m^2")
    print(f"    Solar wind: n = {n/1e6:.0f} cm^-3, v = {v_sw/1e3:.0f} km/s")
    print(f"    Mass density: rho = n * m_p = {rho:.3e} kg/m^3")
    print(f"    KE flux (power): 0.5 * rho * v^3 * A = {KE_power:.3e} W")
    print(f"    KE over 1 hour: {KE_total:.3e} J")
    print(f"    Ratio E_ring / E_sw = {E_ring / KE_total:.4f}")
    print(f"    => Only ~{E_ring / KE_total * 100:.2f}% of solar wind KE is "
          f"captured as ring current energy")

    # (b) Hurricane comparison
    E_hurricane = 3e18  # J
    print(f"\n(b) Comparison with hurricane kinetic energy:")
    print(f"    Hurricane KE: {E_hurricane:.1e} J")
    print(f"    Ring current energy: {E_ring:.1e} J")
    print(f"    Ratio E_ring / E_hurricane = {E_ring / E_hurricane:.4f}")
    print(f"    => Ring current energy is only ~{E_ring / E_hurricane * 100:.2f}% "
          f"of a hurricane's KE")
    print(f"\n    Interpretation: The coupling efficiency between solar wind and")
    print(f"    magnetosphere is very low (~{E_ring / KE_total * 100:.1f}%), "
          f"and even a major geomagnetic")
    print(f"    storm stores far less energy than terrestrial weather systems.")


def exercise_3():
    """
    Exercise 3: Carrington-Class Recurrence

    If Carrington-class events have a return period of ~150 years, what is the
    probability of at least one such event in the next 30 years?
    Assume a Poisson process.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Carrington-Class Recurrence (Poisson Process)")
    print("=" * 70)

    T_return = 150  # years (return period)
    lam_rate = 1 / T_return  # events per year
    t_window = 30  # years

    # Expected number of events in 30 years
    mu = lam_rate * t_window
    print(f"\n    Return period: {T_return} years")
    print(f"    Rate: lambda = 1/{T_return} = {lam_rate:.4f} events/year")
    print(f"    Time window: {t_window} years")
    print(f"    Expected events in {t_window} years: mu = lambda * t = {mu:.4f}")

    # P(at least 1) = 1 - P(0) = 1 - exp(-mu)
    P_zero = poisson.pmf(0, mu)
    P_at_least_one = 1 - P_zero
    print(f"\n    P(X = 0) = exp(-mu) = exp(-{mu:.4f}) = {P_zero:.4f}")
    print(f"    P(X >= 1) = 1 - P(X = 0) = {P_at_least_one:.4f}")
    print(f"    => {P_at_least_one * 100:.1f}% probability of at least one "
          f"Carrington-class event in {t_window} years")

    # Show probabilities for different numbers of events
    print(f"\n    Full Poisson distribution (mu = {mu:.3f}):")
    for k in range(5):
        p = poisson.pmf(k, mu)
        print(f"      P(X = {k}) = {p:.6f}")

    # Discussion
    print(f"\n    Discussion of Poisson assumption:")
    print(f"    - Poisson assumes events are independent and uniformly distributed")
    print(f"    - Solar events cluster around solar maximum (~11-year cycle)")
    print(f"    - Extreme events may follow a power-law distribution, not Poisson")
    print(f"    - The 150-year return period is poorly constrained (few data points)")
    print(f"    - Some studies suggest higher probability (~12% per decade)")


def exercise_4():
    """
    Exercise 4: Infrastructure Vulnerability

    500 HV transformers, each with 2% damage probability during a severe storm.
    Calculate expected damaged transformers and P(more than 15 damaged).
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Infrastructure Vulnerability (Binomial)")
    print("=" * 70)

    n = 500   # number of transformers
    p = 0.02  # damage probability per transformer

    # Expected number
    E_X = n * p
    std_X = np.sqrt(n * p * (1 - p))
    print(f"\n    Parameters: n = {n}, p = {p}")
    print(f"    Expected damaged: E[X] = n*p = {E_X:.1f}")
    print(f"    Standard deviation: sigma = sqrt(n*p*(1-p)) = {std_X:.2f}")

    # P(X > 15)
    P_le_15 = binom.cdf(15, n, p)
    P_gt_15 = 1 - P_le_15
    print(f"\n    P(X > 15) = 1 - P(X <= 15)")
    print(f"    P(X <= 15) = {P_le_15:.8f}")
    print(f"    P(X > 15) = {P_gt_15:.8f} = {P_gt_15:.6e}")
    print(f"    => About {P_gt_15 * 100:.4f}% chance of more than 15 failures")

    # Show distribution around the mean
    print(f"\n    Probability distribution near the mean:")
    for k in range(max(0, int(E_X) - 5), int(E_X) + 10):
        prob = binom.pmf(k, n, p)
        bar = "*" * int(prob * 500)
        print(f"      P(X = {k:2d}) = {prob:.4f} {bar}")

    # Discussion
    print(f"\n    Key assumption flaw: Independence!")
    print(f"    - GICs depend on local geology and grid topology")
    print(f"    - Transformers in the same region experience correlated fields")
    print(f"    - A cascading failure can propagate through the grid")
    print(f"    - Actual risk is likely higher than the independent model predicts")
    print(f"    - Copula models or network simulations would be more realistic")


def exercise_5():
    """
    Exercise 5: DSCOVR Warning Time Analysis

    DSCOVR at L1 (~1.5 million km upstream).
    Calculate warning time for solar wind speeds 400, 600, 1000 km/s.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: DSCOVR Warning Time Analysis")
    print("=" * 70)

    d_L1 = 1.5e9  # 1.5 million km = 1.5e9 m
    speeds = [400e3, 600e3, 1000e3]  # m/s

    print(f"\n    DSCOVR distance from Earth: {d_L1/1e9:.1f} million km")
    print(f"\n    {'Speed (km/s)':<15} {'Warning Time':>15} {'Minutes':>10}")
    print(f"    {'-'*40}")

    for v in speeds:
        t = d_L1 / v
        t_min = t / 60
        print(f"    {v/1e3:>8.0f}       {t:>12.0f} s   {t_min:>8.1f}")

    print(f"\n    Why actual warning time may be shorter during extreme events:")
    print(f"    1. CME-driven shocks accelerate ahead of the main structure;")
    print(f"       the shock arrives at DSCOVR only minutes before hitting Earth")
    print(f"    2. At 1000+ km/s, the 1.5M km provides only ~25 min warning")
    print(f"    3. DSCOVR data transmission and processing introduces delays")
    print(f"    4. The CME magnetic field orientation (Bz) cannot be predicted")
    print(f"       until it passes DSCOVR, reducing effective forecast lead time")
    print(f"    5. In extreme events, solar wind speed may increase between")
    print(f"       L1 and Earth, further reducing the actual warning time")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
