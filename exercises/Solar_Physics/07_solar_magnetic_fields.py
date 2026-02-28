"""
Exercises for Lesson 07: Solar Magnetic Fields
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
k_B = 1.381e-23        # Boltzmann constant [J/K]
m_p = 1.673e-27        # proton mass [kg]
c = 3.0e8              # speed of light [m/s]
e = 1.602e-19          # elementary charge [C]
m_e = 9.109e-31        # electron mass [kg]
mu_0 = 4.0e-7 * np.pi  # vacuum permeability [T m/A]
R_sun = 6.957e8        # solar radius [m]

# Atomic masses
m_Fe = 55.845 * 1.661e-27  # iron atomic mass [kg]


def exercise_1():
    """
    Problem 1: Zeeman Splitting Calculation

    Fe I 6302.5 A line, g_eff = 2.5.
    (a) Splitting for B = 2500 G (sunspot).
    (b) Thermal Doppler width at T = 4500 K. Compare with splitting.
    (c) Is splitting resolved in sunspot? What about B = 50 G (quiet Sun)?
    """
    lambda_0 = 6302.5e-10   # wavelength [m]
    g_eff = 2.5
    T = 4500.0               # K

    # Zeeman splitting: Delta_lambda_B = g_eff * e * lambda_0^2 * B / (4 pi m_e c)
    # Or: Delta_lambda_B = 4.67e-13 * g_eff * lambda_0^2 [A] * B [G]  in Angstroms

    # (a) Sunspot B = 2500 G
    B_sunspot = 2500.0e-4    # convert G to T (1 G = 1e-4 T)

    # Using SI:
    Delta_lambda_B = g_eff * e * lambda_0**2 * B_sunspot / (4.0 * np.pi * m_e * c)
    Delta_lambda_B_A = Delta_lambda_B * 1e10  # convert to Angstroms

    print(f"  (a) Fe I {lambda_0*1e10:.1f} A, g_eff = {g_eff}")
    print(f"      B = 2500 G = {B_sunspot:.4f} T")
    print(f"      Zeeman splitting: Delta_lambda_B = g_eff e lambda^2 B / (4 pi m_e c)")
    print(f"                      = {Delta_lambda_B_A:.4f} A = {Delta_lambda_B_A*1e3:.2f} mA")

    # (b) Thermal Doppler width
    # Delta_lambda_D = lambda_0 * sqrt(2 k_B T / (m_Fe c^2))
    Delta_lambda_D = lambda_0 * np.sqrt(2.0 * k_B * T / (m_Fe * c**2))
    Delta_lambda_D_A = Delta_lambda_D * 1e10

    print(f"\n  (b) Thermal Doppler width at T = {T:.0f} K:")
    print(f"      Delta_lambda_D = lambda_0 * sqrt(2 k_B T / (m_Fe c^2))")
    print(f"                     = {Delta_lambda_D_A:.4f} A = {Delta_lambda_D_A*1e3:.2f} mA")

    ratio = Delta_lambda_B_A / Delta_lambda_D_A
    print(f"      Delta_lambda_B / Delta_lambda_D = {ratio:.2f}")

    # (c) Resolution check
    print(f"\n  (c) For the sunspot (B = 2500 G):")
    if ratio > 1:
        print(f"      The Zeeman splitting ({Delta_lambda_B_A*1e3:.1f} mA) EXCEEDS the")
        print(f"      Doppler width ({Delta_lambda_D_A*1e3:.1f} mA) => RESOLVED splitting")
    else:
        print(f"      The Zeeman splitting is smaller than the Doppler width => unresolved")

    # Quiet Sun: B = 50 G
    B_quiet = 50.0e-4  # T
    Delta_lambda_B_quiet = g_eff * e * lambda_0**2 * B_quiet / (4.0 * np.pi * m_e * c)
    Delta_lambda_B_quiet_A = Delta_lambda_B_quiet * 1e10
    ratio_quiet = Delta_lambda_B_quiet_A / Delta_lambda_D_A

    print(f"\n      For quiet Sun (B = 50 G):")
    print(f"      Delta_lambda_B = {Delta_lambda_B_quiet_A*1e3:.2f} mA")
    print(f"      Delta_lambda_B / Delta_lambda_D = {ratio_quiet:.3f}")
    print(f"      The splitting ({Delta_lambda_B_quiet_A*1e3:.2f} mA) is much smaller than")
    print(f"      the Doppler width ({Delta_lambda_D_A*1e3:.1f} mA) => UNRESOLVED")
    print(f"      => Must use Stokes V polarimetry (longitudinal Zeeman effect)")


def exercise_2():
    """
    Problem 2: PFSS Model (Dipole)

    l=1, m=0 PFSS solution (dipole).
    (a) General solution for Phi(r, theta).
    (b) Apply source surface BC.
    (c) Limits: R_ss -> inf and R_ss -> R_sun.
    """
    print(f"  (a) For l=1, m=0, the potential satisfies Laplace's equation")
    print(f"      in spherical coordinates. The general solution is:")
    print(f"      Phi(r, theta) = [a_10 * r + b_10 * r^(-2)] * cos(theta)")
    print(f"      ")
    print(f"      The radial field components are:")
    print(f"      B_r = -dPhi/dr = -[a_10 - 2*b_10*r^(-3)] * cos(theta)")
    print(f"      B_theta = -(1/r)*dPhi/dtheta = [a_10*r + b_10*r^(-2)] * sin(theta) / r")

    print(f"\n  (b) Source surface BC: B_theta(R_ss) = 0  (radial field at R_ss)")
    print(f"      B_theta ~ [a_10 * R_ss + b_10 * R_ss^(-2)] = 0")
    print(f"      => b_10 = -a_10 * R_ss^3")
    print(f"      ")
    print(f"      Substituting back:")
    print(f"      Phi = a_10 * [r - R_ss^3/r^2] * cos(theta)")
    print(f"      B_r = -a_10 * [1 + 2*R_ss^3/r^3] * cos(theta)")

    # (c) Limits
    print(f"\n  (c) Limit R_ss -> infinity:")
    print(f"      b_10 = -a_10 * R_ss^3 -> -infinity, but Phi/r^2 terms dominate")
    print(f"      Phi ~ a_10 * r * cos(theta) + ... => standard dipole (B ~ 1/r^3)")
    print(f"      Actually, with b_10 = -a_10*R_ss^3:")
    print(f"      Phi = a_10 * [r - R_ss^3/r^2] * cos(theta)")
    print(f"      For R_ss -> inf: the r term dominates at finite r, giving uniform field.")
    print(f"      More properly, redefine: let a_10 = C/R_ss^3, then:")
    print(f"      Phi = C * [r/R_ss^3 - 1/r^2] * cos(theta)")
    print(f"      As R_ss -> inf: Phi -> -C/r^2 * cos(theta) = standard dipole potential")
    print(f"      B_r = -2C*cos(theta)/r^3 (pure dipole, no source surface effect)")

    print(f"\n      Limit R_ss -> R_sun:")
    print(f"      b_10 = -a_10 * R_sun^3")
    print(f"      At r = R_sun: B_theta ~ [a_10*R_sun + b_10/R_sun^2]")
    print(f"                           = [a_10*R_sun - a_10*R_sun] = 0")
    print(f"      The field is purely radial at the surface itself.")
    print(f"      This is the limit of a completely open field configuration.")

    # Numerical illustration
    print(f"\n      Numerical illustration (R_ss in R_sun units):")
    R_ss_values = [2.5, 5.0, 10.0, 50.0, 1000.0]
    print(f"      {'R_ss/R':>8} {'B_r ratio at r=R':>20}")
    for R_ss in R_ss_values:
        # B_r(R_sun) ~ [1 + 2*(R_ss/R_sun)^3/1] with a_10 = const
        # But we need to normalize. At the surface:
        # B_r(R) = -a_10 * [1 + 2*R_ss^3/R^3] * cos(theta)
        ratio = (1 + 2 * R_ss**3) / 3.0  # relative to R_ss=1 case
        print(f"      {R_ss:8.1f} {1 + 2*R_ss**3:20.1f}")


def exercise_3():
    """
    Problem 3: Force-Free Parameter

    B = 100 G, J = 10 mA/m^2.
    (a) Calculate alpha = mu_0 * J / B.
    (b) Length scale L = 1/alpha vs typical AR size.
    (c) Physical meaning of large vs small alpha.
    """
    B = 100.0e-4         # 100 G in Tesla
    J = 10.0e-3          # 10 mA/m^2

    # (a) Force-free parameter
    alpha = mu_0 * J / B  # 1/m
    alpha_Mm = alpha * 1.0e6  # Mm^-1

    print(f"  (a) B = 100 G = {B:.4f} T")
    print(f"      J = {J*1e3:.0f} mA/m^2 = {J:.3f} A/m^2")
    print(f"      alpha = mu_0 * J / B")
    print(f"            = {mu_0:.4e} * {J:.3f} / {B:.4f}")
    print(f"            = {alpha:.4e} m^-1")
    print(f"            = {alpha_Mm:.3f} Mm^-1")

    # (b) Length scale
    L_scale = 1.0 / alpha
    L_scale_Mm = L_scale / 1.0e6

    print(f"\n  (b) Characteristic length: L = 1/alpha = {L_scale_Mm:.0f} Mm")
    AR_size = 100.0  # typical AR size in Mm
    print(f"      Typical active region size: ~{AR_size:.0f} Mm")

    if L_scale_Mm > AR_size:
        print(f"      L ({L_scale_Mm:.0f} Mm) > AR size ({AR_size:.0f} Mm)")
        print(f"      => The field is close to potential (weakly non-potential)")
    else:
        print(f"      L ({L_scale_Mm:.0f} Mm) < AR size ({AR_size:.0f} Mm)")
        print(f"      => Significant twist/shear on scales smaller than the AR")

    # (c) Physical meaning
    print(f"\n  (c) Physical meaning of alpha:")
    print(f"      alpha = mu_0 J_parallel / B = curl(B) dot B / B^2")
    print(f"      - Small |alpha| (L >> AR size): nearly potential field,")
    print(f"        currents are weak, little free energy stored")
    print(f"      - Large |alpha| (L << AR size): highly sheared/twisted field,")
    print(f"        strong currents, significant free magnetic energy")
    print(f"        available for flares and CMEs")
    print(f"      - Sign of alpha: positive = right-handed twist (dextral),")
    print(f"        negative = left-handed twist (sinistral)")
    print(f"      - Hemispheric helicity rule: alpha < 0 in north, > 0 in south")


def exercise_4():
    """
    Problem 4: Magnetic Helicity and CMEs

    Helicity injection rate: dH/dt = 5e40 Mx^2/hour.
    (a) Helicity accumulated over 3 days.
    (b) Days of accumulation per CME (H_CME ~ 1e42 Mx^2).
    (c) Total over AR lifetime, number of CMEs.
    """
    dH_dt = 5.0e40    # Mx^2 / hour

    # (a) Helicity over 3 days
    t_3days = 3.0 * 24.0  # hours
    H_3days = dH_dt * t_3days

    print(f"  (a) Helicity injection rate: dH/dt = {dH_dt:.1e} Mx^2/hour")
    print(f"      Over 3 days ({t_3days:.0f} hours):")
    print(f"      H = {dH_dt:.1e} * {t_3days:.0f} = {H_3days:.2e} Mx^2")

    # (b) Days per CME
    H_CME = 1.0e42   # Mx^2
    t_CME_hr = H_CME / dH_dt
    t_CME_days = t_CME_hr / 24.0

    print(f"\n  (b) CME helicity: H_CME ~ {H_CME:.0e} Mx^2")
    print(f"      Accumulation time: H_CME / (dH/dt) = {t_CME_hr:.0f} hours = {t_CME_days:.1f} days")
    print(f"      One CME's worth of helicity accumulates every ~{t_CME_days:.0f} days.")

    # (c) Over AR lifetime
    AR_lifetime_days = 21.0  # 2-4 weeks, use 3 weeks
    AR_lifetime_hr = AR_lifetime_days * 24.0
    H_total = dH_dt * AR_lifetime_hr
    N_CMEs = H_total / H_CME

    print(f"\n  (c) AR lifetime: ~{AR_lifetime_days:.0f} days (3 weeks)")
    print(f"      Total helicity: {H_total:.2e} Mx^2")
    print(f"      Number of CMEs to shed helicity: N = {N_CMEs:.1f}")

    # Also show for 14 and 28 days
    for lifetime in [14, 21, 28]:
        H = dH_dt * lifetime * 24.0
        N = H / H_CME
        print(f"      {lifetime:2d} days: H = {H:.1e} Mx^2, ~{N:.0f} CMEs")

    print(f"\n      Implications:")
    print(f"      - Active regions need to produce several CMEs during their lifetime")
    print(f"        to remove the accumulated helicity.")
    print(f"      - This is consistent with observations: productive ARs can launch")
    print(f"        multiple CMEs over days to weeks.")
    print(f"      - If helicity is not removed by CMEs, it may contribute to")
    print(f"        increasingly complex/unstable magnetic configurations.")


def exercise_5():
    """
    Problem 5: Squashing Factor and Reconnection

    2D mapping: (x1,y1) -> (x2,y2) = (x1 + L*tanh(x1/w), y1)
    (a) Jacobian elements.
    (b) Squashing factor Q.
    (c) Where Q is maximized, Q_max.
    (d) Numerical example: L=50 Mm, w=1 Mm.
    """
    # (a) Jacobian of the mapping
    # x2 = x1 + L * tanh(x1/w), y2 = y1
    # a = dx2/dx1 = 1 + L/w * sech^2(x1/w)
    # b = dx2/dy1 = 0
    # c = dy2/dx1 = 0
    # d = dy2/dy1 = 1
    print(f"  (a) Mapping: (x1, y1) -> (x1 + L*tanh(x1/w), y1)")
    print(f"      Jacobian:")
    print(f"      a = dx2/dx1 = 1 + (L/w) * sech^2(x1/w)")
    print(f"      b = dx2/dy1 = 0")
    print(f"      c = dy2/dx1 = 0")
    print(f"      d = dy2/dy1 = 1")

    # (b) Squashing factor
    # Q = (a^2 + b^2 + c^2 + d^2) / |ad - bc|
    # With b = c = 0, d = 1:
    # Q = (a^2 + 1) / |a|
    print(f"\n  (b) Squashing factor:")
    print(f"      Q = (a^2 + b^2 + c^2 + d^2) / |ad - bc|")
    print(f"        = (a^2 + 0 + 0 + 1) / |a * 1 - 0|")
    print(f"        = (a^2 + 1) / |a|")
    print(f"        = |a| + 1/|a|")
    print(f"      where a = 1 + (L/w) * sech^2(x1/w)")

    # (c) Q is maximized where sech^2(x1/w) is maximized => x1 = 0
    # At x1 = 0: sech^2(0) = 1
    # a_max = 1 + L/w
    # Q_max = (1 + L/w) + 1/(1 + L/w)
    print(f"\n  (c) sech^2(x1/w) is maximized at x1 = 0 (sech^2(0) = 1)")
    print(f"      a_max = 1 + L/w")
    print(f"      Q_max = a_max + 1/a_max = (1 + L/w) + 1/(1 + L/w)")
    print(f"      For L/w >> 1: Q_max ~ L/w")

    # (d) Numerical: L = 50 Mm, w = 1 Mm
    L = 50.0    # Mm
    w = 1.0     # Mm

    a_max = 1.0 + L / w
    Q_max = a_max + 1.0 / a_max

    print(f"\n  (d) For L = {L:.0f} Mm, w = {w:.0f} Mm:")
    print(f"      L/w = {L/w:.0f}")
    print(f"      a_max = 1 + L/w = {a_max:.0f}")
    print(f"      Q_max = {a_max:.0f} + 1/{a_max:.0f} = {Q_max:.2f}")
    print(f"      ")
    print(f"      A typical QSL is defined by Q > ~100 (often log Q > 2).")
    print(f"      With Q_max = {Q_max:.0f}, this is a strong QSL (log Q = {np.log10(Q_max):.2f}).")
    print(f"      Such QSLs are sites where field lines change connectivity rapidly,")
    print(f"      making them favorable locations for current sheet formation and")
    print(f"      magnetic reconnection during eruptions.")

    # Plot Q vs x1 for illustration
    x1 = np.linspace(-5 * w, 5 * w, 100)
    a = 1.0 + (L / w) / np.cosh(x1 / w)**2
    Q = a + 1.0 / a

    print(f"\n      Q profile along x1 (in units of w):")
    print(f"      {'x1/w':>8} {'a':>10} {'Q':>10} {'log10(Q)':>10}")
    for xi in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
        ai = 1.0 + (L / w) / np.cosh(xi)**2
        Qi = ai + 1.0 / ai
        print(f"      {xi:8.1f} {ai:10.2f} {Qi:10.2f} {np.log10(Qi):10.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Zeeman Splitting Calculation ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: PFSS Model ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Force-Free Parameter ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Magnetic Helicity and CMEs ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Squashing Factor and Reconnection ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
