"""
Exercise Solutions for Lesson 07: Radiation Belts

Topics covered:
  - First adiabatic invariant and adiabatic transport
  - Loss cone angle calculation for dipole field
  - Drift period and ULF wave resonance
  - Phase space density (PSD) analysis for source identification
  - South Atlantic Anomaly (SAA) radiation dose estimation
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Adiabatic Invariant and Transport

    1 MeV electron at L=5, equatorial pitch angle 45 deg.
    (a) Calculate first adiabatic invariant mu.
    (b) Adiabatic transport to L=3: new perpendicular energy.
    (c) Factor of total KE increase.
    """
    print("=" * 70)
    print("Exercise 1: Adiabatic Invariant and Transport")
    print("=" * 70)

    eV = 1.602e-19
    m_e = 9.109e-31    # kg
    c = 2.998e8         # m/s
    B0 = 3.1e-5         # T

    E_kin = 1.0e6 * eV  # 1 MeV in Joules
    L1 = 5
    L2 = 3
    alpha_eq = np.radians(45)  # equatorial pitch angle

    # Equatorial B at L=5: B_eq = B0 / L^3
    B_eq1 = B0 / L1**3
    B_eq2 = B0 / L2**3

    print(f"\n    E = 1 MeV electron, L = {L1}, alpha_eq = 45 deg")
    print(f"    B_eq(L={L1}) = B0/L^3 = {B0:.1e}/{L1}^3 = {B_eq1:.3e} T "
          f"= {B_eq1*1e9:.1f} nT")
    print(f"    B_eq(L={L2}) = B0/L^3 = {B0:.1e}/{L2}^3 = {B_eq2:.3e} T "
          f"= {B_eq2*1e9:.1f} nT")

    # (a) First adiabatic invariant
    # For relativistic electron: mu = p_perp^2 / (2*m*B)
    # where p is the relativistic momentum
    # E_total = E_kin + m*c^2 = gamma*m*c^2
    # p*c = sqrt(E_total^2 - (m*c^2)^2)
    E_rest = m_e * c**2
    E_total = E_kin + E_rest
    p = np.sqrt(E_total**2 - E_rest**2) / c  # relativistic momentum

    # p_perp = p * sin(alpha)
    p_perp = p * np.sin(alpha_eq)

    # Non-relativistic mu = p_perp^2 / (2*m_e*B)
    # For relativistic: mu = p_perp^2 / (2*m_e*B) still works as adiabatic invariant
    mu = p_perp**2 / (2 * m_e * B_eq1)

    print(f"\n(a) First adiabatic invariant mu:")
    print(f"    E_rest = m_e*c^2 = {E_rest/eV:.0f} eV = {E_rest/eV/1e3:.0f} keV")
    print(f"    E_total = E_kin + E_rest = {E_total/eV/1e6:.3f} MeV")
    print(f"    p = sqrt(E_total^2 - E_rest^2)/c = {p:.3e} kg*m/s")
    print(f"    p_perp = p*sin(45 deg) = {p_perp:.3e} kg*m/s")
    print(f"    mu = p_perp^2 / (2*m_e*B_eq)")
    print(f"    mu = {p_perp**2:.3e} / (2*{m_e:.3e}*{B_eq1:.3e})")
    print(f"    mu = {mu:.3e} J/T = {mu/eV:.3e} eV/T")

    # (b) At L=3, mu is conserved
    # p_perp_new^2 = 2 * m_e * mu * B_eq2
    p_perp_new_sq = 2 * m_e * mu * B_eq2
    p_perp_new = np.sqrt(p_perp_new_sq)

    # Perpendicular kinetic energy (non-relativistic approximation for clarity)
    E_perp_new = p_perp_new_sq / (2 * m_e)  # non-relativistic
    # For better estimate with relativistic:
    # E_perp involves the full momentum, but mu conservation gives p_perp

    print(f"\n(b) After adiabatic transport to L = {L2}:")
    print(f"    mu conserved: p_perp_new^2 = 2*m_e*mu*B_eq(L={L2})")
    print(f"    p_perp_new = {p_perp_new:.3e} kg*m/s")

    # The perpendicular momentum increases by sqrt(B2/B1)
    B_ratio = B_eq2 / B_eq1
    p_ratio = np.sqrt(B_ratio)
    print(f"    B_eq ratio: B(L={L2})/B(L={L1}) = {B_ratio:.2f}")
    print(f"    p_perp increases by factor sqrt({B_ratio:.2f}) = {p_ratio:.2f}")

    # New perpendicular energy
    E_perp_old = p_perp**2 / (2 * m_e)
    E_perp_ratio = B_ratio
    print(f"    E_perp_new / E_perp_old = B2/B1 = {B_ratio:.2f}")
    print(f"    Old E_perp = {E_perp_old/eV/1e3:.1f} keV")
    print(f"    New E_perp = {E_perp_new_sq/(2*m_e)/eV/1e3:.1f} keV")

    # (c) Total KE increase factor
    # Assuming parallel energy increases proportionally (second invariant conserved)
    # For equatorial pitch angle of 45 deg, E_perp = E_par = E/2
    # Both scale with B, so total E scales with B ratio
    KE_ratio = B_ratio
    E_new = E_kin * KE_ratio

    print(f"\n(c) Total kinetic energy increase:")
    print(f"    If parallel energy also scales with B (both invariants conserved):")
    print(f"    E_new / E_old = B(L={L2})/B(L={L1}) = (L1/L2)^3 = "
          f"({L1}/{L2})^3 = {KE_ratio:.2f}")
    print(f"    E_new = {E_kin/eV/1e6:.1f} MeV * {KE_ratio:.2f} = "
          f"{E_new/eV/1e6:.2f} MeV")
    print(f"    A {KE_ratio:.1f}x energy increase from inward radial transport!")
    print(f"    This is the betatron acceleration mechanism for radiation belts.")


def exercise_2():
    """
    Exercise 2: Loss Cone

    Dipole field at L=4.
    (a) Equatorial B field.
    (b) B at foot of field line.
    (c) Loss cone angle.
    (d) Fraction of isotropic distribution in loss cone.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Loss Cone Calculation")
    print("=" * 70)

    B0 = 3.1e-5  # T
    L = 4

    # (a) Equatorial field
    B_eq = B0 / L**3

    print(f"\n    L = {L}")
    print(f"\n(a) Equatorial magnetic field:")
    print(f"    B_eq = B0/L^3 = {B0:.1e}/{L}^3 = {B_eq:.3e} T = {B_eq*1e9:.1f} nT")

    # (b) Field at foot of field line
    # Foot latitude: cos^2(lambda_f) = 1/L
    lambda_f = np.arccos(1 / np.sqrt(L))
    lambda_f_deg = np.degrees(lambda_f)

    # B(lambda) = B_eq * sqrt(1 + 3*sin^2(lambda)) / cos^6(lambda)
    sin_lam = np.sin(lambda_f)
    cos_lam = np.cos(lambda_f)
    B_foot = B_eq * np.sqrt(1 + 3 * sin_lam**2) / cos_lam**6

    print(f"\n(b) Field at foot of field line:")
    print(f"    cos^2(lambda_f) = 1/L = 1/{L} => lambda_f = {lambda_f_deg:.1f} deg")
    print(f"    B(lambda) = B_eq * sqrt(1+3*sin^2(lambda)) / cos^6(lambda)")
    print(f"    B_foot = {B_eq:.3e} * sqrt(1+3*{sin_lam**2:.4f}) / {cos_lam**6:.6f}")
    print(f"    B_foot = {B_foot:.3e} T = {B_foot*1e9:.0f} nT")

    # Alternatively, use B_foot ~ B_surface at the footpoint
    # B_surface at lambda_f = B0 * sqrt(1+3*sin^2(lambda_f)) / cos^6(lambda_f) / L^3
    # which should be the same
    B_surface = B0 * np.sqrt(1 + 3 * sin_lam**2)
    print(f"    Cross-check: B at surface at lambda_f = {B_surface*1e9:.0f} nT")

    # (c) Loss cone angle
    # sin^2(alpha_LC) = B_eq / B_foot
    mirror_ratio = B_foot / B_eq
    sin2_alpha = 1 / mirror_ratio
    alpha_LC = np.degrees(np.arcsin(np.sqrt(sin2_alpha)))

    print(f"\n(c) Loss cone angle:")
    print(f"    Mirror ratio: B_foot/B_eq = {mirror_ratio:.2f}")
    print(f"    sin^2(alpha_LC) = B_eq/B_foot = 1/{mirror_ratio:.2f} = {sin2_alpha:.6f}")
    print(f"    alpha_LC = arcsin(sqrt({sin2_alpha:.6f})) = {alpha_LC:.2f} deg")

    # (d) Fraction of isotropic distribution in loss cone
    # Fractional solid angle = 1 - cos(alpha_LC)
    frac = 1 - np.cos(np.radians(alpha_LC))

    print(f"\n(d) Fraction in loss cone (isotropic distribution):")
    print(f"    Fractional solid angle = 1 - cos(alpha_LC)")
    print(f"    = 1 - cos({alpha_LC:.2f} deg) = 1 - {np.cos(np.radians(alpha_LC)):.6f}")
    print(f"    = {frac:.6f} = {frac*100:.4f}%")
    print(f"    Including both hemispheres: {2*frac*100:.4f}%")
    print(f"    => Only a tiny fraction of trapped particles are in the loss cone")
    print(f"    This is why radiation belts can persist for long periods")


def exercise_3():
    """
    Exercise 3: Drift Period and ULF Resonance

    T_d = 2*pi*q*B0*R_E^2 / (3*L*E)
    (a) 500 keV electron at L=5.
    (b) 5 MeV proton at L=2.
    (c) L-shell for drift resonance with 300 s ULF wave for 1 MeV electrons.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Drift Period and ULF Resonance")
    print("=" * 70)

    eV = 1.602e-19
    q = 1.602e-19
    B0 = 3.1e-5
    R_E = 6.371e6

    # (a) 500 keV electron at L=5
    E_a = 500e3 * eV
    L_a = 5
    T_a = 2 * np.pi * q * B0 * R_E**2 / (3 * L_a * E_a)

    print(f"\n(a) 500 keV electron at L = {L_a}:")
    print(f"    T_d = 2*pi*q*B0*R_E^2 / (3*L*E)")
    print(f"    = 2*pi*{q:.3e}*{B0:.1e}*({R_E:.3e})^2 / (3*{L_a}*{E_a:.3e})")
    print(f"    T_d = {T_a:.1f} s = {T_a/60:.1f} min")

    # (b) 5 MeV proton at L=2
    E_b = 5e6 * eV
    L_b = 2
    T_b = 2 * np.pi * q * B0 * R_E**2 / (3 * L_b * E_b)

    print(f"\n(b) 5 MeV proton at L = {L_b}:")
    print(f"    T_d = 2*pi*q*B0*R_E^2 / (3*{L_b}*{E_b:.3e})")
    print(f"    T_d = {T_b:.1f} s = {T_b/60:.1f} min")
    print(f"    (Same formula — drift period depends on energy per charge,")
    print(f"     not on mass. Protons and electrons of same energy drift equally fast.)")

    # (c) ULF resonance at 300 s for 1 MeV electrons
    T_ULF = 300  # s
    E_c = 1e6 * eV
    # T_d = 2*pi*q*B0*R_E^2 / (3*L*E) = T_ULF
    # L = 2*pi*q*B0*R_E^2 / (3*E*T_ULF)
    L_res = 2 * np.pi * q * B0 * R_E**2 / (3 * E_c * T_ULF)

    print(f"\n(c) Drift resonance with {T_ULF} s ULF wave (1 MeV electrons):")
    print(f"    Set T_d = T_ULF = {T_ULF} s")
    print(f"    L = 2*pi*q*B0*R_E^2 / (3*E*T_ULF)")
    print(f"    L = {L_res:.2f}")
    print(f"    This is in the heart of the outer radiation belt (L ~ {L_res:.1f})")
    print(f"    ULF waves at Pc5 periods (~150-600 s) are known to efficiently")
    print(f"    accelerate MeV electrons through drift resonance at L ~ 4-6")


def exercise_4():
    """
    Exercise 4: Phase Space Density Analysis

    j = 1e4 particles/(cm^2 s sr MeV) at L*=4.5, E=1 MeV.
    (a) Relativistic momentum for 1 MeV electron.
    (b) Convert flux to PSD.
    (c) Compare PSD at L*=5.5 and L*=3.5 to determine source mechanism.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Phase Space Density Analysis")
    print("=" * 70)

    eV = 1.602e-19
    m_e = 9.109e-31
    c = 2.998e8

    j = 1e4         # particles/(cm^2 s sr MeV)
    E_kin = 1.0e6    # eV
    E_kin_J = E_kin * eV

    # (a) Relativistic momentum
    E_rest = m_e * c**2  # rest energy in Joules
    E_rest_eV = E_rest / eV
    E_total_J = E_kin_J + E_rest
    p = np.sqrt(E_total_J**2 - E_rest**2) / c

    # Express p in MeV/c
    p_MeV_c = p * c / (1e6 * eV)

    print(f"\n    j = {j:.0e} particles/(cm^2 s sr MeV) at E = 1 MeV")

    print(f"\n(a) Relativistic momentum of 1 MeV electron:")
    print(f"    E_rest = m_e*c^2 = {E_rest_eV/1e3:.1f} keV = {E_rest_eV/1e6:.3f} MeV")
    print(f"    E_total = E_kin + E_rest = {E_kin/1e6:.3f} + {E_rest_eV/1e6:.3f} "
          f"= {(E_kin + E_rest_eV)/1e6:.3f} MeV")
    print(f"    p*c = sqrt(E_total^2 - E_rest^2)")
    print(f"    p = {p:.3e} kg*m/s")
    print(f"    p = {p_MeV_c:.3f} MeV/c")

    # (b) PSD: f = j / p^2
    # In GEM units: c^3 / (MeV^3 * cm^3)
    # j in particles/(cm^2 s sr MeV), p in MeV/c
    # f = j / p^2  with p in MeV/c gives units of c^3/(MeV^3 cm^3) * (1/sr)
    # Actually, j is differential directional flux:
    # f = j / p^2 gives [particles/(cm^2 s sr MeV)] / [(MeV/c)^2]
    # = particles * c^2 / (cm^2 s sr MeV^3)
    # GEM convention: f = j / p^2 (with j in /cm^2/s/sr/MeV, p in MeV/c)
    f_45 = j / p_MeV_c**2

    print(f"\n(b) Phase space density:")
    print(f"    f = j / p^2")
    print(f"    = {j:.0e} / ({p_MeV_c:.3f})^2")
    print(f"    = {f_45:.3e} c^3/(MeV^3 cm^3)")

    # (c) PSD comparison
    f_55 = 1.5e-5   # at L*=5.5
    f_35 = 2e-6     # at L*=3.5

    print(f"\n(c) PSD radial profile comparison (at same mu):")
    print(f"    L* = 3.5: f = {f_35:.1e} c^3/(MeV^3 cm^3)")
    print(f"    L* = 4.5: f = {f_45:.3e} c^3/(MeV^3 cm^3)")
    print(f"    L* = 5.5: f = {f_55:.1e} c^3/(MeV^3 cm^3)")

    print(f"\n    Analysis:")
    if f_45 > f_55 and f_45 > f_35:
        print(f"    PSD has a LOCAL PEAK at L* = 4.5")
        print(f"    This indicates LOCAL ACCELERATION (not radial diffusion)")
        print(f"    Radial diffusion would produce a monotonically decreasing")
        print(f"    PSD profile from source to sink, not a local peak.")
        print(f"    The local acceleration is likely driven by VLF chorus waves")
        print(f"    interacting with seed electrons in the outer belt.")
    elif f_55 > f_45 > f_35:
        print(f"    PSD increases outward: consistent with INWARD RADIAL DIFFUSION")
        print(f"    from an external source")
    elif f_35 > f_45 > f_55:
        print(f"    PSD decreases outward: consistent with OUTWARD RADIAL DIFFUSION")
        print(f"    from an internal source")


def exercise_5():
    """
    Exercise 5: SAA Radiation Dose

    Satellite: 600 km, 28.5 deg inclination.
    6 SAA passes/day, 10 min each.
    Proton flux (>30 MeV): 1e3 protons/(cm^2 s sr).
    (a) Daily fluence.
    (b) Annual fluence.
    (c) Annual TID (dose per proton = 1e-8 rad per proton/cm^2 behind 3mm Al).
    """
    print("\n" + "=" * 70)
    print("Exercise 5: SAA Radiation Dose")
    print("=" * 70)

    n_passes = 6           # per day
    t_pass = 10 * 60       # 10 minutes in seconds
    flux = 1e3             # protons/(cm^2 s sr)

    # (a) Daily fluence
    # For isotropic flux, fluence = flux * 4*pi (integrate over all solid angles)
    # Actually, for a unidirectional detector or total omnidirectional flux:
    # Omnidirectional fluence rate = flux * 4*pi (integrating over full sphere)
    # But if the flux is given as directional (per sr), the total fluence
    # impinging on a surface from one hemisphere = flux * pi (cosine-weighted)
    # For radiation dose, typically use 4*pi*j (omnidirectional):
    fluence_per_pass = flux * 4 * np.pi * t_pass
    fluence_daily = n_passes * fluence_per_pass

    print(f"\n    SAA parameters:")
    print(f"    Passes per day: {n_passes}")
    print(f"    Duration per pass: {t_pass/60:.0f} min = {t_pass} s")
    print(f"    Proton flux (>30 MeV): {flux:.0e} protons/(cm^2 s sr)")

    print(f"\n(a) Daily fluence (assuming isotropic flux):")
    print(f"    Omnidirectional fluence rate = flux * 4*pi")
    print(f"    = {flux:.0e} * 4*pi = {flux*4*np.pi:.2e} protons/(cm^2 s)")
    print(f"    Per pass: {fluence_per_pass:.3e} protons/cm^2")
    print(f"    Daily: {n_passes} passes * {fluence_per_pass:.3e}")
    print(f"    = {fluence_daily:.3e} protons/cm^2 per day")

    # (b) Annual fluence
    fluence_annual = fluence_daily * 365
    print(f"\n(b) Annual fluence:")
    print(f"    = {fluence_daily:.3e} * 365")
    print(f"    = {fluence_annual:.3e} protons/cm^2 per year")

    # (c) Annual TID
    dose_per_proton = 1e-8  # rad per proton/cm^2
    TID_annual = fluence_annual * dose_per_proton

    print(f"\n(c) Annual total ionizing dose (TID):")
    print(f"    Dose per proton = {dose_per_proton:.0e} rad per proton/cm^2")
    print(f"    TID = fluence * dose_per_proton")
    print(f"    = {fluence_annual:.3e} * {dose_per_proton:.0e}")
    print(f"    = {TID_annual:.1f} rad/year")
    print(f"    = {TID_annual/1e3:.3f} krad/year")

    print(f"\n    Context:")
    print(f"    - Typical component tolerance: 10-100 krad total")
    print(f"    - At {TID_annual/1e3:.3f} krad/year from SAA protons alone,")
    print(f"      a 5-year mission accumulates ~{5*TID_annual/1e3:.2f} krad")
    print(f"    - This is the dose behind 3mm Al shielding; unshielded dose")
    print(f"      would be much higher")
    print(f"    - Additional dose comes from trapped electrons and solar events")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
