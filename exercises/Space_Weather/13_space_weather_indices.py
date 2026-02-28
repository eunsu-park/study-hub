"""
Exercise Solutions for Lesson 13: Space Weather Indices

Topics covered:
  - Dst calculation from magnetometer data and DPS relation
  - K-index determination and Kp/ap conversion
  - AE/AU/AL/AO auroral electrojet indices
  - F10.7 to sunspot number conversion
  - NOAA space weather scale classification (G, S, R)
"""

import numpy as np


def exercise_1():
    """
    Exercise 1: Dst from Magnetometer and DPS Relation

    Station at magnetic latitude 30 deg, measures Delta_H = -86.6 nT.
    Calculate Dst contribution. Estimate ring current energy if all 4 stations
    show similar values.
    """
    print("=" * 70)
    print("Exercise 1: Dst from Magnetometer Data")
    print("=" * 70)

    mu0 = 4 * np.pi * 1e-7
    B0 = 3.1e-5
    R_E = 6.371e6

    lat = 30  # magnetic latitude
    Delta_H = -86.6  # nT

    # Dst correction for latitude: Dst_station = Delta_H / cos(lat)
    # Dst is the average over all stations of H-perturbation / cos(latitude)
    cos_lat = np.cos(np.radians(lat))
    Dst_station = Delta_H / cos_lat

    print(f"\n    Station: magnetic latitude = {lat} deg")
    print(f"    Measured Delta_H = {Delta_H} nT")

    print(f"\n    Dst contribution from this station:")
    print(f"    Dst = Delta_H / cos(lat) = {Delta_H} / cos({lat} deg)")
    print(f"    = {Delta_H} / {cos_lat:.4f}")
    print(f"    = {Dst_station:.1f} nT")

    # If all 4 Dst stations show similar values
    Dst = Dst_station  # average over stations

    print(f"\n    If all 4 stations show similar corrected values:")
    print(f"    Dst ~ {Dst:.0f} nT")

    # Ring current energy from DPS
    E_rc = abs(Dst * 1e-9) * B0 * R_E**3 / (2 * mu0)

    print(f"\n    Ring current energy (DPS relation):")
    print(f"    E_rc = |Dst| * B0 * R_E^3 / (2*mu0)")
    print(f"    = {abs(Dst)*1e-9:.1e} * {B0:.1e} * ({R_E:.3e})^3 / (2*{mu0:.3e})")
    print(f"    = {E_rc:.3e} J")
    print(f"    This is a moderate storm (~{abs(Dst):.0f} nT Dst)")


def exercise_2():
    """
    Exercise 2: K-Index, Kp, and ap

    3-hour interval: max deviation +45 nT, min deviation -55 nT.
    Station K=5 threshold: 70 nT.
    If Kp = 5o, find ap and NOAA G-scale.
    """
    print("\n" + "=" * 70)
    print("Exercise 2: K-Index, Kp, and ap")
    print("=" * 70)

    max_dev = 45     # nT
    min_dev = -55    # nT
    K5_threshold = 70  # nT

    # K-index is based on the range (max - min) of deviations
    dev_range = max_dev - min_dev

    print(f"\n    3-hour interval deviations: max = +{max_dev} nT, min = {min_dev} nT")
    print(f"    Range = {max_dev} - ({min_dev}) = {dev_range} nT")

    # K-index lookup: The range is compared to the station's K thresholds
    # K=5 threshold is 70 nT, so the thresholds are typically:
    # K: 0  1  2  3  4  5  6  7  8  9
    # For a station with K5=70, thresholds scale approximately:
    # K=0: 0-5, K=1: 5-10, K=2: 10-20, K=3: 20-40, K=4: 40-70,
    # K=5: 70-120, K=6: 120-200, K=7: 200-330, K=8: 330-500, K=9: >500

    print(f"    Station K=5 threshold: {K5_threshold} nT")
    print(f"    Range {dev_range} nT > {K5_threshold} nT (K=5 threshold)")

    # Check if it exceeds K=6 threshold (~120 nT for this scaling)
    K6_threshold = int(K5_threshold * 120 / 70)  # approximate
    if dev_range >= K6_threshold:
        K_local = 6
    elif dev_range >= K5_threshold:
        K_local = 5
    else:
        K_local = 4

    print(f"    Approximate K=6 threshold: ~{K6_threshold} nT")
    print(f"    {dev_range} nT is between K=5 ({K5_threshold}) and K=6 (~{K6_threshold})")
    print(f"    Local K = {K_local}")

    # Kp and ap conversion
    # Kp = 5o (5 minus) means Kp = 4.67 in the thirds system
    # Kp thirds: 5- = 4.67, 5o = 5.0, 5+ = 5.33
    Kp = 5.0  # 5o

    # ap conversion table (standard):
    # Kp:  0  0+  1-  1  1+  2-  2  2+  3-  3  3+  4-  4  4+  5-  5  5+  6-  6  6+  7-  7  7+  8-  8  8+  9-  9
    # ap:  0   2   3   4   5   6   7   9  12  15  18  22  27  32  39  48  56  67  80  94 111 132 154 179 207 236 300 400
    ap_table = {
        0: 0, 0.33: 2, 0.67: 3, 1: 4, 1.33: 5, 1.67: 6, 2: 7, 2.33: 9,
        2.67: 12, 3: 15, 3.33: 18, 3.67: 22, 4: 27, 4.33: 32, 4.67: 39,
        5: 48, 5.33: 56, 5.67: 67, 6: 80, 6.33: 94, 6.67: 111, 7: 132,
        7.33: 154, 7.67: 179, 8: 207, 8.33: 236, 8.67: 300, 9: 400
    }

    ap = ap_table.get(Kp, "unknown")

    print(f"\n    Kp = 5o (= {Kp:.1f} in decimal)")
    print(f"    From standard Kp-to-ap table:")
    print(f"    ap = {ap}")

    # NOAA G-scale
    # G1: Kp=5, G2: Kp=6, G3: Kp=7, G4: Kp=8, G5: Kp=9
    G_scale = {5: "G1 (Minor)", 6: "G2 (Moderate)", 7: "G3 (Strong)",
               8: "G4 (Severe)", 9: "G5 (Extreme)"}
    G_level = G_scale.get(int(Kp), "< G1")

    print(f"\n    NOAA G-scale:")
    print(f"    Kp = {Kp:.0f} -> {G_level}")
    print(f"    Impacts: Minor power grid fluctuations, minor impact on")
    print(f"    satellite operations, aurora visible at high latitudes")


def exercise_3():
    """
    Exercise 3: AE/AU/AL/AO Computation

    Auroral stations: A: +350, B: -620, C: +180, D: -480, E: +50 nT.
    """
    print("\n" + "=" * 70)
    print("Exercise 3: AE/AU/AL/AO Indices")
    print("=" * 70)

    stations = {'A': 350, 'B': -620, 'C': 180, 'D': -480, 'E': 50}

    H_values = list(stations.values())
    names = list(stations.keys())

    AU = max(H_values)
    AL = min(H_values)
    AE = AU - AL
    AO = (AU + AL) / 2

    AU_station = names[H_values.index(AU)]
    AL_station = names[H_values.index(AL)]

    print(f"\n    Station H-deviations (nT):")
    for name, val in stations.items():
        print(f"    Station {name}: {val:+.0f} nT")

    print(f"\n    AU (maximum upper envelope): {AU} nT (Station {AU_station})")
    print(f"    AL (minimum lower envelope): {AL} nT (Station {AL_station})")
    print(f"    AE = AU - AL = {AU} - ({AL}) = {AE} nT")
    print(f"    AO = (AU + AL) / 2 = ({AU} + {AL}) / 2 = {AO:.0f} nT")

    print(f"\n    Interpretation:")
    print(f"    Station {AU_station} (+{AU} nT): most likely under the EASTWARD electrojet")
    print(f"    (positive H deviation from eastward current overhead)")
    print(f"    Station {AL_station} ({AL} nT): most likely under the WESTWARD electrojet")
    print(f"    (negative H deviation from westward current overhead)")
    print(f"    AE = {AE} nT indicates {'strong' if AE > 500 else 'moderate'} "
          f"auroral activity")
    print(f"    AO = {AO:.0f} nT indicates the asymmetry between the electrojets")


def exercise_4():
    """
    Exercise 4: F10.7 to Sunspot Number

    F10.7 = 185 SFU. R ~ 1.1 * (F10.7 - 67).
    Observer sees 8 groups, 63 spots. Find k-factor.
    """
    print("\n" + "=" * 70)
    print("Exercise 4: F10.7 and Sunspot Number")
    print("=" * 70)

    F107 = 185  # SFU

    # Approximate relationship: R ~ 1.1 * (F10.7 - 67)
    R_est = 1.1 * (F107 - 67)

    print(f"\n    F10.7 = {F107} SFU")
    print(f"    R ~ 1.1 * (F10.7 - 67)")
    print(f"    R ~ 1.1 * ({F107} - 67) = 1.1 * {F107-67} = {R_est:.0f}")

    # Wolf sunspot number: R = k * (10*g + s)
    # where g = number of groups, s = number of individual spots
    g = 8     # groups
    s = 63    # individual spots
    R_raw = 10 * g + s

    # If R matches our estimate:
    k = R_est / R_raw

    print(f"\n    Observer's raw count:")
    print(f"    Groups: g = {g}, Individual spots: s = {s}")
    print(f"    Raw Wolf number = 10*g + s = 10*{g} + {s} = {R_raw}")

    print(f"\n    Wolf number: R = k * (10*g + s)")
    print(f"    If R = {R_est:.0f} (from F10.7 estimate):")
    print(f"    k = R / (10*g + s) = {R_est:.0f} / {R_raw} = {k:.3f}")

    print(f"\n    The k-factor ({k:.3f}) is the observer's personal correction factor")
    print(f"    It accounts for telescope aperture, seeing conditions, and")
    print(f"    observer experience. k < 1 means the observer over-counts")
    print(f"    (good telescope/conditions); k > 1 means under-counting.")


def exercise_5():
    """
    Exercise 5: NOAA Scale Classification

    Peak proton flux = 5000 pfu (>10 MeV), X5 flare, Kp = 8.
    Classify on G, S, R scales. Describe one impact per scale.
    """
    print("\n" + "=" * 70)
    print("Exercise 5: NOAA Scale Classification")
    print("=" * 70)

    proton_flux = 5000  # pfu (>10 MeV)
    flare_class = "X5"
    Kp = 8

    print(f"\n    Observed conditions:")
    print(f"    Proton flux (>10 MeV): {proton_flux} pfu")
    print(f"    X-ray flare class: {flare_class}")
    print(f"    Kp: {Kp}")

    # G-scale (Geomagnetic storms based on Kp)
    # G1: Kp=5, G2: Kp=6, G3: Kp=7, G4: Kp=8, G5: Kp=9
    print(f"\n    G-SCALE (Geomagnetic Storms):")
    print(f"    Kp = {Kp} -> G4 (Severe)")
    print(f"    Impact: Widespread voltage control problems in power grids.")
    print(f"    Some grid protective systems may trip key assets. Satellite")
    print(f"    surface charging and tracking difficulties. Aurora visible")
    print(f"    at mid-latitudes (~45 deg geographic latitude).")

    # S-scale (Solar Radiation Storms based on >10 MeV proton flux)
    # S1: 10, S2: 100, S3: 1000, S4: 10000, S5: 100000
    if proton_flux >= 1e5:
        S = "S5 (Extreme)"
    elif proton_flux >= 1e4:
        S = "S4 (Severe)"
    elif proton_flux >= 1e3:
        S = "S3 (Strong)"
    elif proton_flux >= 1e2:
        S = "S2 (Moderate)"
    elif proton_flux >= 10:
        S = "S1 (Minor)"
    else:
        S = "Below S1"

    print(f"\n    S-SCALE (Solar Radiation Storms):")
    print(f"    Proton flux = {proton_flux} pfu -> {S}")
    print(f"    Impact: Elevated radiation risk for polar flights and EVA.")
    print(f"    Satellite memory device problems (SEU rate increase).")
    print(f"    Degraded satellite navigation accuracy at high latitudes.")

    # R-scale (Radio Blackouts based on X-ray class)
    # R1: M1, R2: M5, R3: X1, R4: X10, R5: X20+
    # X5 -> between X1 (R3) and X10 (R4)
    print(f"\n    R-SCALE (Radio Blackouts):")
    print(f"    Flare class: {flare_class} -> R3 (Strong)")
    print(f"    (X1=R3, X10=R4, X5 falls in R3 range)")
    print(f"    Impact: Wide-area HF radio blackout on the sunlit side")
    print(f"    lasting approximately 1 hour. Loss of HF communication")
    print(f"    for transoceanic aviation and maritime operations.")
    print(f"    Low-frequency navigation signals degraded for ~1 hour.")

    # Combined assessment
    print(f"\n    COMBINED ASSESSMENT:")
    print(f"    This is a severe multi-hazard space weather event.")
    print(f"    All three scales are elevated simultaneously:")
    print(f"    G4 + S3 + R3 = a major coordinated event")
    print(f"    Operators should activate emergency procedures across")
    print(f"    all sectors: power, satellites, aviation, and communications.")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
