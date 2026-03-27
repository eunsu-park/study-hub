"""
Exercises for Lesson 15: Modern Solar Missions
Topic: Solar_Physics
Solutions to practice problems from the lesson.
"""

import numpy as np


# --- Physical constants ---
sigma_SB = 5.670e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]
R_sun = 6.957e8        # solar radius [m]
AU = 1.496e11          # astronomical unit [m]
S_0 = 1361.0           # solar constant at 1 AU [W/m^2]


def exercise_1():
    """
    Problem 1: PSP Heat Shield Temperature

    Closest perihelion: 9.86 R_sun.
    Solar constant at 1 AU = 1361 W/m^2.
    TPS reflectivity = 0.6.
    Estimate equilibrium temperature of heat shield.
    """
    r_perihelion = 9.86  # R_sun
    reflectivity = 0.6
    # 1 AU = 215 R_sun
    AU_Rsun = AU / R_sun  # ~ 215

    # Solar intensity at perihelion (inverse-square law)
    F_perihelion = S_0 * (AU / (r_perihelion * R_sun))**2
    # Equivalently: F = S_0 * (AU_Rsun / r_perihelion)^2

    print(f"  PSP closest perihelion: {r_perihelion} R_sun")
    print(f"  Distance ratio: 1 AU / r_p = {AU_Rsun:.0f} / {r_perihelion} = {AU_Rsun/r_perihelion:.0f}")
    print(f"  Solar intensity at perihelion:")
    print(f"    F = S_0 * (d_AU / d_p)^2")
    print(f"      = {S_0} * ({AU_Rsun/r_perihelion:.0f})^2")
    print(f"      = {F_perihelion:.0f} W/m^2")
    print(f"      = {F_perihelion/S_0:.0f} x solar constant")

    # Absorbed power per unit area
    absorptivity = 1.0 - reflectivity
    F_absorbed = absorptivity * F_perihelion

    print(f"\n  TPS reflectivity: {reflectivity}")
    print(f"  Absorptivity: {absorptivity}")
    print(f"  Absorbed flux: F_abs = {absorptivity} * {F_perihelion:.0f} = {F_absorbed:.0f} W/m^2")

    # Equilibrium temperature (re-radiating from one face as blackbody)
    # F_absorbed = sigma * T^4
    T_eq = (F_absorbed / sigma_SB)**0.25

    print(f"\n  Equilibrium temperature (radiating from one face):")
    print(f"    F_abs = sigma T^4")
    print(f"    T = (F_abs / sigma)^(1/4)")
    print(f"      = ({F_absorbed:.0f} / {sigma_SB:.3e})^0.25")
    print(f"      = {T_eq:.0f} K = {T_eq - 273:.0f} C")

    # PSP actual shield temperature
    print(f"\n  The actual PSP TPS temperature at closest approach is ~1400 C (~1670 K).")
    print(f"  Our estimate of ~{T_eq-273:.0f} C is in the right ballpark.")
    print(f"  Differences arise from: geometry (not a flat plate), edge effects,")
    print(f"  thermal conductivity, and the shield radiating from both sides.")


def exercise_2():
    """
    Problem 2: Solar Orbiter Inclination

    Max orbital inclination: 33 degrees.
    Sun's rotation axis tilt: 7.25 degrees from ecliptic normal.
    """
    i_orbit = 33.0     # degrees
    tilt = 7.25        # degrees (solar axis tilt from ecliptic normal)

    print(f"  Solar Orbiter max orbital inclination: {i_orbit} deg")
    print(f"  Sun's rotation axis tilt from ecliptic normal: {tilt} deg")

    # Highest observable heliographic latitude
    lat_max = i_orbit  # degrees (direct imaging from above the equatorial plane)
    print(f"\n  Highest heliographic latitude directly imaged: {lat_max} deg")
    print(f"  (Spacecraft is at {i_orbit} deg above ecliptic plane)")

    # With favorable geometry (spacecraft over pole when Sun's axis is tilted toward it)
    lat_max_favorable = i_orbit + tilt
    lat_max_unfavorable = i_orbit - tilt

    print(f"\n  Solar axis tilt effect:")
    print(f"    Favorable geometry (axis tilted toward spacecraft):")
    print(f"      Max latitude = {i_orbit} + {tilt} = {lat_max_favorable} deg")
    print(f"    Unfavorable geometry (axis tilted away):")
    print(f"      Max latitude = {i_orbit} - {tilt} = {lat_max_unfavorable} deg")

    print(f"\n  With optimal timing, Solar Orbiter can view latitudes up to")
    print(f"  ~{lat_max_favorable:.1f} deg. However, this is still short of a true polar view (90 deg).")

    print(f"\n  Why no previous mission imaged the solar poles:")
    print(f"    - Ulysses (1994-2009) reached 80 deg heliographic latitude")
    print(f"      but carried NO imager (only in situ instruments)")
    print(f"    - Getting high inclination requires large delta-v or gravity assists")
    print(f"    - Solar Orbiter uses Venus gravity assists to increase inclination")
    print(f"    - STEREO observed from the ecliptic plane at different longitudes")
    print(f"    - The poles are important for understanding the solar dynamo")
    print(f"      (polar magnetic fields seed the next cycle)")


def exercise_3():
    """
    Problem 3: SDO/AIA Data Rate

    4096x4096 pixels, 16-bit, 10 channels, every 12 seconds.
    Calculate raw data rate and total mission data volume.
    """
    pixels = 4096 * 4096
    bits_per_pixel = 16
    channels = 10
    cadence = 12.0        # seconds
    mission_years = 15.0

    # (a) Raw data rate
    bytes_per_image = pixels * bits_per_pixel / 8
    bytes_per_set = bytes_per_image * channels
    MB_per_set = bytes_per_set / (1024**2)

    data_rate_Bps = bytes_per_set / cadence  # bytes/s
    data_rate_MBps = data_rate_Bps / (1024**2)

    print(f"  SDO/AIA parameters:")
    print(f"    Image size: 4096 x 4096 = {pixels:,} pixels")
    print(f"    Bit depth: {bits_per_pixel} bits/pixel")
    print(f"    Channels: {channels}")
    print(f"    Cadence: {cadence:.0f} seconds")
    print(f"  ")
    print(f"  Bytes per image: {pixels} * {bits_per_pixel}/8 = {bytes_per_image:,.0f} bytes")
    print(f"                 = {bytes_per_image/(1024**2):.0f} MB")
    print(f"  Bytes per channel set: {MB_per_set:.0f} MB")
    print(f"  ")
    print(f"  Raw data rate: {MB_per_set:.0f} MB / {cadence:.0f} s = {data_rate_MBps:.1f} MB/s")

    # Total over mission
    seconds_per_year = 365.25 * 24 * 3600
    total_seconds = mission_years * seconds_per_year
    total_bytes = data_rate_Bps * total_seconds
    total_TB = total_bytes / (1024**4)
    total_PB = total_bytes / (1024**5)

    print(f"\n  Over {mission_years:.0f}-year mission:")
    print(f"    Total data: {data_rate_MBps:.1f} MB/s * {total_seconds:.2e} s")
    print(f"              = {total_TB:.0f} TB = {total_PB:.1f} PB")

    # Also count images
    total_images = total_seconds / cadence * channels
    print(f"    Total images: ~{total_images:.1e}")

    print(f"\n  Why machine learning became essential:")
    print(f"    - {total_PB:.1f} PB of data is far too large for manual inspection")
    print(f"    - ~{total_images:.0e} images require automated classification")
    print(f"    - ML excels at pattern recognition: flare detection, AR classification,")
    print(f"      coronal hole identification, CME onset detection")
    print(f"    - Deep learning can extract features impossible to code manually")
    print(f"    - Real-time processing needed for space weather forecasting")
    print(f"    - Transfer learning leverages SDO data for other missions")


def exercise_4():
    """
    Problem 4: DKIST Diffraction Limit and Resolution

    DKIST 4m mirror.
    At 500 nm: 0.031 arcsec (22 km).
    Granules ~1000 km, bright points 100-200 km.
    Fe XIII 1074.7 nm coronal line.
    """
    D = 4.0             # mirror diameter [m]
    lambda_vis = 500.0e-9    # m
    lambda_IR = 1074.7e-9    # m
    arcsec_to_km = 725.0

    # Diffraction limit at 500 nm (given: 0.031 arcsec = 22 km)
    theta_vis = 1.22 * lambda_vis / D * 206265  # arcsec
    res_vis_km = theta_vis * arcsec_to_km

    print(f"  DKIST 4m mirror diffraction limit:")
    print(f"    At 500 nm: {theta_vis:.3f} arcsec = {res_vis_km:.0f} km on the Sun")

    # Granules
    granule_km = 1000.0
    elements_per_granule = granule_km / res_vis_km
    print(f"\n  Granule size: ~{granule_km:.0f} km")
    print(f"  Resolution elements across one granule: {granule_km}/{res_vis_km:.0f} = {elements_per_granule:.0f}")

    # Bright points
    bp_min, bp_max = 100.0, 200.0
    print(f"\n  Bright points: {bp_min:.0f}-{bp_max:.0f} km")
    print(f"  Can DKIST resolve them? Resolution = {res_vis_km:.0f} km")
    if res_vis_km < bp_min:
        print(f"  YES -- DKIST can resolve even the smallest bright points.")
    elif res_vis_km < bp_max:
        print(f"  Partially -- DKIST can resolve the larger ones ({bp_max:.0f} km)")
    else:
        print(f"  NO -- bright points are below the resolution limit.")

    # Fe XIII 1074.7 nm
    theta_IR = 1.22 * lambda_IR / D * 206265  # arcsec
    res_IR_km = theta_IR * arcsec_to_km

    print(f"\n  At Fe XIII 1074.7 nm:")
    print(f"    Diffraction limit: {theta_IR:.3f} arcsec = {res_IR_km:.0f} km")

    print(f"\n  Why this is still valuable for coronal science:")
    print(f"    - {res_IR_km:.0f} km ({theta_IR:.2f} arcsec) can resolve individual")
    print(f"      coronal loops (typical width: 1-10 Mm = 1000-10000 km)")
    print(f"    - DKIST provides spectropolarimetry of the Fe XIII line:")
    print(f"      -> coronal MAGNETIC FIELD measurement (unique capability)")
    print(f"    - No space-based EUV imager measures coronal magnetic fields")
    print(f"    - The resolution is better than any previous coronal magnetometry")
    print(f"    - Even at {theta_IR:.2f} arcsec, DKIST surpasses most coronagraphs")
    print(f"    - Combines magnetic field + velocity + temperature diagnostics")


def exercise_5():
    """
    Problem 5: Vigil L5 CME Prediction

    Halo CME at L1: 500 km/s plane-of-sky speed.
    Vigil at L5: 1200 km/s true radial speed.
    Deceleration to 800 km/s by 1 AU. Estimate transit time.
    """
    v_L1 = 500.0         # km/s (plane-of-sky from L1)
    v_true = 1200.0       # km/s (true radial from L5)
    v_final = 800.0       # km/s (at 1 AU after deceleration)
    r_launch = 2.0        # R_sun

    print(f"  SOHO/LASCO at L1 (head-on view): v_POS = {v_L1:.0f} km/s")
    print(f"  Vigil at L5 (side view): v_true = {v_true:.0f} km/s")

    # (a) Explain discrepancy
    print(f"\n  Speed discrepancy:")
    print(f"    v_true / v_POS = {v_true/v_L1:.1f}")
    print(f"    A halo CME seen from L1 is moving TOWARD the observer.")
    print(f"    The coronagraph measures only the PLANE-OF-SKY (POS) speed,")
    print(f"    which is the component perpendicular to the line of sight.")
    print(f"    For an Earth-directed CME, most of the velocity is ALONG the")
    print(f"    line of sight (radial), which the coronagraph cannot measure.")
    print(f"    Result: severe underestimate of the true speed (projection effect).")
    print(f"    ")
    print(f"    Vigil at L5 (60 deg from Sun-Earth line) sees the CME from the")
    print(f"    side, measuring the true radial propagation directly in the POS.")

    # (b) Transit time estimate
    # Average speed: (v_true + v_final) / 2 assuming linear deceleration
    v_avg = (v_true + v_final) / 2.0  # km/s

    # Distance from launch to 1 AU
    d_launch = r_launch * R_sun / 1e3  # km
    d_AU = AU / 1e3  # km
    d_travel = d_AU - d_launch  # km

    t_transit_s = d_travel / v_avg
    t_transit_hr = t_transit_s / 3600.0

    print(f"\n  Transit time estimate:")
    print(f"    Launch distance: {r_launch:.0f} R_sun = {d_launch:.0f} km")
    print(f"    Distance to 1 AU: {d_travel:.3e} km")
    print(f"    Initial speed: {v_true:.0f} km/s, final speed: {v_final:.0f} km/s")
    print(f"    Average speed (linear deceleration): {v_avg:.0f} km/s")
    print(f"    Transit time: d/v_avg = {t_transit_hr:.0f} hours = {t_transit_hr/24:.1f} days")

    # With only L1 data (using v_POS = 500 km/s)
    v_avg_L1 = (v_L1 + v_L1 * 0.8) / 2.0  # assume similar deceleration fraction
    t_L1 = d_travel / v_avg_L1
    t_L1_hr = t_L1 / 3600.0

    print(f"\n  If using only L1 data (v = {v_L1:.0f} km/s):")
    print(f"    Estimated transit: ~{t_L1_hr:.0f} hours = {t_L1_hr/24:.1f} days")
    print(f"    ACTUAL transit: ~{t_transit_hr:.0f} hours = {t_transit_hr/24:.1f} days")
    print(f"    Prediction error: {abs(t_L1_hr - t_transit_hr):.0f} hours!")

    print(f"\n  Current prediction uncertainty: 12-18 hours")
    print(f"  How Vigil improves the forecast:")
    print(f"    - Measures true CME speed (not projected), eliminating the")
    print(f"      largest source of error for Earth-directed events")
    print(f"    - Provides 3D reconstruction when combined with L1 data")
    print(f"    - Enables better drag model inputs (true initial speed)")
    print(f"    - Can reduce transit time uncertainty to ~6 hours or less")
    print(f"    - Also images the CME structure/size from the side")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: PSP Heat Shield Temperature ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Solar Orbiter Inclination ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: SDO Data Rate ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: DKIST Resolution ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Vigil L5 CME Prediction ===")
    print("=" * 70)
    exercise_5()

    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
