;+
; Exercise 03: Map Projections
;-

PRO exercise_03

    ; === Exercise 1: Four projections ===
    ; Display the same data on 4 different map projections in a 2x2 layout
    ; TODO: Use MAP_SET with Orthographic, Mollweide, Aitoff, Stereographic
    ; TODO: Add MAP_CONTINENTS and MAP_GRID to each

    ; === Exercise 2: Disk-to-heliographic conversion ===
    ; Convert the following disk positions (arcsec) to heliographic coordinates:
    ; (a) (0, 0) — disk center
    ; (b) (500, 0) — east limb region
    ; (c) (0, 800) — near north pole
    ; Assume R_sun = 960 arcsec, B0 = 0, L0 = 0
    rsun = 960.0
    positions = [[0, 0], [500, 0], [0, 800]]
    ; TODO: Compute heliographic lat/lon for each position
    ; TODO: Check if position is on-disk (rho < 1)
    ; Hint: rho = sqrt(x^2 + y^2) / R_sun

    ; === Exercise 3: Differential rotation ===
    ; Compute how far a feature at latitude 30 degrees drifts in Carrington
    ; longitude over 10 days. Use Snodgrass & Ulrich coefficients.
    ; TODO: Implement diff_rot_rate function
    ; TODO: Compute drift = omega(30) * 10 days
    ; Hint: omega(lat) = 14.713 - 2.396*sin^2(lat) - 1.787*sin^4(lat) deg/day

END
