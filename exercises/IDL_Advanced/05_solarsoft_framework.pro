;+
; Exercise 05: SolarSoft Framework
; Requires: SolarSoft (sswidl)
;-

PRO exercise_05

    ; === Exercise 1: Time conversion ===
    ; Convert '2024-07-04T15:30:00' to all available formats:
    ; CCSDS, VMS, ECS, MJD, and external (array)
    ; TODO: Use ANYTIM with different output keywords

    ; === Exercise 2: Time arithmetic ===
    ; Given a start time '2024-01-01T00:00:00' and end time '2024-12-31T23:59:59':
    ; (a) Compute the difference in days
    ; (b) Generate a time grid with 1-day cadence
    ; (c) Convert each time to Carrington rotation number
    ; TODO: Use ANYTIM, TIMEGRID, TIM2CARR

    ; === Exercise 3: Solar ephemeris ===
    ; Compute B0, L0, P angle, and solar radius for the solstices and equinoxes of 2024:
    ; Mar 20, Jun 20, Sep 22, Dec 21
    ; TODO: Use PB0R or GET_SUN for each date
    ; TODO: Print a table of results

    ; === Exercise 4: SSW map operations ===
    ; Create a synthetic SSW map structure and use PLOT_MAP
    ; TODO: Create a map using MAKE_MAP or manually set structure fields
    ; Hint: map = {data: DIST(256), xc: 0., yc: 0., dx: 0.6, dy: 0.6, ...}

    ; === Exercise 5: VSO search ===
    ; Search for AIA 304 data from 2024-01-01 00:00 to 00:10
    ; Print the number of results and file sizes
    ; TODO: Use VSO_SEARCH (do NOT download)

END
