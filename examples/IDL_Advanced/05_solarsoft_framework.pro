;+
; 05_solarsoft_framework.pro — Lesson 05: SolarSoft Framework
;
; Demonstrates SSW time utilities and data structures.
; Requires: SolarSoft (sswidl)
;-

PRO solarsoft_demo
    ; Time conversions
    t_ccsds = ANYTIM('2024-01-15 12:00:00', /CCSDS)
    t_vms   = ANYTIM('2024-01-15 12:00:00', /VMS)
    PRINT, 'CCSDS: ', t_ccsds
    PRINT, 'VMS:   ', t_vms

    ; Time arithmetic
    t0 = ANYTIM('2024-01-15T12:00:00')
    t1 = t0 + 3600.0
    PRINT, '1 hour later: ', ANYTIM(t1, /CCSDS)

    ; Carrington rotation
    carr = TIM2CARR('2024-01-15T12:00:00')
    PRINT, 'Carrington rotation: ', LONG(carr)

    ; Solar ephemeris
    pbr = PB0R('2024-01-15T12:00:00')
    PRINT, 'P angle:  ', pbr[0], ' deg'
    PRINT, 'B0 angle: ', pbr[1], ' deg'
    PRINT, 'R_sun:    ', pbr[2], ' arcmin'

    ; Installed instruments
    PRINT, 'SSW instruments: ', SSW_INSTR()
END

solarsoft_demo
END
