;+
; 03_map_projections.pro — Lesson 03: Map Projections
;
; Demonstrates:
;   - MAP_SET with different projections
;   - MAP_CONTINENTS and MAP_GRID overlays
;   - Simple solar coordinate conversion
;-

PRO map_projections_demo
    WINDOW, 0, XSIZE=800, YSIZE=600
    !P.MULTI = [0, 2, 2]

    ; Orthographic
    MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC, TITLE='Orthographic'
    MAP_CONTINENTS, /FILL, COLOR=200
    MAP_GRID, /LABEL, LATDEL=30, LONDEL=30

    ; Mollweide
    MAP_SET, 0, 0, 0, /MOLLWEIDE, /ISOTROPIC, TITLE='Mollweide'
    MAP_CONTINENTS & MAP_GRID

    ; Mercator
    MAP_SET, 0, 0, 0, /MERCATOR, TITLE='Mercator'
    MAP_CONTINENTS & MAP_GRID

    ; Aitoff
    MAP_SET, 0, 0, 0, /AITOFF, /ISOTROPIC, TITLE='Aitoff'
    MAP_CONTINENTS & MAP_GRID

    !P.MULTI = 0

    ; Disk-to-heliographic conversion
    rsun = 960.0  ; arcsec
    x_arc = 200.0 & y_arc = 300.0
    rho = SQRT(x_arc^2 + y_arc^2) / rsun
    IF rho LT 1.0 THEN BEGIN
        theta = ASIN(rho)
        phi = ATAN(x_arc, y_arc)
        lat = ASIN(SIN(0)*COS(theta) + COS(0)*SIN(theta)*COS(phi)) * !RADEG
        lon = ASIN(SIN(theta)*SIN(phi) / COS(lat*!DTOR)) * !RADEG
        PRINT, 'Disk (', x_arc, ',', y_arc, ') arcsec -> Helio lat=', lat, ' lon=', lon
    ENDIF
END

map_projections_demo
END
