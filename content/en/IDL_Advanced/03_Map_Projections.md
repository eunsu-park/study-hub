# 03. Map Projections

**Previous**: [Advanced Plotting](./02_Advanced_Plotting.md) | **Next**: [Object-Oriented IDL](./04_Object_Oriented_IDL.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up map projections using MAP_SET with various projection types
2. Overlay continental boundaries and coordinate grids
3. Work with heliographic and Carrington coordinate systems
4. Understand the World Coordinate System (WCS) for solar data
5. Project solar data onto maps with coordinate transforms

---

## 1. MAP_SET — Setting Up a Projection

`MAP_SET` establishes a map projection for subsequent plotting.

```idl
; Basic setup: Orthographic projection centered on lat=0, lon=0
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC, $
    TITLE='Orthographic Projection'

; Add continents and grid
MAP_CONTINENTS, /FILL, COLOR=200
MAP_GRID, /LABEL, LATDEL=30, LONDEL=30
```

### MAP_SET Syntax

```idl
MAP_SET, lat_center, lon_center, rotation, $
    /PROJECTION_NAME, $
    LIMIT=[lat_min, lon_min, lat_max, lon_max], $
    /ISOTROPIC, $
    TITLE='...'
```

### Common Projections

```idl
; Orthographic (globe view)
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC
MAP_CONTINENTS & MAP_GRID

; Mollweide (equal-area, full sky)
MAP_SET, 0, 0, 0, /MOLLWEIDE, /ISOTROPIC
MAP_CONTINENTS & MAP_GRID

; Mercator (cylindrical)
MAP_SET, 0, 0, 0, /MERCATOR
MAP_CONTINENTS & MAP_GRID

; Aitoff (equal-area, used in astronomy)
MAP_SET, 0, 0, 0, /AITOFF, /ISOTROPIC
MAP_CONTINENTS & MAP_GRID

; Stereographic (conformal, used for polar regions)
MAP_SET, 90, 0, 0, /STEREOGRAPHIC, /ISOTROPIC, $
    LIMIT=[60, -180, 90, 180]
MAP_CONTINENTS & MAP_GRID

; Gnomonic (great circles appear straight)
MAP_SET, 0, 0, 0, /GNOMONIC, /ISOTROPIC, $
    LIMIT=[-45, -45, 45, 45]
MAP_CONTINENTS & MAP_GRID
```

---

## 2. MAP_CONTINENTS and MAP_GRID

### Continental Boundaries

```idl
; Outline continents
MAP_SET, 0, 0, 0, /MOLLWEIDE, /ISOTROPIC
MAP_CONTINENTS, COLOR=100

; Fill continents
MAP_CONTINENTS, /FILL, COLOR=200

; Countries and rivers
MAP_CONTINENTS, /COUNTRIES, COLOR=150
MAP_CONTINENTS, /RIVERS, COLOR=100

; Coastline only (faster)
MAP_CONTINENTS, /COASTS, COLOR=80

; With custom data file
; MAP_CONTINENTS, FILENAME='custom_coast.dat'
```

### Coordinate Grid

```idl
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC

; Basic grid
MAP_GRID

; Customized grid
MAP_GRID, LATDEL=15, LONDEL=15, $   ; Grid spacing in degrees
    /LABEL, $                         ; Label grid lines
    CHARSIZE=0.8, $
    LONLAB=-85, LATLAB=5, $          ; Label positions
    COLOR=150, $
    LINESTYLE=1                       ; Dotted lines

; Specific lat/lon lines
MAP_GRID, LATS=[-23.5, 0, 23.5, 66.5], $  ; Tropics, Equator, Arctic
    LONS=FINDGEN(12)*30
```

---

## 3. Solar Coordinate Systems

Solar physics uses specialized coordinate systems. SolarSoft provides routines for coordinate transforms.

### Heliographic Coordinates

```idl
; Heliographic Stonyhurst (HGS):
; - Longitude: 0 at central meridian, -180 to +180 (or 0 to 360)
; - Latitude: -90 to +90, 0 at solar equator
; - Fixed to the Sun's rotation axis

; Heliographic Carrington (HGC):
; - Same latitude as Stonyhurst
; - Longitude rotates with the Sun (Carrington rotation period = 27.2753 days)
; - Carrington longitude 0 defined by Carrington's reference meridian

; Convert between Stonyhurst and Carrington
; In SSW:
; carr_lon = stonyhurst_lon + (L0)
; where L0 is the Carrington longitude of the central meridian
; L0 changes continuously as the Sun rotates
```

### Heliocentric Coordinates

```idl
; Helioprojective Cartesian (HPC):
; - Theta_x (arcsec): angular displacement from Sun center in solar-X
; - Theta_y (arcsec): angular displacement in solar-Y
; - This is what you see in AIA/HMI images (pixel coordinates map to arcsec)

; Heliocentric Cartesian (HCC):
; - X, Y, Z in solar radii or km
; - Origin at Sun center

; Conversion between HPC and HGS (SSW routine)
; wcs_convert_from_coord, wcs, 'hpc', theta_x, theta_y, $
;     'hgs', hg_lon, hg_lat
```

### SolarSoft Coordinate Utilities

```idl
; Get solar B0 angle (tilt of solar axis toward Earth)
; and L0 (Carrington longitude of central meridian)
; and P angle (position angle of solar north pole)
sun_data = PB0R('2024-01-15T12:00:00')
; sun_data = [P_angle, B0_angle, R_sun_arcmin]
PRINT, 'P angle:  ', sun_data[0]
PRINT, 'B0 angle: ', sun_data[1]
PRINT, 'R_sun:    ', sun_data[2], ' arcmin'

; Carrington rotation number from date
carr = TIM2CARR('2024-01-15T12:00:00')
PRINT, 'Carrington rotation: ', carr

; Convert between coordinate systems
; arcmin2hel — helioprojective to heliographic
; hel2arcmin — heliographic to helioprojective
arcmin_x = 5.0  ; arcminutes from disk center
arcmin_y = 3.0
hel = ARCMIN2HEL(arcmin_x, arcmin_y, DATE='2024-01-15')
PRINT, 'Heliographic lat: ', hel[0]
PRINT, 'Heliographic lon: ', hel[1]
```

---

## 4. World Coordinate System (WCS)

WCS provides a standardized way to describe the relationship between pixel coordinates and physical coordinates in FITS files.

### WCS in Solar FITS Files

```idl
; Read a FITS file and extract WCS information
; (Using SSW routines)
fits_file = 'aia_171_image.fits'
data = READFITS(fits_file, header)

; Parse WCS from FITS header
wcs = FITSHEAD2WCS(header)

; Key WCS parameters for solar images:
; CRPIX1, CRPIX2 — reference pixel
; CRVAL1, CRVAL2 — coordinate at reference pixel (arcsec)
; CDELT1, CDELT2 — pixel scale (arcsec/pixel)
; CTYPE1, CTYPE2 — coordinate type ('HPLN-TAN', 'HPLT-TAN')
; NAXIS1, NAXIS2 — image size

PRINT, 'Reference pixel: ', wcs.crpix
PRINT, 'Reference coord: ', wcs.crval, ' arcsec'
PRINT, 'Pixel scale:     ', wcs.cdelt, ' arcsec/pixel'
```

### Pixel-to-World Conversion

```idl
; Convert pixel coordinates to world coordinates (arcsec)
pixel_x = 2048.0
pixel_y = 2048.0

; Method 1: Manual calculation
arcsec_x = (pixel_x - wcs.crpix[0]) * wcs.cdelt[0] + wcs.crval[0]
arcsec_y = (pixel_y - wcs.crpix[1]) * wcs.cdelt[1] + wcs.crval[1]

; Method 2: SSW WCS routines
wcs_coord = WCS_GET_COORD(wcs, [pixel_x, pixel_y])
PRINT, 'World coord: ', wcs_coord, ' arcsec'

; Convert world coordinates back to pixels
pixel = WCS_GET_PIXEL(wcs, [arcsec_x, arcsec_y])
```

### Sub-Region Extraction with WCS

```idl
; Extract a sub-region using arcsec coordinates
; Define region of interest in arcsec
x_center = -200.0  ; arcsec from Sun center
y_center = 300.0
half_fov = 100.0    ; arcsec

; Convert corners to pixels
ll = WCS_GET_PIXEL(wcs, [x_center - half_fov, y_center - half_fov])
ur = WCS_GET_PIXEL(wcs, [x_center + half_fov, y_center + half_fov])

; Extract sub-image
x0 = ROUND(ll[0]) > 0
y0 = ROUND(ll[1]) > 0
x1 = ROUND(ur[0]) < (wcs.naxis[0]-1)
y1 = ROUND(ur[1]) < (wcs.naxis[1]-1)
subimg = data[x0:x1, y0:y1]
```

---

## 5. Plotting Solar Data on Maps

### Heliographic Map of Solar Data

```idl
; Create a Carrington map from a synoptic chart
; Typical synoptic maps: 3600 x 1800 (0.1 deg resolution)
nx_carr = 3600 & ny_carr = 1800
carr_lon = FINDGEN(nx_carr) * 360.0 / nx_carr
carr_lat = FINDGEN(ny_carr) * 180.0 / ny_carr - 90.0

; Simulated magnetic field synoptic chart
bfield = RANDOMN(seed, nx_carr, ny_carr) * 10.0  ; Gauss

; Plot on Mollweide projection
MAP_SET, 0, 180, 0, /MOLLWEIDE, /ISOTROPIC, $
    TITLE='Carrington Synoptic Map'

; Remap data to projection
result = MAP_IMAGE(bfield, startx, starty, $
    LATMIN=-90, LATMAX=90, LONMIN=0, LONMAX=360, $
    /BILINEAR)

LOADCT, 0  ; Grayscale
TV, BYTSCL(result, MIN=-20, MAX=20), startx, starty
MAP_GRID, /LABEL, LATDEL=30, LONDEL=30, COLOR=200
```

### Overlaying Contours on Solar Disk

```idl
; Create an orthographic view of the solar disk
; Useful for overlaying magnetograms on EUV images

MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC, $
    LIMIT=[-90, -90, 90, 90], $
    TITLE='Solar Disk Overlay'

; Plot EUV image as background
; MAP_IMAGE projects data onto the map projection
euv_projected = MAP_IMAGE(euv_data, sx, sy, $
    LATMIN=-90, LATMAX=90, LONMIN=-90, LONMAX=90)
LOADCT, 3
TV, BYTSCL(euv_projected), sx, sy

; Overlay magnetic field contours
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC, /NOERASE, $
    LIMIT=[-90, -90, 90, 90]
CONTOUR, bfield, blon, blat, /OVERPLOT, $
    LEVELS=[-100, -50, 50, 100], $
    C_COLORS=[50, 100, 200, 250], $
    C_THICK=2
```

---

## 6. Coordinate Transform Examples

### Disk Position to Heliographic

```idl
; Given a position on the solar disk in arcsec,
; convert to heliographic coordinates

; Solar parameters
rsun_arcsec = 960.0  ; Solar radius in arcsec
b0 = 0.0             ; Solar B0 angle (degrees)
l0 = 0.0             ; Carrington longitude of central meridian

; Disk position
x_arcsec = 200.0     ; East-West (positive = West)
y_arcsec = 300.0     ; North-South (positive = North)

; Normalize to solar radii
rho = SQRT(x_arcsec^2 + y_arcsec^2) / rsun_arcsec

IF rho LT 1.0 THEN BEGIN
    ; On-disk: compute heliographic coordinates
    theta = ASIN(rho)  ; angular distance from disk center

    ; Position angle
    phi = ATAN(x_arcsec, y_arcsec)

    ; Heliographic latitude
    lat = ASIN(SIN(b0*!DTOR)*COS(theta) + $
               COS(b0*!DTOR)*SIN(theta)*COS(phi))
    lat = lat * !RADEG

    ; Heliographic longitude (relative to central meridian)
    lon = ASIN(SIN(theta)*SIN(phi) / COS(lat*!DTOR))
    lon = lon * !RADEG

    PRINT, 'Heliographic lat: ', lat, ' degrees'
    PRINT, 'Heliographic lon: ', lon, ' degrees from CM'

    ; Carrington longitude
    carr_lon = lon + l0
    PRINT, 'Carrington lon:   ', carr_lon, ' degrees'
ENDIF ELSE BEGIN
    PRINT, 'Position is off-disk'
ENDELSE
```

### Differential Rotation Correction

```idl
; The Sun rotates differentially: faster at equator, slower at poles
; Synodic rotation rate (Snodgrass & Ulrich, 1990):
;   omega(lat) = A + B*sin^2(lat) + C*sin^4(lat)  [deg/day]
;   A = 14.713, B = -2.396, C = -1.787

FUNCTION diff_rot_rate, lat_deg
    ; Returns rotation rate in deg/day
    lat_rad = lat_deg * !DTOR
    sin2 = SIN(lat_rad)^2
    sin4 = sin2^2
    RETURN, 14.713 - 2.396*sin2 - 1.787*sin4
END

; Correct a Carrington longitude for differential rotation
; Given: position at time t0, compute position at time t1
lat = 30.0   ; degrees
lon0 = 45.0  ; Carrington longitude at t0
dt = 5.0     ; days between t0 and t1

omega = diff_rot_rate(lat)
lon1 = lon0 + omega * dt  ; New Carrington longitude
PRINT, 'Rotation rate at lat=', lat, ': ', omega, ' deg/day'
PRINT, 'New longitude after ', dt, ' days: ', lon1
```

---

## 7. Full-Disk Solar Image with WCS Grid

```idl
; Complete example: display a solar image with coordinate grid

; Read AIA 171 image (4096x4096)
data = READFITS('aia_171.fits', header)
wcs = FITSHEAD2WCS(header)

; Rebin for display
img = REBIN(data, 1024, 1024)

; Display
LOADCT, 3
WINDOW, 0, XSIZE=1024, YSIZE=1024
TV, BYTSCL(img, MIN=0, MAX=5000)

; Compute arcsec ranges
nx = wcs.naxis[0] & ny = wcs.naxis[1]
x_arcsec = (FINDGEN(nx) - wcs.crpix[0]) * wcs.cdelt[0] + wcs.crval[0]
y_arcsec = (FINDGEN(ny) - wcs.crpix[1]) * wcs.cdelt[1] + wcs.crval[1]

; Scale for rebinned image
x_disp = CONGRID(x_arcsec, 1024)
y_disp = CONGRID(y_arcsec, 1024)

; Overlay coordinate axes
PLOT, x_disp, y_disp, /NODATA, /NOERASE, $
    POSITION=[0, 0, 1, 1], $
    XRANGE=[MIN(x_disp), MAX(x_disp)], $
    YRANGE=[MIN(y_disp), MAX(y_disp)], $
    XSTYLE=1, YSTYLE=1, $
    XTITLE='Solar-X (arcsec)', YTITLE='Solar-Y (arcsec)', $
    COLOR=255

; Draw solar limb circle
theta = FINDGEN(361) * !DTOR
rsun = 960.0  ; arcsec
PLOTS, rsun*COS(theta), rsun*SIN(theta), COLOR=255, THICK=2

; Draw heliographic grid on the disk
FOR lat = -60, 60, 30 DO BEGIN
    phi = FINDGEN(361) * !DTOR
    ; Simple projection (ignoring B0 for illustration)
    x_grid = rsun * COS(lat*!DTOR) * SIN(phi)
    y_grid = rsun * SIN(lat*!DTOR) * REPLICATE(1.0, 361)
    ; Only plot visible portion
    visible = WHERE(COS(lat*!DTOR)*COS(phi) GT 0, nvis)
    IF nvis GT 0 THEN $
        PLOTS, x_grid[visible], y_grid[visible], COLOR=200, LINESTYLE=1
ENDFOR
```

---

## Summary

| Topic | Key Routines | Purpose |
|-------|-------------|---------|
| Map setup | `MAP_SET` | Establish projection |
| Boundaries | `MAP_CONTINENTS`, `MAP_GRID` | Geographic overlays |
| Image projection | `MAP_IMAGE` | Project data onto map |
| Solar coordinates | `PB0R`, `ARCMIN2HEL`, `TIM2CARR` | Solar geometry |
| WCS | `FITSHEAD2WCS`, `WCS_GET_COORD` | FITS coordinate system |
| Heliographic | `ARCMIN2HEL`, `HEL2ARCMIN` | Disk-to-heliographic |
| Differential rotation | Custom functions | Rotation correction |

---

**Previous**: [Advanced Plotting](./02_Advanced_Plotting.md) | **Next**: [Object-Oriented IDL](./04_Object_Oriented_IDL.md)
