# 05. SolarSoft Framework

**Previous**: [Object-Oriented IDL](./04_Object_Oriented_IDL.md) | **Next**: [SDO/AIA Analysis](./06_SDO_AIA_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and configure SolarSoft (SSW) on your system
2. Navigate the SSW directory structure and instrument trees
3. Use SSW environment variables and the sswidl startup
4. Work with SSW time-handling utilities (ANYTIM, UTC2TAI, etc.)
5. Query data using SSW_JSOC and the Virtual Solar Observatory (VSO)

---

## 1. What is SolarSoft?

SolarSoft (SSW) is a comprehensive IDL library system for solar physics. It provides:

- **Instrument calibration pipelines**: SDO/AIA, SDO/HMI, SOHO/EIT, Hinode, STEREO, etc.
- **Coordinate transforms**: heliographic, heliocentric, Carrington
- **Time utilities**: flexible time parsing, TAI/UTC conversion
- **Data access**: JSOC, VSO, SDAC
- **Analysis tools**: DEM, spectral fitting, image processing
- **Standard data formats**: FITS I/O with solar-specific keywords

### SSW Architecture

```
$SSW/                           # Root directory
├── gen/                        # General-purpose libraries
│   ├── idl/                    # Core IDL routines
│   │   ├── time/               # Time conversion (anytim, utc2tai, etc.)
│   │   ├── fits/               # FITS I/O (mreadfits, mwritefits)
│   │   ├── maps/               # Solar map structure routines
│   │   ├── image/              # Image processing
│   │   └── util/               # General utilities
│   └── setup/                  # Environment setup scripts
│       └── setup.ssw           # Main SSW setup script
├── sdo/                        # SDO mission
│   ├── aia/                    # AIA instrument
│   │   ├── idl/                # AIA-specific routines
│   │   └── response/           # AIA response functions
│   └── hmi/                    # HMI instrument
├── soho/                       # SOHO mission
│   ├── eit/                    # EIT instrument
│   └── mdi/                    # MDI instrument
├── hinode/                     # Hinode mission
│   ├── eis/                    # EIS spectrometer
│   └── xrt/                    # X-Ray Telescope
├── stereo/                     # STEREO mission
│   └── secchi/                 # SECCHI instrument suite
├── goes/                       # GOES satellites
├── hessi/                      # RHESSI mission
└── packages/                   # Additional packages
    ├── chianti/                # CHIANTI atomic database
    └── forward/                # Forward modeling
```

---

## 2. Installation and Setup

### Installing SolarSoft

```bash
# Set SSW root directory
export SSW=/usr/local/ssw

# Download the installer
mkdir -p $SSW
cd $SSW
wget https://www.lmsal.com/solarsoft/ssw_install.tar
tar xf ssw_install.tar

# Run the IDL installer script
# (Requires IDL to be installed first)
idl -e "ssw_install, /sdo, /aia, /hmi, /goes, /hessi, /gen"
```

### Alternative: ssw_install from IDL

```idl
; From within IDL
ssw_install, /sdo, /aia, /hmi
ssw_install, /goes
ssw_install, /hessi
ssw_install, /soho, /eit
```

### Environment Variables

```bash
# Required environment variables (add to ~/.bashrc or ~/.cshrc)
export SSW=/usr/local/ssw
export SSW_INSTR="aia hmi goes hessi eit"

# Optional: specify IDL directory
export IDL_DIR=/usr/local/harris/idl

# Source SSW setup
source $SSW/gen/setup/setup.ssw
```

### Starting SolarSoft IDL

```bash
# The sswidl command starts IDL with SSW paths configured
sswidl

# Or manually configure paths in IDL:
# @$SSW/gen/setup/ssw_idl
```

### Verifying Installation

```idl
; Check SSW is loaded
PRINT, GETENV('SSW')           ; Should print SSW root path
PRINT, SSW_INSTR()              ; List loaded instruments

; Check available instrument routines
ssw_path, /aia
PRINT, !PATH                    ; Should include AIA directories

; Test a basic SSW routine
PRINT, ANYTIM('2024-01-15T12:00:00', /CCSDS)
```

---

## 3. SSW Time Utilities

SSW's time-handling routines are among its most valuable tools. They accept almost any time format and convert between them.

### ANYTIM — Universal Time Parser

```idl
; ANYTIM converts virtually any time string to a standard format
; Default output: double-precision seconds since 1-Jan-1979

; Various input formats — all produce the same result:
t1 = ANYTIM('2024-01-15 12:00:00')
t2 = ANYTIM('15-Jan-2024 12:00:00')
t3 = ANYTIM('2024/01/15 12:00')
t4 = ANYTIM('20240115_120000')

; Output format options:
t_ccsds  = ANYTIM('2024-01-15', /CCSDS)    ; '2024-01-15T00:00:00.000'
t_vms    = ANYTIM('2024-01-15', /VMS)       ; '15-Jan-2024 00:00:00.000'
t_ecs    = ANYTIM('2024-01-15', /ECS)       ; '2024/01/15 00:00:00'
t_tai    = ANYTIM('2024-01-15', /TAI)       ; TAI seconds
t_mjd    = ANYTIM('2024-01-15', /MJD)       ; Modified Julian Date
t_int    = ANYTIM('2024-01-15', /EXTERNAL)  ; [hh,mm,ss,msec,dd,mm,yy]

PRINT, t_ccsds
PRINT, t_vms
```

### Time Arithmetic

```idl
; Add/subtract time in seconds
t0 = ANYTIM('2024-01-15T12:00:00')
t1 = t0 + 3600.0  ; Add 1 hour
t2 = t0 - 86400.0  ; Subtract 1 day

PRINT, ANYTIM(t1, /CCSDS)  ; '2024-01-15T13:00:00.000'
PRINT, ANYTIM(t2, /CCSDS)  ; '2024-01-14T12:00:00.000'

; Time difference
dt = ANYTIM('2024-01-16') - ANYTIM('2024-01-15')
PRINT, dt, ' seconds'     ; 86400.000 seconds
PRINT, dt/3600.0, ' hours' ; 24.000 hours
```

### SSW_FILE2TIME — Extract Time from Filename

```idl
; Many solar data files encode the time in the filename
; SSW_FILE2TIME extracts it

file = 'aia.lev1.171A_2024-01-15T12_00_00.00Z.image_lev1.fits'
time = SSW_FILE2TIME(file)
PRINT, ANYTIM(time, /CCSDS)
```

### TIM2CARR — Time to Carrington Rotation

```idl
; Convert time to Carrington rotation number and longitude
carr = TIM2CARR('2024-01-15T12:00:00')
PRINT, 'Carrington rotation: ', LONG(carr)

; Get both rotation number and longitude
carr = TIM2CARR('2024-01-15T12:00:00', /DC)
; DC = decimal Carrington, includes fractional rotation

; Reverse: Carrington rotation to time
time = CARR2TIM(2277.0)
PRINT, ANYTIM(time, /CCSDS)
```

### TIMEGRID — Generate Time Arrays

```idl
; Generate evenly-spaced time array
t_start = '2024-01-15T00:00:00'
t_end   = '2024-01-15T23:59:59'
cadence = 720.0  ; seconds (12 minutes)

times = TIMEGRID(t_start, t_end, cadence, /SECONDS)
PRINT, N_ELEMENTS(times), ' time steps'
PRINT, ANYTIM(times[0], /CCSDS)
PRINT, ANYTIM(times[-1], /CCSDS)
```

---

## 4. SSW Data Structures

### The Index/Data Paradigm

SSW follows a consistent pattern: `read_xxx` returns both an **index** (FITS header as structure) and **data** (image array).

```idl
; Generic pattern:
; read_instrument, files, index, data
;
; index: structure array with header keywords
;   index.date_obs  — observation time
;   index.naxis1    — image width
;   index.naxis2    — image height
;   index.crpix1    — reference pixel x
;   index.crval1    — reference value x
;   index.cdelt1    — pixel scale x
;   index.exptime   — exposure time
;   index.wavelnth  — wavelength (for multi-channel instruments)

; Example with MREADFITS (generic SSW FITS reader)
MREADFITS, 'solar_image.fits', index, data
HELP, index, /STRUCTURE
PRINT, 'Date: ', index.date_obs
PRINT, 'Size: ', index.naxis1, ' x ', index.naxis2
```

### SSW Map Structure

```idl
; SSW maps combine image data with coordinate metadata
; Common pattern: index2map converts index/data to map structure

; Create a map from index and data
index2map, index, data, map

; Map structure fields:
;   map.data   — 2D image array
;   map.xc     — x center (arcsec)
;   map.yc     — y center (arcsec)
;   map.dx     — x pixel size (arcsec)
;   map.dy     — y pixel size (arcsec)
;   map.time   — observation time
;   map.id     — instrument/channel ID

; Plot a map
PLOT_MAP, map, /LIMB, TITLE=map.id + ' ' + map.time

; Map operations
map_sub = SUB_MAP(map, XRANGE=[-500, 500], YRANGE=[-500, 500])
PLOT_MAP, map_sub, /LIMB
```

---

## 5. Data Access: VSO and JSOC

### Virtual Solar Observatory (VSO)

```idl
; Search for data
vso_search, '2024-01-15T00:00:00', '2024-01-15T01:00:00', $
    INSTRUMENT='aia', WAVE='171', results, /FLAT

; Display results
PRINT, results.fileid
PRINT, results.size

; Download data
vso_get, results, OUT_DIR='./data/', FILENAMES=downloaded_files
PRINT, downloaded_files
```

### JSOC (Joint Science Operations Center) for SDO

```idl
; SSW_JSOC provides access to SDO data through JSOC
; (Requires JSOC registration: http://jsoc.stanford.edu/)

; Search for AIA data
ssw_jsoc_main, date='2024-01-15T12:00:00', $
    ins='aia', wave='171', cadence='60s', $
    timespan='1h', /search

; Download via URL
; JSOC also supports direct wget/curl:
; http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_fetch?...
```

### Using cutout service

```idl
; JSOC cutout service for sub-field data (saves bandwidth)
; Specify center (arcsec) and field of view

; SSW routine for cutout requests
ssw_cutout_service, '2024-01-15T12:00:00', $
    INSTRUMENT='aia', WAVELENGTH=171, $
    XCEN=-200, YCEN=300, $
    FOVX=500, FOVY=500, $
    CADENCE=60, DURATION=3600, $
    OUT_DIR='./cutouts/'
```

---

## 6. SSW Utility Routines

### File Utilities

```idl
; FILE_SEARCH with pattern matching
files = FILE_SEARCH('/data/aia/171/*.fits', COUNT=nfiles)
PRINT, nfiles, ' files found'

; Sort files by time
times = SSW_FILE2TIME(files)
sorted_idx = SORT(times)
files = files[sorted_idx]

; SSW_CONCAT_STRUCT: merge structure arrays
; Useful for combining headers from multiple reads
MREADFITS, files[0:4], idx1, dat1
MREADFITS, files[5:9], idx2, dat2
all_idx = SSW_CONCAT_STRUCT(idx1, idx2)
```

### String Utilities

```idl
; SSW string routines
str = SSW_STRSPLIT('2024-01-15T12:00:00', 'T', /HEAD, TAIL=tail)
PRINT, str    ; '2024-01-15'
PRINT, tail   ; '12:00:00'

; STRJOIN — join array of strings
parts = ['aia', '171', '2024', '01', '15']
filename = STRJOIN(parts, '_') + '.fits'
PRINT, filename  ; 'aia_171_2024_01_15.fits'
```

### Solar Ephemeris

```idl
; GET_SUN — comprehensive solar ephemeris
sun = GET_SUN('2024-01-15T12:00:00')
PRINT, 'B0 angle:     ', sun.b0, ' deg'
PRINT, 'L0 (Carr lon):', sun.l0, ' deg'
PRINT, 'P angle:      ', sun.p0, ' deg'
PRINT, 'R_sun:        ', sun.sd, ' arcsec'
PRINT, 'Earth-Sun:    ', sun.dist, ' AU'

; PB0R — quick B0, P, R_sun
pbr = PB0R('2024-01-15T12:00:00')
PRINT, 'P: ', pbr[0], ' B0: ', pbr[1], ' R_sun: ', pbr[2], ' arcmin'
```

---

## 7. Working with SSW Instrument Trees

### Listing Available Instruments

```idl
; Check which instruments are installed
ssw_packages = SSW_INSTR()
PRINT, ssw_packages

; Add an instrument at runtime
ssw_path, /aia
ssw_path, /hmi
ssw_path, /goes

; Check if a specific instrument is available
IF SSW_INSTR(/AIA) THEN PRINT, 'AIA is available'
```

### Instrument-Specific Setup

```idl
; AIA
aia_lct, wave=171, /LOAD    ; Load AIA 171 color table
aia_lct, wave=304, /LOAD    ; Load AIA 304 color table

; Each instrument provides:
; 1. Reader routine (read_sdo, read_eit, etc.)
; 2. Calibration routine (aia_prep, eit_prep, etc.)
; 3. Response functions
; 4. Instrument-specific utilities
```

---

## 8. Common SSW Patterns

### Reading and Calibrating a Batch of Files

```idl
; Standard SSW workflow:
; 1. Find files
; 2. Read raw data
; 3. Calibrate
; 4. Analyze

; Step 1: Find files
files = FILE_SEARCH('/data/sdo/aia/171/2024/01/15/*.fits', COUNT=nf)
PRINT, nf, ' files found'

; Step 2: Read
MREADFITS, files, index, data

; Step 3: Calibrate (instrument-specific)
aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

; Step 4: Create maps for analysis
index2map, oindex, odata, maps

; Step 5: Plot
FOR i = 0, N_ELEMENTS(maps)-1 DO BEGIN
    PLOT_MAP, maps[i], /LIMB
    WAIT, 0.5
ENDFOR
```

### Error Handling in SSW

```idl
; SSW routines often use STATUS and ERRMSG keywords
vso_search, t_start, t_end, INSTRUMENT='aia', $
    results, STATUS=status, ERRMSG=errmsg

IF status NE 0 THEN BEGIN
    PRINT, 'VSO search failed: ', errmsg
    RETURN
ENDIF

; Check for valid data
IF N_ELEMENTS(results) EQ 0 THEN BEGIN
    PRINT, 'No data found'
    RETURN
ENDIF
```

---

## Summary

| Topic | Key Routines | Purpose |
|-------|-------------|---------|
| Installation | `ssw_install`, `setup.ssw` | Set up SSW |
| Environment | `$SSW`, `$SSW_INSTR`, `sswidl` | Configure paths |
| Time | `ANYTIM`, `UTC2TAI`, `TIM2CARR` | Time conversion |
| Data I/O | `MREADFITS`, `MWRITEFITS` | FITS file access |
| Maps | `INDEX2MAP`, `PLOT_MAP`, `SUB_MAP` | Solar map structures |
| Data access | `VSO_SEARCH`, `VSO_GET`, `SSW_JSOC` | Download data |
| Ephemeris | `GET_SUN`, `PB0R` | Solar geometry |
| Instruments | `ssw_path`, `SSW_INSTR` | Instrument trees |

---

**Previous**: [Object-Oriented IDL](./04_Object_Oriented_IDL.md) | **Next**: [SDO/AIA Analysis](./06_SDO_AIA_Analysis.md)
