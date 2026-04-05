# FITS File Handling

**Previous**: [Image Display](./11_Image_Display.md) | **Next**: [Date and Time](./13_Date_and_Time.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the FITS file format and its role in astronomy
2. Read FITS files with READFITS and MRDFITS
3. Write FITS files with WRITEFITS and MWRFITS
4. Manipulate FITS headers with SXPAR and FXADDPAR
5. Work with multi-extension FITS (MEF) files
6. Handle different FITS data types (images, tables, binary tables)
7. Read SDO/AIA and HMI data as practical examples

---

FITS (Flexible Image Transport System) is the standard file format in astronomy and space science. It was developed in the 1970s and has been the primary data format for virtually all astronomical observations. Understanding FITS is essential for working with data from telescopes, satellites, and space missions.

## What is FITS?

### Format Structure

A FITS file consists of one or more Header-Data Units (HDUs):

```
┌─────────────────────────────────┐
│ Primary HDU                     │
│   ┌─────────────────────────┐   │
│   │ Header (ASCII keywords) │   │
│   │ 2880-byte blocks        │   │
│   ├─────────────────────────┤   │
│   │ Data (optional)         │   │
│   │ Image or table          │   │
│   └─────────────────────────┘   │
├─────────────────────────────────┤
│ Extension HDU 1 (optional)      │
│   ┌─────────────────────────┐   │
│   │ Header                  │   │
│   ├─────────────────────────┤   │
│   │ Data                    │   │
│   └─────────────────────────┘   │
├─────────────────────────────────┤
│ Extension HDU 2 (optional)      │
│   ...                           │
└─────────────────────────────────┘
```

### Header Keywords

FITS headers consist of 80-character keyword records:

```
SIMPLE  =                    T / file does conform to FITS standard
BITPIX  =                  -32 / number of bits per data pixel
NAXIS   =                    2 / number of data axes
NAXIS1  =                 4096 / length of data axis 1
NAXIS2  =                 4096 / length of data axis 2
TELESCOP= 'SDO/AIA '          / telescope name
WAVELNTH=                  171 / wavelength in Angstroms
DATE-OBS= '2024-07-15T12:00:00.000' / observation date and time
END
```

### BITPIX Values

| BITPIX | Data Type | IDL Type |
|--------|-----------|----------|
| 8 | Unsigned byte | BYTE |
| 16 | 16-bit signed integer | INT |
| 32 | 32-bit signed integer | LONG |
| -32 | 32-bit floating point | FLOAT |
| -64 | 64-bit floating point | DOUBLE |

---

## Reading FITS Files

### READFITS

`READFITS` is the simplest way to read a FITS file (from the IDL Astronomy User's Library):

```idl
; Read data and header
data = READFITS('image.fits', header)

; Check what was read
HELP, data
; DATA            FLOAT     = Array[4096, 4096]

PRINT, N_ELEMENTS(header), ' header lines'

; Read only the header (no data)
header = HEADFITS('image.fits')
```

### Examining the Header

```idl
; Read the header
data = READFITS('aia_171.fits', header)

; Print the entire header
FOR i = 0, N_ELEMENTS(header) - 1 DO PRINT, header[i]

; Extract specific keywords with SXPAR
naxis1 = SXPAR(header, 'NAXIS1')
naxis2 = SXPAR(header, 'NAXIS2')
bitpix = SXPAR(header, 'BITPIX')
telescop = SXPAR(header, 'TELESCOP')
wavelnth = SXPAR(header, 'WAVELNTH')
date_obs = SXPAR(header, 'DATE-OBS')

PRINT, 'Dimensions:', naxis1, 'x', naxis2
PRINT, 'BITPIX:', bitpix
PRINT, 'Telescope:', telescop
PRINT, 'Wavelength:', wavelnth, ' A'
PRINT, 'Date:', date_obs

; SXPAR handles comments and missing keywords
exptime = SXPAR(header, 'EXPTIME', COMMENT=comment)
PRINT, 'Exposure:', exptime, ' ; ', comment

; Check if keyword exists
value = SXPAR(header, 'MISSING_KEY', COUNT=count)
IF count EQ 0 THEN PRINT, 'Keyword not found'
```

---

## Writing FITS Files

### WRITEFITS

```idl
; Create a simple FITS file
data = DIST(256)

; Create a minimal header
MKHDR, header, data    ; Auto-generate header from data

; Add custom keywords
SXADDPAR, header, 'TELESCOP', 'Simulated', 'Telescope name'
SXADDPAR, header, 'DATE-OBS', '2024-07-15T12:00:00', 'Observation date'
SXADDPAR, header, 'WAVELNTH', 171, 'Wavelength in Angstroms'
SXADDPAR, header, 'EXPTIME', 2.0, 'Exposure time in seconds'

; Write the file
WRITEFITS, 'output.fits', data, header
PRINT, 'File written: output.fits'
```

### FXADDPAR — More Flexible Header Editing

```idl
; FXADDPAR offers more control than SXADDPAR
data = FLTARR(512, 512)
MKHDR, header, data

; Add keywords with specific format and position
FXADDPAR, header, 'OBJECT', 'Sun', ' Target object'
FXADDPAR, header, 'CRPIX1', 256.5, ' Reference pixel X', FORMAT='(F8.1)'
FXADDPAR, header, 'CRPIX2', 256.5, ' Reference pixel Y', FORMAT='(F8.1)'
FXADDPAR, header, 'CDELT1', 0.6, ' Pixel scale X (arcsec)', FORMAT='(F8.4)'
FXADDPAR, header, 'CDELT2', 0.6, ' Pixel scale Y (arcsec)', FORMAT='(F8.4)'
FXADDPAR, header, 'CRVAL1', 0.0, ' Reference coordinate X'
FXADDPAR, header, 'CRVAL2', 0.0, ' Reference coordinate Y'
FXADDPAR, header, 'CTYPE1', 'HPLN-TAN', ' Coordinate type X'
FXADDPAR, header, 'CTYPE2', 'HPLT-TAN', ' Coordinate type Y'

; Add COMMENT and HISTORY
SXADDPAR, header, 'COMMENT', 'This is a simulated solar image'
SXADDPAR, header, 'HISTORY', 'Created with IDL on ' + SYSTIME()

WRITEFITS, 'solar_sim.fits', data, header
```

---

## Multi-Extension FITS (MEF)

### MRDFITS — Read Multi-Extension FITS

```idl
; Read the primary HDU (extension 0)
primary = MRDFITS('multi.fits', 0, primary_header)

; Read extension 1
ext1 = MRDFITS('multi.fits', 1, ext1_header)

; Read extension 2
ext2 = MRDFITS('multi.fits', 2, ext2_header)

; MRDFITS can read binary tables as structures
table = MRDFITS('catalog.fits', 1, header)
; table is a structure array if the extension contains a binary table
HELP, table, /STRUCTURE
PRINT, table.ra          ; Array of RA values
PRINT, table.dec         ; Array of DEC values
```

### MWRFITS — Write Multi-Extension FITS

```idl
; Write primary HDU
primary_data = DIST(256)
MKHDR, header, primary_data
MWRFITS, primary_data, 'multi_output.fits', header, /CREATE

; Append extensions
image2 = RANDOMN(seed, 256, 256)
MWRFITS, image2, 'multi_output.fits'

; Write a binary table extension
n = 100
catalog = REPLICATE({ra: 0.0D, dec: 0.0D, flux: 0.0, name: ''}, n)
catalog.ra = RANDOMU(seed, n) * 360.0D
catalog.dec = (RANDOMU(seed, n) - 0.5D) * 180.0D
catalog.flux = RANDOMU(seed, n) * 1000.0
catalog.name = 'Star_' + STRTRIM(INDGEN(n) + 1, 2)

MWRFITS, catalog, 'catalog.fits', /CREATE
; This creates a binary table FITS file
```

### Querying MEF Structure

```idl
; Find how many extensions a file has
FITS_OPEN, 'multi.fits', fcb
PRINT, 'Number of extensions:', fcb.NEXTEND
FOR i = 0, fcb.NEXTEND DO BEGIN
  PRINT, FORMAT='("  Extension ", I0, ": ", A)', i, fcb.EXTNAME[i]
ENDFOR
FITS_CLOSE, fcb
```

---

## Working with AIA/HMI FITS Files

### SDO/AIA Example

```idl
; Read an AIA 171 Angstrom image
data = READFITS('aia_171_image.fits', header)

; Extract key metadata
wavelength = SXPAR(header, 'WAVELNTH')
date_obs = SXPAR(header, 'DATE-OBS')
exptime = SXPAR(header, 'EXPTIME')
crpix1 = SXPAR(header, 'CRPIX1')
crpix2 = SXPAR(header, 'CRPIX2')
cdelt1 = SXPAR(header, 'CDELT1')
r_sun = SXPAR(header, 'RSUN_OBS')  ; Solar radius in arcsec

PRINT, 'AIA ', wavelength, ' A'
PRINT, 'Date: ', date_obs
PRINT, 'Exposure: ', exptime, ' s'
PRINT, 'Plate scale: ', cdelt1, ' arcsec/pixel'
PRINT, 'Solar radius: ', r_sun, ' arcsec'

; Normalize by exposure time
data_norm = data / exptime

; Display with logarithmic scaling
display = BYTSCL(ALOG10(data_norm > 1.0), MIN=0, MAX=4)
LOADCT, 1, /SILENT    ; Blue-white for EUV
WINDOW, XSIZE=512, YSIZE=512
TV, CONGRID(display, 512, 512)
XYOUTS, 0.5, 0.96, 'AIA ' + STRTRIM(wavelength, 2) + ' A - ' + date_obs, $
  /NORMAL, ALIGNMENT=0.5, CHARSIZE=1.3
```

### SDO/HMI Example

```idl
; Read an HMI magnetogram
mag = READFITS('hmi_magnetogram.fits', header)

; Extract metadata
date_obs = SXPAR(header, 'DATE-OBS')
cdelt1 = SXPAR(header, 'CDELT1')

; Display magnetogram (positive = white, negative = black)
display = BYTSCL(mag, MIN=-500, MAX=500)
LOADCT, 0, /SILENT    ; Grayscale
WINDOW, XSIZE=512, YSIZE=512
TV, CONGRID(display, 512, 512)
XYOUTS, 0.5, 0.96, 'HMI Magnetogram - ' + date_obs, $
  /NORMAL, ALIGNMENT=0.5, CHARSIZE=1.3
```

---

## Header Manipulation

### Modifying Headers

```idl
; Read existing file
data = READFITS('original.fits', header)

; Modify keywords
SXADDPAR, header, 'EXPTIME', 4.0, 'Updated exposure time'

; Delete a keyword
SXDELPAR, header, 'OBSOLETE_KEY'

; Add history
SXADDPAR, header, 'HISTORY', 'Processed on ' + SYSTIME()
SXADDPAR, header, 'HISTORY', 'Dark subtracted and flat fielded'

; Write modified file
WRITEFITS, 'processed.fits', data, header
```

### Creating Headers from Scratch

```idl
; For a 2D float image
nx = 1024L
ny = 1024L
data = FLTARR(nx, ny)

; Create minimal header
MKHDR, header, data

; Add WCS (World Coordinate System) keywords
SXADDPAR, header, 'CRPIX1', nx/2.0 + 0.5, 'Reference pixel X'
SXADDPAR, header, 'CRPIX2', ny/2.0 + 0.5, 'Reference pixel Y'
SXADDPAR, header, 'CRVAL1', 0.0, 'Reference coordinate X (arcsec)'
SXADDPAR, header, 'CRVAL2', 0.0, 'Reference coordinate Y (arcsec)'
SXADDPAR, header, 'CDELT1', 1.0, 'Plate scale X (arcsec/pixel)'
SXADDPAR, header, 'CDELT2', 1.0, 'Plate scale Y (arcsec/pixel)'
SXADDPAR, header, 'CTYPE1', 'HPLN-TAN', 'Helioprojective longitude'
SXADDPAR, header, 'CTYPE2', 'HPLT-TAN', 'Helioprojective latitude'
SXADDPAR, header, 'CUNIT1', 'arcsec', 'Units for axis 1'
SXADDPAR, header, 'CUNIT2', 'arcsec', 'Units for axis 2'

WRITEFITS, 'wcs_image.fits', data, header
```

---

## Data Types in FITS

```idl
; Write different data types
; Integer image
int_data = FIX(DIST(256))
MKHDR, hdr, int_data
WRITEFITS, 'int_image.fits', int_data, hdr
; BITPIX = 16

; Byte image
byte_data = BYTSCL(DIST(256))
MKHDR, hdr, byte_data
WRITEFITS, 'byte_image.fits', byte_data, hdr
; BITPIX = 8

; Double precision
dbl_data = DOUBLE(DIST(256))
MKHDR, hdr, dbl_data
WRITEFITS, 'double_image.fits', dbl_data, hdr
; BITPIX = -64

; When reading, data type is determined by BITPIX
data = READFITS('int_image.fits', hdr)
PRINT, SIZE(data, /TNAME)        ; INT
PRINT, SXPAR(hdr, 'BITPIX')     ; 16
```

---

## FITS Libraries

### IDL Astronomy User's Library (astrolib)

The most widely used FITS library in IDL:

```
Key routines:
  READFITS / WRITEFITS     — Simple FITS I/O
  HEADFITS                 — Read header only
  MRDFITS / MWRFITS        — Multi-extension FITS
  SXPAR                    — Read header keyword
  SXADDPAR / SXDELPAR      — Add/delete header keyword
  FXADDPAR                 — Add keyword with formatting
  MKHDR                    — Create minimal FITS header
  SXADDHIST                — Add HISTORY records

Installation:
  https://idlastro.gsfc.nasa.gov/
  git clone https://github.com/wlandsman/IDLAstro.git
```

### SolarSoft (SSW)

For solar physics, SolarSoft provides additional FITS routines:

```
Key routines:
  READ_SDO           — Read SDO data
  AIA_PREP           — Calibrate AIA images
  HMI_PREP           — Calibrate HMI images
  FITSHEAD2STRUCT     — Convert header to structure
  STRUCT2FITSHEAD     — Convert structure to header
```

---

## Practical Example: Batch FITS Processing

```idl
PRO batch_process_fits, input_dir, output_dir
  ; Find all FITS files
  files = FILE_SEARCH(input_dir + '/*.fits', COUNT=n_files)
  IF n_files EQ 0 THEN BEGIN
    PRINT, 'No FITS files found in: ' + input_dir
    RETURN
  ENDIF

  ; Create output directory
  FILE_MKDIR, output_dir

  PRINT, FORMAT='("Processing ", I0, " FITS files...")', n_files

  FOR i = 0, n_files - 1 DO BEGIN
    ; Read file
    data = READFITS(files[i], header)
    IF N_ELEMENTS(data) LE 1 THEN CONTINUE

    ; Get metadata
    filename = FILE_BASENAME(files[i])
    exptime = SXPAR(header, 'EXPTIME')
    IF exptime LE 0 THEN exptime = 1.0

    ; Process: normalize by exposure time
    processed = FLOAT(data) / exptime

    ; Update header
    SXADDPAR, header, 'BUNIT', 'DN/s', 'Data units'
    SXADDPAR, header, 'HISTORY', 'Normalized by exposure time'
    SXADDPAR, header, 'HISTORY', 'Processed: ' + SYSTIME()

    ; Write output
    outfile = output_dir + '/' + filename
    WRITEFITS, outfile, processed, header

    IF (i + 1) MOD 10 EQ 0 THEN $
      PRINT, FORMAT='("  ", I0, "/", I0, " complete")', i + 1, n_files
  ENDFOR

  PRINT, 'Batch processing complete.'
END
```

---

## Summary

| Function/Procedure | Description |
|-------------------|-------------|
| `READFITS(file, header)` | Read FITS file |
| `WRITEFITS, file, data, header` | Write FITS file |
| `HEADFITS(file)` | Read header only |
| `MRDFITS(file, ext, header)` | Read multi-extension FITS |
| `MWRFITS, data, file` | Write/append FITS extension |
| `SXPAR(header, key)` | Read header keyword value |
| `SXADDPAR, header, key, val` | Add/modify header keyword |
| `SXDELPAR, header, key` | Delete header keyword |
| `FXADDPAR` | Add keyword with format control |
| `MKHDR, header, data` | Create minimal header |
| `FITS_OPEN / FITS_CLOSE` | Open/close for querying structure |

---

**Previous**: [Image Display](./11_Image_Display.md) | **Next**: [Date and Time](./13_Date_and_Time.md)
