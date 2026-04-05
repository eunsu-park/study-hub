# 13. IDL-Python Bridge

**Previous**: [NetCDF and HDF5](./12_NetCDF_and_HDF5.md) | **Next**: [Performance and Large Data](./14_Performance_and_Large_Data.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Call Python functions from IDL using the built-in Python bridge
2. Call IDL from Python using pIDLy and hissw
3. Map data types between IDL and Python
4. Design hybrid IDL-Python workflows
5. Plan a migration strategy from IDL to Python (SunPy, Astropy)

---

## 1. IDL's Built-in Python Bridge

IDL 8.5+ includes a native Python bridge that allows calling Python code directly from IDL.

### Basic Python Calls from IDL

```idl
; Import a Python module
np = PYTHON.IMPORT('numpy')

; Call numpy functions
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
PRINT, np.mean(arr)
PRINT, np.std(arr)

; Create numpy arrays
x = np.linspace(0, 10, 100)
y = np.sin(x)

; Convert Python objects to IDL arrays
x_idl = FLOAT(x)
y_idl = FLOAT(y)
PLOT, x_idl, y_idl
```

### Using SunPy from IDL

```idl
; Import SunPy
sunpy = PYTHON.IMPORT('sunpy')
sunpy_map = PYTHON.IMPORT('sunpy.map')

; Load a FITS file through SunPy
smap = sunpy_map.Map('aia_171.fits')
PRINT, smap.date
PRINT, smap.wavelength
PRINT, smap.dimensions

; Get data as IDL array
data = FLOAT(smap.data)
PRINT, SIZE(data, /DIMENSIONS)

; Access WCS
PRINT, smap.scale
PRINT, smap.reference_coordinate
```

### Using Astropy from IDL

```idl
; Import Astropy
fits = PYTHON.IMPORT('astropy.io.fits')
units = PYTHON.IMPORT('astropy.units')
coords = PYTHON.IMPORT('astropy.coordinates')

; Read FITS with Astropy
hdulist = fits.open('image.fits')
header = hdulist[0].header
data = FLOAT(hdulist[0].data)
hdulist.close()
```

### Matplotlib from IDL

```idl
; Use matplotlib for advanced plotting
plt = PYTHON.IMPORT('matplotlib.pyplot')

x = PYTHON.IMPORT('numpy').linspace(0, 10, 100)
y = PYTHON.IMPORT('numpy').sin(x)

plt.figure(figsize=PYTHON.TUPLE(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Matplotlib from IDL')
plt.savefig('plot_from_idl.png', dpi=150)
plt.close()
```

---

## 2. Data Type Mapping

### IDL to Python

| IDL Type | Python Type | Notes |
|----------|-------------|-------|
| `BYTE` | `numpy.uint8` | |
| `INT` | `numpy.int16` | |
| `LONG` | `numpy.int32` | |
| `LONG64` | `numpy.int64` | |
| `FLOAT` | `numpy.float32` | |
| `DOUBLE` | `numpy.float64` | |
| `STRING` | `str` | |
| `COMPLEX` | `numpy.complex64` | |
| `DCOMPLEX` | `numpy.complex128` | |
| Array | `numpy.ndarray` | Column-major to row-major! |
| Structure | `dict` | Requires manual conversion |
| Pointer | Not supported | Dereference first |

### Array Order Warning

```idl
; CRITICAL: IDL is column-major (Fortran order), Python is row-major (C order)
; The bridge handles this, but be aware of dimension ordering

; IDL array [3, 4] = 3 columns, 4 rows
arr_idl = INDGEN(3, 4)
PRINT, SIZE(arr_idl, /DIMENSIONS)  ; 3 4

; When passed to Python, it becomes shape (4, 3) in numpy
; (the memory layout is preserved, but interpretation differs)

np = PYTHON.IMPORT('numpy')
arr_py = np.array(arr_idl)
PRINT, arr_py.shape  ; (4, 3) in Python convention
```

### Converting Complex Structures

```idl
; IDL structures need manual conversion
; Create a dict for Python
py_dict = PYTHON.IMPORT('builtins').dict()
py_dict['name'] = 'AIA 171'
py_dict['wavelength'] = 171.0
py_dict['data'] = FLOAT(image_data)

; Pass to Python function
my_module = PYTHON.IMPORT('my_analysis')
result = my_module.analyze(py_dict)
```

---

## 3. Calling IDL from Python

### Using pIDLy

```python
# pIDLy: Python-IDL bridge (runs IDL as subprocess)
# Install: pip install pidly

import pidly
idl = pidly.IDL()

# Execute IDL commands
idl('x = FINDGEN(100) * 0.1')
idl('y = SIN(x)')

# Get variables from IDL
x = idl.x  # Returns numpy array
y = idl.y

# Pass variables to IDL
import numpy as np
data = np.random.randn(100, 100)
idl.data = data
idl('PRINT, SIZE(data, /DIMENSIONS)')  # 100 100

# Call IDL procedures
idl('read_sdo, "aia_171.fits", index, data')
aia_data = idl.data
aia_header = idl.index  # Returned as dict

# Close
idl.close()
```

### Using hissw

```python
# hissw: simplified SSW access from Python
# Install: pip install hissw

import hissw

# Configure SSW environment
ssw_env = hissw.Environment(ssw_packages=['sdo/aia', 'sdo/hmi'])

# Run SSW IDL code
script = """
read_sdo, '{{ file }}', index, data
aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER
"""

inputs = {'file': 'aia_171.fits'}
outputs = ssw_env.run(script, args=inputs, save=['oindex', 'odata'])

# Access results
calibrated_data = outputs['odata']
header = outputs['oindex']
```

### Using idlpy (IDL's official Python module)

```python
# idlpy: official Python-to-IDL bridge (requires IDL license)
from idlpy import IDL

# Execute IDL code
IDL.run('PRINT, !VERSION.RELEASE')

# Call IDL functions
result = IDL.dist(256)  # Returns numpy array
print(result.shape)     # (256, 256)

# Pass arguments
import numpy as np
data = np.random.randn(100).astype(np.float32)
smoothed = IDL.smooth(data, 5)
```

---

## 4. Hybrid Workflows

### Pattern: IDL for Calibration, Python for Analysis

```idl
; Step 1: Calibrate in IDL (SSW has the best calibration pipelines)
files = FILE_SEARCH('/data/aia/171/*.fits', COUNT=nf)
FOR i = 0, nf-1 DO BEGIN
    read_sdo, files[i], index, data
    aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

    ; Save calibrated data in a Python-friendly format
    outfile = '/data/aia/171/prepped/frame_' + STRING(i, FORMAT='(I04)') + '.fits'
    mwritefits, oindex, odata, OUTFILE=outfile
ENDFOR
```

```python
# Step 2: Analyze in Python
import sunpy.map
import numpy as np
from glob import glob

files = sorted(glob('/data/aia/171/prepped/frame_*.fits'))
maps = sunpy.map.Map(files)

# Time series analysis with scipy
from scipy import signal
# ... (Python has richer analysis ecosystem)
```

### Pattern: Python Data Retrieval, IDL Processing

```python
# Step 1: Download data with Python (Fido is easier than SSW for downloads)
from sunpy.net import Fido, attrs as a

result = Fido.search(
    a.Time('2024-01-15 12:00', '2024-01-15 13:00'),
    a.Instrument('AIA'),
    a.Wavelength(171*u.angstrom)
)
files = Fido.fetch(result, path='/data/aia/171/')
```

```idl
; Step 2: Process in IDL with SSW-specific tools
files = FILE_SEARCH('/data/aia/171/*.fits', COUNT=nf)
read_sdo, files, index, data
aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER
; ... use SSW-specific analysis tools
```

---

## 5. Migration from IDL to Python

### Equivalent Libraries

| IDL / SSW | Python Equivalent |
|-----------|-------------------|
| Core IDL | NumPy, SciPy |
| READFITS / WRITEFITS | astropy.io.fits |
| PLOT, CONTOUR, SURFACE | Matplotlib |
| SMOOTH, MEDIAN, CONVOL | scipy.ndimage |
| FFT | numpy.fft, scipy.fft |
| CURVEFIT, MPFIT | scipy.optimize.curve_fit, lmfit |
| MAP_SET, MAP_CONTINENTS | Cartopy |
| SolarSoft (general) | SunPy |
| AIA_PREP | aiapy |
| WCS routines | astropy.wcs, sunpy.coordinates |
| ANYTIM, UTC2TAI | astropy.time, sunpy.time |
| RHESSI/OSPEX | sunpy.timeseries, xrstools |
| CHIANTI | ChiantiPy, fiasco |
| Widget programming | PyQt, tkinter, Jupyter widgets |

### SunPy Equivalents for Common SSW Tasks

```python
# SunPy equivalent of key SSW operations

# Reading and displaying solar data
import sunpy.map
smap = sunpy.map.Map('aia_171.fits')
smap.peek()  # Quick display

# AIA calibration (aiapy)
import aiapy.calibrate as ac
smap_prepped = ac.register(ac.update_pointing(smap))
smap_normalized = ac.normalize_exposure(smap_prepped)

# Time handling
from sunpy.time import parse_time
t = parse_time('2024-01-15 12:00:00')

# Coordinate transforms
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
coord = SkyCoord(200, 300, unit='arcsec', frame=frames.Helioprojective,
                 observer='earth', obstime='2024-01-15')
hgs = coord.transform_to(frames.HeliographicStonyhurst)

# GOES data
from sunpy.timeseries import TimeSeries
goes = TimeSeries('goes_xrs_data.nc')
goes.peek()

# Data download
from sunpy.net import Fido, attrs
result = Fido.search(attrs.Time('2024-01-15', '2024-01-16'),
                     attrs.Instrument('AIA'),
                     attrs.Wavelength(171))
```

### Migration Strategy

1. **Phase 1: Side-by-side** — Use hissw/pIDLy for SSW calibration, Python for new analysis
2. **Phase 2: Replace I/O** — Switch to SunPy/Astropy for data access and basic operations
3. **Phase 3: Replace calibration** — Use aiapy, sunkit-instruments when mature enough
4. **Phase 4: Full Python** — Only keep IDL for legacy code that has no Python equivalent

### What to Keep in IDL

- Mission-specific calibration pipelines (some have no Python equivalent yet)
- Legacy analysis code that is well-tested and not worth rewriting
- OSPEX spectral fitting (partially available in Python, but IDL version is more mature)
- Interactive widget-based tools (SSW has many specialized GUIs)

---

## 6. Practical: Cross-Language Workflow

```idl
; IDL side: calibrate and save
PRO calibrate_for_python, input_dir, output_dir
    files = FILE_SEARCH(input_dir + '/*.fits', COUNT=nf)
    IF nf EQ 0 THEN BEGIN
        PRINT, 'No files found'
        RETURN
    ENDIF

    FILE_MKDIR, output_dir

    FOR i = 0, nf-1 DO BEGIN
        read_sdo, files[i], index, data
        aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

        ; Save as standard FITS (readable by astropy)
        outfile = output_dir + '/cal_' + FILE_BASENAME(files[i])
        mwritefits, oindex, odata, OUTFILE=outfile
        PRINT, 'Wrote: ', outfile
    ENDFOR
END
```

```python
# Python side: analyze calibrated data
import sunpy.map
import numpy as np
from pathlib import Path

cal_dir = Path('/data/calibrated/')
files = sorted(cal_dir.glob('cal_*.fits'))

# Create a map sequence
maps = [sunpy.map.Map(f) for f in files]

# Extract light curve from a region
import astropy.units as u
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord

# Define ROI
bottom_left = SkyCoord(-300*u.arcsec, 200*u.arcsec,
                        frame=frames.Helioprojective)
top_right = SkyCoord(-100*u.arcsec, 400*u.arcsec,
                      frame=frames.Helioprojective)

intensities = []
times = []
for m in maps:
    submap = m.submap(bottom_left, top_right=top_right)
    intensities.append(np.mean(submap.data))
    times.append(m.date)

# Analysis with scipy, matplotlib, etc.
```

---

## Summary

| Approach | Direction | Best For |
|----------|-----------|----------|
| IDL Python bridge | IDL -> Python | Quick Python library access |
| pIDLy | Python -> IDL | Scripting IDL from Python |
| hissw | Python -> SSW IDL | SSW calibration pipelines |
| idlpy | Python -> IDL | Official bridge |
| SunPy/Astropy | Pure Python | New projects |

---

**Previous**: [NetCDF and HDF5](./12_NetCDF_and_HDF5.md) | **Next**: [Performance and Large Data](./14_Performance_and_Large_Data.md)
