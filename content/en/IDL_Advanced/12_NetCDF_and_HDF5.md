# 12. NetCDF and HDF5

**Previous**: [Curve Fitting](./11_Curve_Fitting.md) | **Next**: [IDL-Python Bridge](./13_IDL_Python_Bridge.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Read and write NetCDF files using IDL's NCDF routines
2. Read and write HDF5 files using IDL's H5 routines
3. Work with CDF files for space physics data (CDAWeb/SPDF)
4. Navigate hierarchical data structures in HDF5 and NetCDF
5. Handle metadata (attributes, dimensions, groups)

---

## 1. NetCDF File I/O

NetCDF (Network Common Data Form) is widely used in atmospheric science, climate modeling, and space physics.

### Reading NetCDF Files

```idl
; Open a NetCDF file
ncid = NCDF_OPEN('solar_wind.nc', /NOWRITE)

; Get file information
info = NCDF_INQUIRE(ncid)
PRINT, 'Dimensions: ', info.ndims
PRINT, 'Variables:  ', info.nvars
PRINT, 'Attributes: ', info.ngatts

; List dimensions
FOR i = 0, info.ndims-1 DO BEGIN
    NCDF_DIMINQ, ncid, i, dim_name, dim_size
    PRINT, 'Dim ', i, ': ', dim_name, ' = ', dim_size
ENDFOR

; List variables
FOR i = 0, info.nvars-1 DO BEGIN
    var_info = NCDF_VARINQ(ncid, i)
    PRINT, 'Var ', i, ': ', var_info.name, $
           ' type=', var_info.datatype, $
           ' ndims=', var_info.ndims
ENDFOR

; Read a variable by name
varid = NCDF_VARID(ncid, 'velocity')
NCDF_VARGET, ncid, varid, velocity

; Read with start/count for subsetting
; Read 100 time steps starting at index 50
NCDF_VARGET, ncid, varid, vel_sub, $
    OFFSET=[0, 50], COUNT=[3, 100]

; Read global attribute
NCDF_ATTGET, ncid, /GLOBAL, 'title', title_bytes
title = STRING(title_bytes)
PRINT, 'Title: ', title

; Read variable attribute
NCDF_ATTGET, ncid, varid, 'units', units_bytes
units = STRING(units_bytes)
PRINT, 'Units: ', units

; Close
NCDF_CLOSE, ncid
```

### Writing NetCDF Files

```idl
; Create a new NetCDF file
ncid = NCDF_CREATE('output.nc', /CLOBBER)

; Define dimensions
time_dimid = NCDF_DIMDEF(ncid, 'time', /UNLIMITED)
x_dimid = NCDF_DIMDEF(ncid, 'x', 100)
y_dimid = NCDF_DIMDEF(ncid, 'y', 100)

; Define variables
time_varid = NCDF_VARDEF(ncid, 'time', [time_dimid], /DOUBLE)
data_varid = NCDF_VARDEF(ncid, 'magnetic_field', $
    [x_dimid, y_dimid, time_dimid], /FLOAT)

; Add attributes
NCDF_ATTPUT, ncid, /GLOBAL, 'title', 'Solar Magnetic Field Data'
NCDF_ATTPUT, ncid, /GLOBAL, 'institution', 'Solar Physics Lab'
NCDF_ATTPUT, ncid, time_varid, 'units', 'seconds since 2024-01-01'
NCDF_ATTPUT, ncid, data_varid, 'units', 'Gauss'
NCDF_ATTPUT, ncid, data_varid, 'long_name', 'Line-of-sight magnetic field'

; End define mode (switch to data mode)
NCDF_CONTROL, ncid, /ENDEF

; Write data
time_data = DINDGEN(50) * 720.0  ; 720s cadence
NCDF_VARPUT, ncid, time_varid, time_data

; Write 3D data
bfield = RANDOMN(seed, 100, 100, 50) * 100.0
NCDF_VARPUT, ncid, data_varid, bfield

; Close
NCDF_CLOSE, ncid
PRINT, 'Wrote output.nc'
```

### NetCDF-4 with Compression

```idl
; Create NetCDF-4 file with compression
ncid = NCDF_CREATE('compressed.nc', /CLOBBER, /NETCDF4_FORMAT)

time_dim = NCDF_DIMDEF(ncid, 'time', /UNLIMITED)
x_dim = NCDF_DIMDEF(ncid, 'x', 1024)
y_dim = NCDF_DIMDEF(ncid, 'y', 1024)

; Define variable with chunking and compression
varid = NCDF_VARDEF(ncid, 'image', [x_dim, y_dim, time_dim], /FLOAT, $
    CHUNK_DIMENSIONS=[256, 256, 1], $  ; Chunk size
    GZIP=6)                             ; Compression level (1-9)

NCDF_CONTROL, ncid, /ENDEF
NCDF_VARPUT, ncid, varid, RANDOMN(seed, 1024, 1024, 10)
NCDF_CLOSE, ncid
```

---

## 2. HDF5 File I/O

HDF5 (Hierarchical Data Format 5) is used by many space missions and supports complex hierarchical data structures.

### Reading HDF5 Files

```idl
; Open HDF5 file
file_id = H5F_OPEN('satellite_data.h5')

; List top-level groups
n_obj = H5G_GET_NMEMBERS(file_id, '/')
FOR i = 0, n_obj-1 DO BEGIN
    name = H5G_GET_MEMBER_NAME(file_id, '/', i)
    PRINT, 'Object: ', name
ENDFOR

; Open a dataset
dataset_id = H5D_OPEN(file_id, '/measurements/magnetic_field')

; Get dataset info
space_id = H5D_GET_SPACE(dataset_id)
dims = H5S_GET_SIMPLE_EXTENT_DIMS(space_id)
PRINT, 'Dataset dimensions: ', dims

; Read the entire dataset
data = H5D_READ(dataset_id)
PRINT, SIZE(data, /DIMENSIONS)

; Read attributes
attr_id = H5A_OPEN_NAME(dataset_id, 'units')
units = H5A_READ(attr_id)
H5A_CLOSE, attr_id
PRINT, 'Units: ', units

; Close
H5S_CLOSE, space_id
H5D_CLOSE, dataset_id
H5F_CLOSE, file_id
```

### Reading HDF5 Subsets

```idl
; Read a subset (hyperslab) of an HDF5 dataset
file_id = H5F_OPEN('large_data.h5')
dataset_id = H5D_OPEN(file_id, '/data/images')

; Get dataspace
space_id = H5D_GET_SPACE(dataset_id)
dims = H5S_GET_SIMPLE_EXTENT_DIMS(space_id)

; Select hyperslab: read [100:200, 100:200, 0:9]
; start, count, stride, block
H5S_SELECT_HYPERSLAB, space_id, [100, 100, 0], [101, 101, 10], $
    /RESET

; Create memory dataspace
memspace_id = H5S_CREATE_SIMPLE([101, 101, 10])

; Read
data_sub = H5D_READ(dataset_id, FILE_SPACE=space_id, $
    MEMORY_SPACE=memspace_id)

H5S_CLOSE, memspace_id
H5S_CLOSE, space_id
H5D_CLOSE, dataset_id
H5F_CLOSE, file_id
```

### Writing HDF5 Files

```idl
; Create an HDF5 file
file_id = H5F_CREATE('output.h5')

; Create a group
group_id = H5G_CREATE(file_id, 'solar_data')

; Create a dataspace
dims = [512, 512, 100]
space_id = H5S_CREATE_SIMPLE(dims)

; Create a dataset
type_id = H5T_IDL_CREATE(FLTARR(1))  ; Float type
dataset_id = H5D_CREATE(group_id, 'aia_171', type_id, space_id)

; Write data
data = FLTARR(512, 512, 100)
; (fill with actual data)
H5D_WRITE, dataset_id, data

; Add attributes
attr_space = H5S_CREATE_SIMPLE([1])
attr_type = H5T_IDL_CREATE('')

attr_id = H5A_CREATE(dataset_id, 'wavelength', attr_type, attr_space)
H5A_WRITE, attr_id, '171 Angstrom'
H5A_CLOSE, attr_id

attr_id = H5A_CREATE(dataset_id, 'units', attr_type, attr_space)
H5A_WRITE, attr_id, 'DN/s'
H5A_CLOSE, attr_id

; Close everything
H5S_CLOSE, attr_space
H5T_CLOSE, attr_type
H5T_CLOSE, type_id
H5S_CLOSE, space_id
H5D_CLOSE, dataset_id
H5G_CLOSE, group_id
H5F_CLOSE, file_id
```

### H5_PARSE — Quick HDF5 Exploration

```idl
; H5_PARSE reads the entire file structure into an IDL structure
result = H5_PARSE('satellite_data.h5', /READ_DATA)

HELP, result, /STRUCTURE
; Navigate the structure hierarchy
; result._NAME, result._TYPE, result._DATA (if /READ_DATA)

; Quick way to read a specific dataset
data = H5_PARSE('file.h5', '/group/dataset', /READ_DATA)
```

---

## 3. CDF Files (Common Data Format)

CDF is the standard format for space physics data distributed through NASA's CDAWeb/SPDF.

### Reading CDF Files

```idl
; Open CDF file
cdf_id = CDF_OPEN('ace_swepam_data.cdf')

; Get file information
info = CDF_INQUIRE(cdf_id)
PRINT, 'Variables: ', info.nvars, ' rVars + ', info.nzvars, ' zVars'

; List variables
CDF_CONTROL, cdf_id, GET_NUMZVARS=nzvars
FOR i = 0, nzvars-1 DO BEGIN
    CDF_CONTROL, cdf_id, VARIABLE=i, /ZVAR, GET_VAR_INFO=varinfo
    PRINT, 'zVar ', i, ': ', varinfo
ENDFOR

; Read a variable
CDF_VARGET, cdf_id, 'Epoch', epoch_data, /ZVAR
CDF_VARGET, cdf_id, 'V_GSE', velocity, /ZVAR
CDF_VARGET, cdf_id, 'Np', density, /ZVAR

; Convert CDF epoch to readable time
; CDF epoch = milliseconds since 0 AD
; Use CDF_EPOCH to convert
n_records = N_ELEMENTS(epoch_data)
time_strings = STRARR(n_records)
FOR i = 0, n_records-1 DO BEGIN
    CDF_EPOCH, epoch_data[i], yr, mo, dy, hr, mn, sc, ms, /BREAKDOWN_EPOCH
    time_strings[i] = STRING(yr, mo, dy, hr, mn, sc, $
        FORMAT='(I4, "-", I02, "-", I02, " ", I02, ":", I02, ":", I02)')
ENDFOR

PRINT, 'Time range: ', time_strings[0], ' to ', time_strings[-1]

; Close
CDF_CLOSE, cdf_id
```

### Reading CDF Variable Attributes

```idl
cdf_id = CDF_OPEN('data.cdf')

; Get variable attributes
CDF_ATTGET, cdf_id, 'UNITS', 'Np', units
PRINT, 'Density units: ', units

CDF_ATTGET, cdf_id, 'VALIDMIN', 'Np', vmin
CDF_ATTGET, cdf_id, 'VALIDMAX', 'Np', vmax
PRINT, 'Valid range: ', vmin, ' to ', vmax

; Filter invalid data
valid = WHERE(density GE vmin AND density LE vmax, n_valid)
PRINT, 'Valid records: ', n_valid, ' / ', N_ELEMENTS(density)

CDF_CLOSE, cdf_id
```

### Writing CDF Files

```idl
; Create a CDF file
cdf_id = CDF_CREATE('output.cdf', /CLOBBER, /COL_MAJOR)

; Define variables
epoch_varid = CDF_VARCREATE(cdf_id, 'Epoch', /CDF_EPOCH, /REC_VARY, /ZVAR)
vel_varid = CDF_VARCREATE(cdf_id, 'Velocity', /CDF_FLOAT, $
    DIM=3, DIMVAR=[1], /REC_VARY, /ZVAR)
temp_varid = CDF_VARCREATE(cdf_id, 'Temperature', /CDF_FLOAT, $
    /REC_VARY, /ZVAR)

; Add attributes
CDF_ATTCREATE, cdf_id, 'UNITS', /VARIABLE_SCOPE
CDF_ATTPUT, cdf_id, 'UNITS', vel_varid, 'km/s', /ZVAR
CDF_ATTPUT, cdf_id, 'UNITS', temp_varid, 'K', /ZVAR

; Write data
n_records = 100
CDF_VARPUT, cdf_id, epoch_varid, epoch_array, /ZVAR
CDF_VARPUT, cdf_id, vel_varid, velocity_data, /ZVAR  ; [3, 100]
CDF_VARPUT, cdf_id, temp_varid, temp_data, /ZVAR      ; [100]

CDF_CLOSE, cdf_id
```

---

## 4. Practical Examples

### Reading ACE Solar Wind Data (CDF)

```idl
; ACE solar wind data from CDAWeb
; Download from: https://cdaweb.gsfc.nasa.gov/

file = 'ac_h0_swe_20240115_v001.cdf'
cdf_id = CDF_OPEN(file)

; Read key variables
CDF_VARGET, cdf_id, 'Epoch', epoch, /ZVAR
CDF_VARGET, cdf_id, 'Np', density, /ZVAR       ; Proton density
CDF_VARGET, cdf_id, 'Vp', speed, /ZVAR         ; Proton speed
CDF_VARGET, cdf_id, 'Tpr', temperature, /ZVAR  ; Proton temperature

CDF_CLOSE, cdf_id

; Filter fill values
fill_value = -1e31
good = WHERE(density GT 0 AND speed GT 0 AND density NE fill_value)
density = density[good]
speed = speed[good]
temperature = temperature[good]

; Plot
!P.MULTI = [0, 1, 3]
PLOT, density, TITLE='Proton Density', YTITLE='cm!U-3!N'
PLOT, speed, TITLE='Solar Wind Speed', YTITLE='km/s'
PLOT, temperature, TITLE='Proton Temperature', YTITLE='K'
!P.MULTI = 0
```

### Converting Between Formats

```idl
; Read NetCDF, write HDF5
ncid = NCDF_OPEN('input.nc')
varid = NCDF_VARID(ncid, 'data')
NCDF_VARGET, ncid, varid, data
NCDF_CLOSE, ncid

; Write to HDF5
file_id = H5F_CREATE('output.h5')
space_id = H5S_CREATE_SIMPLE(SIZE(data, /DIMENSIONS))
type_id = H5T_IDL_CREATE(data[0])
dset_id = H5D_CREATE(file_id, 'data', type_id, space_id)
H5D_WRITE, dset_id, data
H5D_CLOSE, dset_id
H5T_CLOSE, type_id
H5S_CLOSE, space_id
H5F_CLOSE, file_id
```

---

## 5. Format Comparison

| Feature | FITS | NetCDF | HDF5 | CDF |
|---------|------|--------|------|-----|
| Primary domain | Astronomy | Climate/Atmos | General science | Space physics |
| Hierarchical | No | Groups (v4) | Yes | No |
| Compression | Rice, GZIP | GZIP (v4) | GZIP, SZIP | GZIP |
| Unlimited dims | No | Yes | Yes | Records |
| IDL support | READFITS | NCDF_* | H5*_ | CDF_* |
| SSW support | MREADFITS | Limited | Limited | Some |
| Parallel I/O | No | Yes (v4) | Yes | No |
| Self-describing | Header | Attributes | Attributes | Attributes |

---

## Summary

| Format | Key Routines | Open/Create |
|--------|-------------|-------------|
| NetCDF | `NCDF_OPEN`, `NCDF_VARGET`, `NCDF_ATTGET` | `NCDF_CREATE` |
| HDF5 | `H5F_OPEN`, `H5D_READ`, `H5A_READ` | `H5F_CREATE` |
| CDF | `CDF_OPEN`, `CDF_VARGET`, `CDF_ATTGET` | `CDF_CREATE` |
| Quick explore | — / `H5_PARSE` / — | — |

---

**Previous**: [Curve Fitting](./11_Curve_Fitting.md) | **Next**: [IDL-Python Bridge](./13_IDL_Python_Bridge.md)
