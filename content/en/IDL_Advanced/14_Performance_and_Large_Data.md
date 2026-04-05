# 14. Performance and Large Data

**Previous**: [IDL-Python Bridge](./13_IDL_Python_Bridge.md) | **Next**: [Capstone: Solar Event Analysis](./15_Capstone_Solar_Event_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use ASSOC for memory-mapped file access to large datasets
2. Benchmark IDL code with SYSTIME and PROFILER
3. Write vectorized code that avoids slow scalar loops
4. Manage memory with TEMPORARY, HEAP_GC, and PTR_FREE
5. Optimize batch processing of large file collections
6. Handle multi-gigabyte image cubes efficiently

---

## 1. Vectorization vs Loops

The single most important optimization in IDL: **replace scalar loops with array operations**.

### Performance Comparison

```idl
; Task: compute sqrt(x^2 + y^2) for 10 million points
n = 10000000L
x = RANDOMU(seed, n)
y = RANDOMU(seed, n)

; Method 1: Scalar loop (SLOW)
t0 = SYSTIME(1)
result_loop = FLTARR(n)
FOR i = 0L, n-1 DO result_loop[i] = SQRT(x[i]^2 + y[i]^2)
t_loop = SYSTIME(1) - t0
PRINT, 'Loop:       ', t_loop, ' seconds'

; Method 2: Vectorized (FAST)
t0 = SYSTIME(1)
result_vec = SQRT(x^2 + y^2)
t_vec = SYSTIME(1) - t0
PRINT, 'Vectorized: ', t_vec, ' seconds'
PRINT, 'Speedup:    ', t_loop / t_vec, 'x'

; Typical speedup: 50-200x
```

### Common Vectorization Patterns

```idl
; Pattern 1: Conditional assignment
; SLOW:
FOR i = 0L, n-1 DO IF data[i] LT 0 THEN data[i] = 0.0
; FAST:
data = data > 0.0
; or:
neg = WHERE(data LT 0, count)
IF count GT 0 THEN data[neg] = 0.0

; Pattern 2: Accumulation
; SLOW:
total = 0.0
FOR i = 0L, n-1 DO total = total + data[i]
; FAST:
total = TOTAL(data)

; Pattern 3: Element-wise with condition
; SLOW:
result = FLTARR(n)
FOR i = 0L, n-1 DO BEGIN
    IF mask[i] THEN result[i] = data[i] * scale $
    ELSE result[i] = 0.0
ENDFOR
; FAST:
result = data * scale * mask

; Pattern 4: Running operation across array
; SLOW (pixel-by-pixel smoothing):
FOR i = 0L, nx-1 DO FOR j = 0L, ny-1 DO $
    out[i,j] = MEAN(img[i-2>0:i+2<nx-1, j-2>0:j+2<ny-1])
; FAST:
out = SMOOTH(img, 5, /EDGE_TRUNCATE)

; Pattern 5: Multi-array operations
; SLOW:
FOR i = 0L, n-1 DO result[i] = a[i] * b[i] + c[i]
; FAST:
result = a * b + c
```

---

## 2. The TEMPORARY Function

`TEMPORARY` frees the input variable's memory immediately, reusing it for the output. This halves peak memory usage for large arrays.

```idl
; Without TEMPORARY:
; Peak memory = 2 * sizeof(data) during the operation
data = FLTARR(4096, 4096)  ; ~64 MB
result = ALOG10(data + 1.0)  ; Needs another 64 MB for result
; data is still allocated (128 MB total)

; With TEMPORARY:
data = FLTARR(4096, 4096)
result = ALOG10(TEMPORARY(data) + 1.0)
; data is now undefined (freed), only result exists (~64 MB total)
PRINT, SIZE(data, /TYPE)  ; 0 (undefined)

; Chaining TEMPORARY:
a = FLTARR(1000000L)
b = FLTARR(1000000L)
; Instead of: result = SQRT(a^2 + b^2)  ; Needs 3 arrays in memory
; Do:
a2 = TEMPORARY(a)^2        ; a freed, a2 = a^2
b2 = TEMPORARY(b)^2        ; b freed, b2 = b^2
sum = TEMPORARY(a2) + TEMPORARY(b2)  ; a2, b2 freed
result = SQRT(TEMPORARY(sum))         ; sum freed
; Only 'result' remains in memory
```

---

## 3. Memory Management

### Monitoring Memory Usage

```idl
; Check current memory usage
HELP, /MEMORY
; Outputs: Current/Maximum/Total heap memory in bytes

; Get system memory info
mem = MEMORY(/CURRENT)
PRINT, 'Current heap: ', mem / 1e6, ' MB'
mem_max = MEMORY(/HIGHWATER)
PRINT, 'Peak heap:    ', mem_max / 1e6, ' MB'
```

### Freeing Memory

```idl
; Explicitly free variables
large_array = FLTARR(4096, 4096, 100)
; ... use it ...
large_array = 0  ; Frees memory (replaces with scalar 0)
; or:
DELVAR, large_array  ; At command line only (not in procedures)

; Free pointers
ptr = PTR_NEW(FLTARR(1000000L))
; ... use it ...
PTR_FREE, ptr

; Free objects
obj = OBJ_NEW('my_class')
OBJ_DESTROY, obj

; Garbage collection for orphaned heap variables
HEAP_GC  ; Find and free unreferenced pointers/objects
; Use /VERBOSE to see what was freed:
HEAP_GC, /VERBOSE
```

### Memory-Efficient Patterns

```idl
; Process files one at a time instead of loading all into memory
; BAD: Load everything
all_data = FLTARR(4096, 4096, 1000)  ; 64 GB!

; GOOD: Process sequentially
result = FLTARR(4096, 4096)
FOR i = 0, nfiles-1 DO BEGIN
    data = READFITS(files[i])
    result += TEMPORARY(data)  ; Free each frame immediately
ENDFOR
result /= nfiles
```

---

## 4. ASSOC — Memory-Mapped File Access

`ASSOC` maps a file directly to an IDL variable without reading the entire file into memory. Essential for files larger than available RAM.

### Basic ASSOC Usage

```idl
; Create a large binary file
nx = 4096L & ny = 4096L & nt = 500L
; This file would be 32 GB — too large for RAM
; Write frame by frame:
OPENW, lun, 'large_cube.dat', /GET_LUN
frame = FLTARR(nx, ny)
FOR t = 0L, nt-1 DO BEGIN
    ; Generate or read each frame
    frame[*] = RANDOMU(seed, nx, ny) * 1000.0
    WRITEU, lun, frame
ENDFOR
FREE_LUN, lun

; Read using ASSOC (memory-mapped)
OPENR, lun, 'large_cube.dat', /GET_LUN
cube = ASSOC(lun, FLTARR(nx, ny))
; cube is NOT in memory — it's a file mapping

; Access individual frames (reads only that frame from disk)
frame0 = cube[0]      ; Read frame 0
frame100 = cube[100]   ; Read frame 100
frame499 = cube[499]   ; Read last frame

PRINT, SIZE(frame0, /DIMENSIONS)  ; 4096 4096
PRINT, MEAN(frame0)

; Compute temporal mean without loading entire file
temporal_sum = FLTARR(nx, ny)
FOR t = 0L, nt-1 DO temporal_sum += cube[t]
temporal_mean = temporal_sum / nt

FREE_LUN, lun
```

### ASSOC with Offsets

```idl
; Skip a file header
header_bytes = 2880L  ; FITS header size (for example)
OPENR, lun, 'data_with_header.bin', /GET_LUN
cube = ASSOC(lun, FLTARR(nx, ny), header_bytes)
; Now cube[0] starts after the header
frame = cube[0]
FREE_LUN, lun
```

---

## 5. Benchmarking with SYSTIME

```idl
; Basic timing
t0 = SYSTIME(1)          ; Wall clock time in seconds
; ... code to benchmark ...
elapsed = SYSTIME(1) - t0
PRINT, 'Elapsed: ', elapsed, ' seconds'

; Multiple runs for stable timing
n_runs = 10
times = FLTARR(n_runs)
FOR r = 0, n_runs-1 DO BEGIN
    t0 = SYSTIME(1)
    result = FFT(RANDOMU(seed, 4096, 4096))
    times[r] = SYSTIME(1) - t0
ENDFOR
PRINT, 'FFT 4096x4096 — Mean: ', MEAN(times), ' Std: ', STDDEV(times), ' s'
```

### PROFILER

```idl
; Profile procedure execution
PROFILER, /SYSTEM    ; Start profiling
PROFILER, /RESET     ; Clear previous data

; Run the code to profile
my_analysis_procedure, data

; Get results
PROFILER, /REPORT    ; Print timing report
; Shows: procedure name, # calls, total time, self time

; Or get structured results
PROFILER, /REPORT, DATA=prof_data
; prof_data: structure array with timing information
```

---

## 6. SAVE/RESTORE Optimization

```idl
; SAVE/RESTORE for caching intermediate results
cache_file = 'calibrated_cache.sav'

IF FILE_TEST(cache_file) THEN BEGIN
    ; Load cached data (fast)
    RESTORE, cache_file
    PRINT, 'Loaded from cache'
ENDIF ELSE BEGIN
    ; Expensive computation
    files = FILE_SEARCH('*.fits', COUNT=nf)
    data_cube = FLTARR(512, 512, nf)
    FOR i = 0, nf-1 DO BEGIN
        read_sdo, files[i], idx, dat
        aia_prep, idx, dat, oi, od, /NORMALIZE, /REGISTER
        data_cube[*, *, i] = REBIN(od, 512, 512)
    ENDFOR

    ; Save cache
    SAVE, data_cube, FILENAME=cache_file
    PRINT, 'Saved cache: ', cache_file
ENDELSE

; SAVE with compression
SAVE, data_cube, FILENAME='compressed.sav', /COMPRESS
```

---

## 7. Batch Processing Patterns

### Sequential File Processing

```idl
PRO batch_process, input_dir, output_dir
    files = FILE_SEARCH(input_dir + '/*.fits', COUNT=nf)
    IF nf EQ 0 THEN BEGIN
        PRINT, 'No files found in ', input_dir
        RETURN
    ENDIF

    FILE_MKDIR, output_dir
    t_start = SYSTIME(1)

    FOR i = 0L, nf-1 DO BEGIN
        ; Read
        read_sdo, files[i], index, data

        ; Process
        aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER

        ; Compute derived quantities
        submap = odata[1000:3000, 1000:3000]
        ; ... analysis ...

        ; Save result
        outfile = output_dir + '/' + FILE_BASENAME(files[i])
        mwritefits, oindex, odata, OUTFILE=outfile

        ; Progress report
        IF (i MOD 50) EQ 0 THEN BEGIN
            elapsed = SYSTIME(1) - t_start
            rate = (i+1) / elapsed
            eta = (nf - i - 1) / rate
            PRINT, STRING(i+1, nf, elapsed, eta, $
                FORMAT='(I5, "/", I5, "  Elapsed: ", F7.1, "s  ETA: ", F7.1, "s")')
        ENDIF

        ; Free memory each iteration
        data = 0 & odata = 0
    ENDFOR

    total_time = SYSTIME(1) - t_start
    PRINT, STRING(nf, total_time, total_time/nf, $
        FORMAT='("Processed ", I0, " files in ", F7.1, "s (", F5.2, "s/file)")')
END
```

### Collecting Results

```idl
; Collect summary statistics from many files
PRO collect_statistics, file_list, output_csv
    nf = N_ELEMENTS(file_list)

    ; Pre-allocate result arrays
    times = DBLARR(nf)
    means = FLTARR(nf)
    maxvals = FLTARR(nf)
    total_flux = DBLARR(nf)

    FOR i = 0L, nf-1 DO BEGIN
        read_sdo, file_list[i], idx, dat
        times[i] = ANYTIM(idx.date_obs)
        means[i] = MEAN(dat)
        maxvals[i] = MAX(dat)
        total_flux[i] = TOTAL(DOUBLE(dat))
        dat = 0  ; Free memory
    ENDFOR

    ; Write CSV
    OPENW, lun, output_csv, /GET_LUN
    PRINTF, lun, 'time,mean,max,total_flux'
    FOR i = 0L, nf-1 DO BEGIN
        PRINTF, lun, ANYTIM(times[i], /CCSDS), ',', $
            means[i], ',', maxvals[i], ',', total_flux[i]
    ENDFOR
    FREE_LUN, lun
END
```

---

## 8. Handling Large Image Cubes

```idl
; Strategy for processing a cube that fits in memory but is large
; (e.g., 4096 x 4096 x 200 = 6.4 GB as float)

; Option 1: Spatial rebinning first
cube = FLTARR(1024, 1024, 200)  ; Rebin from 4096 to 1024 (400 MB)
FOR i = 0, 199 DO BEGIN
    dat = READFITS(files[i])
    cube[*, *, i] = REBIN(TEMPORARY(dat), 1024, 1024)
ENDFOR

; Option 2: Process in spatial tiles
tile_size = 512
n_tiles_x = 4096 / tile_size  ; 8 tiles
n_tiles_y = 4096 / tile_size  ; 8 tiles
n_times = 200

FOR tx = 0, n_tiles_x-1 DO BEGIN
    FOR ty = 0, n_tiles_y-1 DO BEGIN
        x0 = tx * tile_size
        y0 = ty * tile_size

        ; Load tile time series
        tile_cube = FLTARR(tile_size, tile_size, n_times)
        FOR t = 0, n_times-1 DO BEGIN
            dat = READFITS(files[t])
            tile_cube[*, *, t] = dat[x0:x0+tile_size-1, y0:y0+tile_size-1]
            dat = 0
        ENDFOR

        ; Process this tile (temporal analysis, etc.)
        tile_mean = MEAN(tile_cube, DIMENSION=3)
        tile_std = FLTARR(tile_size, tile_size)
        FOR ix = 0, tile_size-1 DO FOR iy = 0, tile_size-1 DO $
            tile_std[ix, iy] = STDDEV(tile_cube[ix, iy, *])

        ; Save tile results
        ; ...
    ENDFOR
ENDFOR

; Option 3: Use ASSOC for random access
OPENR, lun, 'cube.dat', /GET_LUN
cube_assoc = ASSOC(lun, FLTARR(4096, 4096))

; Now process frame-by-frame without full cube in memory
temporal_max = FLTARR(4096, 4096) - 1e30
FOR t = 0L, 199 DO BEGIN
    frame = cube_assoc[t]
    temporal_max = temporal_max > frame
ENDFOR
FREE_LUN, lun
```

---

## Summary

| Technique | Key Functions | Impact |
|-----------|-------------|--------|
| Vectorization | Array operators | 50-200x speedup |
| TEMPORARY | `TEMPORARY()` | Halves peak memory |
| ASSOC | `ASSOC()` | Files larger than RAM |
| Profiling | `SYSTIME(1)`, `PROFILER` | Identify bottlenecks |
| Memory mgmt | `HEAP_GC`, `MEMORY()` | Prevent crashes |
| SAVE/RESTORE | `SAVE`, `RESTORE` | Cache results |
| Batch patterns | Sequential + progress | Large file sets |

---

**Previous**: [IDL-Python Bridge](./13_IDL_Python_Bridge.md) | **Next**: [Capstone: Solar Event Analysis](./15_Capstone_Solar_Event_Analysis.md)
