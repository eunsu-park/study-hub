# Debugging and Best Practices

**Previous**: [Date and Time](./13_Date_and_Time.md) | **Next**: [Project: Solar Light Curve](./15_Project_Solar_Light_Curve.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use STOP for breakpoints and .CONTINUE for resuming
2. Inspect variables with HELP and PRINT during debugging
3. Manage compilation with .COMPILE, .RUN, and RETALL
4. Use RESOLVE_ALL for dependency checking
5. Manage memory with HEAP_GC and pointer cleanup
6. Follow IDL coding conventions for readable, maintainable code
7. Write efficient code by vectorizing operations instead of using loops

---

Debugging and writing clean code are skills that separate productive IDL programmers from frustrated ones. IDL provides useful debugging tools, and following best practices can prevent many bugs before they occur.

## Debugging Tools

### STOP — Breakpoints

`STOP` halts execution and returns control to the IDL command prompt, preserving all variables:

```idl
PRO process_data, filename
  data = READFITS(filename, header)

  ; Check data at this point
  STOP    ; Execution pauses here

  ; After inspecting variables at the prompt, type .CONTINUE to resume
  processed = data / SXPAR(header, 'EXPTIME')
  STOP    ; Another checkpoint

  PRINT, 'Processing complete'
END
```

At the STOP point, you can inspect any variable:

```idl
IDL> process_data, 'test.fits'
% Stop encountered: PROCESS_DATA         3 /path/to/process_data.pro
IDL> HELP, data
DATA            FLOAT     = Array[512, 512]
IDL> PRINT, MIN(data), MAX(data)
     0.00000      9876.54
IDL> .CONTINUE
% Stop encountered: PROCESS_DATA         8 /path/to/process_data.pro
IDL> HELP, processed
IDL> .CONTINUE
Processing complete
```

### Conditional STOP

```idl
PRO find_anomaly, data
  FOR i = 0, N_ELEMENTS(data) - 1 DO BEGIN
    ; Stop only when we find an anomalous value
    IF data[i] GT 1e10 OR ~FINITE(data[i]) THEN BEGIN
      PRINT, 'Anomaly at index:', i, ' value:', data[i]
      STOP
    ENDIF
  ENDFOR
END
```

### .CONTINUE, .STEP, .SKIP

```idl
; After a STOP:
IDL> .CONTINUE       ; Resume execution (or .CON)
IDL> .STEP           ; Execute one line, then stop (or .S)
IDL> .STEP 5         ; Execute 5 lines
IDL> .SKIP           ; Skip the current line
IDL> .OUT            ; Continue until the current routine returns
```

---

### HELP — Variable Inspection

```idl
; Inspect a single variable
HELP, my_array
; MY_ARRAY        FLOAT     = Array[100, 200]

; Inspect multiple variables
HELP, x, y, z, header

; Show all variables in current scope
HELP

; Show specific types
HELP, /STRUCTURES     ; Show only structures
HELP, /ROUTINES       ; Show compiled routines
HELP, /MEMORY         ; Show memory usage
HELP, /SOURCE_FILES   ; Show source file paths

; Inspect a structure
HELP, my_struct, /STRUCTURE

; Show heap variables (pointers and objects)
HELP, /HEAP_VARIABLES
```

### PRINT for Debugging

```idl
; Strategic print statements
PRO debug_example, data
  PRINT, '--- Entering debug_example ---'
  PRINT, 'Input type:', SIZE(data, /TNAME)
  PRINT, 'Input dimensions:', SIZE(data, /DIMENSIONS)
  PRINT, 'Input range:', MIN(data), MAX(data)

  ; Processing step 1
  cleaned = data
  bad = WHERE(~FINITE(data), n_bad)
  PRINT, 'Bad values found:', n_bad
  IF n_bad GT 0 THEN cleaned[bad] = MEDIAN(data[WHERE(FINITE(data))])

  ; Processing step 2
  result = SMOOTH(cleaned, 5)
  PRINT, 'Result range:', MIN(result), MAX(result)
  PRINT, '--- Leaving debug_example ---'
END
```

---

## Session Management

### RETALL — Return to Main Level

```idl
; If you're stuck deep in nested calls after an error:
IDL> RETALL
; Returns to the main program level, clearing the call stack

; View the call stack
IDL> HELP, /TRACEBACK
; Shows where execution is currently paused
```

### .COMPILE and .RUN

```idl
; Recompile after editing source code
IDL> .COMPILE my_routine        ; Compile my_routine.pro
IDL> .RUN my_script             ; Compile and run main-level script

; Force recompilation
IDL> .COMPILE -v my_routine     ; Verbose compilation

; Reset and recompile everything
IDL> .RESET_SESSION             ; Clear ALL variables and compiled routines
IDL> .FULL_RESET_SESSION        ; Complete reset including graphics
```

### RESOLVE_ALL

```idl
; Compile all unresolved dependencies
IDL> RESOLVE_ALL
; Finds and compiles all routines called by currently compiled code

; Useful before SAVE to ensure everything is included
IDL> RESOLVE_ALL
IDL> SAVE, /ROUTINES, FILENAME='my_library.sav'
```

---

## Error Handling

### CATCH

```idl
; Modern error handling with CATCH
PRO safe_process, filename
  CATCH, error_status
  IF error_status NE 0 THEN BEGIN
    PRINT, 'Error caught: ' + !ERROR_STATE.MSG
    CATCH, /CANCEL    ; Cancel the error handler
    RETURN
  ENDIF

  ; Code that might fail
  data = READFITS(filename, header)
  result = data / SXPAR(header, 'EXPTIME')
  WRITEFITS, 'output.fits', result, header

  CATCH, /CANCEL    ; Clean up error handler
  PRINT, 'Processing successful'
END
```

### ON_ERROR

```idl
; ON_ERROR controls behavior when an error occurs
; ON_ERROR, 0   ; Stop and enter debugger (default for interactive)
; ON_ERROR, 1   ; Return to main level
; ON_ERROR, 2   ; Return to caller
; ON_ERROR, 3   ; Stop at the point of error

PRO my_routine
  ON_ERROR, 2    ; Return to caller on error
  ; ... code that might fail ...
END
```

### MESSAGE

```idl
; Raise informational or error messages
PRO validate_input, data, name
  IF N_ELEMENTS(data) EQ 0 THEN $
    MESSAGE, 'Input ' + name + ' is undefined'

  IF SIZE(data, /N_DIMENSIONS) NE 2 THEN $
    MESSAGE, 'Input ' + name + ' must be 2D, got ' + $
    STRTRIM(SIZE(data, /N_DIMENSIONS), 2) + 'D'

  ; Informational message (does not halt execution)
  MESSAGE, 'Input validated: ' + name + $
    ' [' + STRJOIN(STRTRIM(SIZE(data, /DIMENSIONS), 2), 'x') + ']', $
    /INFORMATIONAL
END
```

---

## Memory Management

### HEAP_GC — Garbage Collection

```idl
; IDL uses reference counting for heap variables (pointers and objects)
; Orphaned heap variables waste memory

; Manual garbage collection
HEAP_GC

; With verbose output
HEAP_GC, /VERBOSE

; Check for orphaned pointers
HELP, /HEAP_VARIABLES
```

### Pointer Cleanup

```idl
; Always free pointers when done
ptr = PTR_NEW(FINDGEN(1000000))
; ... use ptr ...
PTR_FREE, ptr

; Free all pointers in a structure
PRO free_struct_pointers, s
  tags = TAG_NAMES(s)
  FOR i = 0, N_TAGS(s) - 1 DO BEGIN
    IF SIZE(s.(i), /TYPE) EQ 10 THEN BEGIN  ; Type 10 = POINTER
      IF PTR_VALID(s.(i)) THEN PTR_FREE, s.(i)
    ENDIF
  ENDFOR
END
```

### Memory-Efficient Practices

```idl
; Use TEMPORARY to avoid copies
big_array = FLTARR(4096, 4096)
; This creates a copy:
result = big_array * 2.0

; This reuses the memory:
result = TEMPORARY(big_array) * 2.0
; big_array is now undefined

; Check memory usage
HELP, /MEMORY
; Heap Var Bytes  Active   Free
;    Pointers:      1024      2
;    Objects:           0      0
```

---

## Coding Conventions

### File and Routine Naming

```idl
; One routine per file, filename matches routine name
; File: read_solar_data.pro
PRO read_solar_data, filename, data, header
  ; ...
END

; Functions follow the same convention
; File: compute_temperature.pro
FUNCTION compute_temperature, flux, wavelength
  ; ...
END
```

### Commenting Style

```idl
;+
; NAME:
;   compute_temperature
;
; PURPOSE:
;   Calculate brightness temperature from flux and wavelength
;   using the Planck function.
;
; CALLING SEQUENCE:
;   temp = compute_temperature(flux, wavelength)
;
; INPUTS:
;   flux       - Spectral flux in W/m^2/Hz
;   wavelength - Wavelength in meters
;
; KEYWORDS:
;   /RAYLEIGH_JEANS - Use Rayleigh-Jeans approximation
;
; OUTPUTS:
;   Returns brightness temperature in Kelvin
;
; MODIFICATION HISTORY:
;   2024-07-15  Author  Initial version
;-
FUNCTION compute_temperature, flux, wavelength, RAYLEIGH_JEANS=rj
  h = 6.626D-34    ; Planck constant
  c = 3.0D8         ; Speed of light
  k = 1.381D-23     ; Boltzmann constant

  IF KEYWORD_SET(rj) THEN BEGIN
    ; Rayleigh-Jeans approximation
    RETURN, flux * wavelength^2 / (2.0D0 * k)
  ENDIF

  ; Full Planck function inversion
  nu = c / wavelength
  RETURN, h * nu / (k * ALOG(2.0D0 * h * nu^3 / (c^2 * flux) + 1.0D0))
END
```

### Variable Naming

```idl
; Use descriptive names
; Good:
exposure_time = SXPAR(header, 'EXPTIME')
solar_radius_arcsec = SXPAR(header, 'RSUN_OBS')
pixel_scale = SXPAR(header, 'CDELT1')

; Bad:
et = SXPAR(header, 'EXPTIME')
r = SXPAR(header, 'RSUN_OBS')
s = SXPAR(header, 'CDELT1')

; Loop variables: short names are fine
FOR i = 0, nx - 1 DO ...
FOR ix = 0, nx - 1 DO ...    ; Even better: directional index
```

---

## Efficiency Tips: Vectorize!

The single most important performance tip in IDL: **avoid loops when array operations will do**.

### Bad: Loop Over Elements

```idl
; SLOW: Element-by-element processing
n = 1000000L
data = RANDOMN(seed, n)
result = FLTARR(n)

t0 = SYSTIME(1)
FOR i = 0L, n - 1 DO BEGIN
  IF data[i] GT 0 THEN result[i] = SQRT(data[i]) $
  ELSE result[i] = 0.0
ENDFOR
PRINT, 'Loop time:', SYSTIME(1) - t0, ' seconds'
```

### Good: Vectorized Operations

```idl
; FAST: Array operations
t0 = SYSTIME(1)
positive = WHERE(data GT 0, count, COMPLEMENT=negative)
result = FLTARR(n)
IF count GT 0 THEN result[positive] = SQRT(data[positive])
PRINT, 'Vectorized time:', SYSTIME(1) - t0, ' seconds'
; Typically 10-100x faster than the loop
```

### More Vectorization Examples

```idl
; BAD: Loop to compute distances
distances = FLTARR(n)
cx = 256.0
cy = 256.0
FOR i = 0L, n - 1 DO BEGIN
  distances[i] = SQRT((x[i] - cx)^2 + (y[i] - cy)^2)
ENDFOR

; GOOD: Vectorized distance
distances = SQRT((x - cx)^2 + (y - cy)^2)

; BAD: Loop to threshold
FOR i = 0L, N_ELEMENTS(data) - 1 DO BEGIN
  IF data[i] LT 0 THEN data[i] = 0
ENDFOR

; GOOD: Vectorized threshold
data = data > 0    ; min operator clips to 0

; Or using WHERE
neg = WHERE(data LT 0, count)
IF count GT 0 THEN data[neg] = 0

; BAD: Loop to accumulate
total_val = 0.0
FOR i = 0L, N_ELEMENTS(data) - 1 DO total_val += data[i]

; GOOD: Built-in function
total_val = TOTAL(data)
```

### When Loops Are Necessary

```idl
; Loops are sometimes unavoidable:

; 1. When current element depends on previous result
result = FLTARR(n)
result[0] = data[0]
FOR i = 1L, n - 1 DO result[i] = 0.9 * result[i-1] + 0.1 * data[i]

; 2. When processing files
files = FILE_SEARCH('*.fits')
FOR i = 0, N_ELEMENTS(files) - 1 DO BEGIN
  data = READFITS(files[i])
  ; ... process ...
ENDFOR

; 3. Complex conditional logic that cannot be vectorized
```

---

## Common Pitfalls

### Integer Division

```idl
; WRONG
PRINT, 1/3              ;        0
; RIGHT
PRINT, 1.0/3.0          ;     0.333333
```

### Array Size Mismatch

```idl
; Always check dimensions before operations
IF N_ELEMENTS(a) NE N_ELEMENTS(b) THEN BEGIN
  MESSAGE, 'Array size mismatch', /INFORMATIONAL
  RETURN
ENDIF
```

### Forgetting BEGIN/END

```idl
; WRONG: Only first statement is in the IF
IF x GT 0 THEN
  PRINT, 'Positive'       ; This is the IF body
  y = SQRT(x)              ; This ALWAYS executes!

; RIGHT:
IF x GT 0 THEN BEGIN
  PRINT, 'Positive'
  y = SQRT(x)
ENDIF
```

### Not Checking WHERE Results

```idl
; WRONG: Will crash if no matches
idx = WHERE(data GT threshold)
good_data = data[idx]    ; Fails if idx = [-1]

; RIGHT:
idx = WHERE(data GT threshold, count)
IF count GT 0 THEN good_data = data[idx] $
ELSE PRINT, 'No values above threshold'
```

### Not Freeing LUNs

```idl
; WRONG: LUN leak
OPENR, lun, 'data.txt', /GET_LUN
; ... error occurs, FREE_LUN never called

; RIGHT: Use error handling
OPENR, lun, 'data.txt', /GET_LUN
ON_IOERROR, cleanup
; ... read data ...
FREE_LUN, lun
RETURN

cleanup:
PRINT, 'Error reading file'
IF N_ELEMENTS(lun) GT 0 THEN FREE_LUN, lun
```

---

## Profiling

```idl
; Time a specific operation
t0 = SYSTIME(1)
; ... operation to time ...
t1 = SYSTIME(1)
PRINT, FORMAT='("Elapsed: ", F8.3, " seconds")', t1 - t0

; IDL Profiler (commercial IDL)
PROFILER, /SYSTEM    ; Start profiling
; ... run your code ...
PROFILER, /REPORT    ; Print profiling report
PROFILER, /RESET     ; Reset counters
```

---

## Summary

| Tool | Description |
|------|-------------|
| `STOP` | Insert breakpoint |
| `.CONTINUE` | Resume after STOP |
| `.STEP` | Execute one line |
| `HELP, var` | Inspect variable |
| `RETALL` | Return to main level |
| `.COMPILE` | Recompile a routine |
| `RESOLVE_ALL` | Compile all dependencies |
| `CATCH` | Structured error handling |
| `MESSAGE` | Raise error/info messages |
| `HEAP_GC` | Garbage collect heap variables |
| `TEMPORARY()` | Reuse memory |
| `SYSTIME(1)` | Timing measurements |

### Best Practices Checklist

- Vectorize operations instead of looping
- Use descriptive variable and routine names
- Add header documentation to every routine
- Check WHERE results before using indices
- Always FREE_LUN after opening files
- Use DOUBLE for precision-critical calculations
- Free pointers and objects when done
- Test with edge cases (empty arrays, NaN, zero division)

---

**Previous**: [Date and Time](./13_Date_and_Time.md) | **Next**: [Project: Solar Light Curve](./15_Project_Solar_Light_Curve.md)
