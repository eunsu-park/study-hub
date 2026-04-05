# Procedures and Functions

**Previous**: [Control Flow](./05_Control_Flow.md) | **Next**: [String Processing](./07_String_Processing.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define and call procedures with the PRO keyword
2. Define and call functions with the FUNCTION keyword
3. Use positional parameters and keyword parameters
4. Check parameters with KEYWORD_SET, N_PARAMS, and N_ELEMENTS
5. Pass extra keywords with _EXTRA and _REF_EXTRA
6. Compile programs with .COMPILE and .RUN
7. Understand variable scope and COMMON blocks
8. Use RESOLVE_ALL for dependency checking

---

Procedures and functions are the fundamental building blocks for organizing IDL code. A procedure performs an action, while a function computes and returns a value. Understanding how to write, call, and parameterize these routines is essential for any non-trivial IDL program.

## Procedures

### Defining a Procedure

```idl
; hello.pro
PRO hello
  PRINT, 'Hello, World!'
END
```

```idl
IDL> .COMPILE hello
IDL> hello
Hello, World!
```

### Procedures with Positional Parameters

```idl
; greet.pro
PRO greet, name, title
  IF N_PARAMS() EQ 0 THEN name = 'World'
  IF N_PARAMS() LT 2 THEN title = ''

  IF STRLEN(title) GT 0 THEN BEGIN
    PRINT, 'Hello, ' + title + ' ' + name + '!'
  ENDIF ELSE BEGIN
    PRINT, 'Hello, ' + name + '!'
  ENDELSE
END
```

```idl
IDL> greet
Hello, World!
IDL> greet, 'Alice'
Hello, Alice!
IDL> greet, 'Smith', 'Dr.'
Hello, Dr. Smith!
```

### Procedures with Keyword Parameters

```idl
; print_stats.pro
PRO print_stats, data, VERBOSE=verbose, TITLE=title, OUTPUT=output
  ; Default values
  IF ~KEYWORD_SET(title) THEN title = 'Statistics'

  mn = MEAN(data)
  sd = STDDEV(data)
  med = MEDIAN(data)
  mn_val = MIN(data, min_idx)
  mx_val = MAX(data, max_idx)

  PRINT, '=== ' + title + ' ==='
  PRINT, FORMAT='("  Mean:    ", G12.5)', mn
  PRINT, FORMAT='("  Std Dev: ", G12.5)', sd
  PRINT, FORMAT='("  Median:  ", G12.5)', med
  PRINT, FORMAT='("  Min:     ", G12.5, " at index ", I0)', mn_val, min_idx
  PRINT, FORMAT='("  Max:     ", G12.5, " at index ", I0)', mx_val, max_idx

  IF KEYWORD_SET(verbose) THEN BEGIN
    PRINT, FORMAT='("  N:       ", I0)', N_ELEMENTS(data)
    PRINT, FORMAT='("  Total:   ", G12.5)', TOTAL(data)
    PRINT, FORMAT='("  Variance:", G12.5)', VARIANCE(data)
  ENDIF

  ; Return statistics through an output keyword
  output = {mean: mn, stddev: sd, median: med, min: mn_val, max: mx_val}
END
```

```idl
IDL> data = RANDOMN(seed, 100)
IDL> print_stats, data, TITLE='Random Data', /VERBOSE, OUTPUT=stats
IDL> HELP, stats, /STRUCTURE
```

---

## Functions

### Defining a Function

```idl
; circle_area.pro
FUNCTION circle_area, radius
  RETURN, !PI * radius^2
END
```

```idl
IDL> .COMPILE circle_area
IDL> area = circle_area(5.0)
IDL> PRINT, area
      78.5398
```

### Functions with Multiple Parameters

```idl
; distance.pro
FUNCTION distance, x1, y1, x2, y2
  RETURN, SQRT((x2 - x1)^2 + (y2 - y1)^2)
END
```

```idl
IDL> d = distance(0.0, 0.0, 3.0, 4.0)
IDL> PRINT, d
      5.00000
```

### Functions with Keywords

```idl
; normalize.pro
FUNCTION normalize, data, MIN_VAL=min_val, MAX_VAL=max_val, $
                    MEAN_CENTER=mean_center
  result = DOUBLE(data)

  IF KEYWORD_SET(mean_center) THEN BEGIN
    ; Zero-mean normalization
    result = result - MEAN(result)
    result = result / STDDEV(result)
  ENDIF ELSE BEGIN
    ; Min-max normalization
    IF N_ELEMENTS(min_val) EQ 0 THEN min_val = MIN(result)
    IF N_ELEMENTS(max_val) EQ 0 THEN max_val = MAX(result)
    range = max_val - min_val
    IF range EQ 0 THEN RETURN, REPLICATE(0.0D, N_ELEMENTS(data))
    result = (result - min_val) / range
  ENDELSE

  RETURN, result
END
```

```idl
IDL> data = [10.0, 20.0, 30.0, 40.0, 50.0]
IDL> PRINT, normalize(data)
;      0.00000     0.250000     0.500000     0.750000      1.00000
IDL> PRINT, normalize(data, /MEAN_CENTER)
;     -1.26491    -0.632456     0.000000     0.632456      1.26491
```

---

## Parameter Handling

### N_PARAMS

`N_PARAMS()` returns the number of positional parameters passed to a routine:

```idl
PRO flexible_proc, a, b, c, d
  np = N_PARAMS()
  PRINT, 'Number of parameters:', np

  CASE np OF
    0: PRINT, 'No parameters'
    1: PRINT, 'One parameter: a =', a
    2: PRINT, 'Two parameters: a =', a, ' b =', b
    3: PRINT, 'Three parameters: a, b, c'
    4: PRINT, 'All four parameters'
    ELSE: ; Cannot happen for this routine
  ENDCASE
END
```

### KEYWORD_SET

`KEYWORD_SET` returns 1 if a keyword is defined AND non-zero:

```idl
PRO example, data, VERBOSE=verbose, PLOT=do_plot, COUNT=count
  ; KEYWORD_SET returns 1 if keyword is set and non-zero
  ; Returns 0 if keyword is not set, or set to 0, or undefined

  IF KEYWORD_SET(verbose) THEN PRINT, 'Verbose mode on'
  IF KEYWORD_SET(do_plot) THEN PRINT, 'Will create plot'

  ; For keywords that may have a value of 0, use N_ELEMENTS instead
  IF N_ELEMENTS(count) GT 0 THEN BEGIN
    PRINT, 'Count was specified:', count
  ENDIF ELSE BEGIN
    count = 10  ; Default
    PRINT, 'Using default count:', count
  ENDELSE
END
```

**Important distinction**:
- `KEYWORD_SET(kw)` — True if keyword is set and non-zero. Fails for valid value of 0.
- `N_ELEMENTS(kw) GT 0` — True if keyword was passed (even if value is 0). Preferred for numeric keywords.

### Checking for Undefined Parameters

```idl
FUNCTION safe_divide, a, b, DEFAULT=default
  ; N_ELEMENTS returns 0 for undefined variables
  IF N_ELEMENTS(a) EQ 0 OR N_ELEMENTS(b) EQ 0 THEN BEGIN
    PRINT, 'Error: Both arguments required'
    RETURN, !VALUES.F_NAN
  ENDIF

  IF N_ELEMENTS(default) EQ 0 THEN default = !VALUES.F_NAN

  ; Handle division by zero
  zero_idx = WHERE(b EQ 0, n_zero)
  result = FLOAT(a) / FLOAT(b)
  IF n_zero GT 0 THEN result[zero_idx] = default

  RETURN, result
END
```

---

## Keyword Inheritance: _EXTRA and _REF_EXTRA

### _EXTRA

`_EXTRA` collects unrecognized keywords into a structure and passes them to called routines:

```idl
; my_plot.pro — wrapper around PLOT with custom defaults
PRO my_plot, x, y, TITLE=title, _EXTRA=extra
  ; Set defaults
  IF ~KEYWORD_SET(title) THEN title = 'My Custom Plot'

  ; Pass unrecognized keywords through to PLOT
  PLOT, x, y, TITLE=title, $
    CHARSIZE=1.5, THICK=2, $
    _EXTRA=extra
END
```

```idl
IDL> x = FINDGEN(100) / 10.0
IDL> y = SIN(x)
IDL> my_plot, x, y, LINESTYLE=2, COLOR=255, XRANGE=[0,5]
; LINESTYLE, COLOR, XRANGE are passed through _EXTRA to PLOT
```

### _REF_EXTRA

`_REF_EXTRA` passes keywords by reference, allowing the called routine to return values through them:

```idl
; wrapper.pro
PRO wrapper, data, RESULT=result, _REF_EXTRA=extra
  ; _REF_EXTRA allows called routines to modify keyword values
  ; that are then visible to the caller
  process_data, data, OUTPUT=result, _EXTRA=extra
END
```

---

## Compilation and Execution

### .COMPILE vs .RUN

```idl
; .COMPILE — compile only, do not execute
IDL> .COMPILE my_routine
% Compiled module: MY_ROUTINE.

; .RUN — compile and execute main-level programs
; (For named routines, .RUN just compiles like .COMPILE)
IDL> .RUN main_script

; Compile and run a function/procedure explicitly
IDL> .COMPILE circle_area
IDL> result = circle_area(5.0)
```

### Automatic Compilation

IDL can automatically find and compile routines if they are on the `!PATH`:

```idl
; If circle_area.pro is in a directory on !PATH,
; IDL compiles it automatically when first called:
IDL> area = circle_area(10.0)
% Compiled module: CIRCLE_AREA.
      314.159
```

### RESOLVE_ALL

`RESOLVE_ALL` compiles all unresolved procedures and functions:

```idl
; Before creating a .sav file or distributing code
IDL> RESOLVE_ALL
; This ensures all dependencies are compiled

; Resolve for a specific routine
IDL> RESOLVE_ROUTINE, 'my_routine', /EITHER
```

### RESOLVE_ROUTINE

```idl
; Compile a specific routine
RESOLVE_ROUTINE, 'plot_data'           ; Procedure
RESOLVE_ROUTINE, 'calc_mean', /IS_FUNCTION  ; Function
RESOLVE_ROUTINE, 'helper', /EITHER     ; Either procedure or function
```

---

## Variable Scope

### Local Scope

Variables defined inside a procedure or function are local by default:

```idl
PRO scope_demo
  local_var = 42
  PRINT, 'Inside procedure: local_var =', local_var
END
```

```idl
IDL> scope_demo
Inside procedure: local_var =       42
IDL> PRINT, local_var
% PRINT: Variable is undefined: LOCAL_VAR.
```

### Parameters Are Passed by Reference

IDL passes parameters by reference, so modifications inside a routine affect the caller:

```idl
PRO double_it, x
  x = x * 2
END
```

```idl
IDL> a = 5
IDL> double_it, a
IDL> PRINT, a
      10
; a was modified because it was passed by reference
```

**Exception**: Expressions and constants are passed by value:

```idl
IDL> double_it, 5        ; 5 is a constant — not modified
IDL> double_it, a + 1    ; a + 1 is an expression — not modified
```

### COMMON Blocks

COMMON blocks allow variables to be shared between routines without passing them as parameters:

```idl
; init_config.pro
PRO init_config
  COMMON config_block, data_dir, verbose_flag, max_iter
  data_dir = '/data/solar/'
  verbose_flag = 1
  max_iter = 100
  PRINT, 'Configuration initialized'
END

; process_data.pro
PRO process_data
  COMMON config_block, data_dir, verbose_flag, max_iter

  IF verbose_flag THEN PRINT, 'Data directory:', data_dir
  PRINT, 'Max iterations:', max_iter

  ; Process data...
  FOR i = 1, max_iter DO BEGIN
    IF verbose_flag THEN PRINT, 'Iteration:', i
    ; ... processing code ...
    BREAK  ; Just for demonstration
  ENDFOR
END
```

```idl
IDL> init_config
Configuration initialized
IDL> process_data
Data directory: /data/solar/
Max iterations:      100
Iteration:       1
```

**Caution**: COMMON blocks create hidden coupling between routines. Modern IDL code should prefer passing data through parameters or using structures/objects instead.

---

## Multiple Routines in One File

Although the convention is one routine per file, you can define helper routines in the same file. They must appear **before** the main routine:

```idl
; solar_analysis.pro
; Helper function (defined first)
FUNCTION compute_temperature, wavelength, intensity
  h = 6.626D-34   ; Planck constant
  c = 3.0D8        ; Speed of light
  k = 1.381D-23    ; Boltzmann constant
  RETURN, h * c / (wavelength * k * ALOG(2.0D * h * c^2 / (wavelength^5 * intensity) + 1.0D))
END

; Helper procedure
PRO print_separator, char, width
  IF N_ELEMENTS(char) EQ 0 THEN char = '-'
  IF N_ELEMENTS(width) EQ 0 THEN width = 60
  PRINT, STRJOIN(REPLICATE(char, width))
END

; Main procedure (last in file, name matches filename)
PRO solar_analysis, wavelengths, intensities
  print_separator, '='
  PRINT, 'Solar Spectrum Analysis'
  print_separator, '='

  n = N_ELEMENTS(wavelengths)
  FOR i = 0, n - 1 DO BEGIN
    temp = compute_temperature(wavelengths[i], intensities[i])
    PRINT, FORMAT='("  Lambda = ", E10.3, " m  ->  T = ", F8.0, " K")', $
      wavelengths[i], temp
  ENDFOR

  print_separator
END
```

---

## Practical Examples

### Modular Data Pipeline

```idl
; read_solar_data.pro
FUNCTION read_solar_data, filename, HEADER=header, STATUS=status
  status = 0
  ON_IOERROR, read_error

  IF ~FILE_TEST(filename) THEN BEGIN
    PRINT, 'File not found: ' + filename
    status = -1
    RETURN, !NULL
  ENDIF

  ; Read data (simplified)
  data = READFITS(filename, header)
  status = 1
  RETURN, data

  read_error:
  PRINT, 'Error reading: ' + filename
  status = -2
  RETURN, !NULL
END

; process_solar_data.pro
FUNCTION process_solar_data, data, DARK=dark, FLAT=flat, $
                              NORMALIZE=normalize
  result = DOUBLE(data)

  ; Dark subtraction
  IF N_ELEMENTS(dark) GT 0 THEN result = result - dark

  ; Flat field correction
  IF N_ELEMENTS(flat) GT 0 THEN BEGIN
    good = WHERE(flat NE 0, count)
    IF count GT 0 THEN result[good] = result[good] / flat[good]
  ENDIF

  ; Normalize
  IF KEYWORD_SET(normalize) THEN BEGIN
    max_val = MAX(result)
    IF max_val GT 0 THEN result = result / max_val
  ENDIF

  RETURN, result
END

; plot_solar_data.pro
PRO plot_solar_data, data, TITLE=title, COLORBAR=colorbar, _EXTRA=extra
  IF ~KEYWORD_SET(title) THEN title = 'Solar Image'

  TVSCL, data
  XYOUTS, 0.5, 0.95, title, /NORMAL, ALIGNMENT=0.5, CHARSIZE=2.0

  IF KEYWORD_SET(colorbar) THEN BEGIN
    ; Add a simple colorbar
    PRINT, 'Range: [', MIN(data), ', ', MAX(data), ']'
  ENDIF
END
```

### Statistical Function Library

```idl
; weighted_mean.pro
FUNCTION weighted_mean, values, weights, ERROR=error
  IF N_PARAMS() LT 2 THEN BEGIN
    PRINT, 'Usage: result = weighted_mean(values, weights)'
    RETURN, !VALUES.F_NAN
  ENDIF

  IF N_ELEMENTS(values) NE N_ELEMENTS(weights) THEN BEGIN
    PRINT, 'Error: values and weights must have same length'
    RETURN, !VALUES.F_NAN
  ENDIF

  w_sum = TOTAL(weights)
  IF w_sum EQ 0 THEN RETURN, !VALUES.F_NAN

  wmean = TOTAL(values * weights) / w_sum

  ; Weighted standard error
  IF ARG_PRESENT(error) THEN BEGIN
    wvar = TOTAL(weights * (values - wmean)^2) / w_sum
    error = SQRT(wvar / N_ELEMENTS(values))
  ENDIF

  RETURN, wmean
END
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `PRO name` | Define a procedure (no return value) |
| `FUNCTION name` | Define a function (returns a value with RETURN) |
| Positional parameters | Passed by position in the call |
| Keyword parameters | Named with `KEY=value` syntax |
| `N_PARAMS()` | Number of positional parameters passed |
| `KEYWORD_SET(kw)` | True if keyword is set and non-zero |
| `N_ELEMENTS(kw) GT 0` | True if keyword was passed (any value) |
| `_EXTRA` | Collect/pass unrecognized keywords (by value) |
| `_REF_EXTRA` | Collect/pass unrecognized keywords (by reference) |
| `.COMPILE` | Compile a file without executing |
| `.RUN` | Compile and execute main-level programs |
| `RESOLVE_ALL` | Compile all unresolved dependencies |
| COMMON | Shared variable block (use sparingly) |
| Pass by reference | Parameters are references; modifications affect caller |

---

**Previous**: [Control Flow](./05_Control_Flow.md) | **Next**: [String Processing](./07_String_Processing.md)
