; 02 Variables and Data Types
; ===========================
; Demonstrates IDL data types, type conversion,
; variable inspection, and special values.

PRO example_02_variables

  ; Numeric types
  b = 255B             ; BYTE
  i = 42               ; INT
  l = 100000L          ; LONG
  f = 3.14             ; FLOAT
  d = 3.14159265D0     ; DOUBLE
  z = COMPLEX(3.0, 4.0); COMPLEX

  PRINT, '--- Type Inspection ---'
  HELP, b, i, l, f, d, z

  ; Type conversion
  PRINT, '--- Type Conversion ---'
  PRINT, 'FIX(3.7) =', FIX(3.7)
  PRINT, 'FLOAT(42) =', FLOAT(42)
  PRINT, 'DOUBLE(!PI) =', DOUBLE(!PI)
  PRINT, 'STRING(42) = [', STRTRIM(STRING(42), 2), ']'
  PRINT, 'BYTE("A") =', BYTE('A')

  ; SIZE function
  arr = FLTARR(100, 200)
  PRINT, '--- SIZE ---'
  PRINT, 'N_DIMENSIONS:', SIZE(arr, /N_DIMENSIONS)
  PRINT, 'DIMENSIONS:', SIZE(arr, /DIMENSIONS)
  PRINT, 'TYPE NAME:', SIZE(arr, /TNAME)
  PRINT, 'N_ELEMENTS:', N_ELEMENTS(arr)

  ; Type promotion
  PRINT, '--- Type Promotion ---'
  r1 = 5 + 3.0
  HELP, r1       ; FLOAT
  r2 = 3.14 + 1.0D0
  HELP, r2       ; DOUBLE

  ; Special values
  PRINT, '--- Special Values ---'
  PRINT, 'NaN:', !VALUES.F_NAN
  PRINT, 'Infinity:', !VALUES.F_INFINITY
  PRINT, 'FINITE(42.0):', FINITE(42.0)
  PRINT, 'FINITE(NaN):', FINITE(!VALUES.F_NAN)

  ; Complex arithmetic
  z1 = COMPLEX(3.0, 4.0)
  PRINT, '--- Complex ---'
  PRINT, 'z =', z1
  PRINT, '|z| =', ABS(z1)
  PRINT, 'Real:', REAL_PART(z1)
  PRINT, 'Imag:', IMAGINARY(z1)

  PRINT, 'Example 02 complete.'
END
