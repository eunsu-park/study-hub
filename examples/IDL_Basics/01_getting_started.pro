; 01 Getting Started with IDL
; ===========================
; Demonstrates basic IDL commands: PRINT, variables,
; simple arithmetic, and system variables.

PRO example_01_getting_started

  ; Hello World
  PRINT, 'Hello, World!'
  PRINT, 'Welcome to IDL programming!'

  ; Basic arithmetic
  a = 10
  b = 3
  PRINT, 'a =', a, '  b =', b
  PRINT, 'a + b =', a + b
  PRINT, 'a * b =', a * b
  PRINT, 'a / b =', a / b, '  (integer division)'
  PRINT, 'FLOAT(a) / b =', FLOAT(a) / b

  ; System variables
  PRINT, 'Pi =', !PI
  PRINT, 'Degrees to radians:', !DTOR
  PRINT, 'IDL Version:', !VERSION.RELEASE
  PRINT, 'OS:', !VERSION.OS

  ; Simple array and statistics
  data = [4.5, 3.2, 7.8, 1.1, 9.6, 5.3]
  PRINT, 'Data:', data
  PRINT, 'Sum:', TOTAL(data)
  PRINT, 'Mean:', MEAN(data)
  PRINT, 'Sorted:', data[SORT(data)]

  ; Quick function test
  x = FINDGEN(10)
  PRINT, 'FINDGEN(10):', x
  PRINT, 'SIN of first 5:', SIN(x[0:4])

  PRINT, 'Example 01 complete.'
END
