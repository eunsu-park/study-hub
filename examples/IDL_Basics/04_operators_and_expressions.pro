; 04 Operators and Expressions
; ============================
; Demonstrates arithmetic, relational, logical operators,
; min/max operators, and ternary expressions.

PRO example_04_operators

  ; Arithmetic
  PRINT, '--- Arithmetic ---'
  PRINT, '17 MOD 5 =', 17 MOD 5
  PRINT, '2^10 =', 2^10
  PRINT, '7/2 =', 7/2, '  (integer division)'
  PRINT, '7.0/2.0 =', 7.0/2.0

  ; Relational
  PRINT, '--- Relational ---'
  PRINT, '10 EQ 20:', 10 EQ 20
  PRINT, '10 LT 20:', 10 LT 20
  PRINT, '10 GE 10:', 10 GE 10

  ; Min/Max operators
  PRINT, '--- Min/Max Operators ---'
  PRINT, '5 < 3 =', 5 < 3
  PRINT, '5 > 3 =', 5 > 3
  x = 150
  PRINT, 'Clamp 150 to [0,100]:', 0 > x < 100

  ; Logical with arrays
  PRINT, '--- Logical with Arrays ---'
  data = [1, 5, 3, 8, 2, 7, 4, 9]
  idx = WHERE(data GE 3 AND data LE 7, count)
  PRINT, 'Between 3 and 7:', data[idx]

  ; Ternary operator
  PRINT, '--- Ternary ---'
  score = 85
  grade = (score GE 90) ? 'A' : ((score GE 80) ? 'B' : 'C')
  PRINT, 'Score:', score, '  Grade:', grade

  ; String concatenation
  PRINT, '--- String ---'
  msg = 'Hello' + ' ' + 'IDL' + '!'
  PRINT, msg

  PRINT, 'Example 04 complete.'
END
