; 05 Control Flow
; ===============
; Demonstrates IF/THEN/ELSE, FOR, WHILE, REPEAT,
; CASE, BREAK, CONTINUE.

PRO example_05_control_flow

  ; IF/THEN/ELSE
  x = 15
  IF x GT 10 THEN BEGIN
    PRINT, 'x is greater than 10'
  ENDIF ELSE BEGIN
    PRINT, 'x is 10 or less'
  ENDELSE

  ; FOR loop
  PRINT, '--- FOR loop: squares ---'
  FOR i = 0, 5 DO PRINT, FORMAT='("  ", I2, "^2 = ", I4)', i, i^2

  ; FOR with step
  PRINT, '--- Even numbers ---'
  FOR i = 0, 10, 2 DO PRINT, FORMAT='("  ", I2)', i

  ; WHILE loop: Newton's method for sqrt(2)
  guess = 1.0D0
  WHILE ABS(guess^2 - 2.0D0) GT 1.0D-12 DO $
    guess = 0.5D0 * (guess + 2.0D0 / guess)
  PRINT, FORMAT='("sqrt(2) = ", F18.15)', guess

  ; REPEAT/UNTIL
  count = 0
  REPEAT BEGIN
    count += 1
  ENDREP UNTIL count GE 5
  PRINT, 'REPEAT count:', count

  ; CASE
  day = 3
  CASE day OF
    1: PRINT, 'Monday'
    2: PRINT, 'Tuesday'
    3: PRINT, 'Wednesday'
    ELSE: PRINT, 'Other day'
  ENDCASE

  ; BREAK and CONTINUE
  PRINT, '--- Odd numbers (CONTINUE) ---'
  FOR i = 0, 9 DO BEGIN
    IF i MOD 2 EQ 0 THEN CONTINUE
    PRINT, FORMAT='("  ", I2)', i
  ENDFOR

  PRINT, 'Example 05 complete.'
END
