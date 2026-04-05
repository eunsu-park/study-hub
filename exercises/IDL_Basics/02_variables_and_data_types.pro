; Exercise 02: Variables and Data Types
;
; Practice creating variables, type conversion, and inspection.

; Exercise 1: Create variables of each numeric type and use HELP
; to verify their types.
PRO exercise_02a
  ; TODO: Create one variable of each type:
  ;   BYTE, INT, LONG, FLOAT, DOUBLE, COMPLEX
  ; TODO: Use HELP to display all of them
  ; Hint: Use suffixes like B, L, D0 or conversion functions
END

; Exercise 2: Demonstrate the precision difference between
; FLOAT and DOUBLE by computing 1/3 and printing with many digits.
PRO exercise_02b
  ; TODO: Compute result_float = 1.0 / 3.0
  ; TODO: Compute result_double = 1.0D0 / 3.0D0
  ; TODO: Print both with FORMAT showing 15 decimal places
  ; Hint: FORMAT='("Float:  ", F18.15)' and FORMAT='("Double: ", D18.15)'
END

; Exercise 3: Create an array with some NaN values, filter them out,
; and compute the mean of the valid values.
PRO exercise_02c
  ; TODO: Create data = [1.0, !VALUES.F_NAN, 3.0, 4.0, !VALUES.F_NAN, 6.0]
  ; TODO: Use WHERE and FINITE to find valid elements
  ; TODO: Print the mean of valid elements only
  ; Hint: good = WHERE(FINITE(data), count)
END

; Exercise 4: Write a function that takes any numeric input
; and returns its type name and byte size.
FUNCTION exercise_02d, x
  ; TODO: Return a structure with fields 'typename' and 'bytes'
  ; TODO: Use SIZE(x, /TNAME) for type name
  ; TODO: Use SIZE(x, /TYPE) and a lookup for byte size
  ; Hint: RETURN, {typename: ..., bytes: ...}
  RETURN, !NULL
END
