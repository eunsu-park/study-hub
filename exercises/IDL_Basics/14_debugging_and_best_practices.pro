; Exercise 14: Debugging and Best Practices
;
; Practice error handling, vectorization, and code quality.

; Exercise 1: Rewrite this loop-based code using vectorized operations.
; Measure the speedup.
PRO exercise_14a
  n = 50000L
  seed = 42L
  data = RANDOMN(seed, n) * 100.0

  ; SLOW: Loop version (given)
  t0 = SYSTIME(1)
  result = FLTARR(n)
  FOR i = 0L, n - 1 DO BEGIN
    IF data[i] GT 0 AND data[i] LT 50 THEN result[i] = data[i]^2 $
    ELSE result[i] = 0.0
  ENDFOR
  t_loop = SYSTIME(1) - t0
  PRINT, 'Loop time:', t_loop

  ; TODO: Write the vectorized version
  ; TODO: Use WHERE to find indices where data GT 0 AND data LT 50
  ; TODO: Compute result_vec using array operations
  ; TODO: Time it and print the speedup
END

; Exercise 2: Write a procedure with proper CATCH error handling
; that attempts to read a file and processes it.
PRO exercise_14b, filename
  ; TODO: Set up CATCH
  ; TODO: If error, print message and return gracefully
  ; TODO: Check FILE_TEST first
  ; TODO: Read and print first 5 lines
  ; TODO: Cancel CATCH when done
END

; Exercise 3: Fix the bugs in this procedure (there are 4 bugs).
PRO exercise_14c_buggy, data
  ; Bug 1: Integer division
  half = N_ELEMENTS(data) / 2
  mean_val = TOTAL(data) / N_ELEMENTS(data)

  ; Bug 2: Missing BEGIN/END
  IF mean_val GT 0 THEN
    PRINT, 'Positive mean'
    scaled = data / mean_val

  ; Bug 3: WHERE without count check
  big = WHERE(data GT 1000)
  PRINT, 'Big values:', data[big]

  ; Bug 4: Not freeing LUN
  OPENW, lun, 'temp.txt', /GET_LUN
  PRINTF, lun, 'test'
  ; Missing: FREE_LUN, lun
END

; TODO: Write exercise_14c_fixed that fixes all 4 bugs
PRO exercise_14c_fixed, data
  ; TODO: Fix all bugs from exercise_14c_buggy
END

; Exercise 4: Write a well-documented function following IDL
; conventions with header comments, input validation, and error handling.
FUNCTION exercise_14d, wavelength, temperature
  ; TODO: Add a proper IDL documentation header (NAME, PURPOSE, INPUTS, etc.)
  ; TODO: Validate inputs (check N_PARAMS, check for positive values)
  ; TODO: Compute the Planck function B(lambda, T)
  ; TODO: B = 2*h*c^2 / (lambda^5 * (exp(h*c/(lambda*k*T)) - 1))
  ; TODO: Return the result
  RETURN, 0.0D0
END
