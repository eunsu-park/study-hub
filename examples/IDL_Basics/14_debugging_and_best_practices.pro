; 14 Debugging and Best Practices
; ================================
; Demonstrates CATCH error handling, TEMPORARY,
; vectorization benchmarks, and common pitfalls.

PRO example_14_debugging

  ; Error handling with CATCH
  PRINT, '--- CATCH Error Handling ---'
  CATCH, error_status
  IF error_status NE 0 THEN BEGIN
    PRINT, 'Caught error: ' + !ERROR_STATE.MSG
    CATCH, /CANCEL
    GOTO, skip_error
  ENDIF
  ; This will cause an error (divide by zero in integer)
  ; result = 1 / 0   ; Uncomment to test
  CATCH, /CANCEL
  skip_error:

  ; Vectorization benchmark
  PRINT, '--- Vectorization Benchmark ---'
  n = 100000L
  data = RANDOMN(seed, n)

  ; Loop version
  t0 = SYSTIME(1)
  result_loop = FLTARR(n)
  FOR i = 0L, n - 1 DO BEGIN
    IF data[i] GT 0 THEN result_loop[i] = SQRT(data[i]) $
    ELSE result_loop[i] = 0.0
  ENDFOR
  t_loop = SYSTIME(1) - t0

  ; Vectorized version
  t0 = SYSTIME(1)
  result_vec = FLTARR(n)
  pos = WHERE(data GT 0, count)
  IF count GT 0 THEN result_vec[pos] = SQRT(data[pos])
  t_vec = SYSTIME(1) - t0

  PRINT, FORMAT='("  Loop time:       ", F8.4, " seconds")', t_loop
  PRINT, FORMAT='("  Vectorized time: ", F8.4, " seconds")', t_vec
  PRINT, FORMAT='("  Speedup:         ", F6.1, "x")', t_loop / (t_vec > 1e-6)

  ; TEMPORARY example
  PRINT, '--- TEMPORARY ---'
  big = FLTARR(1000, 1000)
  HELP, big
  result = TEMPORARY(big) * 2.0
  HELP, result
  ; big is now undefined

  ; Common pitfall: WHERE check
  PRINT, '--- WHERE Safety ---'
  data = [1, 2, 3]
  idx = WHERE(data GT 100, count)
  IF count GT 0 THEN $
    PRINT, 'Found:', data[idx] $
  ELSE $
    PRINT, 'No values > 100 (safe check)'

  PRINT, 'Example 14 complete.'
END
