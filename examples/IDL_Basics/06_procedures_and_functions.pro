; 06 Procedures and Functions
; ===========================
; Demonstrates PRO, FUNCTION, keywords, N_PARAMS,
; KEYWORD_SET, and _EXTRA.

FUNCTION weighted_average, values, weights
  RETURN, TOTAL(values * weights) / TOTAL(weights)
END

PRO greet_user, name, FORMAL=formal
  IF N_PARAMS() EQ 0 THEN name = 'World'
  IF KEYWORD_SET(formal) THEN $
    PRINT, 'Good day, ' + name + '.' $
  ELSE $
    PRINT, 'Hello, ' + name + '!'
END

PRO compute_stats, data, TITLE=title, VERBOSE=verbose
  IF ~KEYWORD_SET(title) THEN title = 'Statistics'
  PRINT, '=== ' + title + ' ==='
  PRINT, FORMAT='("  Mean:   ", G12.5)', MEAN(data)
  PRINT, FORMAT='("  StdDev: ", G12.5)', STDDEV(data)
  PRINT, FORMAT='("  Min:    ", G12.5)', MIN(data)
  PRINT, FORMAT='("  Max:    ", G12.5)', MAX(data)
  IF KEYWORD_SET(verbose) THEN $
    PRINT, FORMAT='("  N:      ", I0)', N_ELEMENTS(data)
END

PRO example_06_procedures

  ; Call procedure with different args
  greet_user
  greet_user, 'Alice'
  greet_user, 'Dr. Smith', /FORMAL

  ; Call function
  vals = [80.0, 90.0, 70.0, 85.0]
  wts = [1.0, 2.0, 1.0, 1.5]
  avg = weighted_average(vals, wts)
  PRINT, FORMAT='("Weighted average: ", F6.2)', avg

  ; Keywords
  data = RANDOMN(seed, 200) * 10.0 + 50.0
  compute_stats, data, TITLE='Random Data', /VERBOSE

  PRINT, 'Example 06 complete.'
END
