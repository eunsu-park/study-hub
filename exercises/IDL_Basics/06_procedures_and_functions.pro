; Exercise 06: Procedures and Functions
;
; Practice writing procedures and functions with keywords.

; Exercise 1: Write a function that computes the distance between
; two points in 2D or 3D space using keyword parameters.
FUNCTION exercise_06a, x1, y1, x2, y2, Z1=z1, Z2=z2
  ; TODO: Compute 2D distance by default
  ; TODO: If Z1 and Z2 are provided, compute 3D distance
  ; TODO: Use N_ELEMENTS to check for z keywords
  ; Hint: dist = SQRT((x2-x1)^2 + (y2-y1)^2 [+ (z2-z1)^2])
  RETURN, 0.0
END

; Exercise 2: Write a procedure that prints a formatted table header
; and data rows using keyword parameters for column names and widths.
PRO exercise_06b, data, COLUMNS=columns, TITLE=title
  ; TODO: Default title = 'Data Table'
  ; TODO: Default columns = ['Col1', 'Col2', 'Col3']
  ; TODO: Print title, separator line, column headers, data rows
  ; Hint: Use STRJOIN for headers, FORMAT for data
END

; Exercise 3: Write a wrapper around PLOT that sets default
; thick=2, charsize=1.3, and passes all other keywords through.
PRO exercise_06c, x, y, _EXTRA=extra
  ; TODO: Call PLOT with x, y, THICK=2, CHARSIZE=1.3, _EXTRA=extra
  ; Hint: Just one line is needed
END

; Exercise 4: Write a function that normalizes data to [0,1] range
; with optional MIN_VAL and MAX_VAL keywords for clipping.
FUNCTION exercise_06d, data, MIN_VAL=min_val, MAX_VAL=max_val
  ; TODO: If MIN_VAL not provided, use MIN(data)
  ; TODO: If MAX_VAL not provided, use MAX(data)
  ; TODO: Return (data - min_val) / (max_val - min_val)
  ; TODO: Handle case where min_val == max_val
  ; Hint: Use N_ELEMENTS to check keyword presence
  RETURN, data
END
