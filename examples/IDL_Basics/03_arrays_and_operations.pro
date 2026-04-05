; 03 Arrays and Operations
; ========================
; Demonstrates array creation, indexing, WHERE function,
; array arithmetic, and manipulation.

PRO example_03_arrays

  ; Array creation
  a = INDGEN(10)
  b = FINDGEN(5)
  c = FLTARR(3, 4)
  d = MAKE_ARRAY(5, VALUE=99.0)
  PRINT, 'INDGEN(10):', a
  PRINT, 'FINDGEN(5):', b
  PRINT, 'MAKE_ARRAY:', d

  ; 2D array
  arr2d = INDGEN(4, 3)
  PRINT, '--- 2D Array ---'
  PRINT, arr2d

  ; Indexing
  PRINT, '--- Indexing ---'
  data = [10, 20, 30, 40, 50]
  PRINT, 'data[0]:', data[0]
  PRINT, 'data[2:4]:', data[2:4]
  PRINT, 'data[-1]:', data[-1]

  ; WHERE function
  PRINT, '--- WHERE ---'
  values = [3, 7, 1, 9, 4, 6, 2, 8, 5]
  idx = WHERE(values GT 5, count)
  PRINT, 'Values > 5:', values[idx], '  Count:', count

  ; NaN filtering
  mixed = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
  good = WHERE(FINITE(mixed), n_good)
  PRINT, 'Good values:', mixed[good]

  ; Array arithmetic
  x = FINDGEN(5) + 1.0
  PRINT, '--- Array Arithmetic ---'
  PRINT, 'x:', x
  PRINT, 'x^2:', x^2
  PRINT, 'SQRT(x):', SQRT(x)
  PRINT, 'TOTAL:', TOTAL(x), '  MEAN:', MEAN(x)

  ; REFORM
  flat = INDGEN(12)
  reshaped = REFORM(flat, 3, 4)
  PRINT, '--- REFORM ---'
  PRINT, reshaped

  ; SORT
  unsorted = [3, 1, 4, 1, 5, 9, 2]
  PRINT, 'Sorted:', unsorted[SORT(unsorted)]

  PRINT, 'Example 03 complete.'
END
