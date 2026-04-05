; Exercise 08: File I/O
;
; Practice reading and writing text and binary files.

; Exercise 1: Write a procedure that creates a data file with
; 100 rows of (x, sin(x), cos(x)) with a header line.
PRO exercise_08a, filename
  IF N_PARAMS() EQ 0 THEN filename = 'trig_data.txt'
  ; TODO: Open file for writing with /GET_LUN
  ; TODO: Write header: '# x  sin(x)  cos(x)'
  ; TODO: Loop x from 0 to 2*pi in 100 steps
  ; TODO: Write each row with FORMAT
  ; TODO: FREE_LUN
END

; Exercise 2: Write a function that reads the file from Exercise 1
; and returns the data as a structure {x: array, sinx: array, cosx: array}.
FUNCTION exercise_08b, filename
  IF N_PARAMS() EQ 0 THEN filename = 'trig_data.txt'
  ; TODO: Count lines with FILE_LINES
  ; TODO: Open, skip header, read data
  ; TODO: Return structure with three arrays
  ; Hint: Subtract 1 for header line
  RETURN, !NULL
END

; Exercise 3: Save multiple variables to a .sav file, then restore
; them and verify the values match.
PRO exercise_08c
  ; TODO: Create x = DINDGEN(50), y = SIN(x/5), label = 'test data'
  ; TODO: SAVE to 'exercise08.sav'
  ; TODO: Clear variables (x = !NULL etc.)
  ; TODO: RESTORE 'exercise08.sav'
  ; TODO: Verify N_ELEMENTS(x) EQ 50
  ; TODO: Clean up the .sav file
END

; Exercise 4: Write a procedure that safely reads a file,
; handling the case where the file does not exist.
PRO exercise_08d, filename
  ; TODO: Check FILE_TEST first
  ; TODO: If not found, print error and RETURN
  ; TODO: Use ON_IOERROR or CATCH for read errors
  ; TODO: Read and print file contents
  ; TODO: Always FREE_LUN even if error occurs
END
