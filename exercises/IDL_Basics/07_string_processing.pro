; Exercise 07: String Processing
;
; Practice string manipulation, formatting, and regex.

; Exercise 1: Parse a FITS-style date string 'YYYY-MM-DDTHH:MM:SS.sss'
; into individual components and print them.
PRO exercise_07a
  date_str = '2024-07-15T14:30:45.123'
  ; TODO: Extract year, month, day, hour, minute, second using STRMID
  ; TODO: Print each component
  ; TODO: Also try using STRSPLIT with '-T:.' as delimiters
END

; Exercise 2: Write a function that creates a formatted filename
; from components: instrument, wavelength, date, and sequence number.
; Example: 'aia_171_20240715_001.fits'
FUNCTION exercise_07b, instrument, wavelength, date_str, seq_num
  ; TODO: Build filename from components
  ; TODO: Ensure wavelength is zero-padded to 4 digits
  ; TODO: Ensure seq_num is zero-padded to 3 digits
  ; Hint: STRING(wavelength, FORMAT='(I04)')
  RETURN, ''
END

; Exercise 3: Use STREGEX to extract all numbers from a string
; and compute their sum.
FUNCTION exercise_07c, text
  ; TODO: Use STREGEX with a loop to find all numbers
  ; TODO: Convert each to float and sum them
  ; Hint: Use STRPOS to track position, STREGEX with /EXTRACT
  ; Example: 'The temp is 293.15 K at altitude 500 m' -> 793.15
  RETURN, 0.0
END

; Exercise 4: Write a simple CSV line formatter that takes
; a structure and returns a comma-separated string of its values.
FUNCTION exercise_07d, s
  ; TODO: Loop over TAG_NAMES to get each field
  ; TODO: Convert each value to string with STRTRIM
  ; TODO: Join with commas using STRJOIN
  ; Hint: Access fields by tag number: s.(i)
  RETURN, ''
END
