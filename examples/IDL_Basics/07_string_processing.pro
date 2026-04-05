; 07 String Processing
; ====================
; Demonstrates string functions: STRMID, STRPOS, STRSPLIT,
; STRJOIN, FORMAT, and STREGEX.

PRO example_07_strings

  s = 'Hello, World!'
  PRINT, 'String:', s
  PRINT, 'Length:', STRLEN(s)
  PRINT, 'Upper:', STRUPCASE(s)
  PRINT, 'Lower:', STRLOWCASE(s)
  PRINT, 'Substr [0:5]:', STRMID(s, 0, 5)
  PRINT, 'Pos of World:', STRPOS(s, 'World')

  ; Trim
  padded = '   IDL   '
  PRINT, 'Trimmed: [' + STRTRIM(padded, 2) + ']'

  ; Split and join
  csv = 'Alice,30,Engineer,NYC'
  fields = STRSPLIT(csv, ',', /EXTRACT)
  PRINT, '--- CSV Fields ---'
  FOR i = 0, N_ELEMENTS(fields) - 1 DO PRINT, '  ', fields[i]
  PRINT, 'Rejoined:', STRJOIN(fields, ' | ')

  ; FORMAT
  PRINT, '--- Formatting ---'
  PRINT, FORMAT='("Pi = ", F10.7)', !PI
  PRINT, FORMAT='("Name: ", A-10, " Score: ", I5)', 'Alice', 95

  ; READS
  line = '42 3.14 Hello'
  a = 0 & b = 0.0 & c = ''
  READS, line, a, b, c
  PRINT, FORMAT='("Parsed: a=", I0, " b=", F5.2, " c=", A)', a, b, c

  ; STREGEX
  date = 'Observed: 2024-07-15T14:30:00'
  parts = STREGEX(date, '([0-9]{4})-([0-9]{2})-([0-9]{2})', /SUBEXPR, /EXTRACT)
  PRINT, 'Year:', parts[1], '  Month:', parts[2], '  Day:', parts[3]

  PRINT, 'Example 07 complete.'
END
