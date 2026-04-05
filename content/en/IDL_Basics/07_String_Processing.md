# String Processing

**Previous**: [Procedures and Functions](./06_Procedures_and_Functions.md) | **Next**: [File I/O](./08_File_IO.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Manipulate strings with STRMID, STRPOS, STRTRIM, STRLEN
2. Split and join strings with STRSPLIT and STRJOIN
3. Change case with STRUPCASE and STRLOWCASE
4. Format output with the STRING function and FORMAT keyword
5. Use printf-style formatting codes
6. Parse strings with READS
7. Apply regular expressions with STREGEX

---

String processing is essential for parsing data files, formatting output, constructing filenames, and working with FITS headers. IDL provides a comprehensive set of string functions.

## Basic String Operations

### String Creation and Properties

```idl
; String assignment
s = 'Hello, World!'
s2 = "Double quotes also work"
empty = ''

; String length
PRINT, STRLEN(s)         ;       13
PRINT, STRLEN(empty)     ;        0

; String arrays
names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
PRINT, STRLEN(names)     ;        5       3       7       5       3
```

### STRTRIM — Remove Whitespace

```idl
s = '   Hello   '

; STRTRIM(string, flag)
; flag=0: trim trailing (right)
; flag=1: trim leading (left)
; flag=2: trim both sides

PRINT, '[' + STRTRIM(s, 0) + ']'   ; [   Hello]
PRINT, '[' + STRTRIM(s, 1) + ']'   ; [Hello   ]
PRINT, '[' + STRTRIM(s, 2) + ']'   ; [Hello]

; Common pattern: trim numeric string
n = 42
PRINT, '[' + STRING(n) + ']'              ; [      42]
PRINT, '[' + STRTRIM(STRING(n), 2) + ']'  ; [42]
; Shortcut:
PRINT, '[' + STRTRIM(n, 2) + ']'          ; [42]
```

### STRMID — Extract Substring

```idl
s = 'Hello, World!'

; STRMID(string, start_position [, length])
PRINT, STRMID(s, 0, 5)     ; Hello
PRINT, STRMID(s, 7, 5)     ; World
PRINT, STRMID(s, 7)        ; World!  (to end of string)

; Extract date components from 'YYYY-MM-DD'
date_str = '2024-07-15'
year = STRMID(date_str, 0, 4)
month = STRMID(date_str, 5, 2)
day = STRMID(date_str, 8, 2)
PRINT, 'Year:', year, ' Month:', month, ' Day:', day
```

### STRPOS — Find Substring Position

```idl
s = 'The quick brown fox jumps over the lazy dog'

; STRPOS(string, search_string [, start_position])
PRINT, STRPOS(s, 'fox')        ;       16
PRINT, STRPOS(s, 'cat')        ;       -1  (not found)

; Search from a specific position
PRINT, STRPOS(s, 'the')        ;       -1  (case-sensitive! 'the' != 'The')
PRINT, STRPOS(STRLOWCASE(s), 'the')  ;        0

; Find second occurrence
first = STRPOS(s, 'the')
; Since case-sensitive and 'The' is at 0, search lowercase:
sl = STRLOWCASE(s)
first = STRPOS(sl, 'the', 0)    ;        0
second = STRPOS(sl, 'the', first + 1)  ;       31

; Reverse search (find last occurrence)
PRINT, STRPOS(sl, 'the', /REVERSE_SEARCH)  ;       31
```

### STRPUT — Replace Substring In Place

```idl
s = 'Hello, World!'
STRPUT, s, 'IDL', 7
PRINT, s             ; Hello, IDL!d!
; Note: STRPUT overwrites characters starting at position 7
; The string length does not change
```

---

## Case Conversion

```idl
s = 'Hello, World!'

PRINT, STRUPCASE(s)     ; HELLO, WORLD!
PRINT, STRLOWCASE(s)    ; hello, world!

; Works on arrays
names = ['alice', 'bob', 'charlie']
PRINT, STRUPCASE(names)  ; ALICE  BOB  CHARLIE
```

---

## String Splitting and Joining

### STRSPLIT

```idl
; Split by whitespace (default)
s = 'The quick brown fox'
words = STRSPLIT(s, ' ', /EXTRACT)
PRINT, words
; The
; quick
; brown
; fox

; Without /EXTRACT, returns indices
idx = STRSPLIT(s, ' ')
PRINT, idx               ;        0       4      10      16

; Split CSV data
csv_line = 'Alice,30,Engineer,New York'
fields = STRSPLIT(csv_line, ',', /EXTRACT)
PRINT, fields
; Alice
; 30
; Engineer
; New York

; Split with multiple delimiters
data = 'x=10; y=20; z=30'
parts = STRSPLIT(data, '=; ', /EXTRACT)
PRINT, parts             ; x  10  y  20  z  30

; Keep count
parts = STRSPLIT(data, ';', /EXTRACT, COUNT=n_parts)
PRINT, 'Number of parts:', n_parts    ;        3

; Split with regex
s = 'value1   value2     value3'
parts = STRSPLIT(s, ' +', /EXTRACT, /REGEX)
PRINT, parts             ; value1  value2  value3
```

### STRJOIN

```idl
; Join array elements into a single string
words = ['Hello', 'World', 'from', 'IDL']

PRINT, STRJOIN(words)         ; HelloWorldfromIDL
PRINT, STRJOIN(words, ' ')    ; Hello World from IDL
PRINT, STRJOIN(words, ', ')   ; Hello, World, from, IDL
PRINT, STRJOIN(words, ' | ')  ; Hello | World | from | IDL

; Build a CSV line
values = STRTRIM([10, 20, 30, 40, 50], 2)
csv = STRJOIN(values, ',')
PRINT, csv                    ; 10,20,30,40,50
```

---

## String Formatting with STRING and FORMAT

### The STRING Function

```idl
; Convert numbers to strings
PRINT, STRING(42)                  ;       42
PRINT, STRING(3.14159)             ;      3.14159
PRINT, STRING(42B)                 ; *   (ASCII character 42 = '*')

; With FORMAT keyword
PRINT, STRING(42, FORMAT='(I6)')          ;    42
PRINT, STRING(3.14159, FORMAT='(F8.4)')   ;  3.1416
PRINT, STRING(1.23e10, FORMAT='(E12.4)')  ;  1.2300E+10
```

### Printf-Style Format Codes

IDL uses Fortran-style format codes:

| Code | Description | Example | Output |
|------|-------------|---------|--------|
| `I` | Integer | `FORMAT='(I5)'` | `   42` |
| `F` | Fixed-point float | `FORMAT='(F8.3)'` | `   3.142` |
| `E` | Scientific notation | `FORMAT='(E12.4)'` | ` 3.1416E+00` |
| `G` | General float | `FORMAT='(G12.5)'` | `  3.1416` |
| `D` | Double precision | `FORMAT='(D15.8)'` | `  3.14159265` |
| `A` | String | `FORMAT='(A10)'` | `     Hello` |
| `X` | Spaces | `FORMAT='(5X)'` | `     ` |
| `/` | Newline | `FORMAT='(A, /, A)'` | Two lines |

```idl
; Formatted output examples
PRINT, FORMAT='("Name: ", A-15, " Age: ", I3)', 'Alice', 30
; Name: Alice            Age:  30

PRINT, FORMAT='("Pi = ", F10.7)', !PI
; Pi =  3.1415927

PRINT, FORMAT='("Value = ", E12.5, " +/- ", E12.5)', 1.234D5, 5.678D2
; Value =  1.23400E+05 +/-  5.67800E+02

; Repeat count
PRINT, FORMAT='(5(I4, :, ", "))', INDGEN(5)
;    0,    1,    2,    3,    4

; Right-justified and left-justified
PRINT, FORMAT='("|", A10, "|")', 'right'    ; |     right|
PRINT, FORMAT='("|", A-10, "|")', 'left'    ; |left      |
```

### READS — Parse String into Variables

```idl
; READS parses a string into variables (like READF for strings)
line = '42 3.14 Hello'

a = 0
b = 0.0
c = ''
READS, line, a, b, c
PRINT, 'a =', a, ' b =', b, ' c = ', c
; a =      42 b =      3.14000 c = Hello

; With FORMAT
line = '2024-07-15 14:30:00'
year = 0 & month = 0 & day = 0
hour = 0 & minute = 0 & second = 0
READS, line, year, month, day, hour, minute, second, $
  FORMAT='(I4, 1X, I2, 1X, I2, 1X, I2, 1X, I2, 1X, I2)'
PRINT, year, month, day, hour, minute, second
```

---

## Regular Expressions with STREGEX

### Basic Pattern Matching

```idl
; STREGEX(string, pattern [, /BOOLEAN] [, /EXTRACT])
s = 'The temperature is 293.15 K'

; Check if pattern matches
PRINT, STREGEX(s, '[0-9]+', /BOOLEAN)     ;        1

; Extract the match
match = STREGEX(s, '[0-9]+\.?[0-9]*', /EXTRACT)
PRINT, match                                ; 293.15

; Get position and length
pos = STREGEX(s, '[0-9]+\.?[0-9]*', LENGTH=len)
PRINT, 'Position:', pos, ' Length:', len    ; Position: 19  Length: 6
```

### Subexpressions (Capture Groups)

```idl
; Use parentheses for capture groups
s = 'Date: 2024-07-15'
; Capture year, month, day separately
result = STREGEX(s, '([0-9]{4})-([0-9]{2})-([0-9]{2})', /SUBEXPR, /EXTRACT)
PRINT, 'Full match:', result[0]    ; 2024-07-15
PRINT, 'Year:', result[1]          ; 2024
PRINT, 'Month:', result[2]         ; 07
PRINT, 'Day:', result[3]           ; 15
```

### STREGEX on Arrays

```idl
; Apply to string array
filenames = ['data_20240715.fits', 'dark_20240715.fits', $
             'flat_20240716.fits', 'data_20240716.fits', $
             'readme.txt']

; Find FITS files
is_fits = STREGEX(filenames, '\.fits$', /BOOLEAN)
PRINT, is_fits           ;        1       1       1       1       0

fits_idx = WHERE(is_fits, count)
PRINT, 'FITS files:', count
FOR i = 0, count - 1 DO PRINT, '  ', filenames[fits_idx[i]]

; Extract dates from filenames
dates = STREGEX(filenames, '([0-9]{8})', /SUBEXPR, /EXTRACT)
; dates is a 2D array: [2, n_files]
FOR i = 0, N_ELEMENTS(filenames) - 1 DO BEGIN
  IF dates[0, i] NE '' THEN PRINT, filenames[i], ' -> ', dates[1, i]
ENDFOR
```

### Common Patterns

```idl
; Email validation
email = 'user@example.com'
valid = STREGEX(email, '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', /BOOLEAN)
PRINT, 'Valid email:', valid

; FITS keyword extraction
header_line = "NAXIS1  =                 4096 / Number of pixels"
parts = STREGEX(header_line, '([A-Z0-9_]+)\s*=\s*([^ /]+)', /SUBEXPR, /EXTRACT)
PRINT, 'Keyword:', STRTRIM(parts[1], 2)
PRINT, 'Value:', STRTRIM(parts[2], 2)

; IP address pattern
ip = '192.168.1.100'
valid_ip = STREGEX(ip, '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$', /BOOLEAN)
```

---

## String Replacement

IDL does not have a built-in string replace function, but you can create one:

```idl
; Simple string replacement using STRJOIN and STRSPLIT
FUNCTION str_replace, source, search, replace
  parts = STRSPLIT(source, search, /EXTRACT, /REGEX, /PRESERVE_NULL)
  RETURN, STRJOIN(parts, replace)
END
```

```idl
s = 'Hello World Hello IDL'
PRINT, str_replace(s, 'Hello', 'Hi')    ; Hi World Hi IDL
PRINT, str_replace(s, ' ', '_')          ; Hello_World_Hello_IDL
```

### STRCOMPRESS — Remove Repeated Spaces

```idl
s = 'This   has    many     spaces'
PRINT, STRCOMPRESS(s)              ; This has many spaces (compress multiples to single)
PRINT, STRCOMPRESS(s, /REMOVE_ALL) ; Thishasmanyspaces (remove all spaces)
```

---

## Practical Examples

### Parsing FITS Header Lines

```idl
PRO parse_fits_header, header
  ; header is a string array from READFITS
  n_lines = N_ELEMENTS(header)

  FOR i = 0, n_lines - 1 DO BEGIN
    line = header[i]

    ; Skip blank lines and END
    IF STRTRIM(line, 2) EQ '' THEN CONTINUE
    IF STRMID(line, 0, 3) EQ 'END' THEN BREAK

    ; Skip COMMENT and HISTORY
    keyword = STRMID(line, 0, 8)
    IF STRTRIM(keyword, 2) EQ 'COMMENT' THEN CONTINUE
    IF STRTRIM(keyword, 2) EQ 'HISTORY' THEN CONTINUE

    ; Extract keyword and value
    eq_pos = STRPOS(line, '=')
    IF eq_pos GT 0 THEN BEGIN
      key = STRTRIM(STRMID(line, 0, eq_pos), 2)
      rest = STRMID(line, eq_pos + 1)

      ; Remove comment (after /)
      slash_pos = STRPOS(rest, '/')
      IF slash_pos GT 0 THEN BEGIN
        value = STRTRIM(STRMID(rest, 0, slash_pos), 2)
        comment = STRTRIM(STRMID(rest, slash_pos + 1), 2)
      ENDIF ELSE BEGIN
        value = STRTRIM(rest, 2)
        comment = ''
      ENDELSE

      PRINT, FORMAT='(A-10, " = ", A-20, " ; ", A)', key, value, comment
    ENDIF
  ENDFOR
END
```

### Building File Paths

```idl
; Construct observation file paths
base_dir = '/data/solar/aia/'
wavelengths = ['171', '193', '211', '304']
date = '2024/07/15'

FOR i = 0, N_ELEMENTS(wavelengths) - 1 DO BEGIN
  filepath = base_dir + date + '/aia_' + wavelengths[i] + '_*.fits'
  PRINT, 'Pattern:', filepath

  ; Find matching files
  files = FILE_SEARCH(filepath, COUNT=n_files)
  PRINT, '  Found:', n_files, ' files'
ENDFOR
```

---

## Summary

| Function | Description |
|----------|-------------|
| `STRLEN(s)` | String length |
| `STRMID(s, pos, len)` | Extract substring |
| `STRPOS(s, search)` | Find substring position |
| `STRTRIM(s, flag)` | Remove whitespace (0=right, 1=left, 2=both) |
| `STRSPLIT(s, delim, /EXTRACT)` | Split string into array |
| `STRJOIN(arr, sep)` | Join array into string |
| `STRUPCASE(s)` | Convert to uppercase |
| `STRLOWCASE(s)` | Convert to lowercase |
| `STRCOMPRESS(s)` | Remove extra spaces |
| `STRING(val, FORMAT=fmt)` | Format value as string |
| `READS, s, vars` | Parse string into variables |
| `STREGEX(s, pattern)` | Regular expression matching |
| `STRPUT, s, repl, pos` | Overwrite characters in place |

---

**Previous**: [Procedures and Functions](./06_Procedures_and_Functions.md) | **Next**: [File I/O](./08_File_IO.md)
