# File I/O

**Previous**: [String Processing](./07_String_Processing.md) | **Next**: [Structures](./09_Structures.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Open files with OPENR, OPENW, OPENU and manage logical units with GET_LUN/FREE_LUN
2. Read and write text files with READF and PRINTF
3. Read and write binary files with READU and WRITEU
4. Use POINT_LUN to navigate within files and EOF to detect end-of-file
5. Save and restore IDL variables with SAVE/RESTORE
6. Read structured text files with READ_ASCII and READ_CSV
7. Search for files with FILE_SEARCH and test file properties with FILE_TEST

---

File I/O is a core skill for scientific computing. Whether you are reading observational data, writing processed results, or managing configuration files, IDL provides a flexible set of I/O routines.

## Opening and Closing Files

### OPENR, OPENW, OPENU

```idl
; OPENR — Open for Reading
; OPENW — Open for Writing (creates or truncates)
; OPENU — Open for Updating (read and write, existing file)

; Basic syntax: OPENx, logical_unit_number, filename

; Method 1: Specify a logical unit number (1-128)
OPENR, 1, 'data.txt'
; ... read data ...
CLOSE, 1

; Method 2: Let IDL assign a logical unit number (preferred)
OPENR, lun, 'data.txt', /GET_LUN
; ... read data ...
FREE_LUN, lun
```

### GET_LUN and FREE_LUN

Using `/GET_LUN` is the recommended approach because it avoids conflicts when multiple routines use files simultaneously:

```idl
; GET_LUN assigns an available logical unit
OPENR, lun, 'input.txt', /GET_LUN
PRINT, 'Assigned LUN:', lun    ; e.g., 100

; Always free the LUN when done
FREE_LUN, lun   ; Closes and frees the logical unit

; Alternatively, get a LUN first
GET_LUN, lun
OPENW, lun, 'output.txt'
; ... write data ...
FREE_LUN, lun
```

---

## Reading Text Files

### READF — Read Formatted Data

```idl
; Read a simple text file line by line
OPENR, lun, 'data.txt', /GET_LUN
line = ''
WHILE ~EOF(lun) DO BEGIN
  READF, lun, line
  PRINT, line
ENDWHILE
FREE_LUN, lun
```

### Reading Structured Data

```idl
; data.txt contains:
; 1  2.5  100.0
; 2  3.7  200.0
; 3  4.2  150.0

; Read into variables
n = 3
x = INTARR(n)
y = FLTARR(n)
z = FLTARR(n)

OPENR, lun, 'data.txt', /GET_LUN
FOR i = 0, n - 1 DO BEGIN
  READF, lun, xi, yi, zi
  x[i] = xi
  y[i] = yi
  z[i] = zi
ENDFOR
FREE_LUN, lun

; Or read an entire row at once
OPENR, lun, 'data.txt', /GET_LUN
row = FLTARR(3)
FOR i = 0, n - 1 DO BEGIN
  READF, lun, row
  PRINT, row
ENDFOR
FREE_LUN, lun
```

### Reading with FORMAT

```idl
; Precise control over parsing
; data.txt:
; Alice     30  98.5
; Bob       25  87.3
; Charlie   35  92.1

name = ''
age = 0
score = 0.0

OPENR, lun, 'data.txt', /GET_LUN
WHILE ~EOF(lun) DO BEGIN
  READF, lun, FORMAT='(A10, I5, F6.1)', name, age, score
  PRINT, STRTRIM(name, 2), age, score
ENDWHILE
FREE_LUN, lun
```

### Skipping Header Lines

```idl
; File has 3 header lines followed by data
OPENR, lun, 'data_with_header.txt', /GET_LUN

; Skip header
header = ''
FOR i = 0, 2 DO READF, lun, header

; Read data
line = ''
WHILE ~EOF(lun) DO BEGIN
  READF, lun, line
  ; Process data line
  values = FLOAT(STRSPLIT(line, /EXTRACT))
  PRINT, values
ENDWHILE
FREE_LUN, lun
```

---

## Writing Text Files

### PRINTF — Print Formatted

```idl
; Write data to a text file
OPENW, lun, 'output.txt', /GET_LUN

; Simple output
PRINTF, lun, 'This is line 1'
PRINTF, lun, 'This is line 2'

; Formatted output
x = FINDGEN(10)
y = SIN(x)
PRINTF, lun, '# X        Y'
FOR i = 0, N_ELEMENTS(x) - 1 DO BEGIN
  PRINTF, lun, FORMAT='(F8.3, "  ", F10.6)', x[i], y[i]
ENDFOR

FREE_LUN, lun
```

### Writing CSV Files

```idl
; Write data as CSV
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 35]
scores = [98.5, 87.3, 92.1]

OPENW, lun, 'people.csv', /GET_LUN
PRINTF, lun, 'Name,Age,Score'
FOR i = 0, N_ELEMENTS(names) - 1 DO BEGIN
  PRINTF, lun, names[i] + ',' + STRTRIM(ages[i], 2) + ',' + $
    STRTRIM(STRING(scores[i], FORMAT='(F5.1)'), 2)
ENDFOR
FREE_LUN, lun
```

### Appending to Files

```idl
; Open existing file for appending
OPENU, lun, 'log.txt', /GET_LUN, /APPEND
PRINTF, lun, SYSTIME() + ': New log entry'
FREE_LUN, lun
```

---

## Binary File I/O

### WRITEU — Write Unformatted (Binary)

```idl
; Write binary data
data = FINDGEN(1000)
OPENW, lun, 'data.bin', /GET_LUN
WRITEU, lun, data
FREE_LUN, lun

; Write multiple variables
nx = 512L
ny = 512L
image = FLTARR(nx, ny)
header_str = 'Image data v1.0'

OPENW, lun, 'image.bin', /GET_LUN
WRITEU, lun, STRLEN(header_str)
WRITEU, lun, BYTE(header_str)
WRITEU, lun, nx, ny
WRITEU, lun, image
FREE_LUN, lun
```

### READU — Read Unformatted (Binary)

```idl
; Read binary data — must know the exact format
data = FLTARR(1000)
OPENR, lun, 'data.bin', /GET_LUN
READU, lun, data
FREE_LUN, lun
PRINT, 'Read', N_ELEMENTS(data), 'values'

; Read the complex binary file
header_len = 0L
OPENR, lun, 'image.bin', /GET_LUN
READU, lun, header_len
header_bytes = BYTARR(header_len)
READU, lun, header_bytes
header_str = STRING(header_bytes)
PRINT, 'Header:', header_str

nx = 0L & ny = 0L
READU, lun, nx, ny
PRINT, 'Dimensions:', nx, ny

image = FLTARR(nx, ny)
READU, lun, image
FREE_LUN, lun
PRINT, 'Image range:', MIN(image), MAX(image)
```

### POINT_LUN — File Position

```idl
; Move to a specific position in a file
OPENR, lun, 'data.bin', /GET_LUN

; Get current position
POINT_LUN, -lun, current_pos
PRINT, 'Current position:', current_pos

; Seek to a specific byte offset
POINT_LUN, lun, 1000L   ; Move to byte 1000

; Read 100 floats starting at byte 1000
subset = FLTARR(100)
READU, lun, subset

FREE_LUN, lun
```

### EOF — End of File

```idl
; Read until end of file
OPENR, lun, 'data.txt', /GET_LUN
line = ''
count = 0L
WHILE ~EOF(lun) DO BEGIN
  READF, lun, line
  count = count + 1
ENDWHILE
PRINT, 'Total lines:', count
FREE_LUN, lun
```

---

## SAVE and RESTORE

IDL's SAVE and RESTORE provide a convenient way to save any IDL variable to a `.sav` file:

### SAVE

```idl
; Save specific variables
x = FINDGEN(100)
y = SIN(x / 10.0)
metadata = {date: SYSTIME(), n_points: 100, source: 'simulation'}
SAVE, x, y, metadata, FILENAME='results.sav'

; Save all variables in the current session
SAVE, /ALL, FILENAME='session.sav'

; Save with description
SAVE, x, y, FILENAME='data.sav', DESCRIPTION='Sine wave data'

; Save compiled routines
SAVE, /ROUTINES, FILENAME='my_library.sav'
```

### RESTORE

```idl
; Restore all variables from a .sav file
RESTORE, 'results.sav'
PRINT, 'x has', N_ELEMENTS(x), 'elements'
PRINT, 'Metadata:', metadata.date

; Restore with verbose output
RESTORE, 'results.sav', /VERBOSE
; % RESTORE: Restored variable: X.
; % RESTORE: Restored variable: Y.
; % RESTORE: Restored variable: METADATA.

; Check what's in a .sav file without restoring
RESTORE, 'results.sav', /VERBOSE, RESTORED_OBJECTS=obj_names
```

---

## High-Level File Reading

### READ_ASCII

```idl
; Read a structured text file into a structure
; data.txt:
; # X  Y  Z
; 1.0  2.0  3.0
; 4.0  5.0  6.0
; 7.0  8.0  9.0

; Define template (or use ASCII_TEMPLATE() interactively)
template = {version: 1.0, $
            datastart: 1L, $         ; Skip 1 header line
            delimiter: 32B, $        ; Space delimiter
            missingvalue: !VALUES.F_NAN, $
            commentsymbol: '#', $
            fieldcount: 3L, $
            fieldtypes: [4, 4, 4], $ ; FLOAT
            fieldnames: ['X', 'Y', 'Z'], $
            fieldlocations: [0, 5, 10], $
            fieldgroups: [0, 1, 2]}

data = READ_ASCII('data.txt', TEMPLATE=template)
PRINT, data.X
PRINT, data.Y
PRINT, data.Z
```

### READ_CSV

```idl
; Read a CSV file (IDL 8+)
; people.csv:
; Name,Age,Score
; Alice,30,98.5
; Bob,25,87.3

data = READ_CSV('people.csv', HEADER=header)
PRINT, 'Headers:', header
PRINT, 'Column 1:', data.FIELD1  ; Names
PRINT, 'Column 2:', data.FIELD2  ; Ages
PRINT, 'Column 3:', data.FIELD3  ; Scores
```

---

## File System Operations

### FILE_SEARCH

```idl
; Find files matching a pattern
fits_files = FILE_SEARCH('/data/solar/*.fits', COUNT=n_files)
PRINT, 'Found', n_files, 'FITS files'
FOR i = 0, (n_files < 5) - 1 DO PRINT, '  ', fits_files[i]

; Recursive search
all_pro = FILE_SEARCH('/home/user/idl/**/*.pro', COUNT=n_pro)
PRINT, 'Found', n_pro, '.pro files'

; Search with multiple patterns
data_files = FILE_SEARCH('/data/', '*.{fits,dat,csv}', COUNT=n)
```

### FILE_TEST

```idl
; Test file existence and properties
filename = '/data/solar/image.fits'

PRINT, FILE_TEST(filename)                ; 1 if exists
PRINT, FILE_TEST(filename, /READ)         ; 1 if readable
PRINT, FILE_TEST(filename, /WRITE)        ; 1 if writable
PRINT, FILE_TEST(filename, /DIRECTORY)    ; 1 if directory
PRINT, FILE_TEST(filename, /REGULAR)      ; 1 if regular file

; Use in error checking
IF ~FILE_TEST(filename) THEN BEGIN
  PRINT, 'Error: File not found: ' + filename
  RETURN
ENDIF
```

### FILE_MKDIR, FILE_DELETE, FILE_COPY

```idl
; Create directory
FILE_MKDIR, '/tmp/idl_output'
FILE_MKDIR, '/tmp/idl_output/images'  ; Creates parent if needed

; Copy files
FILE_COPY, 'source.dat', '/tmp/backup/source.dat'

; Delete files
FILE_DELETE, 'temp.dat', /ALLOW_NONEXISTENT  ; No error if missing

; Get file information
info = FILE_INFO('/data/solar/image.fits')
PRINT, 'Size:', info.SIZE, 'bytes'
PRINT, 'Modified:', SYSTIME(0, info.MTIME)
```

---

## Practical Examples

### Reading a Multi-Column Data File

```idl
PRO read_multicolumn, filename, time, flux, error
  ; Count lines first
  n_lines = FILE_LINES(filename)

  ; Subtract header lines
  n_header = 2
  n_data = n_lines - n_header

  ; Allocate arrays
  time = DBLARR(n_data)
  flux = DBLARR(n_data)
  error = DBLARR(n_data)

  ; Read data
  OPENR, lun, filename, /GET_LUN
  line = ''
  FOR i = 0, n_header - 1 DO READF, lun, line  ; Skip header

  t = 0.0D0 & f = 0.0D0 & e = 0.0D0
  FOR i = 0, n_data - 1 DO BEGIN
    READF, lun, t, f, e
    time[i] = t
    flux[i] = f
    error[i] = e
  ENDFOR
  FREE_LUN, lun

  PRINT, FORMAT='("Read ", I0, " data points from ", A)', n_data, filename
END
```

### Safe File I/O with Error Handling

```idl
FUNCTION safe_read_data, filename
  ; Check file existence
  IF ~FILE_TEST(filename, /READ) THEN BEGIN
    PRINT, 'Error: Cannot read file: ' + filename
    RETURN, !NULL
  ENDIF

  ; Set up I/O error handler
  ON_IOERROR, io_error

  OPENR, lun, filename, /GET_LUN

  ; Read header
  line = ''
  READF, lun, line
  n_cols = N_ELEMENTS(STRSPLIT(line, /EXTRACT))

  ; Count remaining lines
  n_lines = FILE_LINES(filename) - 1

  ; Read data
  data = DBLARR(n_cols, n_lines)
  row = DBLARR(n_cols)
  FOR i = 0, n_lines - 1 DO BEGIN
    READF, lun, row
    data[*, i] = row
  ENDFOR

  FREE_LUN, lun
  RETURN, data

  io_error:
  PRINT, 'I/O Error reading: ' + filename
  IF N_ELEMENTS(lun) GT 0 THEN FREE_LUN, lun
  RETURN, !NULL
END
```

---

## Summary

| Operation | Procedure/Function | Description |
|-----------|-------------------|-------------|
| Open for reading | `OPENR, lun, file, /GET_LUN` | Open existing file |
| Open for writing | `OPENW, lun, file, /GET_LUN` | Create/truncate file |
| Open for update | `OPENU, lun, file, /GET_LUN` | Read/write existing file |
| Close file | `FREE_LUN, lun` | Close and free LUN |
| Read text | `READF, lun, vars` | Read formatted text |
| Write text | `PRINTF, lun, vars` | Write formatted text |
| Read binary | `READU, lun, vars` | Read unformatted binary |
| Write binary | `WRITEU, lun, vars` | Write unformatted binary |
| File position | `POINT_LUN, lun, pos` | Seek to byte position |
| End of file | `EOF(lun)` | Test for end of file |
| Save variables | `SAVE, vars, FILENAME=f` | Save to .sav file |
| Restore variables | `RESTORE, filename` | Load from .sav file |
| Find files | `FILE_SEARCH(pattern)` | Search for files |
| Test file | `FILE_TEST(file)` | Check file properties |

---

**Previous**: [String Processing](./07_String_Processing.md) | **Next**: [Structures](./09_Structures.md)
