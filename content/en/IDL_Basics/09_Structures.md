# Structures

**Previous**: [File I/O](./08_File_IO.md) | **Next**: [Basic Plotting](./10_Basic_Plotting.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create anonymous and named structures
2. Access structure fields by name and by tag number
3. Use CREATE_STRUCT to build structures dynamically
4. Work with structure arrays
5. Create nested structures
6. Inspect structures with TAG_NAMES, N_TAGS, and HELP
7. Understand structure inheritance

---

Structures in IDL are composite data types that group related data under a single variable. They are IDL's equivalent of C structs or Python dictionaries (but with fixed fields once created). Structures are used extensively for organizing metadata, passing multiple values between routines, and representing complex data records.

## Anonymous Structures

Anonymous structures have no type name and are the simplest form:

```idl
; Create an anonymous structure with curly braces
person = {name: 'Alice', age: 30, score: 98.5}

; Access fields with dot notation
PRINT, person.name       ; Alice
PRINT, person.age        ;       30
PRINT, person.score      ;      98.5000

; Inspect the structure
HELP, person
; PERSON          STRUCT    = -> <Anonymous> Array[1]

HELP, person, /STRUCTURE
; ** Structure <anonymous>, 3 tags, length=28:
;    NAME            STRING    'Alice'
;    AGE             INT             30
;    SCORE           FLOAT           98.5000
```

### Modifying Fields

```idl
; Modify a field value
person.name = 'Bob'
person.age = 25
PRINT, person.name, person.age    ; Bob      25

; You CANNOT add new fields to an existing structure
; person.email = 'bob@example.com'  ; ERROR!
```

---

## Named Structures

Named structures have a type name, and all instances share the same field layout:

```idl
; Define a named structure
star = {STAR, name: '', ra: 0.0D0, dec: 0.0D0, magnitude: 0.0, spectral_type: ''}

; Create instances of the same type
sirius = {STAR, name: 'Sirius', ra: 101.287D, dec: -16.716D, $
          magnitude: -1.46, spectral_type: 'A1V'}
vega = {STAR, name: 'Vega', ra: 279.235D, dec: 38.784D, $
        magnitude: 0.03, spectral_type: 'A0V'}

PRINT, sirius.name, sirius.magnitude
PRINT, vega.name, vega.magnitude
```

### Named Structure Rules

```idl
; Once defined, the field layout is fixed for the session
; All subsequent instances must have the same fields
star1 = {STAR}              ; Create with default values
star1.name = 'Betelgeuse'
star1.magnitude = 0.42

; This would cause an error because STAR is already defined:
; bad = {STAR, name: '', ra: 0.0D0, extra_field: 0}  ; ERROR!

; To redefine, you must restart IDL or use:
; .RESET_SESSION
```

---

## CREATE_STRUCT — Dynamic Construction

`CREATE_STRUCT` builds structures at runtime, useful when field names are determined dynamically:

```idl
; Build a structure field by field
s = CREATE_STRUCT('x', 1.0)
s = CREATE_STRUCT(s, 'y', 2.0)
s = CREATE_STRUCT(s, 'z', 3.0)
HELP, s, /STRUCTURE
; ** Structure <anonymous>, 3 tags:
;    X               FLOAT           1.00000
;    Y               FLOAT           2.00000
;    Z               FLOAT           3.00000

; Create from arrays of names and values
names = ['TEMP', 'PRESSURE', 'HUMIDITY']
values = [25.0, 1013.25, 0.65]

s = CREATE_STRUCT(names[0], values[0])
FOR i = 1, N_ELEMENTS(names) - 1 DO BEGIN
  s = CREATE_STRUCT(s, names[i], values[i])
ENDFOR
HELP, s, /STRUCTURE

; Create a named structure
obs = CREATE_STRUCT(NAME='OBSERVATION', 'time', 0.0D0, 'value', 0.0, 'flag', 0B)
HELP, obs, /STRUCTURE

; Merge two structures
s1 = {a: 1, b: 2}
s2 = {c: 3, d: 4}
merged = CREATE_STRUCT(s1, s2)
HELP, merged, /STRUCTURE
; a:1, b:2, c:3, d:4
```

---

## Structure Arrays

Arrays of structures are extremely useful for tabular data:

```idl
; Create a structure array
n = 5
stars = REPLICATE({STAR}, n)

; Fill in data
stars[0].name = 'Sirius'   & stars[0].magnitude = -1.46
stars[1].name = 'Canopus'  & stars[1].magnitude = -0.72
stars[2].name = 'Arcturus' & stars[2].magnitude = -0.04
stars[3].name = 'Vega'     & stars[3].magnitude = 0.03
stars[4].name = 'Capella'  & stars[4].magnitude = 0.08

; Access fields across the array
PRINT, stars.name         ; Array of names
PRINT, stars.magnitude    ; Array of magnitudes

; Sort by magnitude
idx = SORT(stars.magnitude)
sorted = stars[idx]
FOR i = 0, n-1 DO PRINT, sorted[i].name, sorted[i].magnitude

; Filter with WHERE
bright = WHERE(stars.magnitude LT 0, count)
PRINT, 'Stars brighter than 0:', count
FOR i = 0, count-1 DO PRINT, '  ', stars[bright[i]].name
```

### Creating Structure Arrays Efficiently

```idl
; Method 1: REPLICATE
template = {time: 0.0D0, flux: 0.0D0, error: 0.0D0, quality: 0B}
data = REPLICATE(template, 1000)

; Method 2: Array of predefined structures
obs = REPLICATE({OBSERVATION}, 500)

; Fill from arrays
n = 100
times = DINDGEN(n)
fluxes = RANDOMN(seed, n) + 10.0D0
errors = REPLICATE(0.1D0, n)

records = REPLICATE({time: 0.0D0, flux: 0.0D0, error: 0.0D0}, n)
records.time = times
records.flux = fluxes
records.error = errors
```

---

## Nested Structures

Structures can contain other structures as fields:

```idl
; Nested structure for astronomical observation
obs = {observation, $
       target: {name: '', ra: 0.0D0, dec: 0.0D0}, $
       instrument: {name: '', wavelength: 0.0, exposure: 0.0}, $
       data: {nx: 0L, ny: 0L, values: PTR_NEW()}, $
       timestamp: 0.0D0}

; Fill in nested fields
obs.target.name = 'Sun'
obs.target.ra = 0.0D0
obs.instrument.name = 'AIA'
obs.instrument.wavelength = 171.0
obs.instrument.exposure = 2.0
obs.data.nx = 4096L
obs.data.ny = 4096L
obs.timestamp = SYSTIME(/JULIAN)

PRINT, obs.target.name
PRINT, obs.instrument.wavelength
PRINT, obs.data.nx, 'x', obs.data.ny
```

---

## Accessing Fields by Tag Number

You can access structure fields by their index number using parentheses:

```idl
s = {name: 'Alice', age: 30, score: 98.5}

; Access by tag number (0-based)
PRINT, s.(0)     ; Alice   (same as s.name)
PRINT, s.(1)     ;    30   (same as s.age)
PRINT, s.(2)     ; 98.5    (same as s.score)

; Useful for iterating over all fields
FOR i = 0, N_TAGS(s) - 1 DO BEGIN
  PRINT, (TAG_NAMES(s))[i], ' = ', s.(i)
ENDFOR
```

---

## Inspecting Structures

### TAG_NAMES

```idl
s = {name: 'Alice', age: 30, score: 98.5}
tags = TAG_NAMES(s)
PRINT, tags          ; NAME  AGE  SCORE

; Check if a tag exists
target = 'AGE'
found = (WHERE(TAG_NAMES(s) EQ target))[0] NE -1
PRINT, 'Has AGE tag:', found

; Get the structure name (for named structures)
star = {STAR}
PRINT, TAG_NAMES(star, /STRUCTURE_NAME)   ; STAR
```

### N_TAGS

```idl
s = {a: 1, b: 2, c: 3, d: 4}
PRINT, N_TAGS(s)     ;        4

; Get data length in bytes
PRINT, N_TAGS(s, /DATA_LENGTH)
```

---

## Structure Inheritance

Named structures can inherit fields from a parent structure:

```idl
; Define a base structure
base = {CELESTIAL_OBJECT, name: '', ra: 0.0D0, dec: 0.0D0, distance: 0.0D0}

; Define derived structures that inherit base fields
star_def = {STAR_OBJ, INHERITS CELESTIAL_OBJECT, $
            magnitude: 0.0, spectral_type: '', mass: 0.0D0}

galaxy_def = {GALAXY_OBJ, INHERITS CELESTIAL_OBJECT, $
              morphology: '', redshift: 0.0D0, luminosity: 0.0D0}

; Create instances
my_star = {STAR_OBJ}
my_star.name = 'Proxima Centauri'
my_star.ra = 217.429D0
my_star.dec = -62.680D0
my_star.distance = 4.24D0
my_star.magnitude = 11.13
my_star.spectral_type = 'M5.5V'

HELP, my_star, /STRUCTURE
; Has all CELESTIAL_OBJECT fields plus STAR_OBJ-specific fields
```

---

## Practical Examples

### FITS Header as Structure

```idl
; Build a FITS-like header structure from keyword-value pairs
FUNCTION build_header_struct, keywords, values
  IF N_ELEMENTS(keywords) NE N_ELEMENTS(values) THEN BEGIN
    PRINT, 'Error: keywords and values must have same length'
    RETURN, !NULL
  ENDIF

  s = CREATE_STRUCT(keywords[0], values[0])
  FOR i = 1, N_ELEMENTS(keywords) - 1 DO BEGIN
    s = CREATE_STRUCT(s, keywords[i], values[i])
  ENDFOR

  RETURN, s
END

; Usage:
; keys = ['NAXIS1', 'NAXIS2', 'BITPIX', 'TELESCOP']
; vals = [4096L, 4096L, -32L, 'SDO/AIA']
; hdr = build_header_struct(keys, vals)
```

### Observation Database

```idl
PRO create_obs_database
  ; Define record structure
  template = {obs_record, $
              id: 0L, $
              date: '', $
              target: '', $
              instrument: '', $
              wavelength: 0.0, $
              exposure: 0.0, $
              quality: 0B, $
              filename: ''}

  ; Create sample database
  n_obs = 5
  db = REPLICATE({obs_record}, n_obs)

  db[0] = {obs_record, 1L, '2024-07-15', 'Sun', 'AIA', 171.0, 2.0, 1B, 'aia_171_001.fits'}
  db[1] = {obs_record, 2L, '2024-07-15', 'Sun', 'AIA', 193.0, 2.0, 1B, 'aia_193_001.fits'}
  db[2] = {obs_record, 3L, '2024-07-15', 'Sun', 'HMI', 6173.0, 3.5, 1B, 'hmi_mag_001.fits'}
  db[3] = {obs_record, 4L, '2024-07-16', 'Sun', 'AIA', 304.0, 2.0, 0B, 'aia_304_001.fits'}
  db[4] = {obs_record, 5L, '2024-07-16', 'Sun', 'AIA', 171.0, 2.0, 1B, 'aia_171_002.fits'}

  ; Query: Find all AIA observations
  aia_idx = WHERE(db.instrument EQ 'AIA', n_aia)
  PRINT, 'AIA observations:', n_aia
  FOR i = 0, n_aia - 1 DO BEGIN
    rec = db[aia_idx[i]]
    PRINT, FORMAT='("  ", I3, "  ", A10, "  ", F6.1, " A  ", A)', $
      rec.id, rec.date, rec.wavelength, rec.filename
  ENDFOR

  ; Query: Good quality only
  good = WHERE(db.quality EQ 1B, n_good)
  PRINT, 'Good quality observations:', n_good

  ; Save database
  SAVE, db, FILENAME='obs_database.sav'
END
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Anonymous structure | `s = {field1: val1, field2: val2}` |
| Named structure | `s = {TYPE_NAME, field1: val1, ...}` |
| Field access | `s.field_name` or `s.(tag_number)` |
| CREATE_STRUCT | Build structures dynamically |
| REPLICATE | Create structure arrays |
| Nested structures | Structures within structures |
| TAG_NAMES(s) | Get field name array |
| N_TAGS(s) | Get number of fields |
| INHERITS | Structure inheritance |
| `s.field = value` | Modify a field |

---

**Previous**: [File I/O](./08_File_IO.md) | **Next**: [Basic Plotting](./10_Basic_Plotting.md)
