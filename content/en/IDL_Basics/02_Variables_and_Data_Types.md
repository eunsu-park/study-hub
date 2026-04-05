# Variables and Data Types

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Arrays and Operations](./03_Arrays_and_Operations.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and assign variables of all IDL numeric types
2. Understand the full range of IDL data types (BYTE through COMPLEX)
3. Use type conversion functions (FIX, FLOAT, DOUBLE, STRING, BYTE)
4. Inspect variables with HELP, SIZE, and N_ELEMENTS
5. Check variable types using ISA and TYPENAME
6. Work with special values (NaN, Infinity, NULL)
7. Understand IDL's type promotion rules

---

IDL is a dynamically typed language. You do not declare variable types explicitly; instead, the type is determined by the value you assign. However, understanding data types is crucial because numerical precision, memory usage, and performance all depend on the types you choose.

## Variable Assignment

Variables in IDL are created by simple assignment with the `=` operator:

```idl
; Scalar variables
x = 42           ; Integer (INT, 16-bit signed)
y = 3.14         ; Floating point (FLOAT, 32-bit)
name = 'Alice'   ; String

; IDL variable names are case-insensitive
MyVar = 100
PRINT, myvar     ;       100
PRINT, MYVAR     ;       100
```

### Variable Naming Rules

- Names must start with a letter or underscore
- Can contain letters, digits, and underscores
- Maximum 255 characters
- Case-insensitive (`myVar` and `MYVAR` are the same variable)
- Cannot use IDL reserved words (AND, OR, NOT, EQ, NE, etc.)

```idl
; Valid names
count = 0
_temp = 3.14
data_2024 = [1, 2, 3]
longVariableName = 'OK'

; Invalid names (these will cause errors)
; 2data = 5        ; Cannot start with a digit
; my-var = 10      ; Hyphens not allowed
; for = 1          ; Reserved word
```

---

## IDL Data Types

IDL provides a rich set of data types optimized for scientific computing:

### Numeric Types

| Type | IDL Name | Bytes | Range | Creation Syntax |
|------|----------|-------|-------|-----------------|
| Byte | BYTE | 1 | 0 to 255 | `x = 0B` or `x = BYTE(42)` |
| Integer | INT | 2 | -32,768 to 32,767 | `x = 0` or `x = 0S` |
| Unsigned Integer | UINT | 2 | 0 to 65,535 | `x = 0U` or `x = UINT(42)` |
| Long | LONG | 4 | -2^31 to 2^31-1 | `x = 0L` |
| Unsigned Long | ULONG | 4 | 0 to 2^32-1 | `x = 0UL` |
| 64-bit Long | LONG64 | 8 | -2^63 to 2^63-1 | `x = 0LL` |
| Unsigned 64-bit | ULONG64 | 8 | 0 to 2^64-1 | `x = 0ULL` |
| Float | FLOAT | 4 | ~1.2e-38 to 3.4e38 | `x = 0.0` or `x = 0.0E0` |
| Double | DOUBLE | 8 | ~2.2e-308 to 1.8e308 | `x = 0.0D0` or `x = 0D` |
| Complex | COMPLEX | 8 | Two floats (real, imag) | `x = COMPLEX(1.0, 2.0)` |
| Double Complex | DCOMPLEX | 16 | Two doubles (real, imag) | `x = DCOMPLEX(1.0D, 2.0D)` |

### Non-Numeric Types

| Type | Description | Example |
|------|-------------|---------|
| STRING | Character string | `s = 'Hello'` |
| POINTER | Pointer to a heap variable | `p = PTR_NEW(42)` |
| OBJECT | Object reference | `obj = OBJ_NEW('classname')` |
| LIST | Ordered collection (IDL 8+) | `lst = LIST(1, 'a', 3.14)` |
| HASH | Key-value pairs (IDL 8+) | `h = HASH('key', 'value')` |

---

## Creating Variables of Specific Types

### Byte (BYTE)

Bytes are unsigned 8-bit integers (0-255), commonly used for image data:

```idl
; Using the B suffix
b1 = 0B
b2 = 255B

; Using the BYTE() function
b3 = BYTE(42)
b4 = BYTE('A')     ; ASCII value of 'A' = 65

HELP, b1, b2, b3, b4
; B1              BYTE      =    0
; B2              BYTE      =  255
; B3              BYTE      =   42
; B4              BYTE      =   65

; Byte arrays from strings
bytes = BYTE('Hello')
PRINT, bytes
;   72  101  108  108  111
```

### Integer (INT) and Long (LONG)

```idl
; Integer (16-bit signed, default for integer literals)
i = 42
i2 = -100
i3 = 0S          ; Explicit INT suffix

; Long (32-bit signed)
big = 100000L
big2 = -500000L

; Check the difference
HELP, i, big
; I               INT       =       42
; BIG             LONG      =       100000

; Integer overflow warning
x = 32767        ; Maximum INT value
PRINT, x + 1     ; Will promote to LONG or wrap depending on context
```

### Unsigned Types

```idl
; Unsigned integer
ui = 50000U
HELP, ui
; UI              UINT      =  50000

; Unsigned long
ul = 3000000000UL
HELP, ul
; UL              ULONG     = 3000000000

; 64-bit integers
big64 = 9000000000000LL
ubig64 = 18000000000000ULL
HELP, big64, ubig64
; BIG64           LONG64    =      9000000000000
; UBIG64          ULONG64   =     18000000000000
```

### Float (FLOAT) and Double (DOUBLE)

```idl
; Float (32-bit, ~7 decimal digits of precision)
f = 3.14
f2 = 1.0E6        ; Scientific notation: 1,000,000.0
f3 = 2.5E-3       ; 0.0025

; Double (64-bit, ~15 decimal digits of precision)
d = 3.14159265358979D0
d2 = 1.0D6         ; Scientific notation with double precision
d3 = 2.5D-3

HELP, f, d
; F               FLOAT     =       3.14000
; D               DOUBLE    =        3.1415926535898

; Precision matters in scientific computing
PRINT, 1.0 / 3.0           ;     0.333333  (7 digits)
PRINT, 1.0D0 / 3.0D0       ;      0.33333333333333  (15 digits)

; Pi constants
PRINT, !PI                  ;     3.14159  (float)
PRINT, !DPI                 ;      3.1415926535898  (double)
```

### Complex Numbers

```idl
; Complex (pair of floats)
z1 = COMPLEX(3.0, 4.0)     ; 3 + 4i
PRINT, z1                   ; (     3.00000,     4.00000)

; Double complex
z2 = DCOMPLEX(1.0D0, -2.0D0)
PRINT, z2                   ; (      1.0000000,     -2.0000000)

; Extract real and imaginary parts
PRINT, REAL_PART(z1)        ;     3.00000
PRINT, IMAGINARY(z1)        ;     4.00000

; Complex arithmetic
z3 = z1 * COMPLEX(2.0, 1.0)
PRINT, z3                   ; (     2.00000,     11.0000)

; Magnitude (absolute value)
PRINT, ABS(z1)              ;     5.00000   (sqrt(3^2 + 4^2))
```

### Strings

```idl
; String creation
s1 = 'Hello, World!'
s2 = "Double quotes also work"
s3 = ''                     ; Empty string

; String length
PRINT, STRLEN(s1)           ;       13

; String concatenation
greeting = 'Hello' + ', ' + 'World!'
PRINT, greeting             ; Hello, World!

; Include quotes in strings
s4 = "It's a test"
s5 = 'She said "hello"'

HELP, s1
; S1              STRING    = 'Hello, World!'
```

---

## Type Conversion Functions

IDL provides explicit conversion functions to change between types:

```idl
; BYTE() — Convert to byte
PRINT, BYTE(65)              ;   65
PRINT, BYTE(-1)              ;  255  (wraps around)

; FIX() — Convert to integer (INT)
PRINT, FIX(3.7)              ;       3  (truncates, does not round)
PRINT, FIX('42')             ;      42

; LONG() — Convert to long integer
PRINT, LONG(100000.5)        ;       100000

; FLOAT() — Convert to float
PRINT, FLOAT(42)             ;      42.0000
PRINT, FLOAT('3.14')         ;      3.14000

; DOUBLE() — Convert to double
PRINT, DOUBLE(42)            ;       42.000000
PRINT, DOUBLE(!PI)           ;       3.1415927  (promotes from float !PI)

; STRING() — Convert to string
PRINT, STRING(42)            ;       42
PRINT, STRING(3.14, FORMAT='(F6.3)')  ;  3.140

; UINT(), ULONG(), LONG64(), ULONG64() — Unsigned and 64-bit conversions
PRINT, UINT(42)
PRINT, LONG64(9000000000)

; COMPLEX() — Convert to complex
PRINT, COMPLEX(3)            ; (     3.00000,     0.00000)
```

### Type Conversion Table

```idl
; Summary of type conversion functions:
;
;   BYTE(x)      -> BYTE
;   FIX(x)       -> INT       (also: FIX(x, TYPE=type_code))
;   UINT(x)      -> UINT
;   LONG(x)      -> LONG
;   ULONG(x)     -> ULONG
;   LONG64(x)    -> LONG64
;   ULONG64(x)   -> ULONG64
;   FLOAT(x)     -> FLOAT
;   DOUBLE(x)    -> DOUBLE
;   COMPLEX(x)   -> COMPLEX
;   DCOMPLEX(x)  -> DCOMPLEX
;   STRING(x)    -> STRING
```

---

## Inspecting Variables

### HELP

The `HELP` procedure displays the name, type, and value (or dimensions) of variables:

```idl
a = 42
b = 3.14D0
c = 'Hello'
d = FINDGEN(10)
e = FLTARR(3, 4)

HELP, a, b, c, d, e
; A               INT       =       42
; B               DOUBLE    =        3.1400000000000
; C               STRING    = 'Hello'
; D               FLOAT     = Array[10]
; E               FLOAT     = Array[3, 4]

; Show all variables in the current scope
HELP

; Show only structures
HELP, /STRUCTURES

; Show memory usage
HELP, /MEMORY
```

### SIZE

The `SIZE` function returns detailed information about a variable's dimensions and type:

```idl
x = FLTARR(100, 200)

; Default: returns an array of dimension info
info = SIZE(x)
PRINT, info
;          2         100         200           4       20000
; [n_dims, dim1, dim2, type_code, n_elements]

; Type codes:
;  0 = Undefined   1 = BYTE      2 = INT       3 = LONG
;  4 = FLOAT       5 = DOUBLE    6 = COMPLEX   7 = STRING
;  8 = STRUCT      9 = DCOMPLEX 10 = POINTER  11 = OBJREF
; 12 = UINT       13 = ULONG    14 = LONG64   15 = ULONG64

; Get specific information with keywords
PRINT, SIZE(x, /N_DIMENSIONS)   ;        2
PRINT, SIZE(x, /DIMENSIONS)     ;      100     200
PRINT, SIZE(x, /TYPE)           ;        4  (FLOAT)
PRINT, SIZE(x, /TNAME)          ; FLOAT
PRINT, SIZE(x, /N_ELEMENTS)     ;    20000
```

### N_ELEMENTS

`N_ELEMENTS` returns the total number of elements in a variable:

```idl
arr = FINDGEN(5, 3)
PRINT, N_ELEMENTS(arr)      ;       15  (5 * 3)

scalar = 42
PRINT, N_ELEMENTS(scalar)   ;        1

; N_ELEMENTS of undefined variable returns 0 — useful for existence checks
PRINT, N_ELEMENTS(undefined_var)  ;        0
```

### ISA and TYPENAME (IDL 8+)

```idl
x = 3.14D0
arr = FINDGEN(10)
s = {name: 'Alice', age: 30}

; ISA checks if a variable is of a certain type
PRINT, ISA(x, 'DOUBLE')         ;    1  (TRUE)
PRINT, ISA(x, 'FLOAT')          ;    0  (FALSE)
PRINT, ISA(arr, /ARRAY)         ;    1  (TRUE)
PRINT, ISA(x, /NUMBER)          ;    1  (TRUE)
PRINT, ISA(s, /STRUCTURE)       ;    1  (TRUE)

; TYPENAME returns the type name as a string
PRINT, TYPENAME(x)              ; DOUBLE
PRINT, TYPENAME(arr)            ; FLOAT
PRINT, TYPENAME(s)              ; ANONYMOUS
```

---

## Type Promotion Rules

When you mix types in an expression, IDL promotes the result to the more precise type:

```idl
; INT + FLOAT -> FLOAT
result = 5 + 3.0
HELP, result
; RESULT          FLOAT     =       8.00000

; FLOAT + DOUBLE -> DOUBLE
result = 3.14 + 1.0D0
HELP, result
; RESULT          DOUBLE    =        4.1400000000000

; INT + LONG -> LONG
result = 5 + 100000L
HELP, result
; RESULT          LONG      =       100005

; Promotion hierarchy (lowest to highest):
; BYTE < INT < LONG < LONG64
;                          ↘
;                    FLOAT < DOUBLE
;                          ↘
;                    COMPLEX < DCOMPLEX
```

### Precision Pitfall

```idl
; This is a common source of bugs:
; 1/3 is integer division!
PRINT, 1/3           ;       0  (integer division)
PRINT, 1.0/3.0       ;     0.333333  (float division)
PRINT, 1.0D0/3.0D0   ;      0.33333333333333  (double division)

; Always use float or double literals in scientific calculations
wavelength = 6563.0    ; Angstroms (float)
frequency = 3.0D14     ; Hz (double for precision)
```

---

## Special Values

### NaN (Not a Number) and Infinity

```idl
; Create NaN and Infinity
nan_float = !VALUES.F_NAN
inf_float = !VALUES.F_INFINITY
nan_double = !VALUES.D_NAN
inf_double = !VALUES.D_INFINITY

; NaN results from undefined operations
PRINT, 0.0 / 0.0         ; NaN
PRINT, ALOG(-1.0)         ; NaN

; Infinity results from overflow
PRINT, 1.0 / 0.0          ; Inf

; Testing for NaN (NaN is not equal to anything, including itself)
PRINT, FINITE(nan_float)             ;        0  (FALSE)
PRINT, FINITE(inf_float)             ;        0  (FALSE)
PRINT, FINITE(42.0)                  ;        1  (TRUE)
PRINT, FINITE(nan_float, /NAN)       ;        1  (TRUE - it IS NaN)
PRINT, FINITE(inf_float, /INFINITY)  ;        1  (TRUE - it IS Infinity)

; Replace NaN values in an array
data = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
good = WHERE(FINITE(data), count)
IF count GT 0 THEN PRINT, 'Good values:', data[good]
; Good values:      1.00000      3.00000      5.00000
```

### NULL (IDL 8+)

```idl
; NULL represents an undefined variable
x = !NULL
HELP, x
; X               UNDEFINED = !NULL

; Useful for initializing variables before a loop
result = !NULL
FOR i = 0, 4 DO result = [result, i^2]
PRINT, result
;        0       1       4       9      16
```

---

## Practical Examples

### Scientific Data Types

```idl
; Solar physics example: photon wavelength
wavelength_angstrom = 1216.0D0    ; Lyman-alpha in Angstroms (double)
wavelength_nm = wavelength_angstrom / 10.0D0  ; Convert to nm
wavelength_m = wavelength_angstrom * 1.0D-10  ; Convert to meters

PRINT, 'Lyman-alpha wavelength:'
PRINT, FORMAT='("  ", F8.2, " Angstroms")', wavelength_angstrom
PRINT, FORMAT='("  ", F8.2, " nm")', wavelength_nm
PRINT, FORMAT='("  ", E12.4, " m")', wavelength_m

; Complex impedance in electromagnetic calculations
Z = COMPLEX(50.0, 30.0)  ; 50 + 30i Ohms
PRINT, 'Impedance:', Z
PRINT, 'Magnitude:', ABS(Z), ' Ohms'
PRINT, 'Phase:', ATAN(IMAGINARY(Z), REAL_PART(Z)) * !RADEG, ' degrees'
```

### Type Checking in Routines

```idl
; A function that validates input type
FUNCTION safe_sqrt, x
  ; Ensure input is numeric
  IF ~ISA(x, /NUMBER) THEN BEGIN
    PRINT, 'Error: Input must be numeric'
    RETURN, !VALUES.F_NAN
  ENDIF

  ; Convert to double for precision
  dx = DOUBLE(x)

  ; Check for negative values
  neg = WHERE(dx LT 0, n_neg)
  IF n_neg GT 0 THEN BEGIN
    PRINT, 'Warning: Negative values set to NaN'
    result = SQRT(ABS(dx))
    result[neg] = !VALUES.D_NAN
    RETURN, result
  ENDIF

  RETURN, SQRT(dx)
END
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Dynamic typing | Variables take the type of their assigned value |
| Type suffixes | `B` (byte), `S` (int), `L` (long), `LL` (long64), `U` (unsigned), `D` (double) |
| HELP | Displays variable name, type, and value/dimensions |
| SIZE | Returns dimension and type information |
| N_ELEMENTS | Returns total element count (0 for undefined) |
| ISA | Type checking with keywords (/NUMBER, /ARRAY, etc.) |
| Type conversion | FIX, FLOAT, DOUBLE, STRING, BYTE, LONG, etc. |
| Type promotion | Mixed-type expressions promote to the more precise type |
| Special values | `!VALUES.F_NAN`, `!VALUES.F_INFINITY`, `!NULL` |
| FINITE | Test for NaN and Infinity |

---

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Arrays and Operations](./03_Arrays_and_Operations.md)
