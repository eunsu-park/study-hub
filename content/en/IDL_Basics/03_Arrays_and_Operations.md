# Arrays and Operations

**Previous**: [Variables and Data Types](./02_Variables_and_Data_Types.md) | **Next**: [Operators and Expressions](./04_Operators_and_Expressions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create arrays using generator functions (INDGEN, FINDGEN, DINDGEN) and zero-fill functions (INTARR, FLTARR, DBLARR, BYTARR)
2. Use MAKE_ARRAY for flexible array construction
3. Index and slice arrays with subscript notation
4. Use the WHERE function to find elements matching conditions
5. Perform element-wise arithmetic on arrays
6. Reshape arrays with REFORM and understand array dimensions
7. Distinguish between N_ELEMENTS and N_PARAMS

---

Arrays are the fundamental data structure in IDL. Unlike many languages where you loop over individual elements, IDL is designed to operate on entire arrays at once. This array-oriented approach is both more concise and dramatically faster than element-by-element processing.

## Array Creation

### Generator Functions (Index Arrays)

These functions create arrays filled with consecutive index values:

```idl
; INDGEN — Integer index array
a = INDGEN(5)
PRINT, a             ;        0       1       2       3       4

; FINDGEN — Float index array
b = FINDGEN(5)
PRINT, b             ;      0.00000      1.00000      2.00000      3.00000      4.00000

; DINDGEN — Double index array
c = DINDGEN(5)
PRINT, c             ;       0.0000000       1.0000000       2.0000000       3.0000000       4.0000000

; BINDGEN — Byte index array
d = BINDGEN(5)
PRINT, d             ;    0   1   2   3   4

; LINDGEN — Long index array
e = LINDGEN(5)
PRINT, e             ;            0           1           2           3           4

; SINDGEN — String index array (string representations of indices)
f = SINDGEN(5)
PRINT, f             ;          0          1          2          3          4

; CINDGEN — Complex index array
g = CINDGEN(5)
PRINT, g
```

### Multi-Dimensional Index Arrays

```idl
; 2D array (3 columns x 4 rows)
arr2d = INDGEN(3, 4)
PRINT, arr2d
;        0       1       2
;        3       4       5
;        6       7       8
;        9      10      11

; 3D array (2 x 3 x 4)
arr3d = FINDGEN(2, 3, 4)
HELP, arr3d
; ARR3D           FLOAT     = Array[2, 3, 4]

PRINT, N_ELEMENTS(arr3d)   ;       24
```

### Zero-Fill Arrays

These functions create arrays initialized to zero:

```idl
; BYTARR — Byte array of zeros
ba = BYTARR(5)
PRINT, ba            ;    0   0   0   0   0

; INTARR — Integer array of zeros
ia = INTARR(5)
PRINT, ia            ;        0       0       0       0       0

; LONARR — Long array of zeros
la = LONARR(5)

; FLTARR — Float array of zeros
fa = FLTARR(3, 4)
HELP, fa             ; FA              FLOAT     = Array[3, 4]

; DBLARR — Double array of zeros
da = DBLARR(100)

; COMPLEXARR — Complex array of zeros
ca = COMPLEXARR(10)

; STRARR — String array of empty strings
sa = STRARR(5)
PRINT, sa            ; (five empty strings)
```

### MAKE_ARRAY

`MAKE_ARRAY` is a flexible function that can create arrays of any type:

```idl
; Create by specifying type code
arr = MAKE_ARRAY(5, 3, TYPE=4)     ; 5x3 FLOAT array of zeros
HELP, arr
; ARR             FLOAT     = Array[5, 3]

; Create with initial value
arr = MAKE_ARRAY(10, VALUE=99.0)
PRINT, arr
;      99.0000      99.0000  ... (10 values)

; Create with /INDEX keyword (like *INDGEN)
arr = MAKE_ARRAY(5, /FLOAT, /INDEX)
PRINT, arr
;      0.00000      1.00000      2.00000      3.00000      4.00000

; Type keywords
arr_byte = MAKE_ARRAY(5, /BYTE, VALUE=128B)
arr_double = MAKE_ARRAY(3, 3, /DOUBLE)
arr_string = MAKE_ARRAY(4, /STRING, VALUE='empty')
```

### Array Literals

```idl
; Create arrays directly with square brackets
x = [1, 2, 3, 4, 5]
y = [1.0, 2.0, 3.0]
names = ['Alice', 'Bob', 'Charlie']

; Nested brackets create multi-dimensional arrays
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
HELP, matrix
; MATRIX          INT       = Array[3, 3]
PRINT, matrix
;        1       4       7
;        2       5       8
;        3       6       9
; Note: IDL is column-major! [[1,2,3], [4,5,6], [7,8,9]] creates 3 columns.

; Concatenation with brackets
a = [1, 2, 3]
b = [4, 5, 6]
c = [a, b]
PRINT, c             ;        1       2       3       4       5       6
```

---

## Array Indexing

IDL uses zero-based indexing with square brackets:

```idl
arr = [10, 20, 30, 40, 50]

; Single element
PRINT, arr[0]        ;       10
PRINT, arr[4]        ;       50

; Negative indexing (from the end) — IDL 8+ only
PRINT, arr[-1]       ;       50
PRINT, arr[-2]       ;       40

; Multiple indices
PRINT, arr[[0, 2, 4]]  ;       10      30      50
```

### Array Slicing (Ranges)

```idl
arr = INDGEN(10)     ; [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

; Range with colon
PRINT, arr[2:5]      ;        2       3       4       5

; From start
PRINT, arr[0:3]      ;        0       1       2       3

; To end (using * for last element)
PRINT, arr[7:*]      ;        7       8       9

; With stride (IDL 8.4+)
PRINT, arr[0:*:2]    ;        0       2       4       6       8  (every other element)
PRINT, arr[1:*:3]    ;        1       4       7
```

### Multi-Dimensional Indexing

```idl
; Create a 4x3 array
arr = INDGEN(4, 3)
PRINT, arr
;        0       1       2       3
;        4       5       6       7
;        8       9      10      11

; Single element: arr[column, row]
PRINT, arr[1, 2]     ;        9

; Row slice (all columns of row 1)
PRINT, arr[*, 1]     ;        4       5       6       7

; Column slice (all rows of column 2)
PRINT, arr[2, *]     ;        2       6      10

; Sub-array
PRINT, arr[1:2, 0:1]
;        1       2
;        5       6

; 3D indexing
cube = INDGEN(3, 4, 5)  ; 3 x 4 x 5 cube
PRINT, cube[1, 2, 3]     ; Single element
PRINT, cube[*, *, 0]     ; First "slice" (3 x 4)
PRINT, cube[0, *, *]     ; First column across all slices (4 x 5)
```

---

## The WHERE Function

`WHERE` is one of the most important functions in IDL. It returns the indices of array elements that satisfy a condition:

```idl
data = [3, 7, 1, 9, 4, 6, 2, 8, 5]

; Find elements greater than 5
idx = WHERE(data GT 5, count)
PRINT, 'Indices:', idx       ; Indices:       1       3       5       7
PRINT, 'Count:', count       ; Count:       4
PRINT, 'Values:', data[idx]  ; Values:       7       9       6       8

; Find elements equal to a value
idx = WHERE(data EQ 4)
PRINT, 'Index of 4:', idx    ;        4

; Find elements in a range
idx = WHERE(data GE 3 AND data LE 6, count)
PRINT, 'Between 3 and 6:', data[idx]  ;        3       4       6       5

; No match returns -1
idx = WHERE(data GT 100, count)
PRINT, 'Count:', count       ;        0
PRINT, 'Index:', idx         ;       -1
```

### WHERE with Complement

```idl
data = FINDGEN(10)

; Get both matching and non-matching indices
good = WHERE(data GE 5, n_good, COMPLEMENT=bad, NCOMPLEMENT=n_bad)
PRINT, 'Good:', data[good]   ; Good:      5.00000      6.00000 ... 9.00000
PRINT, 'Bad:', data[bad]     ; Bad:      0.00000      1.00000 ... 4.00000
PRINT, 'N_good:', n_good     ;        5
PRINT, 'N_bad:', n_bad       ;        5
```

### Common WHERE Patterns

```idl
; Replace values
data = FINDGEN(10)
idx = WHERE(data LT 3)
IF idx[0] NE -1 THEN data[idx] = -999.0
PRINT, data
; -999.000    -999.000    -999.000      3.00000  ...

; Remove NaN values
data = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
good = WHERE(FINITE(data), count)
IF count GT 0 THEN clean_data = data[good]
PRINT, clean_data    ;      1.00000      3.00000      5.00000

; Count occurrences
flags = [0, 1, 1, 0, 1, 0, 0, 1, 1]
n_true = TOTAL(flags EQ 1)
PRINT, 'Number of 1s:', n_true   ;        5

; Threshold an image
image = BYTSCL(DIST(256))
bright = WHERE(image GT 200B, count)
PRINT, 'Bright pixels:', count
```

---

## Array Arithmetic

IDL performs arithmetic element-wise on arrays, which is both concise and fast:

```idl
a = [1.0, 2.0, 3.0, 4.0, 5.0]
b = [10.0, 20.0, 30.0, 40.0, 50.0]

; Element-wise operations
PRINT, a + b         ;      11.0000      22.0000 ...
PRINT, a * b         ;      10.0000      40.0000 ...
PRINT, b / a         ;      10.0000      10.0000 ...
PRINT, a ^ 2         ;      1.00000      4.00000 ...
PRINT, SQRT(a)       ;      1.00000      1.41421 ...

; Scalar-array operations
PRINT, a * 10        ;      10.0000      20.0000 ...
PRINT, a + 100       ;      101.000      102.000 ...

; Mathematical functions work element-wise
x = FINDGEN(100) / 99.0 * 2.0 * !PI
y = SIN(x)
PRINT, MIN(y), MAX(y)
```

### Array Statistics

```idl
data = RANDOMN(seed, 1000)   ; 1000 random numbers from normal distribution

; Basic statistics
PRINT, 'Mean:', MEAN(data)
PRINT, 'Median:', MEDIAN(data)
PRINT, 'Std Dev:', STDDEV(data)
PRINT, 'Variance:', VARIANCE(data)
PRINT, 'Min:', MIN(data)
PRINT, 'Max:', MAX(data)
PRINT, 'Total:', TOTAL(data)

; Min/Max with index
min_val = MIN(data, min_idx)
max_val = MAX(data, max_idx)
PRINT, 'Min value:', min_val, ' at index:', min_idx
PRINT, 'Max value:', max_val, ' at index:', max_idx

; MOMENT function returns [mean, variance, skewness, kurtosis]
moments = MOMENT(data)
PRINT, 'Moments:', moments
```

### Matrix Operations

```idl
; Matrix multiplication with # and ##
A = [[1, 2], [3, 4]]   ; 2x2 matrix
B = [[5, 6], [7, 8]]   ; 2x2 matrix

; # operator: column of first * row of second (standard matrix multiply for [m,n] # [n,p])
C = A # B
PRINT, C

; ## operator: row of first * column of second
D = A ## B
PRINT, D

; Transpose
PRINT, TRANSPOSE(A)

; Identity matrix
I = IDENTITY(3, /DOUBLE)  ; or use DIAG_MATRIX(REPLICATE(1.0D, 3))

; Determinant
PRINT, DETERM(DOUBLE(A))

; Matrix inverse
A_inv = INVERT(DOUBLE(A))
PRINT, A_inv
```

---

## Array Manipulation

### REFORM — Reshape Arrays

```idl
; Reshape a 1D array into 2D
arr = INDGEN(12)
arr2d = REFORM(arr, 3, 4)
PRINT, arr2d
;        0       1       2
;        3       4       5
;        6       7       8
;        9      10      11

; Reshape 2D to 3D
arr3d = REFORM(arr, 2, 2, 3)
HELP, arr3d
; ARR3D           INT       = Array[2, 2, 3]

; Remove degenerate dimensions (squeeze)
x = FLTARR(10, 1, 5)
HELP, x              ; Array[10, 1, 5]
y = REFORM(x)
HELP, y              ; Array[10, 5]   — removed the size-1 dimension

; Explicit dimensions with REFORM
z = FLTARR(1, 10)    ; Array[1, 10]
z = REFORM(z, 10)    ; Array[10]
```

### REBIN — Resize by Integer Factors

```idl
; Expand an array by integer multiples
small = INDGEN(3, 2)
PRINT, small
;        0       1       2
;        3       4       5

big = REBIN(small, 6, 4)   ; Each dimension multiplied by 2
PRINT, big

; Shrink an array (dimensions must divide evenly)
big = FINDGEN(100)
small = REBIN(big, 10)     ; Average every 10 elements
PRINT, small
```

### CONGRID — Resize to Arbitrary Dimensions

```idl
; Resize to any dimension (uses interpolation)
arr = FINDGEN(10)
new_arr = CONGRID(arr, 25)        ; Resize 10 elements to 25
HELP, new_arr

; 2D resize
image = DIST(64)                   ; 64x64 image
big_image = CONGRID(image, 256, 256)
HELP, big_image
```

### REVERSE

```idl
arr = [1, 2, 3, 4, 5]
PRINT, REVERSE(arr)  ;        5       4       3       2       1

; Reverse along a specific dimension in multidimensional arrays
arr2d = INDGEN(3, 3)
PRINT, REVERSE(arr2d, 1)  ; Reverse columns
PRINT, REVERSE(arr2d, 2)  ; Reverse rows
```

### SORT

```idl
data = [3, 1, 4, 1, 5, 9, 2, 6]

; SORT returns indices that would sort the array
idx = SORT(data)
PRINT, data[idx]     ;        1       1       2       3       4       5       6       9

; Sort in descending order
idx = REVERSE(SORT(data))
PRINT, data[idx]     ;        9       6       5       4       3       2       1       1
```

### UNIQ

```idl
; UNIQ returns indices of unique elements (input must be sorted)
data = [1, 1, 2, 3, 3, 3, 4, 5, 5]
u = UNIQ(data)
PRINT, data[u]       ;        1       2       3       4       5

; For unsorted data, sort first
data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
u = UNIQ(data, SORT(data))
PRINT, data[u]       ;        1       2       3       4       5       6       9
```

### SHIFT

```idl
arr = [1, 2, 3, 4, 5]

; Shift right by 2
PRINT, SHIFT(arr, 2)  ;        4       5       1       2       3

; Shift left by 1
PRINT, SHIFT(arr, -1) ;        2       3       4       5       1

; 2D shift
arr2d = INDGEN(4, 4)
shifted = SHIFT(arr2d, 1, 0)  ; Shift columns right by 1
```

---

## N_ELEMENTS vs N_PARAMS

These are two different functions that beginners sometimes confuse:

```idl
; N_ELEMENTS — number of elements in a variable
arr = FINDGEN(100)
PRINT, N_ELEMENTS(arr)       ;      100

; N_PARAMS — number of positional parameters passed to a routine
PRO example_proc, a, b, c
  PRINT, 'Number of parameters:', N_PARAMS()
  PRINT, 'N_ELEMENTS of a:', N_ELEMENTS(a)
END

; Calling with different numbers of arguments
example_proc, [1,2,3]              ; N_PARAMS = 1, N_ELEMENTS(a) = 3
example_proc, [1,2,3], 'hello'     ; N_PARAMS = 2
example_proc, [1,2,3], 'hi', 42   ; N_PARAMS = 3
```

---

## Practical Examples

### Solar Image Processing

```idl
; Simulate a solar image (512 x 512 pixels)
nx = 512
ny = 512
image = RANDOMN(seed, nx, ny) * 100.0 + 1000.0

; Find bright regions (above 3 sigma)
mean_val = MEAN(image)
sigma = STDDEV(image)
threshold = mean_val + 3.0 * sigma
bright = WHERE(image GT threshold, n_bright)
PRINT, 'Bright pixels:', n_bright
PRINT, 'Fraction:', FLOAT(n_bright) / N_ELEMENTS(image)

; Extract a subregion (central 100x100)
x0 = nx/2 - 50
y0 = ny/2 - 50
subregion = image[x0:x0+99, y0:y0+99]
HELP, subregion
PRINT, 'Subregion mean:', MEAN(subregion)
```

### Time Series Analysis

```idl
; Create sample time series data
n = 1000
time = DINDGEN(n) / 100.0D0          ; 0 to 10 seconds
signal = SIN(2.0D * !DPI * time) + $  ; 1 Hz sine
         0.5D * SIN(2.0D * !DPI * 3.0D * time) + $  ; 3 Hz harmonic
         RANDOMN(seed, n) * 0.2D       ; noise

; Smooth the data with running average
kernel_size = 11
smoothed = SMOOTH(signal, kernel_size, /EDGE_TRUNCATE)

; Find zero crossings
signs = signal GT 0
crossings = WHERE(signs[1:*] NE signs[0:N_ELEMENTS(signs)-2], n_cross)
PRINT, 'Zero crossings:', n_cross

; Peak detection
; Find local maxima: signal[i] > signal[i-1] AND signal[i] > signal[i+1]
n_pts = N_ELEMENTS(signal)
peaks = WHERE(signal[1:n_pts-2] GT signal[0:n_pts-3] AND $
              signal[1:n_pts-2] GT signal[2:n_pts-1], n_peaks)
peaks = peaks + 1  ; Adjust for offset
PRINT, 'Number of peaks:', n_peaks
```

---

## Summary

| Concept | Description |
|---------|-------------|
| INDGEN, FINDGEN, DINDGEN | Create arrays with index values |
| INTARR, FLTARR, DBLARR | Create zero-filled arrays |
| MAKE_ARRAY | Flexible array construction |
| `arr[i]`, `arr[i:j]` | Indexing and slicing |
| `arr[*, n]`, `arr[n, *]` | Row/column selection |
| WHERE | Find indices matching a condition |
| Element-wise operations | `+`, `-`, `*`, `/`, `^` on arrays |
| REFORM | Reshape array dimensions |
| REBIN / CONGRID | Resize arrays |
| SORT, UNIQ, REVERSE, SHIFT | Array manipulation utilities |
| N_ELEMENTS | Count elements in a variable |
| N_PARAMS | Count positional parameters in a routine |

---

**Previous**: [Variables and Data Types](./02_Variables_and_Data_Types.md) | **Next**: [Operators and Expressions](./04_Operators_and_Expressions.md)
