# Operators and Expressions

**Previous**: [Arrays and Operations](./03_Arrays_and_Operations.md) | **Next**: [Control Flow](./05_Control_Flow.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use arithmetic operators (+, -, *, /, ^, MOD) on scalars and arrays
2. Apply relational operators (EQ, NE, LT, GT, LE, GE)
3. Combine conditions with logical operators (AND, OR, NOT, XOR)
4. Understand bitwise operations
5. Concatenate strings with the + operator
6. Follow IDL's operator precedence rules
7. Write complex expressions using ternary-like patterns

---

Operators are the building blocks of expressions in IDL. Because IDL is array-oriented, most operators work element-wise on arrays, making it straightforward to perform computations on entire datasets without explicit loops.

## Arithmetic Operators

```idl
; Basic arithmetic with scalars
a = 15
b = 4

PRINT, a + b       ;       19   Addition
PRINT, a - b       ;       11   Subtraction
PRINT, a * b       ;       60   Multiplication
PRINT, a / b       ;        3   Integer division (truncates)
PRINT, a ^ 2       ;      225   Exponentiation
PRINT, a MOD b     ;        3   Modulo (remainder)
```

### Integer Division Pitfall

```idl
; Integer operands produce integer results
PRINT, 7 / 2        ;        3   (not 3.5!)
PRINT, 1 / 3        ;        0   (not 0.333!)

; Use floating-point operands for decimal results
PRINT, 7.0 / 2.0    ;      3.50000
PRINT, 1.0 / 3.0    ;     0.333333
PRINT, FLOAT(7) / FLOAT(2)  ;      3.50000
```

### Arithmetic with Arrays

```idl
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [10.0, 20.0, 30.0, 40.0, 50.0]

; Element-wise operations
PRINT, x + y     ;      11.0000      22.0000      33.0000      44.0000      55.0000
PRINT, x * y     ;      10.0000      40.0000      90.0000      160.000      250.000
PRINT, y / x     ;      10.0000      10.0000      10.0000      10.0000      10.0000
PRINT, x ^ 2     ;      1.00000      4.00000      9.00000      16.0000      25.0000

; Scalar broadcast to array
PRINT, x * 100   ;      100.000      200.000      300.000      400.000      500.000
PRINT, x + 1000  ;      1001.00      1002.00      1003.00      1004.00      1005.00
```

### Exponentiation Details

```idl
; ^ operator for powers
PRINT, 2 ^ 10       ;     1024
PRINT, 2.0 ^ 0.5    ;      1.41421   (square root)
PRINT, 10.0 ^ (-3)  ;   0.00100000

; Array exponentiation
bases = FINDGEN(5) + 1.0
PRINT, bases ^ 3    ;      1.00000      8.00000      27.0000      64.0000      125.000

; Use SQRT for square root (faster than ^0.5)
PRINT, SQRT(2.0)    ;      1.41421
```

### MOD Operator

```idl
; Modulo — remainder after division
PRINT, 17 MOD 5     ;        2
PRINT, 10 MOD 3     ;        1
PRINT, 15 MOD 5     ;        0

; Works with floats
PRINT, 7.5 MOD 2.0  ;      1.50000

; Common use: check even/odd
n = INDGEN(10)
even = WHERE(n MOD 2 EQ 0)
PRINT, 'Even:', n[even]    ;        0       2       4       6       8
```

### Unary Minus

```idl
x = 5
PRINT, -x            ;       -5

arr = [1, -2, 3, -4]
PRINT, -arr          ;       -1       2      -3       4
```

---

## Relational Operators

IDL uses keyword-style relational operators (not symbols like `<` or `>`):

```idl
a = 10
b = 20

; Relational operators return 1 (true) or 0 (false)
PRINT, a EQ b    ;        0   Equal
PRINT, a NE b    ;        1   Not equal
PRINT, a LT b    ;        1   Less than
PRINT, a GT b    ;        0   Greater than
PRINT, a LE b    ;        1   Less than or equal
PRINT, a GE b    ;        0   Greater than or equal
```

### Relational Operators on Arrays

```idl
data = [15, 8, 22, 3, 17, 11, 29, 5]

; Element-wise comparison returns an array of 0s and 1s
PRINT, data GT 10
;        1       0       1       0       1       1       1       0

; Use with WHERE
big = WHERE(data GT 10, count)
PRINT, 'Values > 10:', data[big]
PRINT, 'Count:', count

; Combined with arithmetic for masking
mask = data GE 10 AND data LE 20
PRINT, 'Mask:', mask
;        1       0       0       0       1       1       0       0
masked_data = data * mask
PRINT, 'Masked:', masked_data
;       15       0       0       0      17      11       0       0
```

### The < and > Operators (Min/Max)

IDL overloads `<` and `>` as minimum and maximum operators (not relational!):

```idl
; < returns the smaller of two values
PRINT, 5 < 3        ;        3
PRINT, 10 < 20      ;       10

; > returns the larger of two values
PRINT, 5 > 3        ;        5
PRINT, 10 > 20      ;       20

; Clamping a value to a range
x = 150
clamped = 0 > x < 100     ; Clamp to [0, 100]
PRINT, clamped             ;      100

; Array clamping
data = [−5, 10, 150, 50, 200, −20, 80]
clamped = 0 > data < 100
PRINT, clamped
;        0      10     100      50     100       0      80
```

**Important**: In IDL, `<` and `>` are NOT relational operators. Use `LT`, `GT`, `LE`, `GE` for comparisons.

---

## Logical Operators

```idl
; AND — logical (and bitwise) AND
PRINT, 1 AND 1      ;        1
PRINT, 1 AND 0      ;        0

; OR — logical (and bitwise) OR
PRINT, 1 OR 0       ;        1
PRINT, 0 OR 0       ;        0

; NOT — bitwise complement (NOT logical NOT!)
PRINT, NOT 0         ;       -1  (all bits flipped: 0000...0000 -> 1111...1111)
PRINT, NOT 1         ;       -2

; XOR — exclusive OR
PRINT, 1 XOR 0      ;        1
PRINT, 1 XOR 1      ;        0

; For logical NOT, use ~ (IDL 8+) or the pattern (x EQ 0)
PRINT, ~0            ;       -1  (same as NOT — bitwise)

; Preferred logical NOT pattern:
flag = 1
PRINT, flag EQ 0     ;        0  (logical NOT of 1)
PRINT, (flag NE 0)   ;        1  (logical TRUE)
```

### Logical Operators with Arrays

```idl
x = [1, 5, 3, 8, 2, 7, 4, 9]

; Compound conditions
idx = WHERE(x GE 3 AND x LE 7, count)
PRINT, 'Between 3 and 7:', x[idx]

; OR condition
idx = WHERE(x LT 2 OR x GT 8, count)
PRINT, 'Less than 2 or greater than 8:', x[idx]
```

### Short-Circuit Evaluation with && and || (IDL 8+)

```idl
; && and || perform short-circuit evaluation (scalars only)
; The second operand is not evaluated if the result is already determined

a = 5
b = 0

; Safe division — b is only used if it's non-zero
IF b NE 0 && (a/b GT 2) THEN PRINT, 'Large ratio'

; || short-circuits on first true
IF a GT 0 || b GT 0 THEN PRINT, 'At least one positive'

; Note: && and || work only with scalar expressions
; For arrays, use AND and OR
```

---

## Bitwise Operators

The `AND`, `OR`, `NOT`, and `XOR` operators also serve as bitwise operators when used with integers:

```idl
; Bitwise AND
PRINT, 12 AND 10    ;        8
; 12 = 1100
; 10 = 1010
; AND= 1000 = 8

; Bitwise OR
PRINT, 12 OR 10     ;       14
; OR = 1110 = 14

; Bitwise XOR
PRINT, 12 XOR 10    ;        6
; XOR= 0110 = 6

; Bitwise NOT (complement)
PRINT, NOT 0B        ;  255  (BYTE: 11111111)
PRINT, NOT 0         ;   -1  (INT: all 1s in two's complement)

; Bit shifting
PRINT, ISHFT(1, 4)   ;       16   (shift left by 4: 1 -> 10000)
PRINT, ISHFT(16, -2)  ;        4   (shift right by 2: 10000 -> 100)

; Check if a specific bit is set
value = 42           ; Binary: 101010
bit3 = (value AND ISHFT(1, 3)) NE 0  ; Check bit 3
PRINT, 'Bit 3 set:', bit3    ;        1 (TRUE, since 101010 has bit 3)
```

---

## String Operators

### String Concatenation

```idl
; The + operator concatenates strings
first = 'Hello'
second = 'World'
greeting = first + ', ' + second + '!'
PRINT, greeting      ; Hello, World!

; Concatenation with numbers requires STRING() conversion
name = 'Temperature'
value = 98.6
unit = 'F'
msg = name + ' = ' + STRING(value, FORMAT='(F5.1)') + ' ' + unit
PRINT, msg           ; Temperature = 98.6 F

; STRJOIN for array concatenation
words = ['IDL', 'is', 'powerful']
sentence = STRJOIN(words, ' ')
PRINT, sentence      ; IDL is powerful
```

### String Comparison

```idl
; String comparison uses the same relational operators
s1 = 'apple'
s2 = 'banana'

PRINT, s1 EQ s2     ;        0
PRINT, s1 NE s2     ;        1
PRINT, s1 LT s2     ;        1  (alphabetical comparison)

; Case-sensitive by default
PRINT, 'ABC' EQ 'abc'   ;        0
PRINT, STRUPCASE('abc') EQ 'ABC'  ;        1
```

---

## Operator Precedence

IDL evaluates expressions according to this precedence (highest to lowest):

| Precedence | Operators | Description |
|-----------|-----------|-------------|
| 1 (highest) | `()` | Parentheses |
| 2 | `^` | Exponentiation |
| 3 | `*`, `/`, `MOD` | Multiplication, division, modulo |
| 4 | `+`, `-` | Addition, subtraction (and unary minus) |
| 5 | `<`, `>` | Minimum, maximum |
| 6 | `EQ`, `NE`, `LT`, `GT`, `LE`, `GE` | Relational |
| 7 | `NOT`, `~` | Bitwise NOT |
| 8 | `AND` | Bitwise/logical AND |
| 9 | `OR`, `XOR` | Bitwise/logical OR, XOR |
| 10 | `&&` | Short-circuit AND |
| 11 (lowest) | `\|\|` | Short-circuit OR |

```idl
; Precedence examples
PRINT, 2 + 3 * 4        ;       14  (multiplication first)
PRINT, (2 + 3) * 4      ;       20  (parentheses override)
PRINT, 2 ^ 3 ^ 2        ;      512  (right-to-left: 2^(3^2) = 2^9)
PRINT, 10 - 3 - 2       ;        5  (left-to-right)

; Relational before logical
x = 5
PRINT, x GT 3 AND x LT 10  ;        1  (evaluated as (x GT 3) AND (x LT 10))

; Always use parentheses for clarity in complex expressions
result = ((a + b) * c) / (d - e)
```

---

## Ternary-Like Patterns

IDL does not have a ternary operator (`?:` in C), but you can achieve similar results:

```idl
; Pattern 1: Arithmetic trick (for numeric values)
x = 5
sign_label = (['negative', 'non-negative'])[x GE 0]
PRINT, sign_label    ; non-negative

; Pattern 2: Single-line IF (IDL 8+)
result = (x GT 0) ? 'positive' : 'non-positive'
PRINT, result        ; positive

; Pattern 3: Using WHERE on scalars
; Less common, but useful for array operations
values = [100, 200]
idx = x GT 0
PRINT, values[idx]   ;      200
```

### The Ternary Operator (?:) in IDL 8+

```idl
; Syntax: condition ? true_value : false_value
x = 42
label = (x MOD 2 EQ 0) ? 'even' : 'odd'
PRINT, label         ; even

; Nested ternary (use sparingly for readability)
score = 85
grade = (score GE 90) ? 'A' : ((score GE 80) ? 'B' : ((score GE 70) ? 'C' : 'F'))
PRINT, grade         ; B
```

---

## Practical Examples

### Unit Conversion

```idl
; Temperature conversion
temp_celsius = [-40.0, 0.0, 20.0, 37.0, 100.0]
temp_fahrenheit = temp_celsius * 9.0/5.0 + 32.0
temp_kelvin = temp_celsius + 273.15

FOR i = 0, N_ELEMENTS(temp_celsius) - 1 DO BEGIN
  PRINT, FORMAT='(F7.1, " C = ", F7.1, " F = ", F7.1, " K")', $
    temp_celsius[i], temp_fahrenheit[i], temp_kelvin[i]
ENDFOR
```

### Signal Thresholding

```idl
; Generate a noisy signal
n = 500
t = FINDGEN(n) / FLOAT(n)
signal = SIN(2.0 * !PI * 5.0 * t) + RANDOMN(seed, n) * 0.3

; Threshold: keep only values above 0.5
threshold = 0.5
above = WHERE(signal GT threshold, n_above)
below = WHERE(signal LE threshold, n_below)
PRINT, 'Above threshold:', n_above
PRINT, 'Below threshold:', n_below

; Clip signal to range [-1, 1]
clipped = (-1.0) > signal < 1.0
PRINT, 'Original range:', MIN(signal), MAX(signal)
PRINT, 'Clipped range:', MIN(clipped), MAX(clipped)
```

### Solar Flux Classification

```idl
; Classify solar X-ray flux levels (GOES classes)
; A < 1e-7, B < 1e-6, C < 1e-5, M < 1e-4, X >= 1e-4

flux = [3.2e-8, 5.1e-7, 2.3e-6, 8.7e-5, 1.2e-4, 4.5e-7, 6.3e-5]

FOR i = 0, N_ELEMENTS(flux) - 1 DO BEGIN
  f = flux[i]
  class = (f GE 1e-4) ? 'X' : $
          ((f GE 1e-5) ? 'M' : $
          ((f GE 1e-6) ? 'C' : $
          ((f GE 1e-7) ? 'B' : 'A')))
  PRINT, FORMAT='("Flux: ", E9.2, "  Class: ", A1)', f, class
ENDFOR
```

---

## Summary

| Category | Operators | Notes |
|----------|-----------|-------|
| Arithmetic | `+`, `-`, `*`, `/`, `^`, `MOD` | Integer division truncates |
| Relational | `EQ`, `NE`, `LT`, `GT`, `LE`, `GE` | Return 0 or 1 |
| Min/Max | `<`, `>` | NOT relational — return smaller/larger value |
| Logical | `AND`, `OR`, `NOT`, `XOR` | Also bitwise for integers |
| Short-circuit | `&&`, `\|\|` | Scalar only (IDL 8+) |
| String | `+` | Concatenation |
| Ternary | `? :` | IDL 8+ conditional expression |
| Bit shift | `ISHFT(value, bits)` | Positive=left, negative=right |

---

**Previous**: [Arrays and Operations](./03_Arrays_and_Operations.md) | **Next**: [Control Flow](./05_Control_Flow.md)
