# Integer and Floating-Point Representation

**Previous**: [Data Representation Basics](./02_Data_Representation_Basics.md) | **Next**: [Logic Gates](./04_Logic_Gates.md)

**Difficulty**: ⭐⭐ (Intermediate)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why two's complement is preferred over other negative number representations
2. Perform two's complement arithmetic (addition, subtraction, negation)
3. Describe the IEEE 754 floating-point format (sign, exponent, mantissa) for single and double precision
4. Identify special IEEE 754 values (infinity, NaN, denormalized numbers)
5. Explain why 0.1 + 0.2 does not equal 0.3 in floating-point arithmetic
6. Compare fixed-point and floating-point representations and their trade-offs

---

The infamous question "why does 0.1 + 0.2 not equal 0.3 in most programming languages?" has its answer in IEEE 754 floating-point representation. Understanding how computers represent integers and real numbers explains a whole class of bugs that have caused real-world disasters -- from the Ariane 5 rocket explosion (integer overflow) to the Patriot missile failure (floating-point drift).

## Table of Contents

1. [Integer Representation Overview](#1-integer-representation-overview)
2. [Sign-Magnitude Representation](#2-sign-magnitude-representation)
3. [One's Complement Representation](#3-ones-complement-representation)
4. [Two's Complement Representation](#4-twos-complement-representation)
5. [Integer Overflow](#5-integer-overflow)
6. [Fixed-Point Representation](#6-fixed-point-representation)
7. [IEEE 754 Floating-Point](#7-ieee-754-floating-point)
8. [Floating-Point Precision Issues](#8-floating-point-precision-issues)
9. [Practice Problems](#9-practice-problems)

---

## 1. Integer Representation Overview

### Unsigned Integer

```
n-bit unsigned integer:
- Range: 0 ~ 2ⁿ - 1
- All bits used to represent magnitude

8-bit example:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  2⁷ │  2⁶ │  2⁵ │  2⁴ │  2³ │  2² │  2¹ │  2⁰ │
│ 128 │  64 │  32 │  16 │  8  │  4  │  2  │  1  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

00000000 = 0
11111111 = 255
Range: 0 ~ 255 (256 values)
```

### Signed Integer Representation Methods

```
┌─────────────────────────────────────────────────────────────┐
│          Signed Integer Representation Comparison            │
├───────────────┬───────────────────────────────────────────────┤
│    Method     │              Characteristics                  │
├───────────────┼───────────────────────────────────────────────┤
│  Sign-        │  MSB is sign, rest is magnitude              │
│  Magnitude    │  Both +0 and -0 exist, complex arithmetic    │
├───────────────┼───────────────────────────────────────────────┤
│  One's        │  Negative = bitwise invert of positive       │
│  Complement   │  Both +0 and -0 exist                        │
├───────────────┼───────────────────────────────────────────────┤
│  Two's        │  Negative = one's complement + 1             │
│  Complement   │  Unique zero, efficient arithmetic (modern   │
│               │  standard)                                    │
└───────────────┴───────────────────────────────────────────────┘
```

---

## 2. Sign-Magnitude Representation

### Structure

```
Sign-Magnitude Representation:

In n bits:
┌─────────┬──────────────────────────────────────────┐
│  Sign   │              Magnitude                    │
│ (1 bit) │              (n-1 bits)                   │
└─────────┴──────────────────────────────────────────┘
    ↓
  0 = positive
  1 = negative

8-bit example:
+45 = 0 0101101
       ↑ └─────┘
     sign  magnitude(45)

-45 = 1 0101101
       ↑ └─────┘
     sign  magnitude(45)
```

### Advantages and Disadvantages

```
┌─────────────────────────────────────────────────────────────┐
│          Sign-Magnitude Representation Characteristics       │
├─────────────────────────────────────────────────────────────┤
│  Advantages:                                                │
│  - Intuitive and easy to understand                         │
│  - Simple sign checking (just check MSB)                    │
│  - Easy absolute value calculation                          │
│                                                             │
│  Disadvantages:                                             │
│  - Two zeros: +0 (00000000) and -0 (10000000)              │
│  - Complex addition/subtraction (sign comparison required)  │
│  - Complex hardware implementation                          │
├─────────────────────────────────────────────────────────────┤
│  n-bit range: -(2^(n-1) - 1) ~ +(2^(n-1) - 1)             │
│  8-bit range: -127 ~ +127 (255 values, two zeros)          │
└─────────────────────────────────────────────────────────────┘
```

### 8-Bit Sign-Magnitude Examples

```
┌──────────┬────────────┬──────────┐
│ Decimal  │   Binary   │  Note    │
├──────────┼────────────┼──────────┤
│   +127   │  01111111  │ Maximum  │
│   +45    │  00101101  │          │
│   +1     │  00000001  │          │
│   +0     │  00000000  │ Positive │
│   -0     │  10000000  │ Negative │
│   -1     │  10000001  │          │
│   -45    │  10101101  │          │
│   -127   │  11111111  │ Minimum  │
└──────────┴────────────┴──────────┘
```

---

## 3. One's Complement Representation

### Structure

```
One's Complement Representation:

Positive: represented as is
Negative: all bits inverted

8-bit example:
+45 = 00101101

-45 = 11010010  (all bits inverted)
      ↓↓↓↓↓↓↓↓
      00101101 → 11010010
```

### Bit Inversion Process

```
Positive → Negative conversion:

  +45 = 0 0 1 0 1 1 0 1
        ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓  (invert each bit)
  -45 = 1 1 0 1 0 0 1 0

Verification:
  00101101 = 45
+ 11010010 = 210
────────────
  11111111 = 255 = 2⁸ - 1 ✓

One's complement property: N + (-N) = 2ⁿ - 1
```

### One's Complement Addition

```
One's complement addition feature: End-around carry

Example: 5 + (-3) in 4 bits

  5 = 0101
 -3 = 1100 (one's complement of 3)

    0 1 0 1  (+5)
  + 1 1 0 0  (-3)
  ─────────
  1 0 0 0 1
  ↑
  End-around carry (add it back)

    0 0 0 1
  +       1
  ─────────
    0 0 1 0  = +2 ✓
```

### Zero Representation Problem

```
Zero representation in one's complement:

+0 = 00000000
-0 = 11111111 (one's complement of 0)

Both values mean zero but have different bit patterns
→ Requires additional logic for comparisons and branching
```

---

## 4. Two's Complement Representation

### Structure

```
Two's Complement Representation:

Positive: represented as is
Negative: one's complement + 1

8-bit example:
+45 = 00101101

-45 calculation:
  1) One's complement: 11010010
  2) +1:              11010011  ← two's complement of -45
```

### Two's Complement Calculation Methods

```
Method 1: One's complement + 1

  45 = 00101101
       ↓↓↓↓↓↓↓↓  invert
       11010010
             + 1
       ────────
 -45 = 11010011

Method 2: Keep from right up to first 1, invert rest

  45 = 0 0 1 0 1 1 0 1
              └───┘ keep
       1 1 0 1 0 1 0 1
       └─────┘ invert

 -45 = 1 1 0 1 0 0 1 1

Method 3: 2ⁿ - N

 -45 = 256 - 45 = 211 = 11010011
```

### Advantages of Two's Complement

```
┌─────────────────────────────────────────────────────────────┐
│              Advantages of Two's Complement                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Unique representation of zero                           │
│     Only 00000000 represents 0                              │
│     -0 = two's complement(00000000) = 00000000 (itself)     │
│                                                             │
│  2. Addition/subtraction use same operation                 │
│     A - B = A + (-B) = A + (two's complement of B)          │
│                                                             │
│  3. No end-around carry needed                              │
│     Simply discard MSB carry                                │
│                                                             │
│  4. Simple hardware implementation                          │
│     Single adder handles both addition and subtraction      │
│                                                             │
│  5. Asymmetric range (one more negative)                    │
│     n bits: -2^(n-1) ~ +2^(n-1) - 1                         │
│     8 bits: -128 ~ +127                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Two's Complement Addition/Subtraction

```
Example 1: Positive + Positive (8-bit)
  45 + 30 = 75

    00101101  (+45)
  + 00011110  (+30)
  ──────────
    01001011  (+75) ✓


Example 2: Positive + Negative
  45 + (-30) = 15

  Two's complement of -30: 11100010

    00101101  (+45)
  + 11100010  (-30)
  ──────────
  1 00001111
  ↑
  Discard carry

  Result: 00001111 = +15 ✓


Example 3: Negative + Negative
  -45 + (-30) = -75

  -45 = 11010011
  -30 = 11100010

    11010011  (-45)
  + 11100010  (-30)
  ──────────
  1 10110101
  ↑
  Discard carry

  Result: 10110101
  Convert to positive: 01001011 = 75
  Therefore: -75 ✓
```

### 8-Bit Two's Complement Table

```
┌──────────┬────────────┬────────────────────────────────────┐
│ Decimal  │   Binary   │           Description              │
├──────────┼────────────┼────────────────────────────────────┤
│   +127   │  01111111  │  Maximum positive                  │
│   +126   │  01111110  │                                    │
│    ...   │    ...     │                                    │
│    +2    │  00000010  │                                    │
│    +1    │  00000001  │                                    │
│     0    │  00000000  │  Unique zero                       │
│    -1    │  11111111  │  All bits are 1                    │
│    -2    │  11111110  │                                    │
│    ...   │    ...     │                                    │
│   -127   │  10000001  │                                    │
│   -128   │  10000000  │  Minimum negative (asymmetric)     │
└──────────┴────────────┴────────────────────────────────────┘

Two's complement number line:

-128  -127        -1   0   1         126  127
  │     │          │   │   │           │    │
  └─────┴──── ─────┴───┴───┴───────────┴────┘
10000000     11111111 00000000 00000001  01111111
```

### Comparison of Three Methods

```
4-bit representation comparison:

┌────────┬────────────┬────────────┬────────────┐
│Decimal │Sign-Mag.   │1's Comp.   │2's Comp.   │
├────────┼────────────┼────────────┼────────────┤
│   +7   │   0111     │   0111     │   0111     │
│   +6   │   0110     │   0110     │   0110     │
│   +5   │   0101     │   0101     │   0101     │
│   +4   │   0100     │   0100     │   0100     │
│   +3   │   0011     │   0011     │   0011     │
│   +2   │   0010     │   0010     │   0010     │
│   +1   │   0001     │   0001     │   0001     │
│   +0   │   0000     │   0000     │   0000     │
│   -0   │   1000     │   1111     │   N/A      │
│   -1   │   1001     │   1110     │   1111     │
│   -2   │   1010     │   1101     │   1110     │
│   -3   │   1011     │   1100     │   1101     │
│   -4   │   1100     │   1011     │   1100     │
│   -5   │   1101     │   1010     │   1011     │
│   -6   │   1110     │   1001     │   1010     │
│   -7   │   1111     │   1000     │   1001     │
│   -8   │   N/A      │   N/A      │   1000     │
└────────┴────────────┴────────────┴────────────┘

Ranges:
  Sign-magnitude: -7 ~ +7 (15 values, two zeros)
  One's complement: -7 ~ +7 (15 values, two zeros)
  Two's complement: -8 ~ +7 (16 values, one zero)
```

---

## 5. Integer Overflow

### What is Overflow?

```
┌─────────────────────────────────────────────────────────────┐
│                        Overflow                              │
├─────────────────────────────────────────────────────────────┤
│  When operation result exceeds representable range           │
│                                                             │
│  8-bit two's complement example:                            │
│  - Range: -128 ~ +127                                       │
│  - 127 + 1 = ? (exceeds range!)                             │
└─────────────────────────────────────────────────────────────┘

Positive overflow:
    01111111  (+127)
  + 00000001  (+1)
  ──────────
    10000000  (-128)  ← Unexpected negative!

Negative overflow (underflow):
    10000000  (-128)
  - 00000001  (+1)
  ──────────
    01111111  (+127)  ← Unexpected positive!
```

### Overflow Detection

```
Overflow detection conditions for signed integers:

1. Positive + Positive = Negative → Overflow
2. Negative + Negative = Positive → Overflow
3. Positive + Negative → No overflow possible

┌─────────────────────────────────────────────────────────────┐
│              Overflow Detection (V flag)                     │
├─────────────────────────────────────────────────────────────┤
│  V = C_in(MSB) XOR C_out(MSB)                               │
│                                                             │
│  C_in:  Carry input to MSB                                  │
│  C_out: Carry output from MSB                               │
│                                                             │
│  V = 1 → Overflow occurred                                  │
│  V = 0 → No overflow                                        │
└─────────────────────────────────────────────────────────────┘

Example: 127 + 1 (8-bit)
      01111111
    + 00000001
    ──────────
C_in →  1 (carry from bit 6 to bit 7)
C_out → 0 (no carry out from bit 7)

V = 1 XOR 0 = 1 → Overflow!
```

### Overflow in Programming

```c
// C language example
#include <stdio.h>
#include <limits.h>

int main() {
    // Signed integer overflow (undefined behavior!)
    int max = INT_MAX;  // 2147483647
    printf("%d + 1 = %d\n", max, max + 1);  // Unpredictable

    // Unsigned integer overflow (defined behavior, wraps around)
    unsigned int umax = UINT_MAX;  // 4294967295
    printf("%u + 1 = %u\n", umax, umax + 1);  // 0

    return 0;
}
```

```python
# Python example - arbitrary precision integers
a = 2**63 - 1  # 9223372036854775807
b = a + 1      # 9223372036854775808 (no overflow!)

# Python automatically handles large integers
huge = 10**100
print(huge)  # Prints normally
```

### Overflow Prevention Strategies

```
┌─────────────────────────────────────────────────────────────┐
│              Overflow Prevention Strategies                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Check before operation                                  │
│     if (a > 0 && b > INT_MAX - a) → Overflow will occur    │
│                                                             │
│  2. Use larger data types                                   │
│     long instead of int, long long instead of long         │
│                                                             │
│  3. Use arbitrary precision libraries                       │
│     Python, Java BigInteger, GMP, etc.                     │
│                                                             │
│  4. Use compiler options                                    │
│     -ftrapv (GCC): Halt program on overflow                │
│                                                             │
│  5. Safe integer arithmetic libraries                       │
│     SafeInt (C++), checked_add (Rust)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Fixed-Point Representation

### Fixed-Point Concept

```
Fixed-Point:
Representation with fixed decimal point position

Q notation: Qm.n
- m: Number of integer bits
- n: Number of fractional bits
- Total bits = m + n (excluding sign) or 1 + m + n (with sign)

8-bit Q3.4 example (unsigned):
┌───────────────────────────────────────┐
│  Integer (3 bits)  │ Fraction (4 bits)│
└───────────────────────────────────────┘
    2²  2¹  2⁰    .   2⁻¹ 2⁻² 2⁻³ 2⁻⁴
    4   2   1     .   0.5 0.25 0.125 0.0625

Example: 01011010 in Q3.4
= 0×4 + 1×2 + 0×1 + 1×0.5 + 1×0.25 + 0×0.125 + 1×0.0625 + 0×0.03125
= 2 + 0.5 + 0.25 + 0.0625
= 2.8125
```

### Fixed-Point Operations

```
Fixed-point addition/subtraction:
- Same Q format allows integer-like operations

Fixed-point multiplication:
- Result fractional bits = sum of operand fractional bits
- Requires increased result bits to prevent overflow

Q4.4 × Q4.4 = Q8.8 (8-bit × 8-bit = 16-bit)

Example:
  2.5 × 1.5 (Q4.4)
  2.5 = 00101000 (0010.1000)
  1.5 = 00011000 (0001.1000)

  Multiplication result: 00101000 × 00011000 = 0000001011100000 (Q8.8)
  = 3.75 (0000001111.00000000)
```

---

## 7. IEEE 754 Floating-Point

### Floating-Point Concept

```
Floating-Point:
Representation with flexible decimal point position

Similar to scientific notation:
6.022 × 10²³ (Avogadro's number)
  ↑     ↑ ↑
mantissa base exponent

Binary floating-point:
1.01011 × 2⁵
  ↑        ↑
mantissa exponent
```

### IEEE 754 Format

```
IEEE 754 structure:

┌─────────┬────────────────────┬───────────────────────────────┐
│  Sign   │     Exponent       │         Mantissa               │
│ (Sign)  │    (Exponent)      │         (Mantissa)            │
│ 1 bit   │     E bits         │          M bits               │
└─────────┴────────────────────┴───────────────────────────────┘

Value = (-1)^S × 1.M × 2^(E - bias)

┌───────────────┬───────────┬───────────┬───────────┬────────────┐
│    Format     │ Total Bits│ Exponent(E)│ Mantissa(M)│   Bias    │
├───────────────┼───────────┼───────────┼───────────┼────────────┤
│  Single       │    32     │     8     │    23     │    127     │
│  Precision    │           │           │           │            │
├───────────────┼───────────┼───────────┼───────────┼────────────┤
│  Double       │    64     │    11     │    52     │    1023    │
│  Precision    │           │           │           │            │
├───────────────┼───────────┼───────────┼───────────┼────────────┤
│  Half         │    16     │     5     │    10     │     15     │
│  Precision    │           │           │           │            │
└───────────────┴───────────┴───────────┴───────────┴────────────┘
```

### Single Precision (32-bit) Details

```
32-bit single precision (float):

  31 30    23 22                    0
┌───┬────────┬───────────────────────┐
│ S │   E    │          M           │
│ 1 │   8    │         23           │
└───┴────────┴───────────────────────┘

Value = (-1)^S × 1.M × 2^(E - 127)

Example: Convert -6.625 to single precision

Step 1: Binary conversion
  6 = 110₂
  0.625 = 0.101₂
  6.625 = 110.101₂

Step 2: Normalize
  110.101 = 1.10101 × 2²

Step 3: Determine each field
  S = 1 (negative)
  E = 2 + 127 = 129 = 10000001₂
  M = 10101000000000000000000 (after decimal point, padded to 23 bits)

Result: 1 10000001 10101000000000000000000
     = 0xC0D40000
```

### Double Precision (64-bit) Details

```
64-bit double precision (double):

  63 62         52 51                                  0
┌───┬─────────────┬──────────────────────────────────────┐
│ S │      E      │                  M                   │
│ 1 │     11      │                 52                   │
└───┴─────────────┴──────────────────────────────────────┘

Value = (-1)^S × 1.M × 2^(E - 1023)

Range comparison:
┌─────────────┬───────────────────────────────────────────────┐
│   Format    │                   Range                        │
├─────────────┼───────────────────────────────────────────────┤
│  Single     │  ±1.18×10⁻³⁸ ~ ±3.4×10³⁸                      │
│  (float)    │  ~7 significant digits                        │
├─────────────┼───────────────────────────────────────────────┤
│  Double     │  ±2.23×10⁻³⁰⁸ ~ ±1.8×10³⁰⁸                    │
│  (double)   │  ~15-16 significant digits                    │
└─────────────┴───────────────────────────────────────────────┘
```

### Special Values

```
IEEE 754 special values:

┌───────────────────────────────────────────────────────────────┐
│                   Special Value Representation                 │
├─────────┬──────────────┬─────────────┬────────────────────────┤
│  Value  │ Exponent(E)  │ Mantissa(M) │       Meaning          │
├─────────┼──────────────┼─────────────┼────────────────────────┤
│  +0     │  00000000    │  000...0    │  Positive zero         │
│  -0     │  00000000    │  000...0    │  Negative zero (S=1)   │
├─────────┼──────────────┼─────────────┼────────────────────────┤
│  +∞     │  11111111    │  000...0    │  Positive infinity     │
│  -∞     │  11111111    │  000...0    │  Negative infinity(S=1)│
├─────────┼──────────────┼─────────────┼────────────────────────┤
│  NaN    │  11111111    │  ≠ 0       │  Not a Number          │
│         │              │             │  (0/0, ∞-∞, etc.)     │
├─────────┼──────────────┼─────────────┼────────────────────────┤
│ Denorm  │  00000000    │  ≠ 0       │  Very small numbers    │
│         │              │             │  0.M × 2^(-126)        │
└─────────┴──────────────┴─────────────┴────────────────────────┘

NaN properties:
- NaN ≠ NaN (not even equal to itself)
- All comparisons with NaN are false
- All operations with NaN result in NaN
```

### Floating-Point Operations

```
Addition process:
1. Align exponents (adjust smaller to larger)
2. Add mantissas
3. Normalize
4. Round

Example: 1.5 + 0.25 (single precision)

1.5  = 1.1 × 2⁰
0.25 = 1.0 × 2⁻²

Align exponents:
0.25 → 0.01 × 2⁰

Addition:
  1.10
+ 0.01
──────
  1.11 × 2⁰ = 1.75 ✓


Multiplication process:
1. Determine sign (XOR)
2. Add exponents (adjust bias)
3. Multiply mantissas
4. Normalize
5. Round

Example: 1.5 × 2.0

1.5 = 1.1 × 2⁰
2.0 = 1.0 × 2¹

Sign: positive × positive = positive
Exponent: 0 + 1 = 1
Mantissa: 1.1 × 1.0 = 1.1

Result: 1.1 × 2¹ = 3.0 ✓
```

---

## 8. Floating-Point Precision Issues

### Representation Errors

```
┌─────────────────────────────────────────────────────────────┐
│                Representation Error Problem                  │
├─────────────────────────────────────────────────────────────┤
│  Many decimal fractions cannot be exactly represented        │
│  in binary                                                   │
│                                                             │
│  Example: 0.1 (decimal)                                     │
│  = 0.0001100110011001100110011... (binary, infinite repeat)│
│                                                             │
│  Truncated when stored, causing error                       │
└─────────────────────────────────────────────────────────────┘

Exactly representable:
- Numbers expressible as sums of powers of 2
- Examples: 0.5 (2⁻¹), 0.25 (2⁻²), 0.75 (2⁻¹ + 2⁻²)

Not exactly representable:
- Most decimal fractions
- Examples: 0.1, 0.2, 0.3, etc.
```

### Code Examples

```python
# Python example
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False

>>> format(0.1, '.20f')
'0.10000000000000000555'

>>> format(0.2, '.20f')
'0.20000000000000001110'

>>> format(0.1 + 0.2, '.20f')
'0.30000000000000004441'
```

```c
// C example
#include <stdio.h>

int main() {
    float a = 0.1f;
    float sum = 0.0f;

    for (int i = 0; i < 10; i++) {
        sum += a;
    }

    printf("0.1 * 10 = %.10f\n", sum);
    // Output: 0.1 * 10 = 1.0000001192
    // Expected: 1.0000000000

    return 0;
}
```

### Precision Loss Situations

```
┌─────────────────────────────────────────────────────────────┐
│                Precision Loss Situations                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Adding large and small numbers                          │
│     1000000.0 + 0.0000001 ≈ 1000000.0                      │
│     (small number outside significant digit range)          │
│                                                             │
│  2. Subtracting similar-sized numbers                       │
│     (Catastrophic Cancellation)                             │
│     1.0000001 - 1.0000000 = 0.0000001                      │
│     Significant digits drastically reduced                  │
│                                                             │
│  3. Repeated operations                                     │
│     Small errors accumulate to large errors                 │
│                                                             │
│  4. Infinite loop risk                                      │
│     for (float x = 0.0f; x != 1.0f; x += 0.1f)             │
│     → May never terminate!                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Solutions

```
┌─────────────────────────────────────────────────────────────┐
│              Solutions to Precision Problems                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Use epsilon tolerance                                   │
│     |a - b| < epsilon  instead of  a == b                   │
│                                                             │
│  2. Use integer arithmetic                                  │
│     Money: $1.99 → store as 199 cents                      │
│                                                             │
│  3. Use Decimal type                                        │
│     Python: from decimal import Decimal                    │
│     Java: BigDecimal                                        │
│                                                             │
│  4. Optimize operation order                                │
│     Add small numbers first, then larger numbers           │
│                                                             │
│  5. Kahan Summation (compensated summation)                 │
│     Track and compensate for accumulated error             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Epsilon Comparison Examples

```python
# Python - epsilon comparison
import math

def float_equals(a, b, rel_tol=1e-9, abs_tol=1e-9):
    """Floating-point comparison"""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# Or use math.isclose from Python 3.5+
print(math.isclose(0.1 + 0.2, 0.3))  # True

# Using Decimal
from decimal import Decimal, getcontext

getcontext().prec = 50  # Set precision

a = Decimal('0.1')
b = Decimal('0.2')
c = Decimal('0.3')

print(a + b == c)  # True
```

```c
// C - epsilon comparison
#include <math.h>
#include <float.h>

int float_equals(float a, float b) {
    return fabs(a - b) < FLT_EPSILON * fmax(fabs(a), fabs(b));
}

int double_equals(double a, double b) {
    return fabs(a - b) < DBL_EPSILON * fmax(fabs(a), fabs(b));
}
```

---

## 9. Practice Problems

### Basic Problems

**1. Represent the following numbers in 8-bit two's complement:**
   - (a) +50
   - (b) -50
   - (c) -1
   - (d) -128

**2. Convert the following 8-bit two's complement values to decimal:**
   - (a) 01100100
   - (b) 11001110
   - (c) 10000000
   - (d) 11111111

**3. Calculate the following operations in 8-bit two's complement:**
   - (a) 45 + 30
   - (b) 45 - 30
   - (c) -45 - 30
   - (d) -1 + 1

### IEEE 754 Problems

**4. Convert the following decimal numbers to IEEE 754 single precision (32-bit):**
   - (a) 5.75
   - (b) -0.375
   - (c) 1.0

**5. Convert the following IEEE 754 single precision bit patterns to decimal:**
   - (a) 0 10000001 01000000000000000000000
   - (b) 1 01111111 00000000000000000000000
   - (c) 0 00000000 00000000000000000000000

### Advanced Problems

**6. Determine if overflow occurs in the following 8-bit two's complement operations:**
   - (a) 01111111 + 00000001
   - (b) 10000000 + 11111111
   - (c) 01000000 + 01000000

**7. Explain why 0.1 + 0.2 is not exactly 0.3.**

**8. Explain the Kahan Summation algorithm and its advantages compared to regular addition.**

**9. Explain why both +0 and -0 exist in IEEE 754 and their differences.**

**10. Identify the problem in the following code and suggest a solution:**
```c
float sum = 0.0f;
for (int i = 0; i < 1000000; i++) {
    sum += 0.0001f;
}
printf("Expected: 100.0, Actual: %f\n", sum);
```

---

<details>
<summary>Answers</summary>

**1. 8-bit two's complement representation**
- (a) +50 = 00110010
- (b) -50 = 11001110 (two's complement of 50)
- (c) -1 = 11111111
- (d) -128 = 10000000

**2. Two's complement → Decimal**
- (a) 01100100 = +100 (MSB is 0, so positive)
- (b) 11001110 = -50 (MSB is 1, so negative; complement = 00110010 = 50)
- (c) 10000000 = -128
- (d) 11111111 = -1

**3. Two's complement operations**
- (a) 45 + 30 = 00101101 + 00011110 = 01001011 = +75
- (b) 45 - 30 = 00101101 + 11100010 = 00001111 = +15
- (c) -45 - 30 = 11010011 + 11100010 = 10110101 = -75
- (d) -1 + 1 = 11111111 + 00000001 = 00000000 = 0

**4. IEEE 754 single precision conversion**
- (a) 5.75 = 101.11 = 1.0111 × 2² → 0 10000001 01110000000000000000000
- (b) -0.375 = -0.011 = -1.1 × 2⁻² → 1 01111101 10000000000000000000000
- (c) 1.0 = 1.0 × 2⁰ → 0 01111111 00000000000000000000000

**5. IEEE 754 → Decimal**
- (a) S=0, E=129, M=0.25 → +1.25 × 2² = +5.0
- (b) S=1, E=127, M=0 → -1.0 × 2⁰ = -1.0
- (c) S=0, E=0, M=0 → +0.0

**6. Overflow detection**
- (a) 01111111 + 00000001 = 10000000 → Overflow! (positive+positive=negative)
- (b) 10000000 + 11111111 = 01111111 → Overflow! (negative+negative=positive)
- (c) 01000000 + 01000000 = 10000000 → Overflow! (64+64=-128)

**7.** 0.1 and 0.2 cannot be exactly represented in binary floating-point. Both are infinite repeating fractions, so they're truncated when stored, and their errors add up.

**8.** Kahan Summation tracks rounding error from each addition in a separate variable and compensates in the next addition. This minimizes error accumulation when adding many numbers.

**9.** +0 and -0 have different bit patterns but compare as equal. -0 occurs when a very small negative number underflows, and preserves sign information: 1/+0 = +∞, 1/-0 = -∞.

**10.** 0.0001f is not exactly representable in binary, causing errors that accumulate over 1 million iterations. Solutions: use double, Kahan Summation, or integer arithmetic followed by division.

</details>

---

## Next Steps

- [04_Logic_Gates.md](./04_Logic_Gates.md) - Basic logic gates and Boolean algebra

---

## References

- IEEE 754-2019 Standard for Floating-Point Arithmetic
- What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg)
- Computer Organization and Design (Patterson & Hennessy)
- [Float Toy - Interactive IEEE 754 Visualization](https://evanw.github.io/float-toy/)
- [Floating Point Guide](https://floating-point-gui.de/)
