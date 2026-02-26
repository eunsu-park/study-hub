# Data Representation Basics

**Previous**: [Computer System Overview](./01_Computer_System_Overview.md) | **Next**: [Integer and Floating-Point Representation](./03_Integer_Float_Representation.md)

**Difficulty**: ⭐ (Basic)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Convert numbers between binary, octal, decimal, and hexadecimal
2. Explain why computers use binary representation
3. Perform binary addition and subtraction with carry/borrow
4. Represent negative numbers using sign-magnitude, 1's complement, and 2's complement
5. Explain overflow detection in binary arithmetic
6. Distinguish between data units (bit, nibble, byte, word) and their roles

---

Computers do not understand decimal numbers, letters, or images -- they only understand sequences of 0s and 1s. Everything in computing, from the color of a pixel to the price of a stock, must be encoded as binary. Understanding data representation reveals why integer overflow bugs happen, why floating-point arithmetic has surprises, and how data is stored at the hardware level.

## Table of Contents

1. [Understanding Number Systems](#1-understanding-number-systems)
2. [Binary (Base-2)](#2-binary-base-2)
3. [Octal (Base-8)](#3-octal-base-8)
4. [Hexadecimal (Base-16)](#4-hexadecimal-base-16)
5. [Base Conversion](#5-base-conversion)
6. [Data Units](#6-data-units)
7. [Concept of Complements](#7-concept-of-complements)
8. [Practice Problems](#8-practice-problems)

---

## 1. Understanding Number Systems

### What is a Number System?

A number system is a way of representing numbers using a base (radix).

```
┌─────────────────────────────────────────────────────────┐
│                  Types of Number Systems                 │
├─────────────┬─────────┬─────────────────────────────────┤
│   System    │  Base   │         Digits Used             │
├─────────────┼─────────┼─────────────────────────────────┤
│  Binary     │   2     │  0, 1                           │
│  Octal      │   8     │  0, 1, 2, 3, 4, 5, 6, 7         │
│  Decimal    │   10    │  0, 1, 2, 3, 4, 5, 6, 7, 8, 9   │
│  Hexadecimal│   16    │  0-9, A, B, C, D, E, F          │
└─────────────┴─────────┴─────────────────────────────────┘
```

### Place Value Principle

In all number systems, each position has a value based on powers of the base.

```
Place values in decimal 1234:
    1    2    3    4
   10³  10²  10¹  10⁰
 =1000 =100  =10   =1

1234 = 1×10³ + 2×10² + 3×10¹ + 4×10⁰
     = 1000 + 200 + 30 + 4
     = 1234

Place values in binary 1101:
    1    1    0    1
   2³   2²   2¹   2⁰
   =8   =4   =2   =1

1101₂ = 1×2³ + 1×2² + 0×2¹ + 1×2⁰
      = 8 + 4 + 0 + 1
      = 13₁₀
```

---

## 2. Binary (Base-2)

### Binary Characteristics

Why computers use binary:

```
┌─────────────────────────────────────────────────────────┐
│           Why Do Computers Use Binary?                  │
├─────────────────────────────────────────────────────────┤
│  1. Corresponds to two states of electrical signals     │
│     (ON/OFF)                                            │
│  2. Robust against noise (only need to distinguish      │
│     two states)                                         │
│  3. Direct correspondence with logic operations         │
│     (true/false)                                        │
│  4. Simpler circuit design                              │
└─────────────────────────────────────────────────────────┘

     Voltage
       │
   5V ─┤  ████████           ████████
       │          │         │
   0V ─┤──────────┴─────────┴────────── Time
       │
         HIGH(1)    LOW(0)   HIGH(1)
```

### Binary Notation

```
Binary notation methods:
- Subscript: 1010₂
- Prefix: 0b1010 (commonly used in programming)
- Suffix: 1010b

Powers of 2 (recommended to memorize):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 2⁰  │ 2¹  │ 2²  │ 2³  │ 2⁴  │ 2⁵  │ 2⁶  │ 2⁷  │ 2⁸  │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  1  │  2  │  4  │  8  │ 16  │ 32  │ 64  │ 128 │ 256 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

┌─────┬─────┬──────┬──────┬───────┐
│ 2⁹  │ 2¹⁰ │ 2¹¹  │ 2¹²  │ 2¹⁶   │
├─────┼─────┼──────┼──────┼───────┤
│ 512 │1024 │ 2048 │ 4096 │ 65536 │
└─────┴─────┴──────┴──────┴───────┘
```

### Binary Addition

```
Binary addition rules:
  0 + 0 = 0
  0 + 1 = 1
  1 + 0 = 1
  1 + 1 = 10 (write 0, carry 1)
  1 + 1 + 1 = 11 (write 1, carry 1)

Example: 1011 + 1101
        1 1     (carry)
        1 0 1 1
      + 1 1 0 1
      ─────────
      1 1 0 0 0

Verification: 11₁₀ + 13₁₀ = 24₁₀ = 11000₂ ✓
```

### Binary Subtraction

```
Binary subtraction rules:
  0 - 0 = 0
  1 - 0 = 1
  1 - 1 = 0
  0 - 1 = 1 (borrow from higher bit)

Example: 1101 - 1011
        0 10    (borrow)
        1 1 0 1
      - 1 0 1 1
      ─────────
        0 0 1 0

Verification: 13₁₀ - 11₁₀ = 2₁₀ = 10₂ ✓
```

---

## 3. Octal (Base-8)

### Octal Characteristics

```
Octal uses digits 0-7.
Since 2³ = 8, three binary digits correspond to one octal digit.

Relationship between binary and octal:
┌────────┬────────┐
│ Binary │ Octal  │
├────────┼────────┤
│  000   │   0    │
│  001   │   1    │
│  010   │   2    │
│  011   │   3    │
│  100   │   4    │
│  101   │   5    │
│  110   │   6    │
│  111   │   7    │
└────────┴────────┘
```

### Octal Notation

```
Octal notation methods:
- Subscript: 753₈
- Prefix: 0o753 (Python) or 0753 (C)

Example: Binary → Octal
110 101 011₂
 6   5   3₈ = 653₈

Example: Octal → Binary
752₈ = 111 101 010₂
```

### Uses of Octal

```
┌─────────────────────────────────────────────────────────┐
│                  Applications of Octal                   │
├─────────────────────────────────────────────────────────┤
│  1. Unix/Linux file permissions (chmod 755)             │
│     - 755 = rwxr-xr-x                                   │
│     - 7 = 111₂ = rwx (read+write+execute)              │
│     - 5 = 101₂ = r-x (read+execute)                    │
│                                                         │
│  2. Legacy computer systems (PDP-8, etc.)               │
│  3. ASCII code representation                           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Hexadecimal (Base-16)

### Hexadecimal Characteristics

```
Hexadecimal uses 0-9 and A-F (10-15).
Since 2⁴ = 16, four binary digits correspond to one hex digit.

Relationship between binary and hexadecimal:
┌────────┬────────┬────────┐
│ Decimal│ Binary │  Hex   │
├────────┼────────┼────────┤
│   0    │  0000  │   0    │
│   1    │  0001  │   1    │
│   2    │  0010  │   2    │
│   3    │  0011  │   3    │
│   4    │  0100  │   4    │
│   5    │  0101  │   5    │
│   6    │  0110  │   6    │
│   7    │  0111  │   7    │
│   8    │  1000  │   8    │
│   9    │  1001  │   9    │
│  10    │  1010  │   A    │
│  11    │  1011  │   B    │
│  12    │  1100  │   C    │
│  13    │  1101  │   D    │
│  14    │  1110  │   E    │
│  15    │  1111  │   F    │
└────────┴────────┴────────┘
```

### Hexadecimal Notation

```
Hexadecimal notation methods:
- Subscript: 2AF₁₆
- Prefix: 0x2AF (most commonly used)
- Suffix: 2AFh

Example: Binary → Hexadecimal
1010 1111 0011₂
 A    F    3₁₆ = 0xAF3

Example: Hexadecimal → Binary
0x3E8 = 0011 1110 1000₂
```

### Uses of Hexadecimal

```
┌─────────────────────────────────────────────────────────┐
│              Applications of Hexadecimal                 │
├─────────────────────────────────────────────────────────┤
│  1. Memory address representation                       │
│     - 0x7FFE1234                                        │
│                                                         │
│  2. Color codes (RGB)                                   │
│     - #FF5733 = Red(FF) Green(57) Blue(33)              │
│     - Each color ranges from 0-255 (0x00-0xFF)          │
│                                                         │
│  3. MAC addresses                                       │
│     - 00:1A:2B:3C:4D:5E                                 │
│                                                         │
│  4. Machine code/assembly language                      │
│     - MOV AX, 0x1234                                    │
│                                                         │
│  5. Unicode/ASCII codes                                 │
│     - 'A' = 0x41 = 65                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Base Conversion

### 5.1 Decimal → Binary Conversion

**Method 1: Division (Integer part)**

```
Convert decimal 25 to binary:

25 ÷ 2 = 12 ... remainder 1  ↑
12 ÷ 2 =  6 ... remainder 0  │
 6 ÷ 2 =  3 ... remainder 0  │ Read bottom to top
 3 ÷ 2 =  1 ... remainder 1  │
 1 ÷ 2 =  0 ... remainder 1  │

Result: 25₁₀ = 11001₂
```

**Method 2: Multiplication (Fractional part)**

```
Convert decimal 0.625 to binary:

0.625 × 2 = 1.25  → integer part 1  ↓
0.25  × 2 = 0.5   → integer part 0  │
0.5   × 2 = 1.0   → integer part 1  │ Read top to bottom
0.0   (done)                        ↓

Result: 0.625₁₀ = 0.101₂
```

### 5.2 Binary → Decimal Conversion

```
Convert binary 110101 to decimal:

   1    1    0    1    0    1
  2⁵   2⁴   2³   2²   2¹   2⁰
  32   16    8    4    2    1

= 1×32 + 1×16 + 0×8 + 1×4 + 0×2 + 1×1
= 32 + 16 + 0 + 4 + 0 + 1
= 53₁₀
```

### 5.3 Binary ↔ Hexadecimal Conversion

```
Binary → Hexadecimal: Group by 4 bits

  1011 1010 0110 1111₂
   B    A    6    F
= 0xBA6F

Hexadecimal → Binary: Expand each digit to 4 bits

0x2F5 = 0010 1111 0101₂
        2    F    5
```

### 5.4 Decimal ↔ Hexadecimal Conversion

```
Decimal → Hexadecimal: Divide by 16

500 ÷ 16 = 31 ... remainder 4   ↑
 31 ÷ 16 =  1 ... remainder 15 (F)  │
  1 ÷ 16 =  0 ... remainder 1   │

500₁₀ = 0x1F4

Hexadecimal → Decimal: Calculate place values

0x1F4 = 1×16² + 15×16¹ + 4×16⁰
      = 256 + 240 + 4
      = 500₁₀
```

### Base Conversion Summary Diagram

```
           ┌──────────────────┐
           │     Decimal      │
           └────────┬─────────┘
          ÷2       │        ÷16
      reverse      │    reverse
     remainders    │   remainders
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              │              ▼
┌───────┐          │         ┌────────┐
│ Binary│←─────────┼─────────→│  Hex   │
└───────┘    group by 4      └────────┘
            expand each
            to 4 bits
```

### Quick Base Conversion Reference Table

```
┌──────────┬──────────┬──────────┬──────────┐
│  Decimal │  Binary  │  Octal   │   Hex    │
├──────────┼──────────┼──────────┼──────────┤
│    0     │   0000   │    0     │    0     │
│    1     │   0001   │    1     │    1     │
│    2     │   0010   │    2     │    2     │
│    3     │   0011   │    3     │    3     │
│    4     │   0100   │    4     │    4     │
│    5     │   0101   │    5     │    5     │
│    6     │   0110   │    6     │    6     │
│    7     │   0111   │    7     │    7     │
│    8     │   1000   │   10     │    8     │
│    9     │   1001   │   11     │    9     │
│   10     │   1010   │   12     │    A     │
│   11     │   1011   │   13     │    B     │
│   12     │   1100   │   14     │    C     │
│   13     │   1101   │   15     │    D     │
│   14     │   1110   │   16     │    E     │
│   15     │   1111   │   17     │    F     │
│   16     │  10000   │   20     │   10     │
│   32     │ 100000   │   40     │   20     │
│   64     │ 1000000  │  100     │   40     │
│  128     │10000000  │  200     │   80     │
│  255     │11111111  │  377     │   FF     │
│  256     │100000000 │  400     │  100     │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 6. Data Units

### Basic Units

```
┌─────────────────────────────────────────────────────────┐
│                  Data Unit Hierarchy                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Bit                                                    │
│    └── 0 or 1, smallest unit of information            │
│                                                         │
│  Nibble                                                 │
│    └── 4 bits = one hexadecimal digit                  │
│                                                         │
│  Byte                                                   │
│    └── 8 bits = can represent one character            │
│                                                         │
│  Word                                                   │
│    └── Data size processed by CPU in one operation     │
│        (16-bit, 32-bit, 64-bit, etc.)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Bits and Bytes

```
1 byte = 8 bits

┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ bit7│ bit6│ bit5│ bit4│ bit3│ bit2│ bit1│ bit0│
│(MSB)│     │     │     │     │     │     │(LSB)│
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  2⁷    2⁶    2⁵    2⁴    2³    2²    2¹    2⁰
 128    64    32    16     8     4     2     1

MSB (Most Significant Bit): Leftmost bit (highest order)
LSB (Least Significant Bit): Rightmost bit (lowest order)

Values representable in 1 byte:
- Unsigned integer: 0 ~ 255 (2⁸ = 256 values)
- Signed integer: -128 ~ 127
```

### Word Size

```
┌──────────────────────────────────────────────────────────┐
│              Word Size by System Type                     │
├────────────────┬──────────┬───────────────────────────────┤
│    System      │Word Size │          Value Range          │
├────────────────┼──────────┼───────────────────────────────┤
│  8-bit system  │  1 byte  │  0 ~ 255                      │
│  16-bit system │  2 bytes │  0 ~ 65,535                   │
│  32-bit system │  4 bytes │  0 ~ 4,294,967,295            │
│  64-bit system │  8 bytes │  0 ~ 18,446,744,073,709,551,615│
└────────────────┴──────────┴───────────────────────────────┘
```

### Large-Scale Units

```
┌────────────────────────────────────────────────────────────┐
│                    Storage Capacity Units                   │
├────────────┬────────────────────┬──────────────────────────┤
│    Unit    │   Binary Prefix    │     Decimal Prefix       │
│            │   (IEC Standard)   │    (SI Standard)         │
├────────────┼────────────────────┼──────────────────────────┤
│  Kilo (K)  │ 1 KiB = 2¹⁰ = 1,024│ 1 KB = 10³ = 1,000      │
│  Mega (M)  │ 1 MiB = 2²⁰        │ 1 MB = 10⁶              │
│  Giga (G)  │ 1 GiB = 2³⁰        │ 1 GB = 10⁹              │
│  Tera (T)  │ 1 TiB = 2⁴⁰        │ 1 TB = 10¹²             │
│  Peta (P)  │ 1 PiB = 2⁵⁰        │ 1 PB = 10¹⁵             │
│  Exa (E)   │ 1 EiB = 2⁶⁰        │ 1 EB = 10¹⁸             │
└────────────┴────────────────────┴──────────────────────────┘

Real capacity difference example:
- 1 TB hard drive (decimal): 1,000,000,000,000 bytes
- OS display (binary): approximately 931 GiB

Calculation: 1,000,000,000,000 ÷ 1,073,741,824 ≈ 931 GiB
```

---

## 7. Concept of Complements

### What is a Complement?

```
A complement is the value obtained by subtracting a number from
a specific reference point. In computers, complements are used
to convert subtraction into addition.

┌─────────────────────────────────────────────────────────┐
│                  Types of Complements                    │
├─────────────────────────────────────────────────────────┤
│  (r-1)'s complement: Subtract each digit from (r-1)     │
│  r's complement: (r-1)'s complement + 1                 │
│                                                         │
│  For binary:                                            │
│  - One's complement: Invert each bit (0↔1)             │
│  - Two's complement: One's complement + 1               │
└─────────────────────────────────────────────────────────┘
```

### One's Complement

```
One's complement: Invert all bits (0→1, 1→0)

Example: One's complement of 00101101 in 8 bits

  Original:     0 0 1 0 1 1 0 1
                ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
  1's comp:     1 1 0 1 0 0 1 0

Verification: 45₁₀ → 1's complement → 210₁₀
      45 + 210 = 255 = 2⁸ - 1 ✓
```

### Two's Complement

```
Two's complement: One's complement + 1
(or keep rightmost 1, invert left side)

Example: Two's complement of 00101101 in 8 bits

Method 1: One's complement + 1
  Original:     0 0 1 0 1 1 0 1  (45)
  1's comp:     1 1 0 1 0 0 1 0  (210)
             +                 1
  ─────────────────────────────────
  2's comp:     1 1 0 1 0 0 1 1  (211)

Method 2: Keep up to first 1 from right, invert the rest
  Original:     0 0 1 0 1 1 0 1
                      ↑
              Keep up to here, invert left side
  2's comp:     1 1 0 1 0 0 1 1

Verification: 45 + 211 = 256 = 2⁸ ✓
```

### Importance of Two's Complement

```
┌─────────────────────────────────────────────────────────┐
│          Why Two's Complement is Important               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Unique representation of zero                       │
│     - One's complement: +0 (00000000), -0 (11111111)    │
│     - Two's complement: 0 (00000000) only               │
│                                                         │
│  2. Converts subtraction to addition                    │
│     A - B = A + (-B) = A + (two's complement of B)      │
│                                                         │
│  3. Single adder can handle both addition/subtraction   │
│                                                         │
│  4. Easy overflow detection                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Subtraction Using Two's Complement

```
Example: Calculate 7 - 3 using 8-bit two's complement

  7 = 00000111
  3 = 00000011

  Two's complement of 3 = 11111101 (= -3)

    0 0 0 0 0 1 1 1   (7)
  + 1 1 1 1 1 1 0 1   (-3, two's complement of 3)
  ─────────────────────
  1 0 0 0 0 0 1 0 0
  ↑
  Discard carry (since 8-bit)

  Result: 00000100 = 4 ✓
```

### Signed Integer Range

```
n-bit two's complement range: -2^(n-1) ~ 2^(n-1) - 1

┌───────────┬──────────────────────────────────────────────┐
│  Bit Size │               Value Range                     │
├───────────┼──────────────────────────────────────────────┤
│  4-bit    │  -8 ~ 7                                      │
│  8-bit    │  -128 ~ 127                                  │
│  16-bit   │  -32,768 ~ 32,767                            │
│  32-bit   │  -2,147,483,648 ~ 2,147,483,647              │
│  64-bit   │  approx -9.2×10¹⁸ ~ 9.2×10¹⁸                 │
└───────────┴──────────────────────────────────────────────┘

8-bit two's complement number circle:

         0 (00000000)
          ↑
    -1 (11111111)  1 (00000001)
         ↖        ↗
          ·      ·
           ·    ·
    -128    ·  ·     127
  (10000000)  (01111111)
```

---

## 8. Practice Problems

### Basic Problems

**1. Convert the following decimal numbers to binary:**
   - (a) 42
   - (b) 100
   - (c) 255

**2. Convert the following binary numbers to decimal:**
   - (a) 10110
   - (b) 11111111
   - (c) 10000000

**3. Convert the following binary numbers to hexadecimal:**
   - (a) 11011010
   - (b) 101111110000
   - (c) 11111111111111111111111111111111

**4. Convert the following hexadecimal numbers to binary:**
   - (a) 0xAB
   - (b) 0x1234
   - (c) 0xDEADBEEF

### Calculation Problems

**5. Perform the following binary additions:**
   - (a) 1011 + 1101
   - (b) 11111111 + 00000001
   - (c) 10101010 + 01010101

**6. In 8-bit two's complement representation:**
   - (a) What is the binary representation of -45?
   - (b) What is the decimal value of 11101100?
   - (c) What is the result of 7 - 12?

**7. Convert the following decimal fractions to binary (up to 4 decimal places):**
   - (a) 0.5
   - (b) 0.25
   - (c) 0.1

### Advanced Problems

**8. What is the difference in bytes between 1 GiB and 1 GB?**

**9. Explain why a 32-bit system is limited to addressing a maximum of 4GB of memory.**

**10. Analyze the following scenario:**
```c
signed char a = 127;
a = a + 1;
// What is the value of a?
```

---

<details>
<summary>Answers</summary>

**1. Decimal → Binary**
- (a) 42 = 101010
- (b) 100 = 1100100
- (c) 255 = 11111111

**2. Binary → Decimal**
- (a) 10110 = 22
- (b) 11111111 = 255
- (c) 10000000 = 128

**3. Binary → Hexadecimal**
- (a) 11011010 = 0xDA
- (b) 101111110000 = 0xBF0
- (c) 11111111111111111111111111111111 = 0xFFFFFFFF

**4. Hexadecimal → Binary**
- (a) 0xAB = 10101011
- (b) 0x1234 = 0001001000110100
- (c) 0xDEADBEEF = 11011110101011011011111011101111

**5. Binary Addition**
- (a) 1011 + 1101 = 11000 (11 + 13 = 24)
- (b) 11111111 + 00000001 = 100000000 (result depends on overflow handling)
- (c) 10101010 + 01010101 = 11111111 (170 + 85 = 255)

**6. Two's Complement**
- (a) -45 = 11010011 (45 = 00101101, complement = 11010011)
- (b) 11101100 = -20 (MSB is 1, so negative; complement = 00010100 = 20)
- (c) 7 - 12 = -5 = 11111011

**7. Decimal Fractions → Binary**
- (a) 0.5 = 0.1
- (b) 0.25 = 0.01
- (c) 0.1 ≈ 0.0001 (actually 0.0001100110011... repeating infinitely)

**8.**
- 1 GiB = 1,073,741,824 bytes
- 1 GB = 1,000,000,000 bytes
- Difference = 73,741,824 bytes ≈ 70.3 MiB

**9.** A 32-bit system uses 32-bit addresses, so it can address 2³² = 4,294,967,296 addresses. Since each address points to 1 byte, the maximum directly accessible memory is 4GB (approximately 4GiB).

**10.** a = -128. The range of signed char is -128~127, so 127+1 causes overflow and wraps to -128. This is because in two's complement representation, 01111111(127) + 1 = 10000000(-128).

</details>

---

## Next Steps

- [03_Integer_Float_Representation.md](./03_Integer_Float_Representation.md) - Detailed integer and floating-point representation

---

## References

- Computer Organization and Design (Patterson & Hennessy)
- Digital Design (Morris Mano)
- [Binary, Hexadecimal, Octal - CS101](https://web.stanford.edu/class/cs101/)
- [Two's Complement - Wikipedia](https://en.wikipedia.org/wiki/Two%27s_complement)
