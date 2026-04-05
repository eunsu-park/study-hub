# Control Flow

**Previous**: [Operators and Expressions](./04_Operators_and_Expressions.md) | **Next**: [Procedures and Functions](./06_Procedures_and_Functions.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write IF/THEN/ELSE statements in both single-line and block form
2. Use FOR loops with various step values
3. Implement WHILE and REPEAT/UNTIL loops
4. Select among multiple branches with CASE and SWITCH
5. Use BREAK and CONTINUE for loop control
6. Understand BEGIN...END block syntax
7. Recognize when to use GOTO (and when not to)

---

Control flow statements determine the order in which IDL executes code. IDL provides familiar constructs — conditionals, loops, and multi-way branches — but with syntax that differs from C-family languages. The key thing to remember is that multi-statement blocks require `BEGIN...END` delimiters.

## IF / THEN / ELSE

### Single-Line IF

When you have only one statement to execute, use the single-line form:

```idl
x = 15

; Simple IF
IF x GT 10 THEN PRINT, 'x is greater than 10'

; IF with ELSE
IF x MOD 2 EQ 0 THEN PRINT, 'Even' ELSE PRINT, 'Odd'

; Nested single-line IF (hard to read — prefer block form)
IF x GT 10 THEN IF x LT 20 THEN PRINT, 'Between 10 and 20'
```

### Block IF with BEGIN...END

For multiple statements, use `BEGIN...END` (or `BEGIN...ENDIF` as a shorthand that also closes the IF):

```idl
x = 15

; IF block
IF x GT 10 THEN BEGIN
  PRINT, 'x is greater than 10'
  PRINT, 'x = ', x
ENDIF

; IF/ELSE block
IF x MOD 2 EQ 0 THEN BEGIN
  PRINT, 'x is even'
  half = x / 2
  PRINT, 'Half of x:', half
ENDIF ELSE BEGIN
  PRINT, 'x is odd'
  next_even = x + 1
  PRINT, 'Next even number:', next_even
ENDELSE

; IF/ELSE IF/ELSE chain
score = 85
IF score GE 90 THEN BEGIN
  grade = 'A'
ENDIF ELSE IF score GE 80 THEN BEGIN
  grade = 'B'
ENDIF ELSE IF score GE 70 THEN BEGIN
  grade = 'C'
ENDIF ELSE IF score GE 60 THEN BEGIN
  grade = 'D'
ENDIF ELSE BEGIN
  grade = 'F'
ENDELSE
PRINT, 'Grade:', grade
```

### Block Delimiters

IDL uses matching pairs of BEGIN/END keywords:

| Structure | Begin | End |
|-----------|-------|-----|
| IF...THEN | `BEGIN` | `ENDIF` |
| ELSE | `BEGIN` | `ENDELSE` |
| FOR | `BEGIN` | `ENDFOR` |
| WHILE | `BEGIN` | `ENDWHILE` |
| REPEAT | `BEGIN` | `ENDREP` |
| CASE/SWITCH clause | `BEGIN` | `END` |

You can also use plain `BEGIN...END` everywhere, but the specific keywords (ENDIF, ENDFOR, etc.) help IDL catch mismatched blocks.

---

## FOR Loops

### Basic FOR Loop

```idl
; Count from 0 to 9
FOR i = 0, 9 DO PRINT, i

; Block form
FOR i = 0, 9 DO BEGIN
  PRINT, 'Iteration:', i, '  Square:', i^2
ENDFOR

; With step (increment by 2)
FOR i = 0, 10, 2 DO PRINT, i
; Output: 0  2  4  6  8  10

; Counting down (negative step)
FOR i = 10, 0, -1 DO PRINT, i

; Floating-point loop variable
FOR x = 0.0, 1.0, 0.1 DO PRINT, FORMAT='(F4.1)', x
```

### Iterating Over Arrays

```idl
; Using index variable
names = ['Alice', 'Bob', 'Charlie', 'Diana']
FOR i = 0, N_ELEMENTS(names) - 1 DO BEGIN
  PRINT, 'Name ', STRTRIM(i+1, 2), ': ', names[i]
ENDFOR

; Processing array elements
data = [4.5, 3.2, 7.8, 1.1, 9.6]
sum = 0.0
FOR i = 0, N_ELEMENTS(data) - 1 DO BEGIN
  sum = sum + data[i]
ENDFOR
PRINT, 'Sum:', sum
PRINT, 'Mean:', sum / N_ELEMENTS(data)

; Note: The above loop is much slower than TOTAL(data).
; Use array operations whenever possible!
```

### Nested FOR Loops

```idl
; Create a multiplication table
n = 5
table = INTARR(n, n)
FOR i = 1, n DO BEGIN
  FOR j = 1, n DO BEGIN
    table[i-1, j-1] = i * j
  ENDFOR
ENDFOR
PRINT, table

; Fill a 2D array with distance from center
nx = 10
ny = 10
dist_arr = FLTARR(nx, ny)
cx = nx / 2.0
cy = ny / 2.0
FOR ix = 0, nx-1 DO BEGIN
  FOR iy = 0, ny-1 DO BEGIN
    dist_arr[ix, iy] = SQRT((ix - cx)^2 + (iy - cy)^2)
  ENDFOR
ENDFOR
; Note: DIST(nx, ny) does this much faster!
```

---

## WHILE Loops

The WHILE loop executes as long as a condition is true:

```idl
; Basic WHILE loop
count = 0
WHILE count LT 5 DO BEGIN
  PRINT, 'Count:', count
  count = count + 1
ENDWHILE

; Single-line form
x = 1
WHILE x LT 100 DO x = x * 2
PRINT, 'x:', x    ;      128

; Newton's method for square root of 2
guess = 1.0D0
target = 2.0D0
tolerance = 1.0D-10
iter = 0
WHILE ABS(guess^2 - target) GT tolerance DO BEGIN
  guess = 0.5D0 * (guess + target / guess)
  iter = iter + 1
ENDWHILE
PRINT, FORMAT='("sqrt(2) = ", F18.15, " in ", I0, " iterations")', guess, iter
```

### Avoiding Infinite Loops

```idl
; Always ensure the loop condition will eventually become false
max_iter = 10000
iter = 0
x = 100.0
WHILE x GT 1.0 AND iter LT max_iter DO BEGIN
  x = x / 1.01
  iter = iter + 1
ENDWHILE
IF iter GE max_iter THEN PRINT, 'Warning: Maximum iterations reached'
PRINT, 'Final x:', x, ' after', iter, ' iterations'
```

---

## REPEAT / UNTIL

REPEAT executes the body at least once, then checks the condition:

```idl
; Basic REPEAT/UNTIL (do-while equivalent)
count = 0
REPEAT BEGIN
  PRINT, 'Count:', count
  count = count + 1
ENDREP UNTIL count GE 5

; Single-line form
x = 1
REPEAT x = x * 2 UNTIL x GE 100
PRINT, 'x:', x    ;      128

; Input validation pattern (conceptual — no actual input in batch)
attempts = 0
password = ''
REPEAT BEGIN
  ; In interactive mode: password = ''
  ; READ, password, PROMPT='Enter password: '
  password = (['wrong', 'wrong', 'secret'])[attempts < 2 ? attempts : 2]
  attempts = attempts + 1
  IF password NE 'secret' THEN PRINT, 'Incorrect, try again.'
ENDREP UNTIL password EQ 'secret' OR attempts GE 3
```

---

## CASE Statement

CASE is a multi-way branch that executes the first matching clause and then exits:

```idl
; Basic CASE
day = 3
CASE day OF
  1: PRINT, 'Monday'
  2: PRINT, 'Tuesday'
  3: PRINT, 'Wednesday'
  4: PRINT, 'Thursday'
  5: PRINT, 'Friday'
  6: PRINT, 'Saturday'
  7: PRINT, 'Sunday'
  ELSE: PRINT, 'Invalid day'
ENDCASE

; CASE with blocks
grade = 'B'
CASE grade OF
  'A': BEGIN
    PRINT, 'Excellent!'
    PRINT, 'Score: 90-100'
  END
  'B': BEGIN
    PRINT, 'Good!'
    PRINT, 'Score: 80-89'
  END
  'C': BEGIN
    PRINT, 'Average'
    PRINT, 'Score: 70-79'
  END
  ELSE: BEGIN
    PRINT, 'Below average'
    PRINT, 'Score: below 70'
  END
ENDCASE
```

### CASE with Expressions

```idl
; CASE can match expressions, not just constants
x = 42
CASE 1 OF
  (x LT 0):   PRINT, 'Negative'
  (x EQ 0):   PRINT, 'Zero'
  (x LE 10):  PRINT, 'Small positive'
  (x LE 100): PRINT, 'Medium positive'
  ELSE:        PRINT, 'Large positive'
ENDCASE
; Output: Medium positive
```

---

## SWITCH Statement

SWITCH is similar to CASE but falls through to subsequent clauses (like C's switch without break):

```idl
; SWITCH falls through — rarely used intentionally
level = 2
SWITCH level OF
  1: PRINT, 'Level 1 processing'
  2: PRINT, 'Level 2 processing'
  3: PRINT, 'Level 3 processing'
  ELSE: PRINT, 'Unknown level'
ENDSWITCH
; Output:
; Level 2 processing
; Level 3 processing
; Unknown level

; Use BREAK to prevent fallthrough
level = 2
SWITCH level OF
  1: BEGIN
    PRINT, 'Level 1 only'
    BREAK
  END
  2: BEGIN
    PRINT, 'Level 2 only'
    BREAK
  END
  3: BEGIN
    PRINT, 'Level 3 only'
    BREAK
  END
  ELSE: PRINT, 'Unknown level'
ENDSWITCH
; Output: Level 2 only
```

**Recommendation**: Use CASE instead of SWITCH in most situations. CASE does not fall through and is less error-prone.

---

## BREAK and CONTINUE

### BREAK

`BREAK` exits the innermost loop or CASE/SWITCH:

```idl
; Exit a FOR loop early
FOR i = 0, 100 DO BEGIN
  IF i^2 GT 50 THEN BREAK
  PRINT, i, i^2
ENDFOR
; Prints: 0,0  1,1  2,4  3,9  4,16  5,25  6,36  7,49  (stops at 8)

; Break out of a WHILE loop
data = [3, 7, 2, 9, 4, 1, 8]
target = 9
found = 0
FOR i = 0, N_ELEMENTS(data) - 1 DO BEGIN
  IF data[i] EQ target THEN BEGIN
    found = 1
    PRINT, 'Found', target, 'at index', i
    BREAK
  ENDIF
ENDFOR
IF found EQ 0 THEN PRINT, 'Not found'
```

### CONTINUE

`CONTINUE` skips the rest of the current iteration and moves to the next:

```idl
; Skip even numbers
FOR i = 0, 9 DO BEGIN
  IF i MOD 2 EQ 0 THEN CONTINUE
  PRINT, 'Odd:', i
ENDFOR
; Output: Odd: 1  Odd: 3  Odd: 5  Odd: 7  Odd: 9

; Skip invalid data
data = [1.0, !VALUES.F_NAN, 3.0, !VALUES.F_NAN, 5.0]
sum = 0.0
count = 0
FOR i = 0, N_ELEMENTS(data) - 1 DO BEGIN
  IF ~FINITE(data[i]) THEN CONTINUE
  sum = sum + data[i]
  count = count + 1
ENDFOR
PRINT, 'Mean of valid data:', sum / count
```

---

## GOTO

GOTO transfers control to a labeled statement. It is generally discouraged but occasionally useful for error handling in older IDL code:

```idl
; Basic GOTO (avoid in new code)
x = -5
IF x LT 0 THEN GOTO, handle_error
PRINT, 'Processing x =', x
PRINT, SQRT(x)
GOTO, done

handle_error:
PRINT, 'Error: x cannot be negative'

done:
PRINT, 'Finished'
```

### Error Handling with GOTO (Legacy Pattern)

```idl
; Traditional IDL error handling
PRO read_data_file, filename
  ON_IOERROR, io_error

  OPENR, lun, filename, /GET_LUN
  data = FLTARR(100)
  READF, lun, data
  FREE_LUN, lun

  PRINT, 'Data read successfully'
  PRINT, 'Mean:', MEAN(data)
  RETURN

  io_error:
  PRINT, 'Error reading file: ' + filename
  IF N_ELEMENTS(lun) GT 0 THEN FREE_LUN, lun
END
```

**Note**: Modern IDL (8+) supports `CATCH` for structured error handling, which is preferred over GOTO/ON_IOERROR.

---

## Line Continuation

IDL uses `$` at the end of a line to continue on the next line:

```idl
; Long expressions
result = SIN(x) * COS(y) + $
         TAN(z) * ALOG(w) - $
         SQRT(ABS(v))

; Long IF conditions
IF (temperature GT 1.0e6) AND $
   (density GT 1.0e8) AND $
   (velocity GT 1.0e5) THEN BEGIN
  PRINT, 'Active region detected'
ENDIF

; Long procedure calls
PLOT, time, flux, $
  TITLE='Solar X-ray Flux', $
  XTITLE='Time (hours)', $
  YTITLE='Flux (W/m^2)', $
  XRANGE=[0, 24], $
  YRANGE=[1e-8, 1e-3], $
  /YLOG
```

---

## Practical Examples

### Data Quality Filtering

```idl
; Filter observational data based on quality flags
n_obs = 100
flux = RANDOMN(seed, n_obs) * 100.0 + 500.0
quality = FIX(RANDOMU(seed, n_obs) * 4)  ; 0=good, 1=suspect, 2=bad, 3=missing

good_count = 0
bad_count = 0
FOR i = 0, n_obs - 1 DO BEGIN
  CASE quality[i] OF
    0: BEGIN
      good_count = good_count + 1
    END
    1: BEGIN
      ; Flag suspect data
      flux[i] = flux[i] * 0.9  ; Apply correction factor
      good_count = good_count + 1
    END
    2: BEGIN
      flux[i] = !VALUES.F_NAN
      bad_count = bad_count + 1
    END
    3: BEGIN
      flux[i] = !VALUES.F_NAN
      bad_count = bad_count + 1
    END
  ENDCASE
ENDFOR

PRINT, 'Good observations:', good_count
PRINT, 'Bad observations:', bad_count
good = WHERE(FINITE(flux), n_good)
IF n_good GT 0 THEN PRINT, 'Mean flux:', MEAN(flux[good])
```

### Convergence Loop

```idl
; Iteratively solve for equilibrium temperature
; Stefan-Boltzmann: P = sigma * A * T^4
sigma = 5.67D-8      ; Stefan-Boltzmann constant
solar_flux = 1361.0D0 ; Solar constant (W/m^2)
albedo = 0.3D0        ; Earth's albedo

; Initial guess
T = 250.0D0
tolerance = 0.001D0
max_iter = 100

FOR iter = 1, max_iter DO BEGIN
  ; Absorbed power = emitted power
  P_absorbed = solar_flux * (1.0D0 - albedo) / 4.0D0
  P_emitted = sigma * T^4

  ; Adjust temperature
  T_new = (P_absorbed / sigma)^0.25D0

  IF ABS(T_new - T) LT tolerance THEN BEGIN
    PRINT, FORMAT='("Converged in ", I0, " iterations")', iter
    PRINT, FORMAT='("Equilibrium temperature: ", F7.2, " K")', T_new
    PRINT, FORMAT='("Equilibrium temperature: ", F7.2, " C")', T_new - 273.15D0
    BREAK
  ENDIF

  T = T_new
ENDFOR
```

---

## Summary

| Construct | Syntax | Notes |
|-----------|--------|-------|
| IF/THEN | `IF cond THEN stmt` | Single-line |
| IF/THEN block | `IF cond THEN BEGIN...ENDIF` | Multi-line |
| IF/ELSE | `IF cond THEN BEGIN...ENDIF ELSE BEGIN...ENDELSE` | Block with else |
| FOR | `FOR var=start, stop [, step] DO BEGIN...ENDFOR` | Counted loop |
| WHILE | `WHILE cond DO BEGIN...ENDWHILE` | Pre-test loop |
| REPEAT/UNTIL | `REPEAT BEGIN...ENDREP UNTIL cond` | Post-test loop |
| CASE | `CASE expr OF val: stmt ... ENDCASE` | No fallthrough |
| SWITCH | `SWITCH expr OF val: stmt ... ENDSWITCH` | Falls through |
| BREAK | `BREAK` | Exit loop/case |
| CONTINUE | `CONTINUE` | Skip to next iteration |
| Line continuation | `$` | Continue on next line |

---

**Previous**: [Operators and Expressions](./04_Operators_and_Expressions.md) | **Next**: [Procedures and Functions](./06_Procedures_and_Functions.md)
