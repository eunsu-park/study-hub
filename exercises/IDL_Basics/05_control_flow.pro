; Exercise 05: Control Flow
;
; Practice IF, FOR, WHILE, REPEAT, CASE, BREAK, CONTINUE.

; Exercise 1: Write a procedure that finds all prime numbers up to N
; using a FOR loop and IF statements.
PRO exercise_05a, n
  IF N_PARAMS() EQ 0 THEN n = 50
  ; TODO: Loop from 2 to n
  ; TODO: For each number, check if it's prime (not divisible by 2..sqrt(n))
  ; TODO: Print all primes found
  ; Hint: Use a nested loop and BREAK when a divisor is found
END

; Exercise 2: Implement Newton's method to find the cube root of 27.
; Use a WHILE loop with a tolerance of 1e-10.
PRO exercise_05b
  target = 27.0D0
  ; TODO: Start with initial guess = target / 3.0
  ; TODO: Iterate: guess = (2*guess + target/guess^2) / 3
  ; TODO: Stop when ABS(guess^3 - target) < 1e-10
  ; TODO: Print the result and number of iterations
END

; Exercise 3: Use CASE to convert month numbers (1-12) to month names.
FUNCTION exercise_05c, month_number
  ; TODO: Use CASE to return the month name string
  ; TODO: Return 'Invalid' for numbers outside 1-12
  ; Hint: CASE month_number OF 1: RETURN, 'January' ...
  RETURN, ''
END

; Exercise 4: Process an array of sensor readings. Skip negative
; values (CONTINUE), stop if value exceeds 1000 (BREAK), and
; accumulate the running average of valid readings.
PRO exercise_05d
  readings = [45.2, 67.8, -1.0, 89.3, 52.1, -5.0, 1200.0, 73.4]
  ; TODO: Loop through readings
  ; TODO: Skip negative values with CONTINUE
  ; TODO: Break if value > 1000
  ; TODO: Track running sum and count
  ; TODO: Print the running average of valid readings before the break
END
