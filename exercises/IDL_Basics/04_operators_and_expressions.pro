; Exercise 04: Operators and Expressions
;
; Practice arithmetic, relational, logical operators and expressions.

; Exercise 1: FizzBuzz with IDL operators.
; Print numbers 1-30: "Fizz" for multiples of 3,
; "Buzz" for multiples of 5, "FizzBuzz" for both.
PRO exercise_04a
  ; TODO: Loop from 1 to 30
  ; TODO: Use MOD operator to check divisibility
  ; TODO: Print the number or Fizz/Buzz/FizzBuzz
  ; Hint: Check MOD 15 first, then MOD 3, then MOD 5
END

; Exercise 2: Clamp an array of values to the range [-10, 10]
; using the < and > operators (no loops!).
PRO exercise_04b
  data = [-15.0, 8.0, 25.0, -3.0, 12.0, -20.0, 5.0, 0.0]
  ; TODO: Use the > and < operators to clamp to [-10, 10]
  ; TODO: Print original and clamped arrays
  ; Hint: clamped = (-10.0) > data < 10.0
END

; Exercise 3: Classify solar flare flux values using the ternary operator.
PRO exercise_04c
  flux = [3.2e-8, 5.1e-7, 2.3e-6, 8.7e-5, 1.2e-4]
  ; TODO: For each flux value, determine the class (A/B/C/M/X)
  ; TODO: Use the ternary operator (? :)
  ; TODO: Print flux and class for each
  ; Hint: class = (f GE 1e-4) ? 'X' : ((f GE 1e-5) ? 'M' : ...)
END

; Exercise 4: Without using loops, create a boolean mask for a 10x10
; array that is TRUE only in a circular region of radius 3
; centered at (5, 5).
PRO exercise_04d
  ; TODO: Create x and y index arrays using INDGEN and REFORM
  ; TODO: Compute distance from center (5, 5) vectorized
  ; TODO: Create boolean mask where distance LE 3
  ; TODO: Print the mask
  ; Hint: REPLICATE and transpose tricks for 2D index grids
END
