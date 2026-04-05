; Exercise 01: Getting Started with IDL
;
; Practice basic IDL commands, arithmetic, and system variables.

; Exercise 1: Write a procedure that prints your name and today's date.
; Hint: Use SYSTIME() for the date.
PRO exercise_01a
  ; TODO: Print your name
  ; TODO: Print the current date using SYSTIME()
  ; TODO: Print the IDL version using !VERSION.RELEASE
END

; Exercise 2: Compute and print the circumference and area of a circle
; with radius 7.5. Use !PI for pi.
PRO exercise_01b, radius
  ; TODO: If no argument given, set radius to 7.5
  ; TODO: Compute circumference = 2 * pi * r
  ; TODO: Compute area = pi * r^2
  ; TODO: Print both values formatted to 2 decimal places
  ; Hint: Use FORMAT='("Circumference: ", F8.2)'
END

; Exercise 3: Create an array of 20 values, compute and print
; the sum, mean, min, max, and standard deviation.
PRO exercise_01c
  ; TODO: Create data = FINDGEN(20) + 1.0  (1 through 20)
  ; TODO: Print TOTAL, MEAN, MIN, MAX, STDDEV
  ; Hint: All these are built-in IDL functions
END

; Exercise 4: Convert 98.6 degrees Fahrenheit to Celsius and Kelvin.
; Formula: C = (F - 32) * 5/9, K = C + 273.15
PRO exercise_01d
  ; TODO: Define temp_f = 98.6
  ; TODO: Compute temp_c and temp_k
  ; TODO: Print all three temperatures
  ; Hint: Use 5.0/9.0, not 5/9 (integer division!)
END
