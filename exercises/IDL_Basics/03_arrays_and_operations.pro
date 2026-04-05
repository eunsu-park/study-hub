; Exercise 03: Arrays and Operations
;
; Practice array creation, indexing, WHERE, and array math.

; Exercise 1: Create a 5x5 identity matrix without using a loop.
; Hint: Use INDGEN and array indexing, or DIAG_MATRIX.
PRO exercise_03a
  ; TODO: Create a 5x5 matrix of zeros
  ; TODO: Set the diagonal elements to 1
  ; TODO: Print the result
END

; Exercise 2: Given an array of exam scores, use WHERE to find:
; (a) scores >= 90 (A), (b) scores between 70-89 (B/C),
; (c) scores < 70 (F). Print counts for each.
PRO exercise_03b
  scores = [85, 92, 78, 65, 91, 88, 72, 95, 60, 83, 77, 98]
  ; TODO: Use WHERE to find A grades (>= 90)
  ; TODO: Use WHERE to find B/C grades (70-89)
  ; TODO: Use WHERE to find F grades (< 70)
  ; TODO: Print count and values for each group
END

; Exercise 3: Create two 1D arrays representing x and y coordinates
; of 100 random points. Find all points within distance 0.5 of the
; origin using vectorized operations (no loops!).
PRO exercise_03c
  seed = 42L
  ; TODO: x = RANDOMU(seed, 100) * 2.0 - 1.0  (range -1 to 1)
  ; TODO: y = RANDOMU(seed, 100) * 2.0 - 1.0
  ; TODO: Compute distance = SQRT(x^2 + y^2) vectorized
  ; TODO: Use WHERE to find points within 0.5
  ; TODO: Print count and fraction of total
END

; Exercise 4: Reshape a 1D array of 24 elements into a 2x3x4 cube,
; then extract the 2x3 slice at index 2 of the third dimension.
PRO exercise_03d
  ; TODO: Create flat = INDGEN(24)
  ; TODO: Use REFORM to make a 2x3x4 array
  ; TODO: Extract cube[*, *, 2]
  ; TODO: Print the slice
END
