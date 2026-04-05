;+
; Exercise 14: Performance and Large Data
;-

PRO exercise_14

    ; === Exercise 1: Vectorization challenge ===
    ; Rewrite these loops as vectorized operations and benchmark both:
    ; (a) Set all negative values to zero
    ; (b) Compute running mean (width 5) of a 1D array
    ; (c) Apply a threshold: out[i] = data[i] if data[i] > mean(data), else 0
    ; TODO: Implement loop and vectorized versions, compare with SYSTIME(1)
    ; Hint: (a) data > 0.0 or WHERE + assignment

    ; === Exercise 2: TEMPORARY optimization ===
    ; Compute: result = sqrt((a - mean(a))^2 + (b - mean(b))^2)
    ; for a, b = FLTARR(2000, 2000)
    ; Method 1: Without TEMPORARY
    ; Method 2: With TEMPORARY for each intermediate step
    ; Compare peak memory usage with MEMORY(/HIGHWATER)
    ; TODO: Implement both, compare MEMORY

    ; === Exercise 3: ASSOC for large files ===
    ; Create a 512x512x100 binary file on disk
    ; Using ASSOC, compute the maximum value in each frame
    ; WITHOUT loading the entire file into memory
    ; TODO: Write file, ASSOC, loop through frames

    ; === Exercise 4: Batch processing template ===
    ; Write a procedure that processes N=50 synthetic "files":
    ; - Generate random 128x128 data for each
    ; - Apply SMOOTH(5)
    ; - Compute and store mean of each
    ; - Print progress every 10 files with ETA
    ; TODO: Implement with SYSTIME for progress tracking

    ; === Exercise 5: SAVE/RESTORE caching ===
    ; Implement a computation that checks for a cache file first:
    ; - If cache exists, RESTORE it
    ; - If not, compute (sum of 1000 random 256x256 matrices), SAVE result
    ; Time both paths and print the speedup
    ; TODO: FILE_TEST, RESTORE / compute + SAVE, benchmark

END
