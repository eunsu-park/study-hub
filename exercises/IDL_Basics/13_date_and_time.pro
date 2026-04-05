; Exercise 13: Date and Time
;
; Practice Julian dates, time arithmetic, and date formatting.

; Exercise 1: Write a function that converts an ISO date string
; 'YYYY-MM-DDTHH:MM:SS' to a Julian date.
FUNCTION exercise_13a, date_str
  ; TODO: Parse year, month, day, hour, minute, second from string
  ; TODO: Use JULDAY to convert
  ; TODO: Return the Julian date as DOUBLE
  ; Hint: Use STRMID or STRSPLIT to extract components
  RETURN, 0.0D0
END

; Exercise 2: Write a function that computes the number of days,
; hours, and minutes between two date strings.
FUNCTION exercise_13b, date1, date2
  ; TODO: Parse both dates to Julian dates
  ; TODO: Compute difference in days
  ; TODO: Convert to hours and minutes
  ; TODO: Return structure {days: ..., hours: ..., minutes: ...}
  RETURN, {days: 0.0D0, hours: 0.0D0, minutes: 0.0D0}
END

; Exercise 3: Create a time array covering 7 days at 1-hour cadence,
; starting from 2024-01-01 00:00 UT. Print the first and last 3 times
; in 'YYYY-MM-DD HH:MM' format.
PRO exercise_13c
  ; TODO: start_jd = JULDAY(1, 1, 2024, 0, 0, 0)
  ; TODO: Create time array with 7*24 = 168 points
  ; TODO: Print first 3 and last 3 using CALDAT and FORMAT
END

; Exercise 4: Write a function that takes an array of Julian dates
; and returns an array of Day-of-Year (DOY) integers.
FUNCTION exercise_13d, jd_array
  ; TODO: For each JD, use CALDAT to get year
  ; TODO: Compute DOY = JD - JULDAY(1, 1, year) + 1
  ; TODO: Return integer array of DOY values
  ; Hint: Vectorize by extracting year first, then computing
  RETURN, LONARR(N_ELEMENTS(jd_array))
END
