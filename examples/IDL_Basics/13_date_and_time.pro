; 13 Date and Time
; ================
; Demonstrates SYSTIME, JULDAY, CALDAT, time arithmetic,
; and time array creation.

PRO example_13_date_time

  ; Current time
  PRINT, '--- Current Time ---'
  PRINT, 'SYSTIME():', SYSTIME()
  PRINT, 'Seconds since epoch:', SYSTIME(1)
  PRINT, 'Julian date:', SYSTIME(/JULIAN)

  ; JULDAY and CALDAT
  PRINT, '--- JULDAY / CALDAT ---'
  jd = JULDAY(7, 15, 2024, 12, 0, 0)
  PRINT, 'JD of 2024-07-15 12:00:', jd

  CALDAT, jd, m, d, y, h, mn, s
  PRINT, FORMAT='("Back to calendar: ", I4, "-", I02, "-", I02, " ", I02, ":", I02)', $
    y, m, d, h, mn

  ; Time arithmetic
  PRINT, '--- Time Arithmetic ---'
  start = JULDAY(7, 15, 2024, 0, 0, 0)
  plus_1day = start + 1.0D0
  plus_6h = start + 6.0D0 / 24.0D0

  CALDAT, plus_1day, m, d, y, h, mn, s
  PRINT, FORMAT='("+1 day:  ", I4, "-", I02, "-", I02, " ", I02, ":", I02)', y, m, d, h, mn
  CALDAT, plus_6h, m, d, y, h, mn, s
  PRINT, FORMAT='("+6 hours:", I4, "-", I02, "-", I02, " ", I02, ":", I02)', y, m, d, h, mn

  ; Time difference
  jd1 = JULDAY(7, 15, 2024, 12, 0, 0)
  jd2 = JULDAY(7, 20, 2024, 18, 30, 0)
  PRINT, FORMAT='("Difference: ", F8.4, " days = ", F8.2, " hours")', $
    jd2 - jd1, (jd2 - jd1) * 24.0D0

  ; Day of year
  year = 2024 & month = 7 & day = 15
  doy = LONG(JULDAY(month, day, year) - JULDAY(1, 1, year)) + 1
  PRINT, FORMAT='(I4, "-", I02, "-", I02, " = DOY ", I03)', year, month, day, doy

  ; Time array
  PRINT, '--- Time Array ---'
  n = 10
  time_jd = start + DINDGEN(n) / 24.0D0
  FOR i = 0, n - 1 DO BEGIN
    CALDAT, time_jd[i], m, d, y, h, mn, s
    PRINT, FORMAT='("  ", I02, ":", I02, " UT")', h, mn
  ENDFOR

  PRINT, 'Example 13 complete.'
END
