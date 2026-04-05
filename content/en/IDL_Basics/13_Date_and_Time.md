# Date and Time

**Previous**: [FITS File Handling](./12_FITS_File_Handling.md) | **Next**: [Debugging and Best Practices](./14_Debugging_and_Best_Practices.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Get the current time with SYSTIME
2. Work with Julian dates using JULDAY and CALDAT
3. Parse date strings into numeric components
4. Use ANYTIM for flexible time conversion (SolarSoft)
5. Perform time arithmetic (adding hours, days, finding intervals)
6. Format dates for plot axis labels
7. Create time arrays for time series analysis

---

Date and time handling is essential in scientific data analysis. Observation timestamps, exposure times, and time series all require precise time representation and conversion. IDL provides several approaches to working with dates and times.

## SYSTIME — Current System Time

```idl
; Current time as a string
PRINT, SYSTIME()
; Thu Jul 15 14:30:00 2024

; Current time as seconds since January 1, 1970 (Unix epoch)
PRINT, SYSTIME(1)
; 1.7210694e+09

; UTC (not local time)
PRINT, SYSTIME(/UTC)
; Thu Jul 15 18:30:00 2024

; Julian date of current time
PRINT, SYSTIME(/JULIAN)
; 2460506.3

; Seconds since epoch (double precision)
t0 = SYSTIME(1, /SECONDS)
; ... do some computation ...
t1 = SYSTIME(1, /SECONDS)
PRINT, 'Elapsed:', t1 - t0, 'seconds'

; Convert seconds since epoch to a string
epoch_seconds = SYSTIME(1)
time_string = SYSTIME(0, epoch_seconds)
PRINT, time_string
```

---

## Julian Dates

A Julian Date (JD) is a continuous count of days since January 1, 4713 BC. It provides a uniform time scale for astronomical calculations.

### JULDAY — Convert Calendar to Julian Date

```idl
; JULDAY(month, day, year [, hour, minute, second])
jd = JULDAY(7, 15, 2024, 12, 0, 0)
PRINT, jd
; 2460506.0

; Note the parameter order: Month, Day, Year (not Year, Month, Day)
jd_new_year = JULDAY(1, 1, 2024, 0, 0, 0)
PRINT, 'JD of 2024-01-01:', jd_new_year

; J2000.0 epoch
j2000 = JULDAY(1, 1, 2000, 12, 0, 0)
PRINT, 'J2000.0:', j2000    ; 2451545.0

; Days since J2000.0
days_since_j2000 = jd - j2000
PRINT, 'Days since J2000:', days_since_j2000
```

### CALDAT — Convert Julian Date to Calendar

```idl
; CALDAT, julian_date, month, day, year, hour, minute, second
jd = 2460506.0D0
CALDAT, jd, month, day, year, hour, minute, second

PRINT, FORMAT='(I4, "-", I02, "-", I02, " ", I02, ":", I02, ":", I02)', $
  year, month, day, hour, minute, second
; 2024-07-15 12:00:00
```

### Modified Julian Date (MJD)

```idl
; MJD = JD - 2400000.5
jd = JULDAY(7, 15, 2024, 12, 0, 0)
mjd = jd - 2400000.5D0
PRINT, 'MJD:', mjd

; Convert back
jd_back = mjd + 2400000.5D0
CALDAT, jd_back, m, d, y
PRINT, y, m, d
```

---

## Parsing Date Strings

### Manual Parsing

```idl
; Parse ISO 8601 format: 'YYYY-MM-DDTHH:MM:SS.sss'
date_str = '2024-07-15T14:30:45.123'

year = FIX(STRMID(date_str, 0, 4))
month = FIX(STRMID(date_str, 5, 2))
day = FIX(STRMID(date_str, 8, 2))
hour = FIX(STRMID(date_str, 11, 2))
minute = FIX(STRMID(date_str, 14, 2))
second = DOUBLE(STRMID(date_str, 17))

PRINT, year, month, day, hour, minute, second

; Convert to Julian date
jd = JULDAY(month, day, year, hour, minute, second)
PRINT, 'JD:', jd
```

### Parsing Function

```idl
FUNCTION parse_iso_date, date_str
  ; Parse ISO 8601 date string to Julian date
  ; Handles: 'YYYY-MM-DDTHH:MM:SS.sss' or 'YYYY-MM-DD HH:MM:SS'

  ; Replace T with space for uniform parsing
  s = date_str
  t_pos = STRPOS(s, 'T')
  IF t_pos GT 0 THEN STRPUT, s, ' ', t_pos

  parts = STRSPLIT(s, '- :', /EXTRACT)
  n = N_ELEMENTS(parts)

  year = FIX(parts[0])
  month = FIX(parts[1])
  day = FIX(parts[2])
  hour = (n GE 4) ? FIX(parts[3]) : 0
  minute = (n GE 5) ? FIX(parts[4]) : 0
  second = (n GE 6) ? DOUBLE(parts[5]) : 0.0D0

  RETURN, JULDAY(month, day, year, hour, minute, second)
END
```

---

## ANYTIM — SolarSoft Time Conversion

In SolarSoft (SSW), `ANYTIM` is the universal time conversion function:

```idl
; ANYTIM converts between many time formats
; (Requires SolarSoft installation)

; String to various formats
t = ANYTIM('2024-07-15 14:30:00')        ; Returns seconds from reference
t = ANYTIM('2024-07-15T14:30:00', /TAI)  ; TAI seconds
t = ANYTIM('15-Jul-2024 14:30:00')       ; Various string formats accepted

; Convert between formats
jd = ANYTIM('2024-07-15 14:30:00', /JULIAN)    ; To Julian date
utc = ANYTIM(jd, /UTC_EXT, /JULIAN)             ; To UTC external structure
tai = ANYTIM('2024-07-15 14:30:00', /TAI)       ; To TAI seconds

; Format output
str = ANYTIM('2024-07-15 14:30:00', /CCSDS)     ; ISO 8601 format
str = ANYTIM('2024-07-15 14:30:00', /VMSTIME)   ; VMS format
```

### UTC2TAI and TAI2UTC (SolarSoft)

```idl
; TAI = International Atomic Time (continuous seconds)
; UTC includes leap seconds

; Convert UTC string to TAI seconds
tai = UTC2TAI('2024-07-15 14:30:00')

; Convert TAI back to UTC
utc = TAI2UTC(tai)
PRINT, utc

; TAI seconds are useful for time differences
t1 = UTC2TAI('2024-07-15 14:30:00')
t2 = UTC2TAI('2024-07-15 15:45:30')
dt = t2 - t1    ; Difference in seconds
PRINT, 'Time difference:', dt, ' seconds'
PRINT, 'Time difference:', dt / 60.0, ' minutes'
```

---

## Time Arithmetic

### Adding Time Intervals

```idl
; Using Julian dates (where 1.0 = 1 day)
start_jd = JULDAY(7, 15, 2024, 12, 0, 0)

; Add 1 day
next_day = start_jd + 1.0D0

; Add 6 hours
plus_6h = start_jd + 6.0D0 / 24.0D0

; Add 30 minutes
plus_30m = start_jd + 30.0D0 / (24.0D0 * 60.0D0)

; Add 45 seconds
plus_45s = start_jd + 45.0D0 / 86400.0D0

; Print results
CALDAT, next_day, m, d, y, h, mn, s
PRINT, FORMAT='("+ 1 day:   ", I4, "-", I02, "-", I02, " ", I02, ":", I02)', y, m, d, h, mn
CALDAT, plus_6h, m, d, y, h, mn, s
PRINT, FORMAT='("+ 6 hours: ", I4, "-", I02, "-", I02, " ", I02, ":", I02)', y, m, d, h, mn
```

### Time Differences

```idl
; Calculate time difference between two dates
jd1 = JULDAY(7, 15, 2024, 12, 0, 0)
jd2 = JULDAY(7, 20, 2024, 18, 30, 0)

diff_days = jd2 - jd1
diff_hours = diff_days * 24.0D0
diff_minutes = diff_days * 1440.0D0
diff_seconds = diff_days * 86400.0D0

PRINT, FORMAT='("Difference: ", F10.4, " days")', diff_days
PRINT, FORMAT='("            ", F10.2, " hours")', diff_hours
PRINT, FORMAT='("            ", F10.1, " minutes")', diff_minutes
```

---

## Creating Time Arrays

```idl
; Create a time array at regular intervals
start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
cadence_sec = 12.0D0    ; 12-second cadence (like AIA)
n_steps = 7200           ; 24 hours of data at 12s cadence

; Time array in Julian dates
time_jd = start_jd + DINDGEN(n_steps) * cadence_sec / 86400.0D0

; Convert to hours from start
time_hours = (time_jd - start_jd) * 24.0D0

; Time array with specific step
; Every hour for 7 days
hourly = start_jd + DINDGEN(7*24) / 24.0D0

; Every minute for 1 day
minutely = start_jd + DINDGEN(1440) / 1440.0D0
```

---

## Formatting Dates for Plots

### Custom Tick Labels

```idl
; Plot with time on the X axis
n = 100
start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
time_jd = start_jd + DINDGEN(n) / 24.0D0    ; Hourly data for ~4 days

flux = 1000.0 + 500.0 * SIN(2.0D0 * !DPI * (time_jd - start_jd) / 1.0D0) + $
       RANDOMN(seed, n) * 100.0

; Method 1: Use hours from start
time_hours = (time_jd - start_jd) * 24.0D0
PLOT, time_hours, flux, $
  XTITLE='Hours since 2024-07-15 00:00 UT', $
  YTITLE='Flux (DN/s)', $
  TITLE='Time Series'
```

### Custom Tick Format Function

```idl
; Create a function that formats tick labels as dates
FUNCTION time_tick_format, axis, index, value
  ; value is Julian date
  CALDAT, value, month, day, year, hour, minute
  months = ['Jan','Feb','Mar','Apr','May','Jun',$
            'Jul','Aug','Sep','Oct','Nov','Dec']
  RETURN, STRING(FORMAT='(I02, ":", I02, " ", A3, " ", I02)', $
    hour, minute, months[month-1], day)
END
```

```idl
; Use the custom format function
start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
n = 48
time_jd = start_jd + DINDGEN(n) / 24.0D0
flux = 1000.0 + RANDOMN(seed, n) * 200.0

PLOT, time_jd, flux, $
  XTICKFORMAT='time_tick_format', $
  XTICKS=4, $
  XTITLE='Date/Time (UT)', $
  YTITLE='Flux', $
  TITLE='Solar Light Curve', $
  CHARSIZE=1.2
```

### LABEL_DATE (IDL Built-in)

```idl
; LABEL_DATE provides built-in date formatting for plot axes
; Requires LABEL_DATE format string and Julian dates

; Set up the format
dummy = LABEL_DATE(DATE_FORMAT=['%H:%I', '%M/%D'])

; Plot with date labels
PLOT, time_jd, flux, $
  XTICKFORMAT='LABEL_DATE', $
  XTICKUNITS=['Hours', 'Days'], $
  XTITLE='Date/Time', YTITLE='Flux'
```

---

## Day of Year (DOY)

```idl
; Convert date to day of year
year = 2024
month = 7
day = 15

jd_date = JULDAY(month, day, year)
jd_jan1 = JULDAY(1, 1, year)
doy = LONG(jd_date - jd_jan1) + 1
PRINT, FORMAT='(I4, "-", I02, "-", I02, " = DOY ", I03)', year, month, day, doy
; 2024-07-15 = DOY 197

; Convert DOY back to month/day
jd_from_doy = JULDAY(1, 1, year) + doy - 1
CALDAT, jd_from_doy, m, d, y
PRINT, FORMAT='("DOY ", I03, " = ", I4, "-", I02, "-", I02)', doy, y, m, d

; DOY for an entire year (check leap year)
n_days = JULDAY(12, 31, year) - JULDAY(1, 1, year) + 1
PRINT, 'Days in', year, ':', n_days    ; 366 for leap year, 365 otherwise
```

---

## Practical Examples

### Observation Time Range

```idl
PRO print_obs_timerange, date_start, date_end
  ; Parse date strings
  jd_start = parse_iso_date(date_start)
  jd_end = parse_iso_date(date_end)

  ; Duration
  duration_days = jd_end - jd_start
  duration_hours = duration_days * 24.0D0
  duration_min = duration_days * 1440.0D0

  PRINT, '=== Observation Time Range ==='
  PRINT, 'Start: ' + date_start
  PRINT, 'End:   ' + date_end
  PRINT, FORMAT='("Duration: ", F8.4, " days")', duration_days
  PRINT, FORMAT='("          ", F8.2, " hours")', duration_hours
  PRINT, FORMAT='("          ", F10.1, " minutes")', duration_min

  ; Cadence estimate (if we had N frames)
  n_frames = 100L
  cadence = duration_days * 86400.0D0 / (n_frames - 1)
  PRINT, FORMAT='("Cadence for ", I0, " frames: ", F8.2, " seconds")', n_frames, cadence
END
```

### Time-Tagged Data Processing

```idl
FUNCTION filter_by_timerange, times_jd, data, start_jd, end_jd
  ; Filter data points within a time range
  idx = WHERE(times_jd GE start_jd AND times_jd LE end_jd, count)

  IF count EQ 0 THEN BEGIN
    PRINT, 'No data points in specified range'
    RETURN, !NULL
  ENDIF

  PRINT, FORMAT='("Selected ", I0, " of ", I0, " data points")', count, N_ELEMENTS(data)
  RETURN, {time: times_jd[idx], data: data[idx], count: count}
END
```

---

## Summary

| Function/Procedure | Description |
|-------------------|-------------|
| `SYSTIME()` | Current system time (string or seconds) |
| `SYSTIME(/JULIAN)` | Current Julian date |
| `JULDAY(M, D, Y, H, MN, S)` | Calendar to Julian date |
| `CALDAT, JD, M, D, Y, H, MN, S` | Julian date to calendar |
| `LABEL_DATE` | Format Julian dates for plot axes |
| `ANYTIM` (SSW) | Universal time conversion |
| `UTC2TAI` / `TAI2UTC` (SSW) | UTC/TAI conversion |

| Time Arithmetic | Value |
|----------------|-------|
| 1 day | 1.0 JD |
| 1 hour | 1.0/24.0 JD |
| 1 minute | 1.0/1440.0 JD |
| 1 second | 1.0/86400.0 JD |

---

**Previous**: [FITS File Handling](./12_FITS_File_Handling.md) | **Next**: [Debugging and Best Practices](./14_Debugging_and_Best_Practices.md)
