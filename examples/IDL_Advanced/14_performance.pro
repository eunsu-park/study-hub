;+
; 14_performance.pro — Lesson 14: Performance and Large Data
;
; Demonstrates vectorization speedup, TEMPORARY, and ASSOC.
;-

PRO performance_demo
    ; --- Vectorization benchmark ---
    n = 1000000L
    x = RANDOMU(seed, n)
    y = RANDOMU(seed, n)

    ; Scalar loop
    t0 = SYSTIME(1)
    result1 = FLTARR(n)
    FOR i = 0L, n-1 DO result1[i] = SQRT(x[i]^2 + y[i]^2)
    t_loop = SYSTIME(1) - t0

    ; Vectorized
    t0 = SYSTIME(1)
    result2 = SQRT(x^2 + y^2)
    t_vec = SYSTIME(1) - t0

    PRINT, '--- Vectorization Benchmark (n=', n, ') ---'
    PRINT, 'Loop:       ', t_loop, ' s'
    PRINT, 'Vectorized: ', t_vec, ' s'
    PRINT, 'Speedup:    ', t_loop/t_vec, 'x'

    ; --- TEMPORARY ---
    PRINT, '--- TEMPORARY ---'
    a = FLTARR(1000, 1000)
    PRINT, 'Memory before TEMPORARY: ', MEMORY(/CURRENT)/1e6, ' MB'
    b = ALOG10(TEMPORARY(a) + 1.0)
    PRINT, 'Memory after TEMPORARY:  ', MEMORY(/CURRENT)/1e6, ' MB'
    PRINT, 'a is undefined: ', SIZE(a, /TYPE) EQ 0

    ; --- ASSOC demo ---
    PRINT, '--- ASSOC ---'
    tmpfile = 'assoc_demo.dat'
    nx = 256L & ny = 256L & nt = 20L

    ; Write binary file
    OPENW, lun, tmpfile, /GET_LUN
    FOR t = 0, nt-1 DO WRITEU, lun, DIST(nx) + RANDOMN(seed, nx, ny)*10
    FREE_LUN, lun

    ; Read with ASSOC (memory-mapped)
    OPENR, lun, tmpfile, /GET_LUN
    cube = ASSOC(lun, FLTARR(nx, ny))

    ; Compute mean without loading full file
    mean_frame = FLTARR(nx, ny)
    FOR t = 0, nt-1 DO mean_frame += cube[t]
    mean_frame /= nt

    FREE_LUN, lun
    FILE_DELETE, tmpfile, /QUIET
    PRINT, 'ASSOC mean computed. Shape: ', SIZE(mean_frame, /DIMENSIONS)
END

performance_demo
END
