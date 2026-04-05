;+
; 09_spectral_analysis.pro — Lesson 09: Spectral Analysis
;
; Demonstrates FFT, windowing, and power spectrum computation.
;-

PRO spectral_demo
    n = 1024
    dt = 0.01
    t = FINDGEN(n) * dt
    signal = 3.0*SIN(2*!PI*5.0*t) + 1.5*SIN(2*!PI*12.0*t) + RANDOMN(seed, n)*0.5

    ; FFT and power spectrum
    n_pos = n/2 + 1
    freq = FINDGEN(n_pos) / (n * dt)

    ; Without window
    fft1 = FFT(signal)
    psd1 = ABS(fft1[0:n_pos-1])^2
    psd1[1:n_pos-2] *= 2.0

    ; With Hanning window
    win = HANNING(n)
    fft2 = FFT(signal * win)
    psd2 = ABS(fft2[0:n_pos-1])^2
    psd2[1:n_pos-2] *= 2.0
    psd2 /= (TOTAL(win^2) / n)

    ; Plot
    WINDOW, 0, XSIZE=800, YSIZE=600
    !P.MULTI = [0, 1, 3]
    PLOT, t, signal, XTITLE='Time (s)', TITLE='Signal'
    PLOT, freq[1:*], psd1[1:*] * dt * n, /YLOG, $
        XTITLE='Freq (Hz)', TITLE='PSD (no window)', XRANGE=[0, 50]
    PLOT, freq[1:*], psd2[1:*] * dt * n, /YLOG, $
        XTITLE='Freq (Hz)', TITLE='PSD (Hanning)', XRANGE=[0, 50]
    !P.MULTI = 0

    PRINT, 'Peaks should appear at 5 Hz and 12 Hz'
END

spectral_demo
END
