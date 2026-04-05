;+
; Exercise 09: Spectral Analysis
;-

PRO exercise_09

    ; === Exercise 1: Power spectrum ===
    ; Generate a signal with 3 sinusoids (3, 7, 15 Hz) + noise
    ; Compute and plot the power spectrum. Verify all 3 peaks are detected.
    ; TODO: Create signal, FFT, compute PSD, plot, identify peaks
    ; Hint: n=2048, dt=0.005, signal = A1*sin(2*pi*f1*t) + ...

    ; === Exercise 2: Window comparison ===
    ; Compare the PSD of the same signal with: no window, Hanning, Hamming
    ; Plot all three on the same graph with different line styles
    ; TODO: Apply each window, compute PSD, overplot

    ; === Exercise 3: Bandpass filter ===
    ; Filter the signal to keep only the 5-10 Hz band
    ; Plot original and filtered signals, and their PSDs side by side
    ; TODO: Use DIGITAL_FILTER or frequency-domain masking

    ; === Exercise 4: Lomb-Scargle ===
    ; Generate unevenly sampled data with a known period of 50 time units
    ; Apply Lomb-Scargle periodogram and recover the period
    ; TODO: Create uneven time array, compute LS power, find peak

    ; === Exercise 5: Solar oscillation ===
    ; Create a synthetic sunspot light curve with 3-min and 5-min oscillations
    ; (cadence = 10s, duration = 2 hours)
    ; Detrend, compute PSD, mark the oscillation periods
    ; TODO: Create signal, detrend with SMOOTH, FFT, plot with period axis

END
