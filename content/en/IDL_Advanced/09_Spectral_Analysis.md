# 09. Spectral Analysis

**Previous**: [GOES and RHESSI](./08_GOES_and_RHESSI.md) | **Next**: [Image Processing](./10_Image_Processing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Compute FFT and power spectra with proper normalization
2. Apply windowing functions to reduce spectral leakage
3. Design and apply spectral filters (bandpass, lowpass, highpass)
4. Perform wavelet analysis for time-frequency decomposition
5. Use the Lomb-Scargle periodogram for unevenly sampled data

---

## 1. Fast Fourier Transform (FFT)

### Basic FFT

```idl
; Generate a test signal: two sinusoids + noise
n = 1024
dt = 0.01  ; seconds (100 Hz sampling)
t = FINDGEN(n) * dt
f1 = 5.0   ; Hz
f2 = 12.0  ; Hz
signal = 3.0 * SIN(2*!PI*f1*t) + 1.5 * SIN(2*!PI*f2*t) + $
         RANDOMN(seed, n) * 0.5

; Compute FFT
fft_result = FFT(signal)

; FFT result is complex, n elements
; fft_result[0]        — DC component (mean)
; fft_result[1:n/2-1]  — positive frequencies
; fft_result[n/2]      — Nyquist frequency
; fft_result[n/2+1:*]  — negative frequencies (mirror)

; Frequency array
freq = FINDGEN(n) / (n * dt)  ; Hz
; Only positive frequencies matter
n_pos = n/2 + 1
freq_pos = freq[0:n_pos-1]
```

### Power Spectrum

```idl
; Power spectral density (PSD)
; Two-sided: |FFT|^2
; One-sided: 2 * |FFT[0:n/2]|^2 (double for one-sided, except DC and Nyquist)

psd = ABS(fft_result)^2
psd_onesided = psd[0:n_pos-1]
psd_onesided[1:n_pos-2] = 2.0 * psd_onesided[1:n_pos-2]

; Normalize to physical units (power/Hz)
psd_density = psd_onesided * dt * n  ; [signal_units^2 / Hz]

; Plot
WINDOW, 0, XSIZE=800, YSIZE=500
!P.MULTI = [0, 1, 2]
PLOT, t, signal, XTITLE='Time (s)', YTITLE='Signal', $
    TITLE='Time Series'
PLOT, freq_pos[1:*], psd_density[1:*], /YLOG, $
    XTITLE='Frequency (Hz)', YTITLE='PSD (units!U2!N Hz!U-1!N)', $
    TITLE='Power Spectral Density', XRANGE=[0, 50]
!P.MULTI = 0
```

### 2D FFT for Images

```idl
; 2D FFT for spatial frequency analysis
img = DIST(256)  ; Test image
fft_2d = FFT(img)

; Power spectrum (log scale for visualization)
power_2d = SHIFT(ABS(fft_2d)^2, 128, 128)  ; Center zero-frequency
WINDOW, 0, XSIZE=256, YSIZE=256
TV, BYTSCL(ALOG10(power_2d > 1e-10))
```

---

## 2. Windowing Functions

Windowing reduces spectral leakage by tapering the signal edges to zero.

### Common Windows

```idl
n = 1024

; Hanning window
hanning_win = HANNING(n)

; Hamming window
hamming_win = HANNING(n, ALPHA=0.54)  ; Hamming is HANNING with alpha=0.54

; Blackman window (not built-in, compute manually)
x = FINDGEN(n) / (n-1)
blackman = 0.42 - 0.5*COS(2*!PI*x) + 0.08*COS(4*!PI*x)

; Tukey window (cosine-tapered)
alpha_tukey = 0.5  ; Fraction that is cosine-tapered
tukey = FLTARR(n) + 1.0
n_taper = ROUND(alpha_tukey * n / 2.0)
taper = 0.5 * (1.0 - COS(!PI * FINDGEN(n_taper) / n_taper))
tukey[0:n_taper-1] = taper
tukey[n-n_taper:*] = REVERSE(taper)

; Plot windows
WINDOW, 0, XSIZE=800, YSIZE=400
PLOT, hanning_win, TITLE='Window Functions', $
    YTITLE='Amplitude', XTITLE='Sample'
OPLOT, hamming_win, LINESTYLE=2
OPLOT, blackman, LINESTYLE=3
OPLOT, tukey, LINESTYLE=4
```

### Applying a Window

```idl
; Apply Hanning window before FFT
windowed_signal = signal * HANNING(n)

; Compute FFT
fft_windowed = FFT(windowed_signal)
psd_windowed = ABS(fft_windowed[0:n_pos-1])^2
psd_windowed[1:n_pos-2] = 2.0 * psd_windowed[1:n_pos-2]

; Window correction factor (for power normalization)
; The window reduces total power; compensate with window energy
win_energy = TOTAL(HANNING(n)^2) / n
psd_windowed = psd_windowed / win_energy

; Compare windowed vs unwindowed
!P.MULTI = [0, 1, 2]
PLOT, freq_pos[1:*], psd_density[1:*], /YLOG, $
    TITLE='Without Window', XTITLE='Freq (Hz)', XRANGE=[0, 50]
PLOT, freq_pos[1:*], psd_windowed[1:*] * dt * n, /YLOG, $
    TITLE='With Hanning Window', XTITLE='Freq (Hz)', XRANGE=[0, 50]
!P.MULTI = 0
```

---

## 3. Spectral Filtering

### DIGITAL_FILTER

```idl
; IDL's DIGITAL_FILTER creates a FIR filter
; Bandpass filter: pass 8-15 Hz from a 100 Hz signal

flow = 8.0 / (0.5 / dt)   ; Normalized: fraction of Nyquist
fhigh = 15.0 / (0.5 / dt)

; Create filter coefficients
n_terms = 50  ; Filter order (number of terms)
coeff = DIGITAL_FILTER(flow, fhigh, 50, n_terms)
; Arguments: f_low, f_high, Gibbs_constant, n_terms

; Apply filter via convolution
filtered = CONVOL(signal, coeff, /EDGE_TRUNCATE)

; Plot
!P.MULTI = [0, 1, 3]
PLOT, t, signal, TITLE='Original', XTITLE='Time (s)'
PLOT, t, filtered, TITLE='Bandpass 8-15 Hz', XTITLE='Time (s)'
; Show frequency content
fft_filt = FFT(filtered)
PLOT, freq_pos[1:*], 2*ABS(fft_filt[1:n_pos-1])^2, $
    TITLE='Filtered Spectrum', XTITLE='Freq (Hz)', XRANGE=[0, 50]
!P.MULTI = 0
```

### Manual Frequency-Domain Filtering

```idl
; Lowpass filter in frequency domain
fft_signal = FFT(signal)

; Create frequency mask
freq_full = FINDGEN(n) / (n * dt)  ; Hz
; Make symmetric mask for both positive and negative frequencies
cutoff = 10.0  ; Hz
mask = FLTARR(n) + 1.0
high_freq = WHERE(freq_full GT cutoff AND freq_full LT (1.0/dt - cutoff))
mask[high_freq] = 0.0

; Apply and inverse FFT
fft_filtered = fft_signal * mask
signal_lowpass = REAL_PART(FFT(fft_filtered, /INVERSE))

; Bandpass filter
f_lo = 4.0 & f_hi = 8.0
mask_bp = FLTARR(n)
pass = WHERE((freq_full GE f_lo AND freq_full LE f_hi) OR $
             (freq_full GE (1.0/dt - f_hi) AND freq_full LE (1.0/dt - f_lo)))
mask_bp[pass] = 1.0
signal_bandpass = REAL_PART(FFT(FFT(signal) * mask_bp, /INVERSE))
```

---

## 4. Wavelet Analysis

Wavelets provide time-frequency decomposition: they show how spectral content changes over time.

### Wavelet Transform with SSW

```idl
; SSW includes wavelet analysis routines
; Based on Torrence & Compo (1998)

; Generate a signal with time-varying frequency
n = 2048
dt = 1.0  ; seconds
t = FINDGEN(n) * dt
signal = SIN(2*!PI*0.05*t) * (t LT 1000) + $
         SIN(2*!PI*0.1*t) * (t GE 1000) + $
         RANDOMN(seed, n) * 0.3

; Compute wavelet transform
; SSW wavelet routine (if available)
; WAVE_RESULT = WV_CWT(signal, 'Morlet', 6, SCALE=scale, /PAD)

; Manual Morlet wavelet implementation
n_scales = 64
scales = 2.0^(FINDGEN(n_scales)/8.0 + 1)  ; Logarithmic scale range
periods = scales * dt  ; periods in seconds

wavelet_power = FLTARR(n, n_scales)

FOR j = 0, n_scales-1 DO BEGIN
    ; Morlet wavelet in frequency domain
    omega0 = 6.0  ; Morlet parameter
    s = scales[j]
    freq_arr = FINDGEN(n) / (n * dt)

    ; Wavelet function in frequency space
    psi_hat = FLTARR(n)
    FOR k = 1, n/2 DO BEGIN
        omega = 2.0 * !PI * freq_arr[k]
        psi_hat[k] = SQRT(2.0*!PI*s/dt) * (!PI^(-0.25)) * $
            EXP(-0.5*(s*omega - omega0)^2)
    ENDFOR

    ; Convolution via FFT
    fft_signal = FFT(signal)
    wave_coeff = FFT(fft_signal * psi_hat, /INVERSE)
    wavelet_power[*, j] = ABS(wave_coeff)^2
ENDFOR

; Display wavelet power spectrum
WINDOW, 0, XSIZE=800, YSIZE=500
!P.MULTI = [0, 1, 2]
PLOT, t, signal, TITLE='Signal', XTITLE='Time (s)'
LOADCT, 39
TV, BYTSCL(ALOG10(CONGRID(wavelet_power, 800, 200) > 1e-5)), $
    0, 50, XSIZE=800, YSIZE=200
XYOUTS, 400, 270, 'Wavelet Power', ALIGNMENT=0.5, /DEVICE
!P.MULTI = 0
```

---

## 5. Cross-Spectral Analysis

```idl
; Cross-spectrum between two signals
n = 2048
dt = 0.01
t = FINDGEN(n) * dt

; Two related signals with phase lag
f0 = 10.0  ; Hz
phase_lag = !PI / 4.0  ; 45 degrees
signal1 = SIN(2*!PI*f0*t) + RANDOMN(seed, n) * 0.3
signal2 = SIN(2*!PI*f0*t + phase_lag) + RANDOMN(seed, n) * 0.3

; FFT both
fft1 = FFT(signal1 * HANNING(n))
fft2 = FFT(signal2 * HANNING(n))

; Cross-spectrum
cross_spec = fft1 * CONJ(fft2)

; Cross-power
cross_power = ABS(cross_spec[0:n/2])

; Coherence (requires averaging — use Welch's method)
; Phase difference
phase = ATAN(IMAGINARY(cross_spec), REAL_PART(cross_spec))
phase_deg = phase[0:n/2] * !RADEG

; Plot
freq_arr = FINDGEN(n/2+1) / (n * dt)
!P.MULTI = [0, 1, 2]
PLOT, freq_arr, cross_power, XTITLE='Frequency (Hz)', $
    YTITLE='Cross Power', TITLE='Cross-Power Spectrum', XRANGE=[0, 50]
PLOT, freq_arr, phase_deg, XTITLE='Frequency (Hz)', $
    YTITLE='Phase (degrees)', TITLE='Phase Difference', $
    XRANGE=[0, 50], YRANGE=[-180, 180], PSYM=3
; Should show ~45 degrees at 10 Hz
!P.MULTI = 0
```

---

## 6. Lomb-Scargle Periodogram

For unevenly sampled data (common in astronomical observations), the Lomb-Scargle periodogram is preferred over FFT.

```idl
; Generate unevenly sampled data
n_obs = 200
t_uneven = SORT(RANDOMU(seed, n_obs) * 100.0)  ; Random times in [0, 100]
t_uneven = t_uneven * 100.0 / MAX(t_uneven)     ; Normalize
f_true = 0.15  ; True frequency
signal_uneven = 2.0 * SIN(2*!PI*f_true*t_uneven) + RANDOMN(seed, n_obs) * 0.5

; Lomb-Scargle periodogram
; Frequency grid
n_freq = 500
freq_ls = (FINDGEN(n_freq) + 1) * 0.001  ; 0.001 to 0.5

; Compute periodogram
power_ls = FLTARR(n_freq)

; Subtract mean
y = signal_uneven - MEAN(signal_uneven)
var_y = VARIANCE(y)

FOR i = 0, n_freq-1 DO BEGIN
    omega = 2.0 * !PI * freq_ls[i]

    ; Compute tau (time offset)
    tau = ATAN(TOTAL(SIN(2*omega*t_uneven)), $
               TOTAL(COS(2*omega*t_uneven))) / (2*omega)

    ; Lomb-Scargle power
    cos_term = COS(omega * (t_uneven - tau))
    sin_term = SIN(omega * (t_uneven - tau))

    num_cos = (TOTAL(y * cos_term))^2 / TOTAL(cos_term^2)
    num_sin = (TOTAL(y * sin_term))^2 / TOTAL(sin_term^2)

    power_ls[i] = 0.5 * (num_cos + num_sin) / var_y
ENDFOR

; Plot
WINDOW, 0, XSIZE=800, YSIZE=500
!P.MULTI = [0, 1, 2]
PLOT, t_uneven, signal_uneven, PSYM=1, SYMSIZE=0.5, $
    XTITLE='Time', YTITLE='Signal', TITLE='Unevenly Sampled Data'
PLOT, freq_ls, power_ls, $
    XTITLE='Frequency', YTITLE='Lomb-Scargle Power', $
    TITLE='Lomb-Scargle Periodogram'
; Mark true frequency
PLOTS, [f_true, f_true], [0, MAX(power_ls)], LINESTYLE=2, COLOR=250
!P.MULTI = 0

; Find peak frequency
peak_idx = WHERE(power_ls EQ MAX(power_ls))
PRINT, 'True frequency:     ', f_true
PRINT, 'Detected frequency: ', freq_ls[peak_idx[0]]
```

### False Alarm Probability

```idl
; Significance level for Lomb-Scargle
; FAP (False Alarm Probability) for a given power level z:
; FAP = 1 - (1 - exp(-z))^M
; where M is the number of independent frequencies

M = n_freq  ; Approximate number of independent frequencies
z_99 = -ALOG(1.0 - (1.0 - 0.01)^(1.0/M))  ; 99% significance
z_95 = -ALOG(1.0 - (1.0 - 0.05)^(1.0/M))  ; 95% significance

PRINT, '99% significance level: ', z_99
PRINT, '95% significance level: ', z_95

; Add to plot
PLOTS, [MIN(freq_ls), MAX(freq_ls)], [z_99, z_99], $
    LINESTYLE=2, COLOR=200
XYOUTS, MAX(freq_ls)*0.8, z_99*1.1, '99%', CHARSIZE=0.8
```

---

## 7. Practical: Solar Oscillation Detection

```idl
; Detect oscillations in a solar EUV time series
; (e.g., 3-minute and 5-minute oscillations in sunspots)

; Simulated solar time series
; 12-second cadence, 4 hours duration
cadence = 12.0  ; seconds
duration = 4.0 * 3600.0  ; seconds
n_pts = LONG(duration / cadence)
t = FINDGEN(n_pts) * cadence

; Simulated signal: 3-min and 5-min oscillations
period_3min = 180.0  ; seconds
period_5min = 300.0  ; seconds
signal = 100.0 + $
    5.0 * SIN(2*!PI*t/period_3min) + $  ; 3-min oscillation
    3.0 * SIN(2*!PI*t/period_5min) + $  ; 5-min oscillation
    RANDOMN(seed, n_pts) * 2.0           ; Noise

; Detrend
signal_detrend = signal - SMOOTH(signal, 101, /EDGE_TRUNCATE)

; Compute power spectrum
win = HANNING(n_pts)
fft_result = FFT(signal_detrend * win)
freq = FINDGEN(n_pts) / (n_pts * cadence)  ; Hz
period = 1.0 / freq[1:n_pts/2]             ; Seconds
psd = 2.0 * ABS(fft_result[1:n_pts/2])^2 / TOTAL(win^2) * n_pts * cadence

; Plot
!P.MULTI = [0, 1, 2]
PLOT, t/60.0, signal_detrend, $
    XTITLE='Time (min)', YTITLE='Detrended Intensity', $
    TITLE='Solar EUV Time Series'
PLOT, period/60.0, psd, /YLOG, $
    XTITLE='Period (min)', YTITLE='PSD', $
    TITLE='Power Spectrum', XRANGE=[0, 10]

; Mark 3-min and 5-min periods
PLOTS, [3.0, 3.0], 10^!Y.CRANGE, LINESTYLE=2, COLOR=250
PLOTS, [5.0, 5.0], 10^!Y.CRANGE, LINESTYLE=2, COLOR=200
XYOUTS, 3.1, MAX(psd)*0.5, '3 min', CHARSIZE=0.8
XYOUTS, 5.1, MAX(psd)*0.5, '5 min', CHARSIZE=0.8
!P.MULTI = 0
```

---

## Summary

| Technique | Key Functions | Purpose |
|-----------|-------------|---------|
| FFT | `FFT`, `FFT(/INVERSE)` | Frequency decomposition |
| Windowing | `HANNING` | Reduce spectral leakage |
| Filtering | `DIGITAL_FILTER`, `CONVOL` | Frequency selection |
| Cross-spectrum | `FFT`, `CONJ` | Phase/coherence analysis |
| Wavelets | Manual/SSW routines | Time-frequency analysis |
| Lomb-Scargle | Manual implementation | Uneven sampling |
| Power spectrum | `ABS(FFT)^2` | Signal power distribution |

---

**Previous**: [GOES and RHESSI](./08_GOES_and_RHESSI.md) | **Next**: [Image Processing](./10_Image_Processing.md)
