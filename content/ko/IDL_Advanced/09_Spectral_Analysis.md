# 09. 스펙트럼 분석

**이전**: [GOES와 RHESSI](./08_GOES_and_RHESSI.md) | **다음**: [이미지 처리](./10_Image_Processing.md)

---

## 학습 목표

1. 적절한 정규화로 FFT와 파워 스펙트럼을 계산한다
2. 스펙트럼 누출을 줄이기 위한 윈도우 함수를 적용한다
3. 스펙트럼 필터(대역통과, 저역통과, 고역통과)를 설계하고 적용한다
4. 시간-주파수 분해를 위한 웨이블릿 분석을 수행한다
5. 불균일 샘플링 데이터에 Lomb-Scargle 피리오도그램을 사용한다

---

## 1. 고속 푸리에 변환 (FFT)

```idl
n = 1024
dt = 0.01  ; 초 (100 Hz 샘플링)
t = FINDGEN(n) * dt
signal = 3.0 * SIN(2*!PI*5.0*t) + 1.5 * SIN(2*!PI*12.0*t) + $
         RANDOMN(seed, n) * 0.5

; FFT 계산
fft_result = FFT(signal)

; 파워 스펙트럼 밀도
n_pos = n/2 + 1
freq_pos = FINDGEN(n_pos) / (n * dt)
psd = ABS(fft_result[0:n_pos-1])^2
psd[1:n_pos-2] = 2.0 * psd[1:n_pos-2]
psd_density = psd * dt * n
```

---

## 2. 윈도우 함수

```idl
; Hanning 윈도우
hanning_win = HANNING(n)

; Hamming 윈도우
hamming_win = HANNING(n, ALPHA=0.54)

; 윈도우 적용 후 FFT
windowed_signal = signal * HANNING(n)
fft_windowed = FFT(windowed_signal)
```

---

## 3. 스펙트럼 필터링

```idl
; DIGITAL_FILTER를 이용한 대역통과 필터
flow = 8.0 / (0.5 / dt)
fhigh = 15.0 / (0.5 / dt)
coeff = DIGITAL_FILTER(flow, fhigh, 50, 50)
filtered = CONVOL(signal, coeff, /EDGE_TRUNCATE)

; 주파수 영역 저역통과 필터
fft_signal = FFT(signal)
freq_full = FINDGEN(n) / (n * dt)
mask = FLTARR(n) + 1.0
mask[WHERE(freq_full GT 10.0 AND freq_full LT (1.0/dt - 10.0))] = 0.0
signal_lowpass = REAL_PART(FFT(fft_signal * mask, /INVERSE))
```

---

## 4. 웨이블릿 분석

웨이블릿은 시간-주파수 분해를 제공합니다: 스펙트럼 내용이 시간에 따라 어떻게 변하는지 보여줍니다.

```idl
; Morlet 웨이블릿 구현
n_scales = 64
scales = 2.0^(FINDGEN(n_scales)/8.0 + 1)
wavelet_power = FLTARR(n, n_scales)

FOR j = 0, n_scales-1 DO BEGIN
    ; 주파수 공간에서 웨이블릿 함수 계산
    ; FFT를 통한 합성곱
    ; ...웨이블릿 파워 계산...
ENDFOR
```

---

## 5. Lomb-Scargle 피리오도그램

불균일하게 샘플링된 데이터(천문 관측에서 흔함)에는 FFT 대신 Lomb-Scargle 피리오도그램이 적합합니다.

```idl
; 불균일 샘플링 데이터 생성
n_obs = 200
t_uneven = SORT(RANDOMU(seed, n_obs) * 100.0) * 100.0 / MAX(...)
signal_uneven = 2.0 * SIN(2*!PI*0.15*t_uneven) + RANDOMN(seed, n_obs) * 0.5

; Lomb-Scargle 계산
; ... 피리오도그램 구현 ...

; 유의 수준 (거짓 경보 확률)
; FAP = 1 - (1 - exp(-z))^M
```

---

## 6. 실전: 태양 진동 검출

```idl
; 태양 EUV 시계열에서 진동 검출
; (예: 흑점의 3분, 5분 진동)

cadence = 12.0  ; 초
duration = 4.0 * 3600.0
n_pts = LONG(duration / cadence)
t = FINDGEN(n_pts) * cadence

signal = 100.0 + $
    5.0 * SIN(2*!PI*t/180.0) + $  ; 3분 진동
    3.0 * SIN(2*!PI*t/300.0) + $  ; 5분 진동
    RANDOMN(seed, n_pts) * 2.0

; 추세 제거 후 파워 스펙트럼 계산
signal_detrend = signal - SMOOTH(signal, 101, /EDGE_TRUNCATE)
fft_result = FFT(signal_detrend * HANNING(n_pts))
```

---

## 요약

| 기법 | 핵심 함수 | 용도 |
|------|----------|------|
| FFT | `FFT`, `FFT(/INVERSE)` | 주파수 분해 |
| 윈도우 | `HANNING` | 스펙트럼 누출 감소 |
| 필터링 | `DIGITAL_FILTER`, `CONVOL` | 주파수 선택 |
| 웨이블릿 | 수동/SSW 루틴 | 시간-주파수 분석 |
| Lomb-Scargle | 수동 구현 | 불균일 샘플링 |

---

**이전**: [GOES와 RHESSI](./08_GOES_and_RHESSI.md) | **다음**: [이미지 처리](./10_Image_Processing.md)
