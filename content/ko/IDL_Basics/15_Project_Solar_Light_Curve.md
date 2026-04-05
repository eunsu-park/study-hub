# 프로젝트: 태양 광도 곡선

**이전**: [디버깅과 모범 사례](./14_Debugging_and_Best_Practices.md)

## 학습 목표

이 프로젝트를 완료하면 다음을 할 수 있습니다:

1. FITS 시계열 데이터 읽기 (GOES X-ray flux)
2. FITS 파일에서 시간과 플럭스 배열 추출하기
3. 과학 데이터의 날짜/시간 변환 처리하기
4. 출판 품질의 광도 곡선 플롯 생성하기
5. 적절한 축 라벨, 범례, 주석 추가하기
6. 저널 제출을 위한 PostScript 형식으로 그림 출력하기
7. 이 과정에서 배운 모든 IDL 기술을 실제 문제에 적용하기

---

이 종합 프로젝트에서는 이전 14개 레슨의 모든 내용을 종합하여 완전한 태양 물리학 데이터 분석 워크플로우를 구축합니다. GOES (Geostationary Operational Environmental Satellite) X-ray 플럭스 데이터를 읽고 처리하여 출판 품질의 광도 곡선 플롯을 만듭니다.

## 프로젝트 개요

GOES X-ray Sensor (XRS) 데이터는 태양 플레어 활동을 모니터링하는 표준 데이터셋입니다:

- **단채널 (0.5-4 A)**: 뜨거운 플레어 플라즈마에 민감
- **장채널 (1-8 A)**: 표준 플레어 분류 채널

태양 플레어는 1-8 A 채널의 최대 플럭스로 분류됩니다:

| 등급 | 플럭스 범위 (W/m^2) |
|------|-------------------|
| A | < 10^-7 |
| B | 10^-7 ~ 10^-6 |
| C | 10^-6 ~ 10^-5 |
| M | 10^-5 ~ 10^-4 |
| X | >= 10^-4 |

## 단계 1: 시뮬레이션된 GOES 데이터 생성

```idl
FUNCTION generate_goes_data, DURATION_HOURS=duration, CADENCE_SEC=cadence
  IF N_ELEMENTS(duration) EQ 0 THEN duration = 24.0D0
  IF N_ELEMENTS(cadence) EQ 0 THEN cadence = 60.0D0

  n_points = LONG(duration * 3600.0D0 / cadence)
  start_jd = JULDAY(7, 15, 2024, 0, 0, 0)
  time_jd = start_jd + DINDGEN(n_points) * cadence / 86400.0D0

  ; 배경 플럭스 (조용한 태양)
  seed = 42L
  background = 5.0D-8 + RANDOMN(seed, n_points) * 5.0D-9
  background = background > 1.0D-9

  ; 플레어 이벤트 추가
  flux_long = background
  flux_long += flare_profile(time_jd, start_jd + 4.0D/24, 3.0D-6, 10.0D, 30.0D)
  flux_long += flare_profile(time_jd, start_jd + 10.5D/24, 2.0D-5, 8.0D, 45.0D)
  flux_long += flare_profile(time_jd, start_jd + 16.0D/24, 5.0D-7, 5.0D, 20.0D)
  flux_long += flare_profile(time_jd, start_jd + 19.0D/24, 1.5D-4, 6.0D, 60.0D)

  flux_short = background * 0.1D + flux_long * 0.1D

  RETURN, {time_jd: time_jd, flux_long: flux_long, flux_short: flux_short, $
           n_points: n_points, start_jd: start_jd, cadence: cadence, duration: duration}
END

FUNCTION flare_profile, time_jd, peak_jd, peak_flux, rise_min, decay_min
  dt = (time_jd - peak_jd) * 1440.0D0
  rise = EXP(-(dt < 0)^2 / (2.0D * rise_min^2))
  decay = EXP(-(dt > 0) / decay_min)
  RETURN, peak_flux * (rise * (dt LE 0) + decay * (dt GT 0))
END
```

## 단계 2: FITS로 저장

```idl
PRO save_goes_fits, goes_data, filename
  MKHDR, header, goes_data.flux_long
  SXADDPAR, header, 'TELESCOP', 'GOES-16'
  SXADDPAR, header, 'BUNIT', 'W/m^2'
  WRITEFITS, filename, goes_data.flux_long, header
  ; 추가 확장으로 단채널과 시간 배열 저장
  WRITEFITS, filename, goes_data.flux_short, /APPEND
  WRITEFITS, filename, goes_data.time_jd, /APPEND
END
```

## 단계 3: 플레어 이벤트 식별

```idl
FUNCTION find_flares, time_jd, flux, THRESHOLD=threshold
  IF N_ELEMENTS(threshold) EQ 0 THEN threshold = 1.0D-6
  n = N_ELEMENTS(flux)

  ; 임계값 이상의 로컬 최대값 찾기
  is_peak = BYTARR(n)
  FOR i = 1L, n - 2 DO BEGIN
    IF flux[i] GT flux[i-1] AND flux[i] GT flux[i+1] AND $
       flux[i] GT threshold THEN is_peak[i] = 1B
  ENDFOR

  peak_idx = WHERE(is_peak, n_peaks)
  IF n_peaks EQ 0 THEN RETURN, !NULL

  ; 플레어 분류
  flares = REPLICATE({peak_time: 0.0D0, peak_flux: 0.0D0, class: ''}, n_peaks)
  FOR i = 0, n_peaks - 1 DO BEGIN
    f = flux[peak_idx[i]]
    flares[i].peak_time = time_jd[peak_idx[i]]
    flares[i].peak_flux = f
    flares[i].class = (f GE 1e-4) ? 'X' : ((f GE 1e-5) ? 'M' : $
                       ((f GE 1e-6) ? 'C' : ((f GE 1e-7) ? 'B' : 'A')))
  ENDFOR

  RETURN, flares
END
```

## 단계 4: 광도 곡선 플롯

```idl
PRO plot_light_curve, goes_data, flares, TO_FILE=to_file
  time = goes_data.time_jd
  flux_long = goes_data.flux_long
  start_jd = MIN(time)
  time_hours = (time - start_jd) * 24.0D0

  IF KEYWORD_SET(to_file) THEN BEGIN
    SET_PLOT, 'PS'
    DEVICE, FILENAME=to_file, /COLOR, /ENCAPSULATED, XSIZE=20, YSIZE=14
  ENDIF

  thick = KEYWORD_SET(to_file) ? 3 : 2

  PLOT, time_hours, flux_long, /YLOG, $
    YRANGE=[1e-8, 1e-3], XRANGE=[0, MAX(time_hours)], $
    XSTYLE=1, YSTYLE=1, $
    TITLE='GOES X-ray Flux', $
    XTITLE='Time (UT hours)', $
    YTITLE='Flux (W m!U-2!N)', THICK=thick

  ; 플레어 분류 수준선
  class_levels = [1e-7, 1e-6, 1e-5, 1e-4]
  class_labels = ['B', 'C', 'M', 'X']
  FOR i = 0, 3 DO $
    OPLOT, [0, MAX(time_hours)], REPLICATE(class_levels[i], 2), LINESTYLE=1

  ; 플레어 표시
  IF N_ELEMENTS(flares) GT 0 THEN BEGIN
    FOR i = 0, N_ELEMENTS(flares) - 1 DO BEGIN
      peak_hour = (flares[i].peak_time - start_jd) * 24.0D0
      XYOUTS, peak_hour, flares[i].peak_flux * 2.5, flares[i].class, $
        ALIGNMENT=0.5, CHARSIZE=0.8
    ENDFOR
  ENDIF

  IF KEYWORD_SET(to_file) THEN BEGIN
    DEVICE, /CLOSE
    SET_PLOT, 'X'
  ENDIF
END
```

## 단계 5: 메인 프로그램 실행

```idl
PRO goes_light_curve
  PRINT, '===== GOES Solar X-ray Light Curve Analysis ====='
  goes_data = generate_goes_data()
  save_goes_fits, goes_data, 'goes_xrs_sim.fits'
  flares = find_flares(goes_data.time_jd, goes_data.flux_long, THRESHOLD=1.0D-7)
  plot_light_curve, goes_data, flares
  plot_light_curve, goes_data, flares, TO_FILE='goes_light_curve.eps'
  PRINT, '===== Analysis Complete ====='
END
```

---

## 확장 과제

1. **다중 일 플롯**: 여러 날의 데이터를 처리하고 X축에 날짜 틱 라벨 표시
2. **피크 감지 개선**: 각 플레어의 상승 시간과 감소 시간 계산
3. **배경 차감**: 러닝 중앙값 배경 차감으로 플레어 방출 분리
4. **미분 플롯**: 플럭스의 시간 미분 (dF/dt)을 보여주는 두 번째 패널 추가
5. **에너지 추정**: 배경 위의 플럭스를 적분하여 각 플레어의 총 복사 에너지 추정

---

## 요약

이 프로젝트에서 통합한 IDL 기술:

| 기술 | 사용 방법 |
|------|----------|
| 변수와 데이터 타입 | 더블 정밀도 시간과 플럭스 배열 |
| 배열과 연산 | 배열 생성, WHERE 필터링, 배열 수학 |
| 연산자 | 플레어 분류를 위한 관계 연산자 |
| 제어 흐름 | FOR 루프, IF/THEN/ELSE, CASE |
| 프로시저와 함수 | 별도 루틴으로 모듈화 설계 |
| 문자열 처리 | 라벨과 FITS 헤더 값 포맷팅 |
| 파일 I/O | READFITS/WRITEFITS를 사용한 FITS 읽기/쓰기 |
| 구조체 | 구조체 배열로 플레어 레코드 |
| 플로팅 | PLOT, OPLOT, XYOUTS로 출판 그림 |
| 날짜와 시간 | 율리우스 날짜, CALDAT, 시간 축 포맷팅 |
| 모범 사례 | 벡터화, 오류 확인, 문서화 |

IDL 기초를 완료한 것을 축하합니다! 이제 태양 물리학, 우주 과학 등에서 IDL/GDL로 과학 데이터를 다루는 기반을 갖추게 되었습니다.

---

**이전**: [디버깅과 모범 사례](./14_Debugging_and_Best_Practices.md)
