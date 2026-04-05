# 15. 캡스톤: 태양 이벤트 분석

**이전**: [성능과 대용량 데이터](./14_Performance_and_Large_Data.md)

---

## 학습 목표

이 캡스톤 프로젝트에서 다음을 수행합니다:

1. 태양 플레어 이벤트의 SDO/AIA 다중 파장 데이터를 다운로드한다
2. AIA_PREP으로 모든 이미지를 보정한다
3. 폭발 역학을 시각화하기 위한 러닝 디퍼런스 이미지를 생성한다
4. 다중 파장 합성 이미지를 만든다
5. 플레어 영역 강도의 시계열을 추출하고 분석한다
6. 광도곡선에서 플레어 발생을 감지한다
7. 출판 품질의 그림 세트를 제작하고 PostScript로 출력한다

---

## 프로젝트 개요

```
데이터 수집 → 보정 → 시각화 → 시계열 → 감지 → 출판 그림
```

---

## 단계 1: 데이터 수집

```idl
; 이벤트 정의
event_date = '2024-01-15'
t_start = '2024-01-15T11:30:00'
t_end = '2024-01-15T13:00:00'
channels = [94, 131, 171, 193, 211, 304, 335]

; VSO를 통해 데이터 다운로드
FOR ic = 0, N_ELEMENTS(channels)-1 DO BEGIN
    vso_search, t_start, t_end, $
        INSTRUMENT='aia', WAVE=STRTRIM(channels[ic], 2), results, /FLAT
    vso_get, results[0:59], OUT_DIR=wave_dir
ENDFOR
```

---

## 단계 2: 보정

```idl
FOR ic = 0, N_ELEMENTS(channels)-1 DO BEGIN
    files = FILE_SEARCH(wave_dir + '*.fits', COUNT=nf)
    FOR i = 0, nf-1 DO BEGIN
        read_sdo, files[i], index, data
        aia_prep, index, data, oindex, odata, /NORMALIZE, /REGISTER
        mwritefits, oindex, odata, OUTFILE=outfile
        data = 0 & odata = 0  ; 메모리 해제
    ENDFOR
ENDFOR
```

---

## 단계 3: 러닝 디퍼런스 이미지

```idl
; 활동 영역 서브필드 추출
wcs = FITSHEAD2WCS(index[0])
pix_ll = WCS_GET_PIXEL(wcs, [xcen-half_fov, ycen-half_fov])
pix_ur = WCS_GET_PIXEL(wcs, [xcen+half_fov, ycen+half_fov])
sub_cube = data_cube[x0:x1, y0:y1, *]

; 러닝 디퍼런스 계산
run_diff = sub_cube[*, *, 1:nf-1] - sub_cube[*, *, 0:nf-2]

; PNG로 저장
LOADCT, 33
FOR t = 0, nf-2 DO BEGIN
    img = BYTSCL(run_diff[*, *, t], MIN=-vmax, MAX=vmax)
    WRITE_PNG, diff_dir + STRING(t, FORMAT='("diff_", I04, ".png")'), $
        CONGRID(img, 512, 512)
ENDFOR
```

---

## 단계 4: 다중 파장 합성 이미지

```idl
; 플레어 피크 시간에 3색 합성 생성
; 304 -> 적색, 171 -> 녹색, 193 -> 청색
r = BYTSCL(ALOG10(od304 > 1), MIN=0.5, MAX=3.5)
g = BYTSCL(ALOG10(od171 > 1), MIN=0.5, MAX=3.5)
b = BYTSCL(ALOG10(od193 > 1), MIN=0.5, MAX=3.5)
WRITE_PNG, 'composite_peak.png', r, g, b
```

---

## 단계 5: 시계열 분석

```idl
; 모든 채널에서 광도곡선 추출
lightcurves = DBLARR(n_channels, n_time_pts)

FOR ic = 0, n_channels-1 DO BEGIN
    FOR t = 0, n_time_pts-1 DO BEGIN
        read_sdo, files_w[t], idx_w, dat_w
        lightcurves[ic, t] = MEAN(dat_w[x0+fx0:x0+fx1, y0+fy0:y0+fy1])
    ENDFOR
ENDFOR
```

---

## 단계 6: 플레어 발생 감지

```idl
lc = REFORM(lightcurves[2, *])  ; 171 A

; 플레어 전 기준선 (처음 10분)
pre_flare_idx = WHERE(t_min LT 10.0)
baseline_mean = MEAN(lc[pre_flare_idx])
baseline_std = STDDEV(lc[pre_flare_idx])

; 발생: 광도곡선이 기준선 + 3*시그마를 처음 초과하는 시점
threshold = baseline_mean + 3.0 * baseline_std
onset_idx = (WHERE(lc GT threshold AND t_min GT 10.0))[0]

PRINT, '플레어 발생: ', ANYTIM(times_sec[onset_idx], /CCSDS)
PRINT, '피크 강도:   ', MAX(lc), ' DN/s'
PRINT, '향상 비율:   ', MAX(lc) / baseline_mean, '배'
```

---

## 단계 7: 출판 품질 그림

```idl
SET_PLOT, 'PS'
DEVICE, FILENAME='figure1_lightcurves.eps', $
    /ENCAPSULATED, /COLOR, BITS=8, XSIZE=18, YSIZE=22

!P.THICK = 3
!P.CHARTHICK = 2
!P.CHARSIZE = 0.9
!P.MULTI = [0, 1, n_channels]

FOR ic = 0, n_channels-1 DO BEGIN
    PLOT, t_min, REFORM(lightcurves[ic, *]), $
        XTITLE=(ic EQ n_channels-1) ? 'Time (min)' : '', $
        YTITLE='DN/s', TITLE=channel_names[ic]
    ; 발생과 피크 표시
    PLOTS, [t_min[onset_idx], t_min[onset_idx]], !Y.CRANGE, LINESTYLE=2
ENDFOR

DEVICE, /CLOSE
SET_PLOT, 'X'
!P.THICK = 0 & !P.CHARTHICK = 0 & !P.MULTI = 0
```

---

## 프로젝트 확장

1. **DEM 분석**: 6개 EUV 채널로 플레어 피크의 DEM 맵 계산
2. **자기장 맥락**: AIA 이미지에 HMI 자기도 등고선 오버레이
3. **GOES 상관**: GOES 1-8 A 광도곡선과 AIA 광도곡선 비교
4. **EUV 파동 검출**: 전체 태양 디스크에서 EUV 파동 검출
5. **스펙트럼 분석**: 플레어 전 광도곡선에 FFT 적용하여 준주기적 맥동 탐색

---

## 체크리스트

- [ ] 이벤트를 포함하는 모든 채널 데이터 다운로드
- [ ] 모든 이미지를 aia_prep으로 보정
- [ ] 러닝 디퍼런스 동영상 생성
- [ ] 플레어 피크의 다중 파장 합성 이미지
- [ ] 모든 채널의 광도곡선 추출
- [ ] 플레어 발생 및 피크 시간 식별
- [ ] 상승 시간 및 감쇠 시간 측정
- [ ] 출판 품질 PostScript 그림 생성
- [ ] 분석 결과 저장 (.sav 파일)

---

**이전**: [성능과 대용량 데이터](./14_Performance_and_Large_Data.md)
