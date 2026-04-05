# 07. SDO/HMI 분석

**이전**: [SDO/AIA 분석](./06_SDO_AIA_Analysis.md) | **다음**: [GOES와 RHESSI](./08_GOES_and_RHESSI.md)

---

## 학습 목표

1. SDO/HMI 데이터 제품(자기도, Dopplergram, 연속광)을 이해한다
2. `read_sdo`와 `hmi_prep`으로 HMI 데이터를 읽고 보정한다
3. 시선 방향 및 벡터 자기장 데이터를 분석한다
4. 총 부호 없는 자기 플럭스와 플럭스 불균형을 계산한다
5. HMI 데이터에서 Carrington 종합 지도를 생성한다

---

## 1. SDO/HMI 개요

| 데이터 제품 | 키워드 | 케이던스 | 설명 |
|------------|--------|---------|------|
| 시선 방향 자기도 | `hmi.M_720s` | 720 s | LOS 자기장 (Gauss) |
| 연속광 강도 | `hmi.Ic_720s` | 720 s | 백색광 강도 |
| Dopplergram | `hmi.V_720s` | 720 s | 시선 방향 속도 (m/s) |
| 벡터 자기도 | `hmi.B_720s` | 720 s | 전체 B 벡터 |
| SHARP 데이터 | `hmi.sharp_720s` | 720 s | 활동 영역 패치 |

- **공간 해상도**: 0.505 arcsec/pixel (4096 x 4096)
- **분광선**: Fe I 6173 A
- **잡음 수준**: ~7 G (LOS), ~100 G (횡방향)

---

## 2. 시선 방향 자기장 분석

```idl
read_sdo, 'hmi_mag.fits', index, mag

; 총 부호 없는 자기 플럭스 계산
cdelt_rad = index.cdelt1 * !DTOR / 3600.0
pixel_area = (cdelt_rad * index.dsun_obs * 100.0)^2  ; cm^2

roi_mag = mag[1800:2200, 2000:2400]
flux_unsigned = TOTAL(ABS(roi_mag)) * pixel_area
PRINT, '총 부호 없는 플럭스: ', flux_unsigned, ' Mx'

; 양/음 플럭스
pos = WHERE(roi_mag GT 10, np)
neg = WHERE(roi_mag LT -10, nn)
flux_pos = (np GT 0) ? TOTAL(roi_mag[pos]) * pixel_area : 0.0
flux_neg = (nn GT 0) ? TOTAL(ABS(roi_mag[neg])) * pixel_area : 0.0
```

---

## 3. LOS에서 방사 방향 자기장 보정

```idl
; mu 보정: B_r ~ B_LOS / mu, mu = cos(theta)
rho = SQRT(xx^2 + yy^2) / index.rsun_obs
mu = SQRT(1.0 - rho^2 < 1.0)

Br_corrected = mag * 0.0
good = WHERE(mu GT 0.3, ngood)
IF ngood GT 0 THEN Br_corrected[good] = mag[good] / mu[good]
```

---

## 4. Carrington 종합 지도

```idl
; Carrington 종합 지도: 한 태양 회전(~27.3일) 동안의
; 중앙 자오선 스트립을 결합

; JSOC에서 미리 계산된 종합 지도 제공:
; hmi.Synoptic_Mr_720s — 방사 방향 자기장 종합
synoptic = READFITS('hmi_synoptic_mr_2277.fits', header)
```

---

## 5. SHARP 키워드 (우주 날씨용)

```idl
read_sdo, 'hmi_sharp.fits', idx, data

PRINT, 'USFLUX:  ', idx.usflux    ; 총 부호 없는 플럭스 (Mx)
PRINT, 'MEANGBT: ', idx.meangbt   ; 평균 총 자기장 기울기 (G/Mm)
PRINT, 'MEANJZD: ', idx.meanjzd   ; 평균 수직 전류 밀도 (mA/m^2)
PRINT, 'MEANSHR: ', idx.meanshr   ; 평균 전단각 (degrees)
; 이 매개변수들은 플레어 예측 모델에 사용됨
```

---

## 요약

| 주제 | 핵심 루틴 | 용도 |
|------|----------|------|
| 데이터 I/O | `read_sdo` | HMI FITS 읽기 |
| 보정 | `hmi_prep` | 표준 보정 |
| LOS 자기장 | 직접 분석 | 자기도 분석 |
| 벡터 자기장 | SHARP 데이터 | 전체 자기 벡터 |
| 플럭스 | `TOTAL`, 픽셀 면적 | 부호 없는 플럭스 |
| 종합 지도 | 중앙 자오선 스트립 | Carrington 지도 |

---

**이전**: [SDO/AIA 분석](./06_SDO_AIA_Analysis.md) | **다음**: [GOES와 RHESSI](./08_GOES_and_RHESSI.md)
