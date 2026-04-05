# 06. SDO/AIA 분석

**이전**: [SolarSoft 프레임워크](./05_SolarSoft_Framework.md) | **다음**: [SDO/HMI 분석](./07_SDO_HMI_Analysis.md)

---

## 학습 목표

1. SDO/AIA 장비 특성(채널, 케이던스, 해상도)을 이해한다
2. `read_sdo`와 `aia_prep`으로 AIA 데이터를 읽고 보정한다
3. AIA 응답 함수로 온도 진단을 수행한다
4. 다중 파장 합성 이미지를 생성한다
5. 기본적인 DEM(Differential Emission Measure) 분석을 수행한다

---

## 1. SDO/AIA 개요

| 채널 (A) | 이온 | log T (K) | 영역 | 케이던스 |
|----------|------|-----------|------|---------|
| 94 | Fe XVIII | 6.8 | 플레어 플라즈마 | 12 s |
| 131 | Fe VIII, XXI | 5.6, 7.0 | 전이 영역 / 플레어 | 12 s |
| 171 | Fe IX | 5.8 | 조용한 코로나, 루프 | 12 s |
| 193 | Fe XII, XXIV | 6.2, 7.3 | 코로나 / 고온 플레어 | 12 s |
| 211 | Fe XIV | 6.3 | 활동 영역 코로나 | 12 s |
| 304 | He II | 4.7 | 채층 / 전이 영역 | 12 s |
| 335 | Fe XVI | 6.4 | 활동 영역 코로나 | 12 s |

- **공간 해상도**: 0.6 arcsec/pixel (4096 x 4096 픽셀)
- **시야**: 41 arcmin (태양 지름의 1.3배)

---

## 2. AIA 데이터 읽기 및 보정

```idl
; AIA FITS 파일 읽기
read_sdo, 'aia_171.fits', index, data

; aia_prep으로 보정
aia_prep, index, data, oindex, odata, $
    /NORMALIZE, $    ; 노출 시간으로 정규화 (DN/s)
    /REGISTER, $     ; 공통 포인팅으로 정합
    /CUTOUT           ; 4096x4096으로 자르기
```

### aia_prep이 수행하는 작업

| 단계 | 설명 |
|------|------|
| 다크/페데스탈 빼기 | CCD 바이어스 제거 |
| 플랫 필드 보정 | 픽셀별 감도 보정 |
| 스파이크 제거 | 우주선 히트 제거 |
| 노출 정규화 | DN/s로 변환 (/NORMALIZE) |
| 롤 보정 | 위성 롤 각도 보정 |
| 포인팅 업데이트 | 최신 포인팅 보정 적용 |

---

## 3. AIA 응답 함수

```idl
; 온도 응답 함수 가져오기
tresp = AIA_GET_RESPONSE(/TEMPERATURE, /DN)

; 응답 함수 플롯
PLOT, tresp.logte, tresp.a171, /YLOG, $
    XTITLE='log T (K)', YTITLE='Response', $
    TITLE='AIA 온도 응답 함수'
OPLOT, tresp.logte, tresp.a193, COLOR=150
OPLOT, tresp.logte, tresp.a304, COLOR=250
```

---

## 4. 다중 파장 합성 이미지

```idl
; 3색 합성: 171 (녹색), 193 (청색), 304 (적색)
r = BYTSCL(ALOG10(od304 > 1), MIN=0, MAX=3.5)  ; 304 -> 적색
g = BYTSCL(ALOG10(od171 > 1), MIN=0, MAX=3.5)  ; 171 -> 녹색
b = BYTSCL(ALOG10(od193 > 1), MIN=0, MAX=3.5)  ; 193 -> 청색

WRITE_PNG, 'aia_composite.png', $
    CONGRID(r, 1024, 1024), $
    CONGRID(g, 1024, 1024), $
    CONGRID(b, 1024, 1024)
```

---

## 5. DEM(Differential Emission Measure) 기초

```
I_channel = integral{ R_channel(T) * DEM(T) * dT }

여기서:
  I_channel = 관측 강도 (DN/s/pixel)
  R_channel(T) = 온도 응답 함수
  DEM(T) = 차분 방출 측정 [cm^-5 K^-1]
```

```idl
; aia_bp_estimate — 빠른 온도 추정
aia_bp_estimate, I_obs, tmap, emmap
PRINT, '최적 log T: ', ALOG10(tmap)
PRINT, '최적 EM:    ', emmap, ' cm^-5'
```

---

## 요약

| 주제 | 핵심 루틴 | 용도 |
|------|----------|------|
| 데이터 I/O | `read_sdo` | AIA FITS 읽기 |
| 보정 | `aia_prep` | 표준 파이프라인 보정 |
| 응답 | `aia_get_response` | 온도 응답 함수 |
| 색상 테이블 | `aia_lct` | 채널별 색상 맵 |
| DEM | `aia_bp_estimate` | 온도 진단 |

---

**이전**: [SolarSoft 프레임워크](./05_SolarSoft_Framework.md) | **다음**: [SDO/HMI 분석](./07_SDO_HMI_Analysis.md)
