# 03. 지도 투영법

**이전**: [고급 플로팅](./02_Advanced_Plotting.md) | **다음**: [객체지향 IDL](./04_Object_Oriented_IDL.md)

---

## 학습 목표

1. MAP_SET으로 다양한 투영 유형의 지도를 설정한다
2. 대륙 경계와 좌표 격자를 오버레이한다
3. 태양 좌표계(heliographic, Carrington)를 이해한다
4. 태양 데이터를 위한 WCS(World Coordinate System)를 이해한다
5. 좌표 변환을 사용하여 태양 데이터를 지도에 투영한다

---

## 1. MAP_SET — 투영법 설정

```idl
; 정사 투영(Orthographic) — 지구본 뷰
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC, TITLE='정사 투영'
MAP_CONTINENTS, /FILL, COLOR=200
MAP_GRID, /LABEL, LATDEL=30, LONDEL=30
```

### 주요 투영법

```idl
MAP_SET, 0, 0, 0, /ORTHOGRAPHIC, /ISOTROPIC  ; 지구본
MAP_SET, 0, 0, 0, /MOLLWEIDE, /ISOTROPIC     ; 등적(전천)
MAP_SET, 0, 0, 0, /MERCATOR                   ; 원통
MAP_SET, 0, 0, 0, /AITOFF, /ISOTROPIC         ; 등적(천문학)
MAP_SET, 90, 0, 0, /STEREOGRAPHIC, /ISOTROPIC ; 극지방용
```

---

## 2. 태양 좌표계

### Heliographic 좌표

```idl
; Heliographic Stonyhurst (HGS):
; - 경도: 중앙 자오선에서 0, -180 ~ +180
; - 위도: -90 ~ +90, 태양 적도에서 0
; - 태양 자전축에 고정

; Heliographic Carrington (HGC):
; - 위도는 Stonyhurst와 동일
; - 경도는 태양과 함께 회전 (Carrington 회전 주기 = 27.2753일)
```

### Helioprojective 좌표

```idl
; Helioprojective Cartesian (HPC):
; - Theta_x (arcsec): 태양 중심으로부터 solar-X 방향 각변위
; - Theta_y (arcsec): solar-Y 방향 각변위
; - AIA/HMI 이미지에서 보는 좌표계 (픽셀이 arcsec에 대응)
```

### SolarSoft 좌표 유틸리티

```idl
; 태양 B0 각도, L0, P 각도 구하기
sun_data = PB0R('2024-01-15T12:00:00')
PRINT, 'P 각도:  ', sun_data[0]
PRINT, 'B0 각도: ', sun_data[1]
PRINT, 'R_sun:   ', sun_data[2], ' arcmin'

; 날짜에서 Carrington 회전 번호
carr = TIM2CARR('2024-01-15T12:00:00')

; 좌표 변환
hel = ARCMIN2HEL(5.0, 3.0, DATE='2024-01-15')
```

---

## 3. World Coordinate System (WCS)

```idl
; FITS 헤더에서 WCS 정보 추출
data = READFITS('aia_171_image.fits', header)
wcs = FITSHEAD2WCS(header)

; 주요 WCS 매개변수:
; CRPIX1, CRPIX2 — 기준 픽셀
; CRVAL1, CRVAL2 — 기준 픽셀의 좌표 (arcsec)
; CDELT1, CDELT2 — 픽셀 스케일 (arcsec/pixel)

; 픽셀-월드 좌표 변환
wcs_coord = WCS_GET_COORD(wcs, [2048.0, 2048.0])

; 월드-픽셀 좌표 변환
pixel = WCS_GET_PIXEL(wcs, [arcsec_x, arcsec_y])
```

---

## 4. 태양 데이터의 지도 투영

```idl
; Carrington 종합 지도를 Mollweide 투영에 표시
MAP_SET, 0, 180, 0, /MOLLWEIDE, /ISOTROPIC, TITLE='Carrington 종합 지도'
result = MAP_IMAGE(bfield, startx, starty, $
    LATMIN=-90, LATMAX=90, LONMIN=0, LONMAX=360, /BILINEAR)
LOADCT, 0
TV, BYTSCL(result, MIN=-20, MAX=20), startx, starty
MAP_GRID, /LABEL, LATDEL=30, LONDEL=30
```

---

## 5. 차분 회전 보정

```idl
; 태양은 차분 회전함: 적도에서 빠르고 극에서 느림
; Snodgrass & Ulrich (1990):
;   omega(lat) = 14.713 - 2.396*sin^2(lat) - 1.787*sin^4(lat) [deg/day]

FUNCTION diff_rot_rate, lat_deg
    lat_rad = lat_deg * !DTOR
    sin2 = SIN(lat_rad)^2
    RETURN, 14.713 - 2.396*sin2 - 1.787*sin2^2
END
```

---

## 요약

| 주제 | 주요 루틴 | 용도 |
|------|----------|------|
| 지도 설정 | `MAP_SET` | 투영법 설정 |
| 경계선 | `MAP_CONTINENTS`, `MAP_GRID` | 지리적 오버레이 |
| 태양 좌표 | `PB0R`, `ARCMIN2HEL`, `TIM2CARR` | 태양 기하학 |
| WCS | `FITSHEAD2WCS`, `WCS_GET_COORD` | FITS 좌표계 |

---

**이전**: [고급 플로팅](./02_Advanced_Plotting.md) | **다음**: [객체지향 IDL](./04_Object_Oriented_IDL.md)
