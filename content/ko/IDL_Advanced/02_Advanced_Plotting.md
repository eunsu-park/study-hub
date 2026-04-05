# 02. 고급 플로팅

**이전**: [고급 배열 기법](./01_Advanced_Array_Techniques.md) | **다음**: [지도 투영법](./03_Map_Projections.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `!P.MULTI`를 사용하여 다중 패널 플롯을 생성한다
2. 사용자 정의 레벨과 채우기 색상으로 등고선 플롯을 만든다
3. 3D 표면 및 음영 표면 플롯을 렌더링한다
4. PLOTS, XYOUTS, 그리기 요소로 주석을 오버레이한다
5. 과학 시각화를 위한 색상 테이블을 관리한다
6. 출판 품질의 PostScript 출력을 생성한다

---

## 1. !P.MULTI를 이용한 다중 패널 플롯

```idl
; !P.MULTI = [remaining, ncols, nrows, nz, order]
; 2열 x 2행 레이아웃
!P.MULTI = [0, 2, 2]

PLOT, FINDGEN(100), SIN(FINDGEN(100)*0.1), TITLE='Sine'
PLOT, FINDGEN(100), COS(FINDGEN(100)*0.1), TITLE='Cosine'
PLOT, FINDGEN(100), EXP(-FINDGEN(100)*0.05), TITLE='Exp Decay'
PLOT, FINDGEN(100), ALOG(FINDGEN(100)+1), TITLE='Log'

!P.MULTI = 0  ; 단일 패널로 복원
```

### POSITION을 이용한 세밀한 패널 배치

```idl
; POSITION = [x0, y0, x1, y1] 정규화 좌표 (0-1)
PLOT, x, y1, POSITION=[0.1, 0.55, 0.95, 0.95], TITLE='상단 패널', /NOERASE
PLOT, x, y2, POSITION=[0.1, 0.08, 0.5, 0.45], TITLE='하단 왼쪽', /NOERASE
PLOT, x, y3, POSITION=[0.55, 0.08, 0.95, 0.45], TITLE='하단 오른쪽', /NOERASE
```

---

## 2. 등고선 플롯

```idl
data = DIST(100)

; 기본 등고선
CONTOUR, data, TITLE='등고선 플롯'

; 채워진 등고선
levels = FINDGEN(10) * 8
LOADCT, 33
CONTOUR, data, LEVELS=levels, /FILL, $
    C_COLORS=BYTSCL(INDGEN(10), TOP=254), $
    TITLE='채워진 등고선'

; 등고선 선 오버레이
CONTOUR, data, LEVELS=levels, /OVERPLOT, $
    C_LABELS=REPLICATE(1, N_ELEMENTS(levels))
```

### 등고선 주요 키워드

| 키워드 | 설명 |
|--------|------|
| `LEVELS` | 등고선 레벨 값 배열 |
| `NLEVELS` | 등간격 레벨 수 |
| `/FILL` | 레벨 사이 채우기 |
| `C_COLORS` | 각 레벨의 색상 인덱스 |
| `C_THICK` | 각 레벨의 선 두께 |
| `C_LABELS` | 레이블 표시할 레벨 (0/1 배열) |
| `/OVERPLOT` | 기존 플롯 위에 그리기 |

---

## 3. 표면 플롯

```idl
data = DIST(50)

; 와이어프레임 표면
SURFACE, data, AX=45, AZ=30, $
    XTITLE='X', YTITLE='Y', ZTITLE='Z', $
    TITLE='회전된 표면'

; 음영 표면 (Gouraud 셰이딩)
LOADCT, 33
SHADE_SURF, data, SHADES=BYTSCL(data), $
    AX=50, AZ=45, TITLE='색상 음영 표면'
```

---

## 4. 오버플롯과 주석

### PLOTS — 플롯 위에 그리기

```idl
PLOT, x, y, TITLE='주석이 달린 플롯'

; 수평선 그리기
PLOTS, [0, 10], [0, 0], LINESTYLE=2, COLOR=200

; 사용자 정의 심볼 (채워진 원)
A = FINDGEN(17) * (!PI * 2 / 16.0)
USERSYM, COS(A), SIN(A), /FILL
PLOTS, peak_x, 1.0, PSYM=8, SYMSIZE=1.5, COLOR=250
```

### XYOUTS — 텍스트 주석

```idl
; 데이터 좌표에 텍스트 추가
XYOUTS, peak_x, 1.05, 'Peak', ALIGNMENT=0.5, CHARSIZE=1.2

; 정규화 좌표에 텍스트 추가
XYOUTS, 0.5, 0.02, 'IDL로 생성됨', /NORMAL, ALIGNMENT=0.5
```

---

## 5. 색상 테이블 관리

```idl
; 내장 색상 테이블 (0-74)
LOADCT, 0     ; 흑백 (회색조)
LOADCT, 3     ; 적색 온도
LOADCT, 13    ; 무지개
LOADCT, 33    ; 청색-적색
LOADCT, 39    ; 무지개+백색

; 사용자 정의 색상 테이블
r = BYTARR(256) & g = BYTARR(256) & b = BYTARR(256)
; 청색-백색-적색 다이버전트 컬러맵 생성
r[0:127] = BINDGEN(128) * 2
g[0:127] = BINDGEN(128) * 2
b[0:127] = 255
r[128:255] = 255
g[128:255] = REVERSE(BINDGEN(128) * 2)
b[128:255] = REVERSE(BINDGEN(128) * 2)
TVLCT, r, g, b
```

---

## 6. 출판 품질 PostScript 출력

```idl
; PostScript 디바이스 열기
SET_PLOT, 'PS'
DEVICE, FILENAME='figure1.ps', /ENCAPSULATED, $
    XSIZE=18, YSIZE=12, /COLOR, BITS_PER_PIXEL=8

; 출판용 두꺼운 선과 큰 텍스트
!P.THICK = 3
!X.THICK = 2
!Y.THICK = 2
!P.CHARSIZE = 1.2
!P.CHARTHICK = 2

LOADCT, 0
x = FINDGEN(200) * 0.05
PLOT, x, SIN(x), XTITLE='Time (s)', YTITLE='Amplitude', $
    TITLE='출판용 플롯', XSTYLE=1

DEVICE, /CLOSE
SET_PLOT, 'X'

; 플롯 매개변수 초기화
!P.THICK = 0 & !X.THICK = 0 & !Y.THICK = 0
!P.CHARSIZE = 0 & !P.CHARTHICK = 0
```

### 다중 페이지 PostScript

```idl
SET_PLOT, 'PS'
DEVICE, FILENAME='multi_page.ps', /COLOR, BITS_PER_PIXEL=8

; 페이지 1
!P.MULTI = [0, 2, 3]
FOR i = 0, 5 DO PLOT, FINDGEN(100), RANDOMN(seed, 100)

; 새 페이지
DEVICE, /ADVANCE

; 페이지 2
!P.MULTI = [0, 1, 2]
CONTOUR, DIST(100), NLEVELS=10, /FILL
SURFACE, DIST(50), AX=45, AZ=30

DEVICE, /CLOSE
SET_PLOT, 'X'
!P.MULTI = 0
```

---

## 7. 고급 플롯 사용자 정의

### 로그 축

```idl
; 로그-로그 플롯
PLOT, x, y, /XLOG, /YLOG, XTITLE='Frequency (Hz)', YTITLE='Power'
```

### 오차 막대

```idl
x = FINDGEN(20) * 0.5
y = SIN(x) + RANDOMN(seed, 20) * 0.1
yerr = REPLICATE(0.1, 20)

PLOT, x, y, PSYM=4, TITLE='오차 막대가 있는 데이터'
ERRPLOT, x, y-yerr, y+yerr
```

---

## 요약

| 기법 | 주요 함수/키워드 | 용도 |
|------|-----------------|------|
| 다중 패널 | `!P.MULTI`, `POSITION` | 여러 플롯 배치 |
| 등고선 | `CONTOUR`, `/FILL` | 2D 필드 시각화 |
| 표면 | `SURFACE`, `SHADE_SURF` | 3D 시각화 |
| 주석 | `PLOTS`, `XYOUTS`, `ARROW` | 레이블과 마커 |
| 색상 테이블 | `LOADCT`, `TVLCT` | 색상 관리 |
| PostScript | `SET_PLOT, 'PS'`, `DEVICE` | 출판 출력 |

---

**이전**: [고급 배열 기법](./01_Advanced_Array_Techniques.md) | **다음**: [지도 투영법](./03_Map_Projections.md)
