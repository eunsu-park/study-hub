# 이미지 표시

**이전**: [기본 플로팅](./10_Basic_Plotting.md) | **다음**: [FITS 파일 처리](./12_FITS_File_Handling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TV와 TVSCL로 이미지 표시하기
2. 이미지용 그래픽 윈도우 생성하고 관리하기
3. LOADCT로 컬러 테이블 로드하고 사용하기
4. BYTSCL로 이미지 데이터 스케일링하기
5. CONGRID와 REBIN으로 이미지 크기 변경하기
6. RGB 컬러 모델과 분해 색상 작업하기
7. !D 시스템 변수로 디바이스 독립 그래픽 이해하기

---

이미지 표시는 태양 디스크 이미지, 스펙트로그램, 시뮬레이션 출력을 보는 데 기본입니다.

## TV와 TVSCL

```idl
; TV — 바이트 배열을 이미지로 표시 (입력은 BYTE여야 함)
image = BYTSCL(DIST(256))
WINDOW, 0, XSIZE=256, YSIZE=256
TV, image

; TVSCL — 자동 스케일링으로 표시 (부동소수점 데이터 직접 표시)
data = DIST(200)
TVSCL, data
```

## BYTSCL — 바이트 스케일링

```idl
; 데이터를 [0, 255] 범위로 선형 스케일링
data = RANDOMN(seed, 256, 256) * 100.0
scaled = BYTSCL(data, MIN=-200, MAX=200)

; 일반적인 패턴: 클리핑으로 표시
image = RANDOMN(seed, 512, 512) + 5.0
mean_val = MEAN(image)
sigma = STDDEV(image)
display = BYTSCL(image, MIN=mean_val - 3*sigma, MAX=mean_val + 3*sigma)
```

## LOADCT — 컬러 테이블

```idl
LOADCT, 0      ; 흑백 (그레이스케일)
LOADCT, 3      ; Red Temperature
LOADCT, 13     ; Rainbow
LOADCT, 39     ; Rainbow + White

; 사용자 정의 컬러 테이블
n = 256
r = BINDGEN(n)
g = BYTARR(n)
b = REVERSE(BINDGEN(n))
TVLCT, r, g, b
```

## 이미지 크기 변경

```idl
; CONGRID — 임의 크기로 변경 (보간)
small = DIST(64)
big = CONGRID(small, 512, 512, /INTERP)

; REBIN — 정수 배율로 변경
big = REBIN(small, 256, 256)
```

## RGB 컬러 모델

```idl
; 인덱스 컬러 모드 vs 분해 (트루 컬러) 모드
DEVICE, GET_DECOMPOSED=mode
DEVICE, DECOMPOSED=0    ; 인덱스 컬러 (8비트)
DEVICE, DECOMPOSED=1    ; 트루 컬러 (24비트)

; RGB 이미지 (3 x nx x ny)
rgb = BYTARR(3, nx, ny)
rgb[0, *, *] = red_channel
rgb[1, *, *] = green_channel
rgb[2, *, *] = blue_channel
TV, rgb, TRUE=1
```

## 이미지 파일 저장

```idl
WRITE_PNG, 'image.png', image
WRITE_JPEG, 'image.jpg', image, QUALITY=95
```

## Z-버퍼 디바이스

디스플레이 없이 메모리에서 이미지를 렌더링합니다 (배치 처리에 유용):

```idl
SET_PLOT, 'Z'
DEVICE, SET_RESOLUTION=[800, 600]
PLOT, x, y, TITLE='Z-Buffer Plot'
snapshot = TVRD()
SET_PLOT, original
WRITE_PNG, 'plot.png', snapshot
```

---

## 요약

| 프로시저/함수 | 설명 |
|--------------|------|
| `TV, image` | 바이트 이미지 표시 |
| `TVSCL, data` | 자동 스케일링으로 표시 |
| `BYTSCL(data)` | 데이터를 0-255 범위로 스케일링 |
| `LOADCT, n` | 컬러 테이블 번호 n 로드 |
| `CONGRID(img, nx, ny)` | 임의 크기로 변경 |
| `REBIN(img, nx, ny)` | 정수 배율로 변경 |
| `WRITE_PNG` | PNG 파일로 저장 |

---

**이전**: [기본 플로팅](./10_Basic_Plotting.md) | **다음**: [FITS 파일 처리](./12_FITS_File_Handling.md)
