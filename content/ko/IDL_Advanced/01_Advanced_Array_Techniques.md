# 01. 고급 배열 기법

**다음**: [고급 플로팅](./02_Advanced_Plotting.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. REFORM으로 배열을 재구성하고 메모리 레이아웃 영향을 이해한다
2. REBIN과 CONGRID로 데이터를 업샘플링/다운샘플링한다
3. TOTAL, MEAN, MEDIAN으로 차원별 축소를 수행한다
4. SMOOTH, MEDIAN, CONVOL로 스무딩 필터를 적용한다
5. IMAGE_STATISTICS로 통계 요약을 추출한다
6. 효율적인 다차원 배열 조작 패턴을 작성한다

---

## 1. REFORM을 이용한 배열 재구성

`REFORM`은 데이터를 복사하지 않고 배열의 차원을 변경합니다. 총 요소 수는 동일해야 합니다.

```idl
; 12개 요소의 1D 배열 생성
arr = INDGEN(12)
PRINT, SIZE(arr, /DIMENSIONS)   ; 12

; 3x4로 재구성
arr2d = REFORM(arr, 3, 4)
PRINT, SIZE(arr2d, /DIMENSIONS) ; 3 4

; 2x2x3으로 재구성
arr3d = REFORM(arr, 2, 2, 3)
PRINT, SIZE(arr3d, /DIMENSIONS) ; 2 2 3

; 1D로 다시 평탄화
flat = REFORM(arr3d, 12)
```

### 주요 사용 사례

```idl
; 크기 1인 차원 제거 (예: 512 x 512 x 1 -> 512 x 512)
img = FLTARR(512, 512, 1)
img = REFORM(img)               ; 이제 512 x 512
PRINT, SIZE(img, /DIMENSIONS)   ; 512 512

; 데이터 큐브에서 시계열 추출
datacube = RANDOMU(seed, 256, 256, 100)
pixel_ts = REFORM(datacube[128, 128, *])  ; [100]
```

### 메모리 레이아웃: 열 우선 순서

IDL은 **열 우선**(Fortran 규약) 순서를 사용합니다. 첫 번째 인덱스가 메모리에서 가장 빠르게 변합니다:

```idl
; arr[3, 4]의 메모리 레이아웃:
;   arr[0,0], arr[1,0], arr[2,0],   <- 첫 번째 열
;   arr[0,1], arr[1,1], arr[2,1],   <- 두 번째 열
;   ...
; 이는 C/Python(행 우선)과 반대입니다

; REFORM은 데이터를 재배열하지 않고 형태 메타데이터만 변경합니다
; 따라서 REFORM(arr, 4, 3)과 TRANSPOSE(arr)는 다른 논리적 뷰를 제공합니다
```

---

## 2. REBIN을 이용한 리비닝

`REBIN`은 정수 배수로 배열 크기를 변경합니다. 축소 시 평균, 확대 시 보간을 수행합니다.

```idl
; 2x2 -> 4x4 (2배 확대)
small = [[1.0, 2.0], [3.0, 4.0]]
big = REBIN(small, 4, 4)

; 1024x1024 -> 256x256 (4배 축소, 평균)
big_img = DIST(1024)
small_img = REBIN(big_img, 256, 256)

; 최근접 이웃 확대 (보간 없음)
big_sample = REBIN(small, 16, 16, /SAMPLE)
```

### 다차원 데이터 리비닝

```idl
; 3D 데이터 큐브: [512, 512, 100] -> [128, 128, 100]
; 공간 차원만 리비닝, 시간 차원 유지
cube = FLTARR(512, 512, 100)
cube_small = REBIN(cube, 128, 128, 100)
```

---

## 3. CONGRID를 이용한 크기 변경

REBIN과 달리 `CONGRID`는 **임의의** 차원으로 크기를 변경할 수 있습니다 (정수 배수 불필요).

```idl
; 100x100에서 256x256으로 크기 변경
img = DIST(100)
img_resized = CONGRID(img, 256, 256)

; 고품질 큐빅 보간
img_cubic = CONGRID(img, 256, 256, /INTERP, CUBIC=-0.5)
```

| 특징 | REBIN | CONGRID |
|------|-------|---------|
| 크기 변경 비율 | 정수 배수만 | 임의 크기 |
| 축소 | 평균 (플럭스 보존) | 보간 (플럭스 비보존) |
| 속도 | 빠름 | 큰 배열에서 느림 |
| 용도 | 과학 데이터 축소 | 디스플레이/시각화 |

---

## 4. TOTAL을 이용한 차원 축소

```idl
arr = FINDGEN(3, 4) + 1

; 모든 요소 합
PRINT, TOTAL(arr)    ; 78.0000

; 차원 1(열)을 따라 합 -> [4] 배열
col_sum = TOTAL(arr, 1)

; 차원 2(행)을 따라 합 -> [3] 배열
row_sum = TOTAL(arr, 2)

; 누적 합
cum = TOTAL(FINDGEN(5)+1, /CUMULATIVE)
```

### 실제 예: 기둥 밀도 계산

```idl
; 3D 밀도 큐브에서 기둥 밀도 계산
density_cube = RANDOMU(seed, 256, 256, 100) * 1e10  ; particles/cm^3
dz = 1e8  ; 셀당 1000 km
column_density = TOTAL(density_cube, 3) * dz  ; [256, 256], particles/cm^2
```

---

## 5. MEAN, MEDIAN, 통계 함수

```idl
; 차원별 평균
cube = RANDOMU(seed, 100, 100, 50)
time_mean = MEAN(cube, DIMENSION=3)  ; [100, 100] 시간 평균

; 중간값 필터 (잡음 제거에 효과적)
img = DIST(256) + RANDOMN(seed, 256, 256) * 20
img_filtered = MEDIAN(img, 3)  ; 3x3 중간값 필터

; MOMENT — 평균, 분산, 왜도, 첨도를 한 번에 계산
data = RANDOMN(seed, 10000)
result = MOMENT(data)
```

---

## 6. SMOOTH를 이용한 스무딩

```idl
; 1D 박스카 스무딩
signal = SIN(FINDGEN(200) * 0.1) + RANDOMN(seed, 200) * 0.3
smoothed = SMOOTH(signal, 11)

; 2D 박스카 스무딩
img_smooth = SMOOTH(img, 5, /EDGE_TRUNCATE)

; 가우시안 스무딩 (커널 생성 후 CONVOL 사용)
sigma = 2.0
ksize = 11
x = FINDGEN(ksize) - ksize/2
kernel_1d = EXP(-x^2 / (2.0 * sigma^2))
kernel_1d = kernel_1d / TOTAL(kernel_1d)
kernel_2d = kernel_1d # kernel_1d
img_gauss = CONVOL(img, kernel_2d, /EDGE_TRUNCATE)
```

---

## 7. CONVOL을 이용한 합성곱

```idl
; 샤프닝 커널 (라플라시안 + 단위행렬)
sharpen_kernel = [[-1, -1, -1], $
                  [-1,  9, -1], $
                  [-1, -1, -1]]
img_sharp = CONVOL(FLOAT(img), sharpen_kernel, /EDGE_TRUNCATE)

; 사용자 정의 커널과 정규화
blur_kernel = FLTARR(5, 5) + 1.0
img_blur = CONVOL(FLOAT(img), blur_kernel, TOTAL(blur_kernel), /EDGE_TRUNCATE)
```

---

## 8. IMAGE_STATISTICS

```idl
img = DIST(512) + RANDOMN(seed, 512, 512) * 10

IMAGE_STATISTICS, img, $
    COUNT=count, MEAN=img_mean, STDDEV=img_stddev, $
    MINIMUM=img_min, MAXIMUM=img_max

; 마스크를 사용한 관심 영역 통계
mask = BYTARR(512, 512)
mask[100:400, 100:400] = 1B
IMAGE_STATISTICS, img, MASK=mask, MEAN=roi_mean, STDDEV=roi_stddev
```

---

## 9. 다차원 배열 연산

### 브로드캐스팅 스타일 연산

```idl
; 2D 배열에서 행별 평균 빼기
data = RANDOMU(seed, 100, 50)
time_mean = MEAN(data, DIMENSION=2)
time_mean_2d = REBIN(time_mean, 100, 50)
data_detrended = data - time_mean_2d
```

### WHERE를 이용한 다차원 인덱싱

```idl
; 데이터 큐브에서 임계값 초과 픽셀 찾기
cube = RANDOMU(seed, 100, 100, 50)
idx = WHERE(cube GT 0.95, count)
subscripts = ARRAY_INDICES(cube, idx)
```

### 효율적인 러닝 디퍼런스

```idl
; 러닝 디퍼런스: diff[t] = data[t] - data[t-1]
nt = 100
data = RANDOMU(seed, 256, 256, nt)
diff = data[*, *, 1:nt-1] - data[*, *, 0:nt-2]
```

---

## 10. 실전 예제: 태양 이미지 큐브 시계열 분석

```idl
; 시뮬레이션된 태양 EUV 이미지 큐브: [512, 512, 200], 12초 간격
nx = 512 & ny = 512 & nt = 200
cadence = 12.0

cube = FLTARR(nx, ny, nt)
FOR t = 0, nt-1 DO $
    cube[*, *, t] = DIST(nx) * (1.0 + 0.1 * SIN(2*!PI*t/50.0))
cube += RANDOMN(seed, nx, ny, nt) * 10.0

; 시간 스무딩 (5프레임 박스카)
cube_smooth = FLTARR(nx, ny, nt)
FOR i = 0, nx-1 DO FOR j = 0, ny-1 DO $
    cube_smooth[i, j, *] = SMOOTH(REFORM(cube[i, j, *]), 5, /EDGE_TRUNCATE)

; 공간 리비닝
cube_rebin = REBIN(cube_smooth, 128, 128, nt)

; 관심 영역 광도곡선
x1 = 50 & x2 = 70 & y1 = 50 & y2 = 70
roi = cube_rebin[x1:x2, y1:y2, *]
lightcurve = REFORM(TOTAL(TOTAL(roi, 1), 1))

; 프레임별 통계
frame_means = FLTARR(nt)
FOR t = 0, nt-1 DO BEGIN
    IMAGE_STATISTICS, cube_rebin[*, *, t], MEAN=m
    frame_means[t] = m
ENDFOR
```

---

## 요약

| 함수 | 용도 | 주요 키워드 |
|------|------|------------|
| `REFORM` | 복사 없이 재구성 | — |
| `REBIN` | 정수 배수 크기 변경 | `/SAMPLE` |
| `CONGRID` | 임의 크기 변경 | `/INTERP`, `CUBIC=` |
| `TOTAL` | 차원별 합 | `/CUMULATIVE`, `/NAN` |
| `SMOOTH` | 박스카 평균 | `/EDGE_TRUNCATE` |
| `CONVOL` | N차원 합성곱 | `/EDGE_TRUNCATE`, `/EDGE_WRAP` |
| `IMAGE_STATISTICS` | 종합 이미지 통계 | `MASK=` |

---

**다음**: [고급 플로팅](./02_Advanced_Plotting.md)
