# 10. 이미지 처리

**이전**: [스펙트럼 분석](./09_Spectral_Analysis.md) | **다음**: [곡선 피팅](./11_Curve_Fitting.md)

---

## 학습 목표

1. 태양 이미지에 스무딩 및 중간값 필터를 적용한다
2. 형태학적 연산(열기, 닫기, 팽창, 침식)을 사용한다
3. Sobel, Roberts, Laplacian 연산자로 에지를 검출한다
4. LABEL_REGION으로 연결 영역을 식별하고 레이블링한다
5. 러닝 디퍼런스 및 베이스 디퍼런스 이미지를 생성한다
6. 태양 이미지 시계열에서 특징을 추적한다

---

## 1. 이미지 필터링

```idl
; 박스카 스무딩
img_boxcar = SMOOTH(img, 5, /EDGE_TRUNCATE)

; 가우시안 스무딩
img_gauss = CONVOL(img, kernel_2d, /EDGE_TRUNCATE)

; 중간값 필터 (소금-후추 잡음에 효과적)
img_median = MEDIAN(img, 5)

; 비선명 마스킹 (선명화)
img_unsharp = img + 2.0 * (img - SMOOTH(img, 11, /EDGE_TRUNCATE))
```

---

## 2. 에지 검출

```idl
; Sobel 연산자
img_edges = SOBEL(img)

; Roberts 교차
img_roberts = ROBERTS(img)

; Laplacian
laplacian_kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
img_laplacian = CONVOL(FLOAT(img), laplacian_kernel, /EDGE_TRUNCATE)
```

---

## 3. 형태학적 연산

```idl
; 구조 요소 생성
radius = 3
se_disk = SHIFT(DIST(2*radius+1), radius, radius) LE radius

; 침식 — 밝은 영역 축소
eroded = ERODE(binary, se_disk)

; 팽창 — 밝은 영역 확대
dilated = DILATE(binary, se_disk)

; 열기 = 침식 후 팽창 (작은 밝은 특징 제거)
opened = MORPH_OPEN(binary, se_disk)

; 닫기 = 팽창 후 침식 (작은 어두운 구멍 채우기)
closed = MORPH_CLOSE(binary, se_disk)
```

---

## 4. 연결 구성요소 레이블링

```idl
labels = LABEL_REGION(binary)
n_regions = MAX(labels)

FOR i = 1, n_regions DO BEGIN
    region_pixels = WHERE(labels EQ i, n_pix)
    xy = ARRAY_INDICES(binary, region_pixels)
    PRINT, '영역 ', i, ': ', n_pix, ' 픽셀'
ENDFOR
```

### 활동 영역 검출 응용

```idl
; 자기도에서 활동 영역 검출
ar_binary = BYTE(ABS(magnetogram) GT 100)
ar_binary = MORPH_CLOSE(MORPH_OPEN(ar_binary, se), se)
labels = LABEL_REGION(ar_binary)
```

---

## 5. 러닝 디퍼런스 이미지

```idl
; 러닝 디퍼런스: diff[t] = image[t] - image[t-1]
diff_cube = cube[*, *, 1:nt-1] - cube[*, *, 0:nt-2]

; 베이스 디퍼런스: diff[t] = image[t] - image[0]
base_diff = cube - REBIN(cube[*, *, 0], nx, ny, nt)

; 백분율 러닝 디퍼런스
; 이미지 전체의 강도 변화를 정규화
```

---

## 6. 특징 추적

```idl
; 밝은 특징의 무게 중심 추적
x_track = FLTARR(nt)
y_track = FLTARR(nt)

FOR t = 1, nt-1 DO BEGIN
    subimg = cube[x0:x1, y0:y1, t]
    total_intensity = TOTAL(subimg)
    IF total_intensity GT 0 THEN BEGIN
        x_track[t] = TOTAL(xx2d * subimg) / total_intensity
        y_track[t] = TOTAL(yy2d * subimg) / total_intensity
    ENDIF
ENDFOR
```

---

## 요약

| 기법 | 핵심 함수 | 용도 |
|------|----------|------|
| 스무딩 | `SMOOTH`, `MEDIAN`, `CONVOL` | 잡음 감소 |
| 에지 검출 | `SOBEL`, `ROBERTS` | 경계 찾기 |
| 형태학 | `ERODE`, `DILATE`, `MORPH_OPEN` | 형태 정리 |
| 레이블링 | `LABEL_REGION` | 연결 구성요소 |
| 러닝 디퍼런스 | 배열 뺄셈 | 시간 변화 |
| 특징 추적 | 무게 중심 계산 | 운동 분석 |

---

**이전**: [스펙트럼 분석](./09_Spectral_Analysis.md) | **다음**: [곡선 피팅](./11_Curve_Fitting.md)
