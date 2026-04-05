# 기본 플로팅

**이전**: [구조체](./09_Structures.md) | **다음**: [이미지 표시](./11_Image_Display.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PLOT 프로시저로 선 그래프 만들기
2. OPLOT으로 데이터 오버레이하기
3. XYOUTS로 텍스트 주석 추가하기
4. AXIS와 플롯 키워드로 축 커스터마이즈하기
5. 선 스타일, 색상, 심볼, 두께 제어하기
6. !P.MULTI 시스템 변수로 다중 패널 플롯 만들기
7. 출판 품질의 PostScript 출력 생성하기

---

IDL의 내장 플로팅 시스템은 가장 큰 강점 중 하나입니다. 몇 가지 명령만으로 과학 논문을 위한 출판 품질 그림을 만들 수 있습니다.

## PLOT 프로시저

```idl
x = FINDGEN(360) * !DTOR
y = SIN(x)
PLOT, x / !DTOR, y, $
  TITLE='Sine Function', $
  XTITLE='Angle (degrees)', $
  YTITLE='sin(x)', $
  CHARSIZE=1.5
```

### 축 범위 제어

```idl
PLOT, x, y, $
  XRANGE=[0, 80], YRANGE=[-1, 1], $
  XSTYLE=1, YSTYLE=1    ; 정확한 범위
```

## 선 스타일, 심볼, 색상

```idl
; LINESTYLE: 0=실선, 1=점선, 2=대시, 3=대시-점, 4=대시-점-점, 5=긴 대시
; PSYM: 1=+, 2=*, 4=다이아몬드, 5=삼각형, 6=사각형, 7=X, 10=히스토그램
; 음수 PSYM: 심볼을 선으로 연결

PLOT, x, SIN(x), LINESTYLE=0, THICK=3
OPLOT, x, COS(x), LINESTYLE=2, THICK=2
```

## OPLOT — 오버플로팅

```idl
PLOT, x, SIN(x), TITLE='Trigonometric Functions', YRANGE=[-1.5, 1.5]
OPLOT, x, COS(x), LINESTYLE=2
OPLOT, x, SIN(x) + COS(x), LINESTYLE=3
```

## XYOUTS — 텍스트 주석

```idl
; 데이터 좌표에 텍스트 추가
XYOUTS, 1.57, 1.0, 'Maximum', CHARSIZE=1.2, ALIGNMENT=0.5

; 정규화 좌표 (0-1 범위)에 텍스트 추가
XYOUTS, 0.15, 0.85, 'y = sin(x)', /NORMAL, CHARSIZE=1.5

; 위첨자와 아래첨자
; !U = 위첨자 시작, !D = 아래첨자 시작, !N = 정상으로 복귀
XYOUTS, 0.5, 0.70, 'x!U2!N + y!U2!N = r!U2!N', /NORMAL
```

## 다중 패널 플롯

```idl
!P.MULTI = [0, 2, 2]    ; [남은 수, 열, 행]
PLOT, x, SIN(x), TITLE='sin(x)'
PLOT, x, COS(x), TITLE='cos(x)'
PLOT, x, TAN(x) < 5 > (-5), TITLE='tan(x)'
PLOT, x, EXP(-x/5.0), TITLE='exp(-x/5)'
!P.MULTI = 0    ; 단일 패널로 리셋
```

## 로그 플롯

```idl
PLOT, x, y, /YLOG, TITLE='Semi-Log Plot'
PLOT, x, y, /XLOG, /YLOG, TITLE='Log-Log Plot'
```

## PostScript 출력

```idl
original_device = !D.NAME
SET_PLOT, 'PS'
DEVICE, FILENAME='figure.eps', /COLOR, /ENCAPSULATED, $
  XSIZE=18, YSIZE=12

PLOT, x, y, TITLE='My Figure', THICK=3, CHARSIZE=1.2

DEVICE, /CLOSE
SET_PLOT, original_device
```

---

## 요약

| 프로시저 | 설명 |
|---------|------|
| `PLOT, x, y` | 새 플롯 생성 |
| `OPLOT, x, y` | 기존 플롯 위에 오버레이 |
| `XYOUTS, x, y, text` | 텍스트 주석 추가 |
| `AXIS` | 축 추가/수정 |
| `ERRPLOT` | 오차 막대 추가 |
| `SET_PLOT, 'PS'` | PostScript 디바이스로 전환 |
| `!P.MULTI` | 다중 패널 레이아웃 |

| 키워드 | 설명 |
|--------|------|
| TITLE, XTITLE, YTITLE | 플롯 및 축 제목 |
| XRANGE, YRANGE | 축 범위 |
| LINESTYLE | 선 패턴 (0-5) |
| PSYM | 플롯 심볼 (1-10) |
| THICK | 선 두께 |
| COLOR | 선/심볼 색상 |
| /YLOG, /XLOG | 로그 축 |

---

**이전**: [구조체](./09_Structures.md) | **다음**: [이미지 표시](./11_Image_Display.md)
