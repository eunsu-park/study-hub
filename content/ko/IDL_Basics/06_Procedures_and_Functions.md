# 프로시저와 함수

**이전**: [제어 흐름](./05_Control_Flow.md) | **다음**: [문자열 처리](./07_String_Processing.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PRO 키워드로 프로시저 정의하고 호출하기
2. FUNCTION 키워드로 함수 정의하고 호출하기
3. 위치 매개변수와 키워드 매개변수 사용하기
4. KEYWORD_SET, N_PARAMS, N_ELEMENTS로 매개변수 확인하기
5. _EXTRA와 _REF_EXTRA로 추가 키워드 전달하기
6. .COMPILE과 .RUN으로 프로그램 컴파일하기
7. 변수 스코프와 COMMON 블록 이해하기
8. RESOLVE_ALL로 종속성 확인하기

---

프로시저와 함수는 IDL 코드를 구성하는 기본 빌딩 블록입니다. 프로시저는 동작을 수행하고, 함수는 값을 계산하여 반환합니다.

## 프로시저

```idl
; greet.pro
PRO greet, name, title
  IF N_PARAMS() EQ 0 THEN name = 'World'
  IF N_PARAMS() LT 2 THEN title = ''
  IF STRLEN(title) GT 0 THEN $
    PRINT, 'Hello, ' + title + ' ' + name + '!' $
  ELSE PRINT, 'Hello, ' + name + '!'
END
```

### 키워드 매개변수가 있는 프로시저

```idl
PRO print_stats, data, VERBOSE=verbose, TITLE=title
  IF ~KEYWORD_SET(title) THEN title = 'Statistics'
  PRINT, '=== ' + title + ' ==='
  PRINT, FORMAT='("  Mean: ", G12.5)', MEAN(data)
  PRINT, FORMAT='("  Std:  ", G12.5)', STDDEV(data)
  IF KEYWORD_SET(verbose) THEN $
    PRINT, FORMAT='("  N:    ", I0)', N_ELEMENTS(data)
END
```

## 함수

```idl
FUNCTION circle_area, radius
  RETURN, !PI * radius^2
END
```

```idl
IDL> area = circle_area(5.0)
IDL> PRINT, area
      78.5398
```

## 매개변수 처리

### N_PARAMS

`N_PARAMS()`는 루틴에 전달된 위치 매개변수 수를 반환합니다.

### KEYWORD_SET vs N_ELEMENTS

```idl
PRO example, data, VERBOSE=verbose, COUNT=count
  ; KEYWORD_SET: 키워드가 설정되고 0이 아니면 1 반환
  IF KEYWORD_SET(verbose) THEN PRINT, 'Verbose mode on'

  ; N_ELEMENTS: 키워드가 전달되었으면 (값이 0이더라도) GT 0
  ; 숫자 키워드에 선호됨
  IF N_ELEMENTS(count) GT 0 THEN $
    PRINT, 'Count:', count $
  ELSE count = 10  ; 기본값
END
```

**중요한 구분**:
- `KEYWORD_SET(kw)` — 키워드가 설정되고 0이 아니면 참. 유효한 값 0에 대해 실패.
- `N_ELEMENTS(kw) GT 0` — 키워드가 전달되었으면 참 (값이 0이더라도). 숫자 키워드에 선호.

## 키워드 상속: _EXTRA

```idl
PRO my_plot, x, y, TITLE=title, _EXTRA=extra
  IF ~KEYWORD_SET(title) THEN title = 'My Plot'
  PLOT, x, y, TITLE=title, THICK=2, _EXTRA=extra
END
```

`_EXTRA`는 인식되지 않는 키워드를 구조체로 수집하여 호출된 루틴에 전달합니다.

## 컴파일과 실행

```idl
IDL> .COMPILE my_routine        ; 컴파일만
IDL> .RUN my_script             ; 컴파일 후 메인 레벨 프로그램 실행
IDL> RESOLVE_ALL                ; 모든 미해결 종속성 컴파일
```

## 변수 스코프

프로시저나 함수 내에서 정의된 변수는 기본적으로 로컬입니다. 매개변수는 참조로 전달되므로 루틴 내부의 수정이 호출자에 영향을 줍니다.

### COMMON 블록

```idl
PRO init_config
  COMMON config_block, data_dir, verbose_flag
  data_dir = '/data/solar/'
  verbose_flag = 1
END

PRO process_data
  COMMON config_block, data_dir, verbose_flag
  IF verbose_flag THEN PRINT, 'Data directory:', data_dir
END
```

**주의**: COMMON 블록은 루틴 간 숨겨진 결합을 만듭니다. 현대 IDL 코드는 매개변수나 구조체를 통해 데이터를 전달하는 것을 선호합니다.

---

## 요약

| 개념 | 설명 |
|------|------|
| `PRO name` | 프로시저 정의 (반환값 없음) |
| `FUNCTION name` | 함수 정의 (RETURN으로 값 반환) |
| 위치 매개변수 | 호출 시 위치로 전달 |
| 키워드 매개변수 | `KEY=value` 구문으로 명명 |
| `N_PARAMS()` | 전달된 위치 매개변수 수 |
| `KEYWORD_SET(kw)` | 키워드가 설정되고 0이 아니면 참 |
| `_EXTRA` | 인식되지 않는 키워드 수집/전달 |
| `.COMPILE` | 실행 없이 파일 컴파일 |
| `RESOLVE_ALL` | 모든 미해결 종속성 컴파일 |
| COMMON | 공유 변수 블록 (절제하여 사용) |

---

**이전**: [제어 흐름](./05_Control_Flow.md) | **다음**: [문자열 처리](./07_String_Processing.md)
