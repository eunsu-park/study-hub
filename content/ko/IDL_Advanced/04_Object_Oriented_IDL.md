# 04. 객체지향 IDL

**이전**: [지도 투영법](./03_Map_Projections.md) | **다음**: [SolarSoft 프레임워크](./05_SolarSoft_Framework.md)

---

## 학습 목표

1. `__DEFINE` 프로시저로 IDL 클래스를 정의한다
2. INIT, CLEANUP, GetProperty, SetProperty 메서드를 구현한다
3. 상속과 메서드 해석을 사용한다
4. IDL 위젯 프로그래밍으로 대화형 GUI를 구축한다
5. XMANAGER와 이벤트 콜백으로 위젯 이벤트를 처리한다

---

## 1. IDL 객체 시스템 개요

| OOP 개념 | IDL 구현 |
|----------|---------|
| 클래스 정의 | `PRO classname__DEFINE` |
| 생성자 | `FUNCTION classname::INIT` |
| 소멸자 | `PRO classname::CLEANUP` |
| 메서드 호출 | `object->MethodName()` |
| 인스턴스화 | `obj = OBJ_NEW('classname', args...)` |
| 소멸 | `OBJ_DESTROY, obj` |
| 상속 | 구조체 정의에서 `INHERITS parent_class` |

---

## 2. 클래스 정의

```idl
; 파일: star__define.pro
PRO star__DEFINE
    void = {star, $
        name: '', $
        ra: 0.0D, $
        dec: 0.0D, $
        magnitude: 0.0, $
        spectral_type: '' $
    }
END

FUNCTION star::INIT, name, ra, dec, magnitude=magnitude, spectral_type=spectral_type
    self.name = name
    self.ra = ra
    self.dec = dec
    self.magnitude = N_ELEMENTS(magnitude) GT 0 ? magnitude : 99.0
    self.spectral_type = N_ELEMENTS(spectral_type) GT 0 ? spectral_type : 'Unknown'
    RETURN, 1  ; 성공 시 1, 실패 시 0 반환
END

PRO star::CLEANUP
    PRINT, '별 소멸: ', self.name
END
```

---

## 3. 프로퍼티 접근

```idl
PRO star::GetProperty, name=name, ra=ra, dec=dec, magnitude=magnitude
    IF ARG_PRESENT(name) THEN name = self.name
    IF ARG_PRESENT(ra) THEN ra = self.ra
    IF ARG_PRESENT(dec) THEN dec = self.dec
    IF ARG_PRESENT(magnitude) THEN magnitude = self.magnitude
END

PRO star::SetProperty, magnitude=magnitude, spectral_type=spectral_type
    IF N_ELEMENTS(magnitude) GT 0 THEN self.magnitude = magnitude
    IF N_ELEMENTS(spectral_type) GT 0 THEN self.spectral_type = spectral_type
END
```

---

## 4. 상속

```idl
PRO binary_star__DEFINE
    void = {binary_star, $
        INHERITS star, $
        companion_name: '', $
        period_years: 0.0D $
    }
END

FUNCTION binary_star::INIT, name, ra, dec, _EXTRA=extra
    IF ~self->star::INIT(name, ra, dec, _EXTRA=extra) THEN RETURN, 0
    RETURN, 1
END

; ISA 확인
PRINT, OBJ_ISA(algol, 'binary_star')  ; 1
PRINT, OBJ_ISA(algol, 'star')         ; 1 (상속)
```

---

## 5. 위젯 프로그래밍

```idl
PRO simple_gui_event, event
    WIDGET_CONTROL, event.id, GET_UVALUE=uvalue
    CASE uvalue OF
        'PLOT': BEGIN
            WIDGET_CONTROL, event.top, GET_UVALUE=state
            WSET, state.draw_id
            x = FINDGEN(100) * 0.1
            PLOT, x, SIN(x + RANDOMU(seed) * !PI)
        END
        'QUIT': WIDGET_CONTROL, event.top, /DESTROY
        ELSE:
    ENDCASE
END

PRO simple_gui
    base = WIDGET_BASE(TITLE='간단한 GUI', /COLUMN)
    draw = WIDGET_DRAW(base, XSIZE=480, YSIZE=350)
    button_base = WIDGET_BASE(base, /ROW)
    WIDGET_BUTTON(button_base, VALUE='Plot', UVALUE='PLOT')
    WIDGET_BUTTON(button_base, VALUE='Quit', UVALUE='QUIT')

    WIDGET_CONTROL, base, /REALIZE
    WIDGET_CONTROL, draw, GET_VALUE=draw_id
    state = {draw_id: draw_id}
    WIDGET_CONTROL, base, SET_UVALUE=state
    XMANAGER, 'simple_gui', base, /NO_BLOCK
END
```

---

## 6. 객체 프로퍼티에서의 포인터 사용

```idl
PRO timeseries__DEFINE
    void = {timeseries, $
        name: '', $
        time: PTR_NEW(), $
        data: PTR_NEW(), $
        npoints: 0L $
    }
END

PRO timeseries::CLEANUP
    ; 메모리 누수 방지를 위해 반드시 포인터 해제
    PTR_FREE, self.time
    PTR_FREE, self.data
END
```

---

## 요약

| 주제 | 핵심 구성 요소 | 용도 |
|------|---------------|------|
| 클래스 정의 | `classname__DEFINE` | 구조체/필드 정의 |
| 생성자 | `classname::INIT` | 객체 초기화 |
| 소멸자 | `classname::CLEANUP` | 리소스 해제 |
| 상속 | `INHERITS parent` | 코드 재사용 |
| 위젯 | `WIDGET_BASE`, `WIDGET_DRAW` | GUI 구축 |
| 이벤트 | `XMANAGER`, 이벤트 프로시저 | 상호작용 |

---

**이전**: [지도 투영법](./03_Map_Projections.md) | **다음**: [SolarSoft 프레임워크](./05_SolarSoft_Framework.md)
