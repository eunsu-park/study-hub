# 04. Object-Oriented IDL

**Previous**: [Map Projections](./03_Map_Projections.md) | **Next**: [SolarSoft Framework](./05_SolarSoft_Framework.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define IDL classes with `__DEFINE` procedures
2. Implement INIT, CLEANUP, GetProperty, and SetProperty methods
3. Use inheritance and method resolution in IDL's object system
4. Build interactive GUIs with IDL widget programming
5. Handle widget events with XMANAGER and event callbacks

---

## 1. IDL Object System Overview

IDL supports object-oriented programming (OOP) with named structures as classes and procedures/functions as methods.

### Key Concepts

| OOP Concept | IDL Implementation |
|------------|-------------------|
| Class definition | `PRO classname__DEFINE` |
| Constructor | `FUNCTION classname::INIT` |
| Destructor | `PRO classname::CLEANUP` |
| Method call | `object->MethodName()` or `object.MethodName()` |
| Instantiation | `obj = OBJ_NEW('classname', args...)` |
| Destruction | `OBJ_DESTROY, obj` |
| Inheritance | `INHERITS parent_class` in structure definition |
| Property access | `GetProperty` / `SetProperty` methods |

---

## 2. Defining a Class

### The __DEFINE Procedure

Every IDL class requires a `classname__DEFINE` procedure that creates a named structure:

```idl
; File: star__define.pro
; Convention: one class per file, filename matches classname

PRO star__DEFINE
    ; Define the class structure
    void = {star, $
        name: '', $              ; Star name
        ra: 0.0D, $             ; Right ascension (degrees)
        dec: 0.0D, $            ; Declination (degrees)
        magnitude: 0.0, $       ; Apparent magnitude
        spectral_type: '', $    ; Spectral classification
        distance_pc: 0.0D $     ; Distance in parsecs
    }
END
```

### The INIT Method (Constructor)

```idl
FUNCTION star::INIT, name, ra, dec, magnitude=magnitude, $
    spectral_type=spectral_type, distance_pc=distance_pc

    ; Set required properties
    self.name = name
    self.ra = ra
    self.dec = dec

    ; Set optional properties with defaults
    self.magnitude = N_ELEMENTS(magnitude) GT 0 ? magnitude : 99.0
    self.spectral_type = N_ELEMENTS(spectral_type) GT 0 ? spectral_type : 'Unknown'
    self.distance_pc = N_ELEMENTS(distance_pc) GT 0 ? distance_pc : 0.0D

    ; INIT must return 1 for success, 0 for failure
    RETURN, 1
END
```

### The CLEANUP Method (Destructor)

```idl
PRO star::CLEANUP
    ; Free any heap variables (pointers, objects) owned by this object
    ; Simple value types (string, float) are freed automatically
    PRINT, 'Destroying star: ', self.name
END
```

### Creating and Destroying Objects

```idl
; Create an object
sun = OBJ_NEW('star', 'Sun', 0.0D, 0.0D, $
    MAGNITUDE=-26.74, SPECTRAL_TYPE='G2V', DISTANCE_PC=4.848e-6)

sirius = OBJ_NEW('star', 'Sirius', 101.287D, -16.716D, $
    MAGNITUDE=-1.46, SPECTRAL_TYPE='A1V', DISTANCE_PC=2.637)

; Check if valid
PRINT, OBJ_VALID(sun)    ; 1
PRINT, OBJ_ISA(sun, 'star')  ; 1

; Destroy
OBJ_DESTROY, sun
PRINT, OBJ_VALID(sun)    ; 0
```

---

## 3. Properties: GetProperty and SetProperty

```idl
PRO star::GetProperty, name=name, ra=ra, dec=dec, $
    magnitude=magnitude, spectral_type=spectral_type, $
    distance_pc=distance_pc

    IF ARG_PRESENT(name) THEN name = self.name
    IF ARG_PRESENT(ra) THEN ra = self.ra
    IF ARG_PRESENT(dec) THEN dec = self.dec
    IF ARG_PRESENT(magnitude) THEN magnitude = self.magnitude
    IF ARG_PRESENT(spectral_type) THEN spectral_type = self.spectral_type
    IF ARG_PRESENT(distance_pc) THEN distance_pc = self.distance_pc
END

PRO star::SetProperty, name=name, magnitude=magnitude, $
    spectral_type=spectral_type, distance_pc=distance_pc

    ; RA and Dec are read-only (not included here)
    IF N_ELEMENTS(name) GT 0 THEN self.name = name
    IF N_ELEMENTS(magnitude) GT 0 THEN self.magnitude = magnitude
    IF N_ELEMENTS(spectral_type) GT 0 THEN self.spectral_type = spectral_type
    IF N_ELEMENTS(distance_pc) GT 0 THEN self.distance_pc = distance_pc
END
```

### Using Properties

```idl
star = OBJ_NEW('star', 'Vega', 279.234D, 38.783D, $
    MAGNITUDE=0.03, SPECTRAL_TYPE='A0V')

; Get properties
star->GetProperty, NAME=n, MAGNITUDE=m
PRINT, n, m   ; Vega   0.0300000

; Set properties
star->SetProperty, DISTANCE_PC=7.68D
star->GetProperty, DISTANCE_PC=d
PRINT, 'Distance: ', d, ' pc'

; Clean up
OBJ_DESTROY, star
```

---

## 4. Custom Methods

```idl
; Calculate absolute magnitude
FUNCTION star::AbsoluteMagnitude
    IF self.distance_pc LE 0 THEN RETURN, !VALUES.F_NAN
    RETURN, self.magnitude - 5.0 * ALOG10(self.distance_pc) + 5.0
END

; Calculate luminosity relative to Sun
FUNCTION star::Luminosity
    abs_mag = self->AbsoluteMagnitude()
    IF ~FINITE(abs_mag) THEN RETURN, !VALUES.F_NAN
    ; L/L_sun = 10^((M_sun - M_star) / 2.5)
    RETURN, 10.0^((4.83 - abs_mag) / 2.5)
END

; Print summary
PRO star::Print
    PRINT, '========================='
    PRINT, 'Star: ', self.name
    PRINT, 'RA:   ', self.ra, ' deg'
    PRINT, 'Dec:  ', self.dec, ' deg'
    PRINT, 'Mag:  ', self.magnitude
    PRINT, 'Type: ', self.spectral_type
    PRINT, 'Dist: ', self.distance_pc, ' pc'
    abs_mag = self->AbsoluteMagnitude()
    IF FINITE(abs_mag) THEN $
        PRINT, 'M_abs:', abs_mag
    PRINT, '========================='
END
```

---

## 5. Inheritance

### Defining a Child Class

```idl
; File: binary_star__define.pro
PRO binary_star__DEFINE
    void = {binary_star, $
        INHERITS star, $         ; Inherit all star properties
        companion_name: '', $    ; Companion star name
        separation_arcsec: 0.0, $ ; Angular separation
        period_years: 0.0D $     ; Orbital period
    }
END

FUNCTION binary_star::INIT, name, ra, dec, $
    companion_name=companion_name, $
    separation_arcsec=separation_arcsec, $
    period_years=period_years, $
    _EXTRA=extra

    ; Call parent INIT
    IF ~self->star::INIT(name, ra, dec, _EXTRA=extra) THEN RETURN, 0

    ; Set child-specific properties
    IF N_ELEMENTS(companion_name) GT 0 THEN $
        self.companion_name = companion_name
    IF N_ELEMENTS(separation_arcsec) GT 0 THEN $
        self.separation_arcsec = separation_arcsec
    IF N_ELEMENTS(period_years) GT 0 THEN $
        self.period_years = period_years

    RETURN, 1
END

PRO binary_star::CLEANUP
    ; Call parent cleanup
    self->star::CLEANUP
END

PRO binary_star::Print
    ; Override parent Print method
    self->star::Print  ; Call parent method first
    PRINT, 'Companion:  ', self.companion_name
    PRINT, 'Separation: ', self.separation_arcsec, ' arcsec'
    PRINT, 'Period:     ', self.period_years, ' years'
END
```

### Using Inheritance

```idl
; Create a binary star
algol = OBJ_NEW('binary_star', 'Algol A', 47.042D, 40.956D, $
    MAGNITUDE=2.12, SPECTRAL_TYPE='B8V', DISTANCE_PC=28.5D, $
    COMPANION_NAME='Algol B', SEPARATION_ARCSEC=0.0, $
    PERIOD_YEARS=2.867D/365.25)

; ISA checks
PRINT, OBJ_ISA(algol, 'binary_star')  ; 1
PRINT, OBJ_ISA(algol, 'star')         ; 1 (inheritance)

; Call inherited method
abs_mag = algol->AbsoluteMagnitude()

; Call overridden method
algol->Print

OBJ_DESTROY, algol
```

---

## 6. IDL_Object Base Class

IDL 8.0+ provides `IDL_Object` as a built-in base class with operator overloading:

```idl
PRO modern_star__DEFINE
    void = {modern_star, $
        INHERITS IDL_Object, $   ; Enables operator overloading
        name: '', $
        magnitude: 0.0 $
    }
END

; Overload the print operator
FUNCTION modern_star::_overloadPrint
    RETURN, STRING(self.name, self.magnitude, $
        FORMAT='("Star: ", A-15, " Mag: ", F6.2)')
END

; Overload comparison operator
FUNCTION modern_star::_overloadLT, left, right
    ; Compare by magnitude (brighter = smaller magnitude)
    left->GetProperty, MAGNITUDE=m1
    right->GetProperty, MAGNITUDE=m2
    RETURN, m1 LT m2
END

; Usage:
star = OBJ_NEW('modern_star')
PRINT, star  ; Calls _overloadPrint
```

---

## 7. Widget Programming Basics

IDL widgets provide a toolkit for building interactive GUIs.

### Widget Hierarchy

```
WIDGET_BASE (top-level container)
├── WIDGET_BUTTON (clickable button)
├── WIDGET_TEXT (text input/display)
├── WIDGET_LABEL (static text)
├── WIDGET_LIST (selection list)
├── WIDGET_SLIDER (numeric slider)
├── WIDGET_DROPLIST (dropdown menu)
├── WIDGET_DRAW (drawing canvas)
└── WIDGET_BASE (nested container)
    ├── WIDGET_BUTTON
    └── WIDGET_BUTTON
```

### Simple Widget Application

```idl
PRO simple_gui_event, event
    ; Event handler: called whenever a widget event occurs
    WIDGET_CONTROL, event.id, GET_UVALUE=uvalue

    CASE uvalue OF
        'PLOT': BEGIN
            ; Get the draw widget window
            WIDGET_CONTROL, event.top, GET_UVALUE=state
            WSET, state.draw_id

            ; Generate and plot data
            x = FINDGEN(100) * 0.1
            y = SIN(x + RANDOMU(seed) * !PI)
            PLOT, x, y, TITLE='Random Phase Sine Wave'
        END

        'CLEAR': BEGIN
            WIDGET_CONTROL, event.top, GET_UVALUE=state
            WSET, state.draw_id
            ERASE
        END

        'QUIT': BEGIN
            WIDGET_CONTROL, event.top, /DESTROY
        END

        ELSE:
    ENDCASE
END

PRO simple_gui
    ; Create the widget hierarchy
    base = WIDGET_BASE(TITLE='Simple GUI', /COLUMN, $
        XSIZE=500, YSIZE=450)

    ; Drawing area
    draw = WIDGET_DRAW(base, XSIZE=480, YSIZE=350)

    ; Button row
    button_base = WIDGET_BASE(base, /ROW)
    btn_plot = WIDGET_BUTTON(button_base, VALUE='Plot', UVALUE='PLOT')
    btn_clear = WIDGET_BUTTON(button_base, VALUE='Clear', UVALUE='CLEAR')
    btn_quit = WIDGET_BUTTON(button_base, VALUE='Quit', UVALUE='QUIT')

    ; Realize the widget (display it)
    WIDGET_CONTROL, base, /REALIZE

    ; Get the draw window ID
    WIDGET_CONTROL, draw, GET_VALUE=draw_id

    ; Store state information
    state = {draw_id: draw_id}
    WIDGET_CONTROL, base, SET_UVALUE=state

    ; Start the event loop
    XMANAGER, 'simple_gui', base, /NO_BLOCK
END
```

### Widget with Text Input

```idl
PRO text_gui_event, event
    WIDGET_CONTROL, event.id, GET_UVALUE=uvalue
    WIDGET_CONTROL, event.top, GET_UVALUE=state

    CASE uvalue OF
        'CALC': BEGIN
            ; Read text field
            WIDGET_CONTROL, state.text_id, GET_VALUE=text
            value = FLOAT(text[0])

            ; Compute and display result
            result = STRING(value^2, FORMAT='("Square: ", F10.3)')
            WIDGET_CONTROL, state.result_id, SET_VALUE=result
        END
        ELSE:
    ENDCASE
END

PRO text_gui
    base = WIDGET_BASE(TITLE='Calculator', /COLUMN)

    WIDGET_LABEL(base, VALUE='Enter a number:')
    text = WIDGET_TEXT(base, /EDITABLE, XSIZE=20, UVALUE='INPUT')
    btn = WIDGET_BUTTON(base, VALUE='Square it', UVALUE='CALC')
    result = WIDGET_LABEL(base, VALUE='Result: ', XSIZE=200)

    WIDGET_CONTROL, base, /REALIZE
    state = {text_id: text, result_id: result}
    WIDGET_CONTROL, base, SET_UVALUE=state
    XMANAGER, 'text_gui', base, /NO_BLOCK
END
```

---

## 8. Interactive Image Viewer Widget

```idl
PRO image_viewer_event, event
    WIDGET_CONTROL, event.id, GET_UVALUE=uvalue
    WIDGET_CONTROL, event.top, GET_UVALUE=state

    CASE uvalue OF
        'DRAW': BEGIN
            ; Mouse event on draw widget
            WSET, state.draw_id
            IF event.type EQ 0 THEN BEGIN  ; Button press
                ; Display cursor position in arcsec
                x_arc = (event.x - state.nx/2) * state.cdelt
                y_arc = (event.y - state.ny/2) * state.cdelt
                status = STRING(x_arc, y_arc, $
                    FORMAT='("X: ", F8.1, " arcsec  Y: ", F8.1, " arcsec")')
                WIDGET_CONTROL, state.status_id, SET_VALUE=status
            ENDIF
        END

        'LOADCT': BEGIN
            XLOADCT  ; Interactive color table selector
            WSET, state.draw_id
            TV, BYTSCL(state.image)
        END

        'QUIT': WIDGET_CONTROL, event.top, /DESTROY
        ELSE:
    ENDCASE
END

PRO image_viewer, image, cdelt=cdelt
    IF N_ELEMENTS(cdelt) EQ 0 THEN cdelt = 0.6  ; arcsec/pixel (AIA)

    sz = SIZE(image, /DIMENSIONS)
    nx = sz[0] & ny = sz[1]

    base = WIDGET_BASE(TITLE='Solar Image Viewer', /COLUMN)
    draw = WIDGET_DRAW(base, XSIZE=nx, YSIZE=ny, $
        /BUTTON_EVENTS, /MOTION_EVENTS, UVALUE='DRAW')

    ctrl_base = WIDGET_BASE(base, /ROW)
    WIDGET_BUTTON(ctrl_base, VALUE='Color Table', UVALUE='LOADCT')
    WIDGET_BUTTON(ctrl_base, VALUE='Quit', UVALUE='QUIT')

    status = WIDGET_LABEL(base, VALUE='Click on image for coordinates', $
        XSIZE=400)

    WIDGET_CONTROL, base, /REALIZE
    WIDGET_CONTROL, draw, GET_VALUE=draw_id

    state = {draw_id: draw_id, image: image, $
             nx: nx, ny: ny, cdelt: cdelt, status_id: status}
    WIDGET_CONTROL, base, SET_UVALUE=state

    WSET, draw_id
    TV, BYTSCL(image)

    XMANAGER, 'image_viewer', base, /NO_BLOCK
END
```

---

## 9. Pointers in Object Properties

For dynamic-size data (arrays of unknown size), use pointers:

```idl
PRO timeseries__DEFINE
    void = {timeseries, $
        name: '', $
        time: PTR_NEW(), $      ; Pointer to time array
        data: PTR_NEW(), $      ; Pointer to data array
        npoints: 0L $
    }
END

FUNCTION timeseries::INIT, name, time, data
    self.name = name
    self.time = PTR_NEW(time)
    self.data = PTR_NEW(data)
    self.npoints = N_ELEMENTS(time)
    RETURN, 1
END

PRO timeseries::CLEANUP
    ; MUST free pointers to avoid memory leaks
    PTR_FREE, self.time
    PTR_FREE, self.data
END

FUNCTION timeseries::GetTime
    RETURN, *self.time
END

FUNCTION timeseries::GetData
    RETURN, *self.data
END

PRO timeseries::Plot, _EXTRA=extra
    PLOT, *self.time, *self.data, $
        TITLE=self.name, _EXTRA=extra
END

; Usage:
t = FINDGEN(1000) * 0.01
d = SIN(2*!PI*t*5.0) + RANDOMN(seed, 1000) * 0.2
ts = OBJ_NEW('timeseries', 'Channel A', t, d)
ts->Plot, XTITLE='Time (s)', YTITLE='Signal'
OBJ_DESTROY, ts  ; Frees pointers in CLEANUP
```

---

## Summary

| Topic | Key Constructs | Purpose |
|-------|---------------|---------|
| Class definition | `classname__DEFINE` | Define structure/fields |
| Constructor | `classname::INIT` | Initialize object, return 1/0 |
| Destructor | `classname::CLEANUP` | Free resources |
| Properties | `GetProperty`, `SetProperty` | Encapsulated access |
| Inheritance | `INHERITS parent` | Code reuse |
| IDL_Object | `_overloadPrint`, `_overloadLT` | Operator overloading |
| Widgets | `WIDGET_BASE`, `WIDGET_DRAW` | GUI construction |
| Events | `XMANAGER`, event procedures | Interactivity |
| Pointers | `PTR_NEW`, `PTR_FREE` | Dynamic data in objects |

---

**Previous**: [Map Projections](./03_Map_Projections.md) | **Next**: [SolarSoft Framework](./05_SolarSoft_Framework.md)
