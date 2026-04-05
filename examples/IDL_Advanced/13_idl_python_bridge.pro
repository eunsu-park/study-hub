;+
; 13_idl_python_bridge.pro — Lesson 13: IDL-Python Bridge
;
; Demonstrates calling Python from IDL using the built-in bridge.
; Requires: IDL 8.5+ with Python bridge configured
;-

PRO python_bridge_demo
    ; Check if Python bridge is available
    CATCH, err
    IF err NE 0 THEN BEGIN
        PRINT, 'Python bridge not available: ', !ERROR_STATE.MSG
        CATCH, /CANCEL
        RETURN
    ENDIF

    ; Import numpy
    np = PYTHON.IMPORT('numpy')

    ; Create and manipulate arrays
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    PRINT, 'numpy mean of sin(x): ', np.mean(y)
    PRINT, 'numpy std of sin(x):  ', np.std(y)

    ; Convert to IDL arrays and plot
    x_idl = FLOAT(x)
    y_idl = FLOAT(y)
    WINDOW, 0, XSIZE=600, YSIZE=400
    PLOT, x_idl, y_idl, TITLE='numpy sin(x) plotted in IDL', $
        XTITLE='X', YTITLE='sin(X)'

    ; Use scipy for integration
    integrate = PYTHON.IMPORT('scipy.integrate')
    result = integrate.quad(np.sin, 0, np.pi)
    PRINT, 'Integral of sin(x) from 0 to pi: ', result

    PRINT, 'Python bridge demo complete'
    CATCH, /CANCEL
END

python_bridge_demo
END
