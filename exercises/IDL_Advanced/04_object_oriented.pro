;+
; Exercise 04: Object-Oriented IDL
;-

PRO exercise_04

    ; === Exercise 1: Define a Planet class ===
    ; Create planet__define.pro with fields: name, mass_kg, radius_km, n_moons
    ; Implement INIT, CLEANUP, GetProperty, SetProperty
    ; TODO: Write the class definition
    ; TODO: Create Earth and Mars objects, print their properties

    ; === Exercise 2: Add methods ===
    ; Add a SurfaceGravity method: g = G*M/R^2
    ; Add a Density method: rho = M / (4/3 * pi * R^3)
    ; TODO: Implement and test with Earth (g~9.8 m/s^2, rho~5500 kg/m^3)

    ; === Exercise 3: Inheritance ===
    ; Create a gas_giant class that inherits from planet
    ; Add fields: ring_system (boolean), magnetic_moment (A*m^2)
    ; TODO: Create Jupiter and Saturn objects

    ; === Exercise 4: Simple calculator widget ===
    ; Create a widget with:
    ; - Two text input fields
    ; - Buttons for +, -, *, /
    ; - A label showing the result
    ; TODO: Use WIDGET_BASE, WIDGET_TEXT, WIDGET_BUTTON, WIDGET_LABEL
    ; TODO: Implement event handler
    ; Hint: WIDGET_CONTROL, text_id, GET_VALUE=text

    ; === Exercise 5: Pointers for dynamic data ===
    ; Create a spectrum class with pointer-based wavelength and flux arrays
    ; Implement a Plot method and proper CLEANUP
    ; TODO: Use PTR_NEW in INIT, PTR_FREE in CLEANUP

END
