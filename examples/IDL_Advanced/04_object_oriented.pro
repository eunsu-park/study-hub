;+
; 04_object_oriented.pro — Lesson 04: Object-Oriented IDL
;
; Demonstrates:
;   - Class definition with __DEFINE
;   - INIT, CLEANUP, GetProperty methods
;   - Inheritance
;   - Simple widget GUI
;-

PRO star__DEFINE
    void = {star, name: '', ra: 0.0D, dec: 0.0D, magnitude: 0.0}
END

FUNCTION star::INIT, name, ra, dec, magnitude=magnitude
    self.name = name
    self.ra = ra
    self.dec = dec
    self.magnitude = N_ELEMENTS(magnitude) GT 0 ? magnitude : 99.0
    RETURN, 1
END

PRO star::CLEANUP
    PRINT, 'Destroying star: ', self.name
END

PRO star::GetProperty, name=name, magnitude=magnitude
    IF ARG_PRESENT(name) THEN name = self.name
    IF ARG_PRESENT(magnitude) THEN magnitude = self.magnitude
END

FUNCTION star::AbsoluteMagnitude, distance_pc
    IF distance_pc LE 0 THEN RETURN, !VALUES.F_NAN
    RETURN, self.magnitude - 5.0 * ALOG10(distance_pc) + 5.0
END

PRO oop_demo
    sun = OBJ_NEW('star', 'Sun', 0.0D, 0.0D, MAGNITUDE=-26.74)
    sirius = OBJ_NEW('star', 'Sirius', 101.287D, -16.716D, MAGNITUDE=-1.46)

    sun->GetProperty, NAME=n1, MAGNITUDE=m1
    sirius->GetProperty, NAME=n2, MAGNITUDE=m2
    PRINT, n1, ' apparent mag: ', m1
    PRINT, n2, ' apparent mag: ', m2
    PRINT, 'Sirius abs mag (d=2.64pc): ', sirius->AbsoluteMagnitude(2.64)

    OBJ_DESTROY, sun
    OBJ_DESTROY, sirius
END

oop_demo
END
