; 09 Structures
; =============
; Demonstrates anonymous/named structures, CREATE_STRUCT,
; structure arrays, and inspection.

PRO example_09_structures

  ; Anonymous structure
  person = {name: 'Alice', age: 30, score: 98.5}
  PRINT, '--- Anonymous Structure ---'
  HELP, person, /STRUCTURE
  PRINT, 'Name:', person.name, '  Age:', person.age

  ; CREATE_STRUCT
  PRINT, '--- Dynamic Construction ---'
  s = CREATE_STRUCT('temp', 25.0, 'pressure', 1013.25, 'humidity', 0.65)
  HELP, s, /STRUCTURE

  ; Structure array
  PRINT, '--- Structure Array ---'
  template = {star_name: '', magnitude: 0.0, distance: 0.0D0}
  stars = REPLICATE(template, 4)
  stars[0] = {star_name: 'Sirius', magnitude: -1.46, distance: 8.6D0}
  stars[1] = {star_name: 'Canopus', magnitude: -0.72, distance: 310.0D0}
  stars[2] = {star_name: 'Vega', magnitude: 0.03, distance: 25.0D0}
  stars[3] = {star_name: 'Capella', magnitude: 0.08, distance: 43.0D0}

  ; Sort by magnitude
  idx = SORT(stars.magnitude)
  FOR i = 0, 3 DO $
    PRINT, FORMAT='("  ", A-10, " mag=", F6.2, "  d=", F7.1, " ly")', $
      stars[idx[i]].star_name, stars[idx[i]].magnitude, stars[idx[i]].distance

  ; Tag access by number
  PRINT, '--- Tag Access ---'
  PRINT, 'Tag names:', TAG_NAMES(person)
  PRINT, 'N_TAGS:', N_TAGS(person)
  FOR i = 0, N_TAGS(person) - 1 DO $
    PRINT, '  ', (TAG_NAMES(person))[i], ':', person.(i)

  ; Nested structure
  obs = {target: {name: 'Sun', type: 'star'}, $
         instrument: 'AIA', wavelength: 171.0}
  PRINT, '--- Nested ---'
  PRINT, 'Target:', obs.target.name
  PRINT, 'Instrument:', obs.instrument

  PRINT, 'Example 09 complete.'
END
