; Exercise 09: Structures
;
; Practice creating and manipulating structures.

; Exercise 1: Create a structure to represent a solar active region
; with fields: id, location (nested struct with lat/lon),
; area, and classification.
PRO exercise_09a
  ; TODO: Create a structure with the fields described above
  ; TODO: Fill in sample data for 3 active regions
  ; TODO: Print all fields using a loop over TAG_NAMES
  ; Hint: Use nested struct: location: {lat: 0.0, lon: 0.0}
END

; Exercise 2: Create a structure array of 10 observations and
; sort them by the 'flux' field in descending order.
PRO exercise_09b
  ; TODO: Define template = {time: 0.0D0, flux: 0.0, quality: 0B}
  ; TODO: Create obs = REPLICATE(template, 10)
  ; TODO: Fill with random data
  ; TODO: Sort by flux descending using REVERSE(SORT(...))
  ; TODO: Print top 3
END

; Exercise 3: Write a function that merges two structures
; dynamically using CREATE_STRUCT.
FUNCTION exercise_09c, s1, s2
  ; TODO: Use CREATE_STRUCT to combine s1 and s2
  ; TODO: Handle the case where field names might conflict
  ; Hint: merged = CREATE_STRUCT(s1, s2)
  RETURN, !NULL
END

; Exercise 4: Create a mini "database" of FITS header keywords
; as a structure array. Implement a search function that finds
; a keyword by name and returns its value.
FUNCTION exercise_09d, db, keyword
  ; TODO: db is a structure array with fields {name: '', value: ''}
  ; TODO: Search for keyword in db.name
  ; TODO: Return the value if found, '' if not found
  ; Hint: Use WHERE(db.name EQ STRUPCASE(keyword), count)
  RETURN, ''
END
