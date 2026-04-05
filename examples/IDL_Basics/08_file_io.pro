; 08 File I/O
; ==========
; Demonstrates text file reading/writing, binary I/O,
; and SAVE/RESTORE.

PRO example_08_file_io

  ; Write a text file
  outfile = 'example_data.txt'
  OPENW, lun, outfile, /GET_LUN
  PRINTF, lun, '# X    Y    Z'
  FOR i = 0, 9 DO BEGIN
    PRINTF, lun, FORMAT='(F6.2, "  ", F8.4, "  ", F8.4)', $
      FLOAT(i), SIN(FLOAT(i)), COS(FLOAT(i))
  ENDFOR
  FREE_LUN, lun
  PRINT, 'Wrote:', outfile

  ; Read the text file back
  OPENR, lun, outfile, /GET_LUN
  line = ''
  READF, lun, line   ; Skip header
  PRINT, 'Header:', line
  x = 0.0 & y = 0.0 & z = 0.0
  WHILE ~EOF(lun) DO BEGIN
    READF, lun, x, y, z
    PRINT, FORMAT='("  x=", F5.2, " y=", F8.4, " z=", F8.4)', x, y, z
  ENDWHILE
  FREE_LUN, lun

  ; Binary I/O
  data = FINDGEN(100)
  binfile = 'example_data.bin'
  OPENW, lun, binfile, /GET_LUN
  WRITEU, lun, data
  FREE_LUN, lun

  read_data = FLTARR(100)
  OPENR, lun, binfile, /GET_LUN
  READU, lun, read_data
  FREE_LUN, lun
  PRINT, 'Binary read range:', MIN(read_data), MAX(read_data)

  ; SAVE/RESTORE
  x = FINDGEN(50)
  y = SIN(x / 5.0)
  SAVE, x, y, FILENAME='example_save.sav'
  PRINT, 'Saved to example_save.sav'

  ; Clean up temp files
  FILE_DELETE, outfile, binfile, 'example_save.sav', /ALLOW_NONEXISTENT

  PRINT, 'Example 08 complete.'
END
