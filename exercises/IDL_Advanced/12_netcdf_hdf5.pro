;+
; Exercise 12: NetCDF and HDF5
;-

PRO exercise_12

    ; === Exercise 1: Create a NetCDF file ===
    ; Write a NetCDF file with:
    ; - Dimensions: x(100), y(100), time(UNLIMITED)
    ; - Variable: temperature(x, y, time) [float]
    ; - Attributes: units='Kelvin', title='Exercise Data'
    ; Write 10 time steps of random data
    ; TODO: NCDF_CREATE, NCDF_DIMDEF, NCDF_VARDEF, NCDF_VARPUT, NCDF_CLOSE

    ; === Exercise 2: Read and analyze NetCDF ===
    ; Read back the file from Exercise 1
    ; Compute and print the temporal mean and standard deviation at pixel (50,50)
    ; TODO: NCDF_OPEN, NCDF_VARGET, compute stats

    ; === Exercise 3: Create an HDF5 file ===
    ; Write an HDF5 file with:
    ; - Group: /solar/aia
    ; - Dataset: /solar/aia/image_171 (256x256 float)
    ; - Attribute on dataset: wavelength='171 Angstrom'
    ; TODO: H5F_CREATE, H5G_CREATE, H5D_CREATE, H5A_CREATE

    ; === Exercise 4: Read HDF5 with H5_PARSE ===
    ; Read the HDF5 file from Exercise 3 using H5_PARSE
    ; Print the structure hierarchy
    ; TODO: H5_PARSE with /READ_DATA, HELP /STRUCTURE

    ; === Exercise 5: Format conversion ===
    ; Read a FITS file (or create synthetic data)
    ; Write the data to both NetCDF and HDF5 formats
    ; Read both back and verify the data matches
    ; TODO: READFITS, NCDF_*, H5*_, compare with TOTAL(ABS(diff))

END
