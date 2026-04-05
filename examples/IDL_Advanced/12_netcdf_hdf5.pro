;+
; 12_netcdf_hdf5.pro — Lesson 12: NetCDF and HDF5
;
; Demonstrates creating and reading NetCDF and HDF5 files.
;-

PRO netcdf_hdf5_demo
    ; --- NetCDF write and read ---
    PRINT, '--- NetCDF ---'
    ncfile = 'test_output.nc'
    ncid = NCDF_CREATE(ncfile, /CLOBBER)
    xdim = NCDF_DIMDEF(ncid, 'x', 50)
    ydim = NCDF_DIMDEF(ncid, 'y', 50)
    varid = NCDF_VARDEF(ncid, 'temperature', [xdim, ydim], /FLOAT)
    NCDF_ATTPUT, ncid, varid, 'units', 'Kelvin'
    NCDF_ATTPUT, ncid, /GLOBAL, 'title', 'Test NetCDF File'
    NCDF_CONTROL, ncid, /ENDEF

    temp = RANDOMU(seed, 50, 50) * 100.0 + 200.0
    NCDF_VARPUT, ncid, varid, temp
    NCDF_CLOSE, ncid
    PRINT, 'Wrote: ', ncfile

    ; Read back
    ncid = NCDF_OPEN(ncfile)
    vid = NCDF_VARID(ncid, 'temperature')
    NCDF_VARGET, ncid, vid, temp_read
    NCDF_CLOSE, ncid
    PRINT, 'Read back — Mean temp: ', MEAN(temp_read), ' K'

    ; --- HDF5 write and read ---
    PRINT, '--- HDF5 ---'
    h5file = 'test_output.h5'
    fid = H5F_CREATE(h5file)
    gid = H5G_CREATE(fid, 'data')
    dims = [100, 100]
    sid = H5S_CREATE_SIMPLE(dims)
    tid = H5T_IDL_CREATE(0.0)
    did = H5D_CREATE(gid, 'image', tid, sid)
    H5D_WRITE, did, RANDOMU(seed, 100, 100)
    H5D_CLOSE, did & H5T_CLOSE, tid & H5S_CLOSE, sid
    H5G_CLOSE, gid & H5F_CLOSE, fid
    PRINT, 'Wrote: ', h5file

    ; Read back
    fid = H5F_OPEN(h5file)
    did = H5D_OPEN(fid, '/data/image')
    img = H5D_READ(did)
    H5D_CLOSE, did & H5F_CLOSE, fid
    PRINT, 'Read back — Image size: ', SIZE(img, /DIMENSIONS)

    ; Cleanup
    FILE_DELETE, ncfile, h5file, /QUIET
END

netcdf_hdf5_demo
END
