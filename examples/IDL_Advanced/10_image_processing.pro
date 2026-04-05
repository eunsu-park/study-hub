;+
; 10_image_processing.pro — Lesson 10: Image Processing
;
; Demonstrates edge detection, morphology, and connected component labeling.
;-

PRO image_processing_demo
    img = DIST(256) + RANDOMN(seed, 256, 256) * 15.0

    ; Edge detection
    edges_sobel = SOBEL(img)
    edges_roberts = ROBERTS(img)

    ; Morphological operations on binary image
    binary = BYTARR(256, 256)
    binary[30:80, 40:90] = 1B
    binary[120:180, 100:170] = 1B
    binary[200:220, 30:50] = 1B

    se = SHIFT(DIST(7), 3, 3) LE 3
    opened = MORPH_OPEN(binary, se)
    closed = MORPH_CLOSE(binary, se)

    ; Label connected regions
    labels = LABEL_REGION(binary)
    n_regions = MAX(labels)
    PRINT, 'Detected ', n_regions, ' connected regions'
    FOR i = 1, n_regions DO BEGIN
        pix = WHERE(labels EQ i, np)
        xy = ARRAY_INDICES(binary, pix)
        PRINT, 'Region ', i, ': ', np, ' pixels, center=(', $
            MEAN(xy[0,*]), ',', MEAN(xy[1,*]), ')'
    ENDFOR

    ; Display
    WINDOW, 0, XSIZE=768, YSIZE=512
    !P.MULTI = [0, 3, 2]
    LOADCT, 0
    TV, BYTSCL(CONGRID(img, 256, 256)), 0
    TV, BYTSCL(CONGRID(edges_sobel, 256, 256)), 1
    TV, BYTSCL(CONGRID(edges_roberts, 256, 256)), 2
    TV, CONGRID(binary*255B, 256, 256), 3
    TV, CONGRID(opened*255B, 256, 256), 4
    TV, CONGRID(BYTSCL(labels), 256, 256), 5
    !P.MULTI = 0
END

image_processing_demo
END
