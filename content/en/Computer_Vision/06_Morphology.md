# Morphological Operations

## Overview

Morphological Operations are operations based on the shape of binary or grayscale images. They are mainly used for noise removal, object separation, hole filling, and more. This document covers everything from the concept of structuring elements to various applications of morphological operations.

**Difficulty**: ⭐⭐ (Beginner-Intermediate)

## Learning Objectives

After completing this lesson, you will be able to:

1. Understanding Structuring Elements
2. Erosion and Dilation operations
3. Opening and Closing operations
4. Gradient, Top-hat, and Black-hat operations
5. Noise removal and object separation applications

---

## Table of Contents

1. [Morphological Operations Overview](#1-morphological-operations-overview)
2. [Structuring Element - getStructuringElement()](#2-structuring-element---getstructuringelement)
3. [Erosion - erode()](#3-erosion---erode)
4. [Dilation - dilate()](#4-dilation---dilate)
5. [Opening and Closing - morphologyEx()](#5-opening-and-closing---morphologyex)
6. [Gradient, Top-hat, Black-hat](#6-gradient-top-hat-black-hat)
7. [Practical Applications](#7-practical-applications)
8. [Exercises](#8-exercises)
9. [Next Steps](#9-next-steps)
10. [References](#10-references)

---

## 1. Morphological Operations Overview

Pixel-level filters like Gaussian blur treat all pixels equally, but many real-world tasks require reasoning about object *shape* — removing speckle noise without blurring object boundaries, or separating two touching cells. Morphological operations fill this gap by probing image structure with a shaped mask, making them the standard toolkit for binary image cleanup and shape analysis.

### What is Morphology?

```
┌─────────────────────────────────────────────────────────────────┐
│                  Morphological Operations Overview               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Morphology = Study of shape                                   │
│   Operations based on the shape of images                       │
│                                                                 │
│   Main Uses:                                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. Noise removal     - Remove small noise dots          │   │
│   │  2. Hole filling      - Fill holes inside objects        │   │
│   │  3. Object separation - Separate connected objects       │   │
│   │  4. Object connection - Connect disconnected parts       │   │
│   │  5. Edge detection    - Morphological gradient           │   │
│   │  6. Skeletonization  - Extract object skeleton           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   Basic Operations:                                             │
│   - Erosion: Shrink objects                                     │
│   - Dilation: Expand objects                                    │
│                                                                 │
│   Combined Operations:                                          │
│   - Opening = Erosion → Dilation                                │
│   - Closing = Dilation → Erosion                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Operating Principle

Morphological operations determine pixel values by moving a small mask called a **Structuring Element** across the image.

---

## 2. Structuring Element - getStructuringElement()

### What is a Structuring Element?

```
┌─────────────────────────────────────────────────────────────────┐
│                        Structuring Element                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Structuring Element = A small binary matrix used in operations│
│                                                                 │
│   Main Shapes:                                                  │
│                                                                 │
│   MORPH_RECT (Rectangle)   MORPH_CROSS (Cross)    MORPH_ELLIPSE │
│   ┌───┬───┬───┐           ┌───┬───┬───┐        ┌───┬───┬───┐  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 1 │ 1 │ 1 │        │ 1 │ 1 │ 1 │  │
│   ├───┼───┼───┤           ├───┼───┼───┤        ├───┼───┼───┤  │
│   │ 1 │ 1 │ 1 │           │ 0 │ 1 │ 0 │        │ 0 │ 1 │ 0 │  │
│   └───┴───┴───┘           └───┴───┴───┘        └───┴───┴───┘  │
│   All directions          Vertical/Horizontal   Elliptical      │
│                                                                 │
│   Effect by Size:                                               │
│   - Small size (3x3): Fine processing                           │
│   - Large size (7x7, 9x9): Strong effect                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Creating Structuring Elements

```python
import cv2
import numpy as np

# getStructuringElement(shape, ksize, anchor=(-1,-1))
# shape: Structuring element shape
# ksize: (width, height) size
# anchor: Reference point (default: center)

# Why MORPH_RECT: treats all directions equally — use when objects have
# roughly rectangular or straight edges (text, PCB traces)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
print("RECT (5x5):\n", rect_kernel)

# Cross
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
print("\nCROSS (5x5):\n", cross_kernel)

# Why MORPH_ELLIPSE: approximates a disk — preferred for circular/rounded
# objects (cells, coins) because it avoids introducing rectangular artifacts
# at diagonals that MORPH_RECT would create
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("\nELLIPSE (5x5):\n", ellipse_kernel)

# Custom structuring element
custom_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)
```

### Visualizing Structuring Elements

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

shapes = [
    ('RECT', cv2.MORPH_RECT),
    ('CROSS', cv2.MORPH_CROSS),
    ('ELLIPSE', cv2.MORPH_ELLIPSE)
]

sizes = [(5, 5), (7, 7), (11, 11)]

fig, axes = plt.subplots(len(shapes), len(sizes), figsize=(12, 10))

for i, (name, shape) in enumerate(shapes):
    for j, size in enumerate(sizes):
        kernel = cv2.getStructuringElement(shape, size)
        axes[i, j].imshow(kernel, cmap='gray')
        axes[i, j].set_title(f'{name} {size}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
```

---

## 3. Erosion - erode()

Erosion answers the question "does the foreground completely cover this kernel-shaped region?" — if not, the pixel is eliminated. This makes it the go-to tool for removing isolated noise dots (which can never fully cover the kernel) and for breaking thin bridges between touching objects before counting them separately.

### Erosion Operation Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                         Erosion                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 only if all pixels under the         │
│     structuring element are 1                                   │
│   - If any pixel is 0, center pixel becomes 0                   │
│                                                                 │
│   Effect:                                                       │
│   - Shrinks foreground (white) area                             │
│   - Removes small noise                                         │
│   - Separates connected objects                                 │
│   - Smooths boundaries                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Erosion (3x3):                      │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│ ──▶ │   ████████  │                          │
│   │ ████████████│     │   ████████  │                          │
│   │ ████████████│     │             │                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders shrink by 1 pixel                                     │
│                                                                 │
│   Noise Removal:                                                │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██  ■  ████ │     │ ██     ███  │                          │
│   │ ████  ████  │ ──▶ │  ██    ██   │  Small dots (■) removed  │
│   │    ■  ████  │     │       ███   │                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Using Erosion

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why MORPH_RECT (3x3): the smallest kernel that still has a meaningful neighborhood;
# larger kernels erode more aggressively and may destroy the objects you want to keep
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# erode(src, kernel, iterations=1)
# Why iterations: repeating erosion N times is equivalent to eroding with a larger
# kernel but cheaper to compute — use iterations to tune removal strength incrementally
eroded_1 = cv2.erode(binary, kernel, iterations=1)
eroded_2 = cv2.erode(binary, kernel, iterations=2)
eroded_3 = cv2.erode(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Eroded 1x', eroded_1)
cv2.imshow('Eroded 2x', eroded_2)
cv2.imshow('Eroded 3x', eroded_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Creating Erosion Test Images

```python
import cv2
import numpy as np

# Create test image
img = np.zeros((300, 400), dtype=np.uint8)

# Large rectangle
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Small noise dots
for _ in range(50):
    x, y = np.random.randint(200, 350), np.random.randint(50, 250)
    cv2.circle(img, (x, y), 2, 255, -1)

# Connected circles
cv2.circle(img, (280, 150), 40, 255, -1)
cv2.circle(img, (320, 150), 40, 255, -1)

# Apply erosion
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Original', img)
cv2.imshow('Eroded', eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 4. Dilation - dilate()

Dilation is the dual of erosion: it asks "does the foreground touch any part of this kernel region?" If yes, the pixel is set. This makes it ideal for reconnecting broken strokes and filling small gaps, and it is always paired with erosion (as opening or closing) so the overall object size is preserved.

### Dilation Operation Principle

```
┌─────────────────────────────────────────────────────────────────┐
│                         Dilation                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Principle:                                                    │
│   - Move the structuring element across the image               │
│   - Set center pixel to 1 if any pixel under the               │
│     structuring element is 1                                    │
│   - Opposite of erosion                                         │
│                                                                 │
│   Effect:                                                       │
│   - Expands foreground (white) area                             │
│   - Fills holes                                                 │
│   - Connects broken parts                                       │
│   - Emphasizes objects                                          │
│                                                                 │
│   Example:                                                      │
│   Original:           After Dilation (3x3):                     │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │   ██████    │     │ ████████████│                          │
│   │   ██████    │ ──▶ │ ████████████│                          │
│   │   ██████    │     │ ████████████│                          │
│   └─────────────┘     └─────────────┘                          │
│   Borders expand by 1 pixel                                     │
│                                                                 │
│   Connect Broken Parts:                                         │
│   ┌─────────────┐     ┌─────────────┐                          │
│   │ ██      ██  │     │ ████    ████│                          │
│   │ ██  ..  ██  │ ──▶ │ ██████████  │  Dotted line connected  │
│   │ ██      ██  │     │ ████    ████│                          │
│   └─────────────┘     └─────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Using Dilation

```python
import cv2
import numpy as np

# Prepare binary image
img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# dilate(src, kernel, iterations=1)
dilated_1 = cv2.dilate(binary, kernel, iterations=1)
dilated_2 = cv2.dilate(binary, kernel, iterations=2)
dilated_3 = cv2.dilate(binary, kernel, iterations=3)

cv2.imshow('Original', binary)
cv2.imshow('Dilated 1x', dilated_1)
cv2.imshow('Dilated 2x', dilated_2)
cv2.imshow('Dilated 3x', dilated_3)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Comparing Erosion and Dilation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 20, 0, -1)  # Inner hole

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

eroded = cv2.erode(img, kernel, iterations=1)
dilated = cv2.dilate(img, kernel, iterations=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title('Eroded (Shrink)')

axes[2].imshow(dilated, cmap='gray')
axes[2].set_title('Dilated (Expand)')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 5. Opening and Closing - morphologyEx()

Raw erosion shrinks objects permanently; raw dilation expands them. Opening and closing combine both operations so the net effect targets a *specific defect type* (noise dots or holes) while keeping object size approximately the same — this symmetry is why they are preferred over bare erosion/dilation in almost every practical pipeline.

### Opening

```
┌─────────────────────────────────────────────────────────────────┐
│                      Opening                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Opening = Erosion → Dilation                                  │
│                                                                 │
│   Effect:                                                       │
│   - Removes small noise (dots)                                  │
│   - Maintains overall object size approximately                 │
│   - Breaks thin connections                                     │
│                                                                 │
│   Original    Erosion      Dilation (Opening result)            │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██ ■ █│    │█     │    │██   █│                              │
│   │██████│ ─▶ │ ████ │ ─▶ │██████│                              │
│   │  ■ ██│    │    █ │    │    ██│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Small dots (■) removed                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Closing

```
┌─────────────────────────────────────────────────────────────────┐
│                      Closing                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Closing = Dilation → Erosion                                  │
│                                                                 │
│   Effect:                                                       │
│   - Fills small holes                                           │
│   - Maintains overall object size approximately                 │
│   - Connects broken parts                                       │
│                                                                 │
│   Original    Dilation     Erosion (Closing result)             │
│   ┌──────┐    ┌──────┐    ┌──────┐                              │
│   │██████│    │██████│    │██████│                              │
│   │██○ ██│ ─▶ │██████│ ─▶ │██████│                              │
│   │██████│    │██████│    │██████│                              │
│   └──────┘    └──────┘    └──────┘                              │
│   Inner hole (○) filled                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Using morphologyEx()

```python
import cv2
import numpy as np

img = cv2.imread('binary_image.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Why (5,5): kernel must be larger than the noise/holes you want to remove;
# a 5x5 kernel removes features smaller than ~5 pixels in diameter
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# morphologyEx(src, op, kernel, iterations=1)
# op: Operation type

# Why MORPH_OPEN first: erosion removes small noise dots; the subsequent dilation
# restores the larger objects to their original size (erode → dilate = opening)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: Hole filling (dilate → erode; expands to fill holes, then contracts back)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Why open before close: opening on the raw image avoids noise dots being "healed"
# into the object by the closing step — order matters
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', binary)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Open + Close', clean)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Comparing Opening and Closing Test

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Test image: Rectangle with noise + holes
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

# Add noise (small dots)
noise = img.copy()
for _ in range(30):
    x, y = np.random.randint(10, 45), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)
for _ in range(30):
    x, y = np.random.randint(155, 190), np.random.randint(10, 190)
    cv2.circle(noise, (x, y), 2, 255, -1)

# Add holes (inside object)
holes = noise.copy()
for _ in range(10):
    x, y = np.random.randint(60, 140), np.random.randint(60, 140)
    cv2.circle(holes, (x, y), 3, 0, -1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

opening = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, kernel)
both = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(holes, cmap='gray')
axes[0, 0].set_title('Original (Noise + Holes)')

axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title('Opening (Noise Removed)')

axes[1, 0].imshow(closing, cmap='gray')
axes[1, 0].set_title('Closing (Holes Filled)')

axes[1, 1].imshow(both, cmap='gray')
axes[1, 1].set_title('Open + Close')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. Gradient, Top-hat, Black-hat

### Morphological Gradient

```
┌─────────────────────────────────────────────────────────────────┐
│                   Morphological Gradient                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Gradient = Dilation - Erosion                                 │
│                                                                 │
│   Effect: Extract object outline (boundary)                     │
│                                                                 │
│   Original          Dilation           Erosion                  │
│   ┌──────┐         ┌──────┐         ┌──────┐                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   │ ████ │    -    │██████│    =    │  ██  │                   │
│   │ ████ │         │██████│         │  ██  │                   │
│   └──────┘         └──────┘         └──────┘                   │
│                                                                 │
│   Gradient Result:                                              │
│   ┌──────┐                                                      │
│   │ ████ │  → Only outline remains                              │
│   │ █  █ │                                                      │
│   │ ████ │                                                      │
│   └──────┘                                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Top-hat and Black-hat

```
┌─────────────────────────────────────────────────────────────────┐
│                    Top-hat / Black-hat                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Top-hat = Original - Opening                                  │
│   - Extract small bright parts from bright areas                │
│   - Detect small objects brighter than background               │
│                                                                 │
│   Black-hat = Closing - Original                                │
│   - Extract small dark parts from dark areas                    │
│   - Detect small holes/objects darker than background           │
│                                                                 │
│   Applications:                                                 │
│   - Correct images with uneven illumination                     │
│   - Remove shadows from document images                         │
│   - Detect small defects                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation and Usage

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

# Morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top-hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black-hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Manual calculation (for verification)
dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient_manual = dilated - eroded
tophat_manual = img - opening
blackhat_manual = closing - img

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(gradient, cmap='gray')
axes[0, 1].set_title('Gradient (Edge)')

axes[0, 2].imshow(tophat, cmap='gray')
axes[0, 2].set_title('Top Hat (Bright spots)')

axes[1, 0].imshow(blackhat, cmap='gray')
axes[1, 0].set_title('Black Hat (Dark spots)')

# Enhance contrast using top-hat + black-hat
enhanced = cv2.add(img, tophat)
enhanced = cv2.subtract(enhanced, blackhat)
axes[1, 1].imshow(enhanced, cmap='gray')
axes[1, 1].set_title('Enhanced (Top+Black Hat)')

for ax in axes.flatten():
    ax.axis('off')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
```

### Summary of All Morphological Operations

```python
import cv2

# List of operations available in morphologyEx()
operations = {
    cv2.MORPH_ERODE: "Erode",
    cv2.MORPH_DILATE: "Dilate",
    cv2.MORPH_OPEN: "Open (Erode + Dilate)",
    cv2.MORPH_CLOSE: "Close (Dilate + Erode)",
    cv2.MORPH_GRADIENT: "Gradient (Dilate - Erode)",
    cv2.MORPH_TOPHAT: "Top Hat (Src - Open)",
    cv2.MORPH_BLACKHAT: "Black Hat (Close - Src)",
    cv2.MORPH_HITMISS: "Hit-Miss (Pattern Matching)"
}

for op, name in operations.items():
    print(f"{op}: {name}")
```

---

## 7. Practical Applications

### Noise Removal Pipeline

```python
import cv2
import numpy as np

def remove_noise_morphology(binary_img, noise_size=3):
    """
    Remove noise using morphological operations

    Parameters:
    - binary_img: Binary image
    - noise_size: Maximum size of noise to remove
    """
    # Why noise_size * 2 + 1: the kernel must fully contain the largest noise dot
    # (radius noise_size → diameter noise_size*2) and be odd for a centered anchor
    kernel_size = noise_size * 2 + 1
    # Why MORPH_ELLIPSE: circular objects (cells, blobs) are better modeled with a
    # disk-shaped kernel — avoids introducing rectangular bias at diagonals
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    # Opening to remove small noise dots (erode kills noise, dilate restores objects)
    cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Closing to fill small holes (dilate bridges gaps, erode restores boundaries)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# Usage example
img = cv2.imread('noisy_document.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cleaned = remove_noise_morphology(binary, noise_size=2)
```

### Object Separation

```python
import cv2
import numpy as np

def separate_objects(binary_img, erosion_iterations=3):
    """
    Separate connected objects
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Why multiple erosion iterations instead of a large kernel: iterating with a
    # small kernel is cheaper and lets you tune separation strength without rebuilding
    # the structuring element; each pass peels off one layer from every boundary
    eroded = cv2.erode(binary_img, kernel, iterations=erosion_iterations)

    # Distance transform to find center points — the peak of the distance map is
    # the point farthest from any background pixel, i.e., the object center
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    # Why 0.5 * max: keeps only the top half of distance values, retaining confident
    # object cores while discarding ambiguous border regions
    _, sure_fg = cv2.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    return eroded, sure_fg


# Usage example
img = cv2.imread('connected_circles.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
separated, centers = separate_objects(binary)
```

### Document Image Preprocessing

```python
import cv2
import numpy as np

def preprocess_document(img):
    """
    Document image preprocessing (shadow removal + binarization)
    """
    # Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Top-hat to extract bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    # Black-hat to correct shadows/dark areas
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Subtract black-hat from original (shadow removal effect)
    no_shadow = cv2.add(gray, blackhat)

    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        no_shadow, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 15
    )

    # Noise removal
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)

    return binary


# Usage example
img = cv2.imread('document_with_shadow.jpg')
result = preprocess_document(img)
```

### Skeletonization

```python
import cv2
import numpy as np

def skeletonize(img):
    """
    Extract skeleton using morphological operations
    """
    skeleton = np.zeros_like(img)
    temp = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Opening operation
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)

        # Calculate difference
        diff = cv2.subtract(temp, opened)

        # Erosion
        temp = cv2.erode(temp, kernel)

        # Add to skeleton
        skeleton = cv2.bitwise_or(skeleton, diff)

        # Stop if no more white pixels
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


# Usage example
img = cv2.imread('character.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binary)
```

---

## 8. Exercises

### Exercise 1: Compare Structuring Element Effects

Apply erosion and dilation to the same binary image using three different structuring elements (RECT, CROSS, ELLIPSE) and analyze the differences in results.

### Exercise 2: Adjust Character Thickness

Write a function to adjust the thickness of characters in handwriting images:
- Positive value: Thicken with dilation
- Negative value: Thin with erosion

```python
def adjust_stroke_width(img, amount):
    """
    amount > 0: Thicken
    amount < 0: Thin
    """
    pass
```

### Exercise 3: Compare Boundary Extraction

Extract object boundaries using three different methods and compare:
1. Morphological gradient
2. Canny edge detection
3. findContours

### Exercise 4: Braille Recognition Preprocessing

Design a preprocessing pipeline to detect individual dots in braille images. (Hint: Use erosion to separate dots)

### Exercise 5: Cell Separation (Watershed Preprocessing)

Implement preprocessing to separate connected cells in microscope cell images:
1. Binarization
2. Noise removal (opening/closing)
3. Find sure background area (dilation)
4. Find sure foreground area (distance transform + threshold)

---

## 9. Next Steps

In [07_Thresholding.md](./07_Thresholding.md), you'll learn various binarization methods and thresholding techniques!

**Next Topics**:
- Global thresholding (`cv2.threshold`)
- OTSU automatic threshold
- Adaptive thresholding
- HSV-based thresholding

---

## 10. References

### Official Documentation

- [erode() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511f2fb1c)
- [dilate() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)
- [morphologyEx() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)
- [getStructuringElement() documentation](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc)

### Related Materials

| File | Related Content |
|------|-----------------|
| [05_Image_Filtering.md](./05_Image_Filtering.md) | Filtering basics |
| [09_Contours.md](./09_Contours.md) | Contour detection after preprocessing |

### Additional References

- [Morphological Operations Tutorial](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
- [Mathematical Morphology Theory](https://en.wikipedia.org/wiki/Mathematical_morphology)
