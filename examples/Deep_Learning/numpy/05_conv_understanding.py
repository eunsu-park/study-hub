"""
05. Understanding Convolution - NumPy Version (Educational)

Understand the principles of convolution operations with NumPy.
For actual CNN training, use PyTorch!

This file is for understanding how convolution works.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("NumPy Convolution Understanding (Educational)")
print("=" * 60)


# ============================================
# 1. Basic 2D Convolution
# ============================================
print("\n[1] Basic 2D Convolution")
print("-" * 40)

def conv2d_basic(image, kernel):
    """
    Most basic 2D convolution implementation

    Args:
        image: 2D array (H, W)
        kernel: 2D array (kH, kW)

    Returns:
        Output (H-kH+1, W-kW+1)
    """
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Extract region
            region = image[i:i+kh, j:j+kw]
            # Element-wise multiplication then sum
            output[i, j] = np.sum(region * kernel)

    return output

# Test
image = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1]
], dtype=float)

kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=float)

output = conv2d_basic(image, kernel)
print(f"Input image (4x4):\n{image}")
print(f"\nKernel (2x2):\n{kernel}")
print(f"\nOutput (3x3):\n{output}")
print(f"\nExample computation (top-left):")
print(f"  {image[0,0]}x{kernel[0,0]} + {image[0,1]}x{kernel[0,1]} + {image[1,0]}x{kernel[1,0]} + {image[1,1]}x{kernel[1,1]}")
print(f"  = 1x1 + 2x0 + 0x0 + 1x(-1) = 0")


# ============================================
# 2. Padding and Stride
# ============================================
print("\n[2] Padding and Stride")
print("-" * 40)

def conv2d_with_padding(image, kernel, padding=0, stride=1):
    """Convolution with padding and stride support"""
    # Apply padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    h, w = image.shape
    kh, kw = kernel.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+kh, sj:sj+kw]
            output[i, j] = np.sum(region * kernel)

    return output

# Test
image = np.ones((4, 4))
kernel = np.ones((3, 3))

print("Input: 4x4, Kernel: 3x3")
for p in [0, 1]:
    for s in [1, 2]:
        out = conv2d_with_padding(image, kernel, padding=p, stride=s)
        print(f"  padding={p}, stride={s} -> output: {out.shape}")


# ============================================
# 3. Edge Detection Filters
# ============================================
print("\n[3] Edge Detection Filters")
print("-" * 40)

# Generate sample image
def create_sample_image():
    """Generate a simple pattern image"""
    img = np.zeros((8, 8))
    img[2:6, 2:6] = 1  # Center square
    return img

image = create_sample_image()

# Edge detection filters
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]])

# Apply filters
edge_x = conv2d_with_padding(image, sobel_x, padding=1)
edge_y = conv2d_with_padding(image, sobel_y, padding=1)
edge_laplace = conv2d_with_padding(image, laplacian, padding=1)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(sobel_x, cmap='RdBu')
axes[0, 1].set_title('Sobel X Filter')
axes[0, 2].imshow(sobel_y, cmap='RdBu')
axes[0, 2].set_title('Sobel Y Filter')
axes[1, 0].imshow(edge_x, cmap='gray')
axes[1, 0].set_title('Sobel X Edge')
axes[1, 1].imshow(edge_y, cmap='gray')
axes[1, 1].set_title('Sobel Y Edge')
axes[1, 2].imshow(edge_laplace, cmap='gray')
axes[1, 2].set_title('Laplacian Edge')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('numpy_edge_detection.png', dpi=100)
plt.close()
print("Edge detection saved: numpy_edge_detection.png")


# ============================================
# 4. Pooling Operations
# ============================================
print("\n[4] Pooling Operations")
print("-" * 40)

def max_pool2d(image, pool_size=2, stride=2):
    """Max Pooling implementation"""
    h, w = image.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+pool_size, sj:sj+pool_size]
            output[i, j] = np.max(region)

    return output

def avg_pool2d(image, pool_size=2, stride=2):
    """Average Pooling implementation"""
    h, w = image.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            si, sj = i * stride, j * stride
            region = image[si:si+pool_size, sj:sj+pool_size]
            output[i, j] = np.mean(region)

    return output

# Test
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

print(f"Input:\n{image}")
print(f"\nMax Pooling (2x2):\n{max_pool2d(image)}")
print(f"\nAvg Pooling (2x2):\n{avg_pool2d(image)}")


# ============================================
# 5. Multi-Channel Convolution
# ============================================
print("\n[5] Multi-Channel Convolution")
print("-" * 40)

def conv2d_multichannel(image, kernels, bias=0):
    """
    Multi-channel convolution (e.g., RGB images)

    Args:
        image: (C, H, W) - C channels
        kernels: (C, kH, kW) - kernel for each channel
        bias: Bias

    Returns:
        Output: (H-kH+1, W-kW+1)
    """
    c, h, w = image.shape
    _, kh, kw = kernels.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    # Convolve each channel and sum
    for ch in range(c):
        output += conv2d_basic(image[ch], kernels[ch])

    return output + bias

# RGB image example
rgb_image = np.random.rand(3, 8, 8)  # (C, H, W)
kernels = np.random.rand(3, 3, 3)    # (C, kH, kW)

output = conv2d_multichannel(rgb_image, kernels)
print(f"Input: {rgb_image.shape} (3 channels)")
print(f"Kernel: {kernels.shape} (3x3 per channel)")
print(f"Output: {output.shape}")


# ============================================
# 6. Applying Multiple Filters
# ============================================
print("\n[6] Applying Multiple Filters")
print("-" * 40)

def conv2d_layer(image, filters, biases):
    """
    Conv layer simulation

    Args:
        image: (C_in, H, W)
        filters: (C_out, C_in, kH, kW)
        biases: (C_out,)

    Returns:
        Output: (C_out, oH, oW)
    """
    c_out, c_in, kh, kw = filters.shape
    _, h, w = image.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((c_out, oh, ow))

    for f in range(c_out):
        output[f] = conv2d_multichannel(image, filters[f], biases[f])

    return output

# Example: 3-channel input -> 8-channel output
image = np.random.rand(3, 16, 16)
filters = np.random.rand(8, 3, 3, 3)  # 8 filters
biases = np.zeros(8)

output = conv2d_layer(image, filters, biases)
print(f"Input: {image.shape}")
print(f"Filters: {filters.shape}")
print(f"Output: {output.shape}")


# ============================================
# 7. CNN Forward Pass Simulation
# ============================================
print("\n[7] CNN Forward Pass Simulation")
print("-" * 40)

def relu(x):
    return np.maximum(0, x)

def simple_cnn_forward(image):
    """
    Simple CNN forward pass

    Input (1, 8, 8) -> Conv (2, 6, 6) -> Pool (2, 3, 3) -> FC -> Output
    """
    # Conv1: 1->2 channels, 3x3 kernel
    filters1 = np.random.randn(2, 1, 3, 3) * 0.5
    biases1 = np.zeros(2)

    conv1_out = conv2d_layer(image, filters1, biases1)
    relu1_out = relu(conv1_out)
    print(f"  After Conv1: {relu1_out.shape}")

    # MaxPool: 2x2
    pool_out = np.zeros((2, 3, 3))
    for c in range(2):
        pool_out[c] = max_pool2d(relu1_out[c], 2, 2)
    print(f"  After Pool: {pool_out.shape}")

    # Flatten
    flat = pool_out.flatten()
    print(f"  Flatten: {flat.shape}")

    # FC
    fc_weights = np.random.randn(10, 18) * 0.5
    fc_bias = np.zeros(10)
    output = fc_weights @ flat + fc_bias
    print(f"  FC output: {output.shape}")

    return output

# Test
image = np.random.rand(1, 8, 8)
print(f"Input: {image.shape}")
output = simple_cnn_forward(image)


# ============================================
# Why Use PyTorch?
# ============================================
print("\n" + "=" * 60)
print("Limitations of NumPy CNN")
print("=" * 60)

limitations = """
Problems with NumPy implementation:

1. Speed
   - Pure Python loops are very slow
   - Even 28x28 MNIST is thousands of times slower
   - No GPU acceleration possible

2. Backpropagation
   - Convolution backpropagation is complex to implement
   - Requires optimizations like im2col
   - Error-prone

3. Memory
   - Inefficient memory usage
   - Difficult batch processing

4. Features
   - BatchNorm, Dropout implementation is complex
   - Lacks diverse layers/operations

Why use PyTorch:
   - Convolution optimized with cuDNN
   - Automatic differentiation (auto backpropagation)
   - GPU support
   - Rich layers/functions provided
"""
print(limitations)


# ============================================
# Summary
# ============================================
print("=" * 60)
print("Convolution Key Summary")
print("=" * 60)

summary = """
Convolution operation:
    output[i,j] = sum( input[i+m, j+n] x kernel[m, n] )

Output size:
    output_size = (input - kernel + 2 x padding) / stride + 1

Pooling:
    - MaxPool: Select maximum value in region
    - AvgPool: Average of region

Multi-channel:
    - Apply separate kernel to each channel, then sum
    - Multiple filters = multiple output channels

Training:
    - Kernel weights are learned
    - Optimized via backpropagation

What we learned with NumPy:
    1. Mathematical definition of convolution
    2. Effects of padding and stride
    3. How pooling works
    4. Multi-channel processing approach

For production, use PyTorch!
"""
print(summary)
print("=" * 60)
