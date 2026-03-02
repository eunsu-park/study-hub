"""
Convolution Operations implemented with NumPy

This file implements Convolution forward/backward using pure NumPy.
"""

import numpy as np
from typing import Tuple, Optional


def conv2d_naive(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    2D Convolution (naive implementation with loops)

    Args:
        input: (N, C_in, H, W) - batch input
        kernel: (C_out, C_in, K_h, K_w) - filters
        bias: (C_out,) - bias
        stride: Stride
        padding: Padding

    Returns:
        output: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = kernel.shape

    # Apply padding
    if padding > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input

    # Compute output size
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    output = np.zeros((N, C_out, H_out, W_out))

    # Convolution operation (6-nested loop - very slow)
    for n in range(N):                          # Batch
        for c_out in range(C_out):              # Output channel
            for h in range(H_out):              # Output height
                for w in range(W_out):          # Output width
                    # Receptive field
                    h_start = h * stride
                    h_end = h_start + K_h
                    w_start = w * stride
                    w_end = w_start + K_w

                    # Element-wise product sum of receptive field and kernel
                    receptive_field = input_padded[n, :, h_start:h_end, w_start:w_end]
                    output[n, c_out, h, w] = np.sum(receptive_field * kernel[c_out])

    # Add bias
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def im2col(
    input: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    im2col: Transform image to matrix (for efficient convolution)

    Converts convolution to matrix multiplication:
    - Transform each receptive field into a column vector
    - Transform kernel into a row vector
    - Perform convolution via matrix multiplication

    Args:
        input: (N, C, H, W)
        kernel_size: (K_h, K_w)
        stride: Stride
        padding: Padding

    Returns:
        col: (N, C * K_h * K_w, H_out * W_out)
    """
    N, C, H, W = input.shape
    K_h, K_w = kernel_size

    # Padding
    if padding > 0:
        input_padded = np.pad(
            input,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )
    else:
        input_padded = input

    H_padded, W_padded = input_padded.shape[2], input_padded.shape[3]

    # Output size
    H_out = (H_padded - K_h) // stride + 1
    W_out = (W_padded - K_w) // stride + 1

    # im2col matrix
    col = np.zeros((N, C, K_h, K_w, H_out, W_out))

    for h in range(K_h):
        h_max = h + stride * H_out
        for w in range(K_w):
            w_max = w + stride * W_out
            col[:, :, h, w, :, :] = input_padded[:, :, h:h_max:stride, w:w_max:stride]

    # (N, C, K_h, K_w, H_out, W_out) -> (N, C*K_h*K_w, H_out*W_out)
    col = col.transpose(0, 1, 2, 3, 4, 5).reshape(N, C * K_h * K_w, H_out * W_out)

    return col


def col2im(
    col: np.ndarray,
    input_shape: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    col2im: Inverse of im2col

    Restores gradients to original image shape in backward pass

    Args:
        col: (N, C * K_h * K_w, H_out * W_out)
        input_shape: (N, C, H, W) original input shape
        kernel_size: (K_h, K_w)
        stride: Stride
        padding: Padding

    Returns:
        input_grad: (N, C, H, W)
    """
    N, C, H, W = input_shape
    K_h, K_w = kernel_size

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - K_h) // stride + 1
    W_out = (W_padded - K_w) // stride + 1

    # col reshape: (N, C*K_h*K_w, H_out*W_out) -> (N, C, K_h, K_w, H_out, W_out)
    col = col.reshape(N, C, K_h, K_w, H_out, W_out)

    # Output array (including padding)
    input_padded = np.zeros((N, C, H_padded, W_padded))

    # Accumulate (add values at stride positions)
    for h in range(K_h):
        h_max = h + stride * H_out
        for w in range(K_w):
            w_max = w + stride * W_out
            input_padded[:, :, h:h_max:stride, w:w_max:stride] += col[:, :, h, w, :, :]

    # Remove padding
    if padding > 0:
        return input_padded[:, :, padding:-padding, padding:-padding]
    return input_padded


def conv2d_im2col(
    input: np.ndarray,
    kernel: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    Efficient Convolution using im2col

    Operation: Y = W . col(X) + b
    """
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = kernel.shape

    # im2col transform
    col = im2col(input, (K_h, K_w), stride, padding)  # (N, C_in*K_h*K_w, H_out*W_out)

    # Transform kernel to matrix
    kernel_mat = kernel.reshape(C_out, -1)  # (C_out, C_in*K_h*K_w)

    # Matrix multiplication
    H_out = (H + 2 * padding - K_h) // stride + 1
    W_out = (W + 2 * padding - K_w) // stride + 1

    # (C_out, C_in*K_h*K_w) @ (N, C_in*K_h*K_w, H_out*W_out)
    # -> (N, C_out, H_out*W_out)
    output = np.zeros((N, C_out, H_out * W_out))
    for n in range(N):
        output[n] = kernel_mat @ col[n]

    # Reshape
    output = output.reshape(N, C_out, H_out, W_out)

    # Bias
    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


class Conv2dNumpy:
    """
    NumPy Convolution Layer (trainable)

    Both forward/backward implemented
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming (He) initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * scale
        self.bias = np.zeros(out_channels)

        # Gradient storage
        self.weight_grad = None
        self.bias_grad = None

        # Cache for backward
        self.cache = {}

    def forward(self, input: np.ndarray) -> np.ndarray:
        """Forward pass"""
        N, C, H, W = input.shape

        # im2col
        col = im2col(input, (self.kernel_size, self.kernel_size),
                     self.stride, self.padding)

        # Save to cache
        self.cache['input_shape'] = input.shape
        self.cache['col'] = col

        # Matrix multiplication
        kernel_mat = self.weight.reshape(self.out_channels, -1)

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = np.zeros((N, self.out_channels, H_out * W_out))
        for n in range(N):
            output[n] = kernel_mat @ col[n]

        output = output.reshape(N, self.out_channels, H_out, W_out)
        output += self.bias.reshape(1, -1, 1, 1)

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            grad_output: dL/dY (N, C_out, H_out, W_out)

        Returns:
            grad_input: dL/dX (N, C_in, H, W)
        """
        N, C_out, H_out, W_out = grad_output.shape
        input_shape = self.cache['input_shape']
        col = self.cache['col']

        # Bias gradient: dL/db = sum(dL/dY)
        self.bias_grad = np.sum(grad_output, axis=(0, 2, 3))

        # Transform grad_output to matrix
        grad_output_mat = grad_output.reshape(N, C_out, -1)  # (N, C_out, H_out*W_out)

        # Weight gradient: dL/dW = dL/dY . col(X)^T
        kernel_mat = self.weight.reshape(self.out_channels, -1)
        self.weight_grad = np.zeros_like(kernel_mat)

        for n in range(N):
            self.weight_grad += grad_output_mat[n] @ col[n].T

        self.weight_grad = self.weight_grad.reshape(self.weight.shape)

        # Input gradient: dL/dX = col2im(W^T . dL/dY)
        grad_col = np.zeros_like(col)
        for n in range(N):
            grad_col[n] = kernel_mat.T @ grad_output_mat[n]

        grad_input = col2im(
            grad_col, input_shape,
            (self.kernel_size, self.kernel_size),
            self.stride, self.padding
        )

        return grad_input

    def update(self, lr: float):
        """Weight update"""
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad


# Test
if __name__ == "__main__":
    np.random.seed(42)

    # Test input
    N, C_in, H, W = 2, 3, 8, 8
    C_out, K = 4, 3

    input = np.random.randn(N, C_in, H, W)
    kernel = np.random.randn(C_out, C_in, K, K)
    bias = np.random.randn(C_out)

    # Naive vs im2col comparison
    # Expected: naive vs im2col diff < 1e-10
    output_naive = conv2d_naive(input, kernel, bias, stride=1, padding=1)
    output_im2col = conv2d_im2col(input, kernel, bias, stride=1, padding=1)

    print("Output shape:", output_naive.shape)
    print("Naive vs im2col difference:", np.max(np.abs(output_naive - output_im2col)))

    # Conv2dNumpy test
    conv = Conv2dNumpy(C_in, C_out, K, stride=1, padding=1)
    output = conv.forward(input)
    print("\nConv2dNumpy output shape:", output.shape)

    # Backward test
    grad_output = np.random.randn(*output.shape)
    grad_input = conv.backward(grad_output)
    print("Grad input shape:", grad_input.shape)
    print("Weight grad shape:", conv.weight_grad.shape)

    # Gradient check
    def numerical_gradient(f, x, h=1e-5):
        """Verify gradient using numerical differentiation"""
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            old_val = x[idx]

            x[idx] = old_val + h
            fxh1 = f()

            x[idx] = old_val - h
            fxh2 = f()

            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = old_val

            it.iternext()

        return grad

    # Expected: input gradient diff < 1e-5, weight gradient diff < 1e-5
    print("\n=== Gradient Check ===")

    # Gradient check with small input
    small_input = np.random.randn(1, 2, 4, 4)
    small_conv = Conv2dNumpy(2, 2, 3, stride=1, padding=1)

    def loss_fn():
        out = small_conv.forward(small_input)
        return np.sum(out ** 2)

    # Analytical gradient
    output = small_conv.forward(small_input)
    grad_output = 2 * output  # d(sum(x^2))/dx = 2x
    grad_input = small_conv.backward(grad_output)

    # Numerical gradient (with respect to input)
    num_grad = numerical_gradient(loss_fn, small_input)

    print("Input gradient check:")
    print(f"  Max diff: {np.max(np.abs(grad_input - num_grad)):.2e}")

    # Weight gradient check
    def loss_fn_weight():
        out = small_conv.forward(small_input)
        return np.sum(out ** 2)

    num_grad_weight = numerical_gradient(loss_fn_weight, small_conv.weight)

    # Compute via backward
    output = small_conv.forward(small_input)
    small_conv.backward(2 * output)

    print("Weight gradient check:")
    print(f"  Max diff: {np.max(np.abs(small_conv.weight_grad - num_grad_weight)):.2e}")
