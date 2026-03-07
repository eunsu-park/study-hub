#!/bin/bash
# Exercises for Lesson 17: Linear Algebra in Signal Processing
# Topic: Linear_Algebra
# Solutions to practice problems from the lesson.

# === Exercise 1: DFT as Matrix Multiplication ===
# Problem: Construct the 4x4 DFT matrix and compute the DFT of
# x = [1, 2, 3, 4]. Compare with np.fft.fft.
exercise_1() {
    echo "=== Exercise 1: DFT as Matrix Multiplication ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

N = 4
x = np.array([1, 2, 3, 4], dtype=complex)

# DFT matrix: F[j,k] = omega^{jk}, omega = e^{-2*pi*i/N}
omega = np.exp(-2j * np.pi / N)
F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)])

print(f"DFT matrix F (N={N}):")
print(np.round(F, 4))

# DFT via matrix multiplication
X_matrix = F @ x
X_fft = np.fft.fft(x)

print(f"\nx = {x.real.astype(int)}")
print(f"DFT (matrix): {np.round(X_matrix, 4)}")
print(f"DFT (fft):    {np.round(X_fft, 4)}")
print(f"Match: {np.allclose(X_matrix, X_fft)}")

# F is unitary (up to scaling): F^H F = N * I
print(f"\nF^H @ F / N:\n{np.round(F.conj().T @ F / N, 10)}")
print(f"Unitary (scaled): {np.allclose(F.conj().T @ F, N * np.eye(N))}")

# Inverse DFT: x = F^{-1} X = F^H X / N
x_recovered = F.conj().T @ X_matrix / N
print(f"\nIDFT: {np.round(x_recovered.real, 4)}")
SOLUTION
}

# === Exercise 2: Convolution as Matrix Multiply ===
# Problem: Implement convolution of signal [1,2,3,4,5] with kernel [1,-1]
# using a Toeplitz matrix.
exercise_2() {
    echo "=== Exercise 2: Convolution as Matrix Multiply ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import toeplitz

x = np.array([1, 2, 3, 4, 5], dtype=float)
h = np.array([1, -1], dtype=float)

# Convolution via np.convolve
y_conv = np.convolve(x, h, mode='full')
print(f"x = {x}")
print(f"h = {h}")
print(f"np.convolve: {y_conv}")

# Toeplitz matrix for convolution
# For 'full' output: size = len(x) + len(h) - 1
n_out = len(x) + len(h) - 1
# Build column and row for Toeplitz
col = np.zeros(n_out)
col[:len(h)] = h
row = np.zeros(len(x))
row[0] = h[0]
T = toeplitz(col, row)

y_matrix = T @ x
print(f"\nToeplitz matrix:\n{T}")
print(f"T @ x = {y_matrix}")
print(f"Match: {np.allclose(y_conv, y_matrix)}")
SOLUTION
}

# === Exercise 3: Filtering in Frequency Domain ===
# Problem: Apply a low-pass filter to a signal containing 5 Hz and
# 50 Hz components using the DFT.
exercise_3() {
    echo "=== Exercise 3: Filtering in Frequency Domain ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np

# Create signal: 5 Hz + 50 Hz
fs = 200  # Sampling frequency
t = np.arange(0, 1, 1/fs)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# DFT
X = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), 1/fs)

print(f"Signal: {len(t)} samples at {fs} Hz")
print(f"Contains: 5 Hz (amplitude 1) + 50 Hz (amplitude 0.5)")

# Low-pass filter: zero out frequencies above 20 Hz
cutoff = 20  # Hz
H = np.ones_like(X)
H[np.abs(freqs) > cutoff] = 0

# Apply filter in frequency domain (element-wise multiply)
Y = X * H
y_filtered = np.fft.ifft(Y).real

# Energy analysis
energy_original = np.sum(np.abs(X)**2)
energy_filtered = np.sum(np.abs(Y)**2)

print(f"\nFilter: zero out |f| > {cutoff} Hz")
print(f"Energy retained: {energy_filtered / energy_original * 100:.1f}%")

# Verify filtered signal is ~pure 5 Hz
residual = y_filtered - np.sin(2 * np.pi * 5 * t)
print(f"Residual from pure 5 Hz: {np.linalg.norm(residual):.4f}")
print(f"50 Hz component removed successfully")
SOLUTION
}

# === Exercise 4: Circulant Matrix and DFT ===
# Problem: Show that circulant matrices are diagonalized by the DFT matrix.
exercise_4() {
    echo "=== Exercise 4: Circulant Matrix and DFT ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import numpy as np
from scipy.linalg import circulant

c = np.array([1, 2, 3, 4], dtype=complex)
C = circulant(c)
N = len(c)

print(f"First column: {c.real.astype(int)}")
print(f"Circulant matrix C:\n{C.real.astype(int)}")

# DFT matrix
omega = np.exp(-2j * np.pi / N)
F = np.array([[omega**(j*k) for k in range(N)] for j in range(N)])

# Diagonalization: C = F^H diag(Fc) F / N
Fc = F @ c  # DFT of first column
D = np.diag(Fc)

C_reconstructed = F.conj().T @ D @ F / N
print(f"\nDFT of first column: {np.round(Fc, 4)}")
print(f"C == F^H diag(Fc) F / N: {np.allclose(C, C_reconstructed)}")

# Eigenvalues of circulant = DFT of first column
evals = np.linalg.eigvals(C)
print(f"\nEigenvalues of C: {np.round(np.sort(evals), 4)}")
print(f"DFT of c: {np.round(np.sort(Fc), 4)}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 17: Linear Algebra in Signal Processing"
echo "======================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
