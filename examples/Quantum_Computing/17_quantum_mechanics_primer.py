"""
17_quantum_mechanics_primer.py — Quantum Mechanics Foundations for Computing

Demonstrates:
  - Wave-particle duality: double-slit interference pattern
  - Superposition principle with complex amplitudes
  - Heisenberg uncertainty relation verification
  - Operator eigenvalue problems (position, momentum, energy)
  - Time evolution of a quantum wavepacket
  - Born rule: probability from amplitude

All computations use pure NumPy.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants (natural units: hbar = 1, m = 1)
# ---------------------------------------------------------------------------

HBAR = 1.0
MASS = 1.0


# ---------------------------------------------------------------------------
# Wavepacket construction
# ---------------------------------------------------------------------------

def gaussian_wavepacket(x: np.ndarray, x0: float, sigma: float,
                        k0: float) -> np.ndarray:
    """Create a Gaussian wavepacket centered at x0 with momentum k0.

    ψ(x) = (2πσ²)^{-1/4} exp(-(x-x0)²/(4σ²)) exp(ik0·x)

    Why: The Gaussian wavepacket is the minimum-uncertainty state — it
    saturates the Heisenberg uncertainty bound Δx·Δp = ℏ/2.  It serves
    as the ideal starting point for understanding quantum dynamics.
    """
    norm = (2.0 * np.pi * sigma ** 2) ** (-0.25)
    envelope = np.exp(-(x - x0) ** 2 / (4.0 * sigma ** 2))
    phase = np.exp(1j * k0 * x)
    return norm * envelope * phase


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    """Normalize a wavefunction so that ∫|ψ|² dx = 1."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


# ---------------------------------------------------------------------------
# Expectation values and uncertainty
# ---------------------------------------------------------------------------

def expectation_position(psi: np.ndarray, x: np.ndarray,
                         dx: float) -> float:
    """Compute ⟨x⟩ = ∫ ψ*(x) x ψ(x) dx."""
    return float(np.real(np.sum(psi.conj() * x * psi) * dx))


def expectation_position_sq(psi: np.ndarray, x: np.ndarray,
                            dx: float) -> float:
    """Compute ⟨x²⟩ = ∫ ψ*(x) x² ψ(x) dx."""
    return float(np.real(np.sum(psi.conj() * x ** 2 * psi) * dx))


def expectation_momentum(psi: np.ndarray, dx: float) -> float:
    """Compute ⟨p⟩ using the momentum-space representation.

    Why: In position space, the momentum operator is p = -iℏ d/dx.
    We compute ⟨p⟩ via FFT, which diagonalizes the momentum operator.
    """
    N = len(psi)
    dk = 2.0 * np.pi / (N * dx)
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    psi_k = np.fft.fft(psi) * dx / np.sqrt(2.0 * np.pi)
    return float(np.real(np.sum(psi_k.conj() * HBAR * k * psi_k) * dk))


def expectation_momentum_sq(psi: np.ndarray, dx: float) -> float:
    """Compute ⟨p²⟩ using the momentum-space representation."""
    N = len(psi)
    dk = 2.0 * np.pi / (N * dx)
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    psi_k = np.fft.fft(psi) * dx / np.sqrt(2.0 * np.pi)
    return float(np.real(np.sum(psi_k.conj() * (HBAR * k) ** 2 * psi_k) * dk))


def uncertainty(psi: np.ndarray, x: np.ndarray,
                dx: float) -> Tuple[float, float, float]:
    """Compute Δx, Δp, and Δx·Δp.

    Why: The Heisenberg uncertainty principle states Δx·Δp ≥ ℏ/2.
    For a Gaussian wavepacket, equality holds (minimum uncertainty).
    """
    exp_x = expectation_position(psi, x, dx)
    exp_x2 = expectation_position_sq(psi, x, dx)
    delta_x = np.sqrt(max(exp_x2 - exp_x ** 2, 0))

    exp_p = expectation_momentum(psi, dx)
    exp_p2 = expectation_momentum_sq(psi, dx)
    delta_p = np.sqrt(max(exp_p2 - exp_p ** 2, 0))

    return delta_x, delta_p, delta_x * delta_p


# ---------------------------------------------------------------------------
# Double-slit interference
# ---------------------------------------------------------------------------

def double_slit_pattern(x_screen: np.ndarray, slit_sep: float,
                        slit_width: float, wavelength: float,
                        distance: float) -> np.ndarray:
    """Compute double-slit interference pattern using wave optics.

    Why: The double-slit experiment is the quintessential demonstration
    of quantum superposition.  Even single particles produce an interference
    pattern, revealing their wave nature.  The pattern arises from the
    superposition of amplitudes (not probabilities) from each slit.
    """
    k = 2.0 * np.pi / wavelength

    # Amplitude from each slit (Fraunhofer diffraction)
    # Single-slit envelope: sinc(π·a·sin(θ)/λ)
    theta = x_screen / distance  # small angle approximation
    single_slit = np.sinc(slit_width * theta / wavelength)

    # Double-slit interference: cos(π·d·sin(θ)/λ)
    interference = np.cos(np.pi * slit_sep * theta / wavelength)

    intensity = (single_slit * interference) ** 2
    return intensity / np.max(intensity)


# ---------------------------------------------------------------------------
# Time evolution (split-operator method)
# ---------------------------------------------------------------------------

def free_particle_evolution(psi: np.ndarray, x: np.ndarray,
                            dx: float, dt: float,
                            n_steps: int) -> np.ndarray:
    """Evolve a free-particle wavepacket using the split-operator FFT method.

    Why: For a free particle, H = p²/(2m).  The split-operator method
    uses the FFT to switch between position and momentum representations,
    where each part of the Hamiltonian is diagonal.  This is exact for
    a purely kinetic Hamiltonian.
    """
    N = len(x)
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

    # Kinetic energy propagator in momentum space
    kinetic_prop = np.exp(-1j * HBAR * k ** 2 * dt / (2.0 * MASS))

    state = psi.copy()
    for _ in range(n_steps):
        state_k = np.fft.fft(state)
        state_k *= kinetic_prop
        state = np.fft.ifft(state_k)

    return state


def harmonic_oscillator_eigenstates(x: np.ndarray, omega: float,
                                     n_max: int) -> np.ndarray:
    """Compute harmonic oscillator eigenstates using the recursion relation.

    Why: The quantum harmonic oscillator is exactly solvable and its energy
    levels E_n = ℏω(n + 1/2) are equally spaced.  It is the foundation for
    understanding quantum fields, phonons, and photons.
    """
    states = np.zeros((n_max, len(x)))

    # Ground state: ψ_0(x) = (mω/πℏ)^{1/4} exp(-mωx²/(2ℏ))
    alpha = MASS * omega / HBAR
    states[0] = (alpha / np.pi) ** 0.25 * np.exp(-alpha * x ** 2 / 2.0)

    if n_max > 1:
        # ψ_1(x) = √(2α) x ψ_0(x)
        states[1] = np.sqrt(2.0 * alpha) * x * states[0]

    # Recursion: ψ_{n+1} = √(2α/(n+1)) x ψ_n - √(n/(n+1)) ψ_{n-1}
    for n in range(1, n_max - 1):
        states[n + 1] = (np.sqrt(2.0 * alpha / (n + 1)) * x * states[n]
                         - np.sqrt(n / (n + 1)) * states[n - 1])

    return states


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_superposition():
    """Show superposition principle and Born rule."""
    print("=" * 60)
    print("DEMO 1: Superposition and Born Rule")
    print("=" * 60)

    N = 1024
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    # Why: A superposition of two Gaussian wavepackets shows interference.
    # The probability |ψ_1 + ψ_2|² ≠ |ψ_1|² + |ψ_2|² — the cross term
    # (interference) is the hallmark of quantum mechanics.
    psi1 = gaussian_wavepacket(x, x0=-2.0, sigma=0.8, k0=3.0)
    psi2 = gaussian_wavepacket(x, x0=2.0, sigma=0.8, k0=-3.0)

    psi_sum = normalize(psi1 + psi2, dx)
    prob_sum = np.abs(psi_sum) ** 2
    prob_classical = (np.abs(psi1) ** 2 + np.abs(psi2) ** 2)
    prob_classical /= np.sum(prob_classical) * dx

    print(f"\n  Superposition of two Gaussian wavepackets:")
    print(f"  ψ_1: centered at x=-2, k=+3")
    print(f"  ψ_2: centered at x=+2, k=-3")
    print(f"\n  Normalization check: ∫|ψ|² dx = {np.sum(prob_sum) * dx:.6f}")

    # Show interference fringes
    print(f"\n  {'x':>8} {'|ψ₁+ψ₂|²':>12} {'|ψ₁|²+|ψ₂|²':>14} {'Diff':>10}")
    print(f"  {'─' * 48}")
    for i in range(0, N, N // 20):
        diff = prob_sum[i] - prob_classical[i]
        print(f"  {x[i]:8.2f} {prob_sum[i]:12.6f} {prob_classical[i]:14.6f} "
              f"{diff:10.6f}")


def demo_double_slit():
    """Demonstrate double-slit interference pattern."""
    print("\n" + "=" * 60)
    print("DEMO 2: Double-Slit Interference")
    print("=" * 60)

    x_screen = np.linspace(-0.05, 0.05, 200)
    wavelength = 500e-9  # 500 nm (green light)
    slit_sep = 0.1e-3    # 0.1 mm
    slit_width = 0.02e-3 # 0.02 mm
    distance = 1.0        # 1 m

    pattern = double_slit_pattern(x_screen, slit_sep, slit_width,
                                  wavelength, distance)

    # Why: The fringe spacing Δx = λD/d tells us about the wavelength.
    # This is how de Broglie wavelength λ = h/p was confirmed for electrons.
    fringe_spacing = wavelength * distance / slit_sep

    print(f"\n  Parameters:")
    print(f"    Wavelength: {wavelength * 1e9:.0f} nm")
    print(f"    Slit separation: {slit_sep * 1e3:.2f} mm")
    print(f"    Slit width: {slit_width * 1e3:.3f} mm")
    print(f"    Screen distance: {distance:.1f} m")
    print(f"    Expected fringe spacing: {fringe_spacing * 1e3:.3f} mm")

    # ASCII visualization
    print(f"\n  Intensity pattern:")
    for i in range(0, len(x_screen), 4):
        bar = int(pattern[i] * 40)
        print(f"  {x_screen[i] * 1e3:+6.2f} mm |{'#' * bar}")


def demo_uncertainty():
    """Verify Heisenberg uncertainty principle."""
    print("\n" + "=" * 60)
    print("DEMO 3: Heisenberg Uncertainty Principle")
    print("=" * 60)

    N = 2048
    x = np.linspace(-20, 20, N)
    dx = x[1] - x[0]

    print(f"\n  Gaussian wavepackets with varying width σ:")
    print(f"  (Heisenberg bound: Δx·Δp ≥ ℏ/2 = {HBAR / 2:.4f})")
    print(f"\n  {'σ':>8} {'Δx':>10} {'Δp':>10} {'Δx·Δp':>10} {'≥ ℏ/2?':>10}")
    print(f"  {'─' * 52}")

    for sigma in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]:
        psi = gaussian_wavepacket(x, x0=0.0, sigma=sigma, k0=0.0)
        psi = normalize(psi, dx)
        dx_val, dp_val, product = uncertainty(psi, x, dx)
        satisfies = "Yes" if product >= HBAR / 2 - 1e-6 else "No"
        print(f"  {sigma:8.2f} {dx_val:10.4f} {dp_val:10.4f} "
              f"{product:10.4f} {satisfies:>10}")

    # Why: For Gaussians, Δx·Δp = ℏ/2 exactly (minimum uncertainty).
    # Non-Gaussian states always have Δx·Δp > ℏ/2.


def demo_free_evolution():
    """Show free-particle wavepacket spreading."""
    print("\n" + "=" * 60)
    print("DEMO 4: Free Particle Wavepacket Evolution")
    print("=" * 60)

    N = 2048
    x = np.linspace(-30, 30, N)
    dx = x[1] - x[0]

    sigma = 1.0
    k0 = 3.0
    psi0 = gaussian_wavepacket(x, x0=-5.0, sigma=sigma, k0=k0)
    psi0 = normalize(psi0, dx)

    # Why: A free Gaussian wavepacket moves at the group velocity v_g = ℏk₀/m
    # and spreads over time.  The spreading rate depends on the initial width:
    # σ(t) = σ₀√(1 + (ℏt/(2mσ₀²))²).
    print(f"\n  Initial: x₀ = -5.0, σ = {sigma}, k₀ = {k0}")
    print(f"  Group velocity: v_g = ℏk₀/m = {HBAR * k0 / MASS:.2f}")
    print(f"\n  {'t':>6} {'⟨x⟩':>10} {'Δx':>10} {'⟨p⟩':>10} {'Δx·Δp':>10}")
    print(f"  {'─' * 50}")

    for t_val in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        n_steps = max(1, int(t_val / 0.01)) if t_val > 0 else 0
        dt = t_val / n_steps if n_steps > 0 else 0.01

        if t_val == 0:
            psi_t = psi0.copy()
        else:
            psi_t = free_particle_evolution(psi0, x, dx, dt, n_steps)

        exp_x = expectation_position(psi_t, x, dx)
        dx_val, dp_val, product = uncertainty(psi_t, x, dx)

        exp_p = expectation_momentum(psi_t, dx)
        print(f"  {t_val:6.2f} {exp_x:10.4f} {dx_val:10.4f} "
              f"{exp_p:10.4f} {product:10.4f}")


def demo_harmonic_oscillator():
    """Show quantum harmonic oscillator eigenstates."""
    print("\n" + "=" * 60)
    print("DEMO 5: Quantum Harmonic Oscillator")
    print("=" * 60)

    N = 512
    x = np.linspace(-6, 6, N)
    dx = x[1] - x[0]
    omega = 1.0
    n_states = 6

    states = harmonic_oscillator_eigenstates(x, omega, n_states)

    # Why: Energy levels E_n = ℏω(n + 1/2) are equally spaced.
    # This is the foundation of quantum field theory, where each mode
    # of a field is a harmonic oscillator, and photons/phonons are
    # the quantized excitations (n = number of quanta).
    print(f"\n  ω = {omega}, ℏ = {HBAR}")
    print(f"  E_n = ℏω(n + 1/2)")
    print(f"\n  {'n':>4} {'E_n':>10} {'⟨x⟩':>10} {'⟨x²⟩':>10} {'Norm':>10}")
    print(f"  {'─' * 46}")

    for n in range(n_states):
        psi_n = states[n]
        norm = np.sum(np.abs(psi_n) ** 2) * dx
        exp_x = np.sum(psi_n * x * psi_n) * dx
        exp_x2 = np.sum(psi_n * x ** 2 * psi_n) * dx
        energy = HBAR * omega * (n + 0.5)
        print(f"  {n:4d} {energy:10.4f} {exp_x:10.6f} {exp_x2:10.4f} "
              f"{norm:10.6f}")

    # Verify orthogonality
    print(f"\n  Orthogonality check ⟨ψ_m|ψ_n⟩:")
    print(f"  {'':>4}", end="")
    for n in range(n_states):
        print(f"  {n:>8}", end="")
    print()
    for m in range(n_states):
        print(f"  {m:>4}", end="")
        for n in range(n_states):
            overlap = np.sum(states[m] * states[n]) * dx
            print(f"  {overlap:8.4f}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   Quantum Computing - 17: Quantum Mechanics Primer        |")
    print("+" + "=" * 58 + "+")

    np.random.seed(2026)

    demo_superposition()
    demo_double_slit()
    demo_uncertainty()
    demo_free_evolution()
    demo_harmonic_oscillator()

    print("\n" + "=" * 60)
    print("All demonstrations complete.")
    print("=" * 60)
