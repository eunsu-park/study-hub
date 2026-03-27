"""
Plasma Physics - Lesson 03: Plasma Description Hierarchy
Exercise Solutions

Topics covered:
- Klimontovich to Vlasov derivation (ensemble averaging)
- Moment closure problem in fluid descriptions
- Collisionless vs collisional plasma behavior
- Phase space density conservation (Liouville/Vlasov)
- Fluid vs kinetic treatment of Landau damping
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants
e = 1.602e-19
m_e = 9.109e-31
m_p = 1.673e-27
epsilon_0 = 8.854e-12
k_B = 1.381e-23
eV_to_J = e


def exercise_1():
    """
    Exercise 1: From Klimontovich to Vlasov
    Demonstrate ensemble averaging to go from N-body to kinetic description.
    Show that the Vlasov equation emerges when correlations are neglected.
    """
    print("--- Exercise 1: Klimontovich to Vlasov ---")

    # The Klimontovich distribution: f_K(x, v, t) = sum_i delta(x - x_i(t)) * delta(v - v_i(t))
    # Ensemble average: <f_K> = f(x, v, t) (smooth distribution function)
    # Fluctuation: delta_f = f_K - f

    # The Klimontovich equation is exact:
    # df_K/dt + v * df_K/dx + (q/m)*E_K * df_K/dv = 0
    # where E_K includes the exact (microscopic) electric field

    # Ensemble averaging gives:
    # df/dt + v * df/dx + (q/m)*<E> * df/dv = -(q/m) * <delta_E * delta_f/dv>
    # The RHS is the collision term C[f]
    # Vlasov equation: set C[f] = 0 (neglect correlations)

    print("Derivation summary:")
    print("1. Start with Klimontovich equation (exact N-body):")
    print("   df_K/dt + v * df_K/dx + (q/m)*E_K * df_K/dv = 0")
    print()
    print("2. Decompose: f_K = <f_K> + delta_f = f + delta_f")
    print("   E_K = <E_K> + delta_E = E + delta_E")
    print()
    print("3. Ensemble average:")
    print("   df/dt + v*df/dx + (q/m)*E*df/dv = -(q/m)*<delta_E * d(delta_f)/dv>")
    print()
    print("4. The RHS = C[f] is the collision operator")
    print("   - Contains 2-particle correlations <delta_E * delta_f>")
    print("   - BBGKY hierarchy: 2-point -> 3-point -> ... correlations")
    print()
    print("5. Vlasov approximation: C[f] = 0")
    print("   Valid when N_D >> 1 (many particles in Debye sphere)")
    print("   Correlation effects ~ O(1/N_D)")

    # Numerical demonstration: compare N-body and Vlasov for plasma oscillation
    print("\nNumerical comparison: Plasma oscillation")

    # Simple 1D electrostatic problem
    # Vlasov prediction: omega = omega_pe
    n0 = 1e18
    T_e_eV = 10.0
    T_e = T_e_eV * eV_to_J

    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    v_th = np.sqrt(T_e / m_e)
    lambda_D = v_th / omega_pe
    N_D = (4.0 / 3.0) * np.pi * n0 * lambda_D**3

    print(f"  omega_pe = {omega_pe:.4e} rad/s")
    print(f"  v_th = {v_th:.4e} m/s")
    print(f"  N_D = {N_D:.2e}")
    print(f"  Correlation parameter 1/N_D = {1/N_D:.2e}")
    print(f"  -> Vlasov approximation is excellent for N_D >> 1")
    print()


def exercise_2():
    """
    Exercise 2: Moment Closure Problem
    Derive fluid moments from the Vlasov equation and show the closure problem.
    """
    print("--- Exercise 2: Moment Closure Problem ---")

    print("Taking velocity moments of the Vlasov equation:")
    print()
    print("0th moment (continuity): dn/dt + div(n*u) = 0")
    print("  - Involves: n (density), u (mean velocity)")
    print("  - Closed? No, depends on u from 1st moment")
    print()
    print("1st moment (momentum): m*n*(du/dt + u*grad(u)) = q*n*(E + u x B) - div(P)")
    print("  - Involves: n, u, P (pressure tensor)")
    print("  - Closed? No, depends on P from 2nd moment")
    print()
    print("2nd moment (energy): d(3/2*p)/dt + 5/2*p*div(u) + div(q) + P:grad(u) = ...")
    print("  - Involves: n, u, P, q (heat flux)")
    print("  - Closed? No, depends on q from 3rd moment")
    print()
    print("The hierarchy never closes! Each moment equation involves the next higher moment.")
    print("This is the CLOSURE PROBLEM.")
    print()

    # Common closure approximations
    print("Common closures:")
    print("  1. Cold plasma: P = 0, q = 0 (zero temperature)")
    print("  2. Isothermal: P = nkT*I, T = const, q = 0")
    print("  3. Adiabatic: d/dt(p/n^gamma) = 0, q = 0")
    print("     gamma = (N+2)/N where N = degrees of freedom")
    print("  4. CGL (Chew-Goldberger-Low): separate p_perp, p_parallel")
    print("     double adiabatic: d/dt(p_perp*B/n) = 0, d/dt(p_par*n^2/B^2) = 0")
    print()

    # Numerical example: adiabatic vs isothermal sound speed
    n0 = 1e20
    T_e_eV = 1000  # 1 keV
    T_e = T_e_eV * eV_to_J
    m_i = 2 * m_p  # Deuterium

    # Isothermal sound speed: c_s = sqrt(k_B*T_e / m_i)
    c_s_iso = np.sqrt(T_e / m_i)
    # Adiabatic sound speed: c_s = sqrt(gamma * k_B * T_e / m_i)
    gamma_1d = 3.0  # 1D adiabatic
    gamma_3d = 5.0 / 3.0  # 3D adiabatic
    c_s_ad_1d = np.sqrt(gamma_1d * T_e / m_i)
    c_s_ad_3d = np.sqrt(gamma_3d * T_e / m_i)

    print(f"Sound speed comparison (T_e = {T_e_eV} eV, deuterium):")
    print(f"  Isothermal:     c_s = {c_s_iso/1e3:.1f} km/s")
    print(f"  Adiabatic (1D): c_s = {c_s_ad_1d/1e3:.1f} km/s  (gamma = 3)")
    print(f"  Adiabatic (3D): c_s = {c_s_ad_3d/1e3:.1f} km/s  (gamma = 5/3)")
    print(f"  Choice of closure changes sound speed by factor ~{c_s_ad_1d/c_s_iso:.2f} to {c_s_ad_3d/c_s_iso:.2f}")
    print()


def exercise_3():
    """
    Exercise 3: Collisionless vs Collisional Plasma
    Compare behavior in different collisionality regimes.
    Demonstrate filamentation in collisionless case.
    """
    print("--- Exercise 3: Collisionless vs Collisional Plasma ---")

    # Phase space dynamics: collisionless vs collisional
    # Collisionless: fine-scale filamentation in phase space (phase mixing)
    # Collisional: smoothing of distribution function

    # Simple model: free streaming + optional diffusion in velocity space
    # df/dt + v * df/dx = D * d^2f/dv^2  (D = 0 for collisionless)

    print("Phase space evolution comparison:")
    print()

    # 1D free-streaming with initial perturbation
    Nx = 128
    Nv = 128
    L = 2 * np.pi
    vmax = 5.0

    dx = L / Nx
    dv = 2 * vmax / Nv
    x = np.linspace(0, L - dx, Nx)
    v = np.linspace(-vmax, vmax - dv, Nv)
    X, V = np.meshgrid(x, v, indexing='ij')

    # Initial condition: Maxwellian with sinusoidal density perturbation
    v_th = 1.0
    alpha = 0.1  # Perturbation amplitude
    k = 2 * np.pi / L
    f0 = (1 + alpha * np.cos(k * X)) * np.exp(-V**2 / (2 * v_th**2)) / np.sqrt(2 * np.pi * v_th**2)

    # After free streaming for time t: f(x, v, t) = f0(x - v*t, v)
    # This creates filamentation in (x, v) space

    t_values = [0, 1, 3, 10]
    print("  Collisionless free streaming:")
    print(f"  {'Time':>6} | {'max(f)':>10} | {'min(f)':>10} | Entropy S = -int(f*ln(f))")
    print("  " + "-" * 60)

    for t in t_values:
        # Free-streamed distribution
        x_shifted = np.mod(X - V * t, L)
        f_t = (1 + alpha * np.cos(k * x_shifted)) * np.exp(-V**2 / (2 * v_th**2)) / np.sqrt(2 * np.pi * v_th**2)

        # Entropy (should be conserved in collisionless case)
        f_positive = np.maximum(f_t, 1e-30)
        S = -np.sum(f_positive * np.log(f_positive)) * dx * dv

        print(f"  {t:>6.1f} | {np.max(f_t):>10.6f} | {np.min(f_t):>10.6f} | {S:.6f}")

    print()
    print("  Key observations:")
    print("  - Collisionless: f is conserved along characteristics (Vlasov)")
    print("    -> Fine-scale filamentation develops (phase mixing)")
    print("    -> Entropy S = -int(f*ln(f)) is exactly conserved")
    print("    -> Coarse-grained entropy increases (information moves to finer scales)")
    print()
    print("  - Collisional: velocity-space diffusion smooths filaments")
    print("    -> Distribution relaxes toward Maxwellian")
    print("    -> True entropy increases (irreversibility)")
    print("    -> Time scale: collision time tau_c")
    print()


def exercise_4():
    """
    Exercise 4: Phase Space Density Conservation
    Verify that the Vlasov equation conserves phase space density along orbits.
    Numerically demonstrate for a simple case.
    """
    print("--- Exercise 4: Phase Space Density Conservation ---")

    # The Vlasov equation: df/dt + v*df/dx + (q*E/m)*df/dv = 0
    # This is equivalent to: Df/Dt = 0 along characteristics
    # Characteristics: dx/dt = v, dv/dt = q*E/m

    # Demo: particle orbit in a harmonic potential E = -omega^2 * x
    # (equivalent to plasma oscillation in linearized limit)

    omega = 1.0  # Oscillation frequency

    def equations(t, y):
        x, v = y
        dxdt = v
        dvdt = -omega**2 * x
        return [dxdt, dvdt]

    # Track several particles starting at different (x0, v0) positions
    # Verify that f(x(t), v(t)) = f(x(0), v(0)) for all t

    # Initial distribution: Gaussian
    def f_maxwellian(x, v, x0=0, v0=0, sigma_x=1.0, sigma_v=1.0):
        return np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (v - v0)**2 / (2 * sigma_v**2))) / (
            2 * np.pi * sigma_x * sigma_v)

    print("Tracking particles in harmonic potential (plasma oscillation):")
    print("  E = -omega^2 * x, omega = 1.0")
    print()

    initial_conditions = [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (1.5, -0.3),
        (0.2, 1.2),
    ]

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 101)

    print(f"  {'Particle':>10} {'x0':>8} {'v0':>8} {'f(x0,v0)':>12} {'f(x(T),v(T))':>14} {'Conserved?':>12}")
    print("  " + "-" * 70)

    for i, (x0, v0) in enumerate(initial_conditions):
        sol = solve_ivp(equations, t_span, [x0, v0], t_eval=t_eval, method='RK45', rtol=1e-12)

        f_initial = f_maxwellian(x0, v0)
        x_final = sol.y[0, -1]
        v_final = sol.y[1, -1]
        f_final = f_maxwellian(x_final, v_final)

        # But wait - f is conserved along CHARACTERISTICS, meaning
        # f(x(t), v(t), t) = f(x(0), v(0), 0)
        # The time-dependent solution f(x, v, t) is NOT the initial Maxwellian
        # evaluated at (x(t), v(t))!

        # For the harmonic oscillator, the orbit is: x(t) = x0*cos(wt) + (v0/w)*sin(wt)
        # The distribution function at time t is f(x,v,t) evaluated by tracing back
        # Actually for this demo, we evaluate f0 at the initial point of each characteristic
        # which is f(x0, v0) by definition

        conserved = abs(f_initial - f_final) / max(f_initial, 1e-30) < 1e-6
        print(f"  {i+1:>10d} {x0:>8.2f} {v0:>8.2f} {f_initial:>12.8f} {f_final:>14.8f} "
              f"{'Yes' if conserved else 'No':>12}")

    print()
    print("  Note: f(x(t), v(t), t) = f(x_0, v_0, 0) along characteristics.")
    print("  The initial Maxwellian maps to a rotated Maxwellian at later times")
    print("  (phase space rotation by angle omega*t).")

    # Verify conservation of total particle number, momentum, and energy
    print()
    print("  Conservation laws from Vlasov equation:")
    print("  - Particle number: int f dxdv = const  (0th moment)")
    print("  - Momentum: int m*v*f dxdv = const     (1st moment, if no external force)")
    print("  - Energy: int (1/2)*m*v^2*f dxdv + field energy = const")
    print("  - Entropy: -int f*ln(f) dxdv = const   (H-theorem for collisionless)")
    print("  All conserved exactly in the Vlasov description.")
    print()


def exercise_5():
    """
    Exercise 5: Fluid vs Kinetic Treatment of Landau Damping
    Show that fluid equations cannot capture Landau damping.
    """
    print("--- Exercise 5: Fluid vs Kinetic Landau Damping ---")

    # Fluid model: linearized continuity + momentum + closure
    # dn1/dt + n0 * dv1/dx = 0
    # m*n0 * dv1/dt = -dp1/dx + q*n0*E1
    # Closure: p1 = gamma * T_e * n1 (adiabatic)

    # This gives: omega^2 = omega_pe^2 + gamma * k^2 * v_th^2
    # Real omega only! No damping in fluid model.

    # Kinetic (Vlasov) model gives:
    # omega^2 = omega_pe^2 + 3*k^2*v_th^2 - i*sqrt(pi/8) * omega_pe * (omega_pe/(k*v_th))^3 * exp(-omega_pe^2/(2*k^2*v_th^2))
    # Complex omega -> DAMPING (Landau damping)

    n0 = 1e18
    T_e_eV = 10.0
    T_e = T_e_eV * eV_to_J

    omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
    v_th = np.sqrt(T_e / m_e)
    lambda_D = v_th / omega_pe

    print("Comparing fluid and kinetic dispersion relations:")
    print()
    print(f"  Plasma parameters: n0 = {n0:.0e} m^-3, T_e = {T_e_eV} eV")
    print(f"  omega_pe = {omega_pe:.4e} rad/s")
    print(f"  v_th = {v_th:.4e} m/s")
    print(f"  lambda_D = {lambda_D:.4e} m")
    print()

    k_values = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5]) / lambda_D

    print(f"  {'k*lambda_D':>12} {'omega_fluid/wpe':>16} {'omega_kinetic/wpe':>18} "
          f"{'gamma_kinetic/wpe':>18} {'gamma/omega':>12}")
    print("  " + "-" * 80)

    for k in k_values:
        k_lam = k * lambda_D

        # Fluid (adiabatic, gamma=3 for 1D)
        omega_fluid = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)

        # Kinetic: Bohm-Gross with Landau damping
        omega_r = np.sqrt(omega_pe**2 + 3 * k**2 * v_th**2)

        # Landau damping rate (valid for k*lambda_D << 1)
        zeta = omega_r / (k * v_th * np.sqrt(2))
        gamma_landau = -np.sqrt(np.pi / 8) * omega_pe * (omega_pe / (k * v_th))**3 * \
                       np.exp(-omega_pe**2 / (2 * k**2 * v_th**2))

        print(f"  {k_lam:>12.3f} {omega_fluid/omega_pe:>16.6f} {omega_r/omega_pe:>18.6f} "
              f"{gamma_landau/omega_pe:>18.6e} {abs(gamma_landau)/omega_r:>12.6f}")

    print()
    print("  Key differences:")
    print("  1. Fluid: omega is purely real -> NO damping")
    print("     -> Cannot capture wave-particle resonance (v = omega/k)")
    print()
    print("  2. Kinetic: omega has imaginary part -> Landau DAMPING")
    print("     -> Resonant particles at v ~ omega/k exchange energy with wave")
    print("     -> Damping is exponentially sensitive to k*lambda_D")
    print()
    print("  3. Physical origin: Landau damping requires the velocity-space")
    print("     structure of f(v), which is averaged out in fluid models.")
    print("     The 'closure' approximation p = gamma*n*T destroys this information.")
    print()
    print("  4. Fluid models work well when k*lambda_D << 1 (long wavelengths)")
    print("     where Landau damping is exponentially small.")
    print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    print("All exercises completed!")
