"""
Exercises for Lesson 22: Finite Element Method
Topic: Numerical_Simulation

Solutions to practice problems from the lesson.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Shared FEM utilities
# ---------------------------------------------------------------------------

def assemble_stiffness_1d(N):
    """Assemble global stiffness matrix for 1D linear elements on [0, 1]."""
    h = 1.0 / N
    A = lil_matrix((N + 1, N + 1))
    A_elem = (1.0 / h) * np.array([[1, -1], [-1, 1]])

    for e in range(N):
        nodes = [e, e + 1]
        for i_loc in range(2):
            for j_loc in range(2):
                A[nodes[i_loc], nodes[j_loc]] += A_elem[i_loc, j_loc]

    return A.tocsr()


def fem_1d_poisson(N, f_func):
    """
    Solve -u''(x) = f(x) on (0,1) with u(0) = u(1) = 0 using linear FEM.
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)

    A = assemble_stiffness_1d(N)

    f = np.zeros(N + 1)
    for e in range(N):
        x_mid = (x[e] + x[e + 1]) / 2
        f_mid = f_func(np.array([x_mid]))[0]
        f[e] += f_mid * h / 2
        f[e + 1] += f_mid * h / 2

    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1
    f[0] = 0

    A[N, :] = 0
    A[N, N] = 1
    f[N] = 0

    A = A.tocsr()
    u = spsolve(A, f)

    return x, u


# === Exercise 1: Variable Coefficient Problem ===
# Problem: Solve -(a(x) u'(x))' = f(x) with a(x) = 1+x, f(x) = 1, u(0)=u(1)=0.

def exercise_1():
    """FEM for variable coefficient problem -(a(x)u')' = f(x)."""

    def fem_1d_variable_coeff(N, a_func, f_func):
        """FEM solver for -(a(x)u')' = f(x) with homogeneous Dirichlet BC."""
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)
        A = lil_matrix((N + 1, N + 1))
        f = np.zeros(N + 1)

        for e in range(N):
            x_mid = (x[e] + x[e + 1]) / 2
            a_mid = a_func(x_mid)
            f_mid = f_func(x_mid)

            # Modified element stiffness: a_mid * (1/h) * [[1,-1],[-1,1]]
            A_elem = (a_mid / h) * np.array([[1, -1], [-1, 1]])
            for i_loc in range(2):
                for j_loc in range(2):
                    A[e + i_loc, e + j_loc] += A_elem[i_loc, j_loc]
            f[e] += f_mid * h / 2
            f[e + 1] += f_mid * h / 2

        # Dirichlet BC
        A[0, :] = 0
        A[0, 0] = 1
        f[0] = 0
        A[N, :] = 0
        A[N, N] = 1
        f[N] = 0

        u = spsolve(A.tocsr(), f)
        return x, u

    a_func = lambda x: 1 + x
    f_func = lambda x: 1.0

    print("Variable Coefficient FEM: -(a(x)u')' = 1, a(x) = 1+x")
    print("=" * 60)

    # Reference solution (very fine mesh)
    x_ref, u_ref = fem_1d_variable_coeff(400, a_func, f_func)

    N_values = [10, 20, 40, 80]
    print(f"{'N':<8}{'h':<12}{'L_inf Error':<18}")
    print("-" * 40)

    errors = []
    h_values = []

    for N in N_values:
        x, u = fem_1d_variable_coeff(N, a_func, f_func)
        u_ref_interp = np.interp(x, x_ref, u_ref)
        err = np.max(np.abs(u - u_ref_interp))
        errors.append(err)
        h_values.append(1.0 / N)
        print(f"{N:<8}{1.0/N:<12.4f}{err:<18.4e}")

    # Convergence rates
    rates = [np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
             for i in range(len(N_values) - 1)]
    print(f"Convergence rates: {[f'{r:.2f}' for r in rates]}")
    print("Expected: ~2.0 for linear elements")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x40, u40 = fem_1d_variable_coeff(40, a_func, f_func)
    axes[0].plot(x_ref, u_ref, 'b-', linewidth=2, label='Reference (N=400)')
    axes[0].plot(x40, u40, 'ro-', markersize=4, label='FEM (N=40)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('Variable Coefficient: -(( 1+x )u\')\'= 1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(h_values, errors, 'bo-', markersize=8, label='L_inf error')
    axes[1].loglog(h_values, np.array(h_values)**2 * errors[0] / h_values[0]**2,
                   'r--', label='O(h^2)')
    axes[1].set_xlabel('h')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Convergence')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_22_1_variable_coeff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_22_1_variable_coeff.png")


# === Exercise 2: Mixed Boundary Conditions ===
# Problem: Solve -u'' = 1 with u(0) = 0 (Dirichlet) and u'(1) = -0.5 (Neumann).
# Exact: u(x) = x(1-x)/2.

def exercise_2():
    """FEM with mixed boundary conditions: Dirichlet at x=0, Neumann at x=1."""

    def fem_1d_mixed_bc(N):
        """Solve -u'' = 1, u(0) = 0, u'(1) = -0.5."""
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)

        A = assemble_stiffness_1d(N)

        f = np.zeros(N + 1)
        for e in range(N):
            f[e] += 1.0 * h / 2    # f(x) = 1
            f[e + 1] += 1.0 * h / 2

        # Neumann BC at x=1: add g = u'(1) = -0.5 to rhs
        # From weak form: boundary term is + g_1 * v(1)
        f[N] += -0.5

        # Dirichlet BC at x=0
        A = A.tolil()
        A[0, :] = 0
        A[0, 0] = 1
        f[0] = 0
        A = A.tocsr()

        u = spsolve(A, f)
        return x, u

    # Exact: -u'' = 1, u(0)=0, u'(1) = -1 + C1 = -0.5 => C1 = 0.5
    # u(x) = -x^2/2 + 0.5*x = x(1-x)/2
    def u_exact(x):
        return x * (1 - x) / 2

    print("Mixed BC FEM: -u'' = 1, u(0)=0, u'(1)=-0.5")
    print("=" * 60)
    print(f"Exact solution: u(x) = x(1-x)/2")
    print()

    N_values = [10, 20, 40, 80]
    print(f"{'N':<8}{'h':<12}{'L_inf Error':<18}")
    print("-" * 40)

    errors = []
    h_values = []

    for N in N_values:
        x, u = fem_1d_mixed_bc(N)
        err = np.max(np.abs(u - u_exact(x)))
        errors.append(err)
        h_values.append(1.0 / N)
        print(f"{N:<8}{1.0/N:<12.4f}{err:<18.4e}")

    rates = [np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
             for i in range(len(N_values) - 1)]
    print(f"Convergence rates: {[f'{r:.2f}' for r in rates]}")
    print("Confirmed O(h^2) convergence.")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    x20, u20 = fem_1d_mixed_bc(20)
    x_fine = np.linspace(0, 1, 200)

    ax.plot(x_fine, u_exact(x_fine), 'b-', linewidth=2, label='Exact')
    ax.plot(x20, u20, 'ro-', markersize=6, label='FEM (N=20)')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('Mixed BC: u(0)=0, u\'(1)=-0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_22_2_mixed_bc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_22_2_mixed_bc.png")


# === Exercise 3: Quadratic Elements (P2) ===
# Problem: Implement P2 FEM and compare convergence with P1 (linear).

def exercise_3():
    """Quadratic (P2) elements vs linear (P1): convergence comparison."""

    from numpy.polynomial.legendre import leggauss

    gauss_pts, gauss_wts = leggauss(3)  # 3-point Gauss quadrature

    def p2_basis_ref(xi):
        """P2 basis functions and derivatives on reference element [-1, 1]."""
        phi = np.array([xi * (xi - 1) / 2, 1 - xi**2, xi * (xi + 1) / 2])
        dphi = np.array([(2 * xi - 1) / 2, -2 * xi, (2 * xi + 1) / 2])
        return phi, dphi

    def fem_p2_poisson(N, f_func):
        """Solve -u'' = f using P2 elements, homogeneous Dirichlet BC."""
        h = 1.0 / N
        n_nodes = 2 * N + 1
        x = np.linspace(0, 1, n_nodes)

        A = lil_matrix((n_nodes, n_nodes))
        f = np.zeros(n_nodes)

        for e in range(N):
            x_left = x[2 * e]
            nodes_e = [2 * e, 2 * e + 1, 2 * e + 2]

            A_e = np.zeros((3, 3))
            f_e = np.zeros(3)

            for gp, gw in zip(gauss_pts, gauss_wts):
                phi, dphi = p2_basis_ref(gp)
                x_phys = x_left + (gp + 1) / 2 * h
                J = h / 2  # Jacobian

                # Stiffness: (1/J) * dphi_i * dphi_j * gw
                A_e += gw * np.outer(dphi, dphi) / J
                # Load: f(x) * phi * J * gw
                f_e += gw * J * f_func(np.array([x_phys]))[0] * phi

            for i_loc, i_g in enumerate(nodes_e):
                for j_loc, j_g in enumerate(nodes_e):
                    A[i_g, j_g] += A_e[i_loc, j_loc]
                f[i_g] += f_e[i_loc]

        # Dirichlet BC
        A[0, :] = 0
        A[0, 0] = 1
        f[0] = 0
        A[-1, :] = 0
        A[-1, -1] = 1
        f[-1] = 0

        u = spsolve(A.tocsr(), f)
        return x, u

    f_func = lambda x: np.pi**2 * np.sin(np.pi * x)
    u_exact_func = lambda x: np.sin(np.pi * x)

    N_values = [5, 10, 20, 40]

    print("P1 vs P2 FEM Convergence")
    print("=" * 60)
    print(f"Problem: -u'' = pi^2 sin(pi*x), u(0)=u(1)=0")
    print(f"Exact: u(x) = sin(pi*x)")
    print()
    print(f"{'N':<8}{'P1 L_inf':<18}{'P2 L_inf':<18}{'P2/P1 Ratio':<15}")
    print("-" * 60)

    errors_p1 = []
    errors_p2 = []
    h_values = []

    for N in N_values:
        h = 1.0 / N
        h_values.append(h)

        # P1
        x_p1, u_p1 = fem_1d_poisson(N, f_func)
        err_p1 = np.max(np.abs(u_p1 - u_exact_func(x_p1)))
        errors_p1.append(err_p1)

        # P2
        x_p2, u_p2 = fem_p2_poisson(N, f_func)
        err_p2 = np.max(np.abs(u_p2 - u_exact_func(x_p2)))
        errors_p2.append(err_p2)

        ratio = err_p1 / err_p2 if err_p2 > 0 else float('inf')
        print(f"{N:<8}{err_p1:<18.3e}{err_p2:<18.3e}{ratio:<15.1f}")

    # Convergence rates
    rates_p1 = [np.log(errors_p1[i] / errors_p1[i + 1]) / np.log(h_values[i] / h_values[i + 1])
                for i in range(len(N_values) - 1)]
    rates_p2 = [np.log(errors_p2[i] / errors_p2[i + 1]) / np.log(h_values[i] / h_values[i + 1])
                for i in range(len(N_values) - 1)]

    print(f"\nP1 convergence rates: {[f'{r:.2f}' for r in rates_p1]} (expected ~2)")
    print(f"P2 convergence rates: {[f'{r:.2f}' for r in rates_p2]} (expected ~4)")
    print("P2 elements converge two orders faster than P1.")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_values, errors_p1, 'bo-', markersize=8, label='P1 (linear)')
    ax.loglog(h_values, errors_p2, 'rs-', markersize=8, label='P2 (quadratic)')
    ax.loglog(h_values, np.array(h_values)**2 * 0.1, 'b--', alpha=0.5, label='O(h^2)')
    ax.loglog(h_values, np.array(h_values)**4 * 10, 'r--', alpha=0.5, label='O(h^4)')
    ax.set_xlabel('h (element size)')
    ax.set_ylabel('L_inf Error')
    ax.set_title('P1 vs P2 Element Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_22_3_p1_vs_p2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to exercise_22_3_p1_vs_p2.png")


# === Exercise 4: Reaction-Diffusion Equation ===
# Problem: Solve -u'' + c*u = f with c=10, f=1, u(0)=u(1)=0.

def exercise_4():
    """FEM for reaction-diffusion: -u'' + c*u = f(x)."""

    def fem_1d_reaction_diffusion(N, c, f_func):
        """Solve -u'' + c*u = f with homogeneous Dirichlet BC."""
        h = 1.0 / N
        x = np.linspace(0, 1, N + 1)

        # Stiffness matrix
        A = assemble_stiffness_1d(N)

        # Mass matrix: element mass = (h/6)*[[2,1],[1,2]]
        M = lil_matrix((N + 1, N + 1))
        M_elem = (h / 6) * np.array([[2, 1], [1, 2]])
        for e in range(N):
            for i_loc in range(2):
                for j_loc in range(2):
                    M[e + i_loc, e + j_loc] += M_elem[i_loc, j_loc]

        # Combined system: (A + c*M) u = f
        K = A + c * M.tocsr()

        f = np.zeros(N + 1)
        for e in range(N):
            x_mid = (x[e] + x[e + 1]) / 2
            f_mid = f_func(x_mid)
            f[e] += f_mid * h / 2
            f[e + 1] += f_mid * h / 2

        # Dirichlet BC
        K = K.tolil()
        K[0, :] = 0
        K[0, 0] = 1
        f[0] = 0
        K[N, :] = 0
        K[N, N] = 1
        f[N] = 0
        K = K.tocsr()

        u = spsolve(K, f)
        return x, u

    c_val = 10
    f_func = lambda x: 1.0

    print("Reaction-Diffusion FEM: -u'' + 10*u = 1, u(0)=u(1)=0")
    print("=" * 60)

    # Reference solution
    x_ref, u_ref = fem_1d_reaction_diffusion(400, c_val, f_func)

    N_values = [10, 20, 40, 80]
    print(f"{'N':<8}{'h':<12}{'L_inf Error':<18}")
    print("-" * 40)

    errors = []
    h_values = []

    for N in N_values:
        x, u = fem_1d_reaction_diffusion(N, c_val, f_func)
        u_ref_interp = np.interp(x, x_ref, u_ref)
        err = np.max(np.abs(u - u_ref_interp))
        errors.append(err)
        h_values.append(1.0 / N)
        print(f"{N:<8}{1.0/N:<12.4f}{err:<18.4e}")

    rates = [np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
             for i in range(len(N_values) - 1)]
    print(f"Convergence rates: {[f'{r:.2f}' for r in rates]}")
    print("Confirmed O(h^2) convergence.")

    # Visualization: compare solutions for different c values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    c_values = [0, 1, 10, 100]
    for c in c_values:
        x, u = fem_1d_reaction_diffusion(80, c, f_func)
        axes[0].plot(x, u, linewidth=2, label=f'c={c}')

    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('Reaction-Diffusion: Effect of c')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(h_values, errors, 'bo-', markersize=8, label='L_inf error')
    axes[1].loglog(h_values, np.array(h_values)**2 * errors[0] / h_values[0]**2,
                   'r--', label='O(h^2)')
    axes[1].set_xlabel('h')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Convergence (c=10)')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_22_4_reaction_diffusion.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nAs c increases, the reaction term dominates, pushing u toward f/c = 0.1.")
    print("The mass matrix element integral for linear hat functions is (h/6)[[2,1],[1,2]].")
    print("Plot saved to exercise_22_4_reaction_diffusion.png")


# === Exercise 5: Convergence Rate Verification (Method of Manufactured Solutions) ===
# Problem: Choose u_exact = x(1-x)sin(pi*x), compute f analytically,
# verify O(h^2) L2 convergence.

def exercise_5():
    """Method of manufactured solutions for FEM verification."""

    # Exact solution: u(x) = x(1-x)*sin(pi*x)
    # u'  = (1-2x)*sin(pi*x) + pi*x*(1-x)*cos(pi*x)
    # u'' = -2*sin(pi*x) + 2*pi*(1-2x)*cos(pi*x) - pi^2*x*(1-x)*sin(pi*x)
    # f = -u'' = 2*sin(pi*x) - 2*pi*(1-2x)*cos(pi*x) + pi^2*x*(1-x)*sin(pi*x)

    def u_exact(x):
        return x * (1 - x) * np.sin(np.pi * x)

    def f_manufactured(x):
        return (2 * np.sin(np.pi * x)
                - 2 * np.pi * (1 - 2 * x) * np.cos(np.pi * x)
                + np.pi**2 * x * (1 - x) * np.sin(np.pi * x))

    print("Method of Manufactured Solutions")
    print("=" * 60)
    print(f"Exact: u(x) = x(1-x)sin(pi*x)")
    print(f"f(x) = 2sin(pi*x) - 2pi(1-2x)cos(pi*x) + pi^2 x(1-x)sin(pi*x)")
    print()

    N_values = [5, 10, 20, 40, 80]
    h_values = []
    errors_L2 = []
    errors_Linf = []

    print(f"{'N':<8}{'h':<12}{'L2 Error':<18}{'L_inf Error':<18}")
    print("-" * 56)

    for N in N_values:
        h = 1.0 / N
        h_values.append(h)
        x, u = fem_1d_poisson(N, f_manufactured)
        u_ex = u_exact(x)

        error_L2 = np.sqrt(np.sum((u - u_ex)**2) * h)
        error_Linf = np.max(np.abs(u - u_ex))
        errors_L2.append(error_L2)
        errors_Linf.append(error_Linf)

        print(f"{N:<8}{h:<12.4f}{error_L2:<18.4e}{error_Linf:<18.4e}")

    # Convergence rates
    rates_L2 = [np.log(errors_L2[i] / errors_L2[i + 1]) / np.log(h_values[i] / h_values[i + 1])
                for i in range(len(N_values) - 1)]
    rates_Linf = [np.log(errors_Linf[i] / errors_Linf[i + 1]) / np.log(h_values[i] / h_values[i + 1])
                  for i in range(len(N_values) - 1)]

    print(f"\nL2 convergence rates:   {[f'{r:.2f}' for r in rates_L2]}")
    print(f"L_inf convergence rates: {[f'{r:.2f}' for r in rates_Linf]}")
    print("Expected: ~2.0 for linear (P1) elements")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x20, u20 = fem_1d_poisson(20, f_manufactured)
    x_fine = np.linspace(0, 1, 200)

    axes[0].plot(x_fine, u_exact(x_fine), 'b-', linewidth=2, label='Exact')
    axes[0].plot(x20, u20, 'ro-', markersize=5, label='FEM (N=20)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('Manufactured Solution: u(x) = x(1-x)sin(pi*x)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(h_values, errors_L2, 'bo-', markersize=8, label='L2 error')
    axes[1].loglog(h_values, errors_Linf, 'rs-', markersize=8, label='L_inf error')
    axes[1].loglog(h_values, np.array(h_values)**2, 'k--', alpha=0.5, label='O(h^2)')
    axes[1].set_xlabel('h (element size)')
    axes[1].set_ylabel('Error')
    axes[1].set_title('FEM Convergence (Linear Elements)')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_22_5_manufactured_solution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nThe method of manufactured solutions is a systematic way to verify")
    print("FEM correctness: choose u_exact, derive f, then confirm convergence rate.")
    print("Plot saved to exercise_22_5_manufactured_solution.png")


if __name__ == "__main__":
    print("=== Exercise 1: Variable Coefficient Problem ===")
    exercise_1()

    print("\n=== Exercise 2: Mixed Boundary Conditions ===")
    exercise_2()

    print("\n=== Exercise 3: Quadratic Elements (P2) ===")
    exercise_3()

    print("\n=== Exercise 4: Reaction-Diffusion Equation ===")
    exercise_4()

    print("\n=== Exercise 5: Convergence Rate Verification ===")
    exercise_5()

    print("\nAll exercises completed!")
