# Robot Dynamics

[← Previous: Velocity Kinematics](05_Velocity_Kinematics.md) | [Next: Motion Planning →](07_Motion_Planning.md)

## Learning Objectives

1. Explain why dynamics matters for robot control — the critical step beyond kinematics
2. Derive the equations of motion using the Euler-Lagrange formulation with kinetic energy, potential energy, and the Lagrangian
3. Identify and interpret each term in the standard manipulator dynamics equation: $M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
4. Describe the properties of the inertia matrix and explain the physical origin of Coriolis and centrifugal terms
5. Implement gravity compensation and computed torque control
6. Compare the Euler-Lagrange and Newton-Euler formulations and understand when to use each

---

## Why This Matters

Kinematics tells you *where* the robot can go. Dynamics tells you *how hard you must push* to get there. Without dynamics, you can plan a beautiful path through space but have no idea what motor torques are needed to follow it. The result would be a robot that overshoots, oscillates, or collapses under its own weight.

Dynamics is the bridge between geometry and control. When a robot arm swings joint 1, the centrifugal effects accelerate joint 2 (coupling). When the arm is extended horizontally, gravity creates a torque that the motors must constantly fight. When the arm moves quickly, velocity-dependent forces (Coriolis effects) create unexpected forces that push the end-effector off course. Understanding these effects — and computing them accurately — is essential for high-performance motion control.

> **Analogy**: Dynamics is like understanding not just "where to steer" but "how hard to push the gas pedal." A kinematics-only controller is like driving with a map but no sense of the car's weight, the slope of the road, or the wind resistance. You know the direction, but you have no idea how much throttle to apply.

---

## From Kinematics to Dynamics

### What Kinematics Cannot Tell You

Kinematics gives us:
- **FK**: Joint angles $\to$ end-effector pose
- **IK**: Desired pose $\to$ joint angles
- **Jacobian**: Joint velocities $\to$ end-effector velocity

But it cannot answer: **What torques must the motors apply to achieve a desired joint acceleration?**

This requires dynamics — the relationship between forces/torques and motion, governed by Newton's laws.

### The Manipulator Dynamics Equation

The standard form of the manipulator equations of motion:

$$M(\mathbf{q}) \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

where:
| Term | Symbol | Dimension | Physical Meaning |
|------|--------|-----------|------------------|
| Inertia matrix | $M(\mathbf{q})$ | $n \times n$ | Mass and rotational inertia (configuration-dependent) |
| Coriolis/centrifugal | $C(\mathbf{q}, \dot{\mathbf{q}})$ | $n \times n$ | Velocity-dependent forces (coupling between joints) |
| Gravity | $\mathbf{g}(\mathbf{q})$ | $n \times 1$ | Gravitational torques at each joint |
| Joint torques | $\boldsymbol{\tau}$ | $n \times 1$ | Applied motor torques (the control input) |

---

## The Euler-Lagrange Formulation

### The Lagrangian

The Lagrangian is defined as the difference between kinetic and potential energy:

$$\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = K(\mathbf{q}, \dot{\mathbf{q}}) - P(\mathbf{q})$$

where $K$ is the total kinetic energy and $P$ is the total potential energy of the manipulator.

### The Euler-Lagrange Equations

The equations of motion for the $i$-th joint:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i} = \tau_i, \quad i = 1, \ldots, n$$

These are derived from the principle of least action — the robot's trajectory minimizes the time integral of the Lagrangian. This is a powerful framework because it works in any coordinate system (joint angles, task coordinates, etc.) without needing to compute constraint forces.

```python
import numpy as np
from numpy import sin, cos

# Symbolic derivation for a single pendulum (1-DOF)
# to illustrate the Euler-Lagrange procedure

def single_pendulum_dynamics(theta, theta_dot, m, l, g=9.81):
    """Dynamics of a simple pendulum (1-DOF robot arm).

    Why start with a pendulum? It's the simplest possible robot arm
    (1 link, 1 joint). The derivation illustrates the complete
    Euler-Lagrange procedure without algebraic complexity.

    Step-by-step:
    1. Kinetic energy: K = (1/2) * I * theta_dot^2
       where I = m*l^2 (point mass at end of massless rod)
    2. Potential energy: P = m*g*l*(1 - cos(theta))
       (zero at the bottom, theta=0 = hanging down)
    3. Lagrangian: L = K - P
    4. Euler-Lagrange: d/dt(dL/d_theta_dot) - dL/d_theta = tau

    Result: m*l^2 * theta_ddot + m*g*l*sin(theta) = tau
    """
    I = m * l**2        # moment of inertia
    M = I               # 1x1 "inertia matrix"
    g_term = m * g * l * sin(theta)  # gravity torque

    return M, g_term
    # The equation of motion: M * theta_ddot + g_term = tau
    # Or: theta_ddot = (tau - g_term) / M

# Example: pendulum parameters
m, l = 2.0, 0.5  # 2 kg mass, 0.5 m rod
theta = np.radians(45)
theta_dot = 0.0

M, g_term = single_pendulum_dynamics(theta, theta_dot, m, l)
print(f"Inertia: {M:.4f} kg*m^2")
print(f"Gravity torque at 45 deg: {g_term:.4f} N*m")
print(f"To hold position (tau = g_term): {g_term:.4f} N*m")
```

---

## Kinetic and Potential Energy for a Manipulator

### Kinetic Energy

For link $i$ with mass $m_i$, center of mass velocity $\mathbf{v}_{c_i}$, angular velocity $\boldsymbol{\omega}_i$, and inertia tensor $I_{c_i}$ (about its center of mass):

$$K_i = \frac{1}{2} m_i \mathbf{v}_{c_i}^T \mathbf{v}_{c_i} + \frac{1}{2} \boldsymbol{\omega}_i^T I_{c_i} \boldsymbol{\omega}_i$$

The total kinetic energy is:

$$K = \sum_{i=1}^{n} K_i$$

Using the Jacobian, we can express $\mathbf{v}_{c_i}$ and $\boldsymbol{\omega}_i$ in terms of joint velocities:

$$\mathbf{v}_{c_i} = J_{v_i}(\mathbf{q}) \dot{\mathbf{q}}, \quad \boldsymbol{\omega}_i = J_{\omega_i}(\mathbf{q}) \dot{\mathbf{q}}$$

where $J_{v_i}$ and $J_{\omega_i}$ are the Jacobians for the center of mass and angular velocity of link $i$.

The total kinetic energy becomes:

$$K = \frac{1}{2} \dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$$

where the **inertia matrix** is:

$$M(\mathbf{q}) = \sum_{i=1}^{n} \left[ m_i J_{v_i}^T J_{v_i} + J_{\omega_i}^T I_{c_i} J_{\omega_i} \right]$$

### Potential Energy

$$P = \sum_{i=1}^{n} m_i \mathbf{g}^T \mathbf{p}_{c_i}(\mathbf{q})$$

where $\mathbf{g} = (0, 0, -g)^T$ is the gravity vector and $\mathbf{p}_{c_i}$ is the center of mass position of link $i$.

```python
def compute_link_energy(m, v_cm, omega, I_cm, p_cm, g_vec):
    """Compute kinetic and potential energy for a single link.

    Why separate per-link? Because each link has different mass,
    inertia, and position. The total energy is the sum over all links.
    This modular approach scales to any number of links.
    """
    # Kinetic: translational + rotational
    K_trans = 0.5 * m * np.dot(v_cm, v_cm)
    K_rot = 0.5 * np.dot(omega, I_cm @ omega)
    K = K_trans + K_rot

    # Potential (gravitational)
    P = -m * np.dot(g_vec, p_cm)  # P = -m*g*h (with g pointing down)

    return K, P
```

---

## The 2-Link Planar Arm: Complete Derivation

This is the most important worked example in robot dynamics. We derive every term explicitly.

### Setup

- Link 1: length $l_1$, mass $m_1$, center of mass at distance $l_{c_1}$ from joint 1, inertia $I_1$
- Link 2: length $l_2$, mass $m_2$, center of mass at distance $l_{c_2}$ from joint 2, inertia $I_2$
- Gravity $g$ acts in the $-y$ direction

### Center of Mass Positions

$$x_{c_1} = l_{c_1} \cos\theta_1, \quad y_{c_1} = l_{c_1} \sin\theta_1$$

$$x_{c_2} = l_1 \cos\theta_1 + l_{c_2} \cos(\theta_1 + \theta_2)$$
$$y_{c_2} = l_1 \sin\theta_1 + l_{c_2} \sin(\theta_1 + \theta_2)$$

### Center of Mass Velocities

$$\dot{x}_{c_1} = -l_{c_1} \sin\theta_1 \, \dot{\theta}_1$$
$$\dot{y}_{c_1} = l_{c_1} \cos\theta_1 \, \dot{\theta}_1$$

$$\dot{x}_{c_2} = -l_1 \sin\theta_1 \, \dot{\theta}_1 - l_{c_2} \sin(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)$$
$$\dot{y}_{c_2} = l_1 \cos\theta_1 \, \dot{\theta}_1 + l_{c_2} \cos(\theta_1 + \theta_2)(\dot{\theta}_1 + \dot{\theta}_2)$$

### Kinetic Energy

$$K = \frac{1}{2}(m_1 l_{c_1}^2 + I_1)\dot{\theta}_1^2 + \frac{1}{2}m_2\left[l_1^2 \dot{\theta}_1^2 + l_{c_2}^2(\dot{\theta}_1 + \dot{\theta}_2)^2 + 2 l_1 l_{c_2} \cos\theta_2 \, \dot{\theta}_1(\dot{\theta}_1 + \dot{\theta}_2)\right] + \frac{1}{2}I_2(\dot{\theta}_1 + \dot{\theta}_2)^2$$

This can be written as:

$$K = \frac{1}{2} \dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}}$$

### Inertia Matrix

$$M(\mathbf{q}) = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix}$$

where:
$$M_{11} = m_1 l_{c_1}^2 + m_2(l_1^2 + l_{c_2}^2 + 2l_1 l_{c_2}\cos\theta_2) + I_1 + I_2$$
$$M_{12} = M_{21} = m_2(l_{c_2}^2 + l_1 l_{c_2}\cos\theta_2) + I_2$$
$$M_{22} = m_2 l_{c_2}^2 + I_2$$

### Coriolis and Centrifugal Terms

Using Christoffel symbols $c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{kj}}{\partial q_i} + \frac{\partial M_{ki}}{\partial q_j} - \frac{\partial M_{ij}}{\partial q_k}\right)$:

$$C(\mathbf{q}, \dot{\mathbf{q}}) = \begin{bmatrix} -m_2 l_1 l_{c_2} \sin\theta_2 \, \dot{\theta}_2 & -m_2 l_1 l_{c_2} \sin\theta_2 (\dot{\theta}_1 + \dot{\theta}_2) \\ m_2 l_1 l_{c_2} \sin\theta_2 \, \dot{\theta}_1 & 0 \end{bmatrix}$$

Let $h = m_2 l_1 l_{c_2} \sin\theta_2$. Then:

$$C \dot{\mathbf{q}} = \begin{bmatrix} -h \dot{\theta}_2 \dot{\theta}_1 - h(\dot{\theta}_1 + \dot{\theta}_2)\dot{\theta}_2 \\ h \dot{\theta}_1^2 \end{bmatrix}$$

The first row contains the Coriolis term ($-2h\dot{\theta}_1\dot{\theta}_2$) and the centrifugal term ($-h\dot{\theta}_2^2$). The second row is a centrifugal term from joint 1 affecting joint 2.

### Gravity Terms

$$\mathbf{g}(\mathbf{q}) = \begin{bmatrix} (m_1 l_{c_1} + m_2 l_1)g\cos\theta_1 + m_2 l_{c_2} g\cos(\theta_1 + \theta_2) \\ m_2 l_{c_2} g\cos(\theta_1 + \theta_2) \end{bmatrix}$$

```python
class TwoLinkDynamics:
    """Complete dynamics for a 2-link planar robot arm.

    This is the workhorse example for understanding manipulator dynamics.
    Every concept — inertia, Coriolis, gravity — is visible and verifiable.
    """
    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, g=9.81):
        self.m1 = m1    # link 1 mass
        self.m2 = m2    # link 2 mass
        self.l1 = l1    # link 1 length
        self.l2 = l2    # link 2 length
        self.lc1 = lc1  # link 1 center of mass distance from joint 1
        self.lc2 = lc2  # link 2 center of mass distance from joint 2
        self.I1 = I1    # link 1 rotational inertia about its CM
        self.I2 = I2    # link 2 rotational inertia about its CM
        self.g = g

    def inertia_matrix(self, q):
        """Compute the 2x2 inertia matrix M(q).

        Why is M configuration-dependent? Because the effective inertia
        'seen' by each joint motor depends on how the links are arranged.
        When link 2 is extended, joint 1 sees more inertia (longer moment arm).
        When link 2 is folded, joint 1 sees less inertia.
        """
        t2 = q[1]
        c2 = cos(t2)

        a = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 + \
            2*self.l1*self.lc2*c2) + self.I1 + self.I2
        b = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        d = self.m2*self.lc2**2 + self.I2

        M = np.array([[a, b],
                       [b, d]])
        return M

    def coriolis_matrix(self, q, q_dot):
        """Compute the 2x2 Coriolis/centrifugal matrix C(q, q_dot).

        Why do Coriolis terms exist? When joint 1 rotates, link 2's
        center of mass follows a curved path. This curved motion creates
        centripetal acceleration, which appears as a 'phantom force' in
        the joint-space equations. It's not a real external force — it's
        an artifact of using rotating (non-inertial) reference frames.
        """
        t2 = q[1]
        t1_dot, t2_dot = q_dot

        h = self.m2 * self.l1 * self.lc2 * sin(t2)

        C = np.array([[-h*t2_dot,    -h*(t1_dot + t2_dot)],
                       [ h*t1_dot,    0                    ]])
        return C

    def gravity_vector(self, q):
        """Compute the 2x1 gravity vector g(q).

        Why does gravity depend on q? Because the gravitational torque
        on each joint depends on the horizontal distance of each link's
        center of mass from that joint. As the arm moves, these distances
        change. At theta=0 (horizontal), gravity torque is maximum;
        at theta=90 (vertical), it's zero.
        """
        t1 = q[0]
        t12 = q[0] + q[1]
        g = self.g

        g1 = (self.m1*self.lc1 + self.m2*self.l1)*g*cos(t1) + \
              self.m2*self.lc2*g*cos(t12)
        g2 = self.m2*self.lc2*g*cos(t12)

        return np.array([g1, g2])

    def forward_dynamics(self, q, q_dot, tau):
        """Compute joint accelerations given torques.

        tau = M*q_ddot + C*q_dot + g
        =>  q_ddot = M^{-1} * (tau - C*q_dot - g)

        This is the 'forward dynamics' problem: given forces, find motion.
        Used in simulation.
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        g = self.gravity_vector(q)

        q_ddot = np.linalg.solve(M, tau - C @ q_dot - g)
        return q_ddot

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """Compute required torques for desired motion.

        tau = M*q_ddot + C*q_dot + g

        This is the 'inverse dynamics' problem: given desired motion,
        find required forces. Used in computed torque control.
        """
        M = self.inertia_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        g = self.gravity_vector(q)

        tau = M @ q_ddot + C @ q_dot + g
        return tau


# Create a 2-link arm
arm = TwoLinkDynamics(
    m1=5.0, m2=3.0,       # masses in kg
    l1=0.5, l2=0.4,       # link lengths in m
    lc1=0.25, lc2=0.2,    # center of mass distances
    I1=0.1, I2=0.05       # rotational inertias in kg*m^2
)

# Configuration: both links at 45 degrees
q = np.radians([45, 30])
q_dot = np.array([0.5, -0.3])  # some joint velocities

print("=== 2-Link Arm Dynamics ===")
print(f"Configuration: q = [{np.degrees(q[0]):.0f}, {np.degrees(q[1]):.0f}] deg")
print(f"Velocities: q_dot = {q_dot}")

M = arm.inertia_matrix(q)
C = arm.coriolis_matrix(q, q_dot)
g = arm.gravity_vector(q)

print(f"\nInertia matrix M:")
print(np.round(M, 4))
print(f"\nCoriolis matrix C:")
print(np.round(C, 4))
print(f"\nGravity vector g: {np.round(g, 4)}")

# What torques are needed to hold position (q_ddot = 0, q_dot = 0)?
tau_static = arm.inverse_dynamics(q, np.zeros(2), np.zeros(2))
print(f"\nTorques to hold position: {np.round(tau_static, 4)} N*m")
print(f"(These equal the gravity vector — only gravity acts when stationary)")
```

---

## Inertia Matrix Properties

The inertia matrix $M(\mathbf{q})$ has several important properties:

### 1. Symmetric

$M(\mathbf{q}) = M(\mathbf{q})^T$ always. This follows from the definition as a quadratic form of kinetic energy.

### 2. Positive Definite

$\dot{\mathbf{q}}^T M(\mathbf{q}) \dot{\mathbf{q}} > 0$ for all $\dot{\mathbf{q}} \neq 0$. This means kinetic energy is always positive (unless the robot is stationary).

### 3. Configuration-Dependent

$M$ changes as the robot moves. At some configurations, certain joints see more inertia than at others.

### 4. Bounded

$\lambda_{min}(M) \leq \frac{\dot{\mathbf{q}}^T M \dot{\mathbf{q}}}{\|\dot{\mathbf{q}}\|^2} \leq \lambda_{max}(M)$

The eigenvalues of $M$ are bounded above and below for bounded joint configurations. This is important for control design.

```python
def verify_inertia_properties(arm, n_samples=100):
    """Verify key properties of the inertia matrix.

    Why verify? Because analytical derivations can have sign errors
    or missing terms. Verifying properties catches these bugs —
    a non-positive-definite M means something is wrong in the derivation.
    """
    print("Verifying inertia matrix properties:")

    for _ in range(n_samples):
        q = np.random.uniform(-np.pi, np.pi, 2)
        M = arm.inertia_matrix(q)

        # Symmetry
        assert np.allclose(M, M.T), f"M not symmetric at q={q}"

        # Positive definite (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), f"M not positive definite at q={q}, eigs={eigvals}"

    print(f"  Symmetry: PASS ({n_samples} random configs)")
    print(f"  Positive definiteness: PASS ({n_samples} random configs)")

    # Show how M varies with configuration
    print("\nInertia variation with theta2:")
    for t2_deg in [0, 30, 60, 90, 120, 150, 180]:
        q = np.array([0, np.radians(t2_deg)])
        M = arm.inertia_matrix(q)
        print(f"  theta2={t2_deg:>3d} deg: M11={M[0,0]:.4f}, "
              f"M12={M[0,1]:.4f}, M22={M[1,1]:.4f}")

verify_inertia_properties(arm)
```

---

## Coriolis and Centrifugal Effects

### Physical Origin

**Centrifugal effect**: When joint $i$ rotates, the "outward" pseudo-force on link $i+1$ (and beyond) due to circular motion. Proportional to $\dot{q}_i^2$.

**Coriolis effect**: Cross-coupling between two joints' velocities. When joints $i$ and $j$ both move, there is a force proportional to $\dot{q}_i \dot{q}_j$ on other joints.

### The Skew-Symmetry Property

A fundamental property used in control theory:

$$\dot{M}(\mathbf{q}) - 2C(\mathbf{q}, \dot{\mathbf{q}})$$

is skew-symmetric. This means:

$$\dot{\mathbf{q}}^T [\dot{M} - 2C] \dot{\mathbf{q}} = 0$$

This property is essential for proving the stability of many robot controllers (e.g., computed torque control, adaptive control).

```python
def verify_skew_symmetry(arm, q, q_dot, delta=1e-7):
    """Verify the skew-symmetry property: M_dot - 2C is skew-symmetric.

    Why is this important? Because it implies that the 'power' of the
    Coriolis/centrifugal forces is zero: these forces do no work.
    They only redirect energy between joints, never create or destroy it.
    This is the robotic equivalent of the fact that the Coriolis force
    in rotating frames does no work (it's always perpendicular to velocity).

    This property is used in Lyapunov stability proofs for controllers.
    """
    # Numerical M_dot via finite differences
    M = arm.inertia_matrix(q)
    q_shifted = q + delta * q_dot
    M_shifted = arm.inertia_matrix(q_shifted)
    M_dot = (M_shifted - M) / delta

    C = arm.coriolis_matrix(q, q_dot)

    S = M_dot - 2 * C

    # Check skew-symmetry: S + S^T should be zero
    is_skew = np.allclose(S + S.T, 0, atol=1e-4)
    print(f"M_dot - 2C:")
    print(np.round(S, 6))
    print(f"Skew-symmetric? {is_skew}")

    # Verify: q_dot^T * S * q_dot = 0
    power = q_dot @ S @ q_dot
    print(f"q_dot^T * S * q_dot = {power:.2e} (should be ~0)")

verify_skew_symmetry(arm, np.radians([45, 30]), np.array([1.0, -0.5]))
```

---

## Gravity Compensation

### The Simplest Dynamic Controller

In many applications, the robot needs to hold a position or follow slow trajectories. In these cases, the dominant dynamic effect is gravity. **Gravity compensation** is the simplest and most impactful improvement over pure kinematic control:

$$\boldsymbol{\tau} = \mathbf{g}(\mathbf{q}) + K_p (\mathbf{q}_d - \mathbf{q}) + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}})$$

This is a PD controller with gravity feedforward. Without gravity compensation, the PD controller must generate large torques just to fight gravity, leaving less headroom for tracking.

```python
def gravity_compensation_controller(arm, q, q_dot, q_desired, q_dot_desired,
                                    Kp, Kd):
    """PD controller with gravity compensation.

    Why add gravity compensation? Consider holding the arm horizontal.
    Without it, the PD controller must generate a constant error-correcting
    torque equal to the gravitational torque. This means the arm hangs
    below the desired position (steady-state error for P control) or
    requires very high gains (which cause oscillation).

    Gravity compensation eliminates this: the feedforward term handles
    gravity exactly, and the PD terms only need to handle dynamics.
    """
    g = arm.gravity_vector(q)

    # PD control
    tau_pd = Kp @ (q_desired - q) + Kd @ (q_dot_desired - q_dot)

    # Total torque = gravity compensation + PD
    tau = g + tau_pd

    return tau

# Simulate holding a position with and without gravity compensation
def simulate(arm, controller, q0, q_desired, dt=0.001, duration=2.0):
    """Simple Euler integration simulation.

    Why simulate? Because it reveals the actual behavior of the
    controller, including transient response, steady-state error,
    and stability issues that analysis alone might miss.
    """
    n_steps = int(duration / dt)
    q = q0.copy()
    q_dot = np.zeros(2)
    trajectory = [q.copy()]

    for _ in range(n_steps):
        tau = controller(q, q_dot, q_desired)
        q_ddot = arm.forward_dynamics(q, q_dot, tau)

        # Euler integration
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

        trajectory.append(q.copy())

    return np.array(trajectory)

# Controller without gravity compensation
Kp = np.diag([50, 30])
Kd = np.diag([10, 5])
q_desired = np.radians([45, 30])

def ctrl_pd_only(q, q_dot, q_des):
    return Kp @ (q_des - q) + Kd @ (0 - q_dot)

def ctrl_with_grav(q, q_dot, q_des):
    return arm.gravity_vector(q) + Kp @ (q_des - q) + Kd @ (0 - q_dot)

# Simulate both
traj_pd = simulate(arm, ctrl_pd_only, np.zeros(2), q_desired)
traj_grav = simulate(arm, ctrl_with_grav, np.zeros(2), q_desired)

# Compare final errors
error_pd = np.degrees(np.abs(traj_pd[-1] - q_desired))
error_grav = np.degrees(np.abs(traj_grav[-1] - q_desired))

print("Steady-state error (degrees):")
print(f"  PD only:       joint1={error_pd[0]:.2f}, joint2={error_pd[1]:.2f}")
print(f"  PD + gravity:  joint1={error_grav[0]:.2f}, joint2={error_grav[1]:.2f}")
```

---

## Computed Torque Control

### Full Dynamic Compensation

The **computed torque** (also called **inverse dynamics control** or **feedback linearization**) uses the complete dynamics model to cancel nonlinearities:

$$\boldsymbol{\tau} = M(\mathbf{q}) \mathbf{a} + C(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

where $\mathbf{a}$ is the "virtual acceleration" command:

$$\mathbf{a} = \ddot{\mathbf{q}}_d + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p (\mathbf{q}_d - \mathbf{q})$$

Substituting into the dynamics equation:

$$M \ddot{\mathbf{q}} + C \dot{\mathbf{q}} + \mathbf{g} = M \mathbf{a} + C \dot{\mathbf{q}} + \mathbf{g}$$

$$\ddot{\mathbf{q}} = \mathbf{a} = \ddot{\mathbf{q}}_d + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + K_p (\mathbf{q}_d - \mathbf{q})$$

The resulting error dynamics are linear: $\ddot{\mathbf{e}} + K_d \dot{\mathbf{e}} + K_p \mathbf{e} = 0$

With proper gain selection ($K_p$, $K_d$ making the characteristic polynomial stable), the tracking error converges to zero exponentially.

```python
def computed_torque_controller(arm, q, q_dot, q_desired, q_dot_desired,
                               q_ddot_desired, Kp, Kd):
    """Computed torque (inverse dynamics) controller.

    Why is this the 'gold standard'? Because it exactly linearizes the
    system. After cancellation, the closed-loop behaves like a linear
    spring-damper system, regardless of the robot's nonlinear dynamics.

    The catch: it requires a perfect dynamics model. Model errors
    (inaccurate masses, unmodeled friction) degrade performance.
    In practice, robust or adaptive variants are used.
    """
    # Compute error
    e = q_desired - q
    e_dot = q_dot_desired - q_dot

    # Virtual acceleration command
    a = q_ddot_desired + Kd @ e_dot + Kp @ e

    # Inverse dynamics: compute required torque for acceleration 'a'
    M = arm.inertia_matrix(q)
    C = arm.coriolis_matrix(q, q_dot)
    g = arm.gravity_vector(q)

    tau = M @ a + C @ q_dot + g
    return tau

# Simulate computed torque control
Kp = np.diag([100, 100])   # Higher gains — possible because dynamics are cancelled
Kd = np.diag([20, 20])

def ctrl_computed_torque(q, q_dot, q_des):
    return computed_torque_controller(
        arm, q, q_dot, q_des, np.zeros(2), np.zeros(2), Kp, Kd)

traj_ct = simulate(arm, ctrl_computed_torque, np.zeros(2), q_desired, duration=1.0)
error_ct = np.degrees(np.abs(traj_ct[-1] - q_desired))
print(f"\nComputed torque error: joint1={error_ct[0]:.4f} deg, "
      f"joint2={error_ct[1]:.4f} deg")
```

---

## Newton-Euler Formulation

### Overview

The Newton-Euler approach computes dynamics by a **two-pass recursive algorithm**:

1. **Forward pass** (base to tip): Compute link velocities and accelerations iteratively
2. **Backward pass** (tip to base): Compute forces and torques at each joint using Newton's and Euler's equations

### Comparison with Euler-Lagrange

| Aspect | Euler-Lagrange | Newton-Euler |
|--------|---------------|--------------|
| Formulation | Energy-based | Force/torque balance |
| Computation | Symbolic (analytical) | Recursive numerical |
| Complexity | $O(n^3)$ or $O(n^4)$ | $O(n)$ — **linear!** |
| Insight | Reveals structure (M, C, g) | Efficient computation |
| Use | Analysis, control design | Real-time simulation, feedforward |

```python
def newton_euler_2link(arm, q, q_dot, q_ddot):
    """Newton-Euler recursive dynamics for 2-link planar robot.

    Why Newton-Euler alongside Euler-Lagrange? Because for real-time
    control of robots with 6+ joints, the recursive O(n) algorithm is
    essential. Euler-Lagrange gives beautiful analytical expressions but
    its computational cost grows quickly with the number of joints.

    For a 6-DOF robot:
    - Euler-Lagrange: ~66,000 multiplications
    - Newton-Euler: ~852 multiplications
    That's nearly 80x faster!
    """
    m1, m2 = arm.m1, arm.m2
    l1, l2 = arm.l1, arm.l2
    lc1, lc2 = arm.lc1, arm.lc2
    I1, I2 = arm.I1, arm.I2
    g = arm.g
    t1, t2 = q
    t1d, t2d = q_dot
    t1dd, t2dd = q_ddot

    # === Forward pass: compute velocities and accelerations ===

    # Link 1 angular velocity and acceleration
    w1 = t1d
    alpha1 = t1dd

    # Link 1 center of mass linear acceleration (including gravity)
    # a_c1 = d^2/dt^2 (lc1 * [cos(t1), sin(t1)])
    # We include gravity as a pseudo-acceleration of the base: a0 = [0, g, 0]
    ac1_x = -lc1 * sin(t1) * t1dd - lc1 * cos(t1) * t1d**2
    ac1_y = lc1 * cos(t1) * t1dd - lc1 * sin(t1) * t1d**2 + g

    # Link 2 angular velocity and acceleration
    w2 = t1d + t2d
    alpha2 = t1dd + t2dd

    # Joint 2 position acceleration
    a2_x = -l1 * sin(t1) * t1dd - l1 * cos(t1) * t1d**2
    a2_y = l1 * cos(t1) * t1dd - l1 * sin(t1) * t1d**2 + g

    # Link 2 center of mass acceleration
    t12 = t1 + t2
    w12 = t1d + t2d
    ac2_x = a2_x - lc2 * sin(t12) * (t1dd + t2dd) - lc2 * cos(t12) * w12**2
    ac2_y = a2_y + lc2 * cos(t12) * (t1dd + t2dd) - lc2 * sin(t12) * w12**2

    # === Backward pass: compute forces and torques ===

    # Link 2 forces (Newton: F = m*a)
    f2_x = m2 * ac2_x
    f2_y = m2 * ac2_y

    # Link 2 torque (Euler: tau = I*alpha + r x F)
    tau2 = I2 * alpha2 + lc2 * (cos(t12) * f2_y - sin(t12) * f2_x)
    # Simplification for planar case: tau2 includes moment about joint 2

    # Link 1 forces (including reaction from link 2)
    f1_x = m1 * ac1_x + f2_x
    f1_y = m1 * ac1_y + f2_y

    # Link 1 torque
    tau1 = I1 * alpha1 + lc1 * (cos(t1) * f1_y - sin(t1) * f1_x) + tau2 + \
           l1 * (cos(t1) * f2_y - sin(t1) * f2_x)

    return np.array([tau1, tau2])

# Compare Euler-Lagrange and Newton-Euler
q = np.radians([45, 30])
q_dot = np.array([1.0, -0.5])
q_ddot = np.array([0.5, 0.3])

tau_el = arm.inverse_dynamics(q, q_dot, q_ddot)
tau_ne = newton_euler_2link(arm, q, q_dot, q_ddot)

print("Euler-Lagrange torques:", np.round(tau_el, 4))
print("Newton-Euler torques:  ", np.round(tau_ne, 4))
print(f"Match: {np.allclose(tau_el, tau_ne, atol=1e-3)}")
```

---

## Simulation: Free Fall and Gravity Effects

```python
def simulate_free_fall(arm, q0, dt=0.001, duration=2.0):
    """Simulate the arm falling under gravity (zero applied torque).

    Why simulate free fall? It's the ultimate test of the dynamics model.
    The arm should swing like a double pendulum — chaotic, energy-conserving
    (if no friction), and physically realistic.
    """
    n_steps = int(duration / dt)
    q = q0.copy()
    q_dot = np.zeros(2)
    history = {'t': [], 'q': [], 'q_dot': [], 'energy': []}

    for step in range(n_steps):
        t = step * dt

        # No applied torque — gravity only
        q_ddot = arm.forward_dynamics(q, q_dot, np.zeros(2))

        # Compute energy for conservation check
        M = arm.inertia_matrix(q)
        K = 0.5 * q_dot @ M @ q_dot  # kinetic energy
        # Potential energy (approximate for planar case)
        P = (arm.m1 * arm.lc1 * sin(q[0]) + \
             arm.m2 * (arm.l1 * sin(q[0]) + arm.lc2 * sin(q[0] + q[1]))) * arm.g
        E = K + P

        history['t'].append(t)
        history['q'].append(q.copy())
        history['q_dot'].append(q_dot.copy())
        history['energy'].append(E)

        # Euler integration (use RK4 for better accuracy)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt

    return history

# Drop the arm from horizontal
history = simulate_free_fall(arm, np.radians([90, 0]))

# Check energy conservation (should be constant for conservative system)
energies = np.array(history['energy'])
energy_drift = (energies[-1] - energies[0]) / abs(energies[0])
print(f"Energy drift over 2s: {energy_drift*100:.2f}%")
print(f"  (Non-zero drift is due to Euler integration — use RK4 for better results)")

# Show trajectory
qs = np.array(history['q'])
print(f"\nFinal configuration: q = {np.degrees(qs[-1]).round(1)} deg")
print(f"Final velocity: q_dot = {np.array(history['q_dot'][-1]).round(2)} rad/s")
```

---

## Summary

- **Robot dynamics** relates joint torques to motion: $M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$
- The **Euler-Lagrange** formulation derives dynamics from energy (kinetic - potential), yielding elegant analytical expressions
- The **inertia matrix** $M$ is symmetric, positive definite, and configuration-dependent
- **Coriolis/centrifugal** terms represent velocity-dependent coupling forces; they satisfy the skew-symmetry property with $\dot{M}$
- **Gravity compensation** ($\tau = g(q) + \text{PD}$) is the simplest and most impactful dynamic controller
- **Computed torque control** uses the full model to linearize the system, enabling linear control design
- The **Newton-Euler** recursive algorithm computes inverse dynamics in $O(n)$ time, essential for real-time applications with many joints

---

## Exercises

### Exercise 1: Single Link Dynamics

For a single-link arm (pendulum) with mass $m = 2$ kg, length $l = 0.5$ m, center of mass at $l_c = 0.25$ m, and inertia $I = 0.02$ kg m$^2$:
1. Derive the equation of motion using Euler-Lagrange
2. Compute the gravity torque at $\theta = 0°$, $45°$, $90°$
3. What torque is needed to hold the arm at $\theta = 45°$?
4. Simulate free fall from $\theta = 90°$ using Euler integration

### Exercise 2: Inertia Matrix Analysis

For the 2-link arm from this lesson:
1. Compute $M(q)$ at $\theta_2 = 0°$, $90°$, $180°$ (with $\theta_1 = 0$)
2. Compute the eigenvalues at each configuration. How do they change?
3. At which $\theta_2$ is $M_{11}$ maximized? Minimized? Explain physically
4. Verify positive definiteness at 100 random configurations

### Exercise 3: Coriolis Effects

With the 2-link arm:
1. Set $q = (45°, 60°)$ and $\dot{q} = (2, 0)$ (only joint 1 moving). Compute the Coriolis torque on joint 2
2. Physically explain why moving joint 1 creates a torque on joint 2
3. Verify the skew-symmetry property numerically at 3 different configurations

### Exercise 4: Controller Comparison

Implement and compare three controllers for the 2-link arm tracking a step input from $q = (0, 0)$ to $q_d = (45°, 30°)$:
1. PD control only (no model compensation)
2. PD + gravity compensation
3. Computed torque control
For each, plot the tracking error and joint torques. Identify steady-state errors and transient behavior.

### Exercise 5: Newton-Euler Implementation

1. Implement the Newton-Euler recursive algorithm for a general $n$-DOF serial manipulator
2. Verify against the Euler-Lagrange result for the 2-link arm at 5 random configurations
3. Time both methods for computing inverse dynamics. How does the speed compare?

---

[← Previous: Velocity Kinematics](05_Velocity_Kinematics.md) | [Next: Motion Planning →](07_Motion_Planning.md)
