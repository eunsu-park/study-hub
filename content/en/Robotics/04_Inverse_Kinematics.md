# Inverse Kinematics

[← Previous: Forward Kinematics](03_Forward_Kinematics.md) | [Next: Velocity Kinematics →](05_Velocity_Kinematics.md)

## Learning Objectives

1. Formulate the inverse kinematics (IK) problem and explain why it is fundamentally harder than forward kinematics
2. Solve IK analytically for simple robots (2-link planar, 3-DOF) using geometric and algebraic methods
3. Apply numerical IK methods: Newton-Raphson, Jacobian pseudo-inverse, and damped least squares (Levenberg-Marquardt)
4. Identify and handle multiple solutions, workspace boundaries, and unreachable targets
5. Explain singularities and their effects on IK solutions, including the distinction between boundary and interior singularities
6. Introduce redundancy resolution strategies for robots with more than 6 DOF

---

## Why This Matters

If forward kinematics asks "given joint angles, where is the end-effector?", inverse kinematics asks the *reverse*: "given a desired end-effector pose, what joint angles achieve it?" This is the question that every real robotic task ultimately reduces to. When a surgeon commands "move the scalpel 2mm to the left" or a pick-and-place system needs to reach a box on a shelf, the controller must solve IK in real time.

IK is harder than FK for deep mathematical reasons. FK is a smooth function from joint space to task space; IK tries to invert that function, but the inverse may not exist (target outside workspace), may have multiple branches (elbow-up vs elbow-down), or may be ill-conditioned near singularities. Understanding these challenges — and having robust algorithms to handle them — separates textbook robotics from real-world robotics.

> **Analogy**: IK is like reverse navigation — given a destination, find the turn-by-turn directions. Just as there may be multiple routes to the same destination, some longer, some shorter, and some roads may be closed (singularities), IK must find viable joint configurations among potentially many candidates.

---

## The IK Problem

### Formal Statement

Given a desired end-effector pose $T_d \in SE(3)$ (a 4x4 homogeneous transformation matrix), find the joint configuration $\mathbf{q} = (q_1, q_2, \ldots, q_n)^T$ such that:

$$\text{FK}(\mathbf{q}) = T_d$$

or equivalently:

$$\mathbf{f}(\mathbf{q}) = \mathbf{x}_d$$

where $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ is the forward kinematics function and $\mathbf{x}_d$ is the desired task-space configuration (position + orientation).

### Why IK is Hard

| FK | IK |
|----|-----|
| Single, unique answer | 0, 1, or multiple solutions |
| Smooth, well-conditioned | May be ill-conditioned near singularities |
| Closed-form always exists (just multiply matrices) | Closed-form exists only for specific geometries |
| $O(n)$ computation | May require iterative methods |

### Existence and Uniqueness

- **No solution**: Target is outside the workspace
- **Finite solutions**: Most 6-DOF robots with spherical wrists have up to 16 solutions
- **Infinite solutions**: Redundant robots ($n > 6$) have infinitely many solutions for any reachable target
- **Isolated solutions**: Non-redundant robots at regular points have a discrete (finite) set of solutions

```python
import numpy as np

# Forward kinematics helper (from Lesson 3)
def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ])
```

---

## Analytical (Closed-Form) IK

Analytical IK provides exact solutions using geometric insight or algebraic manipulation. It is preferred when available because it is fast, exact, and enumerates all solutions.

### Geometric Method: 2-Link Planar Robot

The 2-link planar robot is the classic example. Given desired position $(x_d, y_d)$, find $(\theta_1, \theta_2)$.

**Setup**: Link lengths $l_1$, $l_2$; end-effector position:
$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$

**Step 1**: Find $\theta_2$ using the law of cosines.

The distance from the origin to the target is:
$$r^2 = x_d^2 + y_d^2 = l_1^2 + l_2^2 + 2 l_1 l_2 \cos\theta_2$$

Solving for $\theta_2$:
$$\cos\theta_2 = \frac{x_d^2 + y_d^2 - l_1^2 - l_2^2}{2 l_1 l_2}$$

$$\theta_2 = \pm \arccos\left(\frac{x_d^2 + y_d^2 - l_1^2 - l_2^2}{2 l_1 l_2}\right)$$

The $\pm$ gives **two solutions**: **elbow-up** and **elbow-down**.

**Step 2**: Find $\theta_1$ using geometry.

$$\theta_1 = \text{atan2}(y_d, x_d) - \text{atan2}(l_2 \sin\theta_2, \, l_1 + l_2 \cos\theta_2)$$

```python
def ik_2link_planar(x_d, y_d, l1, l2, elbow='up'):
    """Analytical IK for 2-link planar robot.

    Why two solutions? Geometrically, if you draw circles of radius l1
    (centered at origin) and l2 (centered at target), they typically
    intersect at two points — the elbow joint can be above or below
    the line from base to end-effector.

    Parameters:
        x_d, y_d: desired end-effector position
        l1, l2: link lengths
        elbow: 'up' or 'down' selects which solution

    Returns:
        (theta1, theta2) or None if unreachable
    """
    r_sq = x_d**2 + y_d**2
    r = np.sqrt(r_sq)

    # Reachability check
    if r > l1 + l2 or r < abs(l1 - l2):
        print(f"Target ({x_d:.3f}, {y_d:.3f}) is outside workspace!")
        print(f"  Distance: {r:.3f}, Range: [{abs(l1-l2):.3f}, {l1+l2:.3f}]")
        return None

    # Step 1: theta2 from law of cosines
    cos_theta2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # numerical safety

    if elbow == 'up':
        theta2 = np.arccos(cos_theta2)   # positive angle
    else:
        theta2 = -np.arccos(cos_theta2)  # negative angle

    # Step 2: theta1 from geometry
    beta = np.arctan2(y_d, x_d)
    phi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = beta - phi

    return theta1, theta2


# Example: reach point (1.2, 0.8)
l1, l2 = 1.0, 0.8
x_d, y_d = 1.2, 0.8

for elbow in ['up', 'down']:
    result = ik_2link_planar(x_d, y_d, l1, l2, elbow=elbow)
    if result:
        theta1, theta2 = result
        # Verify with FK
        x_fk = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        y_fk = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
        error = np.sqrt((x_fk - x_d)**2 + (y_fk - y_d)**2)
        print(f"Elbow {elbow}: theta1={np.degrees(theta1):.2f} deg, "
              f"theta2={np.degrees(theta2):.2f} deg, FK error={error:.2e}")

# Unreachable target
ik_2link_planar(3.0, 0.0, l1, l2)  # r=3 > l1+l2=1.8
```

### Algebraic Method: Detaching the Wrist

For 6-DOF robots with a **spherical wrist** (joints 4, 5, 6 axes intersect at a point), we can decouple the problem:

1. **Position IK** (joints 1, 2, 3): Find the position of the wrist center
2. **Orientation IK** (joints 4, 5, 6): Find the wrist angles to achieve the desired orientation

The wrist center position $\mathbf{p}_w$ is:

$$\mathbf{p}_w = \mathbf{p}_d - d_6 \cdot {}^{0}R_d \cdot \hat{z}_6$$

where $\mathbf{p}_d$ is the desired end-effector position, ${}^{0}R_d$ is the desired orientation, and $d_6$ is the wrist-to-tool distance.

```python
def ik_6dof_decoupled(T_desired, robot_params):
    """Conceptual framework for decoupled 6-DOF IK.

    Why decoupling works: When the last three joint axes intersect,
    the position of that intersection point (wrist center) depends
    ONLY on joints 1-3. This separates a 6-variable problem into
    two 3-variable problems, each solvable analytically.

    This is the single most important insight in manipulator IK.
    """
    R_d = T_desired[:3, :3]  # desired orientation
    p_d = T_desired[:3, 3]   # desired position
    d6 = robot_params['d6']  # wrist to tool offset

    # Step 1: Wrist center position
    # The tool frame z-axis in world coordinates is the 3rd column of R_d
    z_tool = R_d[:, 2]
    p_wrist = p_d - d6 * z_tool

    # Step 2: Solve for joints 1,2,3 (position IK)
    # This depends on the specific arm geometry
    # For a PUMA-like arm, use geometric/trigonometric methods
    theta1, theta2, theta3 = solve_arm_ik(p_wrist, robot_params)

    # Step 3: Compute the orientation from joints 1-3
    R_03 = compute_R03(theta1, theta2, theta3, robot_params)

    # Step 4: The wrist must provide the remaining rotation
    # R_d = R_03 * R_36  =>  R_36 = R_03^T * R_d
    R_36 = R_03.T @ R_d

    # Step 5: Extract Euler angles from R_36 for the spherical wrist
    theta4, theta5, theta6 = extract_zyz_euler(R_36)

    return np.array([theta1, theta2, theta3, theta4, theta5, theta6])

# Placeholder functions (actual implementation is robot-specific)
def solve_arm_ik(p_wrist, params):
    """Solve position IK for the first 3 joints."""
    # Implementation depends on arm geometry
    return 0, 0, 0  # placeholder

def compute_R03(t1, t2, t3, params):
    """Compute rotation matrix from joints 1-3."""
    return np.eye(3)  # placeholder

def extract_zyz_euler(R):
    """Extract ZYZ Euler angles from rotation matrix."""
    theta5 = np.arccos(np.clip(R[2, 2], -1, 1))
    if abs(np.sin(theta5)) > 1e-10:
        theta4 = np.arctan2(R[1, 2], R[0, 2])
        theta6 = np.arctan2(R[2, 1], -R[2, 0])
    else:
        # Wrist singularity: theta4 + theta6 determined, but not individually
        theta4 = 0
        theta6 = np.arctan2(-R[0, 1], R[0, 0])
    return theta4, theta5, theta6
```

### Number of Solutions

For a general 6-DOF robot with a spherical wrist:
- Up to **8 solutions** for the arm (joints 1-3): left/right shoulder, elbow up/down, wrist flip
- Up to **2 solutions** for the wrist (joints 4-6): wrist flip ($\theta_5$ positive or negative)
- **Total: up to 16** distinct solutions

```python
def enumerate_2link_solutions(x_d, y_d, l1, l2):
    """Find all IK solutions for a 2-link planar robot.

    Why enumerate all? In practice, we choose the solution closest to
    the current configuration (to minimize joint motion) or the one
    that avoids obstacles. But first, we need to know all options.
    """
    solutions = []

    r_sq = x_d**2 + y_d**2
    r = np.sqrt(r_sq)

    if r > l1 + l2 + 1e-10 or r < abs(l1 - l2) - 1e-10:
        return solutions  # empty — unreachable

    cos_theta2 = (r_sq - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)

    for sign in [1, -1]:  # elbow up and down
        theta2 = sign * np.arccos(cos_theta2)
        beta = np.arctan2(y_d, x_d)
        phi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = beta - phi
        solutions.append((theta1, theta2))

    return solutions

# Example with solutions
solutions = enumerate_2link_solutions(0.8, 0.6, 1.0, 0.8)
print(f"Number of solutions: {len(solutions)}")
for i, (t1, t2) in enumerate(solutions):
    x = 1.0 * np.cos(t1) + 0.8 * np.cos(t1 + t2)
    y = 1.0 * np.sin(t1) + 0.8 * np.sin(t1 + t2)
    print(f"  Solution {i+1}: theta1={np.degrees(t1):.2f} deg, "
          f"theta2={np.degrees(t2):.2f} deg -> ({x:.4f}, {y:.4f})")
```

---

## Numerical IK Methods

When closed-form solutions are unavailable (non-standard geometries, redundant robots, constrained IK), we use iterative numerical methods.

### The Jacobian-Based Framework

All numerical IK methods build on the **Jacobian** (detailed in Lesson 5). The Jacobian $J(\mathbf{q})$ relates small changes in joint angles to small changes in end-effector pose:

$$\delta \mathbf{x} = J(\mathbf{q}) \, \delta \mathbf{q}$$

For IK, we need the inverse relationship:

$$\delta \mathbf{q} = J(\mathbf{q})^{-1} \, \delta \mathbf{x}$$

But $J$ may not be square, and even if it is, it may be singular. Different methods handle this differently.

### Method 1: Newton-Raphson

The Newton-Raphson method applies iterative linearization:

$$\mathbf{q}_{k+1} = \mathbf{q}_k + J(\mathbf{q}_k)^{-1} \, (\mathbf{x}_d - \mathbf{f}(\mathbf{q}_k))$$

where $\mathbf{f}(\mathbf{q}_k)$ is the current end-effector pose from FK.

```python
def ik_newton_raphson(fk_func, jacobian_func, x_desired, q_init,
                      max_iter=100, tol=1e-6):
    """Newton-Raphson IK solver.

    Why Newton-Raphson? It has quadratic convergence near the solution,
    meaning the error roughly squares each iteration. A solution accurate
    to 10^-6 typically requires only 5-10 iterations.

    Limitations:
    - Requires a good initial guess (not globally convergent)
    - Fails at singularities (J is not invertible)
    - Only finds ONE solution (the nearest to the initial guess)
    """
    q = q_init.copy()

    for iteration in range(max_iter):
        # Current pose from FK
        x_current = fk_func(q)

        # Error
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"  Converged in {iteration+1} iterations "
                  f"(error={error_norm:.2e})")
            return q

        # Jacobian at current configuration
        J = jacobian_func(q)

        # Solve J * dq = error
        try:
            dq = np.linalg.solve(J, error)
        except np.linalg.LinAlgError:
            print(f"  Singular Jacobian at iteration {iteration+1}")
            return None

        # Update
        q = q + dq

    print(f"  Did not converge after {max_iter} iterations "
          f"(error={error_norm:.2e})")
    return q
```

### Method 2: Jacobian Pseudo-Inverse

When the Jacobian is not square ($m \neq n$, where $m$ is task dimension and $n$ is joint dimension), we use the **Moore-Penrose pseudo-inverse**:

For $m < n$ (redundant robot — more joints than task DOF):
$$\delta \mathbf{q} = J^+ \, \delta \mathbf{x} = J^T (J J^T)^{-1} \, \delta \mathbf{x}$$

This gives the **minimum-norm** solution — the smallest joint motion that achieves the desired task-space change.

For $m > n$ (under-determined — fewer joints than task DOF):
$$\delta \mathbf{q} = J^+ \, \delta \mathbf{x} = (J^T J)^{-1} J^T \, \delta \mathbf{x}$$

This gives the **least-squares** solution — the joint motion that gets as close as possible to the desired change.

```python
def ik_pseudo_inverse(fk_func, jacobian_func, x_desired, q_init,
                      max_iter=200, tol=1e-6, alpha=1.0):
    """Pseudo-inverse IK solver.

    Why pseudo-inverse? It handles non-square Jacobians naturally,
    making it the go-to method for redundant robots (7+ DOF).
    The alpha parameter (step size) can be tuned for stability.

    The pseudo-inverse solution minimizes ||dq||, which tends to
    distribute motion across all joints rather than concentrating
    it in one — generally desirable for smooth motion.
    """
    q = q_init.copy()

    for iteration in range(max_iter):
        x_current = fk_func(q)
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"  Converged in {iteration+1} iterations "
                  f"(error={error_norm:.2e})")
            return q

        J = jacobian_func(q)

        # Pseudo-inverse: J+ = J^T (J J^T)^-1
        # Using numpy's built-in pinv (uses SVD internally)
        J_pinv = np.linalg.pinv(J)

        dq = alpha * J_pinv @ error
        q = q + dq

    print(f"  Did not converge after {max_iter} iterations "
          f"(error={error_norm:.2e})")
    return q
```

### Method 3: Damped Least Squares (Levenberg-Marquardt)

The pseudo-inverse fails near singularities because $J J^T$ becomes near-singular. The **Damped Least Squares** (DLS) method adds a damping term:

$$\delta \mathbf{q} = J^T (J J^T + \lambda^2 I)^{-1} \, \delta \mathbf{x}$$

where $\lambda$ is the **damping factor**. This trades off accuracy for robustness near singularities.

- When $\lambda = 0$: identical to pseudo-inverse
- When $\lambda$ is large: motion is small but well-conditioned
- Near singularities: $\lambda$ prevents explosive joint velocities

```python
def ik_damped_least_squares(fk_func, jacobian_func, x_desired, q_init,
                            max_iter=200, tol=1e-6, lambda_damp=0.1):
    """Damped Least Squares (DLS) IK solver.

    Why DLS? It's the most robust general-purpose IK method. Near
    singularities, the pseudo-inverse produces enormous joint velocities
    (trying to move in a direction the robot physically cannot).
    DLS gracefully degrades: it slows down instead of blowing up.

    The lambda parameter is the key tradeoff:
    - Too small: still vulnerable to singularities
    - Too large: slow convergence, poor accuracy
    - Adaptive lambda (see below) is the best approach
    """
    q = q_init.copy()

    for iteration in range(max_iter):
        x_current = fk_func(q)
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"  Converged in {iteration+1} iterations "
                  f"(error={error_norm:.2e})")
            return q

        J = jacobian_func(q)

        # DLS: J^T (J J^T + lambda^2 I)^-1 * error
        m = J.shape[0]  # task space dimension
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lambda_damp**2 * np.eye(m), error)

        q = q + dq

    print(f"  Did not converge after {max_iter} iterations "
          f"(error={error_norm:.2e})")
    return q

def ik_adaptive_dls(fk_func, jacobian_func, x_desired, q_init,
                    max_iter=200, tol=1e-6, lambda_max=0.5):
    """DLS with adaptive damping based on manipulability.

    Why adaptive? A fixed lambda is either too conservative (far from
    singularity) or too aggressive (near singularity). Adaptive damping
    increases lambda only when needed — when the manipulability measure
    drops below a threshold.
    """
    q = q_init.copy()
    epsilon = 0.01  # manipulability threshold

    for iteration in range(max_iter):
        x_current = fk_func(q)
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"  Converged in {iteration+1} iterations")
            return q

        J = jacobian_func(q)

        # Manipulability: sqrt(det(J J^T))
        JJT = J @ J.T
        w = np.sqrt(max(np.linalg.det(JJT), 0))

        # Adaptive damping
        if w < epsilon:
            lam = lambda_max * (1 - (w / epsilon)**2)
        else:
            lam = 0.0

        dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(J.shape[0]), error)
        q = q + dq

    return q
```

---

## Complete IK Example: 2-Link Planar Robot

Let us build a complete IK solution combining analytical and numerical methods for comparison.

```python
class PlanarRobot2Link:
    """2-link planar robot with both analytical and numerical IK.

    Why both? Analytical IK is faster and finds all solutions, but
    only works for specific geometries. Numerical IK is general.
    Having both lets us verify the numerical method against the
    known analytical solution.
    """
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2

    def fk(self, q):
        """Forward kinematics: returns [x, y]."""
        t1, t2 = q
        x = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        y = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)
        return np.array([x, y])

    def jacobian(self, q):
        """Analytical Jacobian: 2x2 matrix.

        J = d[x,y]/d[theta1, theta2]
        Derived by differentiating the FK equations.
        """
        t1, t2 = q
        s1 = np.sin(t1)
        c1 = np.cos(t1)
        s12 = np.sin(t1 + t2)
        c12 = np.cos(t1 + t2)

        return np.array([
            [-self.l1*s1 - self.l2*s12, -self.l2*s12],
            [ self.l1*c1 + self.l2*c12,  self.l2*c12]
        ])

    def ik_analytical(self, x_d, y_d):
        """All analytical solutions."""
        solutions = []
        r_sq = x_d**2 + y_d**2
        cos_t2 = (r_sq - self.l1**2 - self.l2**2) / (2*self.l1*self.l2)

        if abs(cos_t2) > 1 + 1e-10:
            return solutions

        cos_t2 = np.clip(cos_t2, -1, 1)
        for sign in [1, -1]:
            t2 = sign * np.arccos(cos_t2)
            t1 = np.arctan2(y_d, x_d) - np.arctan2(
                self.l2*np.sin(t2), self.l1 + self.l2*np.cos(t2))
            solutions.append(np.array([t1, t2]))
        return solutions

    def ik_numerical(self, x_desired, q_init, method='dls', **kwargs):
        """Numerical IK using specified method."""
        x_d = np.array(x_desired)

        if method == 'newton':
            return ik_newton_raphson(self.fk, self.jacobian, x_d, q_init, **kwargs)
        elif method == 'pinv':
            return ik_pseudo_inverse(self.fk, self.jacobian, x_d, q_init, **kwargs)
        elif method == 'dls':
            return ik_damped_least_squares(self.fk, self.jacobian, x_d, q_init, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


# Compare analytical and numerical IK
robot = PlanarRobot2Link(l1=1.0, l2=0.8)
target = np.array([0.9, 0.7])

print("=== Analytical IK ===")
analytic_solutions = robot.ik_analytical(target[0], target[1])
for i, sol in enumerate(analytic_solutions):
    fk_result = robot.fk(sol)
    error = np.linalg.norm(fk_result - target)
    print(f"Solution {i+1}: q={np.degrees(sol).round(2)} deg, error={error:.2e}")

print("\n=== Numerical IK (DLS) ===")
# Try with initial guess near each analytical solution
for i, init in enumerate([np.array([0.5, 0.5]), np.array([0.5, -0.5])]):
    print(f"Starting from q0={np.degrees(init).round(1)} deg:")
    q_num = robot.ik_numerical(target, init, method='dls', lambda_damp=0.01)
    if q_num is not None:
        fk_result = robot.fk(q_num)
        error = np.linalg.norm(fk_result - target)
        print(f"  Result: q={np.degrees(q_num).round(2)} deg, error={error:.2e}")
```

---

## Singularities and Their Effects

### What is a Singularity?

A **singularity** is a joint configuration where the Jacobian $J(\mathbf{q})$ loses rank — i.e., $\det(J) = 0$ for square Jacobians, or $\text{rank}(J) < \min(m, n)$ in general.

At a singularity:
- Some end-effector velocities become **impossible** (the robot loses one or more DOF of motion)
- The inverse Jacobian blows up ($\|J^{-1}\| \to \infty$)
- IK solutions become numerically unstable

### Types of Singularities

| Type | Location | Cause | Example |
|------|----------|-------|---------|
| **Boundary** | Edge of workspace | Arm fully extended or fully folded | 2-link: $\theta_2 = 0$ or $\theta_2 = \pi$ |
| **Interior** | Inside workspace | Alignment of joint axes | Wrist: $\theta_5 = 0$ (axes 4,6 align) |
| **Architectural** | Everywhere | Robot design (degenerate geometry) | Rare, indicates design flaw |

```python
def analyze_singularity_2link(robot, q):
    """Analyze proximity to singularity for 2-link planar robot.

    Why monitor singularity proximity? Because near-singular configurations
    require enormous joint velocities for small task-space motions.
    Real controllers limit joint velocities, so the robot slows down
    or stops near singularities — a phenomenon called 'locking up.'
    """
    J = robot.jacobian(q)
    det_J = np.linalg.det(J)
    cond_J = np.linalg.cond(J)

    # Singular values
    _, s, _ = np.linalg.svd(J)
    min_sv = s.min()

    # Manipulability measure (Yoshikawa)
    w = abs(det_J)

    print(f"Configuration: q={np.degrees(q).round(1)} deg")
    print(f"  det(J) = {det_J:.6f}")
    print(f"  cond(J) = {cond_J:.1f}")
    print(f"  min singular value = {min_sv:.6f}")
    print(f"  manipulability = {w:.6f}")

    if min_sv < 0.01:
        print("  WARNING: Near singularity!")
    return w

robot = PlanarRobot2Link(1.0, 0.8)

# Regular configuration
print("--- Regular configuration ---")
analyze_singularity_2link(robot, np.radians([30, 45]))

# Near-singular: elbow almost straight
print("\n--- Near singular (elbow extended) ---")
analyze_singularity_2link(robot, np.radians([30, 5]))

# Singular: elbow fully straight
print("\n--- Singular (elbow fully extended) ---")
analyze_singularity_2link(robot, np.radians([30, 0]))

# Singular: elbow fully folded
print("\n--- Singular (elbow fully folded) ---")
analyze_singularity_2link(robot, np.radians([30, 180]))
```

### Singularity Avoidance Strategies

1. **Damped Least Squares**: Use DLS instead of pseudo-inverse
2. **Singularity-robust task priority**: Reduce task-space DOF near singularities
3. **Joint limit avoidance**: Keep joints away from singular configurations
4. **Path planning**: Plan paths that avoid singular regions (Lesson 7)
5. **Manipulability maximization**: Use redundancy to stay in well-conditioned configurations

---

## Redundancy Resolution

A robot with $n > 6$ joints (for 6-DOF tasks) has **kinematic redundancy** — infinitely many joint configurations reach the same end-effector pose.

### The Null Space

The pseudo-inverse solution $\delta \mathbf{q} = J^+ \delta \mathbf{x}$ is the minimum-norm solution. We can add any vector in the **null space** of $J$ without affecting the end-effector:

$$\delta \mathbf{q} = J^+ \delta \mathbf{x} + (I - J^+ J) \mathbf{z}$$

where $(I - J^+ J)$ is the **null-space projector** and $\mathbf{z}$ is an arbitrary vector.

The null-space component $(I - J^+ J)\mathbf{z}$ moves the joints without moving the end-effector — this is called **self-motion**.

```python
def ik_redundant(fk_func, jacobian_func, x_desired, q_init,
                 null_space_objective=None, max_iter=200, tol=1e-6,
                 alpha=0.5, beta=0.1):
    """IK solver with null-space optimization for redundant robots.

    Why null-space optimization? A 7-DOF robot has infinite solutions
    for any reachable pose. The null space lets us use the extra DOF
    to achieve secondary objectives while maintaining the primary task:
    - Avoid joint limits
    - Avoid obstacles
    - Maximize manipulability
    - Minimize energy

    Parameters:
        null_space_objective: function q -> gradient of secondary objective
        beta: weight for null-space motion
    """
    q = q_init.copy()
    n = len(q)

    for iteration in range(max_iter):
        x_current = fk_func(q)
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            print(f"  Converged in {iteration+1} iterations")
            return q

        J = jacobian_func(q)
        J_pinv = np.linalg.pinv(J)

        # Primary task: minimize end-effector error
        dq_primary = alpha * J_pinv @ error

        # Secondary task: null-space optimization
        if null_space_objective is not None:
            z = null_space_objective(q)
            N = np.eye(n) - J_pinv @ J  # null-space projector
            dq_null = beta * N @ z
        else:
            dq_null = np.zeros(n)

        q = q + dq_primary + dq_null

    return q

# Example secondary objective: stay near joint centers (avoid limits)
def joint_center_gradient(q, joint_centers=None, joint_ranges=None):
    """Gradient that pushes joints toward their center positions.

    Why joint centering? It maximizes the robot's ability to respond
    to future motion commands in any direction. A joint at its limit
    can only move one way — a joint at center can move either way.
    """
    if joint_centers is None:
        joint_centers = np.zeros_like(q)
    if joint_ranges is None:
        joint_ranges = np.ones_like(q) * np.pi

    # Negative gradient of (q - q_center)^2 / range^2
    return -(q - joint_centers) / (joint_ranges**2)
```

---

## Practical Considerations

### Choosing the Right Method

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| 6-DOF with spherical wrist | Analytical (decoupled) | Fast, exact, all solutions |
| Simple geometry (< 4 DOF) | Analytical (geometric) | Fast, exact |
| General geometry | DLS | Robust, handles singularities |
| Redundant robot (7+ DOF) | Pseudo-inverse + null space | Exploits redundancy |
| Many targets in sequence | Numerical from previous solution | Warm-starting |
| Real-time control (1 kHz) | Analytical if possible, else DLS | Speed requirement |

### Joint Limits

Real robots have joint limits. After computing IK, always check:

```python
def enforce_joint_limits(q, limits):
    """Clamp joint values to limits.

    Why not just clamp? Clamping can violate the FK solution.
    Better approaches:
    1. Use constrained optimization
    2. Penalize limit violations in the cost function
    3. Choose IK solutions that respect limits
    """
    q_clamped = q.copy()
    for i, (lo, hi) in enumerate(limits):
        q_clamped[i] = np.clip(q[i], lo, hi)

    # Warning: clamped q may not satisfy FK!
    return q_clamped

def ik_with_limits(robot, x_desired, q_init, limits, max_iter=200, tol=1e-6):
    """IK solver that respects joint limits via gradient projection.

    This method projects the search direction to stay within joint limits,
    rather than clamping after the fact.
    """
    q = q_init.copy()
    lambda_damp = 0.05

    for iteration in range(max_iter):
        x_current = robot.fk(q)
        error = x_desired - x_current
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q

        J = robot.jacobian(q)
        m = J.shape[0]
        JJT = J @ J.T
        dq = J.T @ np.linalg.solve(JJT + lambda_damp**2 * np.eye(m), error)

        # Scale step to respect limits
        q_new = q + dq
        for i, (lo, hi) in enumerate(limits):
            if q_new[i] < lo:
                # Scale down the step so we stop at the limit
                if abs(dq[i]) > 1e-10:
                    scale = (lo - q[i]) / dq[i]
                    dq *= max(0, min(1, scale))
            elif q_new[i] > hi:
                if abs(dq[i]) > 1e-10:
                    scale = (hi - q[i]) / dq[i]
                    dq *= max(0, min(1, scale))

        q = q + dq

    return q
```

### Solution Selection

When multiple IK solutions exist, choose based on:

1. **Closest to current configuration**: Minimizes joint motion (smooth trajectory)
2. **Highest manipulability**: Best conditioned for subsequent motions
3. **Farthest from joint limits**: Maximum future mobility
4. **Obstacle avoidance**: No collisions in the selected configuration

```python
def select_best_solution(solutions, q_current, weights=None):
    """Select the best IK solution from multiple candidates.

    Why not just 'closest'? Sometimes the closest solution passes
    through a singularity, hits a joint limit, or causes a collision.
    Multi-criteria selection balances competing objectives.
    """
    if not solutions:
        return None

    if weights is None:
        weights = {'distance': 1.0, 'manipulability': 0.5}

    best_score = -np.inf
    best_solution = None

    for sol in solutions:
        score = 0
        # Criterion 1: proximity to current configuration
        dist = np.linalg.norm(sol - q_current)
        score -= weights.get('distance', 1.0) * dist

        # Criterion 2: manipulability (would need Jacobian)
        # score += weights.get('manipulability', 0) * compute_manipulability(sol)

        if score > best_score:
            best_score = score
            best_solution = sol

    return best_solution
```

---

## Summary

- **Inverse kinematics** finds joint angles for a desired end-effector pose — the fundamental problem in robot control
- **Analytical methods** (geometric, algebraic, wrist decoupling) give exact, fast solutions for specific robot geometries
- **Numerical methods** (Newton-Raphson, pseudo-inverse, DLS) handle general geometries but require initial guesses and may converge to local minima
- **Damped Least Squares** is the most robust numerical method, gracefully handling singularities
- **Multiple solutions** (up to 16 for 6-DOF) require selection criteria (proximity, manipulability, limits)
- **Singularities** cause loss of DOF and numerical instability; they must be detected and managed
- **Redundant robots** have infinitely many solutions; the null space enables secondary objectives

---

## Exercises

### Exercise 1: 2-Link Analytical IK

For a 2-link planar robot with $l_1 = 1.0$ m, $l_2 = 0.7$ m:
1. Find all IK solutions for target $(1.2, 0.5)$
2. Find all IK solutions for target $(0.3, 0.0)$ — what is special about this case?
3. Attempt IK for target $(2.0, 0.0)$ — what happens and why?
4. Plot both solutions for a reachable target, showing the arm configurations

### Exercise 2: Numerical IK Comparison

Implement Newton-Raphson, pseudo-inverse, and DLS for the 2-link robot. For the target $(0.5, 1.0)$:
1. Compare convergence rates (iterations to tolerance $10^{-8}$) starting from $q_0 = (0, 0)$
2. Compare behavior near the workspace boundary: target at distance $l_1 + l_2 - 0.01$
3. Compare behavior at a singular configuration: start from $q_0 = (0, 0)$ (arm fully extended)

### Exercise 3: Singularity Analysis

For the 2-link robot:
1. Derive the Jacobian analytically
2. Find all configurations where $\det(J) = 0$
3. At the singular configuration $q = (45°, 0°)$, what end-effector velocity is impossible?
4. Plot the manipulability measure $w(\theta_2) = |l_1 l_2 \sin\theta_2|$ and explain its behavior

### Exercise 4: 6-DOF IK Framework

Using the PUMA 560-like DH parameters from Lesson 3:
1. Implement the wrist center computation for decoupled IK
2. Verify that the wrist center position depends only on joints 1-3 by varying joints 4-6 and checking
3. For a desired end-effector pose $T_d$ (your choice), compute the wrist center and find $\theta_1$ geometrically

### Exercise 5: Redundancy

Consider a 3-link planar robot (3 revolute joints) reaching a 2D target (2-DOF task):
1. The robot has 1 DOF of redundancy. What does the self-motion look like?
2. Implement pseudo-inverse IK with null-space joint centering
3. Start from $q_0 = (20°, 40°, 60°)$ and reach target $(0.5, 0.8)$. Compare solutions with and without null-space optimization

---

[← Previous: Forward Kinematics](03_Forward_Kinematics.md) | [Next: Velocity Kinematics →](05_Velocity_Kinematics.md)
