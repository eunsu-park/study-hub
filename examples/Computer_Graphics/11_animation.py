"""
Animation: Keyframes, Quaternion Slerp, FK, and IK
====================================================

Implements core animation concepts:
1. Keyframe interpolation: linear and cubic Bezier
2. Quaternion slerp for smooth rotation
3. Forward Kinematics (FK) chain (2D arm)
4. FABRIK Inverse Kinematics solver (2D)
5. Side-by-side FK vs IK visualization

Animation in computer graphics bridges the gap between static scenes
and motion.  Whether it's a bouncing ball or a walking character,
the fundamental techniques are: interpolation between key poses,
quaternions for rotation, and FK/IK for articulated structures.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# 1. Keyframe Interpolation
# ---------------------------------------------------------------------------


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b.

    The simplest interpolation: move from a to b at constant speed.
    t=0 gives a, t=1 gives b, t=0.5 gives the midpoint.

    Why is lerp fundamental?  Every other interpolation method is built
    on top of it.  Bezier curves use nested lerps (De Casteljau's
    algorithm), and even quaternion slerp reduces to a weighted
    combination at its core.
    """
    return (1 - t) * a + t * b


def lerp_vec(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation for vectors."""
    return (1 - t) * a + t * b


def cubic_bezier_1d(p0: float, p1: float, p2: float, p3: float,
                     t: float) -> float:
    """Evaluate a cubic Bezier curve at parameter t.

    Why cubic Bezier?  Linear interpolation produces abrupt velocity
    changes at keyframes (the motion "jerks").  Cubic Bezier curves
    provide smooth acceleration/deceleration (ease-in/ease-out),
    making motion look natural.

    The four control points:
    - p0: start value
    - p1: influences departure speed and direction
    - p2: influences arrival speed and direction
    - p3: end value

    The curve passes through p0 and p3, but only approaches p1 and p2.
    This is the same curve used in CSS transitions and After Effects.
    """
    u = 1 - t
    return (u**3 * p0 +
            3 * u**2 * t * p1 +
            3 * u * t**2 * p2 +
            t**3 * p3)


def cubic_bezier_vec(p0: np.ndarray, p1: np.ndarray,
                      p2: np.ndarray, p3: np.ndarray,
                      t: float) -> np.ndarray:
    """Evaluate cubic Bezier curve for vectors (per-component)."""
    u = 1 - t
    return (u**3 * p0 +
            3 * u**2 * t * p1 +
            3 * u * t**2 * p2 +
            t**3 * p3)


def ease_in_out(t: float) -> float:
    """Hermite ease-in-ease-out curve (smoothstep).

    Maps [0,1] -> [0,1] with zero velocity at both ends.
    This is equivalent to a specific cubic Bezier with control points
    (0, 0, 1, 1) in the time domain.

    Why useful?  Applying this to the t parameter before linear
    interpolation produces smooth acceleration and deceleration
    without the complexity of full Bezier keyframes.
    """
    return t * t * (3 - 2 * t)


def demo_interpolation():
    """Compare linear vs cubic Bezier interpolation curves."""
    t_values = np.linspace(0, 1, 200)

    # Linear interpolation
    linear_values = [lerp(0, 10, t) for t in t_values]

    # Cubic Bezier (ease-in-ease-out)
    # Control points create a smooth S-curve
    bezier_values = [cubic_bezier_1d(0, 0, 10, 10, t) for t in t_values]

    # Ease-in-ease-out (smoothstep applied to t)
    eased_values = [lerp(0, 10, ease_in_out(t)) for t in t_values]

    # Different Bezier curves for variety
    overshoot_values = [cubic_bezier_1d(0, -3, 13, 10, t) for t in t_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Keyframe Interpolation Methods", fontsize=14, fontweight='bold')

    # Value over time
    ax = axes[0]
    ax.plot(t_values, linear_values, 'r-', linewidth=2, label='Linear')
    ax.plot(t_values, bezier_values, 'b-', linewidth=2, label='Bezier (ease)')
    ax.plot(t_values, eased_values, 'g--', linewidth=2, label='Smoothstep')
    ax.plot(t_values, overshoot_values, 'm-', linewidth=2, label='Bezier (overshoot)')
    ax.set_xlabel('t (normalized time)')
    ax.set_ylabel('Value')
    ax.set_title('Value over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Velocity (derivative) over time
    ax2 = axes[1]
    dt = t_values[1] - t_values[0]
    for values, label, color, style in [
        (linear_values, 'Linear', 'r', '-'),
        (bezier_values, 'Bezier', 'b', '-'),
        (eased_values, 'Smoothstep', 'g', '--'),
        (overshoot_values, 'Overshoot', 'm', '-'),
    ]:
        velocity = np.gradient(values, dt)
        ax2.plot(t_values, velocity, color=color, linestyle=style,
                 linewidth=2, label=label)

    ax2.set_xlabel('t (normalized time)')
    ax2.set_ylabel('Velocity (dValue/dt)')
    ax2.set_title('Velocity Profile (smoothness indicator)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_11_interpolation.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 2. Quaternion Rotation
# ---------------------------------------------------------------------------

class Quaternion:
    """Quaternion for 3D rotation.

    Why quaternions over Euler angles?
    1. No gimbal lock (Euler angles lose a degree of freedom at certain orientations)
    2. Smooth interpolation via slerp (Euler angle interpolation can take odd paths)
    3. Compact (4 numbers vs 3x3 matrix)
    4. Easy to compose (just quaternion multiplication)
    5. Always normalizable (Euler angles have no natural normalization)

    Representation: q = w + xi + yj + zk where (x, y, z) is the vector
    part and w is the scalar part.  A unit quaternion encodes a rotation
    of angle theta about axis (x, y, z) as:
      q = (cos(theta/2), sin(theta/2) * axis)
    """

    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle_deg: float) -> 'Quaternion':
        """Create a quaternion from an axis and angle.

        Why half-angle?  The quaternion double-covers rotation space:
        q and -q represent the same rotation.  The half-angle formula
        arises from the Rodrigues rotation formula and ensures the
        quaternion product correctly composes rotations.
        """
        axis = axis / np.linalg.norm(axis)
        half_angle = np.radians(angle_deg) / 2
        s = np.sin(half_angle)
        return cls(np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s)

    @classmethod
    def identity(cls) -> 'Quaternion':
        return cls(1, 0, 0, 0)

    def normalize(self) -> 'Quaternion':
        n = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if n < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Hamilton product: compose two rotations.

        Why this specific formula?  It's derived from the i^2 = j^2 = k^2
        = ijk = -1 rules.  The product q1 * q2 means "first apply q2,
        then apply q1" (right-to-left, like matrices).
        """
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate_point(self, p: np.ndarray) -> np.ndarray:
        """Rotate a 3D point by this quaternion: q * p * q^-1."""
        p_quat = Quaternion(0, p[0], p[1], p[2])
        result = self * p_quat * self.conjugate()
        return np.array([result.x, result.y, result.z])

    def to_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    def to_angle_deg(self) -> float:
        """Extract the rotation angle in degrees (about whatever axis)."""
        return np.degrees(2 * np.arccos(np.clip(self.w, -1, 1)))


def slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    """Spherical Linear Interpolation between two quaternions.

    Why slerp instead of lerp?  Quaternions live on a 4D unit sphere.
    Linear interpolation (lerp + normalize) doesn't travel along the
    sphere's surface at constant speed -- it cuts through the interior
    and speeds up in the middle.  Slerp follows the great circle arc
    at constant angular velocity, producing visually uniform rotation.

    The formula: slerp(q0, q1, t) = q0 * sin((1-t)*omega) / sin(omega)
                                   + q1 * sin(t*omega) / sin(omega)
    where omega is the angle between q0 and q1.
    """
    # Ensure shortest path: if dot product is negative, negate one quaternion
    # Why?  q and -q represent the same rotation.  Without this check,
    # slerp might take the "long way around" (270 degrees instead of 90).
    dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z

    if dot < 0:
        q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        dot = -dot

    dot = np.clip(dot, -1, 1)

    # If quaternions are very close, fall back to linear interpolation
    # Why?  sin(omega) approaches zero, causing division instability.
    if dot > 0.9995:
        result = Quaternion(
            lerp(q0.w, q1.w, t),
            lerp(q0.x, q1.x, t),
            lerp(q0.y, q1.y, t),
            lerp(q0.z, q1.z, t),
        )
        return result.normalize()

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    s0 = np.sin((1 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    return Quaternion(
        s0 * q0.w + s1 * q1.w,
        s0 * q0.x + s1 * q1.x,
        s0 * q0.y + s1 * q1.y,
        s0 * q0.z + s1 * q1.z,
    ).normalize()


def demo_slerp():
    """Visualize quaternion slerp by rotating a 3D arrow."""
    q_start = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0)
    q_end = Quaternion.from_axis_angle(np.array([0, 0, 1]), 120)

    # Also compare with Euler angle interpolation
    t_values = np.linspace(0, 1, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Quaternion SLERP vs Linear Angle Interpolation",
                 fontsize=14, fontweight='bold')

    # Arrow base
    arrow_tip = np.array([2, 0, 0])

    # Slerp path
    slerp_points = []
    for t in t_values:
        q = slerp(q_start, q_end, t)
        p = q.rotate_point(arrow_tip)
        slerp_points.append(p[:2])
    slerp_points = np.array(slerp_points)

    ax1.plot(slerp_points[:, 0], slerp_points[:, 1], 'b-o',
             markersize=3, linewidth=2, label='SLERP path')

    # Draw arrows at key frames
    for t_val in [0, 0.25, 0.5, 0.75, 1.0]:
        q = slerp(q_start, q_end, t_val)
        p = q.rotate_point(arrow_tip)
        ax1.annotate('', xy=(p[0], p[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(p[0]*1.1, p[1]*1.1, f't={t_val:.2f}', fontsize=8)

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('SLERP: Constant Angular Velocity')
    ax1.legend()

    # Compare angular velocity
    slerp_angles = []
    lerp_angles = []
    for t in t_values:
        q = slerp(q_start, q_end, t)
        slerp_angles.append(q.to_angle_deg())
        lerp_angles.append(lerp(0, 120, t))

    ax2.plot(t_values, slerp_angles, 'b-', linewidth=2, label='SLERP angle')
    ax2.plot(t_values, lerp_angles, 'r--', linewidth=2, label='Linear angle')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Rotation Angle over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_11_slerp.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Forward Kinematics (FK) - 2D Arm
# ---------------------------------------------------------------------------

class FKChain:
    """A 2D Forward Kinematics chain (articulated arm).

    FK computes the position of each joint given the joint angles.
    It's straightforward: each joint's position depends on all previous
    joints' angles and bone lengths.

    Why FK?  It's the foundation -- every animation system needs FK to
    convert joint angles to world positions.  IK then works *on top of* FK,
    finding angles that reach a desired target.
    """

    def __init__(self, bone_lengths: list, base_pos: np.ndarray = None):
        """
        bone_lengths: list of float, length of each bone segment
        base_pos: position of the chain's root joint
        """
        self.bone_lengths = np.array(bone_lengths, dtype=float)
        self.num_bones = len(bone_lengths)
        self.base_pos = base_pos if base_pos is not None else np.array([0, 0], dtype=float)

        # Joint angles in radians (each relative to parent bone)
        self.angles = np.zeros(self.num_bones)

    def compute_joints(self) -> np.ndarray:
        """Compute world positions of all joints.

        Returns array of shape (num_bones + 1, 2):
          [base, joint1, joint2, ..., end_effector]

        Why cumulative angle sum?  Each joint angle is relative to its
        parent bone.  The absolute angle of bone i is the sum of all
        angles from the root to bone i.  This is equivalent to
        multiplying local rotation matrices along the chain.
        """
        joints = np.zeros((self.num_bones + 1, 2))
        joints[0] = self.base_pos

        cumulative_angle = 0
        for i in range(self.num_bones):
            cumulative_angle += self.angles[i]
            dx = self.bone_lengths[i] * np.cos(cumulative_angle)
            dy = self.bone_lengths[i] * np.sin(cumulative_angle)
            joints[i + 1] = joints[i] + np.array([dx, dy])

        return joints

    def end_effector(self) -> np.ndarray:
        """Return the position of the chain's tip."""
        joints = self.compute_joints()
        return joints[-1]


# ---------------------------------------------------------------------------
# 4. FABRIK Inverse Kinematics Solver
# ---------------------------------------------------------------------------

def fabrik_solve(chain: FKChain, target: np.ndarray,
                 tolerance: float = 0.01, max_iterations: int = 50) -> bool:
    """FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.

    Why FABRIK over Jacobian methods?
    1. Much simpler to implement (no matrix math)
    2. Faster convergence for serial chains
    3. Naturally handles joint constraints (though we skip those here)
    4. Intuitive geometric algorithm

    The algorithm alternates two passes:
    - Forward: starting from the end effector, move each joint toward the
      previous one while maintaining bone lengths
    - Backward: starting from the base, move each joint toward the next
      one while maintaining bone lengths

    Each iteration brings the chain closer to a pose that satisfies both
    the base position constraint and the target constraint.

    Returns True if the target was reached within tolerance.
    """
    bone_lengths = chain.bone_lengths
    n = chain.num_bones
    total_length = np.sum(bone_lengths)

    # Check if target is reachable
    dist_to_target = np.linalg.norm(target - chain.base_pos)
    if dist_to_target > total_length:
        # Target is unreachable: stretch toward it
        # Why handle this case?  Without it, the algorithm would oscillate
        # forever trying to reach an impossible target.  Instead, we point
        # the chain directly at the target (fully extended).
        direction = (target - chain.base_pos) / dist_to_target
        angle = np.arctan2(direction[1], direction[0])
        chain.angles[:] = 0
        chain.angles[0] = angle
        return False

    # Start from current joint positions
    joints = chain.compute_joints()

    for iteration in range(max_iterations):
        # Check convergence
        end_pos = joints[-1]
        if np.linalg.norm(end_pos - target) < tolerance:
            break

        # --- Forward pass: end -> base ---
        # Start by placing the end effector at the target
        joints[-1] = target.copy()
        for i in range(n - 1, -1, -1):
            # Move joint i toward joint i+1, maintaining bone length
            direction = joints[i] - joints[i + 1]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction = direction / dist
            else:
                direction = np.array([1, 0])
            joints[i] = joints[i + 1] + direction * bone_lengths[i]

        # --- Backward pass: base -> end ---
        # Fix the base position (it must stay anchored)
        joints[0] = chain.base_pos.copy()
        for i in range(n):
            direction = joints[i + 1] - joints[i]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction = direction / dist
            else:
                direction = np.array([1, 0])
            joints[i + 1] = joints[i] + direction * bone_lengths[i]

    # Convert joint positions back to angles for the FK chain
    # Why convert back?  The FK chain stores joint angles, not positions.
    # We need to update the angles so that compute_joints() matches
    # the positions FABRIK computed.
    cumulative_angle = 0
    for i in range(n):
        direction = joints[i + 1] - joints[i]
        absolute_angle = np.arctan2(direction[1], direction[0])
        chain.angles[i] = absolute_angle - cumulative_angle
        cumulative_angle = absolute_angle

    return np.linalg.norm(joints[-1] - target) < tolerance


# ---------------------------------------------------------------------------
# 5. Animated FK/IK Demo
# ---------------------------------------------------------------------------

def demo_fk_ik_comparison():
    """Animate FK and IK side by side.

    Left panel: FK mode -- joint angles rotate cyclically
    Right panel: IK mode -- end effector tracks a moving target

    Why compare them?  FK and IK solve opposite problems:
    - FK: "given these joint angles, where is the hand?"
    - IK: "the hand should be here; what angles achieve that?"
    Understanding both is essential for character animation.
    """
    bone_lengths = [2.0, 1.5, 1.0]

    fk_chain = FKChain(bone_lengths, base_pos=np.array([0, 0]))
    ik_chain = FKChain(bone_lengths, base_pos=np.array([0, 0]))

    fig, (ax_fk, ax_ik) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Forward Kinematics vs Inverse Kinematics",
                 fontsize=14, fontweight='bold')

    for ax, title in [(ax_fk, "FK: Manual Joint Angles"),
                       (ax_ik, "IK: FABRIK Solver")]:
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12)

    # FK elements
    fk_line, = ax_fk.plot([], [], 'o-', color='steelblue', linewidth=3,
                           markersize=8, markerfacecolor='white',
                           markeredgecolor='steelblue', markeredgewidth=2)
    fk_trace, = ax_fk.plot([], [], 'b.', markersize=1, alpha=0.3)
    fk_trace_data = []

    # IK elements
    ik_line, = ax_ik.plot([], [], 'o-', color='forestgreen', linewidth=3,
                           markersize=8, markerfacecolor='white',
                           markeredgecolor='forestgreen', markeredgewidth=2)
    ik_target, = ax_ik.plot([], [], 'rx', markersize=15, markeredgewidth=3)
    ik_trace, = ax_ik.plot([], [], 'g.', markersize=1, alpha=0.3)
    ik_trace_data = []

    total_frames = 300

    def update(frame):
        t = frame / total_frames * 2 * np.pi

        # --- FK: cycle joint angles ---
        fk_chain.angles[0] = np.sin(t) * 0.8 + np.pi / 4
        fk_chain.angles[1] = np.sin(t * 1.5) * 0.6
        fk_chain.angles[2] = np.cos(t * 2) * 0.4

        fk_joints = fk_chain.compute_joints()
        fk_line.set_data(fk_joints[:, 0], fk_joints[:, 1])

        # Trace the end effector path
        fk_trace_data.append(fk_joints[-1].copy())
        if len(fk_trace_data) > 1:
            trace_arr = np.array(fk_trace_data)
            fk_trace.set_data(trace_arr[:, 0], trace_arr[:, 1])

        # --- IK: track a moving target ---
        # The target moves in a figure-8 pattern
        target = np.array([
            3.0 * np.sin(t),
            2.0 * np.sin(2 * t) + 2.0
        ])

        fabrik_solve(ik_chain, target)
        ik_joints = ik_chain.compute_joints()
        ik_line.set_data(ik_joints[:, 0], ik_joints[:, 1])
        ik_target.set_data([target[0]], [target[1]])

        ik_trace_data.append(ik_joints[-1].copy())
        if len(ik_trace_data) > 1:
            trace_arr = np.array(ik_trace_data)
            ik_trace.set_data(trace_arr[:, 0], trace_arr[:, 1])

        return fk_line, fk_trace, ik_line, ik_target, ik_trace

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=33, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# 6. Static FK/IK visualization
# ---------------------------------------------------------------------------

def demo_fk_ik_static():
    """Show FK and IK in static diagrams for documentation."""
    bone_lengths = [2.0, 1.5, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FK vs IK: How They Work", fontsize=14, fontweight='bold')

    # FK demo: multiple poses from different angles
    ax = axes[0]
    ax.set_title("FK: Set angles -> get position", fontsize=11)
    ax.set_xlim(-5, 6)
    ax.set_ylim(-3, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    angle_sets = [
        ([0.5, 0.3, 0.2], 'blue', 'Pose A'),
        ([1.0, -0.4, 0.6], 'red', 'Pose B'),
        ([0.2, 0.8, -0.3], 'green', 'Pose C'),
    ]

    for angles, color, label in angle_sets:
        chain = FKChain(bone_lengths)
        chain.angles = np.array(angles)
        joints = chain.compute_joints()
        ax.plot(joints[:, 0], joints[:, 1], 'o-', color=color,
                linewidth=2, markersize=6, label=label)

    ax.legend(fontsize=8)

    # IK demo: same target, chain reaches it
    ax = axes[1]
    ax.set_title("IK: Set target -> solve angles", fontsize=11)
    ax.set_xlim(-5, 6)
    ax.set_ylim(-3, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    targets = [
        np.array([3, 2]),
        np.array([-2, 3]),
        np.array([1, 4]),
    ]
    colors = ['blue', 'red', 'green']

    for target, color in zip(targets, colors):
        chain = FKChain(bone_lengths)
        reached = fabrik_solve(chain, target)
        joints = chain.compute_joints()
        ax.plot(joints[:, 0], joints[:, 1], 'o-', color=color,
                linewidth=2, markersize=6)
        marker = 'x' if not reached else '*'
        ax.plot(target[0], target[1], marker, color=color,
                markersize=12, markeredgewidth=2)

    ax.plot([], [], 'k*', markersize=12, label='Target (reachable)')
    ax.plot([], [], 'kx', markersize=12, label='Target (unreachable)')
    ax.legend(fontsize=8)

    # IK reachability
    ax = axes[2]
    ax.set_title("IK: Reachable workspace", fontsize=11)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw reachable workspace as circles
    total_len = sum(bone_lengths)
    min_len = max(bone_lengths[0] - sum(bone_lengths[1:]),
                  0)  # Minimum reach

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(total_len * np.cos(theta), total_len * np.sin(theta),
            'g-', linewidth=2, label=f'Max reach ({total_len:.1f})')
    if min_len > 0:
        ax.plot(min_len * np.cos(theta), min_len * np.sin(theta),
                'r--', linewidth=1, label=f'Min reach ({min_len:.1f})')
    ax.fill_between(total_len * np.cos(theta), total_len * np.sin(theta),
                    alpha=0.1, color='green')
    ax.plot(0, 0, 'ko', markersize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_11_fk_ik.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Keyframe animation of a bouncing ball
# ---------------------------------------------------------------------------

def demo_keyframe_animation():
    """Animate a ball using keyframe interpolation with squash & stretch.

    This demonstrates practical keyframe animation: the ball follows
    a path defined by keyframes, with Bezier easing and the classic
    squash-and-stretch principle from Disney's 12 principles of animation.
    """
    # Keyframes: (time, x, y, scale_x, scale_y)
    # The ball bounces with decreasing height and squashes on impact
    keyframes = [
        (0.0,  -3, 5.0, 1.0, 1.0),  # Start high
        (0.5,  -1, 0.3, 1.3, 0.7),  # First bounce (squash)
        (0.6,  -0.5, 0.5, 1.0, 1.0), # Release
        (1.2,   1, 3.0, 0.9, 1.1),  # Peak (slight stretch)
        (1.7,   2, 0.3, 1.2, 0.8),  # Second bounce
        (1.8,   2.3, 0.5, 1.0, 1.0), # Release
        (2.3,   3, 2.0, 0.95, 1.05), # Smaller peak
        (2.7,   3.5, 0.3, 1.15, 0.85), # Third bounce
        (3.0,   4, 0.5, 1.0, 1.0),  # Settle
    ]

    total_duration = keyframes[-1][0]
    fps = 60
    total_frames = int(total_duration * fps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-5, 6)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_title("Keyframe Animation: Bouncing Ball with Squash & Stretch",
                 fontsize=12, color='white')
    ax.tick_params(colors='gray')

    # Ground line
    ax.axhline(y=0, color='#444', linewidth=2)

    # Ball (ellipse to show squash/stretch)
    from matplotlib.patches import Ellipse
    ball = Ellipse((0, 0), 0.8, 0.8, color='#ff6b35', zorder=5)
    ax.add_patch(ball)

    # Shadow
    shadow = Ellipse((0, 0), 0.8, 0.15, color='black', alpha=0.3, zorder=1)
    ax.add_patch(shadow)

    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], '.', color='#ff6b35', alpha=0.2, markersize=2)

    def find_keyframe_segment(t):
        """Find which two keyframes bracket time t."""
        for i in range(len(keyframes) - 1):
            if keyframes[i][0] <= t <= keyframes[i + 1][0]:
                return i
        return len(keyframes) - 2

    def update(frame):
        t = frame / fps

        seg = find_keyframe_segment(t)
        kf0 = keyframes[seg]
        kf1 = keyframes[seg + 1]

        # Normalize t within this segment
        segment_duration = kf1[0] - kf0[0]
        local_t = (t - kf0[0]) / segment_duration if segment_duration > 0 else 0
        local_t = np.clip(local_t, 0, 1)

        # Apply ease-in-out
        eased_t = ease_in_out(local_t)

        # Interpolate all properties
        x = lerp(kf0[1], kf1[1], eased_t)
        y = lerp(kf0[2], kf1[2], eased_t)
        sx = lerp(kf0[3], kf1[3], eased_t)
        sy = lerp(kf0[4], kf1[4], eased_t)

        ball.set_center((x, y + 0.4 * sy))
        ball.width = 0.8 * sx
        ball.height = 0.8 * sy

        # Shadow scales inversely with height
        shadow.set_center((x, 0.05))
        shadow_scale = max(0.3, 1.0 - y * 0.1)
        shadow.width = 0.8 * shadow_scale
        shadow.set_alpha(0.3 * shadow_scale)

        trail_x.append(x)
        trail_y.append(y + 0.4 * sy)
        trail_line.set_data(trail_x, trail_y)

        return ball, shadow, trail_line

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=1000/fps, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Animation: Keyframes, Quaternion Slerp, FK, and IK")
    print("=" * 60)

    print("\n[1/4] Interpolation methods comparison...")
    demo_interpolation()

    print("\n[2/4] Quaternion slerp...")
    demo_slerp()

    print("\n[3/4] FK vs IK static comparison...")
    demo_fk_ik_static()

    print("\n[4/4] Animated FK/IK (close window to continue)...")
    anim1 = demo_fk_ik_comparison()

    print("\n[Bonus] Keyframe bouncing ball animation...")
    anim2 = demo_keyframe_animation()

    print("\nDone!")


if __name__ == "__main__":
    main()
