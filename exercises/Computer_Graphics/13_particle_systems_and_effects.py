"""
Exercises for Lesson 13: Particle Systems and Effects
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np

matplotlib_available = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    pass


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


# ---------------------------------------------------------------------------
# Exercise 1 -- Particle Fountain with Ground Bounce
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Create a particle system that emits particles upward with initial velocity,
    applies gravity, and has particles bounce off a ground plane (y=0) with
    70% restitution.
    """
    np.random.seed(42)
    num_particles = 200
    dt = 1.0 / 60.0
    gravity = np.array([0, -9.81, 0])
    restitution = 0.7
    ground_y = 0.0

    # Initialize particles
    positions = np.zeros((num_particles, 3))
    velocities = np.zeros((num_particles, 3))
    alive = np.zeros(num_particles, dtype=bool)
    ages = np.zeros(num_particles)
    lifetimes = np.random.uniform(2.0, 5.0, num_particles)

    emit_idx = 0
    emit_rate = 30  # particles per second
    emit_accum = 0.0

    total_frames = 300  # 5 seconds
    bounce_count = 0
    max_height = 0

    for frame in range(total_frames):
        # Emit new particles
        emit_accum += emit_rate * dt
        to_emit = int(emit_accum)
        emit_accum -= to_emit

        for _ in range(to_emit):
            if emit_idx >= num_particles:
                break
            positions[emit_idx] = [0, 0, 0]
            speed = np.random.uniform(5, 10)
            angle = np.random.uniform(-0.3, 0.3)
            velocities[emit_idx] = [
                speed * np.sin(angle),
                speed * np.cos(np.random.uniform(0.3, 0.8)),
                speed * np.sin(np.random.uniform(-0.3, 0.3))
            ]
            alive[emit_idx] = True
            ages[emit_idx] = 0.0
            emit_idx += 1

        # Update
        for i in range(num_particles):
            if not alive[i]:
                continue

            ages[i] += dt
            if ages[i] > lifetimes[i]:
                alive[i] = False
                continue

            # Symplectic Euler
            velocities[i] += gravity * dt
            positions[i] += velocities[i] * dt

            # Ground bounce
            if positions[i, 1] < ground_y:
                positions[i, 1] = ground_y
                velocities[i, 1] = -velocities[i, 1] * restitution
                bounce_count += 1

            max_height = max(max_height, positions[i, 1])

    alive_count = np.sum(alive)
    print(f"  Particle fountain simulation:")
    print(f"    Total particles emitted: {emit_idx}")
    print(f"    Still alive: {alive_count}")
    print(f"    Total bounces: {bounce_count}")
    print(f"    Max height reached: {max_height:.2f}")
    print(f"    Restitution: {restitution}")

    alive_pos = positions[alive]
    if len(alive_pos) > 0:
        print(f"    Alive Y range: [{alive_pos[:,1].min():.2f}, {alive_pos[:,1].max():.2f}]")


# ---------------------------------------------------------------------------
# Exercise 2 -- Vortex Attractor
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Add a vortex force to a fire-like particle system. Experiment with
    different strengths and axis orientations.
    """
    np.random.seed(7)

    class Particle:
        def __init__(self, pos, vel, lifetime):
            self.pos = np.array(pos, dtype=float)
            self.vel = np.array(vel, dtype=float)
            self.age = 0.0
            self.lifetime = lifetime
            self.alive = True

    def vortex_force(pos, center, axis, strength):
        """Compute vortex force that swirls particles around an axis."""
        r = pos - center
        return strength * np.cross(axis, r)

    configs = [
        ("Y-axis, strength=3", np.array([0, 1, 0]), 3.0),
        ("Y-axis, strength=8", np.array([0, 1, 0]), 8.0),
        ("Tilted axis, strength=5", normalize(np.array([1, 2, 0])), 5.0),
    ]

    for label, axis, strength in configs:
        particles = []
        gravity = np.array([0, -2.0, 0])
        dt = 1.0 / 60.0
        center = np.array([0, 0, 0])

        # Emit 100 particles upward
        for _ in range(100):
            speed = np.random.uniform(2, 5)
            angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([0.3 * np.cos(angle), speed, 0.3 * np.sin(angle)])
            particles.append(Particle([0, 0, 0], vel, np.random.uniform(1, 3)))

        # Simulate
        for frame in range(120):
            for p in particles:
                if not p.alive:
                    continue
                p.age += dt
                if p.age > p.lifetime:
                    p.alive = False
                    continue
                f = gravity + vortex_force(p.pos, center, axis, strength)
                f += -0.5 * p.vel  # drag
                p.vel += f * dt
                p.pos += p.vel * dt

        alive_p = [p for p in particles if p.alive]
        if alive_p:
            xs = [p.pos[0] for p in alive_p]
            zs = [p.pos[2] for p in alive_p]
            spread = np.sqrt(np.var(xs) + np.var(zs))
            avg_y = np.mean([p.pos[1] for p in alive_p])
            print(f"  {label}: {len(alive_p)} alive, "
                  f"lateral spread={spread:.2f}, avg_y={avg_y:.2f}")
        else:
            print(f"  {label}: all particles expired")

    print("  Stronger vortex -> wider lateral spread and more swirling motion.")
    print("  Tilted axis creates asymmetric swirl patterns.")


# ---------------------------------------------------------------------------
# Exercise 3 -- Integration Comparison
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Simulate a particle under gravity using (a) Euler, (b) Symplectic Euler,
    and (c) Verlet integration. Compare trajectories and energy conservation
    over 10 seconds. Plot total energy over time.
    """
    g = 9.81
    dt = 0.05
    total_time = 10.0
    steps = int(total_time / dt)
    m = 1.0

    # Initial conditions: toss upward
    y0 = 0.0
    v0 = 15.0

    def total_energy(y, v):
        return 0.5 * m * v ** 2 + m * g * y

    E0 = total_energy(y0, v0)

    # (a) Euler
    y_e, v_e = y0, v0
    euler_energies = []
    for _ in range(steps):
        euler_energies.append(total_energy(y_e, v_e))
        y_e_new = y_e + v_e * dt
        v_e_new = v_e - g * dt
        y_e, v_e = y_e_new, v_e_new
        # Ground bounce
        if y_e < 0:
            y_e = 0
            v_e = abs(v_e)

    # (b) Symplectic Euler
    y_s, v_s = y0, v0
    symp_energies = []
    for _ in range(steps):
        symp_energies.append(total_energy(y_s, v_s))
        v_s = v_s - g * dt     # velocity first
        y_s = y_s + v_s * dt   # then position with new velocity
        if y_s < 0:
            y_s = 0
            v_s = abs(v_s)

    # (c) Verlet
    y_v = y0
    y_v_prev = y0 - v0 * dt  # approximate previous position
    verlet_energies = []
    for _ in range(steps):
        v_approx = (y_v - y_v_prev) / dt
        verlet_energies.append(total_energy(y_v, v_approx))
        y_v_new = 2 * y_v - y_v_prev - g * dt * dt
        y_v_prev = y_v
        y_v = y_v_new
        if y_v < 0:
            y_v = 0
            y_v_prev = y_v + abs(y_v - y_v_prev) * 0.7

    euler_drift = abs(euler_energies[-1] - E0) / E0 * 100
    symp_drift = abs(symp_energies[-1] - E0) / E0 * 100
    verlet_drift = abs(verlet_energies[-1] - E0) / E0 * 100

    print(f"  Integration comparison over {total_time}s (dt={dt}):")
    print(f"  Initial energy: {E0:.4f}")
    print(f"  {'Method':>20s}  {'Final E':>10s}  {'Drift %':>10s}")
    print(f"  {'Euler':>20s}  {euler_energies[-1]:10.4f}  {euler_drift:10.4f}%")
    print(f"  {'Symplectic Euler':>20s}  {symp_energies[-1]:10.4f}  {symp_drift:10.4f}%")
    print(f"  {'Verlet':>20s}  {verlet_energies[-1]:10.4f}  {verlet_drift:10.4f}%")
    print(f"  Symplectic Euler and Verlet conserve energy better than standard Euler.")

    if matplotlib_available:
        t = np.arange(steps) * dt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, euler_energies, label='Euler', alpha=0.8)
        ax.plot(t, symp_energies, label='Symplectic Euler', alpha=0.8)
        ax.plot(t, verlet_energies, label='Verlet', alpha=0.8)
        ax.axhline(E0, color='k', linestyle='--', label=f'E0 = {E0:.2f}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Energy (J)')
        ax.set_title('Energy Conservation Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('integration_energy.png', dpi=100)
        plt.close()
        print("  Saved integration_energy.png")


# ---------------------------------------------------------------------------
# Exercise 4 -- Firework
# ---------------------------------------------------------------------------

def exercise_4():
    """
    Implement a firework: one particle rises, then bursts into 100 sub-
    particles that spread radially, then each sub-particle fades and falls.
    """
    np.random.seed(42)
    dt = 1.0 / 60.0
    gravity = np.array([0, -9.81, 0])

    # Phase 1: Rising rocket
    rocket_pos = np.array([0.0, 0.0, 0.0])
    rocket_vel = np.array([0.5, 25.0, 0.0])
    rocket_lifetime = 2.0
    rocket_age = 0.0

    print("  Phase 1: Rocket rising...")
    while rocket_age < rocket_lifetime:
        rocket_vel += gravity * dt
        rocket_pos += rocket_vel * dt
        rocket_age += dt

    burst_pos = rocket_pos.copy()
    print(f"    Burst position: ({burst_pos[0]:.1f}, {burst_pos[1]:.1f}, {burst_pos[2]:.1f})")

    # Phase 2: Burst into 100 sub-particles
    n_sub = 100
    sub_pos = np.tile(burst_pos, (n_sub, 1))
    sub_vel = np.zeros((n_sub, 3))
    sub_alive = np.ones(n_sub, dtype=bool)
    sub_ages = np.zeros(n_sub)
    sub_lifetimes = np.random.uniform(1.0, 3.0, n_sub)
    sub_colors = np.zeros((n_sub, 3))

    # Radial burst velocities
    for i in range(n_sub):
        direction = normalize(np.random.randn(3))
        speed = np.random.uniform(5, 15)
        sub_vel[i] = direction * speed
        # Random colors
        sub_colors[i] = np.random.uniform(0.5, 1.0, 3)

    print(f"  Phase 2: Burst into {n_sub} sub-particles")

    # Simulate sub-particles
    total_frames = 180  # 3 seconds
    for frame in range(total_frames):
        for i in range(n_sub):
            if not sub_alive[i]:
                continue
            sub_ages[i] += dt
            if sub_ages[i] > sub_lifetimes[i]:
                sub_alive[i] = False
                continue
            # Apply gravity + drag
            drag = -0.3 * sub_vel[i]
            sub_vel[i] += (gravity + drag) * dt
            sub_pos[i] += sub_vel[i] * dt

    alive_mask = sub_alive
    final_alive = np.sum(alive_mask)
    if final_alive > 0:
        alive_positions = sub_pos[alive_mask]
        spread = np.std(alive_positions, axis=0)
        min_y = alive_positions[:, 1].min()
        max_y = alive_positions[:, 1].max()
    else:
        spread = np.zeros(3)
        min_y = max_y = 0

    print(f"    After 3 seconds: {final_alive} sub-particles still alive")
    print(f"    Spread (std): ({spread[0]:.1f}, {spread[1]:.1f}, {spread[2]:.1f})")
    print(f"    Y range: [{min_y:.1f}, {max_y:.1f}]")
    print(f"    Sub-particles spread radially, then fall under gravity.")


# ---------------------------------------------------------------------------
# Exercise 5 -- Simple 2D Ray Marcher
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Implement a 2D ray marcher that renders a circular fog volume. The density
    falls off with distance from the center (Gaussian). Visualize as a 1D
    scanline.
    """
    # Fog volume: circle at (5, 0) with radius 3
    fog_center = np.array([5.0, 0.0])
    fog_radius = 3.0
    fog_density_peak = 2.0
    fog_sigma = 1.5  # Gaussian falloff

    def sample_density(pos):
        """Gaussian density field centered on fog_center."""
        dist = np.linalg.norm(pos - fog_center)
        if dist > fog_radius:
            return 0.0
        return fog_density_peak * np.exp(-dist ** 2 / (2 * fog_sigma ** 2))

    # Ray march along X axis (y=0)
    ray_origin = np.array([0.0, 0.0])
    ray_dir = np.array([1.0, 0.0])
    max_dist = 12.0
    step_size = 0.05
    num_steps = int(max_dist / step_size)

    color = 0.0  # accumulated brightness
    transmittance = 1.0
    fog_color = 1.0  # white fog

    transmittance_log = []
    color_log = []
    positions = []

    for i in range(num_steps):
        pos = ray_origin + i * step_size * ray_dir
        positions.append(pos[0])
        density = sample_density(pos)

        if density > 0:
            # Beer-Lambert extinction
            extinction = np.exp(-density * step_size)
            # Accumulate color
            color += transmittance * (1 - extinction) * fog_color
            transmittance *= extinction

        transmittance_log.append(transmittance)
        color_log.append(color)

        if transmittance < 0.01:
            # Fill remaining
            remaining = num_steps - i - 1
            transmittance_log.extend([transmittance] * remaining)
            color_log.extend([color] * remaining)
            positions.extend([pos[0] + j * step_size for j in range(1, remaining + 1)])
            break

    print(f"  2D Ray Marcher (fog volume):")
    print(f"    Fog center: {fog_center}, radius: {fog_radius}")
    print(f"    Density: Gaussian (peak={fog_density_peak}, sigma={fog_sigma})")
    print(f"    Step size: {step_size}")
    print(f"    Final transmittance: {transmittance:.4f}")
    print(f"    Final accumulated color: {color:.4f}")
    print(f"    Opacity: {1 - transmittance:.4f}")

    if matplotlib_available:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        xs = np.array(positions[:len(transmittance_log)])
        ax1.plot(xs, transmittance_log, 'b-')
        ax1.set_ylabel('Transmittance')
        ax1.set_title('Ray Marching Through Gaussian Fog')
        ax1.axvspan(fog_center[0] - fog_radius, fog_center[0] + fog_radius,
                     alpha=0.1, color='gray', label='Fog volume')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(xs, color_log, 'r-')
        ax2.set_xlabel('Distance along ray')
        ax2.set_ylabel('Accumulated color')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ray_march_fog.png', dpi=100)
        plt.close()
        print("  Saved ray_march_fog.png")


# ---------------------------------------------------------------------------
# Exercise 6 -- GPU Particle System Design
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Design the data layout for a GPU compute shader particle system
    supporting 1 million particles. Specify buffer structure, work group
    size, and how to handle emission and death.
    """

    num_particles = 1_000_000

    # Buffer layout: Structure of Arrays (SoA) for coalesced GPU memory access
    # Each array is a separate SSBO binding
    buffers = {
        "position_x": ("float32", 4 * num_particles),
        "position_y": ("float32", 4 * num_particles),
        "position_z": ("float32", 4 * num_particles),
        "velocity_x": ("float32", 4 * num_particles),
        "velocity_y": ("float32", 4 * num_particles),
        "velocity_z": ("float32", 4 * num_particles),
        "age":        ("float32", 4 * num_particles),
        "lifetime":   ("float32", 4 * num_particles),
        "color_r":    ("float32", 4 * num_particles),
        "color_g":    ("float32", 4 * num_particles),
        "color_b":    ("float32", 4 * num_particles),
        "color_a":    ("float32", 4 * num_particles),
        "size":       ("float32", 4 * num_particles),
        "alive":      ("uint32",  4 * num_particles),
    }

    total_bytes = sum(b[1] for b in buffers.values())
    total_mb = total_bytes / (1024 * 1024)

    work_group_size = 256
    num_work_groups = (num_particles + work_group_size - 1) // work_group_size

    print(f"  GPU Particle System Design: {num_particles:,} particles")
    print(f"\n  Buffer Layout (Structure of Arrays for coalesced access):")
    print(f"  {'Buffer':>15s}  {'Type':>8s}  {'Size (MB)':>10s}")
    print(f"  {'---':>15s}  {'---':>8s}  {'---':>10s}")
    for name, (dtype, size) in buffers.items():
        print(f"  {name:>15s}  {dtype:>8s}  {size/1024/1024:10.2f}")
    print(f"  {'TOTAL':>15s}  {'':>8s}  {total_mb:10.2f}")

    print(f"\n  Compute Dispatch:")
    print(f"    Work group size: {work_group_size}")
    print(f"    Num work groups: {num_work_groups:,}")
    print(f"    Total threads:   {num_work_groups * work_group_size:,}")

    print(f"\n  Emission Strategy:")
    print(f"    - Atomic counter: tracks next free particle index")
    print(f"    - Emission kernel: dispatched with N = particles_to_emit")
    print(f"    - Each thread atomically increments counter, initializes one particle")
    print(f"    - Counter wraps around (ring buffer) for particle recycling")

    print(f"\n  Death/Recycling Strategy:")
    print(f"    - Update kernel checks age >= lifetime, marks alive = 0")
    print(f"    - Stream compaction (prefix sum on alive flags) packs live particles")
    print(f"    - Alternatively: leave dead particles in place, emission overwrites them")
    print(f"    - Dead particle index list via atomic append for O(1) recycling")

    print(f"\n  Rendering:")
    print(f"    - Render as GL_POINTS with gl_PointSize set in vertex shader")
    print(f"    - Or generate billboard quads in a geometry/mesh shader")
    print(f"    - Sort by depth for alpha blending (bitonic sort on GPU)")

    # Simulate what the update kernel would do per thread
    print(f"\n  Per-thread update pseudocode:")
    print(f"    uint idx = gl_GlobalInvocationID.x;")
    print(f"    if (idx >= numParticles || alive[idx] == 0) return;")
    print(f"    age[idx] += dt;")
    print(f"    if (age[idx] >= lifetime[idx]) {{ alive[idx] = 0; return; }}")
    print(f"    velocity_y[idx] += gravity * dt;")
    print(f"    position_x[idx] += velocity_x[idx] * dt;")
    print(f"    position_y[idx] += velocity_y[idx] * dt;")
    print(f"    position_z[idx] += velocity_z[idx] * dt;")
    print(f"    float t = age[idx] / lifetime[idx];")
    print(f"    color_a[idx] = 1.0 - t;  // fade out")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Exercise 1: Particle Fountain with Ground Bounce ===")
    exercise_1()

    print("\n=== Exercise 2: Vortex Attractor ===")
    exercise_2()

    print("\n=== Exercise 3: Integration Comparison ===")
    exercise_3()

    print("\n=== Exercise 4: Firework ===")
    exercise_4()

    print("\n=== Exercise 5: Simple 2D Ray Marcher ===")
    exercise_5()

    print("\n=== Exercise 6: GPU Particle System Design ===")
    exercise_6()

    print("\nAll exercises completed!")
