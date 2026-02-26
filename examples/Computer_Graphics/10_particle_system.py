"""
Particle System
================

Implements a configurable particle system with:
1. Particle emitter with tunable parameters
2. Forces: gravity, wind, drag
3. Verlet integration (more stable than Euler for large timesteps)
4. Particle lifecycle: spawn, age, die
5. Animated matplotlib visualization
6. Multiple effects: fountain, explosion, fire

Particle systems are the foundation of visual effects in games and
film -- smoke, fire, rain, sparks, magic spells, and explosions are
all typically particle-based.  The key insight: complex-looking
phenomena emerge from many simple particles following simple rules.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# 1. Particle data (Structure of Arrays for performance)
# ---------------------------------------------------------------------------


class ParticlePool:
    """Manages a fixed-size pool of particles using Structure-of-Arrays layout.

    Why SoA instead of Array-of-Structures?  When we update ALL positions,
    having all x-coordinates contiguous in memory is faster (cache-friendly
    vectorized NumPy operations) than having each particle's data scattered.
    This is the same layout real game engines use for cache performance.

    Why a fixed-size pool?  Dynamic allocation per-particle is slow.
    Pre-allocating a maximum capacity and reusing dead particles avoids
    memory allocation during gameplay -- critical for 60fps real-time apps.
    """

    def __init__(self, max_particles: int):
        self.max_particles = max_particles

        # Position: (N, 2) for 2D
        self.pos = np.zeros((max_particles, 2))
        # Previous position (for Verlet integration)
        self.prev_pos = np.zeros((max_particles, 2))
        # Velocity (used for initial setup, then Verlet takes over)
        self.vel = np.zeros((max_particles, 2))

        # Particle lifetime
        self.age = np.zeros(max_particles)        # Current age (seconds)
        self.max_age = np.zeros(max_particles)     # Lifetime before death

        # Visual properties
        self.color = np.zeros((max_particles, 4))  # RGBA
        self.size = np.zeros(max_particles)

        # Status
        self.alive = np.zeros(max_particles, dtype=bool)

    @property
    def count_alive(self) -> int:
        return np.sum(self.alive)

    def find_dead_index(self) -> Optional[int]:
        """Find the first dead particle slot for reuse.

        Why linear scan?  For a small pool (<10k), this is fast enough.
        Production systems maintain a free-list or ring buffer for O(1) access.
        """
        dead_indices = np.where(~self.alive)[0]
        if len(dead_indices) == 0:
            return None
        return dead_indices[0]


# ---------------------------------------------------------------------------
# 2. Emitter configuration
# ---------------------------------------------------------------------------

@dataclass
class EmitterConfig:
    """Configuration for a particle emitter.

    Why so many parameters?  Particle systems derive their visual variety
    entirely from parameter tuning.  A fountain, fire, and explosion can
    all use the same code -- only the parameters differ.

    Each parameter with a _var suffix adds random variation (+/- var)
    to create natural-looking diversity.
    """
    # Emission
    rate: float = 50.0              # Particles per second
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    # Initial velocity
    speed: float = 3.0
    speed_var: float = 0.5          # Random speed variation
    angle: float = 90.0             # Emission angle (degrees, 0 = right)
    angle_var: float = 15.0         # Random angle spread

    # Lifetime
    lifetime: float = 2.0           # Base lifetime (seconds)
    lifetime_var: float = 0.3       # Random lifetime variation

    # Visual
    start_color: np.ndarray = field(default_factory=lambda: np.array([1, 0.5, 0, 1.0]))
    end_color: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0.0]))
    start_size: float = 5.0
    end_size: float = 1.0

    # Forces
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.8]))
    wind: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    drag: float = 0.01             # Air resistance coefficient


# ---------------------------------------------------------------------------
# 3. Particle system
# ---------------------------------------------------------------------------

class ParticleSystem:
    """Core particle system with emission, physics, and lifecycle management.

    The update loop follows the standard pattern:
      1. Emit new particles
      2. Apply forces (gravity, wind, drag)
      3. Integrate motion (Verlet)
      4. Age and kill old particles
      5. Update visual properties (color fade, size change)
    """

    def __init__(self, config: EmitterConfig, max_particles: int = 2000):
        self.config = config
        self.pool = ParticlePool(max_particles)
        self.emit_accumulator = 0.0  # Sub-frame emission accumulator
        self.rng = np.random.RandomState(42)

    def emit(self, dt: float):
        """Spawn new particles based on emission rate.

        Why an accumulator?  If rate = 50 particles/sec but dt = 1/60 sec,
        we'd spawn 0.833 particles per frame.  The accumulator carries
        over fractional particles between frames, ensuring the average
        rate is correct without per-frame rounding errors.
        """
        self.emit_accumulator += self.config.rate * dt

        while self.emit_accumulator >= 1.0:
            self.emit_accumulator -= 1.0
            self._spawn_particle()

    def _spawn_particle(self):
        """Initialize a single new particle."""
        idx = self.pool.find_dead_index()
        if idx is None:
            return  # Pool is full -- drop the particle

        cfg = self.config
        pool = self.pool

        pool.alive[idx] = True
        pool.age[idx] = 0.0
        pool.pos[idx] = cfg.position.copy()

        # Randomized initial velocity
        speed = cfg.speed + self.rng.uniform(-cfg.speed_var, cfg.speed_var)
        angle = np.radians(cfg.angle + self.rng.uniform(-cfg.angle_var, cfg.angle_var))
        pool.vel[idx] = speed * np.array([np.cos(angle), np.sin(angle)])

        # Set previous position for Verlet (pos - vel*dt approximation)
        pool.prev_pos[idx] = pool.pos[idx] - pool.vel[idx] * (1/60)

        # Randomized lifetime
        pool.max_age[idx] = max(0.1, cfg.lifetime + self.rng.uniform(
            -cfg.lifetime_var, cfg.lifetime_var))

        # Initial visual properties
        pool.color[idx] = cfg.start_color.copy()
        pool.size[idx] = cfg.start_size

    def update(self, dt: float):
        """Update all alive particles for one timestep.

        Why process all particles vectorized?  NumPy operations on arrays
        are 10-100x faster than Python loops.  This is essential for
        smooth real-time particle systems with thousands of particles.
        """
        mask = self.pool.alive
        if not np.any(mask):
            return

        pool = self.pool
        cfg = self.config

        # --- Age and kill ---
        pool.age[mask] += dt
        expired = pool.age > pool.max_age
        pool.alive[expired] = False
        mask = pool.alive  # Refresh mask after killing

        if not np.any(mask):
            return

        # --- Compute forces ---
        # Acceleration from gravity + wind
        accel = cfg.gravity + cfg.wind

        # Drag force: opposes velocity, proportional to speed
        # F_drag = -drag * v  (simplified linear drag)
        # Why linear drag?  Quadratic drag (proportional to v^2) is more
        # physically accurate but linear is simpler and produces
        # acceptable visual results for particles.
        current_vel = pool.pos[mask] - pool.prev_pos[mask]
        drag_accel = -cfg.drag * current_vel / max(dt, 1e-6)

        total_accel = accel + drag_accel

        # --- Verlet integration ---
        # Why Verlet over Euler?  Verlet is second-order accurate
        # (vs Euler's first-order), time-reversible, and more stable
        # for stiff systems.  It also doesn't store velocity explicitly --
        # velocity is implicit in the position difference.
        #
        # Formula: x_new = 2*x - x_old + a*dt^2
        new_pos = 2 * pool.pos[mask] - pool.prev_pos[mask] + total_accel * dt * dt
        pool.prev_pos[mask] = pool.pos[mask].copy()
        pool.pos[mask] = new_pos

        # --- Update visual properties ---
        # Interpolate color and size based on normalized age (0 to 1)
        t = pool.age[mask] / pool.max_age[mask]
        t = np.clip(t, 0, 1)[:, np.newaxis]

        pool.color[mask] = (1 - t) * cfg.start_color + t * cfg.end_color
        pool.size[mask] = ((1 - t[:, 0]) * cfg.start_size + t[:, 0] * cfg.end_size)


# ---------------------------------------------------------------------------
# 4. Preset effects
# ---------------------------------------------------------------------------

def fountain_config() -> EmitterConfig:
    """Water fountain: particles shoot up, arc down under gravity."""
    return EmitterConfig(
        rate=80,
        position=np.array([0.0, 0.0]),
        speed=6.0, speed_var=1.0,
        angle=90, angle_var=12,
        lifetime=2.5, lifetime_var=0.5,
        start_color=np.array([0.3, 0.6, 1.0, 0.9]),
        end_color=np.array([0.1, 0.3, 0.8, 0.0]),
        start_size=6.0, end_size=2.0,
        gravity=np.array([0.0, -9.8]),
        wind=np.array([0.5, 0.0]),
        drag=0.02,
    )


def explosion_config() -> EmitterConfig:
    """Explosion: particles burst outward in all directions.

    Why high rate and short lifetime?  An explosion is a brief burst
    of many particles, not a continuous stream.  The rate is very high
    but only for a moment (handled by the burst_emit function below).
    """
    return EmitterConfig(
        rate=0,  # We'll use burst emission instead of continuous
        position=np.array([0.0, 0.0]),
        speed=8.0, speed_var=3.0,
        angle=0, angle_var=180,  # Full 360 degrees
        lifetime=1.5, lifetime_var=0.5,
        start_color=np.array([1.0, 0.8, 0.2, 1.0]),
        end_color=np.array([0.6, 0.1, 0.0, 0.0]),
        start_size=8.0, end_size=1.0,
        gravity=np.array([0.0, -3.0]),  # Less gravity for dramatic effect
        wind=np.array([0.0, 0.0]),
        drag=0.05,
    )


def fire_config() -> EmitterConfig:
    """Fire: particles rise with turbulent movement.

    Why negative gravity (upward)?  Fire particles are buoyant -- they
    rise due to convection.  The wind adds horizontal wobble for the
    characteristic flickering appearance.
    """
    return EmitterConfig(
        rate=120,
        position=np.array([0.0, 0.0]),
        speed=2.0, speed_var=1.0,
        angle=90, angle_var=25,
        lifetime=1.0, lifetime_var=0.3,
        start_color=np.array([1.0, 0.9, 0.3, 1.0]),
        end_color=np.array([0.8, 0.1, 0.0, 0.0]),
        start_size=10.0, end_size=2.0,
        gravity=np.array([0.0, 2.0]),   # Upward buoyancy
        wind=np.array([0.0, 0.0]),
        drag=0.1,  # High drag for slower, lazier movement
    )


def burst_emit(system: ParticleSystem, count: int):
    """Emit a burst of particles all at once (for explosion effects).

    Why separate from continuous emission?  An explosion isn't a steady
    stream -- it's a single event that creates many particles simultaneously.
    We manually spawn `count` particles in one go.
    """
    for _ in range(count):
        system._spawn_particle()


# ---------------------------------------------------------------------------
# 5. Animated visualization
# ---------------------------------------------------------------------------

def animate_effect(name: str, config: EmitterConfig,
                   duration: float = 4.0, burst: int = 0,
                   xlim=(-5, 5), ylim=(-2, 10)):
    """Run an animated particle effect.

    Parameters
    ----------
    name    : Display name
    config  : Emitter configuration
    duration: Animation duration in seconds
    burst   : If > 0, emit this many particles as an initial burst
    xlim, ylim : Plot bounds
    """
    system = ParticleSystem(config, max_particles=3000)
    dt = 1 / 60

    if burst > 0:
        burst_emit(system, burst)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')
    ax.set_title(f"Particle Effect: {name}", fontsize=13, color='white',
                 fontweight='bold')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_color('#333')

    scatter = ax.scatter([], [], s=[], c=[], alpha=1.0)
    count_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=10, color='white', verticalalignment='top')

    total_frames = int(duration / dt)

    def update(frame):
        # Emit new particles (continuous effects)
        system.emit(dt)

        # Update physics and lifecycle
        system.update(dt)

        # Gather alive particle data for rendering
        mask = system.pool.alive
        if np.any(mask):
            positions = system.pool.pos[mask]
            colors = system.pool.color[mask]
            sizes = system.pool.size[mask]

            scatter.set_offsets(positions)
            scatter.set_sizes(sizes ** 2)  # Scatter uses area, not radius
            scatter.set_color(colors)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        count_text.set_text(f"Alive: {system.pool.count_alive}")
        return scatter, count_text

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                   interval=dt * 1000, blit=False)
    plt.tight_layout()
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# 6. Static snapshot comparison
# ---------------------------------------------------------------------------

def capture_snapshot(config: EmitterConfig, duration: float = 2.0,
                     burst: int = 0) -> tuple:
    """Run a particle system for `duration` seconds and return final state."""
    system = ParticleSystem(config, max_particles=3000)
    dt = 1 / 60

    if burst > 0:
        burst_emit(system, burst)

    steps = int(duration / dt)
    for _ in range(steps):
        system.emit(dt)
        system.update(dt)

    mask = system.pool.alive
    if np.any(mask):
        return system.pool.pos[mask], system.pool.color[mask], system.pool.size[mask]
    return np.empty((0, 2)), np.empty((0, 4)), np.empty(0)


def demo_effects_comparison():
    """Show all three effects side by side as static snapshots.

    Why static snapshots?  They can be saved to image files and included
    in documentation, unlike animations which require a live viewer.
    """
    effects = [
        ("Fountain", fountain_config(), 2.0, 0, (-6, 6), (-2, 12)),
        ("Explosion", explosion_config(), 1.0, 200, (-12, 12), (-8, 12)),
        ("Fire", fire_config(), 1.5, 0, (-4, 4), (-1, 6)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Particle Effects: Snapshot Comparison", fontsize=14,
                 fontweight='bold')
    fig.patch.set_facecolor('#0a0a1a')

    for ax, (name, config, dur, burst, xlim, ylim) in zip(axes, effects):
        pos, colors, sizes = capture_snapshot(config, dur, burst)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a1a')
        ax.set_title(name, fontsize=12, color='white')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('#333')

        if len(pos) > 0:
            ax.scatter(pos[:, 0], pos[:, 1], s=sizes**2, c=colors, alpha=0.8)

        ax.text(0.02, 0.02, f"{len(pos)} particles",
                transform=ax.transAxes, fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_10_particles.png", dpi=100,
                facecolor=fig.get_facecolor())
    plt.show()


# ---------------------------------------------------------------------------
# 7. Individual animated demos
# ---------------------------------------------------------------------------

def demo_fountain():
    """Animate a water fountain."""
    print("Fountain effect (close window to continue)...")
    return animate_effect("Fountain", fountain_config(),
                          duration=5.0, xlim=(-6, 6), ylim=(-2, 12))


def demo_explosion():
    """Animate an explosion burst."""
    print("Explosion effect (close window to continue)...")
    return animate_effect("Explosion", explosion_config(),
                          duration=3.0, burst=300,
                          xlim=(-15, 15), ylim=(-10, 15))


def demo_fire():
    """Animate a fire."""
    print("Fire effect (close window to continue)...")
    return animate_effect("Fire", fire_config(),
                          duration=5.0, xlim=(-4, 4), ylim=(-1, 6))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Particle System")
    print("=" * 60)

    print("\n[1/4] Static comparison of all effects...")
    demo_effects_comparison()

    print("\n[2/4] Fountain animation...")
    anim1 = demo_fountain()

    print("\n[3/4] Explosion animation...")
    anim2 = demo_explosion()

    print("\n[4/4] Fire animation...")
    anim3 = demo_fire()

    print("\nDone!")


if __name__ == "__main__":
    main()
