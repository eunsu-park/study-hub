"""
Reynolds Flocking: Multi-Robot Swarm Simulation
================================================
Simulate emergent collective behavior from simple local rules.

Craig Reynolds (1987) showed that realistic flocking, schooling, and herding
behavior emerges from just three simple local rules applied to each agent (boid):

  1. Separation: steer away from nearby neighbors to avoid collisions
  2. Alignment: steer toward the average heading of nearby neighbors
  3. Cohesion: steer toward the average position of nearby neighbors

Each agent only uses local information (nearby neighbors within a radius),
yet the collective behavior appears coordinated and intelligent. This is a
classic example of emergence: complex global behavior from simple local rules.

We extend the basic model with:
  - Obstacle avoidance: steer away from obstacles
  - Goal seeking: steer toward a target waypoint
  - Speed limits and boundary enforcement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


class Boid:
    """A single agent in the swarm.

    Each boid has a position and velocity in 2D. At each time step, it
    computes a desired acceleration based on local rules and updates its state.
    """

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.acceleration = np.zeros(2)

    def apply_force(self, force: np.ndarray):
        """Accumulate a steering force.

        Forces are accumulated (summed) over one time step, then applied.
        This allows multiple behaviors to contribute to the final motion.
        """
        self.acceleration += force

    def update(self, dt: float, max_speed: float):
        """Integrate velocity and position using Euler method.

        We clamp the speed to max_speed to keep the simulation stable
        and prevent agents from accelerating indefinitely.
        """
        self.velocity += self.acceleration * dt

        # Clamp speed
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

        self.position += self.velocity * dt
        self.acceleration = np.zeros(2)  # Reset for next step


class SwarmSimulation:
    """Reynolds flocking simulation with obstacle avoidance and goal seeking.

    The simulation manages a collection of boids and computes their
    interactions at each time step. Behavior weights control the relative
    importance of each steering rule.
    """

    def __init__(self, n_boids: int = 50,
                 world_size: float = 100.0,
                 perception_radius: float = 15.0,
                 max_speed: float = 8.0,
                 max_force: float = 5.0):
        """
        Args:
            n_boids: Number of agents
            world_size: Square world dimension (0 to world_size)
            perception_radius: How far each boid can "see" neighbors
            max_speed: Maximum velocity magnitude
            max_force: Maximum steering force magnitude (prevents jerky motion)
        """
        self.world_size = world_size
        self.perception_radius = perception_radius
        self.max_speed = max_speed
        self.max_force = max_force

        # Behavior weights — these control the "personality" of the flock
        # Higher separation weight → more spread out
        # Higher cohesion weight → tighter groups
        # Higher alignment weight → more coordinated heading
        self.w_separation = 2.0
        self.w_alignment = 1.0
        self.w_cohesion = 1.0
        self.w_obstacle = 3.0    # Obstacle avoidance is high priority
        self.w_goal = 0.5        # Gentle pull toward goal
        self.w_boundary = 2.0    # Keep boids in bounds

        # Initialize boids with random positions and velocities
        self.boids = []
        for _ in range(n_boids):
            x = np.random.uniform(20, world_size - 20)
            y = np.random.uniform(20, world_size - 20)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, max_speed)
            self.boids.append(Boid(x, y, speed * np.cos(angle), speed * np.sin(angle)))

        # Obstacles (circles)
        self.obstacles = [
            (np.array([40.0, 50.0]), 8.0),   # (center, radius)
            (np.array([70.0, 30.0]), 6.0),
            (np.array([30.0, 75.0]), 5.0),
        ]

        # Goal waypoint
        self.goal = np.array([80.0, 80.0])

    def _limit_force(self, force: np.ndarray) -> np.ndarray:
        """Clamp a steering force to max_force magnitude.

        Without force limiting, dominant behaviors (like obstacle avoidance
        very close to an obstacle) would create unrealistically large
        accelerations.
        """
        mag = np.linalg.norm(force)
        if mag > self.max_force:
            force = force / mag * self.max_force
        return force

    def _get_neighbors(self, boid: Boid) -> list:
        """Find all boids within the perception radius.

        In a real implementation, you would use a spatial data structure
        (k-d tree, grid) for O(N log N) or O(N) neighbor queries.
        The naive O(N^2) approach here is fine for ~100 agents.
        """
        neighbors = []
        for other in self.boids:
            if other is boid:
                continue
            dist = np.linalg.norm(boid.position - other.position)
            if dist < self.perception_radius:
                neighbors.append(other)
        return neighbors

    def separation(self, boid: Boid, neighbors: list) -> np.ndarray:
        """Steer away from nearby neighbors to avoid crowding.

        The repulsive force is inversely proportional to distance:
        closer neighbors push harder. This creates a "personal space"
        around each agent.
        """
        if not neighbors:
            return np.zeros(2)

        steer = np.zeros(2)
        for other in neighbors:
            diff = boid.position - other.position
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                # Weight inversely by distance (closer = stronger push)
                steer += diff / dist / dist

        return self._limit_force(steer)

    def alignment(self, boid: Boid, neighbors: list) -> np.ndarray:
        """Steer toward the average heading of nearby neighbors.

        This creates coordinated movement: agents tend to fly in the same
        direction as their neighbors. Without alignment, the flock would
        be a disorganized cluster.
        """
        if not neighbors:
            return np.zeros(2)

        avg_vel = np.mean([n.velocity for n in neighbors], axis=0)
        # Desired velocity is the average neighbor velocity
        desired = avg_vel - boid.velocity
        return self._limit_force(desired)

    def cohesion(self, boid: Boid, neighbors: list) -> np.ndarray:
        """Steer toward the average position of nearby neighbors.

        This provides the "glue" that keeps the flock together. Without
        cohesion, alignment alone would cause agents to drift apart
        (parallel but diverging paths).
        """
        if not neighbors:
            return np.zeros(2)

        avg_pos = np.mean([n.position for n in neighbors], axis=0)
        desired = avg_pos - boid.position
        mag = np.linalg.norm(desired)
        if mag > 1e-6:
            desired = desired / mag * self.max_speed - boid.velocity
        return self._limit_force(desired)

    def obstacle_avoidance(self, boid: Boid) -> np.ndarray:
        """Steer away from circular obstacles.

        The avoidance force increases exponentially as the boid gets closer
        to an obstacle. This ensures strong avoidance at close range while
        not affecting distant boids.
        """
        steer = np.zeros(2)
        for center, radius in self.obstacles:
            diff = boid.position - center
            dist = np.linalg.norm(diff) - radius  # Distance to obstacle surface

            if dist < self.perception_radius and dist > 0:
                # Exponential repulsion: strong close up, weak far away
                force_mag = np.exp(-dist / 3.0)
                steer += diff / np.linalg.norm(diff) * force_mag
            elif dist <= 0:
                # Inside obstacle: strong push outward
                if np.linalg.norm(diff) > 1e-6:
                    steer += diff / np.linalg.norm(diff) * self.max_force

        return self._limit_force(steer)

    def goal_seeking(self, boid: Boid) -> np.ndarray:
        """Gentle attraction toward a goal waypoint.

        This gives the flock a global direction without overriding the
        local flocking behaviors. The force is proportional to distance
        but capped at max_force.
        """
        desired = self.goal - boid.position
        dist = np.linalg.norm(desired)
        if dist > 1e-6:
            # Scale desired velocity by distance (slow down near goal)
            speed = min(self.max_speed, dist * 0.1)
            desired = desired / dist * speed - boid.velocity
        return self._limit_force(desired)

    def boundary_force(self, boid: Boid) -> np.ndarray:
        """Keep boids within the world boundaries using soft walls.

        Instead of hard wrapping (which breaks flock continuity), we apply
        a repulsive force that increases as boids approach the boundary.
        """
        margin = 10.0
        steer = np.zeros(2)

        if boid.position[0] < margin:
            steer[0] = (margin - boid.position[0]) / margin * self.max_speed
        elif boid.position[0] > self.world_size - margin:
            steer[0] = (self.world_size - margin - boid.position[0]) / margin * self.max_speed

        if boid.position[1] < margin:
            steer[1] = (margin - boid.position[1]) / margin * self.max_speed
        elif boid.position[1] > self.world_size - margin:
            steer[1] = (self.world_size - margin - boid.position[1]) / margin * self.max_speed

        return self._limit_force(steer)

    def step(self, dt: float = 0.1):
        """Advance the simulation by one time step.

        The update is synchronous: we compute all forces based on the
        current state, then update all positions. This prevents the order
        of iteration from affecting the result (which would happen with
        asynchronous updates).
        """
        # Phase 1: Compute all forces
        for boid in self.boids:
            neighbors = self._get_neighbors(boid)

            # Apply all steering behaviors with weights
            sep = self.separation(boid, neighbors) * self.w_separation
            ali = self.alignment(boid, neighbors) * self.w_alignment
            coh = self.cohesion(boid, neighbors) * self.w_cohesion
            obs = self.obstacle_avoidance(boid) * self.w_obstacle
            goal = self.goal_seeking(boid) * self.w_goal
            boundary = self.boundary_force(boid) * self.w_boundary

            boid.apply_force(sep + ali + coh + obs + goal + boundary)

        # Phase 2: Update all positions
        for boid in self.boids:
            boid.update(dt, self.max_speed)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def demo_swarm():
    """Demonstrate Reynolds flocking with animated visualization."""
    print("=" * 60)
    print("Reynolds Flocking (Swarm Robotics) Demo")
    print("=" * 60)

    np.random.seed(42)
    sim = SwarmSimulation(n_boids=50, world_size=100.0)

    # Run simulation and collect snapshots
    n_steps = 500
    dt = 0.1
    snapshot_steps = [0, 50, 200, 499]
    snapshots = {}

    for step in range(n_steps):
        if step in snapshot_steps:
            positions = np.array([b.position for b in sim.boids])
            velocities = np.array([b.velocity for b in sim.boids])
            snapshots[step] = (positions.copy(), velocities.copy())
        sim.step(dt)

    # Final snapshot
    positions = np.array([b.position for b in sim.boids])
    velocities = np.array([b.velocity for b in sim.boids])
    snapshots[n_steps - 1] = (positions.copy(), velocities.copy())

    # --- Plot snapshots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, step in zip(axes.flat, snapshot_steps):
        pos, vel = snapshots[step]

        # Draw obstacles
        for center, radius in sim.obstacles:
            circle = Circle(center, radius, fill=True, color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Draw goal
        ax.plot(sim.goal[0], sim.goal[1], 'r*', markersize=15, label='Goal')

        # Draw boids as arrows showing their heading
        speeds = np.linalg.norm(vel, axis=1)
        # Normalize velocities for arrow direction
        dirs = vel / (speeds[:, None] + 1e-6)

        ax.quiver(pos[:, 0], pos[:, 1], dirs[:, 0], dirs[:, 1],
                   speeds, cmap='viridis', scale=25, width=0.005,
                   headwidth=4, headlength=5, alpha=0.8)

        ax.set_xlim([0, sim.world_size])
        ax.set_ylim([0, sim.world_size])
        ax.set_aspect('equal')
        ax.set_title(f"Step {step} (t = {step * dt:.1f}s)")
        ax.grid(True, alpha=0.2)

    axes[0, 0].legend(fontsize=8, loc='upper left')
    plt.suptitle("Reynolds Flocking: Swarm Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig("12_swarm_snapshots.png", dpi=120)
    plt.show()

    # --- Animate the swarm ---
    print("\nCreating animation...")
    np.random.seed(42)
    sim2 = SwarmSimulation(n_boids=50, world_size=100.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw static elements
    for center, radius in sim2.obstacles:
        circle = Circle(center, radius, fill=True, color='gray', alpha=0.5)
        ax.add_patch(circle)
    ax.plot(sim2.goal[0], sim2.goal[1], 'r*', markersize=15)

    # Initialize quiver plot
    pos = np.array([b.position for b in sim2.boids])
    vel = np.array([b.velocity for b in sim2.boids])
    speeds = np.linalg.norm(vel, axis=1)
    dirs = vel / (speeds[:, None] + 1e-6)

    quiver = ax.quiver(pos[:, 0], pos[:, 1], dirs[:, 0], dirs[:, 1],
                        speeds, cmap='viridis', scale=25, width=0.005,
                        headwidth=4, headlength=5, alpha=0.8)

    ax.set_xlim([0, sim2.world_size])
    ax.set_ylim([0, sim2.world_size])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    title = ax.set_title("Reynolds Flocking (t = 0.0s)")

    def animate(frame):
        """Update function for animation."""
        for _ in range(3):  # Sub-steps per frame for smoother appearance
            sim2.step(dt)

        pos = np.array([b.position for b in sim2.boids])
        vel = np.array([b.velocity for b in sim2.boids])
        speeds = np.linalg.norm(vel, axis=1)
        dirs = vel / (speeds[:, None] + 1e-6)

        quiver.set_offsets(pos)
        quiver.set_UVC(dirs[:, 0], dirs[:, 1], speeds)
        title.set_text(f"Reynolds Flocking (t = {frame * 3 * dt:.1f}s)")
        return quiver, title

    anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=False)
    plt.tight_layout()

    # Try to save as GIF, fall back to showing
    try:
        anim.save("12_swarm_animation.gif", writer='pillow', fps=20)
        print("Animation saved as 12_swarm_animation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

    plt.show()

    # --- Print statistics ---
    print("\n--- Swarm Statistics (final state) ---")
    final_pos = np.array([b.position for b in sim.boids])
    final_vel = np.array([b.velocity for b in sim.boids])

    centroid = np.mean(final_pos, axis=0)
    spread = np.std(final_pos, axis=0)
    avg_speed = np.mean(np.linalg.norm(final_vel, axis=1))

    # Compute alignment metric (how aligned are the velocities?)
    avg_heading = np.mean(final_vel, axis=0)
    alignment_metric = np.linalg.norm(avg_heading) / avg_speed

    print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
    print(f"  Spread (std): ({spread[0]:.1f}, {spread[1]:.1f})")
    print(f"  Average speed: {avg_speed:.2f}")
    print(f"  Alignment metric: {alignment_metric:.3f} (1.0 = perfect alignment)")
    print(f"  Distance to goal: {np.linalg.norm(centroid - sim.goal):.1f}")


if __name__ == "__main__":
    demo_swarm()
