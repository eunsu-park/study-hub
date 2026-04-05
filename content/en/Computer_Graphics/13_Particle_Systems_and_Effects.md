# 13. Particle Systems and Effects

[← Previous: Animation and Skeletal Systems](12_Animation_and_Skeletal_Systems.md) | [Next: GPU Computing →](14_GPU_Computing.md)

---

## Learning Objectives

1. Understand particle system architecture: emitter, particle pool, update, and render stages
2. Define particle properties and their evolution over a lifetime
3. Implement common forces: gravity, wind, drag, and point attractors
4. Compare numerical integration methods (Euler, Verlet) for particle simulation
5. Explain billboard rendering for camera-facing particle quads
6. Understand GPU particle systems using transform feedback and compute shaders
7. Describe volumetric effects (smoke, fire, fog) and the basics of ray marching
8. Build a complete particle system simulation in Python

---

## Why This Matters

Particle systems are the workhorses of visual effects in games and film. Every explosion, smoke trail, rain shower, fire, sparkle, magic spell, and fountain you see in interactive media is built from particles. William Reeves introduced particle systems at Lucasfilm in 1983 to create the "Genesis effect" in *Star Trek II: The Wrath of Khan*, and they have been a core graphics technique ever since.

What makes particle systems powerful is their simplicity: each particle is just a point with a few properties (position, velocity, age), yet thousands of them together create complex, organic phenomena. Understanding particle systems teaches you the fundamentals of simulation, numerical integration, and GPU-driven rendering that apply broadly across physics, games, and scientific visualization.

---

## 1. Particle System Architecture

### 1.1 Overview

A particle system consists of:

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Emitter  │────▶│ Particle  │────▶│  Update   │────▶│  Render  │
│ (spawn)  │     │   Pool    │     │ (physics) │     │ (draw)   │
└─────────┘     └──────────┘     └──────────┘     └──────────┘
     │                                  │
     │           ┌──────────┐           │
     └──────────▶│  Forces  │◀──────────┘
                 └──────────┘
```

**Emitter**: Spawns new particles with initial properties (position, velocity, color, size). Emission can be:
- **Continuous**: N particles per second
- **Burst**: All particles at once (explosion)
- **Spatial**: From a point, line, circle, sphere, mesh surface, etc.

**Particle pool**: A fixed-size array of particles. Dead particles are recycled to avoid allocation overhead. This is a classic **object pool** pattern.

**Update**: Each frame, advance the simulation by $\Delta t$: apply forces, integrate velocity/position, age particles, kill expired ones.

**Render**: Draw each living particle as a textured quad, point sprite, or mesh.

### 1.2 Particle Lifecycle

```
Born (emitter spawns)
  │
  ▼
Alive (position, velocity, age update each frame)
  │  age += dt
  │  apply forces
  │  integrate motion
  ▼
Dead (age >= lifetime)  →  returned to pool
```

---

## 2. Particle Properties

Each particle carries a set of properties that evolve over its lifetime:

| Property | Type | Description |
|----------|------|-------------|
| `position` | vec3 | Current world position |
| `velocity` | vec3 | Current velocity |
| `acceleration` | vec3 | Accumulated force / mass |
| `age` | float | Time since birth (seconds) |
| `lifetime` | float | Maximum age before death |
| `color` | vec4 | RGBA color (alpha for fade-out) |
| `size` | float | Particle size (radius or quad width) |
| `rotation` | float | Rotation angle (for textured quads) |
| `mass` | float | Used for force calculations |

### 2.1 Property Curves

Properties often change over the particle's lifetime using **curves**:

- **Alpha**: Fade in at birth, full opacity during life, fade out before death
- **Size**: Start small, grow, then shrink
- **Color**: Transition from white → yellow → orange → red → black (fire)

These curves are parameterized by the normalized age $t = \text{age} / \text{lifetime} \in [0, 1]$.

```python
def fade_in_out(t, fade_in=0.1, fade_out=0.3):
    """
    Alpha curve: quick fade-in, sustained, gradual fade-out.
    Why smooth fading: prevents harsh pop-in/pop-out artifacts.
    """
    if t < fade_in:
        return t / fade_in
    elif t > 1.0 - fade_out:
        return (1.0 - t) / fade_out
    else:
        return 1.0
```

---

## 3. Forces

### 3.1 Gravity

The simplest and most common force:

$$\mathbf{F}_{\text{gravity}} = m \mathbf{g}$$

where $\mathbf{g} = (0, -9.81, 0)$ m/s$^2$ on Earth. For stylized effects, $\mathbf{g}$ can be any vector and magnitude.

### 3.2 Wind

A constant or spatially varying force:

$$\mathbf{F}_{\text{wind}} = \mathbf{w}(t)$$

Turbulent wind can be simulated using Perlin or simplex noise:

$$\mathbf{F}_{\text{turbulence}}(\mathbf{p}, t) = A \cdot \text{noise}(\mathbf{p} \cdot s + t \cdot f)$$

where $A$ is amplitude, $s$ is spatial frequency, and $f$ is temporal frequency.

### 3.3 Drag (Air Resistance)

Drag opposes motion and is proportional to velocity:

$$\mathbf{F}_{\text{drag}} = -c_d \cdot \mathbf{v}$$

for linear drag, or:

$$\mathbf{F}_{\text{drag}} = -c_d \cdot \|\mathbf{v}\| \cdot \mathbf{v}$$

for quadratic drag (more physically accurate for high speeds).

**Drag coefficient** $c_d$ determines how quickly particles slow down:
- Low drag: Particles travel far (sparks, bullets)
- High drag: Particles slow quickly (smoke, dust)

### 3.4 Point Attractors and Repellers

A point attractor at position $\mathbf{a}$ pulls particles toward it:

$$\mathbf{F}_{\text{attract}} = k \cdot \frac{\mathbf{a} - \mathbf{p}}{\|\mathbf{a} - \mathbf{p}\|^2 + \epsilon}$$

The $\epsilon$ prevents singularity at the attractor position. Negative $k$ creates a repeller.

### 3.5 Vortex Force

A vortex force creates swirling motion around an axis $\hat{\mathbf{a}}$:

$$\mathbf{F}_{\text{vortex}} = k \cdot (\hat{\mathbf{a}} \times (\mathbf{p} - \mathbf{c}))$$

where $\mathbf{c}$ is the vortex center. This is useful for tornados, whirlpools, and spiral effects.

---

## 4. Numerical Integration

### 4.1 Euler Method

The simplest integrator. Given position $\mathbf{x}$, velocity $\mathbf{v}$, and acceleration $\mathbf{a}$:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$
$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t) \cdot \Delta t$$

**Symplectic Euler** (also called semi-implicit Euler) updates velocity first, then uses the *new* velocity for position:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$
$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t + \Delta t) \cdot \Delta t$$

This small change dramatically improves energy conservation and is the standard for game physics.

### 4.2 Verlet Integration

**Stormer-Verlet** uses positions at the current and previous time steps, without explicit velocity:

$$\mathbf{x}(t + \Delta t) = 2\mathbf{x}(t) - \mathbf{x}(t - \Delta t) + \mathbf{a}(t) \cdot \Delta t^2$$

Velocity is implicit: $\mathbf{v}(t) \approx \frac{\mathbf{x}(t) - \mathbf{x}(t - \Delta t)}{\Delta t}$.

**Advantages**: Second-order accurate, symplectic (energy-conserving), excellent for constraint satisfaction (cloth, ropes).

**Disadvantages**: Adding damping or velocity-dependent forces is less natural.

### 4.3 Comparison

| Method | Order | Energy Conservation | Complexity | Best For |
|--------|-------|--------------------:|------------|----------|
| Euler | 1st | Poor (energy drift) | Very low | Quick prototyping |
| Symplectic Euler | 1st | Good | Very low | Games, particles |
| Verlet | 2nd | Excellent | Low | Constraints, cloth |
| RK4 | 4th | Good | Moderate | High accuracy needs |

For particle systems, **symplectic Euler** is almost always sufficient.

---

## 5. Billboard Rendering

### 5.1 The Problem

Particles are points, but we want to display them as textured shapes (circles, smoke puffs, sparks). We render each particle as a **quad** (two triangles) that always faces the camera.

### 5.2 Camera-Facing Quads

For each particle at position $\mathbf{p}$ with size $s$, construct a quad from the camera's right $\mathbf{r}$ and up $\mathbf{u}$ vectors:

$$\mathbf{v}_0 = \mathbf{p} + s(-\mathbf{r} + \mathbf{u}), \quad \mathbf{v}_1 = \mathbf{p} + s(\mathbf{r} + \mathbf{u})$$
$$\mathbf{v}_2 = \mathbf{p} + s(\mathbf{r} - \mathbf{u}), \quad \mathbf{v}_3 = \mathbf{p} + s(-\mathbf{r} - \mathbf{u})$$

These four vertices form a screen-aligned quad that always faces the camera, regardless of view direction.

### 5.3 Rendering Considerations

- **Blending**: Particles typically use **additive blending** (fire, sparks, magic) or **alpha blending** (smoke, dust). Additive: $C_{\text{out}} = C_{\text{src}} + C_{\text{dst}}$. Alpha: $C_{\text{out}} = \alpha C_{\text{src}} + (1-\alpha)C_{\text{dst}}$.
- **Depth testing**: Write to z-buffer disabled (particles overlap); z-test enabled (particles occluded by solid geometry).
- **Sorting**: For alpha blending, sort particles back-to-front. For additive blending, sorting is unnecessary (additive is commutative).
- **Soft particles**: Fade particles that intersect solid geometry by comparing particle depth with the depth buffer. Prevents hard intersection lines.

### 5.4 Point Sprites

Modern GPUs support **point sprites**: render a single vertex as an automatically-generated screen-aligned quad. In OpenGL: `glEnable(GL_PROGRAM_POINT_SIZE)` and set `gl_PointSize` in the vertex shader. More efficient than generating quad geometry on the CPU.

---

## 6. Python Implementation: Particle System

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable, Optional

@dataclass
class Particle:
    """A single particle with physics properties."""
    position: np.ndarray      # (3,) world position
    velocity: np.ndarray      # (3,) velocity
    color: np.ndarray         # (4,) RGBA
    size: float               # Radius or quad half-size
    age: float = 0.0          # Time since birth
    lifetime: float = 2.0     # Maximum age
    mass: float = 1.0
    alive: bool = True

    @property
    def normalized_age(self):
        """Age as fraction of lifetime [0, 1]."""
        return min(self.age / self.lifetime, 1.0)


class Emitter:
    """Spawns particles with randomized initial properties."""

    def __init__(self, position, rate=50.0, lifetime_range=(1.0, 3.0),
                 speed_range=(1.0, 3.0), size_range=(0.05, 0.15),
                 color_start=np.array([1, 0.8, 0.2, 1]),
                 color_end=np.array([1, 0, 0, 0]),
                 spread_angle=30.0, direction=np.array([0, 1, 0])):
        self.position = np.array(position, dtype=float)
        self.rate = rate               # Particles per second
        self.lifetime_range = lifetime_range
        self.speed_range = speed_range
        self.size_range = size_range
        self.color_start = color_start
        self.color_end = color_end
        self.spread_angle = spread_angle  # Cone half-angle in degrees
        self.direction = direction / np.linalg.norm(direction)
        self._accumulator = 0.0        # Fractional particle accumulator

    def emit(self, dt) -> List[Particle]:
        """Generate new particles for this time step."""
        # Why accumulator: if rate*dt < 1, we still need to emit
        # particles over multiple frames
        self._accumulator += self.rate * dt
        count = int(self._accumulator)
        self._accumulator -= count

        particles = []
        for _ in range(count):
            # Random direction within cone
            vel_dir = self._random_cone_direction()
            speed = np.random.uniform(*self.speed_range)

            lifetime = np.random.uniform(*self.lifetime_range)
            size = np.random.uniform(*self.size_range)

            p = Particle(
                position=self.position.copy(),
                velocity=vel_dir * speed,
                color=self.color_start.copy(),
                size=size,
                lifetime=lifetime,
            )
            particles.append(p)

        return particles

    def _random_cone_direction(self):
        """Generate a random direction within a cone around self.direction."""
        # Build local frame
        d = self.direction
        if abs(d[0]) < 0.9:
            tangent = np.cross(np.array([1, 0, 0]), d)
        else:
            tangent = np.cross(np.array([0, 1, 0]), d)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(d, tangent)

        # Random angle within cone
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(
            np.cos(np.radians(self.spread_angle)), 1.0
        )
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        # Direction in local frame, then transform to world
        local = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
        world = local[0] * tangent + local[1] * bitangent + local[2] * d
        return world / np.linalg.norm(world)


class ForceField:
    """Base class for forces applied to particles."""

    def apply(self, particle: Particle, dt: float) -> np.ndarray:
        """Return force vector (3,)."""
        raise NotImplementedError


class Gravity(ForceField):
    def __init__(self, g=np.array([0, -9.81, 0])):
        self.g = np.array(g, dtype=float)

    def apply(self, particle, dt):
        return particle.mass * self.g


class Wind(ForceField):
    def __init__(self, direction=np.array([1, 0, 0]), strength=2.0):
        self.force = np.array(direction, dtype=float) * strength

    def apply(self, particle, dt):
        return self.force


class Drag(ForceField):
    def __init__(self, coefficient=0.5):
        self.cd = coefficient

    def apply(self, particle, dt):
        # Why linear drag: simple and effective for visual particle systems
        return -self.cd * particle.velocity


class PointAttractor(ForceField):
    def __init__(self, position, strength=10.0, epsilon=0.1):
        self.pos = np.array(position, dtype=float)
        self.strength = strength
        self.epsilon = epsilon

    def apply(self, particle, dt):
        diff = self.pos - particle.position
        dist_sq = np.dot(diff, diff) + self.epsilon
        return self.strength * diff / dist_sq


class ParticleSystem:
    """
    Complete particle system: manages emitters, forces, and particle pool.
    Uses symplectic Euler integration.
    """

    def __init__(self, max_particles=5000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.emitters: List[Emitter] = []
        self.forces: List[ForceField] = []
        self.time = 0.0

    def add_emitter(self, emitter: Emitter):
        self.emitters.append(emitter)

    def add_force(self, force: ForceField):
        self.forces.append(force)

    def update(self, dt: float):
        """Advance the simulation by dt seconds."""
        self.time += dt

        # Emit new particles
        for emitter in self.emitters:
            new_particles = emitter.emit(dt)
            # Respect pool limit
            available = self.max_particles - len(self.particles)
            self.particles.extend(new_particles[:available])

        # Update existing particles
        alive_particles = []
        for p in self.particles:
            # Age the particle
            p.age += dt

            # Kill expired particles
            if p.age >= p.lifetime:
                p.alive = False
                continue

            # Accumulate forces
            total_force = np.zeros(3)
            for force in self.forces:
                total_force += force.apply(p, dt)

            # Symplectic Euler integration
            # Why symplectic: better energy conservation than standard Euler
            acceleration = total_force / p.mass
            p.velocity += acceleration * dt       # Update velocity first
            p.position += p.velocity * dt         # Then use new velocity

            # Update visual properties based on normalized age
            t = p.normalized_age

            # Color interpolation (start -> end over lifetime)
            # Why we store both: allows different start/end per emitter
            emitter = self.emitters[0] if self.emitters else None
            if emitter:
                p.color = (1 - t) * emitter.color_start + t * emitter.color_end

            # Size: grow then shrink
            p.size *= (1.0 - 0.3 * dt)  # Gradual shrink

            alive_particles.append(p)

        self.particles = alive_particles

    def get_positions(self) -> np.ndarray:
        """Return all particle positions as (N, 3) array."""
        if not self.particles:
            return np.zeros((0, 3))
        return np.array([p.position for p in self.particles])

    def get_colors(self) -> np.ndarray:
        """Return all particle colors as (N, 4) array."""
        if not self.particles:
            return np.zeros((0, 4))
        return np.array([p.color for p in self.particles])

    def stats(self) -> dict:
        return {
            "alive": len(self.particles),
            "max": self.max_particles,
            "time": self.time,
        }


# --- Demo: Fire-like particle system ---

system = ParticleSystem(max_particles=2000)

# Emitter: fire shooting upward
fire_emitter = Emitter(
    position=[0, 0, 0],
    rate=200,
    lifetime_range=(0.5, 2.0),
    speed_range=(1.0, 4.0),
    size_range=(0.05, 0.2),
    color_start=np.array([1.0, 0.9, 0.3, 1.0]),   # Bright yellow
    color_end=np.array([0.8, 0.1, 0.0, 0.0]),      # Dark red, transparent
    spread_angle=20.0,
    direction=np.array([0, 1, 0]),
)
system.add_emitter(fire_emitter)

# Forces
system.add_force(Gravity(np.array([0, -2.0, 0])))  # Weak gravity (fire rises)
system.add_force(Wind(np.array([0.5, 0, 0]), strength=0.5))
system.add_force(Drag(coefficient=0.8))

# Simulate
dt = 1.0 / 60.0  # 60 FPS
print("Simulating fire particle system...")
for frame in range(180):  # 3 seconds
    system.update(dt)
    if frame % 30 == 0:
        s = system.stats()
        positions = system.get_positions()
        if len(positions) > 0:
            avg_y = np.mean(positions[:, 1])
            max_y = np.max(positions[:, 1])
            print(f"  Frame {frame:3d}: {s['alive']:4d} particles, "
                  f"avg_y={avg_y:.2f}, max_y={max_y:.2f}")

# Visualization (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    positions = system.get_positions()
    colors = system.get_colors()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
               c=colors[:, :3], s=colors[:, 3] * 20, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(f'Fire Particle System ({len(positions)} particles)')
    plt.tight_layout()
    plt.savefig('particle_fire.png', dpi=150)
    plt.close()
    print("Saved particle_fire.png")
except ImportError:
    print("Install matplotlib for visualization")
```

---

## 7. GPU Particle Systems

### 7.1 Motivation

CPU particle systems are limited to tens of thousands of particles. GPU particle systems can handle **millions** by running the simulation entirely on the GPU.

### 7.2 Transform Feedback Approach (OpenGL)

1. Store particle data in vertex buffer objects (VBOs)
2. In the vertex shader, apply forces and integration
3. Use **transform feedback** to write updated positions/velocities back to a second VBO
4. Swap the two VBOs (ping-pong) each frame
5. Render the output VBO as point sprites

```glsl
// Vertex shader for particle update (transform feedback)
#version 330

in vec3 in_position;
in vec3 in_velocity;
in float in_age;
in float in_lifetime;

out vec3 out_position;
out vec3 out_velocity;
out float out_age;
out float out_lifetime;

uniform float dt;
uniform vec3 gravity;

void main() {
    out_age = in_age + dt;
    out_lifetime = in_lifetime;

    if (out_age >= out_lifetime) {
        // Respawn logic would go here (or use a compute shader)
        out_position = vec3(0.0);
        out_velocity = vec3(0.0);
        out_age = 0.0;
    } else {
        vec3 accel = gravity;
        out_velocity = in_velocity + accel * dt;
        out_position = in_position + out_velocity * dt;
    }
}
```

### 7.3 Compute Shader Approach

Modern GPUs support **compute shaders** that read/write arbitrary buffers:

1. Store particle data in **Shader Storage Buffer Objects (SSBOs)**
2. Dispatch a compute shader with one thread per particle
3. Each thread updates its particle independently
4. No ping-pong needed -- atomic operations or double buffering within the SSBO

```glsl
// Compute shader for particle simulation
#version 430

layout(local_size_x = 256) in;

struct Particle {
    vec4 position;    // xyz = pos, w = size
    vec4 velocity;    // xyz = vel, w = age
    vec4 color;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform float dt;
uniform vec3 gravity;
uniform int numParticles;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numParticles) return;

    Particle p = particles[idx];
    float age = p.velocity.w + dt;

    if (age >= p.position.w) {  // position.w stores lifetime
        // Dead particle -- reset or skip
        age = 0.0;
        // ... respawn logic ...
    }

    vec3 vel = p.velocity.xyz + gravity * dt;
    vec3 pos = p.position.xyz + vel * dt;

    p.position.xyz = pos;
    p.velocity.xyz = vel;
    p.velocity.w = age;

    particles[idx] = p;
}
```

### 7.4 Performance Comparison

| Approach | Particle Count | Bottleneck |
|----------|---------------|------------|
| CPU (Python) | ~1,000-5,000 | Python overhead |
| CPU (C++) | ~50,000-100,000 | Cache misses, single-threaded |
| GPU (transform feedback) | ~500,000 | VBO ping-pong overhead |
| GPU (compute shader) | ~1,000,000-10,000,000 | Memory bandwidth |

---

## 8. Volumetric Effects

### 8.1 Smoke and Fog

Smoke, fog, and clouds are not well represented by individual particles (they fill volumes). Two approaches:

**Particle-based**: Use many overlapping, large, semi-transparent particles with soft textures. Works well for distant smoke plumes but looks "blobby" up close.

**Volume rendering (ray marching)**: Cast rays through a 3D volume (density field) and accumulate color and opacity along the ray:

$$C_{\text{out}} = \sum_{k=0}^{N} T_k \cdot \sigma(\mathbf{x}_k) \cdot c(\mathbf{x}_k) \cdot \Delta s$$

where $T_k = \exp\left(-\sum_{j=0}^{k-1} \sigma(\mathbf{x}_j) \Delta s\right)$ is the transmittance, $\sigma$ is the density (extinction coefficient), and $c$ is the color/emission.

### 8.2 Ray Marching

**Ray marching** steps along a ray in fixed increments, sampling a density field at each step:

```
function ray_march(ray_origin, ray_direction, max_steps, step_size):
    color = (0, 0, 0)
    transmittance = 1.0

    for i in range(max_steps):
        pos = ray_origin + i * step_size * ray_direction
        density = sample_density_field(pos)

        if density > 0:
            // Beer-Lambert extinction
            extinction = exp(-density * step_size)
            // Light contribution at this point
            light = compute_lighting(pos, density)
            // Accumulate
            color += transmittance * (1 - extinction) * light
            transmittance *= extinction

        if transmittance < 0.01:
            break  // Early termination: fully opaque

    return color
```

### 8.3 Fire Rendering

Fire combines particle-based and volumetric approaches:

1. **Temperature field**: Simulate combustion as a density + temperature field
2. **Color mapping**: Map temperature to emission color (black body radiation):
   - Cool: dark red/orange
   - Hot: yellow/white
   - Very hot: blue/white
3. **Noise**: Add procedural noise (Perlin, simplex) to the density field for turbulent appearance
4. **Upward motion**: Advect the density field upward (buoyancy)

### 8.4 Practical Fire Formula

A simple fire color from temperature $T$ (normalized 0-1):

$$R = \min(1, 1.5T), \quad G = \min(1, 1.5T^2), \quad B = T^4$$

This maps cool temperatures to red, medium to yellow, and hot to white.

---

## 9. Advanced Particle Techniques

### 9.1 Collision Detection

Particles can collide with scene geometry:
- **Plane collision**: If particle moves past a plane, reflect velocity component along the normal: $v_n' = -e \cdot v_n$ where $e$ is the coefficient of restitution
- **Sphere collision**: Similar, using the sphere's surface normal
- **Depth buffer collision**: Compare particle depth with the scene depth buffer (GPU-friendly)

### 9.2 Trail Rendering

Particles can spawn trails by recording past positions:
- **Line trails**: Connect current and previous positions (sparks, fireworks)
- **Ribbon trails**: Connect particles in emission order with a textured strip (sword slash, missile trail)

### 9.3 Sub-Emitters

When a particle dies, it can spawn a **sub-emitter** burst:
- Firework: main particle rises, explodes into sub-particles on death
- Spark cascade: impact sparks spawn smaller sparks
- Smoke after explosion: fire particles emit smoke sub-particles

### 9.4 Particle LOD

Distant particle systems can be simplified:
- Reduce emission rate based on distance to camera
- Increase particle size and decrease count
- Switch to a single billboard texture for very distant effects

---

## 10. Common Effects Recipes

| Effect | Particle Count | Lifetime | Forces | Blending | Notes |
|--------|---------------|----------|--------|----------|-------|
| Fire | 200-1000 | 0.5-2s | Weak gravity up, drag, turbulence | Additive | Color: yellow→red→black |
| Smoke | 100-500 | 2-5s | Buoyancy, drag, wind | Alpha | Large, slow, low-alpha |
| Sparks | 50-200 | 0.3-1s | Strong gravity, drag | Additive | Small, bright, fast |
| Rain | 500-2000 | 1-3s | Strong gravity | Alpha | Vertical streaks |
| Snow | 200-1000 | 3-8s | Weak gravity, wind | Alpha | Slow, drifting |
| Explosion | 500 burst | 0.5-2s | Radial velocity, gravity, drag | Additive | Burst emission, sub-emitters |
| Magic/Spell | 100-500 | 1-3s | Attractor, vortex | Additive | Swirling, colorful |
| Dust | 20-100 | 1-4s | Wind, drag | Alpha | Brown, large, slow |

---

## Summary

| Concept | Key Idea |
|---------|----------|
| Particle lifecycle | Emitter spawns → update (forces + integrate) → render → die when age > lifetime |
| Object pool | Fixed-size array; recycle dead particles to avoid allocation |
| Forces | Gravity, wind, drag ($-c_d \mathbf{v}$), attractors ($k/r^2$), vortex |
| Symplectic Euler | Update velocity first, then position; better energy conservation |
| Verlet integration | Second-order; uses current and previous position; good for constraints |
| Billboard | Camera-facing quad; constructed from camera right/up vectors |
| GPU particles | Transform feedback (VBO ping-pong) or compute shaders; millions of particles |
| Ray marching | Step along ray, accumulate density and color; for smoke, fog, fire |
| Beer-Lambert | Transmittance $T = e^{-\sigma \cdot s}$; models light absorption in volumes |

## Exercises

1. **Particle fountain**: Create a particle system that emits particles upward with initial velocity, applies gravity, and has particles bounce off a ground plane (y = 0) with 70% restitution.

2. **Vortex attractor**: Add a vortex force to the fire particle system. Experiment with different strengths and axis orientations. How does the visual effect change?

3. **Integration comparison**: Simulate a particle under gravity using (a) Euler, (b) symplectic Euler, and (c) Verlet integration. Compare the trajectories and energy conservation over 10 seconds. Plot the total energy $E = \frac{1}{2}mv^2 + mgh$ over time for each method.

4. **Firework**: Implement a firework particle system: one particle rises, then bursts into 100 sub-particles that spread radially, then each sub-particle fades and falls with gravity.

5. **Simple ray marcher**: Implement a 2D ray marcher that renders a circular fog volume. The density should fall off with distance from the center (Gaussian profile). Visualize the result as a 1D scanline.

6. **GPU simulation design**: Design (on paper) the data layout for a GPU compute shader particle system supporting 1 million particles. Specify the buffer structure, work group size, and how you would handle emission and death.

## Further Reading

- Reeves, W.T. "Particle Systems -- A Technique for Modeling a Class of Fuzzy Objects." *SIGGRAPH*, 1983. (The original particle systems paper)
- Latta, L. "Building a Million Particle System." *GDC*, 2004. (Practical GPU particle implementation)
- Stam, J. "Real-Time Fluid Dynamics for Games." *GDC*, 2003. (Fluid simulation for smoke/fire; Jos Stam's stable fluids)
- Bridson, R. *Fluid Simulation for Computer Graphics*, 2nd ed. CRC Press, 2015. (Comprehensive fluid dynamics for effects)
- McGuire, M. "The Graphics Codex." Online, 2024. (Excellent particle system and billboard rendering reference)
