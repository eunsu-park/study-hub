# Robotics Overview and Classification

[Next: Rigid Body Transformations →](02_Rigid_Body_Transformations.md)

## Learning Objectives

1. Trace the historical evolution of robotics from early industrial manipulators to modern autonomous systems
2. Classify robots by mechanical structure (manipulators, mobile, aerial, underwater) and understand the distinguishing characteristics of each category
3. Define Degrees of Freedom (DOF) and explain how they determine a robot's workspace and capability
4. Identify the core components of any robotic system — sensors, actuators, and controllers — and describe their roles
5. Survey the major application domains of robotics including manufacturing, medicine, space exploration, agriculture, and logistics
6. Develop intuition for why robotics is an inherently multidisciplinary field combining mechanics, electronics, and computer science

---

## Why This Matters

Robotics sits at the intersection of mechanical engineering, electrical engineering, and computer science. It is arguably the field where abstract mathematics — linear algebra, differential equations, optimization — finds its most tangible expression: a physical machine that moves, senses, and decides. Before we dive into the mathematics of transformations, kinematics, and dynamics, we need a map of the territory. This lesson provides that map.

Understanding how robots are classified, what components they share, and where they are deployed gives you the vocabulary and mental framework to appreciate *why* the subsequent mathematical machinery exists. Every rotation matrix we derive in Lesson 2 exists because a real robot joint needs to know its orientation. Every Jacobian we compute in Lesson 5 exists because a real controller needs to translate desired end-effector motion into motor commands.

> **Analogy**: A robot is like a human body — actuators are muscles, sensors are eyes and ears, the controller is the brain. Just as understanding human anatomy precedes the study of biomechanics, understanding robot anatomy precedes the study of robot mathematics.

---

## A Brief History of Robotics

### The Word "Robot"

The word *robot* entered our vocabulary in 1920 through Karel Capek's play *R.U.R. (Rossum's Universal Robots)*. The Czech word *robota* means "forced labor." Isaac Asimov later coined the term *robotics* in his 1941 short story "Liar!" and formulated the famous Three Laws of Robotics.

### Industrial Beginnings: Unimate (1961)

The modern era of robotics began with **Unimate**, the first industrial robot, installed at a General Motors plant in Ewing, New Jersey, in 1961. Developed by George Devol and Joseph Engelberger, Unimate was a hydraulic manipulator that performed die-casting operations — extracting hot metal parts from machines and stacking them.

Key milestones in the timeline:

| Year | Milestone | Significance |
|------|-----------|--------------|
| 1954 | George Devol patents programmable manipulator | Foundation of industrial robotics |
| 1961 | Unimate installed at GM | First industrial robot deployment |
| 1969 | Stanford Arm (Victor Scheinman) | First electrically-driven, computer-controlled arm |
| 1973 | KUKA Famulus | First industrial robot with 6 electromechanically driven axes |
| 1978 | PUMA (Programmable Universal Machine for Assembly) | Standard for research and light assembly |
| 1997 | Sojourner on Mars | First mobile robot on another planet |
| 2000 | Honda ASIMO | Advanced humanoid bipedal locomotion |
| 2004 | DARPA Grand Challenge | Launched autonomous vehicle research |
| 2011 | da Vinci Surgical System widespread | Robotic surgery becomes mainstream |
| 2013 | Boston Dynamics Atlas | Dynamic humanoid locomotion |
| 2017 | Sophia (Hanson Robotics) | Social robotics enters public awareness |
| 2020s | Warehouse automation, delivery drones | Robotics becomes pervasive in logistics |

### The Three Waves of Robotics

1. **First Wave (1960s-1980s)**: Industrial manipulators for repetitive tasks in structured environments (welding, painting, pick-and-place)
2. **Second Wave (1990s-2010s)**: Service robots operating in semi-structured environments (surgical robots, vacuum cleaners, warehouse AGVs)
3. **Third Wave (2010s-present)**: Autonomous robots in unstructured environments (self-driving cars, delivery drones, agricultural robots), enabled by advances in AI, sensing, and computation

---

## Robot Classification

Robots can be classified along multiple axes. The most fundamental classification is by **mechanical structure**, which determines how the robot moves through space.

### 1. Manipulators (Robot Arms)

A manipulator is a kinematic chain of rigid links connected by joints, typically fixed to a base. They are the workhorses of industrial robotics.

#### Serial Manipulators

In a serial manipulator, links form an open chain from base to end-effector. Each joint adds one degree of freedom.

```
Base ──[Joint 1]── Link 1 ──[Joint 2]── Link 2 ── ... ──[Joint n]── End-Effector
```

Common configurations:

| Configuration | DOF | Joint Types | Workspace Shape | Example |
|--------------|-----|-------------|-----------------|---------|
| Cartesian (PPP) | 3 | 3 Prismatic | Rectangular box | Gantry robots |
| Cylindrical (RPP) | 3 | 1 Rev + 2 Prism | Cylinder | Seiko RT3000 |
| Spherical (RRP) | 3 | 2 Rev + 1 Prism | Sphere portion | Stanford Arm |
| SCARA (RRP) | 4 | 2 Rev + 1 Prism + 1 Rev | Cylinder | Epson SCARA |
| Articulated (RRR...) | 6 | All Revolute | Complex | KUKA, ABB, FANUC |

> **Note**: "R" = Revolute joint (rotation), "P" = Prismatic joint (translation). The letters describe the joint types from base to tip.

```python
# Simple representation of a serial manipulator
# Each joint is defined by its type and axis of motion
class Joint:
    """Represents a single robot joint.

    Why separate joint types? Because revolute joints contribute
    angular velocity while prismatic joints contribute linear velocity
    to the end-effector — this distinction is fundamental to the Jacobian (Lesson 5).
    """
    def __init__(self, joint_type, axis, limits=None):
        self.joint_type = joint_type  # 'revolute' or 'prismatic'
        self.axis = axis              # rotation/translation axis
        self.limits = limits          # (min, max) joint limits
        self.value = 0.0              # current joint value (angle or displacement)

class SerialManipulator:
    """A serial kinematic chain of joints and links.

    Why a list? Because serial manipulators are inherently sequential:
    the pose of joint i depends on the poses of joints 1 through i-1.
    """
    def __init__(self, name, joints):
        self.name = name
        self.joints = joints  # ordered list from base to tip

    @property
    def dof(self):
        """Degrees of freedom equals the number of joints.

        This is a fundamental property: a robot needs at least 6 DOF
        to reach any position AND orientation in 3D space.
        """
        return len(self.joints)

# Example: 6-DOF articulated manipulator (like a KUKA robot)
kuka_joints = [
    Joint('revolute', 'z', limits=(-170, 170)),   # Base rotation
    Joint('revolute', 'y', limits=(-190, 45)),     # Shoulder
    Joint('revolute', 'y', limits=(-120, 156)),    # Elbow
    Joint('revolute', 'z', limits=(-185, 185)),    # Wrist 1
    Joint('revolute', 'y', limits=(-120, 120)),    # Wrist 2
    Joint('revolute', 'z', limits=(-350, 350)),    # Wrist 3 (flange)
]
kuka = SerialManipulator("KUKA KR6", kuka_joints)
print(f"{kuka.name}: {kuka.dof} DOF")  # KUKA KR6: 6 DOF
```

#### Parallel Manipulators

In a parallel manipulator, multiple kinematic chains connect the base to the end-effector platform simultaneously. The most famous example is the **Stewart-Gough platform** (used in flight simulators).

**Advantages over serial manipulators**:
- Higher rigidity and load capacity (forces distributed across multiple chains)
- Better precision (errors don't accumulate along a single chain)
- Higher speed (actuators are near the base, reducing moving mass)

**Disadvantages**:
- Smaller workspace relative to footprint
- More complex kinematics (forward kinematics is harder than inverse!)
- Singular configurations within the workspace

```python
# Contrast: Serial vs Parallel manipulators
# In serial robots, forward kinematics is easy, inverse is hard.
# In parallel robots, it's the opposite!

comparison = {
    "Serial": {
        "FK": "Easy (direct chain multiplication)",
        "IK": "Hard (may have multiple solutions)",
        "Workspace": "Large relative to size",
        "Rigidity": "Lower (cantilevered structure)",
        "Speed": "Lower (moving mass includes actuators)",
        "Example": "KUKA KR6, ABB IRB 6700"
    },
    "Parallel": {
        "FK": "Hard (coupled nonlinear equations)",
        "IK": "Easy (each leg solved independently)",
        "Workspace": "Small relative to size",
        "Rigidity": "Higher (closed-loop structure)",
        "Speed": "Higher (actuators near base)",
        "Example": "Stewart Platform, Delta robot"
    }
}
```

### 2. Mobile Robots

Mobile robots are not fixed to a base — they navigate through their environment.

#### Wheeled Mobile Robots

The most common category, with several drive configurations:

| Type | Description | Holonomic? | Example |
|------|-------------|------------|---------|
| Differential drive | 2 driven wheels + casters | Non-holonomic | Roomba, TurtleBot |
| Ackermann steering | Car-like front steering | Non-holonomic | Autonomous cars |
| Omnidirectional | Mecanum or Swedish wheels | Holonomic | KUKA youBot base |
| Skid-steer | Tank-like differential | Non-holonomic | Clearpath Husky |

> **Key concept — Holonomic vs Non-holonomic**: A holonomic robot can move instantaneously in any direction (like a ball on a flat surface). A non-holonomic robot has constraints on instantaneous motion — a car cannot slide sideways. This distinction profoundly affects motion planning (Lesson 7).

#### Legged Robots

Legged locomotion enables traversal of rough, unstructured terrain where wheels fail.

| Legs | Stability | Gait Complexity | Example |
|------|-----------|-----------------|---------|
| 1 (hopper) | Dynamically stable | Hopping | Boston Dynamics hopper |
| 2 (biped) | Dynamically stable | Walking, running | Atlas, ASIMO, Digit |
| 4 (quadruped) | Statically/dynamically | Walk, trot, gallop | Spot, ANYmal |
| 6 (hexapod) | Statically stable (tripod gait) | Many options | RHex |

**Static stability**: The projection of the center of gravity falls within the support polygon (the convex hull of the feet in contact with the ground).

**Dynamic stability**: The robot may be momentarily unstable (like a human walking — we are essentially "falling forward" with each step) but recovers through active control.

#### Tracked Robots

Tracked (caterpillar) robots use continuous tracks for locomotion. They excel on loose soil, sand, and rubble where wheels would sink. Common in military EOD (Explosive Ordnance Disposal) and search-and-rescue scenarios.

### 3. Aerial Robots (UAVs)

Unmanned Aerial Vehicles (UAVs) have exploded in popularity and capability.

| Type | Mechanism | Characteristics |
|------|-----------|-----------------|
| Multi-rotor (quadcopter) | 4+ fixed-pitch rotors | VTOL, hovering, agile; limited endurance |
| Fixed-wing | Aerodynamic lift | Long range, high speed; no hovering |
| Hybrid (VTOL fixed-wing) | Tilt-rotors or tail-sitter | Combines benefits; complex control |
| Flapping-wing | Biomimetic wing motion | Research stage; extreme agility at small scale |

```python
# UAV classification by flight principle
# Why does this matter? The flight principle determines the
# mathematical model: multi-rotors use rigid body dynamics with
# thrust vectors, while fixed-wing uses aerodynamic lift equations.

class UAV:
    def __init__(self, name, uav_type, max_payload_kg, endurance_min):
        self.name = name
        self.uav_type = uav_type
        self.max_payload_kg = max_payload_kg
        self.endurance_min = endurance_min

# Typical specifications
drones = [
    UAV("DJI Matrice 300", "multi-rotor", 2.7, 55),
    UAV("senseFly eBee X", "fixed-wing", 0.5, 90),
    UAV("Wingtra One", "hybrid-VTOL", 0.8, 59),
]

for d in drones:
    print(f"{d.name} ({d.uav_type}): "
          f"payload={d.max_payload_kg}kg, endurance={d.endurance_min}min")
```

### 4. Underwater Robots

| Type | Description | Example |
|------|-------------|---------|
| ROV (Remotely Operated Vehicle) | Tethered, human-controlled | Jason (WHOI) |
| AUV (Autonomous Underwater Vehicle) | Untethered, autonomous | Bluefin-21 |
| Underwater glider | Buoyancy-driven, very long endurance | Slocum Glider |

Underwater robots face unique challenges: GPS is unavailable (acoustic positioning instead), communication bandwidth is extremely limited (acoustic modems), and water pressure imposes severe structural constraints.

---

## Degrees of Freedom (DOF)

### Definition

The **Degrees of Freedom** of a mechanical system is the minimum number of independent parameters needed to completely specify the configuration of the system.

For a rigid body moving freely in 3D space:
- 3 DOF for position $(x, y, z)$
- 3 DOF for orientation (roll, pitch, yaw)
- **Total: 6 DOF**

This is why most industrial manipulators have 6 joints — to achieve arbitrary position *and* orientation of the end-effector.

### Grubler's Formula

For a mechanism with $n$ links (including the fixed base), $j$ joints, and $f_i$ DOF for joint $i$:

$$\text{DOF} = m(n - 1 - j) + \sum_{i=1}^{j} f_i$$

where $m = 3$ for planar mechanisms and $m = 6$ for spatial mechanisms.

```python
def grubler_formula(n_links, joints, spatial=True):
    """Compute DOF using Grubler's formula.

    Why Grubler's formula? It tells us whether a mechanism is:
    - DOF > 0: a mechanism (can move)
    - DOF = 0: a structure (rigid, no motion)
    - DOF < 0: over-constrained (statically indeterminate)

    Parameters:
        n_links: number of links including the fixed base
        joints: list of (joint_name, dof_of_joint)
        spatial: True for 3D (m=6), False for planar (m=3)
    """
    m = 6 if spatial else 3
    j = len(joints)
    sum_fi = sum(dof for _, dof in joints)

    dof = m * (n_links - 1 - j) + sum_fi
    return dof

# Example 1: Planar 4-bar linkage
# 4 links (including ground), 4 revolute joints (each 1 DOF)
dof_4bar = grubler_formula(4, [("R", 1)]*4, spatial=False)
print(f"4-bar linkage DOF: {dof_4bar}")  # 1 DOF — single input drives the mechanism

# Example 2: 6-DOF serial manipulator
# 7 links (base + 6), 6 revolute joints
dof_6r = grubler_formula(7, [("R", 1)]*6, spatial=True)
print(f"6R manipulator DOF: {dof_6r}")  # 6 DOF

# Example 3: Stewart platform
# 14 links (base + platform + 6 legs x 2),
# 6 universal joints (2 DOF each) at base,
# 6 spherical joints (3 DOF each) at platform,
# 6 prismatic joints (1 DOF each) in legs
stewart_joints = [("U", 2)]*6 + [("S", 3)]*6 + [("P", 1)]*6
dof_stewart = grubler_formula(14, stewart_joints, spatial=True)
print(f"Stewart platform DOF: {dof_stewart}")  # 6 DOF
```

### Workspace

The **workspace** of a robot is the set of all positions (and orientations) that the end-effector can reach.

- **Reachable workspace**: All points the end-effector can reach with at least one orientation
- **Dexterous workspace**: All points the end-effector can reach with *any* orientation

The dexterous workspace is always a subset of the reachable workspace. For a 6-DOF robot, the dexterous workspace is where we have full position and orientation control.

---

## Core Components of a Robot

Every robot, regardless of type, consists of three fundamental subsystems.

### 1. Sensors (Perception)

Sensors provide the robot with information about itself (**proprioceptive**) and its environment (**exteroceptive**).

| Category | Sensor | Measures | Typical Use |
|----------|--------|----------|-------------|
| Proprioceptive | Encoder | Joint angle/position | Joint feedback |
| Proprioceptive | IMU | Acceleration, angular rate | Orientation estimation |
| Proprioceptive | Force/Torque sensor | Contact forces | Force control |
| Proprioceptive | Current sensor | Motor current | Torque estimation |
| Exteroceptive | LiDAR | Distance (3D point cloud) | Mapping, obstacle detection |
| Exteroceptive | Camera (RGB) | Visual images | Object recognition |
| Exteroceptive | Depth camera (RGB-D) | Color + depth | 3D perception |
| Exteroceptive | Ultrasonic | Distance | Proximity detection |
| Exteroceptive | GPS | Global position | Outdoor navigation |

### 2. Actuators (Action)

Actuators convert energy into physical motion.

| Type | Energy Source | Characteristics | Use Case |
|------|-------------|-----------------|----------|
| DC Motor | Electrical | Continuous rotation, precise speed control | Wheel drives |
| Servo Motor | Electrical | Position-controlled, high torque-to-weight | Robot joints |
| Stepper Motor | Electrical | Discrete steps, open-loop positioning | 3D printers, CNC |
| Pneumatic | Compressed air | Fast, compliant, binary | Grippers, soft robots |
| Hydraulic | Pressurized fluid | Very high force, precise | Heavy industrial, excavators |
| Shape Memory Alloy | Thermal | Compact, silent, slow | Micro-actuators |

```python
# Motor selection: a simplified sizing calculation
# Why does this matter? Choosing the wrong motor means the robot
# either can't move its payload or wastes energy on oversized actuators.

import math

def size_motor(payload_kg, arm_length_m, max_angular_vel_rads, safety_factor=2.0):
    """Estimate required motor torque and power for a single joint.

    This is a gross simplification — real sizing considers the full
    dynamics (inertia, Coriolis, friction), but it gives a starting point.
    """
    g = 9.81  # m/s^2

    # Worst case: arm fully extended horizontally
    torque_gravity = payload_kg * g * arm_length_m  # N*m

    # Apply safety factor for dynamic loads, friction, etc.
    torque_required = torque_gravity * safety_factor

    # Power = torque * angular velocity
    power_required = torque_required * max_angular_vel_rads  # Watts

    return torque_required, power_required

torque, power = size_motor(
    payload_kg=5.0,        # 5 kg payload
    arm_length_m=0.5,      # 0.5 m arm
    max_angular_vel_rads=math.pi  # 180 deg/s
)
print(f"Required torque: {torque:.1f} N*m")
print(f"Required power: {power:.1f} W")
```

### 3. Controllers (Decision)

The controller is the "brain" that processes sensor data and commands actuators. Modern robot controllers operate in a hierarchy:

```
┌─────────────────────────────────┐
│  Task Planning (AI / high-level) │  "Pick up the red cup"
├─────────────────────────────────┤
│  Motion Planning (path/trajectory)│  "Move through these waypoints"
├─────────────────────────────────┤
│  Control (servo loop)            │  "Apply these motor torques"
├─────────────────────────────────┤
│  Hardware Interface (drivers)    │  "Send PWM signals to motors"
└─────────────────────────────────┘
```

Each level operates at a different frequency:
- Task planning: seconds to minutes
- Motion planning: 10-100 Hz
- Servo control: 1-10 kHz
- Hardware interface: microseconds

---

## Applications of Robotics

### Manufacturing

The original and still dominant application. Modern factories use robots for:
- **Welding**: Consistent quality, hazardous environment protection
- **Painting**: Uniform coating, VOC exposure reduction
- **Assembly**: High precision, repetitive operations
- **Quality inspection**: Machine vision-based defect detection
- **Material handling**: Palletizing, depalletizing, bin picking

### Medical Robotics

| System | Application | Key Feature |
|--------|-------------|-------------|
| da Vinci | Minimally invasive surgery | Tremor filtering, 3D visualization |
| CyberKnife | Radiation surgery | Sub-millimeter targeting accuracy |
| ROBODOC | Orthopedic surgery | Precise bone cutting |
| Exoskeletons | Rehabilitation | Assistive motion for stroke patients |

### Space Robotics

- **Canadarm2**: 17m robotic arm on the International Space Station
- **Mars rovers**: Sojourner, Spirit, Opportunity, Curiosity, Perseverance
- **Orbital servicing**: Satellite repair and refueling (Northrop Grumman MEV)

### Agriculture

- **Harvesting**: Fruit picking with computer vision (Abundant Robotics, Agrobot)
- **Precision spraying**: Targeted herbicide application, reducing chemical use by 90%
- **Autonomous tractors**: GPS-guided plowing and planting (John Deere)
- **Monitoring**: Drone-based crop health assessment (multispectral imaging)

### Logistics and Warehousing

Amazon's warehouses deploy over 750,000 robots (as of 2024):
- **AGVs/AMRs**: Autonomous mobile robots for shelf transport (Kiva/Amazon Robotics)
- **Sorting**: High-speed parcel sorting systems
- **Last-mile delivery**: Sidewalk delivery robots (Starship), delivery drones (Wing, Amazon Prime Air)

---

## The Multidisciplinary Nature of Robotics

Robotics is not a single discipline — it is the integration of many:

```
        ┌─────────────┐
        │  ROBOTICS    │
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ▼          ▼          ▼
Mechanics  Electronics  Computer
   │          │        Science
   │          │          │
   ├─ Statics ├─ Circuits├─ Algorithms
   ├─ Dynamics├─ Sensors ├─ AI/ML
   ├─ Materials├─ Motors ├─ Computer Vision
   └─ Design  └─ PCBs   ├─ Control Theory
                         └─ Software Eng.
```

This course focuses on the **mathematical foundations** that bridge these disciplines: transformations, kinematics, dynamics, planning, and control. These are the tools that allow a computer scientist to reason about physical motion and a mechanical engineer to program intelligent behavior.

---

## The Road Ahead: Course Overview

| Lesson | Topic | Key Question |
|--------|-------|-------------|
| L01 | Overview and Classification | What are robots and how are they organized? |
| L02 | Rigid Body Transformations | How do we describe position and orientation in 3D? |
| L03 | Forward Kinematics | Given joint angles, where is the end-effector? |
| L04 | Inverse Kinematics | Given a desired pose, what are the joint angles? |
| L05 | Velocity Kinematics (Jacobian) | How do joint velocities map to end-effector velocity? |
| L06 | Robot Dynamics | What forces/torques are needed to produce desired motion? |
| L07 | Motion Planning | How do we find a collision-free path? |
| L08 | Trajectory Planning | How do we time-parameterize the path for smooth execution? |

Each lesson builds on the previous ones. The progression mirrors how a real robot controller works: first we need to describe geometry (L02-L03), then solve for joint angles (L04), understand velocity relationships (L05), account for physics (L06), and finally plan and execute motion (L07-L08).

---

## Summary

- Robotics has evolved from simple industrial manipulators (Unimate, 1961) to autonomous systems operating in unstructured environments
- Robots are classified by structure: **manipulators** (serial, parallel), **mobile** (wheeled, legged, tracked), **aerial** (multi-rotor, fixed-wing), and **underwater** (ROV, AUV)
- **Degrees of Freedom** determines a robot's capability; 6 DOF is the minimum for arbitrary pose in 3D
- Every robot has three core subsystems: **sensors** (perception), **actuators** (action), and **controllers** (decision)
- Robotics is inherently multidisciplinary, and this course focuses on the mathematical foundations that unify the field

---

## Exercises

### Exercise 1: Robot Classification

For each of the following robots, classify by type (manipulator/mobile/aerial/underwater), sub-type, and estimate the DOF:
1. A car-painting robot in a factory
2. A self-driving delivery truck
3. A hexacopter for aerial photography
4. An underwater vehicle inspecting oil pipelines
5. A robotic hand for prosthetics

### Exercise 2: Grubler's Formula

A planar mechanism has 5 links (including ground), 5 revolute joints, and 1 prismatic joint. Use Grubler's formula to compute its DOF. Is this a mechanism or a structure?

### Exercise 3: Motor Sizing

A 2-link planar robot arm has link lengths $l_1 = 0.4$ m and $l_2 = 0.3$ m. The arm must support a 2 kg payload at the end-effector. Ignoring link masses:
1. Calculate the maximum gravitational torque at joint 1 (when both links are horizontal)
2. Calculate the maximum gravitational torque at joint 2 (when link 2 is horizontal)
3. If the maximum joint velocity is $\pi/2$ rad/s, estimate the motor power requirements for each joint (with a safety factor of 2)

### Exercise 4: Workspace Estimation

A 2-DOF planar robot has link lengths $l_1 = 1.0$ m and $l_2 = 0.5$ m, with both joints having full 360-degree rotation.
1. Sketch the reachable workspace (hint: it is an annular region)
2. What are the inner and outer radii?
3. If joint 1 is limited to $[-90^\circ, 90^\circ]$ and joint 2 to $[-180^\circ, 180^\circ]$, how does the workspace change?

### Exercise 5: Research

Choose one application domain (manufacturing, medical, space, agriculture, or logistics). Research a specific robot used in that domain and write a one-page summary covering:
- Robot name and manufacturer
- Classification (type, sub-type, DOF)
- Key sensors and actuators
- Primary application and why a robot is better than a human for this task
- Current limitations

---

[Next: Rigid Body Transformations →](02_Rigid_Body_Transformations.md)
