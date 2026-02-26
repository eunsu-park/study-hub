# 12. 애니메이션과 골격 시스템

[← 이전: 11. 패스 트레이싱과 전역 조명](11_Path_Tracing_and_Global_Illumination.md) | [다음: 13. 파티클 시스템과 이펙트 →](13_Particle_Systems_and_Effects.md)

---

## 학습 목표

1. 키프레임 애니메이션(keyframe animation)과 포즈 간 보간(interpolation)의 역할을 이해한다
2. 회전을 위한 선형 보간(lerp)과 구면 선형 보간(slerp)을 구현한다
3. 베지어 곡선(Bezier curves)과 스플라인(splines)으로 부드러운 애니메이션 경로를 만든다
4. 골격 애니메이션(skeletal animation)을 설명한다: 뼈(bones), 관절(joints), 바인드 포즈(bind pose), 뼈 계층 구조
5. 순방향 기구학(forward kinematics, FK)으로 월드 공간 관절 위치를 계산한다
6. 역방향 기구학(inverse kinematics, IK) 알고리즘 CCD와 FABRIK을 이해한다
7. 버텍스 스키닝(vertex skinning) 방식을 설명한다: 선형 블렌드 스키닝(linear blend skinning, LBS)과 이중 쿼터니언 스키닝(dual quaternion skinning)
8. 얼굴 애니메이션과 변형을 위한 모프 타겟(morph targets, blend shapes)을 적용한다

---

## 왜 중요한가

정적인 3D 씬도 인상적이지만, **애니메이션**이 씬에 생명을 불어넣는다. 게임 세계를 걷는 캐릭터부터 시뮬레이션에서 부품을 조립하는 로봇 팔까지, 애니메이션은 기하학과 스토리텔링을 이어주는 다리다. 이 레슨의 기법들은 모든 애니메이션 영화, 비디오 게임, VR 경험, 로봇 시뮬레이터에서 사용된다.

특히 골격 애니메이션은 컴퓨터 그래픽스에서 가장 우아한 해결책 중 하나다. 수천 개의 버텍스(vertex)를 개별적으로 애니메이션하는 대신, 약 50~200개의 뼈로 구성된 골격을 애니메이션하면 메시(mesh)가 **스키닝(skinning)**을 통해 자동으로 따라온다. 이 제어(골격)와 디테일(메시)의 분리가 캐릭터 애니메이션을 실용적으로 만드는 핵심이다.

---

## 1. 키프레임 애니메이션(Keyframe Animation)

### 1.1 개념

**키프레임 애니메이션**에서 애니메이터는 **키프레임**이라는 특정 시간 순간에 오브젝트의 상태(위치, 회전, 스케일, 색상 등)를 지정한다. 시스템은 **보간(interpolation)**을 통해 중간 상태를 자동으로 계산한다.

```
Time:     0s         0.5s        1.0s        1.5s        2.0s
          |           |           |           |           |
Key:    [start]                 [peak]                 [end]
          ●───────────────────────●───────────────────────●
          pos=(0,0,0)           pos=(0,3,0)           pos=(5,0,0)
```

애니메이터는 키프레임만 설정하고, 보간이 그 사이의 모든 프레임을 채운다.

### 1.2 애니메이션 커브(Animation Curves)

각 애니메이션 속성에는 시간 $t$를 값 $v(t)$로 매핑하는 **애니메이션 커브**(f-curve라고도 함)가 있다. 이 커브의 형태가 동작의 느낌을 결정한다:

- **선형(Linear)**: 일정한 속도; 기계적인 느낌
- **ease-in**: 느리게 시작, 가속
- **ease-out**: 빠르게 시작, 감속
- **ease-in-out**: 부드러운 시작과 끝 (캐릭터 동작에 가장 자연스러움)

이것들은 일반적으로 커브 자체에 대한 3차 베지어 곡선 또는 에르미트 스플라인(Hermite spline)으로 제어된다.

---

## 2. 보간 방법(Interpolation Methods)

### 2.1 선형 보간(Linear Interpolation, Lerp)

두 값 $a$와 $b$ 사이의 가장 단순한 보간:

$$\text{lerp}(a, b, t) = (1 - t) \cdot a + t \cdot b, \quad t \in [0, 1]$$

벡터(위치)의 경우:

$$\mathbf{p}(t) = (1 - t)\mathbf{p}_0 + t\mathbf{p}_1$$

Lerp는 키프레임 사이에서 일정한 속도의 동작을 만든다. 빠르고 많은 속성(위치, 색상, 스케일)에 충분하지만, 행렬이나 쿼터니언으로 표현된 **회전에는 적합하지 않다**.

### 2.2 왜 Lerp가 회전에 실패하는가

두 회전 행렬 사이의 보간을 생각해보자. Lerp는 $(1-t)\mathbf{R}_0 + t\mathbf{R}_1$을 계산하지만, 결과는 일반적으로 **회전 행렬이 아니다** (직교성을 잃음). 보간된 행렬에 전단(shearing)이나 스케일링 아티팩트가 포함될 수 있다.

오일러 각도(Euler angles)에서도 Lerp는 **짐벌 락(gimbal lock)**과 비균일 회전 속도를 유발한다.

### 2.3 쿼터니언 표현(Quaternion Representation)

**쿼터니언** $q = w + xi + yj + zk$은 축 $\mathbf{u}$ 주변의 각도 $\theta$ 회전을 나타낸다:

$$q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(u_x i + u_y j + u_z k)$$

단위 쿼터니언($\|q\| = 1$)은 3D 회전의 **이중 커버(double cover)**를 형성한다: $q$와 $-q$는 같은 회전을 나타낸다. 짐벌 락을 피하고 효율적으로 합성된다.

### 2.4 구면 선형 보간(Spherical Linear Interpolation, Slerp)

**Slerp**는 4D 단위 구의 최단 호를 따라 보간하여 일정한 각속도를 만든다:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} q_0 + \frac{\sin(t\Omega)}{\sin\Omega} q_1$$

여기서 $\Omega = \arccos(q_0 \cdot q_1)$은 쿼터니언 사이의 각도이다.

**중요**: $q_0 \cdot q_1 < 0$이면, 보간이 짧은 경로(180도 미만)를 택하도록 하나를 부정하라.

**$\Omega$가 매우 작을 때** (거의 동일한 회전), slerp는 수치적 문제가 생긴다. 이 경우 정규화된 lerp(nlerp)로 대체한다:

$$\text{nlerp}(q_0, q_1, t) = \frac{(1-t)q_0 + tq_1}{\|(1-t)q_0 + tq_1\|}$$

Nlerp는 일정한 각속도가 없지만 빠르고 특이점(singularity)을 피한다.

### 2.5 구현

```python
import numpy as np

def lerp(a, b, t):
    """Linear interpolation between scalars, vectors, or arrays."""
    return (1.0 - t) * a + t * b


def quaternion_dot(q1, q2):
    """Dot product of two quaternions (as 4D vectors)."""
    return np.dot(q1, q2)


def quaternion_normalize(q):
    """Normalize a quaternion to unit length."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-10 else q


def slerp(q0, q1, t):
    """
    Spherical Linear Interpolation between two unit quaternions.
    Quaternion format: [w, x, y, z].
    """
    # Ensure shortest path: if dot product is negative, negate one
    dot = quaternion_dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    # If quaternions are very close, fall back to normalized lerp
    # Why: sin(omega) approaches 0, causing division-by-zero
    if dot > 0.9995:
        result = (1.0 - t) * q0 + t * q1
        return quaternion_normalize(result)

    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)

    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega

    return s0 * q0 + s1 * q1


def quaternion_from_axis_angle(axis, angle_deg):
    """Create a quaternion from an axis and angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    half = angle_rad / 2.0
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(half),
                     axis[0] * np.sin(half),
                     axis[1] * np.sin(half),
                     axis[2] * np.sin(half)])


def quaternion_to_matrix(q):
    """Convert a unit quaternion [w,x,y,z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


# Demo: interpolate between two rotations
q_start = quaternion_from_axis_angle(np.array([0, 1, 0]), 0)     # No rotation
q_end   = quaternion_from_axis_angle(np.array([0, 1, 0]), 120)   # 120 degrees around Y

for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    q = slerp(q_start, q_end, t)
    # Extract the effective rotation angle
    angle = 2 * np.degrees(np.arccos(np.clip(q[0], -1, 1)))
    print(f"  t={t:.2f}: angle = {angle:.1f} degrees, q = [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")
```

출력:
```
  t=0.00: angle = 0.0 degrees, q = [1.000, 0.000, 0.000, 0.000]
  t=0.25: angle = 30.0 degrees, q = [0.966, 0.000, 0.259, 0.000]
  t=0.50: angle = 60.0 degrees, q = [0.866, 0.000, 0.500, 0.000]
  t=0.75: angle = 90.0 degrees, q = [0.707, 0.000, 0.707, 0.000]
  t=1.00: angle = 120.0 degrees, q = [0.500, 0.000, 0.866, 0.000]
```

각도가 균일하게 증가함을 확인할 수 있다(단계당 30도). 이것이 slerp의 일정한 각속도 특성이다.

---

## 3. 베지어 곡선(Bezier Curves)과 스플라인(Splines)

### 3.1 3차 베지어 곡선(Cubic Bezier Curves)

**3차 베지어 곡선**은 네 개의 제어점 $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3$로 정의된다:

$$\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2 t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3, \quad t \in [0,1]$$

특성:
- 곡선은 $\mathbf{P}_0$에서 시작하고 $\mathbf{P}_3$에서 끝난다
- $\mathbf{P}_0$에서의 접선은 $\mathbf{P}_1$ 방향을 가리키고, $\mathbf{P}_3$에서의 접선은 $\mathbf{P}_2$에서부터의 방향이다
- 곡선은 제어점의 볼록 껍질(convex hull) 내에 있다
- $\mathbf{P}_1$과 $\mathbf{P}_2$는 곡선을 통과하지 않으면서 형태를 잡는 "핸들" 역할을 한다

### 3.2 애니메이션 경로(Animation Paths)

베지어 곡선은 3D 공간에서 부드러운 동작 경로를 정의한다. 오브젝트는 $t$가 0에서 1로 증가함에 따라 곡선을 따라간다:

```python
def bezier_cubic(P0, P1, P2, P3, t):
    """
    Evaluate a cubic Bezier curve at parameter t.
    The curve smoothly moves from P0 to P3, shaped by control points P1, P2.
    """
    u = 1.0 - t
    # Why De Casteljau form: numerically more stable than expanding the polynomial
    return (u**3 * P0 + 3 * u**2 * t * P1
            + 3 * u * t**2 * P2 + t**3 * P3)


def bezier_tangent(P0, P1, P2, P3, t):
    """Tangent (velocity) of the cubic Bezier at parameter t."""
    u = 1.0 - t
    return (3 * u**2 * (P1 - P0) + 6 * u * t * (P2 - P1)
            + 3 * t**2 * (P3 - P2))
```

### 3.3 카탈-롬 스플라인(Catmull-Rom Splines)

애니메이션 경로가 일련의 키프레임 위치를 통과해야 할 때, **카탈-롬 스플라인**은 제어점을 **보간**하기 때문에 편리하다(곡선이 모든 점을 통과):

점 $\mathbf{P}_{i-1}, \mathbf{P}_i, \mathbf{P}_{i+1}, \mathbf{P}_{i+2}$가 주어질 때, $\mathbf{P}_i$와 $\mathbf{P}_{i+1}$ 사이의 세그먼트는:

$$\mathbf{C}(t) = 0.5 \begin{bmatrix} 1 & t & t^2 & t^3 \end{bmatrix}
\begin{bmatrix} 0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1 \end{bmatrix}
\begin{bmatrix} \mathbf{P}_{i-1} \\ \mathbf{P}_i \\ \mathbf{P}_{i+1} \\ \mathbf{P}_{i+2} \end{bmatrix}$$

각 점에서의 접선은 이웃 점으로부터 자동으로 계산되어 $C^1$ 연속성을 보장한다.

---

## 4. 골격 애니메이션(Skeletal Animation)

### 4.1 골격 구조

**골격(skeleton)**은 **뼈(bones)**(또는 **관절(joints)**)의 계층 구조(트리)다. 각 뼈는 다음을 가진다:
- **부모** 뼈 (루트 제외)
- **로컬 변환(local transform)** (부모에 상대적인 회전 + 이동)
- **휴지/바인드 포즈(rest/bind pose)** — 캐릭터가 T-포즈 또는 A-포즈일 때의 기본 구성

```
                    [Root / Hips]
                    /            \
            [Spine]              [Left Leg]
           /       \                  \
     [Chest]    [Right Leg]     [Left Knee]
      /    \                          \
[Left Arm] [Right Arm]          [Left Foot]
    |           |
[Left Elbow] [Right Elbow]
    |           |
[Left Hand] [Right Hand]
```

### 4.2 바인드 포즈와 역 바인드 행렬(Bind Pose and Inverse Bind Matrix)

**바인드 포즈(bind pose)**는 메시가 모델링된 때의 골격 구성이다. 각 뼈 $j$는 뼈 로컬 공간에서 월드 공간으로 매핑하는 **바인드 포즈 월드 변환** $\mathbf{B}_j$를 가진다.

**역 바인드 행렬(inverse bind matrix)** $\mathbf{B}_j^{-1}$은 버텍스를 월드 공간에서 뼈의 로컬 공간(바인드 포즈 기준)으로 변환한다. 한 번만 미리 계산된다:

$$\mathbf{B}_j^{-1} = (\mathbf{B}_j)^{-1}$$

애니메이션 중 뼈의 현재 월드 변환은 $\mathbf{W}_j$다. 뼈 $j$의 **스키닝 행렬(skinning matrix)**은:

$$\mathbf{S}_j = \mathbf{W}_j \cdot \mathbf{B}_j^{-1}$$

이 행렬은 버텍스를 바인드 포즈 위치에서 애니메이션된 위치로 옮긴다.

### 4.3 애니메이션 포즈(Animated Pose)

**애니메이션 클립(animation clip)**은 각 뼈의 로컬 변환에 대한 키프레임을 저장한다(일반적으로 쿼터니언 회전 + 이동). 각 프레임에서:

1. 키프레임 간 보간 (회전에는 slerp, 위치에는 lerp)
2. 각 뼈의 로컬 변환 계산
3. 계층 구조를 통해 전파하여 월드 변환 획득: $\mathbf{W}_j = \mathbf{W}_{\text{parent}(j)} \cdot \mathbf{L}_j$

---

## 5. 순방향 기구학(Forward Kinematics, FK)

### 5.1 정의

**순방향 기구학**은 관절 각도(로컬 회전)가 주어졌을 때 각 뼈의 위치와 방향을 계산한다. 루트에서 시작하여 변환을 체인 아래로 곱해 나간다:

$$\mathbf{W}_0 = \mathbf{L}_0$$
$$\mathbf{W}_j = \mathbf{W}_{\text{parent}(j)} \cdot \mathbf{L}_j$$

최종 효과기(end-effector) 위치(예: 손)는:

$$\mathbf{p}_{\text{end}} = \mathbf{W}_n \cdot \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}$$

### 5.2 FK 구현

```python
import numpy as np

class Bone:
    """A bone in a skeletal hierarchy."""

    def __init__(self, name, length, parent=None):
        self.name = name
        self.length = length         # Bone length along local X axis
        self.parent = parent
        self.children = []

        # Local transform: rotation angle (degrees) around Z axis (2D simplification)
        self.local_angle = 0.0

        # Computed during FK
        self.world_transform = np.eye(3)  # 2D: 3x3 homogeneous
        self.world_position = np.zeros(2)
        self.world_end = np.zeros(2)

        if parent:
            parent.children.append(self)

    def local_matrix(self):
        """Build the local transform: rotate by angle, then translate by length."""
        theta = np.radians(self.local_angle)
        c, s = np.cos(theta), np.sin(theta)
        # Why rotation then translation: rotation happens at the joint,
        # then we offset along the rotated bone direction
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
        T = np.array([[1, 0, self.length],
                      [0, 1, 0],
                      [0, 0, 1]])
        return R @ T


def forward_kinematics(bone, parent_transform=None):
    """
    Recursively compute world transforms for the skeleton.
    Each bone's world transform = parent's world * local transform.
    """
    if parent_transform is None:
        parent_transform = np.eye(3)

    # The joint position is at the origin of the parent transform
    bone.world_transform = parent_transform @ bone.local_matrix()

    # Joint start position: where the parent ends
    bone.world_position = parent_transform[:2, 2]

    # Joint end position: transform the local endpoint
    bone.world_end = bone.world_transform[:2, 2]

    for child in bone.children:
        forward_kinematics(child, bone.world_transform)


# Build a 3-bone arm (2D)
root = Bone("Shoulder", length=0.0)       # Root at origin, no length
upper_arm = Bone("UpperArm", length=2.0, parent=root)
forearm = Bone("Forearm", length=1.5, parent=upper_arm)
hand = Bone("Hand", length=1.0, parent=forearm)

# Set joint angles
root.local_angle = 0.0
upper_arm.local_angle = 45.0    # Shoulder rotated 45 degrees
forearm.local_angle = -30.0     # Elbow bent -30 degrees
hand.local_angle = 10.0         # Wrist slightly rotated

# Compute FK
forward_kinematics(root)

# Print results
for bone in [root, upper_arm, forearm, hand]:
    print(f"  {bone.name:>10}: start=({bone.world_position[0]:.2f}, {bone.world_position[1]:.2f})"
          f"  end=({bone.world_end[0]:.2f}, {bone.world_end[1]:.2f})")
```

---

## 6. 역방향 기구학(Inverse Kinematics, IK)

### 6.1 문제

**역방향 기구학**은 FK의 역이다. 원하는 최종 효과기 위치(예: "손을 여기에 놓아라")가 주어질 때, 이를 달성하는 관절 각도를 찾는다.

FK: 관절 각도 $\rightarrow$ 최종 효과기 위치 (단순)
IK: 최종 효과기 위치 $\rightarrow$ 관절 각도 (비결정적, 다수의 해 존재)

IK가 어려운 이유:
- 시스템이 일반적으로 **비결정적(underdetermined)** (제약보다 자유도(DOF)가 많음)
- 매핑이 **비선형** (삼각 함수 포함)
- 관절 한계를 지켜야 함

### 6.2 해석적 IK (2-뼈 경우)

단순한 2-뼈 체인(예: 상완 + 전완)에서는 **코사인 법칙(law of cosines)**을 사용하여 해석적으로 풀 수 있다:

뼈 길이 $L_1, L_2$와 목표 거리 $d = \|\mathbf{target} - \mathbf{root}\|$가 주어질 때:

$$\cos\theta_2 = \frac{d^2 - L_1^2 - L_2^2}{2L_1 L_2}$$

$$\theta_1 = \text{atan2}(t_y, t_x) - \text{atan2}(L_2 \sin\theta_2, L_1 + L_2\cos\theta_2)$$

이는 정확한 해를 제공한다 (또는 두 해: 팔꿈치-위 및 팔꿈치-아래).

### 6.3 CCD (Cyclic Coordinate Descent, 순환 좌표 하강)

**CCD**는 임의 길이 체인을 위한 반복 알고리즘이다:

1. 체인의 **마지막** 관절에서 시작
2. 최종 효과기가 목표를 향하도록 회전
3. 루트 방향의 다음 관절로 이동하여 반복
4. 최종 효과기가 목표에 충분히 가까워질 때까지 모든 관절을 순환

```
function CCD(chain, target, max_iterations):
    for iter in range(max_iterations):
        for j from last_joint to first_joint:
            // Vector from joint j to end-effector
            to_end = end_effector_pos - joint_pos[j]
            // Vector from joint j to target
            to_target = target - joint_pos[j]
            // Rotate joint j to align to_end with to_target
            angle = angle_between(to_end, to_target)
            joint_angle[j] += angle
            // Recompute FK
            update_chain()
        if distance(end_effector, target) < threshold:
            break
```

**장점**: 단순하고 긴 체인을 처리하며 관절 한계 추가가 쉽다.
**단점**: 수렴이 느릴 수 있고 부자연스러운 포즈(과도한 감김)를 만들 수 있다.

### 6.4 FABRIK (Forward And Backward Reaching Inverse Kinematics)

**FABRIK** (Aristidou & Lasenby, 2011)은 각도 대신 관절 위치를 반복적으로 조정하여 동작한다:

**순방향 패스** (최종 효과기에서 루트로):
1. 최종 효과기를 목표로 이동
2. 뒤에서 두 번째 관절을 최종 효과기 방향으로 이동, 뼈 길이 유지
3. 루트 방향으로 계속

**역방향 패스** (루트에서 최종 효과기로):
1. 루트를 원래 위치로 이동
2. 다음 관절을 루트 방향으로 이동, 뼈 길이 유지
3. 최종 효과기 방향으로 계속

수렴할 때까지 패스를 반복한다.

```python
def fabrik(joint_positions, bone_lengths, target, tolerance=0.01, max_iter=50):
    """
    FABRIK inverse kinematics solver.

    joint_positions: list of 2D positions [(x,y), ...]
    bone_lengths: list of distances between consecutive joints
    target: desired end-effector position (x,y)
    """
    n = len(joint_positions)
    positions = [np.array(p, dtype=float) for p in joint_positions]
    target = np.array(target, dtype=float)

    # Check if target is reachable
    total_length = sum(bone_lengths)
    root_to_target = np.linalg.norm(target - positions[0])
    if root_to_target > total_length:
        # Target unreachable: stretch toward it
        direction = (target - positions[0]) / root_to_target
        for i in range(1, n):
            positions[i] = positions[i-1] + direction * bone_lengths[i-1]
        return positions

    root_pos = positions[0].copy()

    for iteration in range(max_iter):
        # Check convergence
        error = np.linalg.norm(positions[-1] - target)
        if error < tolerance:
            break

        # --- Forward pass: move end-effector to target ---
        positions[-1] = target.copy()
        for i in range(n - 2, -1, -1):
            # Move joint i toward joint i+1, maintaining bone length
            direction = positions[i] - positions[i+1]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction /= dist
            positions[i] = positions[i+1] + direction * bone_lengths[i]

        # --- Backward pass: move root back to original position ---
        positions[0] = root_pos.copy()
        for i in range(1, n):
            # Move joint i toward joint i-1, maintaining bone length
            direction = positions[i] - positions[i-1]
            dist = np.linalg.norm(direction)
            if dist > 1e-10:
                direction /= dist
            positions[i] = positions[i-1] + direction * bone_lengths[i-1]

    return positions


# Demo: 4-joint chain reaching for a target
joints = [(0, 0), (2, 0), (3.5, 0), (4.5, 0)]
lengths = [2.0, 1.5, 1.0]
target = (3.0, 3.0)

result = fabrik(joints, lengths, target, tolerance=0.001)
print("FABRIK result:")
for i, pos in enumerate(result):
    print(f"  Joint {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
end_error = np.linalg.norm(result[-1] - np.array(target))
print(f"  End-effector error: {end_error:.6f}")
```

### 6.5 CCD vs. FABRIK

| 특성 | CCD | FABRIK |
|------|-----|--------|
| 동작 방식 | 각도 (회전 관절) | 위치 (각도는 후에 유도) |
| 수렴 속도 | 보통 | 빠름 (보통 3~5회 반복) |
| 자연스러운 포즈 | 감김 발생 가능 | 일반적으로 더 자연스러움 |
| 관절 제약 | 관절별 추가 용이 | 가능하지만 투영(projection) 필요 |
| 구현 | 단순 | 매우 단순 |
| 사용 사례 | 게임, 로봇공학 | 게임, 절차적 애니메이션 |

---

## 7. 버텍스 스키닝(Vertex Skinning)

### 7.1 문제

골격이 (FK, IK, 또는 키프레임 애니메이션으로) 포즈를 취하면, 메시가 뼈를 따라가도록 **변형(deform)**해야 한다. 각 버텍스는 여러 뼈에 의해 영향을 받을 수 있다(예: 팔꿈치 근처의 버텍스는 상완과 전완 뼈 모두에 영향받음).

### 7.2 선형 블렌드 스키닝(Linear Blend Skinning, LBS)

**LBS**(또는 "스무스 스키닝(smooth skinning)" 또는 "골격 부분 공간 변형(skeletal subspace deformation)")은 표준 기법이다:

$$\mathbf{v}' = \sum_{j=1}^{n} w_j \cdot \mathbf{S}_j \cdot \mathbf{v}$$

여기서:
- $\mathbf{v}$는 바인드 포즈에서의 버텍스 위치
- $\mathbf{S}_j = \mathbf{W}_j \cdot \mathbf{B}_j^{-1}$는 뼈 $j$의 스키닝 행렬
- $w_j$는 이 버텍스에 대한 뼈 $j$의 가중치 ($\sum_j w_j = 1$)
- $\mathbf{v}'$는 변형된 버텍스 위치

일반적으로 각 버텍스는 최대 4개의 뼈에 영향을 받는다 (GPU에 친화적인 한계).

**"캔디 래퍼(candy wrapper)" 아티팩트**: LBS는 변환 행렬을 선형으로 블렌딩한다. 관절이 크게 회전할 때(예: 전완이 180도 비틀릴 때), 보간된 행렬들이 캔디 래퍼가 비틀리는 것처럼 보이는 **체적 붕괴(volume collapse)**를 만든다. 이것이 LBS의 주요 한계다.

### 7.3 이중 쿼터니언 스키닝(Dual Quaternion Skinning)

**이중 쿼터니언 스키닝** (Kavan et al., 2007)은 행렬 블렌딩을 이중 쿼터니언 블렌딩으로 대체하여 체적을 보존한다:

이중 쿼터니언 $\hat{q} = q_r + \epsilon q_d$는 강체 변환(rigid transformation, 회전 $q_r$ + $q_d$에 인코딩된 이동)을 나타낸다. 이중 쿼터니언을 블렌딩하면 체적 붕괴를 피할 수 있다:

$$\hat{q}_{\text{blend}} = \frac{\sum_j w_j \hat{q}_j}{\|\sum_j w_j \hat{q}_j\|}$$

LBS보다 비용이 더 들지만 캔디 래퍼 아티팩트를 제거한다. 대부분의 현대 게임 엔진이 옵션으로 제공한다.

### 7.4 비교

| 특성 | LBS | 이중 쿼터니언 |
|------|-----|--------------|
| 체적 보존 | 아님 (캔디 래퍼) | 예 |
| 속도 | 매우 빠름 | 약간 느림 |
| GPU 지원 | 모든 엔진에서 기본 | 널리 지원됨 |
| 구현 | 단순한 행렬 블렌드 | 이중 쿼터니언 수학 |
| 스케일링 지원 | 예 | 추가 처리 필요 |

---

## 8. 모프 타겟(Morph Targets, 블렌드 셰이프)

### 8.1 개념

**모프 타겟(morph targets)**(또는 **블렌드 셰이프(blend shapes)**)은 메시의 완전히 변형된 버전을 저장한다. 최종 메시는 기본 메시와 하나 이상의 타겟의 가중 블렌드다:

$$\mathbf{v}_{\text{final}} = \mathbf{v}_{\text{base}} + \sum_{k=1}^{K} \alpha_k (\mathbf{v}_{\text{target}_k} - \mathbf{v}_{\text{base}})$$

여기서 $\alpha_k \in [0, 1]$은 타겟 $k$의 블렌드 가중치다.

### 8.2 얼굴 애니메이션

모프 타겟은 **얼굴 애니메이션**의 표준이다:
- 각 타겟은 얼굴 표정 또는 음소(phoneme)를 나타낸다: "smile", "frown", "blink", "mouth_open", "jaw_left" 등
- FACS(Facial Action Coding System, 얼굴 행동 코딩 시스템)은 개별 근육 움직임에 해당하는 표준화된 **행동 단위(action units)**를 정의한다
- 애니메이터는 블렌드 가중치를 키프레임으로 설정하여 발화와 표정을 만든다

일반적인 게임 캐릭터 얼굴은 30~60개의 블렌드 셰이프를 가진다. 영화 캐릭터는 200개 이상일 수 있다.

### 8.3 구현

```python
def apply_blend_shapes(base_vertices, targets, weights):
    """
    Compute blended vertex positions from base mesh and morph targets.

    base_vertices: (N, 3) array of base mesh positions
    targets: list of (N, 3) arrays, each a morph target
    weights: list of floats, blend weight for each target
    """
    result = base_vertices.copy()
    for target, weight in zip(targets, weights):
        if abs(weight) > 1e-6:
            # Why store deltas: saves memory and makes blending a simple add
            delta = target - base_vertices
            result += weight * delta
    return result


# Demo: simple face mesh (triangle of 3 vertices)
base = np.array([[0.0, 0.0, 0.0],    # Left corner of mouth
                  [1.0, 0.0, 0.0],    # Right corner of mouth
                  [0.5, 0.5, 0.0]])   # Upper lip

smile_target = np.array([[-.1, 0.2, 0.0],  # Left corner up
                          [1.1, 0.2, 0.0],  # Right corner up
                          [0.5, 0.6, 0.0]]) # Upper lip slightly up

frown_target = np.array([[0.1, -0.2, 0.0], # Left corner down
                          [0.9, -0.2, 0.0], # Right corner down
                          [0.5, 0.3, 0.0]]) # Upper lip slightly down

# Half smile + slight frown = smirk
result = apply_blend_shapes(base, [smile_target, frown_target], [0.5, 0.2])
print("Blended vertices (smirk):")
for i, v in enumerate(result):
    print(f"  v{i}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})")
```

### 8.4 모프 타겟 vs. 골격 애니메이션

| 측면 | 모프 타겟 | 골격 애니메이션 |
|------|----------|----------------|
| 사용 사례 | 얼굴, 부드러운 변형 | 신체, 강체 사지 |
| 저장 | 타겟당 전체 메시 | 골격 + 가중치 |
| 표현력 | 임의의 변형 가능 | 골격 토폴로지에 제한 |
| 성능 | 버텍스 덧셈 | 행렬 곱셈 |
| 조합 가능성 | 가산 블렌딩 | FK/IK 체인 |
| 일반적 조합 | 얼굴 (블렌드 셰이프) + 신체 (골격) |

---

## 9. 키프레임 보간 파이프라인(Keyframe Interpolation Pipeline)

각 프레임의 완전한 애니메이션 파이프라인:

```
1. Determine current time t within animation clip
2. For each bone:
   a. Find surrounding keyframes (before and after t)
   b. Compute interpolation factor: alpha = (t - t_prev) / (t_next - t_prev)
   c. Interpolate rotation: slerp(q_prev, q_next, alpha)
   d. Interpolate position: lerp(p_prev, p_next, alpha)
   e. Build local transform matrix from interpolated rotation + position
3. Forward Kinematics: propagate local transforms to world transforms
4. Compute skinning matrices: S_j = W_j * B_j_inverse
5. Vertex skinning: deform each vertex using weighted skinning matrices
6. Apply blend shapes (if any)
7. Upload deformed vertices to GPU for rendering
```

4~5단계는 일반적으로 **GPU**에서 버텍스 셰이더를 통해 수행되며, 뼈 행렬은 유니폼 배열로 업로드된다.

---

## 10. 애니메이션 블렌딩(Animation Blending)

### 10.1 크로스페이드(Cross-Fade)

두 애니메이션(예: 걷기에서 달리기) 사이를 출력을 보간하여 부드럽게 전환한다:

$$\mathbf{L}_j^{\text{blend}} = \text{slerp}(\mathbf{L}_j^{\text{walk}}, \mathbf{L}_j^{\text{run}}, \alpha)$$

여기서 $\alpha$는 전환 기간 동안 0에서 1로 증가한다.

### 10.2 레이어드 애니메이션(Layered Animation)

애니메이션을 가산적으로 결합한다:
- **기본 레이어**: 전신 이동 (걷기, 달리기, 점프)
- **상체 레이어**: 무기 조준, 손 흔들기
- **얼굴 레이어**: 표정을 위한 블렌드 셰이프

각 레이어에는 영향을 미치는 뼈를 제한하는 **마스크(mask)**를 가질 수 있다.

### 10.3 애니메이션 상태 기계(Animation State Machines)

게임은 상태 기계로 애니메이션 전환을 관리한다:

```
[Idle] --move--> [Walk] --speed_up--> [Run]
  |                 |                    |
  v                 v                    v
[Jump]           [Jump]              [Jump]
  |                 |                    |
  v                 v                    v
[Land]           [Land]              [Land]
```

전환은 블렌드 지속 시간과 조건(속도 임계값, 입력 이벤트)을 지정한다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| 키프레임 애니메이션 | 핵심 시간에 포즈 정의; 그 사이를 보간 |
| Lerp | $\text{lerp}(a, b, t) = (1-t)a + tb$ — 선형, 회전에는 부적합 |
| Slerp | 쿼터니언 초구(hypersphere)에서의 일정 각속도 보간 |
| 베지어 곡선 | 제어점으로 정의된 부드러운 매개변수 곡선; 애니메이션 경로에 사용 |
| 골격 애니메이션 | 뼈의 계층 구조; 뼈를 애니메이션하면 메시가 따라옴 |
| 순방향 기구학 | 관절 각도 $\to$ 최종 효과기 위치; 체인 아래로 변환 곱하기 |
| 역방향 기구학 | 최종 효과기 위치 $\to$ 관절 각도; CCD, FABRIK, 해석적 |
| LBS | 스키닝 행렬의 가중 합; 빠르지만 캔디 래퍼 아티팩트 |
| 이중 쿼터니언 | 체적 보존 스키닝; 적당한 비용으로 캔디 래퍼 제거 |
| 모프 타겟 | 저장된 메시 변형 간 블렌딩; 얼굴 애니메이션의 표준 |

## 연습문제

1. **Slerp 구현**: slerp를 구현하고 연속적으로 보간된 쿼터니언 사이의 각도를 측정하여 일정한 각속도를 검증한다.

2. **2-뼈 해석적 IK**: 코사인 법칙을 사용하여 해석적 2-뼈 IK 솔버를 구현한다. "팔꿈치-위(elbow-up)"와 "팔꿈치-아래(elbow-down)" 두 해를 시각화한다.

3. **3D FABRIK**: FABRIK 구현을 3D로 확장한다. 5-뼈 체인을 만들고 여러 목표 위치에 대해 풀어본다. 반복에 따른 수렴을 시각화한다.

4. **FK 체인 애니메이션**: 4-뼈 팔을 만든다. 다른 주파수의 사인 함수를 사용하여 관절 각도를 애니메이션한다. 결과로 나온 최종 효과기 경로를 시각화한다.

5. **블렌드 셰이프 얼굴**: 단순화된 얼굴 메시(10~20개 버텍스)를 만든다. 4개의 모프 타겟(smile, frown, surprise, blink)을 정의한다. 시간에 따라 블렌드 가중치를 애니메이션하여 표정 시퀀스를 만든다.

6. **LBS 아티팩트**: 두 개의 뼈를 가진 원통형 메시(튜브)에 LBS를 구현한다. 하나의 뼈를 0, 45, 90, 135, 180도로 회전시킨다. 캔디 래퍼 붕괴를 관찰하고 문서화한다. 이중 쿼터니언 스키닝을 구현하고 비교한다.

## 더 읽을거리

- Parent, R. *Computer Animation: Algorithms and Techniques*, 3rd ed. Morgan Kaufmann, 2012. (포괄적인 애니메이션 레퍼런스)
- Aristidou, A. and Lasenby, J. "FABRIK: A Fast, Iterative Solver for the Inverse Kinematics Problem." *Graphical Models*, 2011. (FABRIK 알고리즘)
- Kavan, L. et al. "Skinning with Dual Quaternions." *I3D*, 2007. (체적 보존 스키닝)
- Shoemake, K. "Animating Rotation with Quaternion Curves." *SIGGRAPH*, 1985. (그래픽스에 slerp 도입)
- Lewis, J.P. et al. "Practice and Theory of Blendshape Facial Models." *Eurographics STAR*, 2014. (블렌드 셰이프 기법 개관)
