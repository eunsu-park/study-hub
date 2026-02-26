# 역운동학(Inverse Kinematics)

[← 이전: 순운동학](03_Forward_Kinematics.md) | [다음: 속도 운동학 →](05_Velocity_Kinematics.md)

## 학습 목표

1. 역운동학(IK) 문제를 형식화하고, 이것이 순운동학보다 근본적으로 어려운 이유를 설명할 수 있다
2. 기하학적 방법과 대수적 방법을 사용하여 단순한 로봇(2-링크 평면, 3-자유도)의 IK를 해석적으로 풀 수 있다
3. 수치적 IK 방법을 적용할 수 있다: 뉴턴-랩슨(Newton-Raphson), 야코비안 유사역행렬(Jacobian pseudo-inverse), 감쇠 최소 제곱법(Levenberg-Marquardt)
4. 다중 해, 작업 공간 경계, 도달 불가 목표 지점을 파악하고 처리할 수 있다
5. 특이점(singularity)과 IK 해에 미치는 영향을 설명하고, 경계 특이점과 내부 특이점을 구별할 수 있다
6. 6 자유도 이상의 로봇을 위한 여유 자유도 해결(redundancy resolution) 전략을 소개할 수 있다

---

## 왜 중요한가

순운동학이 "관절 각도가 주어졌을 때 말단 작동체는 어디에 있는가?"를 묻는다면, 역운동학은 그 *반대*를 묻는다: "원하는 말단 작동체 포즈가 주어졌을 때, 어떤 관절 각도가 이를 구현하는가?" 이것은 모든 실제 로봇 작업이 궁극적으로 귀결되는 질문이다. 외과의사가 "메스를 왼쪽으로 2mm 이동시켜"라고 명령하거나 픽-앤-플레이스(pick-and-place) 시스템이 선반 위의 상자에 도달해야 할 때, 컨트롤러는 실시간으로 IK를 풀어야 한다.

IK는 깊은 수학적 이유로 FK보다 어렵다. FK는 관절 공간에서 작업 공간으로의 매끄러운 함수이지만, IK는 그 함수를 역산하려 한다. 그런데 역산 결과가 존재하지 않을 수도 있고 (목표가 작업 공간 밖), 여러 가지가 있을 수도 있으며 (팔꿈치 위/아래), 특이점 근처에서는 조건이 나쁠 수도 있다. 이러한 도전들을 이해하고 — 이를 처리하는 강건한 알고리즘을 갖추는 것이 교과서 로보틱스와 실세계 로보틱스를 구분한다.

> **비유**: IK는 역방향 내비게이션과 같다 — 목적지가 주어졌을 때 턴-바이-턴(turn-by-turn) 길안내를 찾는 것이다. 같은 목적지까지 여러 경로가 있을 수 있고, 어떤 경로는 더 길고, 어떤 도로는 막혀 있을 수 있듯이 (특이점), IK는 잠재적으로 많은 후보 관절 구성 중에서 실행 가능한 것을 찾아야 한다.

---

## IK 문제

### 형식적 정의

원하는 말단 작동체 포즈 $T_d \in SE(3)$ (4x4 동차 변환 행렬)가 주어졌을 때, 다음을 만족하는 관절 구성 $\mathbf{q} = (q_1, q_2, \ldots, q_n)^T$를 찾는다:

$$\text{FK}(\mathbf{q}) = T_d$$

또는 동치로:

$$\mathbf{f}(\mathbf{q}) = \mathbf{x}_d$$

여기서 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$은 순운동학 함수이고 $\mathbf{x}_d$는 원하는 작업 공간 구성 (위치 + 방위각)이다.

### IK가 어려운 이유

| FK | IK |
|----|-----|
| 단일하고 유일한 답 | 0개, 1개, 또는 여러 개의 해 |
| 매끄럽고 잘 조건화됨 | 특이점 근처에서 조건이 나쁠 수 있음 |
| 닫힌 형태(closed-form)가 항상 존재 (행렬을 곱하기만 하면 됨) | 특정 기하학에서만 닫힌 형태가 존재 |
| $O(n)$ 계산 | 반복적 방법이 필요할 수 있음 |

### 해의 존재와 유일성

- **해 없음**: 목표가 작업 공간 밖
- **유한 개의 해**: 구형 손목(spherical wrist)을 가진 대부분의 6-자유도 로봇은 최대 16개의 해를 가짐
- **무한한 해**: 여유 자유도 로봇 ($n > 6$)은 도달 가능한 모든 목표에 대해 무한히 많은 해를 가짐
- **고립된 해**: 정규 점에서의 비여유 자유도 로봇은 이산적(유한) 해의 집합을 가짐

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

## 해석적(Closed-Form) IK

해석적 IK는 기하학적 통찰이나 대수적 조작을 통해 정확한 해를 제공한다. 이용 가능한 경우 선호되는데, 빠르고 정확하며 모든 해를 열거할 수 있기 때문이다.

### 기하학적 방법: 2-링크 평면 로봇

2-링크 평면 로봇은 전형적인 예제다. 원하는 위치 $(x_d, y_d)$가 주어졌을 때, $(\theta_1, \theta_2)$를 구한다.

**설정**: 링크 길이 $l_1$, $l_2$; 말단 작동체 위치:
$$x = l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2)$$
$$y = l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2)$$

**1단계**: 코사인 법칙을 사용하여 $\theta_2$를 구한다.

원점에서 목표까지의 거리:
$$r^2 = x_d^2 + y_d^2 = l_1^2 + l_2^2 + 2 l_1 l_2 \cos\theta_2$$

$\theta_2$를 풀면:
$$\cos\theta_2 = \frac{x_d^2 + y_d^2 - l_1^2 - l_2^2}{2 l_1 l_2}$$

$$\theta_2 = \pm \arccos\left(\frac{x_d^2 + y_d^2 - l_1^2 - l_2^2}{2 l_1 l_2}\right)$$

$\pm$에 의해 **두 가지 해**가 생긴다: **팔꿈치 위(elbow-up)**와 **팔꿈치 아래(elbow-down)**.

**2단계**: 기하학을 이용하여 $\theta_1$을 구한다.

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

### 대수적 방법: 손목 분리(Detaching the Wrist)

**구형 손목(spherical wrist)**을 가진 6-자유도 로봇(관절 4, 5, 6의 축이 한 점에서 교차)의 경우, 문제를 다음과 같이 분리(decouple)할 수 있다:

1. **위치 IK** (관절 1, 2, 3): 손목 중심(wrist center)의 위치를 구한다
2. **방위각 IK** (관절 4, 5, 6): 원하는 방위각을 달성하기 위한 손목 각도를 구한다

손목 중심 위치 $\mathbf{p}_w$는 다음과 같다:

$$\mathbf{p}_w = \mathbf{p}_d - d_6 \cdot {}^{0}R_d \cdot \hat{z}_6$$

여기서 $\mathbf{p}_d$는 원하는 말단 작동체 위치, ${}^{0}R_d$는 원하는 방위각, $d_6$는 손목에서 도구까지의 거리다.

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

### 해의 수

일반적인 구형 손목을 가진 6-자유도 로봇에서:
- 팔 (관절 1-3)에 대해 최대 **8개의 해**: 좌/우 어깨, 팔꿈치 위/아래, 손목 뒤집기
- 손목 (관절 4-6)에 대해 최대 **2개의 해**: 손목 뒤집기 ($\theta_5$ 양수 또는 음수)
- **합계: 최대 16개**의 고유한 해

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

## 수치적 IK 방법

닫힌 형태 해가 없는 경우 (비표준 기하학, 여유 자유도 로봇, 제약된 IK), 반복적인 수치 방법을 사용한다.

### 야코비안 기반 프레임워크

모든 수치적 IK 방법은 **야코비안(Jacobian)** (Lesson 5에서 자세히 다룸)을 기반으로 한다. 야코비안 $J(\mathbf{q})$는 관절 각도의 미소 변화를 말단 작동체 포즈의 미소 변화와 연관시킨다:

$$\delta \mathbf{x} = J(\mathbf{q}) \, \delta \mathbf{q}$$

IK를 위해서는 역방향 관계가 필요하다:

$$\delta \mathbf{q} = J(\mathbf{q})^{-1} \, \delta \mathbf{x}$$

그러나 $J$가 정방행렬(square matrix)이 아닐 수 있고, 정방행렬이더라도 특이(singular)할 수 있다. 서로 다른 방법들이 이 문제를 다르게 처리한다.

### 방법 1: 뉴턴-랩슨(Newton-Raphson)

뉴턴-랩슨 방법은 반복적 선형화를 적용한다:

$$\mathbf{q}_{k+1} = \mathbf{q}_k + J(\mathbf{q}_k)^{-1} \, (\mathbf{x}_d - \mathbf{f}(\mathbf{q}_k))$$

여기서 $\mathbf{f}(\mathbf{q}_k)$는 FK로부터 구한 현재 말단 작동체 포즈다.

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

### 방법 2: 야코비안 유사역행렬(Jacobian Pseudo-Inverse)

야코비안이 정방행렬이 아닌 경우 ($m \neq n$, 여기서 $m$은 작업 공간 차원, $n$은 관절 차원), **무어-펜로즈 유사역행렬(Moore-Penrose pseudo-inverse)**을 사용한다:

$m < n$ (여유 자유도 로봇 — 작업 자유도보다 관절이 더 많은 경우):
$$\delta \mathbf{q} = J^+ \, \delta \mathbf{x} = J^T (J J^T)^{-1} \, \delta \mathbf{x}$$

이는 **최소 노름(minimum-norm)** 해를 제공한다 — 원하는 작업 공간 변화를 달성하는 최소 관절 운동.

$m > n$ (과소결정(under-determined) — 작업 자유도보다 관절이 더 적은 경우):
$$\delta \mathbf{q} = J^+ \, \delta \mathbf{x} = (J^T J)^{-1} J^T \, \delta \mathbf{x}$$

이는 **최소 제곱(least-squares)** 해를 제공한다 — 원하는 변화에 최대한 가깝게 만드는 관절 운동.

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

### 방법 3: 감쇠 최소 제곱법(Levenberg-Marquardt)

유사역행렬은 $J J^T$가 거의 특이해지는 특이점 근처에서 실패한다. **감쇠 최소 제곱법(Damped Least Squares, DLS)**은 감쇠 항을 추가한다:

$$\delta \mathbf{q} = J^T (J J^T + \lambda^2 I)^{-1} \, \delta \mathbf{x}$$

여기서 $\lambda$는 **감쇠 계수(damping factor)**다. 이는 특이점 근처에서 정확도와 강건성 사이의 균형을 맞춘다.

- $\lambda = 0$일 때: 유사역행렬과 동일
- $\lambda$가 클 때: 운동이 작지만 잘 조건화됨
- 특이점 근처에서: $\lambda$가 폭발적인 관절 속도를 방지

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

## 완전한 IK 예제: 2-링크 평면 로봇

해석적 방법과 수치적 방법을 비교하는 완전한 IK 솔루션을 구현한다.

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

## 특이점과 그 영향

### 특이점이란 무엇인가

**특이점(singularity)**은 야코비안 $J(\mathbf{q})$의 랭크(rank)가 감소하는 관절 구성이다 — 즉, 정방 야코비안에서 $\det(J) = 0$이거나, 일반적으로 $\text{rank}(J) < \min(m, n)$.

특이점에서:
- 일부 말단 작동체 속도가 **불가능**해진다 (로봇이 하나 이상의 운동 자유도를 잃음)
- 역야코비안이 폭발한다 ($\|J^{-1}\| \to \infty$)
- IK 해가 수치적으로 불안정해진다

### 특이점의 유형

| 유형 | 위치 | 원인 | 예시 |
|------|----------|-------|---------|
| **경계(Boundary)** | 작업 공간의 경계 | 팔이 완전히 펼쳐지거나 완전히 접힘 | 2-링크: $\theta_2 = 0$ 또는 $\theta_2 = \pi$ |
| **내부(Interior)** | 작업 공간 내부 | 관절 축의 정렬 | 손목: $\theta_5 = 0$ (축 4, 6이 정렬) |
| **구조적(Architectural)** | 어디서나 | 로봇 설계 (퇴화된 기하학) | 드묾, 설계 결함을 나타냄 |

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

### 특이점 회피 전략

1. **감쇠 최소 제곱법(Damped Least Squares)**: 유사역행렬 대신 DLS 사용
2. **특이점 강건 작업 우선순위(Singularity-robust task priority)**: 특이점 근처에서 작업 공간 자유도를 줄임
3. **관절 한계 회피**: 관절을 특이 구성으로부터 멀리 유지
4. **경로 계획(Path planning)**: 특이 영역을 피하는 경로 계획 (Lesson 7)
5. **조작성 최대화(Manipulability maximization)**: 여유 자유도를 사용하여 잘 조건화된 구성 유지

---

## 여유 자유도 해결(Redundancy Resolution)

6 자유도(6-DOF 작업 기준) 이상의 관절을 가진 로봇은 **운동학적 여유 자유도(kinematic redundancy)**를 가진다 — 무한히 많은 관절 구성이 동일한 말단 작동체 포즈를 실현한다.

### 영공간(Null Space)

유사역행렬 해 $\delta \mathbf{q} = J^+ \delta \mathbf{x}$는 최소 노름 해다. 말단 작동체에 영향을 주지 않으면서 $J$의 **영공간(null space)**에 있는 어떤 벡터도 추가할 수 있다:

$$\delta \mathbf{q} = J^+ \delta \mathbf{x} + (I - J^+ J) \mathbf{z}$$

여기서 $(I - J^+ J)$는 **영공간 투영 행렬(null-space projector)**이고 $\mathbf{z}$는 임의의 벡터다.

영공간 성분 $(I - J^+ J)\mathbf{z}$는 말단 작동체를 움직이지 않고 관절을 움직인다 — 이를 **자기 운동(self-motion)**이라 한다.

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

## 실용적 고려 사항

### 적절한 방법 선택

| 시나리오 | 권장 방법 | 이유 |
|----------|-------------------|--------|
| 구형 손목을 가진 6-자유도 | 해석적 (분리) | 빠르고, 정확하며, 모든 해를 열거 |
| 단순한 기하학 (< 4 자유도) | 해석적 (기하학적) | 빠르고, 정확함 |
| 일반 기하학 | DLS | 강건하고, 특이점 처리 가능 |
| 여유 자유도 로봇 (7+ 자유도) | 유사역행렬 + 영공간 | 여유 자유도 활용 |
| 연속된 다수의 목표 | 이전 해로부터 수치적 방법 | 온기동(warm-starting) |
| 실시간 제어 (1 kHz) | 가능하면 해석적, 아니면 DLS | 속도 요구 사항 |

### 관절 한계(Joint Limits)

실제 로봇에는 관절 한계가 있다. IK 계산 후 항상 확인한다:

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

### 해 선택(Solution Selection)

여러 IK 해가 존재할 때, 다음 기준으로 선택한다:

1. **현재 구성에 가장 가까운 것**: 관절 운동을 최소화 (부드러운 궤적)
2. **가장 높은 조작성**: 이후 운동에 가장 잘 조건화됨
3. **관절 한계에서 가장 먼 것**: 최대 미래 이동 가능성
4. **장애물 회피**: 선택한 구성에서 충돌 없음

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

## 요약

- **역운동학(Inverse kinematics)**은 원하는 말단 작동체 포즈에 대한 관절 각도를 구한다 — 로봇 제어의 근본적인 문제
- **해석적 방법** (기하학적, 대수적, 손목 분리)은 특정 로봇 기하학에 대해 빠르고 정확한 해를 제공한다
- **수치적 방법** (뉴턴-랩슨, 유사역행렬, DLS)은 일반 기하학을 처리하지만 초기 추정값이 필요하고 국소 최솟값(local minima)에 수렴할 수 있다
- **감쇠 최소 제곱법(Damped Least Squares)**은 가장 강건한 수치적 방법으로, 특이점을 우아하게 처리한다
- **다중 해** (6-자유도의 경우 최대 16개)는 선택 기준 (근접성, 조작성, 한계)을 필요로 한다
- **특이점**은 자유도 상실과 수치적 불안정을 야기하므로 감지하고 관리해야 한다
- **여유 자유도 로봇**은 무한히 많은 해를 가지며, 영공간이 부차적인 목표를 가능하게 한다

---

## 연습 문제

### 연습 1: 2-링크 해석적 IK

$l_1 = 1.0$ m, $l_2 = 0.7$ m인 2-링크 평면 로봇에 대해:
1. 목표 $(1.2, 0.5)$에 대한 모든 IK 해를 구한다
2. 목표 $(0.3, 0.0)$에 대한 모든 IK 해를 구한다 — 이 경우의 특별한 점은 무엇인가?
3. 목표 $(2.0, 0.0)$에 대해 IK를 시도한다 — 어떤 일이 발생하고 왜 그런가?
4. 도달 가능한 목표에 대한 두 해를 플롯하여 팔 구성을 보여준다

### 연습 2: 수치적 IK 비교

2-링크 로봇에 대해 뉴턴-랩슨, 유사역행렬, DLS를 구현한다. 목표 $(0.5, 1.0)$에 대해:
1. $q_0 = (0, 0)$에서 시작하여 수렴 속도 (허용 오차 $10^{-8}$까지 필요한 반복 횟수)를 비교한다
2. 작업 공간 경계 근처에서의 동작 비교: $l_1 + l_2 - 0.01$ 거리의 목표
3. 특이 구성에서의 동작 비교: $q_0 = (0, 0)$ (팔이 완전히 펼쳐진 상태)에서 시작

### 연습 3: 특이점 분석

2-링크 로봇에 대해:
1. 야코비안을 해석적으로 유도한다
2. $\det(J) = 0$이 되는 모든 구성을 구한다
3. 특이 구성 $q = (45°, 0°)$에서, 어떤 말단 작동체 속도가 불가능한가?
4. 조작성 척도 $w(\theta_2) = |l_1 l_2 \sin\theta_2|$를 플롯하고 그 동작을 설명한다

### 연습 4: 6-자유도 IK 프레임워크

Lesson 3의 PUMA 560 유사형 DH 매개변수를 사용하여:
1. 분리 IK를 위한 손목 중심 계산을 구현한다
2. 관절 4-6을 변경하면서 손목 중심 위치가 관절 1-3에만 의존하는지 확인한다
3. 원하는 말단 작동체 포즈 $T_d$ (직접 선택)에 대해 손목 중심을 계산하고 $\theta_1$을 기하학적으로 구한다

### 연습 5: 여유 자유도

2D 목표(2-자유도 작업)에 도달하는 3-링크 평면 로봇(회전 관절 3개)을 고려한다:
1. 로봇은 1 자유도의 여유 자유도를 가진다. 자기 운동은 어떤 모양인가?
2. 영공간 관절 중심화(null-space joint centering)를 사용한 유사역행렬 IK를 구현한다
3. $q_0 = (20°, 40°, 60°)$에서 시작하여 목표 $(0.5, 0.8)$에 도달한다. 영공간 최적화를 사용한 경우와 사용하지 않은 경우의 해를 비교한다

---

[← 이전: 순운동학](03_Forward_Kinematics.md) | [다음: 속도 운동학 →](05_Velocity_Kinematics.md)
