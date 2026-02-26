# 9. 로봇 제어

[← 이전: 궤적 계획 및 실행](08_Trajectory_Planning.md) | [다음: 센서와 인식 →](10_Sensors_and_Perception.md)

---

## 학습 목표

1. 로봇 매니퓰레이터를 위한 관절 공간 PID 제어기를 설계하고 튜닝한다
2. 역동역학과 PD 피드백을 이용한 계산 토크 제어를 유도하고 구현한다
3. 임피던스 제어(Impedance Control)와 그 스프링-댐퍼 모델을 순응적 상호작용에 적용하는 방법을 이해한다
4. 힘 제어와 혼합 위치/힘 제어 전략의 차이를 구분한다
5. 모델 불확실성을 다루기 위한 적응 제어(Adaptive Control) 원리를 인식한다
6. 로봇 제어 시스템에서 강인성과 외란 억제를 분석한다

---

아름다운 궤적을 계획할 수 있는 로봇도 물리 세계에서 정확하게 실행하지 못한다면 쓸모가 없다. 제어는 계획된 운동과 실제 운동 사이의 다리 역할을 한다 — 중력, 마찰, 관성, 예상치 못한 외란에도 불구하고 로봇의 관절이 원하는 궤적을 따르도록 올바른 토크, 힘, 명령을 계산하는 학문 분야다. 앞선 레슨에서는 로봇이 *무엇을* 해야 하는지(궤적 계획)를 배웠다. 이번 레슨에서는 *어떻게* 실제로 그것을 수행하게 만드는지를 배운다.

로봇 제어는 특히 매니퓰레이터가 고도로 비선형이고, 결합되어 있으며, 다중 입력 다중 출력(MIMO) 시스템이기 때문에 어렵다. 한 관절에 가해진 토크는 관성 결합(Inertial Coupling)을 통해 다른 모든 관절에 영향을 미친다. 각 관절의 유효 관성은 로봇의 형상(Configuration)에 따라 변한다. 팔이 움직임에 따라 중력 부하도 변한다. 이러한 복잡성으로 인해 대부분의 입문 과정에서 다루는 단순한 PID 제어기를 넘어서는 제어 전략이 필요하다 — 하지만 PID는 여전히 산업 로봇공학의 핵심이므로 여기서 시작하기로 한다.

---

## 1. 관절 공간 PID 제어

### 1.1 독립 관절 제어 패러다임

로봇 제어에 대한 가장 단순한 접근법은 각 관절을 독립적인 단일 입력, 단일 출력(SISO) 시스템으로 취급하는 것이다. 각 관절은 관절 수준의 추적 오차를 기반으로 토크 명령을 계산하는 자체 PID 제어기를 갖는다.

관절 $i$에 대한 제어 법칙은 다음과 같다:

$$\tau_i = K_{p,i} e_i(t) + K_{i,i} \int_0^t e_i(\sigma) \, d\sigma + K_{d,i} \dot{e}_i(t)$$

여기서 $e_i(t) = q_{d,i}(t) - q_i(t)$는 위치 오차, $q_{d,i}$는 원하는 위치, $q_i$는 측정된 위치다.

모든 $n$개 관절에 대한 벡터 형식:

$$\boldsymbol{\tau} = K_p \mathbf{e}(t) + K_i \int_0^t \mathbf{e}(\sigma) \, d\sigma + K_d \dot{\mathbf{e}}(t)$$

여기서 $K_p$, $K_i$, $K_d$는 대각 $n \times n$ 이득 행렬이다.

```python
import numpy as np

class JointPIDController:
    """Independent joint PID controller for an n-DOF robot.

    Why diagonal gain matrices? Each joint is treated independently,
    which simplifies tuning but ignores inter-joint coupling effects.
    This works well when gear ratios are high (reducing coupling)
    or when the robot moves slowly (small dynamic effects).
    """

    def __init__(self, n_joints, kp, ki, kd, dt=0.001):
        # Gain matrices are diagonal — one set of gains per joint
        self.Kp = np.diag(kp)  # Proportional gains [Nm/rad]
        self.Ki = np.diag(ki)  # Integral gains [Nm/(rad*s)]
        self.Kd = np.diag(kd)  # Derivative gains [Nm*s/rad]
        self.dt = dt

        # Integral accumulator and previous error for derivative
        self.integral_error = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)

    def compute(self, q_desired, q_actual, qd_desired=None, qd_actual=None):
        """Compute joint torques from tracking error.

        Why accept velocity signals separately? When velocity measurements
        are available (e.g., from tachometers), we can compute derivative
        action more accurately than by differencing position readings.
        """
        error = q_desired - q_actual
        self.integral_error += error * self.dt

        # Use measured velocities if available; otherwise differentiate error
        if qd_desired is not None and qd_actual is not None:
            derror = qd_desired - qd_actual
        else:
            derror = (error - self.prev_error) / self.dt

        self.prev_error = error.copy()

        tau = self.Kp @ error + self.Ki @ self.integral_error + self.Kd @ derror
        return tau

    def reset(self):
        """Reset integrator state — important when switching setpoints."""
        self.integral_error[:] = 0.0
        self.prev_error[:] = 0.0
```

### 1.2 로봇 관절을 위한 PID 튜닝

로봇 관절을 위한 PID 이득 튜닝은 단순한 선형 시스템보다 더 미묘하다. 유효 플랜트 동역학이 형상에 따라 변하기 때문이다. 팔이 펼쳐진 상태에서 잘 작동하는 이득 세트가 팔이 접혔을 때는 반응이 느리거나 진동할 수 있다.

**실용적인 튜닝 절차**:

1. **PD만으로 시작한다** ($K_i = 0$으로 설정). 초기 튜닝 중 적분기 와인드업(Windup)을 방지한다.
2. **$K_p$를 증가시켜** 관절이 빠르게 반응하지만 진동이 시작되기 직전까지 올린다.
3. **$K_d$를 추가하여** 진동을 감쇠시킨다. 미분 항은 가상의 댐퍼처럼 작용한다.
4. **작은 $K_i$를 추가하는 것은** 정상 상태 오차가 허용할 수 없는 경우에만 한다. 많은 로봇 응용에서 PD 제어만으로도 충분한데, 중력 보상(아래 참조)이 정상 상태 오차의 주된 원인을 제거하기 때문이다.
5. **작업 공간 전체에서 테스트한다** — 팔을 다양한 형상으로 이동시키고 안정성을 확인한다.

**중력 보상(Gravity Compensation)**은 관절 PID의 핵심적인 개선이다:

$$\boldsymbol{\tau} = K_p \mathbf{e} + K_d \dot{\mathbf{e}} + \mathbf{g}(\mathbf{q})$$

여기서 $\mathbf{g}(\mathbf{q})$는 중력 토크 벡터다. 전향(Feedforward) 중력 항을 추가함으로써 큰 $K_i$ 이득이 없으면 극복해야 했을 상수 외란을 제거한다.

```python
def pd_gravity_compensation(q_desired, q, qd, Kp, Kd, gravity_func):
    """PD control with gravity compensation.

    Why add gravity feedforward? Without it, the PD controller must use
    its proportional term to fight gravity, leading to a permanent
    position error (droop). The gravity term cancels this load exactly,
    so the PD only needs to handle dynamic tracking errors.
    """
    error = q_desired - q
    derror = -qd  # Desired velocity is zero for regulation

    tau = Kp @ error + Kd @ derror + gravity_func(q)
    return tau
```

### 1.3 독립 관절 PID의 한계

독립 관절 PID는 로봇의 결합 동역학을 무시한다:

- **형상 의존적 관성**: 각 관절 모터가 받는 유효 관성이 로봇이 움직임에 따라 변하므로, 고정된 이득 세트는 타협의 산물이 된다.
- **코리올리(Coriolis) 및 원심력 결합**: 한 관절에서의 빠른 운동은 독립 PID가 외란으로 처리해야 하는 토크를 다른 관절에 생성한다.
- **제한된 성능 범위**: 독립 관절 PID는 느리고 정밀한 운동(산업용 픽앤플레이스(Pick-and-Place)의 전형)에서는 잘 작동하지만 빠르고 역동적인 운동에서는 어려움을 겪는다.

이러한 한계들이 모델 기반 제어 방법을 동기화한다.

---

## 2. 계산 토크 제어

### 2.1 아이디어: 비선형성을 제거하다

레슨 6에서 로봇 동역학은 다음과 같음을 상기하자:

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

여기서 $M$은 관성 행렬, $C$는 코리올리 및 원심력 항을 포함하고, $\mathbf{g}$는 중력 벡터다.

**계산 토크 제어(Computed Torque Control)**(역동역학 제어(Inverse Dynamics Control) 또는 피드백 선형화(Feedback Linearization)라고도 함)는 동역학 모델을 사용하여 비선형 항을 제거하고, 결과적으로 "선형화된" 시스템에 단순한 선형 제어 법칙을 적용한다.

제어 법칙은 다음과 같다:

$$\boldsymbol{\tau} = M(\mathbf{q})\mathbf{u} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

여기서 $\mathbf{u}$는 보조 제어 입력이다. 동역학에 대입하면:

$$M(\mathbf{q})\ddot{\mathbf{q}} = M(\mathbf{q})\mathbf{u}$$

$M(\mathbf{q})$는 항상 양의 정부호(Positive Definite, 가역)이므로 다음을 얻는다:

$$\ddot{\mathbf{q}} = \mathbf{u}$$

이는 $n$개의 분리된 이중 적분기 — 선형 시스템이다! 이제 어떤 선형 제어 기법을 사용해도 $\mathbf{u}$를 설계할 수 있다. 표준적인 선택은 오차에 대한 PD 제어다:

$$\mathbf{u} = \ddot{\mathbf{q}}_d + K_d \dot{\mathbf{e}} + K_p \mathbf{e}$$

여기서 $\mathbf{e} = \mathbf{q}_d - \mathbf{q}$이다.

폐루프 오차 동역학은 다음과 같다:

$$\ddot{\mathbf{e}} + K_d \dot{\mathbf{e}} + K_p \mathbf{e} = \mathbf{0}$$

이는 임의의 양의 정부호 $K_p$와 $K_d$에 대해 안정적인 2차 시스템이다.

### 2.2 이득 선택

각 관절의 오차 동역학은:

$$\ddot{e}_i + 2\zeta_i \omega_{n,i} \dot{e}_i + \omega_{n,i}^2 e_i = 0$$

일반적인 형식과 비교하면 다음을 설정한다:

$$K_{p,i} = \omega_{n,i}^2, \qquad K_{d,i} = 2\zeta_i \omega_{n,i}$$

임계 감쇠($\zeta = 1$)로 고유 주파수 $\omega_n = 50$ rad/s를 원한다면:

$$K_p = 2500, \qquad K_d = 100$$

```python
class ComputedTorqueController:
    """Computed torque (inverse dynamics + PD) controller.

    Why use the full dynamic model? By computing and canceling the
    nonlinear dynamics (inertia coupling, Coriolis, gravity), we
    convert the control problem from a complex nonlinear MIMO problem
    into n independent linear double-integrator problems.
    """

    def __init__(self, robot_model, kp, kd):
        self.robot = robot_model
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)

    def compute(self, q_desired, qd_desired, qdd_desired, q, qd):
        """Compute control torques using inverse dynamics + PD.

        The three components:
        1. M(q)*u: Feedforward + feedback through the inertia matrix
        2. C(q,qd)*qd: Cancel velocity-dependent forces
        3. g(q): Cancel gravitational forces
        """
        error = q_desired - q
        derror = qd_desired - qd

        # Auxiliary input: desired acceleration + PD correction
        u = qdd_desired + self.Kp @ error + self.Kd @ derror

        # Full inverse dynamics
        M = self.robot.inertia_matrix(q)
        C = self.robot.coriolis_matrix(q, qd)
        g = self.robot.gravity_vector(q)

        tau = M @ u + C @ qd + g
        return tau
```

### 2.3 계산 토크 제어가 실패하는 경우

계산 토크 제어는 $M$, $C$, $\mathbf{g}$에 대한 **완벽한 지식**을 전제로 한다. 실제로는:

- **매개변수 불확실성**: 링크 질량, 관성, 질량 중심 위치는 정확히 알 수 없다.
- **모델링되지 않은 동역학**: 관절 유연성, 구동기 동역학, 기어 백래시(Backlash), 마찰이 강체 모델에서 빠져 있다.
- **계산 비용**: 서보 레이트(1 kHz 이상)에서 전체 동역학을 평가하려면 효율적인 알고리즘(예: 재귀 뉴턴-오일러(Newton-Euler))이 필요하다.

모델이 부정확하면 비선형 소거가 불완전하고, 폐루프 시스템은 더 이상 완벽한 이중 적분기 집합이 아니다. 잔여 비선형성이 외란으로 나타난다. 이것이 **강인(Robust)** 및 **적응 제어** 전략의 동기다.

---

## 3. 임피던스 제어

### 3.1 위치 제어에서 상호작용 제어로

표면을 연마하거나, 구멍에 핀을 삽입하거나, 사람에게 물체를 건네야 하는 로봇을 생각해 보자. 이러한 작업에서 로봇은 필연적으로 환경과 접촉한다. 순수한 위치 제어기는 다음 중 하나가 된다:

- 표면이 예상보다 약간 가까우면 과도한 힘을 가하거나,
- 표면이 약간 멀리 있으면 접촉을 유지하지 못한다.

각각을 독립적으로 제어하는 것이 아니라 **운동과 힘 사이의 관계**를 관리하는 제어기가 필요하다.

> **비유**: 임피던스 제어는 달걀을 쥐는 것과 같다 — 잡을 수 있을 만큼 단단하지만 깨지지 않을 만큼 부드럽게 강성을 조절한다. 위치 제어기는 딱딱한 클램프와 같다: 앞에 무엇이 있든 상관없이 특정 위치로 이동한다. 임피던스 제어기는 스프링-댐퍼 시스템처럼 동작한다: 저항에 부딪히면 양보하고, 평형점에서 벗어나면 밀어낸다.

### 3.2 임피던스 제어 법칙

**임피던스 제어**는 기준 위치로부터의 편차와 환경에 가해지는 힘 사이의 원하는 동적 관계를 명시한다:

$$\mathbf{F} = M_d(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}}) + B_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + K_d(\mathbf{x}_d - \mathbf{x})$$

여기서:
- $M_d$는 원하는 **관성** 행렬(로봇이 가속을 얼마나 저항하는지)
- $B_d$는 원하는 **감쇠(Damping)** 행렬(점성 마찰 동작)
- $K_d$는 원하는 **강성(Stiffness)** 행렬(스프링 같은 동작)
- $\mathbf{x}_d$는 기준 위치, $\mathbf{x}$는 실제 위치

**임피던스 매개변수의 물리적 해석과 단위**:

| 기호 | 이름 | 단위 (병진 / 회전) | 물리적 의미 |
|------|------|-------------------|------------|
| $M_d$ | 원하는 관성(Desired Inertia) | kg / kg$\cdot$m$^2$ | 환경이 느끼는 겉보기 질량; $M_d$가 클수록 가속에 대한 저항이 크다 |
| $B_d$ | 원하는 감쇠(Desired Damping) | N$\cdot$s/m / N$\cdot$m$\cdot$s/rad | 에너지 소산 비율; $B_d$가 클수록 진동을 억제하지만 응답이 느려진다 |
| $K_d$ | 원하는 강성(Desired Stiffness) | N/m / N$\cdot$m/rad | 단위 변위당 복원력; $K_d$가 클수록 위치 추적이 정밀하지만 접촉이 딱딱해진다 |

라플라스(Laplace) 영역에서 임피던스 전달함수는 $Z_d(s) = M_d s^2 + B_d s + K_d$이며, 변위를 힘으로 매핑한다. 이 세 매개변수를 선택하면 모든 주파수에서 로봇이 환경에 어떻게 느껴지는지 완전히 결정된다 — 저주파에서는 $K_d$(정적 강성), 중간 주파수에서는 $B_d$(감쇠), 고주파에서는 $M_d$(관성)가 지배한다.

로봇은 기준 궤적에 연결된 **질량-스프링-댐퍼 시스템**처럼 동작한다. 자유 공간에서는 $\mathbf{x}_d$를 밀접하게 추적한다. 환경과 접촉하면 $\mathbf{x}_d$에서 벗어나고 편차에 비례하는 힘을 가한다.

### 3.3 강성 및 감쇠 설계

$K_d$와 $B_d$의 선택은 작업에 따라 다르다:

| 작업 | 강성 | 감쇠 | 근거 |
|------|------|------|------|
| 연마 | 낮음 | 높음 | 표면에 순응, 튕김 방지 |
| 조립 (핀 삽입) | 삽입 방향으로 낮음, 측면으로 높음 | 중간 | 삽입 방향으로 순응, 측면으로 정밀 |
| 사람에게 건네기 | 매우 낮음 | 낮음 | 사람이 안내하기 쉽게 |
| 정밀 위치 결정 | 높음 | 임계 감쇠 | 강한 추적, 진동 없음 |

각 카르테시안(Cartesian) 자유도에 대한 **임계 감쇠**:

$$B_{d,i} = 2\sqrt{K_{d,i} \cdot M_{d,i}}$$

```python
class ImpedanceController:
    """Cartesian impedance controller.

    Why control impedance instead of position? When the robot interacts
    with an environment that has its own dynamics (stiffness, mass),
    the combined robot-environment system must be stable. Impedance
    control guarantees passivity: the robot absorbs energy on contact
    rather than injecting energy that could cause instability.
    """

    def __init__(self, Md, Bd, Kd):
        # Desired impedance parameters (6x6 for full Cartesian space)
        self.Md = np.array(Md)  # Desired inertia [kg, kg*m^2]
        self.Bd = np.array(Bd)  # Desired damping [Ns/m, Ns*m/rad]
        self.Kd = np.array(Kd)  # Desired stiffness [N/m, Nm/rad]

    def compute_force(self, x_desired, x_actual, xd_desired, xd_actual,
                      xdd_desired=None):
        """Compute desired Cartesian force from impedance model.

        Why separate from joint torque computation? The impedance model
        operates in Cartesian space. We convert to joint torques using
        the Jacobian: tau = J^T * F.
        """
        pos_error = x_desired - x_actual
        vel_error = xd_desired - xd_actual

        F = self.Kd @ pos_error + self.Bd @ vel_error

        if xdd_desired is not None:
            F += self.Md @ xdd_desired

        return F

    def compute_torque(self, F_cartesian, jacobian, q, qd, robot_model):
        """Convert Cartesian force command to joint torques.

        Why add gravity compensation? The impedance model defines the
        desired behavior in task space. Gravity is a joint-space
        disturbance that must be canceled separately.
        """
        # Map Cartesian force to joint torques
        tau = jacobian.T @ F_cartesian

        # Add gravity compensation
        tau += robot_model.gravity_vector(q)

        return tau
```

### 3.4 임피던스 제어 대 어드미턴스 제어(Admittance Control)

상호작용 제어에는 두 가지 이중적 접근법이 있다:

- **임피던스 제어**: 운동을 측정하고 힘을 명령한다. 입력: 변위 → 출력: 힘.
  - 우수한 토크 제어가 가능한 로봇에 적합(직접 구동, 직렬 탄성 구동기(Series Elastic Actuator)).
- **어드미턴스 제어**: 힘을 측정하고 운동을 명령한다. 입력: 힘 → 출력: 변위.
  - 위치 제어가 되는 뻣뻣한 로봇에 적합(높은 기어비를 가진 산업용 로봇).

$$\text{임피던스: } \mathbf{F} = Z(\mathbf{x}_d - \mathbf{x})$$
$$\text{어드미턴스: } \mathbf{x} = Y(\mathbf{F}_{ext})$$

여기서 $Z$는 임피던스 연산자이고 $Y = Z^{-1}$는 어드미턴스 연산자다.

**어떤 것을 선택해야 하는가?** 선택은 로봇의 고유 동역학과 구동 방식에 따라 달라진다. 로봇이 본래 백드라이버블(backdrivable, 저기어비, 직접 구동)하다면 정밀한 토크 명령이 가능하므로, 임피던스 제어(운동 측정, 힘 출력)가 자연스럽다. 높은 기어비를 가진 로봇(딱딱한 위치 제어 산업용 로봇)은 정밀한 위치 명령이 쉽지만 정밀한 토크 명령은 어려우므로, 어드미턴스 제어(센서를 통한 힘 측정, 위치 보정 출력)가 선호된다. 수학적으로, 결합된 로봇-환경 임피던스가 안정성을 위해 수동(passive, 양의 실수(positive real))을 유지해야 한다; 하드웨어의 고유 임피던스를 보완하는 제어 모드를 선택하면 더 간단한 제어기 설계로 이 조건을 만족시킬 수 있다.

---

## 4. 힘 제어

### 4.1 직접 힘 제어

일부 작업에서는 접촉 힘을 직접 제어해야 한다. 예를 들어, 표면을 연삭하는 로봇은 표면의 불규칙성에 관계없이 특정 법선 방향 힘을 유지해야 한다.

가장 단순한 힘 제어기는 힘 오차에 대한 PI 피드백을 사용한다:

$$\boldsymbol{\tau}_f = K_{fp}(\mathbf{F}_d - \mathbf{F}) + K_{fi} \int_0^t (\mathbf{F}_d - \mathbf{F}(\sigma)) \, d\sigma$$

여기서 $\mathbf{F}_d$는 원하는 힘이고 $\mathbf{F}$는 측정된 힘(힘/토크 센서로부터)이다.

**순수 힘 제어의 과제**:

- 힘/토크 센서가 필요하다(비용, 노이즈, 드리프트).
- 로봇이 접촉을 잃으면 힘 오차가 $\mathbf{F}_d$로 가고 제어기는 로봇을 표면에 공격적으로 밀어 넣는다 — 이는 명시적인 접촉 감지와 모드 전환이 필요하다.
- 힘 측정 노이즈가 힘 제어의 대역폭을 제한한다.

### 4.2 혼합 위치/힘 제어

**혼합 제어**(Raibert and Craig, 1981)는 **선택 행렬(Selection Matrix)** $S$를 사용하여 작업 공간을 위치 제어와 힘 제어 부분 공간으로 분리한다:

$$\boldsymbol{\tau} = J^T\left[S \cdot \mathbf{F}_{force} + (I - S) \cdot \mathbf{F}_{position}\right]$$

여기서:
- $S$는 힘 제어 자유도에 대해 1, 위치 제어 자유도에 대해 0을 갖는 대각 선택 행렬
- $\mathbf{F}_{force}$는 힘 제어기로부터
- $\mathbf{F}_{position}$은 위치/임피던스 제어기로부터

**예시**: 테이블 표면 닦기:

| 자유도 | 제어 모드 | 근거 |
|--------|-----------|------|
| $x$ (표면을 따라) | 위치 | 표면을 가로질러 이동 |
| $y$ (표면을 따라) | 위치 | 표면을 가로질러 이동 |
| $z$ (표면 법선 방향) | 힘 | 일정한 누르는 힘 유지 |
| $\theta_x, \theta_y$ | 힘(영) | 표면에 순응하는 기울기 허용 |
| $\theta_z$ | 위치 | 와이퍼의 방향 제어 |

```python
class HybridController:
    """Hybrid position/force controller.

    Why split DOFs between position and force control? In contact tasks,
    some directions are constrained by the environment (e.g., you can't
    move through a wall), so force control is natural. Other directions
    are free (e.g., sliding along the wall), so position control makes sense.
    """

    def __init__(self, pos_controller, force_controller, selection_matrix):
        self.pos_ctrl = pos_controller
        self.force_ctrl = force_controller
        # S: diagonal matrix, 1 = force-controlled, 0 = position-controlled
        self.S = np.diag(selection_matrix)

    def compute(self, x_d, xd_d, F_d, x, xd, F_measured, jacobian, q, robot):
        """Compute joint torques from hybrid position/force control.

        Why use a selection matrix? It provides a clean mathematical
        framework for assigning control modes per DOF. In practice,
        the selection can change dynamically based on the task phase
        (e.g., approach → contact → slide).
        """
        # Force-controlled DOFs
        F_force = self.force_ctrl.compute(F_d, F_measured)

        # Position-controlled DOFs
        F_pos = self.pos_ctrl.compute_force(x_d, x, xd_d, xd)

        # Combine using selection matrix
        F_task = self.S @ F_force + (np.eye(len(self.S)) - self.S) @ F_pos

        # Map to joint torques
        tau = jacobian.T @ F_task + robot.gravity_vector(q)
        return tau
```

---

## 5. 적응 제어

### 5.1 동기

계산 토크 제어는 동역학 매개변수에 대한 정확한 지식이 필요하다:

$$\boldsymbol{\theta} = [m_1, m_1 l_{c1}, I_1, m_2, m_2 l_{c2}, I_2, \ldots]^T$$

실제로 이러한 매개변수에는 불확실성이 있다. **적응 제어**는 로봇을 제어하면서 이러한 매개변수를 온라인으로 추정한다.

### 5.2 매개변수에서의 선형성

로봇 동역학의 핵심 특성은 방정식이 **동역학 매개변수에 대해 선형**이라는 것이다:

$$M(\mathbf{q})\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = Y(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\theta}$$

여기서 $Y$는 **회귀 행렬(Regressor Matrix)**(운동학에 의존하고 미지의 매개변수에는 의존하지 않음)이고 $\boldsymbol{\theta}$는 매개변수 벡터다.

이를 통해 추적 오차를 사용하여 $\hat{\boldsymbol{\theta}}$를 갱신하는 적응 법칙을 설계할 수 있다.

### 5.3 적응 계산 토크

계산 토크 제어의 적응 버전은 실제 매개변수를 추정된 매개변수로 대체한다:

$$\boldsymbol{\tau} = \hat{M}(\mathbf{q})\mathbf{u} + \hat{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \hat{\mathbf{g}}(\mathbf{q})$$

여기서 $\hat{M}$, $\hat{C}$, $\hat{\mathbf{g}}$는 현재 매개변수 추정값 $\hat{\boldsymbol{\theta}}$를 사용한다.

매개변수 갱신 법칙:

$$\dot{\hat{\boldsymbol{\theta}}} = \Gamma Y^T(\mathbf{q}, \dot{\mathbf{q}}, \dot{\mathbf{q}}_r, \ddot{\mathbf{q}}_r) \mathbf{s}$$

여기서:
- $\Gamma > 0$는 적응 이득 행렬
- $\mathbf{s} = \dot{\mathbf{e}} + \Lambda \mathbf{e}$는 슬라이딩 변수(Sliding Variable)
- $\dot{\mathbf{q}}_r = \dot{\mathbf{q}}_d + \Lambda \mathbf{e}$는 기준 속도
- $\Lambda > 0$는 설계 매개변수

```python
class AdaptiveController:
    """Adaptive computed torque controller.

    Why adapt online? In real robots, payloads change (picking up objects),
    joints wear (friction changes), and the environment varies. Adaptive
    control automatically adjusts the internal model to maintain performance
    without manual re-identification.
    """

    def __init__(self, n_joints, n_params, Kd, Lambda, Gamma):
        self.Kd = np.diag(Kd)           # PD gain
        self.Lambda = np.diag(Lambda)    # Sliding surface slope
        self.Gamma = np.diag(Gamma)      # Adaptation rate
        self.theta_hat = np.zeros(n_params)  # Parameter estimates

    def compute(self, q_d, qd_d, qdd_d, q, qd, regressor_func, dt):
        """Compute torque and update parameter estimates.

        Why use a sliding variable s? It combines position and velocity
        errors into a single measure. When s → 0, both the position
        and velocity errors converge to zero. This simplifies the
        stability analysis (Lyapunov-based).
        """
        e = q_d - q
        ed = qd_d - qd

        # Reference velocity and acceleration
        qd_r = qd_d + self.Lambda @ e
        qdd_r = qdd_d + self.Lambda @ ed

        # Sliding variable
        s = ed + self.Lambda @ e

        # Regressor matrix: dynamics = Y(q, qd, qd_r, qdd_r) * theta
        Y = regressor_func(q, qd, qd_r, qdd_r)

        # Control torque using current parameter estimates
        tau = Y @ self.theta_hat + self.Kd @ s

        # Update parameter estimates
        self.theta_hat += self.Gamma @ Y.T @ s * dt

        return tau
```

### 5.4 실용적 고려 사항

- **지속 여기(Persistent Excitation)**: 로봇 운동이 모든 동역학 매개변수를 여기(Excite)할 만큼 "풍부"해야만 매개변수가 수렴한다. 실제로 이는 로봇이 다양한 형상을 통해 이동해야 함을 의미한다.
- **매개변수 투영(Parameter Projection)**: 추정된 매개변수는 물리적으로 의미 있게 유지되어야 한다(예: 질량은 양수여야 함). 투영 알고리즘이 이러한 제약을 적용한다.
- **수렴 속도 대 노이즈 민감도**: 높은 적응 이득 $\Gamma$는 더 빠른 수렴을 의미하지만 측정 노이즈에 더 민감하다.

---

## 6. 강인성과 외란 억제

### 6.1 외란의 원인

실제 로봇은 수많은 외란에 직면한다:

| 원인 | 특성 | 전형적인 크기 |
|------|------|---------------|
| 관절 마찰 | 비선형(쿨롱(Coulomb) + 점성) | 정격 토크의 5-15% |
| 페이로드 불확실성 | 매개변수적 | 작업에 따라 다양 |
| 외부 접촉 | 충격 또는 지속 | 작업 의존적 |
| 센서 노이즈 | 고주파 | 범위의 0.1-1% |
| 모델 불일치 | 구조적 불확실성 | 모델 항의 10-30% |

### 6.2 강인 제어: 슬라이딩 모드(Sliding Mode)

**슬라이딩 모드 제어**는 불연속적인 스위칭 항을 추가하여 유계(Bounded) 모델 불확실성에 대한 강인성을 제공한다:

$$\boldsymbol{\tau} = \hat{M}\mathbf{u} + \hat{C}\dot{\mathbf{q}} + \hat{\mathbf{g}} + K_{robust} \cdot \text{sgn}(\mathbf{s})$$

여기서 $\text{sgn}(\mathbf{s})$는 부호 함수(Signum Function)를 원소별로 적용한 것이다.

스위칭 항은 불확실성에도 불구하고 시스템을 **슬라이딩 곡면(Sliding Surface)** $\mathbf{s} = \dot{\mathbf{e}} + \Lambda \mathbf{e} = \mathbf{0}$으로 밀어낸다.

**채터링(Chattering) 문제**: 불연속적인 $\text{sgn}$ 함수는 실제로 고주파 스위칭(채터링)을 일으킨다. 해결책은 다음을 포함한다:

1. **경계 레이어(Boundary Layer)**: $\text{sgn}(s)$를 $\text{sat}(s/\phi)$로 대체하고, 여기서 $\phi$는 경계 레이어 두께
2. **수퍼-트위스팅(Super-twisting) 알고리즘**: 연속 제어를 생성하는 고차 슬라이딩 모드
3. **적응 경계**: 관찰된 채터링을 기반으로 $\phi$를 조정

```python
def sliding_mode_control(q_d, qd_d, qdd_d, q, qd, robot, Lambda, K_robust, phi):
    """Sliding mode controller with boundary layer.

    Why use a boundary layer? Pure sliding mode causes chattering —
    high-frequency oscillation around the sliding surface. The boundary
    layer replaces the hard switching with a smooth approximation,
    trading some robustness for practical smoothness.
    """
    e = q_d - q
    ed = qd_d - qd

    # Sliding variable
    s = ed + Lambda @ e

    # Reference trajectory
    qd_r = qd_d + Lambda @ e
    qdd_r = qdd_d + Lambda @ ed

    # Nominal computed torque
    M = robot.inertia_matrix(q)
    C = robot.coriolis_matrix(q, qd)
    g = robot.gravity_vector(q)

    u = qdd_r  # Could add PD here for faster convergence
    tau_nominal = M @ u + C @ qd + g

    # Robust switching term with boundary layer (saturation function)
    # sat(s/phi) = s/phi if |s| < phi, else sgn(s)
    sat = np.clip(s / phi, -1.0, 1.0)
    tau_robust = K_robust @ sat

    return tau_nominal + tau_robust
```

### 6.3 외란 관측기(Disturbance Observer, DOB)

**외란 관측기**는 시스템에 작용하는 집중 외란(Lumped Disturbance)을 추정하고 이를 제거한다:

$$\hat{d} = Q(s) \left[\boldsymbol{\tau} - M_n \ddot{\mathbf{q}}\right]$$

여기서 $M_n$은 공칭 모델이고 $Q(s)$는 저역 통과 필터다. 추정된 외란 $\hat{d}$는 모델 불확실성, 마찰, 외력을 포함한다. 이러한 외란을 억제하기 위해 제어 명령에서 $\hat{d}$를 뺀다.

**슬라이딩 모드 대비 장점**: 연속적인 제어 신호(채터링 없음), 어떤 내부 루프 제어기와도 작동, 결함 감지(Fault Detection)에 유용한 외란 추정값 제공.

---

## 7. 제어 아키텍처 개요

### 7.1 계층적 제어

현대 로봇 제어 시스템은 계층적 아키텍처를 사용한다:

```
┌──────────────────────────────────┐
│  Task-Level Planner (1-10 Hz)    │  ← Trajectory waypoints
├──────────────────────────────────┤
│  Cartesian Controller (100 Hz)   │  ← Impedance/force control
├──────────────────────────────────┤
│  Joint Controller (1 kHz)        │  ← PID/computed torque
├──────────────────────────────────┤
│  Motor Driver (10 kHz)           │  ← Current control
└──────────────────────────────────┘
```

각 수준은 다른 속도로 작동하며, 상위 수준이 하위 수준에 설정값을 제공한다.

### 7.2 제어 전략 비교

| 전략 | 필요한 모델 | 접촉 처리 | 강인성 | 복잡도 |
|------|------------|-----------|--------|--------|
| 관절 PID | 없음(최소) | 불량 | 낮음 | 낮음 |
| PD + 중력 | 중력 모델 | 불량 | 중간 | 낮음 |
| 계산 토크 | 전체 동역학 | 불량 | 낮음(민감) | 높음 |
| 임피던스 | 전체 동역학 | 우수 | 중간 | 높음 |
| 혼합 | 전체 동역학 | 우수 | 중간 | 매우 높음 |
| 적응 | 회귀 형식 | 중간 | 높음 | 매우 높음 |
| 슬라이딩 모드 | 불확실성 범위 | 중간 | 매우 높음 | 높음 |

### 7.3 실용적인 권장 사항

- **산업용 픽앤플레이스**: PD + 중력 보상(단순하고, 신뢰할 수 있으며, 높은 기어비의 느린 운동에 충분).
- **고속 매니퓰레이션**: 계산 토크 또는 적응 제어(동적 효과를 보상해야 함).
- **인간-로봇 협업**: 임피던스 제어(순응을 통한 안전).
- **접촉이 많은 작업**: 혼합 위치/힘 또는 임피던스 제어.
- **불확실한 환경**: 적응 또는 슬라이딩 모드 제어.

---

## 8. 종합: 시뮬레이션 예제

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_2link_control(controller_type='pid', T=5.0, dt=0.001):
    """Simulate control of a 2-link planar robot.

    Why simulate before deploying? Control parameters that look
    good on paper can fail catastrophically on a real robot.
    Simulation lets us tune gains, test edge cases (singularities,
    joint limits), and verify stability safely.
    """
    # Robot parameters
    m1, m2 = 1.0, 1.0  # Link masses [kg]
    l1, l2 = 1.0, 1.0  # Link lengths [m]
    lc1, lc2 = 0.5, 0.5  # Center of mass distances [m]
    I1 = m1 * l1**2 / 12  # Moments of inertia [kg*m^2]
    I2 = m2 * l2**2 / 12
    g_acc = 9.81

    def dynamics(q, qd, tau):
        """2-link planar robot dynamics (Euler-Lagrange).
        Returns joint accelerations given state and torques.
        """
        q1, q2 = q
        qd1, qd2 = qd

        # Inertia matrix entries
        d11 = m1*lc1**2 + I1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q2)) + I2
        d12 = m2*(lc2**2 + l1*lc2*np.cos(q2)) + I2
        d22 = m2*lc2**2 + I2

        M = np.array([[d11, d12], [d12, d22]])

        # Coriolis/centrifugal
        h = m2 * l1 * lc2 * np.sin(q2)
        C = np.array([[-h * qd2, -h * (qd1 + qd2)],
                       [h * qd1, 0.0]])

        # Gravity
        g = np.array([
            (m1*lc1 + m2*l1) * g_acc * np.cos(q1) + m2*lc2*g_acc*np.cos(q1+q2),
            m2 * lc2 * g_acc * np.cos(q1 + q2)
        ])

        qdd = np.linalg.solve(M, tau - C @ qd - g)
        return qdd

    def gravity_vec(q):
        q1, q2 = q
        return np.array([
            (m1*lc1 + m2*l1)*g_acc*np.cos(q1) + m2*lc2*g_acc*np.cos(q1+q2),
            m2*lc2*g_acc*np.cos(q1+q2)
        ])

    # Desired trajectory: sinusoidal joint motion
    def desired_trajectory(t):
        q_d = np.array([np.sin(t), 0.5 * np.sin(2 * t)])
        qd_d = np.array([np.cos(t), np.cos(2 * t)])
        qdd_d = np.array([-np.sin(t), -2 * np.sin(2 * t)])
        return q_d, qd_d, qdd_d

    # Initialize
    n_steps = int(T / dt)
    q = np.array([0.0, 0.0])
    qd = np.array([0.0, 0.0])

    q_history = np.zeros((n_steps, 2))
    qd_history = np.zeros((n_steps, 2))
    error_history = np.zeros((n_steps, 2))
    tau_history = np.zeros((n_steps, 2))

    # Controller gains
    Kp = np.diag([100.0, 100.0])
    Kd = np.diag([20.0, 20.0])

    for i in range(n_steps):
        t = i * dt
        q_d, qd_d, qdd_d = desired_trajectory(t)

        e = q_d - q
        ed = qd_d - qd

        if controller_type == 'pid':
            tau = Kp @ e + Kd @ ed + gravity_vec(q)
        elif controller_type == 'computed_torque':
            # Full computed torque (we reuse dynamics components)
            q1, q2 = q
            qd1, qd2 = qd
            d11 = m1*lc1**2+I1+m2*(l1**2+lc2**2+2*l1*lc2*np.cos(q2))+I2
            d12 = m2*(lc2**2+l1*lc2*np.cos(q2))+I2
            d22 = m2*lc2**2+I2
            M = np.array([[d11, d12], [d12, d22]])
            h = m2*l1*lc2*np.sin(q2)
            C = np.array([[-h*qd2, -h*(qd1+qd2)], [h*qd1, 0.0]])
            g = gravity_vec(q)
            u = qdd_d + Kp @ e + Kd @ ed
            tau = M @ u + C @ qd + g
        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        # Simulate dynamics (Euler integration)
        qdd = dynamics(q, qd, tau)
        qd = qd + qdd * dt
        q = q + qd * dt

        q_history[i] = q
        error_history[i] = e
        tau_history[i] = tau

    return q_history, error_history, tau_history

# Run comparison
# q_pid, e_pid, _ = simulate_2link_control('pid')
# q_ct, e_ct, _ = simulate_2link_control('computed_torque')
# Plot tracking errors to see the dramatic improvement of computed torque
```

---

## 요약

| 개념 | 핵심 아이디어 |
|------|--------------|
| 관절 PID | 단순하고, 관절을 독립적으로 처리; 정상 상태 정확도를 위해 중력 보상 추가 |
| 계산 토크 | 모델을 사용하여 비선형 동역학을 소거; 선형 이중 적분기 제어로 변환 |
| 임피던스 제어 | 힘-운동 관계를 제어; 로봇이 프로그래밍 가능한 스프링-댐퍼처럼 동작 |
| 힘 제어 | 힘/토크 센서 피드백을 이용한 접촉 힘의 직접 조절 |
| 혼합 제어 | 작업 공간을 위치 제어와 힘 제어 부분 공간으로 분할 |
| 적응 제어 | 모델 불확실성을 처리하기 위한 온라인 매개변수 추정 |
| 슬라이딩 모드 | 유계 불확실성에 대한 강인성을 위한 불연속 스위칭 |
| DOB | 저역 통과 필터링된 역모델을 이용하여 집중 외란을 추정하고 제거 |

---

## 연습 문제

1. **PID 튜닝 실험**: 중력 하에서 단일 관절 로봇(모터가 있는 진자)을 시뮬레이션한다. $K_p = 50$, $K_d = 10$, $K_i = 0$으로 시작한다. $K_i$를 점진적으로 증가시키고 정상 상태 오차와 진동에 미치는 영향을 관찰한다. 그런 다음 중력 보상을 추가하고 반복한다 — $K_i$가 불필요해지는 이유를 설명하라.

2. **계산 토크 대 PID**: 위의 2링크 시뮬레이션 코드를 사용하여 빠른 정현파 궤적($\omega = 5$ rad/s)에 대한 PD+중력과 계산 토크 제어 사이의 추적 오차를 비교한다. 두 관절의 위치 오차를 플롯한다. 속도가 증가함에 따라 성능 차이가 커지는 이유는?

3. **임피던스 제어 설계**: 로봇이 표면을 연마해야 한다. 3개의 자유도에 대한 임피던스 매개변수($K_d$, $B_d$)를 설계한다: 두 개의 접선 방향(표면을 따라)과 하나의 법선 방향. 로봇은 표면을 따라 원형 경로를 추적하면서 약 10 N의 법선 힘을 유지해야 한다. 제어기를 구현하고 시뮬레이션한다.

4. **혼합 제어기**: 핀 삽입 작업을 위한 혼합 위치/힘 제어기를 구현한다. 핀은 $z$축을 따라 정렬되어 있다. $z$에 힘 제어를 사용하고(5 N 삽입 힘 유지), $x$와 $y$에 위치 제어를 사용한다(핀을 중앙에 배치). 간단한 환경 모델로 시뮬레이션한다.

5. **슬라이딩 모드 채터링**: 경계 레이어가 있는 경우와 없는 경우 슬라이딩 모드 제어기를 구현한다. 두 경우에 대한 제어 토크를 플롯한다. 채터링 진폭과 주파수를 측정한다. 경계 레이어 두께 $\phi$가 강인성과 부드러움 사이의 트레이드오프에 어떤 영향을 미치는가?

---

## 추가 자료

- Siciliano, B. et al. *Robotics: Modelling, Planning and Control*. Springer, 2009. Chapters 8-9. (로봇 제어의 포괄적 처리)
- Slotine, J.-J. E. and Li, W. *Applied Nonlinear Control*. Prentice Hall, 1991. (적응 및 슬라이딩 모드 제어)
- Hogan, N. "Impedance Control: An Approach to Manipulation." *ASME Journal of Dynamic Systems*, 1985. (원본 임피던스 제어 논문)
- Raibert, M. H. and Craig, J. J. "Hybrid Position/Force Control of Manipulators." *ASME Journal of Dynamic Systems*, 1981.

---

[← 이전: 궤적 계획 및 실행](08_Trajectory_Planning.md) | [다음: 센서와 인식 →](10_Sensors_and_Perception.md)
