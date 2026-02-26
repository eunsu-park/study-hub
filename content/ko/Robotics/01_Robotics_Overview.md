# 로봇공학 개요 및 분류

[다음: 강체 변환 →](02_Rigid_Body_Transformations.md)

## 학습 목표

1. 초기 산업용 매니퓰레이터부터 현대의 자율 시스템에 이르기까지 로봇공학의 역사적 발전 과정을 추적한다
2. 기계 구조에 따라 로봇을 분류(매니퓰레이터, 이동 로봇, 공중 로봇, 수중 로봇)하고 각 범주의 특징을 이해한다
3. 자유도(DOF, Degrees of Freedom)를 정의하고, 자유도가 로봇의 작업 공간(workspace)과 능력을 어떻게 결정하는지 설명한다
4. 모든 로봇 시스템의 핵심 구성 요소인 센서, 액추에이터, 컨트롤러를 파악하고 각각의 역할을 설명한다
5. 제조, 의료, 우주 탐사, 농업, 물류 등 로봇공학의 주요 응용 분야를 개관한다
6. 로봇공학이 기계, 전자, 컴퓨터 과학을 결합한 본질적으로 다학제적인 분야임을 직관적으로 이해한다

---

## 왜 이것이 중요한가

로봇공학은 기계공학, 전기공학, 컴퓨터 과학이 교차하는 지점에 위치한다. 선형대수, 미분방정식, 최적화와 같은 추상적인 수학이 가장 구체적으로 표현되는 분야라고 할 수 있다. 움직이고, 감지하고, 결정을 내리는 물리적 기계가 그 산물이다. 변환(transformation), 기구학(kinematics), 동역학(dynamics)의 수학으로 들어가기 전에, 우리는 전체 영역의 지도가 필요하다. 이 레슨이 그 지도를 제공한다.

로봇이 어떻게 분류되는지, 어떤 구성 요소를 공유하는지, 어디에 배치되는지를 이해하면 이후의 수학적 도구들이 *왜* 존재하는지를 이해하는 데 필요한 어휘와 사고 틀이 생긴다. 레슨 2에서 유도하는 모든 회전 행렬(rotation matrix)은 실제 로봇 관절이 자신의 방향을 알아야 하기 때문에 존재한다. 레슨 5에서 계산하는 모든 야코비안(Jacobian)은 실제 컨트롤러가 원하는 끝단 효과기(end-effector) 동작을 모터 명령으로 변환해야 하기 때문에 존재한다.

> **비유**: 로봇은 인체와 같다 — 액추에이터는 근육, 센서는 눈과 귀, 컨트롤러는 뇌다. 인체 해부학을 이해하는 것이 생체역학 연구에 선행하듯, 로봇 해부학을 이해하는 것이 로봇 수학 연구에 선행한다.

---

## 로봇공학의 간략한 역사

### "로봇"이라는 단어

*robot*이라는 단어는 1920년 카렐 차페크(Karel Čapek)의 희곡 *R.U.R. (Rossum's Universal Robots)*를 통해 우리 어휘에 등장했다. 체코어 *robota*는 "강제 노동"을 의미한다. 아이작 아시모프(Isaac Asimov)는 이후 1941년 단편 소설 "거짓말쟁이!(Liar!)"에서 *로봇공학(robotics)*이라는 용어를 만들고 유명한 로봇공학 3원칙을 수립했다.

### 산업적 시작: 유니메이트(Unimate, 1961)

현대 로봇공학의 시대는 1961년 뉴저지 주 이윙(Ewing)의 제너럴 모터스(General Motors) 공장에 설치된 최초의 산업용 로봇 **유니메이트(Unimate)**와 함께 시작되었다. 조지 디볼(George Devol)과 조지프 엔겔버거(Joseph Engelberger)가 개발한 유니메이트는 다이캐스팅 작업을 수행하는 유압식 매니퓰레이터로, 기계에서 뜨거운 금속 부품을 꺼내 쌓는 작업을 담당했다.

연표의 주요 이정표:

| 연도 | 이정표 | 의의 |
|------|-----------|--------------|
| 1954 | 조지 디볼, 프로그래밍 가능한 매니퓰레이터 특허 출원 | 산업 로봇공학의 기초 |
| 1961 | GM에 유니메이트 설치 | 최초의 산업용 로봇 배치 |
| 1969 | 스탠퍼드 암(Stanford Arm, 빅터 샤인만) | 최초의 전기 구동·컴퓨터 제어 팔 |
| 1973 | KUKA Famulus | 6개의 전기기계 구동 축을 갖춘 최초의 산업용 로봇 |
| 1978 | PUMA (Programmable Universal Machine for Assembly) | 연구 및 경조립 분야의 표준 |
| 1997 | 화성의 소저너(Sojourner) | 다른 행성 최초의 이동 로봇 |
| 2000 | 혼다 ASIMO | 진보된 인간형 이족보행 |
| 2004 | DARPA 그랜드 챌린지 | 자율주행차 연구의 시작 |
| 2011 | 다빈치 수술 시스템 보급 | 로봇 수술의 주류화 |
| 2013 | 보스턴 다이나믹스 아틀라스(Atlas) | 동적 인간형 이동 |
| 2017 | 소피아(Sophia, 핸슨 로보틱스) | 소셜 로봇공학의 대중화 |
| 2020년대 | 창고 자동화, 배달 드론 | 물류 분야에서 로봇공학의 보편화 |

### 로봇공학의 세 물결

1. **1세대(1960년대-1980년대)**: 구조화된 환경에서의 반복 작업을 위한 산업용 매니퓰레이터(용접, 도장, 픽앤플레이스)
2. **2세대(1990년대-2010년대)**: 반구조화 환경에서 작동하는 서비스 로봇(수술 로봇, 청소기, 창고 AGV)
3. **3세대(2010년대-현재)**: 비구조화 환경의 자율 로봇(자율주행차, 배달 드론, 농업 로봇), AI·센싱·연산의 발전으로 가능해짐

---

## 로봇 분류

로봇은 여러 기준으로 분류할 수 있다. 가장 근본적인 분류는 **기계 구조**에 의한 것으로, 로봇이 공간에서 어떻게 움직이는지를 결정한다.

### 1. 매니퓰레이터(로봇 팔, Manipulators)

매니퓰레이터는 관절로 연결된 강체 링크의 기구학적 체인으로, 일반적으로 베이스에 고정된다. 산업용 로봇공학의 주력이다.

#### 직렬 매니퓰레이터(Serial Manipulators)

직렬 매니퓰레이터에서 링크들은 베이스에서 끝단 효과기까지 개방 체인(open chain)을 형성한다. 각 관절은 자유도를 하나씩 추가한다.

```
Base ──[Joint 1]── Link 1 ──[Joint 2]── Link 2 ── ... ──[Joint n]── End-Effector
```

일반적인 구성:

| 구성 | 자유도(DOF) | 관절 유형 | 작업 공간 형태 | 예시 |
|--------------|-----|-------------|-----------------|---------|
| 직교형(Cartesian, PPP) | 3 | 직동 3개 | 직육면체 | 갠트리 로봇 |
| 원통형(Cylindrical, RPP) | 3 | 회전 1 + 직동 2 | 원통 | Seiko RT3000 |
| 구형(Spherical, RRP) | 3 | 회전 2 + 직동 1 | 구의 일부 | 스탠퍼드 암 |
| SCARA(RRP) | 4 | 회전 2 + 직동 1 + 회전 1 | 원통 | Epson SCARA |
| 관절형(Articulated, RRR...) | 6 | 모두 회전 | 복잡 | KUKA, ABB, FANUC |

> **참고**: "R" = 회전 관절(Revolute joint), "P" = 직동 관절(Prismatic joint). 글자들은 베이스에서 끝단까지의 관절 유형을 나타낸다.

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

#### 병렬 매니퓰레이터(Parallel Manipulators)

병렬 매니퓰레이터에서는 복수의 기구학적 체인이 베이스와 끝단 효과기 플랫폼을 동시에 연결한다. 가장 유명한 예는 항공 시뮬레이터에 사용되는 **스튜어트-고프 플랫폼(Stewart-Gough platform)**이다.

**직렬 매니퓰레이터 대비 장점**:
- 높은 강성(rigidity)과 부하 용량(여러 체인에 힘이 분산됨)
- 더 높은 정밀도(오차가 단일 체인을 따라 누적되지 않음)
- 더 빠른 속도(액추에이터가 베이스 근처에 있어 이동 질량 감소)

**단점**:
- 풋프린트 대비 작은 작업 공간
- 더 복잡한 기구학(순운동학(forward kinematics)이 역운동학(inverse kinematics)보다 어려움!)
- 작업 공간 내의 특이 구성(singular configurations)

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

### 2. 이동 로봇(Mobile Robots)

이동 로봇은 베이스에 고정되지 않고 환경을 탐색한다.

#### 바퀴형 이동 로봇(Wheeled Mobile Robots)

가장 일반적인 범주로, 여러 가지 구동 구성이 있다:

| 유형 | 설명 | 전방향성(Holonomic)? | 예시 |
|------|-------------|------------|---------|
| 차동 구동(Differential drive) | 구동 바퀴 2개 + 캐스터 | 비전방향성(Non-holonomic) | Roomba, TurtleBot |
| 애커만 조향(Ackermann steering) | 자동차형 앞바퀴 조향 | 비전방향성 | 자율주행차 |
| 전방향(Omnidirectional) | 메카넘 또는 스웨디시 휠 | 전방향성(Holonomic) | KUKA youBot base |
| 스키드 조향(Skid-steer) | 탱크형 차동 | 비전방향성 | Clearpath Husky |

> **핵심 개념 — 전방향성(Holonomic) vs 비전방향성(Non-holonomic)**: 전방향성 로봇은 평면 위의 공처럼 순간적으로 어느 방향으로도 이동할 수 있다. 비전방향성 로봇은 순간적인 움직임에 제약이 있다 — 자동차는 옆으로 미끄러질 수 없다. 이 구분은 경로 계획(Lesson 7)에 깊은 영향을 미친다.

#### 다리형 로봇(Legged Robots)

다리를 이용한 이동은 바퀴가 실패하는 거친 비구조화 지형에서의 이동을 가능하게 한다.

| 다리 수 | 안정성 | 걸음새 복잡도 | 예시 |
|------|-----------|-----------------|---------|
| 1 (단각) | 동적 안정 | 호핑 | Boston Dynamics hopper |
| 2 (이족) | 동적 안정 | 걷기, 달리기 | Atlas, ASIMO, Digit |
| 4 (사족) | 정적/동적 | 걷기, 속보, 구보 | Spot, ANYmal |
| 6 (육족) | 정적 안정 (삼각 걸음새) | 다양한 옵션 | RHex |

**정적 안정성(Static stability)**: 무게중심의 투영이 지지 다각형(지면에 닿은 발의 볼록 껍질) 내에 있는 상태.

**동적 안정성(Dynamic stability)**: 로봇이 순간적으로 불안정할 수 있지만(마치 걷는 인간처럼 — 우리는 본질적으로 매 걸음마다 "앞으로 넘어지는" 것이다) 능동 제어를 통해 회복한다.

#### 궤도형 로봇(Tracked Robots)

궤도(무한궤도)형 로봇은 이동에 연속 트랙을 사용한다. 바퀴가 빠지는 느슨한 흙, 모래, 잔해 위에서 뛰어난 성능을 발휘한다. 군사용 폭발물 처리(EOD, Explosive Ordnance Disposal) 및 수색·구조 시나리오에서 흔히 사용된다.

### 3. 공중 로봇 (UAV)

무인 항공기(UAV, Unmanned Aerial Vehicles)는 인기와 성능이 폭발적으로 증가했다.

| 유형 | 메커니즘 | 특성 |
|------|-----------|-----------------|
| 멀티로터(쿼드콥터) | 4개 이상의 고정 피치 로터 | 수직 이착륙(VTOL), 호버링, 민첩성; 제한된 체공 시간 |
| 고정익 | 공기역학적 양력 | 장거리, 고속; 호버링 불가 |
| 하이브리드 (VTOL 고정익) | 틸트 로터 또는 테일시터 | 두 방식의 장점 결합; 복잡한 제어 |
| 날개짓형(Flapping-wing) | 생체모방 날개 동작 | 연구 단계; 소형에서 극도의 민첩성 |

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

### 4. 수중 로봇(Underwater Robots)

| 유형 | 설명 | 예시 |
|------|-------------|---------|
| ROV(원격 조종 수중기, Remotely Operated Vehicle) | 케이블 연결, 인간 제어 | Jason (WHOI) |
| AUV(자율 수중기, Autonomous Underwater Vehicle) | 케이블 없음, 자율 | Bluefin-21 |
| 수중 글라이더(Underwater glider) | 부력 구동, 매우 긴 체공 시간 | Slocum Glider |

수중 로봇은 독특한 도전에 직면한다: GPS를 사용할 수 없고(음향 측위 대신 사용), 통신 대역폭이 극도로 제한되며(음향 모뎀), 수압이 심각한 구조적 제약을 부과한다.

---

## 자유도(DOF, Degrees of Freedom)

### 정의

기계 시스템의 **자유도(Degrees of Freedom)**는 시스템의 구성(configuration)을 완전히 지정하는 데 필요한 독립 매개변수의 최소 수다.

3D 공간에서 자유롭게 움직이는 강체의 경우:
- 위치 $(x, y, z)$에 대한 자유도 3
- 방향(롤, 피치, 요)에 대한 자유도 3
- **합계: 자유도 6**

이것이 대부분의 산업용 매니퓰레이터가 6개의 관절을 갖는 이유다 — 끝단 효과기의 임의적인 위치 *및* 방향을 달성하기 위해서다.

### 그뤼블러 공식(Grubler's Formula)

$n$개의 링크(고정 베이스 포함), $j$개의 관절, 관절 $i$의 자유도 $f_i$인 메커니즘의 경우:

$$\text{DOF} = m(n - 1 - j) + \sum_{i=1}^{j} f_i$$

여기서 $m = 3$은 평면 메커니즘, $m = 6$은 공간 메커니즘에 해당한다.

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

### 작업 공간(Workspace)

로봇의 **작업 공간(workspace)**은 끝단 효과기가 도달할 수 있는 모든 위치(및 방향)의 집합이다.

- **도달 가능 작업 공간(Reachable workspace)**: 끝단 효과기가 적어도 하나의 방향으로 도달할 수 있는 모든 점
- **기민한 작업 공간(Dexterous workspace)**: 끝단 효과기가 *임의의* 방향으로 도달할 수 있는 모든 점

기민한 작업 공간은 항상 도달 가능 작업 공간의 부분집합이다. 6-DOF 로봇에서 기민한 작업 공간은 완전한 위치 및 방향 제어가 가능한 곳이다.

---

## 로봇의 핵심 구성 요소

모든 로봇은 유형에 관계없이 세 가지 기본 서브시스템으로 구성된다.

### 1. 센서(Sensors, 지각)

센서는 로봇에게 자기 자신(**고유 감각(proprioceptive)**)과 환경(**외부 감각(exteroceptive)**)에 대한 정보를 제공한다.

| 범주 | 센서 | 측정 대상 | 일반적 용도 |
|----------|--------|----------|-------------|
| 고유 감각 | 엔코더(Encoder) | 관절 각도/위치 | 관절 피드백 |
| 고유 감각 | IMU | 가속도, 각속도 | 방향 추정 |
| 고유 감각 | 힘/토크 센서 | 접촉력 | 힘 제어 |
| 고유 감각 | 전류 센서 | 모터 전류 | 토크 추정 |
| 외부 감각 | LiDAR | 거리 (3D 포인트 클라우드) | 지도 작성, 장애물 감지 |
| 외부 감각 | 카메라 (RGB) | 시각 이미지 | 물체 인식 |
| 외부 감각 | 깊이 카메라 (RGB-D) | 색상 + 깊이 | 3D 지각 |
| 외부 감각 | 초음파 | 거리 | 근접 감지 |
| 외부 감각 | GPS | 전역 위치 | 실외 내비게이션 |

### 2. 액추에이터(Actuators, 행동)

액추에이터는 에너지를 물리적 운동으로 변환한다.

| 유형 | 에너지원 | 특성 | 사용 사례 |
|------|-------------|-----------------|----------|
| DC 모터 | 전기 | 연속 회전, 정밀한 속도 제어 | 바퀴 구동 |
| 서보 모터(Servo Motor) | 전기 | 위치 제어, 높은 토크 대 중량비 | 로봇 관절 |
| 스텝 모터(Stepper Motor) | 전기 | 이산적 단계, 개방 루프 위치 결정 | 3D 프린터, CNC |
| 공압식(Pneumatic) | 압축 공기 | 빠름, 유연성, 이진 동작 | 그리퍼, 소프트 로봇 |
| 유압식(Hydraulic) | 가압 유체 | 매우 높은 힘, 정밀 | 대형 산업용, 굴삭기 |
| 형상 기억 합금(Shape Memory Alloy) | 열 | 소형, 저소음, 느림 | 마이크로 액추에이터 |

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

### 3. 컨트롤러(Controllers, 결정)

컨트롤러는 센서 데이터를 처리하고 액추에이터에 명령을 내리는 "두뇌"다. 현대 로봇 컨트롤러는 계층 구조로 동작한다:

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

각 계층은 서로 다른 주파수로 동작한다:
- 작업 계획: 수 초에서 수 분
- 동작 계획: 10-100 Hz
- 서보 제어: 1-10 kHz
- 하드웨어 인터페이스: 마이크로초

---

## 로봇공학의 응용

### 제조업

가장 오래되고 여전히 지배적인 응용 분야다. 현대 공장은 다음을 위해 로봇을 사용한다:
- **용접**: 일관된 품질, 위험 환경 보호
- **도장**: 균일한 코팅, VOC 노출 감소
- **조립**: 높은 정밀도, 반복 작업
- **품질 검사**: 머신 비전 기반 결함 탐지
- **자재 이송**: 팔레타이징(palletizing), 디팔레타이징, 빈 피킹(bin picking)

### 의료 로봇(Medical Robotics)

| 시스템 | 응용 | 주요 특징 |
|--------|-------------|-------------|
| 다빈치(da Vinci) | 최소 침습 수술 | 떨림 필터링, 3D 시각화 |
| 사이버나이프(CyberKnife) | 방사선 수술 | 밀리미터 미만의 표적 정확도 |
| ROBODOC | 정형외과 수술 | 정밀한 뼈 절삭 |
| 외골격(Exoskeleton) | 재활 | 뇌졸중 환자의 보조 운동 |

### 우주 로봇(Space Robotics)

- **캐나담2(Canadarm2)**: 국제 우주 정거장의 17m 로봇 팔
- **화성 탐사차**: 소저너(Sojourner), 스피릿(Spirit), 오퍼튜니티(Opportunity), 큐리오시티(Curiosity), 퍼서비어런스(Perseverance)
- **궤도 서비스**: 위성 수리 및 연료 보급 (Northrop Grumman MEV)

### 농업(Agriculture)

- **수확**: 컴퓨터 비전을 이용한 과일 수확 (Abundant Robotics, Agrobot)
- **정밀 살포**: 표적화된 제초제 살포, 화학물질 사용 90% 감소
- **자율 트랙터**: GPS 유도 경작 및 파종 (John Deere)
- **모니터링**: 드론 기반 작물 건강 평가 (다중 스펙트럼 이미징)

### 물류 및 창고(Logistics and Warehousing)

아마존의 창고에는 75만 대 이상의 로봇이 배치되어 있다 (2024년 기준):
- **AGV/AMR**: 선반 운반을 위한 자율 이동 로봇 (Kiva/Amazon Robotics)
- **분류**: 고속 소포 분류 시스템
- **라스트마일 배송**: 인도 배송 로봇(Starship), 배달 드론 (Wing, Amazon Prime Air)

---

## 로봇공학의 다학제적 성격

로봇공학은 단일 학문이 아니다 — 여러 학문의 통합이다:

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

이 과정은 이 학문들을 연결하는 **수학적 기초** — 변환(transformations), 기구학(kinematics), 동역학(dynamics), 계획(planning), 제어(control) — 에 집중한다. 이것이 컴퓨터 과학자가 물리적 운동에 대해 추론하고, 기계공학자가 지능적 행동을 프로그래밍할 수 있게 하는 도구다.

---

## 앞으로의 여정: 과정 개요

| 레슨 | 주제 | 핵심 질문 |
|--------|-------|-------------|
| L01 | 개요 및 분류 | 로봇이란 무엇이며 어떻게 구성되는가? |
| L02 | 강체 변환 | 3D에서 위치와 방향을 어떻게 기술하는가? |
| L03 | 순운동학 | 관절 각도가 주어졌을 때 끝단 효과기는 어디에 있는가? |
| L04 | 역운동학 | 원하는 자세가 주어졌을 때 관절 각도는 얼마인가? |
| L05 | 속도 기구학 (야코비안) | 관절 속도가 끝단 효과기 속도에 어떻게 매핑되는가? |
| L06 | 로봇 동역학 | 원하는 운동을 생성하는 데 필요한 힘/토크는 무엇인가? |
| L07 | 동작 계획 | 충돌 없는 경로를 어떻게 찾는가? |
| L08 | 궤적 계획 | 부드러운 실행을 위해 경로를 어떻게 시간 매개변수화하는가? |

각 레슨은 이전 레슨들을 기반으로 한다. 이 진행은 실제 로봇 컨트롤러의 작동 방식을 반영한다: 먼저 기하학을 기술(L02-L03)하고, 관절 각도를 구하고(L04), 속도 관계를 이해하고(L05), 물리를 고려하고(L06), 마지막으로 동작을 계획하고 실행한다(L07-L08).

---

## 요약

- 로봇공학은 단순한 산업용 매니퓰레이터(유니메이트, 1961)에서 비구조화 환경에서 작동하는 자율 시스템으로 발전했다
- 로봇은 구조에 따라 분류된다: **매니퓰레이터**(직렬, 병렬), **이동 로봇**(바퀴형, 다리형, 궤도형), **공중 로봇**(멀티로터, 고정익), **수중 로봇**(ROV, AUV)
- **자유도(DOF)**는 로봇의 능력을 결정하며, 3D에서 임의의 자세를 위한 최솟값은 6 DOF이다
- 모든 로봇은 세 가지 핵심 서브시스템을 갖는다: **센서**(지각), **액추에이터**(행동), **컨트롤러**(결정)
- 로봇공학은 본질적으로 다학제적이며, 이 과정은 그 분야를 통합하는 수학적 기초에 집중한다

---

## 연습 문제

### 연습 문제 1: 로봇 분류

다음 각 로봇에 대해 유형(매니퓰레이터/이동/공중/수중), 세부 유형, 예상 자유도를 분류하라:
1. 공장의 자동차 도장 로봇
2. 자율 주행 배송 트럭
3. 항공 사진용 헥사콥터(hexacopter)
4. 석유 파이프라인을 검사하는 수중 기기
5. 보조기 용도의 로봇 손

### 연습 문제 2: 그뤼블러 공식

평면 메커니즘이 5개의 링크(지면 포함), 5개의 회전 관절, 1개의 직동 관절을 가지고 있다. 그뤼블러 공식을 이용해 자유도를 계산하라. 이것은 메커니즘인가 구조물인가?

### 연습 문제 3: 모터 사이징

2-링크 평면 로봇 팔의 링크 길이는 $l_1 = 0.4$ m, $l_2 = 0.3$ m이다. 팔은 끝단 효과기에서 2 kg 부하를 지지해야 한다. 링크 질량은 무시한다:
1. 관절 1에서 최대 중력 토크를 계산하라 (두 링크가 수평일 때)
2. 관절 2에서 최대 중력 토크를 계산하라 (링크 2가 수평일 때)
3. 최대 관절 속도가 $\pi/2$ rad/s일 때, 각 관절에 필요한 모터 출력을 추정하라 (안전 계수 2 적용)

### 연습 문제 4: 작업 공간 추정

2-DOF 평면 로봇이 링크 길이 $l_1 = 1.0$ m, $l_2 = 0.5$ m를 가지며, 두 관절 모두 360도 회전이 가능하다.
1. 도달 가능 작업 공간을 스케치하라 (힌트: 환형 영역이다)
2. 내반경과 외반경은 얼마인가?
3. 관절 1이 $[-90^\circ, 90^\circ]$로, 관절 2가 $[-180^\circ, 180^\circ]$로 제한된다면 작업 공간은 어떻게 변하는가?

### 연습 문제 5: 조사

하나의 응용 분야(제조, 의료, 우주, 농업, 물류)를 선택하라. 그 분야에서 사용되는 특정 로봇을 조사하고 다음 내용을 포함한 한 페이지 요약문을 작성하라:
- 로봇 이름 및 제조사
- 분류 (유형, 세부 유형, 자유도)
- 주요 센서와 액추에이터
- 주요 응용 및 이 작업에 인간보다 로봇이 더 적합한 이유
- 현재의 한계

---

[다음: 강체 변환 →](02_Rigid_Body_Transformations.md)
