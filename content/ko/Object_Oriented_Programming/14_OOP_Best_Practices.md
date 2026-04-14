# 레슨 14: OOP 모범 사례

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 일반적인 OOP 안티패턴을 식별하고 피할 수 있다
2. 의도를 전달하는 명명 규칙을 적용할 수 있다
3. 적절한 세분도의 클래스를 설계할 수 있다 (너무 크지도 작지도 않게)
4. 절차적 코드를 깔끔한 OOP 설계로 리팩토링할 수 있다
5. 의존성 주입을 사용하여 테스트 가능한 클래스를 작성할 수 있다
6. OOP와 실용주의의 균형을 맞출 수 있다 — OOP를 사용하지 않을 때를 알기
7. 실전 OOP 설계 체크리스트를 코드에 적용할 수 있다

## 피해야 할 안티패턴

### 안티패턴 1: 갓 클래스

모든 것을 하는 클래스 — SRP 위반:

```python
# 나쁜 예: 갓 클래스
class Application:
    def register_user(self, name, email): ...
    def authenticate_user(self, email, password): ...
    def create_order(self, user_id, products): ...
    def process_payment(self, order_id, card): ...
    def send_email(self, to, subject, body): ...
    def generate_report(self, report_type): ...
    # 50개 이상의 메서드...

# 좋은 예: 집중된 클래스로 분리
class UserService:
    def register(self, name, email): ...
    def authenticate(self, email, password): ...

class OrderService:
    def create(self, user_id, products): ...

class PaymentService:
    def process(self, order_id, card): ...
```

### 안티패턴 2: 기능 탐욕 (Feature Envy)

자신의 데이터보다 다른 클래스의 데이터를 더 많이 사용하는 메서드:

```python
# 나쁜 예: Order가 Customer의 데이터를 과도하게 접근
class Order:
    def calculate_discount(self, customer):
        if customer.membership_level == "gold":
            if customer.years_active > 5:
                return 0.20
            return 0.10
        return 0.0

# 좋은 예: 데이터가 있는 곳으로 메서드 이동
class Customer:
    def get_discount_rate(self):
        """고객이 자신의 할인 로직을 안다."""
        if self.membership_level == "gold":
            if self.years_active > 5:
                return 0.20
            return 0.10
        return 0.0
```

### 안티패턴 3: 요요 상속

위아래로 왔다갔다해야 하는 깊은 상속 계층:

```
좋지 않은 예:
  Animal → Vertebrate → Mammal → Carnivore → Canidae → Dog → GermanShepherd
  (Dog을 이해하려면 6단계의 클래스를 읽어야 함!)

더 나은 예:
  Animal → Dog (최대 2단계 + 합성으로 세부사항 처리)
```

### 안티패턴 4: 빈약한 도메인 모델

동작이 없는 데이터 가방 클래스:

```python
# 나쁜 예: 모든 로직이 클래스 외부에 있음
class User:
    def __init__(self):
        self.name = ""
        self.email = ""

def validate_user(user): ...  # 로직이 외부에 존재

# 좋은 예: 풍부한 도메인 모델 — 동작이 데이터와 함께
class User:
    def __init__(self, name, email):
        self._validate(name, email)
        self.name = name
        self.email = email

    def _validate(self, name, email):
        if not name:
            raise ValueError("이름이 필요합니다")
        if "@" not in email:
            raise ValueError("잘못된 이메일입니다")
```

## 명명 규칙

### 클래스 이름

```python
# 클래스: PascalCase, 엔티티를 설명하는 명사
class UserAccount: ...
class PaymentProcessor: ...
class DatabaseConnection: ...

# 나쁜 이름
class Manager: ...    # 너무 모호함 — 무엇의 매니저?
class Utils: ...      # 잡동사니, 아마도 SRP 위반
class Data: ...       # 모든 것이 데이터임
```

### 메서드 이름

```python
class Order:
    # 동작: 동사 구문
    def calculate_total(self): ...
    def apply_discount(self, rate): ...

    # 쿼리: is_/has_/can_ (불리언)
    def is_valid(self): ...
    def has_items(self): ...
    def can_cancel(self): ...

    # 프로퍼티: 명사 구문
    @property
    def total(self): ...
```

## 클래스 설계 가이드라인

### 가이드라인 1: 클래스를 집중적으로 유지

```
┌─────────────────────────────────────────────────┐
│  경험 법칙:                                     │
│                                                 │
│  클래스가 하는 일을 "그리고" 또는 "또는"을      │
│  사용하지 않고 한 문장으로 설명할 수 없다면     │
│  아마도 너무 많은 일을 합니다.                   │
│                                                 │
│  좋은 예: "UserRepository는 사용자 데이터를     │
│           영속화합니다"                          │
│  나쁜 예: "AppManager는 사용자 그리고 주문을    │
│           처리하고 이메일 또는 알림을 보냅니다"  │
└─────────────────────────────────────────────────┘
```

### 가이드라인 2: 테스트 가능성을 위한 설계

```python
# 나쁜 예: 자체 의존성을 생성 — 테스트하기 어려움
class OrderProcessor:
    def __init__(self):
        self.db = PostgresDatabase()       # 하드코딩됨
        self.emailer = SMTPEmailService()   # 하드코딩됨

# 좋은 예: 의존성 주입 — 목(mock)으로 쉽게 테스트
class OrderProcessor:
    def __init__(self, db, emailer):
        self.db = db             # 주입됨
        self.emailer = emailer   # 주입됨

# 테스트에서:
class FakeDB:
    def __init__(self):
        self.saved = []
    def save(self, item):
        self.saved.append(item)

processor = OrderProcessor(FakeDB(), FakeEmailer())
```

### 가이드라인 3: 기본적으로 합성 사용

```python
class UserService:
    """집중된 컴포넌트로 합성됨."""

    def __init__(self, repo, validator, logger):
        self._repo = repo
        self._validator = validator
        self._logger = logger

    def create_user(self, data):
        self._validator.validate(data)
        user = User(**data)
        self._repo.save(user)
        self._logger.log(f"사용자 생성됨: {user.name}")
        return user
```

## 설계 체크리스트

```
┌─────────────────────────────────────────────────────────────┐
│  OOP 설계 체크리스트                                        │
├─────────────────────────────────────────────────────────────┤
│  [ ] 각 클래스가 단일하고 명확한 책임을 가지는가?           │
│  [ ] 클래스 이름이 목적을 설명하는 명사인가?                │
│  [ ] 메서드가 동작을 설명하는 동사인가?                     │
│  [ ] 공개 인터페이스가 최소하고 집중적인가?                 │
│  [ ] 의존성이 하드코딩이 아닌 주입되는가?                   │
│  [ ] 상속이 진정한 "is-a"에만 사용되는가?                   │
│  [ ] 계층이 얕은가 (최대 2-3 수준)?                        │
│  [ ] 서브클래스가 리스코프 치환을 만족하는가?               │
│  [ ] __init__에서 가변 기본값이 사용되지 않는가?            │
│  [ ] 내부 상태가 _ 또는 __로 보호되는가?                   │
│  [ ] 불변성이 @property를 통해 강제되는가?                  │
│  [ ] 실제 DB/네트워크 없이 테스트 가능한가?                 │
│  [ ] 여기서 함수가 클래스보다 더 간단하지 않은가?           │
└─────────────────────────────────────────────────────────────┘
```

## OOP를 사용하지 않을 때

```python
# 상태 없는 변환에는 함수가 충분합니다
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32

# 이것을 클래스로 감싸지 마세요:
class TemperatureConverter:  # 불필요!
    def convert(self, celsius):
        return celsius * 9 / 5 + 32
```

### "이것이 클래스인가?" 테스트

```
┌─────────────────────────────────────────────────┐
│  클래스를 사용할 때:                            │
│  - 함께 속하는 데이터 + 동작이 있을 때          │
│  - 공유 로직을 가진 여러 인스턴스가 필요할 때   │
│  - 메서드 호출 간 상태를 유지해야 할 때         │
│  - 상속이나 다형성이 필요할 때                  │
│                                                 │
│  함수를 사용할 때:                              │
│  - 연산이 상태 없을 때                          │
│  - 입력 -> 출력, 부수 상태 없음                 │
│  - 일회성 변환일 때                             │
│  - 클래스에 공개 메서드가 하나뿐일 때           │
└─────────────────────────────────────────────────┘
```

## 요약

- 안티패턴을 피하세요: 갓 클래스, 기능 탐욕, 요요 상속, 조기 추상화, 빈약한 도메인 모델
- 명명 규칙을 따르세요: 클래스는 PascalCase, 메서드는 동사 구문, `is_`/`has_`는 불리언
- 작은 공개 인터페이스로 클래스를 집중적으로 유지하세요
- 의존성 주입을 사용하여 테스트 가능성을 위해 설계하세요
- 기본적으로 상속보다 합성을 선호하세요
- 상태가 있는 동작 풍부한 엔티티에 OOP를, 상태 없는 변환에는 함수를 사용하세요
- 클래스 설계를 확정하기 전에 설계 체크리스트를 적용하세요

## 과정 마무리

객체 지향 프로그래밍 과정을 완료하신 것을 축하합니다! 이제 기초 개념부터 고급 설계 원칙까지 OOP에 대한 깊은 이해를 갖추셨습니다. 핵심 요약:

1. **핵심 개념**: 클래스, 객체, 생성자, 객체 생명주기
2. **네 가지 기둥**: 캡슐화, 상속, 다형성, 추상화
3. **설계 원칙**: SOLID, 상속보다 합성, 디자인 패턴
4. **Python 특화**: 매직 메서드, 데이터클래스, 프로토콜, 현대적 관용구
5. **실용주의**: OOP를 사용할 때와 더 간단한 접근법이 더 나을 때를 아는 것

이 개념들을 결합하는 프로젝트를 구축하며 계속 연습하세요. OOP를 내면화하는 가장 좋은 방법은 실제 시스템을 설계하고 구현하고 리팩토링하는 것입니다.
