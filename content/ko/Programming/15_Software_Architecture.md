# 소프트웨어 아키텍처 기초

> **토픽**: Programming
> **레슨**: 15 of 16
> **선수 지식**: 객체 지향 프로그래밍, 디자인 패턴, 시스템 설계 기초
> **목표**: 고수준 아키텍처 패턴, 모노리스와 마이크로서비스 간의 트레이드오프, 유지보수 가능한 시스템 설계 원칙 이해

## 소개

**소프트웨어 아키텍처**는 시스템이 어떻게 구축되는지를 형성하는 고수준 구조적 결정의 집합입니다. 설계가 개별 구성 요소(클래스, 함수)에 초점을 맞추는 반면, 아키텍처는 **큰 그림**에 초점을 맞춥니다:

- 구성 요소가 어떻게 조직되는가
- 그들이 어떻게 통신하는가
- 어떤 경계가 존재하는가
- 시스템이 어떻게 확장되고, 발전하고, 실패를 처리하는가

좋은 아키텍처:
- 시스템을 **이해하기 쉽게** 만듦(인지 부하)
- 시스템을 **변경하기 쉽게** 만듦(적응성)
- 실패에 **탄력적**이게 만듦
- **비즈니스 목표**와 정렬됨(비용, 출시 시간, 확장성)

나쁜 아키텍처는 시간이 지남에 따라 개발을 늦추는 **기술 부채**를 만듭니다.

## 아키텍처 vs 설계

| 측면 | 아키텍처 | 설계 |
|--------|--------------|--------|
| **범위** | 시스템 수준 | 구성 요소 수준 |
| **결정** | 전략적, 변경하기 어려움 | 전술적, 변경하기 쉬움 |
| **예제** | 모노리스 vs 마이크로서비스, 데이터베이스 선택 | 클래스 구조, 알고리즘 선택 |
| **영향** | 장기적, 전체 팀에 영향 | 단기적, 모듈에 영향 |
| **시기** | 프로젝트 초기 | 프로젝트 전체에 걸쳐 |

**예제**:
- **아키텍처 결정**: "이벤트 기반 통신을 사용하는 마이크로서비스를 사용할 것입니다"
- **설계 결정**: "사용자 서비스는 저장소 패턴을 사용할 것입니다"

## 모노리식 아키텍처

**모노리스(monolith)**는 모든 애플리케이션 논리를 포함하는 단일 배포 가능한 단위입니다.

### 구조

```
monolith-app/
├── controllers/       # HTTP 핸들러
├── services/          # 비즈니스 논리
├── repositories/      # 데이터 접근
├── models/            # 도메인 엔티티
└── main.py            # 진입점
```

**배포**: 하나의 아티팩트(JAR, 실행 파일, Docker 이미지)

### 예제(Python Flask 모노리스)

```python
# app.py
from flask import Flask, jsonify, request
from database import db
from models import User, Order

app = Flask(__name__)

# 사용자 엔드포인트
@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user = User(name=data['name'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

# 주문 엔드포인트
@app.route('/orders', methods=['GET'])
def get_orders():
    orders = Order.query.all()
    return jsonify([o.to_dict() for o in orders])

@app.route('/orders', methods=['POST'])
def create_order():
    data = request.get_json()
    order = Order(user_id=data['user_id'], total=data['total'])
    db.session.add(order)
    db.session.commit()
    return jsonify(order.to_dict()), 201

if __name__ == '__main__':
    app.run()
```

모든 기능이 하나의 코드베이스에 있고 하나의 데이터베이스를 공유합니다.

### 장점

**1. 단순성**: 하나의 코드베이스, 하나의 배포, 하나의 서버
```bash
# 전체 앱 배포
docker build -t myapp .
docker run -p 8000:8000 myapp
```

**2. 쉬운 디버깅**: 모든 코드가 하나의 프로세스에, 단일 디버거 사용
**3. 일관된 데이터**: 기능 간 트랜잭션(ACID 보장)
**4. 성능**: 모듈 간 네트워크 호출 없음(메모리 내 함수 호출)
**5. 개발 속도**: 소규모 팀의 경우, 모노리스가 가장 빠르게 구축

### 단점

**1. 확장 제한**: 전체 앱을 확장해야 하며, 하나의 기능만 리소스가 필요해도
```
# 검색이 CPU 집약적이지만 결제는 아닌 경우에도
# 둘 다 함께 확장해야 함
docker run --replicas=10 myapp
```

**2. 기술 종속**: 전체 앱이 동일한 언어, 프레임워크, 데이터베이스 사용
```python
# 나머지가 Python인데 하나의 기능을 Go로 다시 작성하기 어려움
```

**3. 배포 결합**: 작은 변경에도 전체 앱 배포(위험하고, 느림)
```bash
# 하나의 엔드포인트에서 오타 수정 → 전체 앱 재배포
```

**4. 코드 결합**: 시간이 지남에 따라 기능이 긴밀하게 결합될 수 있음
```python
# 사용자 서비스가 주문 서비스를 직접 호출
def create_user(data):
    user = User(**data)
    db.session.add(user)
    order_service.create_welcome_order(user.id)  # 긴밀한 결합
    db.session.commit()
```

**5. 팀 조정**: 대규모 팀이 서로 방해(병합 충돌, 배포 충돌)

### 모노리스를 사용할 때

✅ **적합**:
- 중소 규모 팀(< 20명 개발자)
- 초기 단계 제품(MVP, 검증 단계)
- 복잡도가 낮은 간단한 도메인
- 제한된 트래픽을 가진 CRUD 애플리케이션

❌ **부적합**:
- 대규모 팀(> 50명 개발자)
- 높은 확장 시스템(하루 수백만 요청)
- 다양한 기술 요구(ML + 웹 + 실시간)

**기억하세요**: 대부분의 성공적인 회사는 모노리스로 시작했습니다. 확장을 위해 조기 최적화하지 마세요.

## 마이크로서비스 아키텍처

**마이크로서비스**는 애플리케이션을 비즈니스 기능을 중심으로 조직된 독립적이고 배포 가능한 서비스로 분할합니다.

### 구조

```
company-system/
├── user-service/          # 사용자 관리
│   ├── src/
│   ├── database/
│   └── Dockerfile
├── order-service/         # 주문 관리
│   ├── src/
│   ├── database/
│   └── Dockerfile
├── payment-service/       # 결제 처리
│   ├── src/
│   ├── database/
│   └── Dockerfile
└── api-gateway/           # 요청 라우팅
    ├── src/
    └── Dockerfile
```

각 서비스:
- **자체 데이터베이스** 보유
- **독립적으로 배포** 가능
- **네트워크**를 통해 통신(HTTP, gRPC, 메시지 큐)
- **단일 팀**이 소유

### 예제(Node.js 마이크로서비스)

**사용자 서비스**:
```javascript
// user-service/server.js
const express = require('express');
const app = express();

app.get('/users/:id', (req, res) => {
  const user = db.findUser(req.params.id);
  res.json(user);
});

app.post('/users', (req, res) => {
  const user = db.createUser(req.body);
  res.status(201).json(user);
});

app.listen(3001);
```

**주문 서비스**:
```javascript
// order-service/server.js
const express = require('express');
const axios = require('axios');
const app = express();

app.post('/orders', async (req, res) => {
  // 사용자 서비스를 호출하여 사용자 검증
  const user = await axios.get(`http://user-service:3001/users/${req.body.userId}`);

  if (!user.data) {
    return res.status(404).send('User not found');
  }

  const order = db.createOrder(req.body);
  res.status(201).json(order);
});

app.listen(3002);
```

**API 게이트웨이**:
```javascript
// api-gateway/server.js
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

app.use('/users', createProxyMiddleware({ target: 'http://user-service:3001' }));
app.use('/orders', createProxyMiddleware({ target: 'http://order-service:3002' }));
app.use('/payments', createProxyMiddleware({ target: 'http://payment-service:3003' }));

app.listen(8080);
```

### 장점

**1. 독립적 확장**: 확장이 필요한 것만 확장
```bash
# 주문 서비스 10배, 사용자 서비스 2배 확장
kubectl scale deployment order-service --replicas=10
kubectl scale deployment user-service --replicas=2
```

**2. 기술 다양성**: 각 서비스가 최적의 기술 사용 가능
```
user-service:   Node.js + MongoDB
order-service:  Python + PostgreSQL
search-service: Go + Elasticsearch
```

**3. 팀 자율성**: 팀이 서비스를 종단 간 소유(조정 감소)
**4. 장애 격리**: 하나의 서비스 실패가 전체 시스템을 망가뜨리지 않음
```javascript
// 결제 서비스가 다운되면 주문 서비스는 대체
try {
  await paymentService.charge(order);
} catch (err) {
  await queue.enqueue('retry-payment', order);
  return res.status(202).send('Payment pending');
}
```

**5. 더 빠른 배포**: 다른 서비스를 건드리지 않고 하나의 서비스 배포

### 단점

**1. 복잡성**: 네트워크 호출, 서비스 발견, 분산 추적
```python
# 모노리스에서 간단
user = get_user(user_id)

# 마이크로서비스에서 복잡
response = requests.get(f'http://user-service/users/{user_id}')
if response.status_code != 200:
    # 네트워크 오류, 타임아웃, 서비스 다운 처리...
user = response.json()
```

**2. 분산 트랜잭션**: 서비스 간 ACID 없음
```javascript
// 모노리스: 원자적 트랜잭션
db.transaction(() => {
  createOrder(order);
  decrementInventory(order.items);
  chargePayment(order.total);
});

// 마이크로서비스: 최종 일관성
await orderService.createOrder(order);
await inventoryService.decrementStock(order.items);  // 실패할 수 있음
await paymentService.charge(order.total);  // 실패할 수 있음
// 사가 패턴이나 보상 논리 필요
```

**3. 네트워크 지연**: 모든 서비스 호출이 10-100ms 추가
**4. 운영 오버헤드**: 더 많은 배포, 모니터링, 로그 관리
**5. 데이터 중복**: 서비스가 다른 서비스의 데이터를 캐시할 수 있음

### 마이크로서비스를 사용하지 말아야 할 때

❌ **마이크로서비스를 피해야 할 경우**:
- MVP를 구축 중(입증되지 않은 제품)
- 팀이 작음(< 10명 개발자)
- DevOps 전문 지식이 없음(모니터링, 오케스트레이션)
- 도메인 경계가 불명확(서비스가 자주 변경될 것)

**유명한 인용**: "마이크로서비스를 사용하려면 이 정도는 되어야 합니다" — Martin Fowler

**모노리스로 시작하세요**. 나중에 고통 지점이 나타나면 마이크로서비스 추출(팀 확장, 성능 병목).

## 계층화(N-Tier) 아키텍처

**계층화 아키텍처**는 코드를 명확한 책임을 가진 수평 레이어로 조직합니다.

### 클래식 3-Tier 아키텍처

```
┌─────────────────────────┐
│   프레젠테이션 계층      │  (UI, 컨트롤러, API 엔드포인트)
├─────────────────────────┤
│   비즈니스 로직 계층     │  (서비스, 도메인 논리)
├─────────────────────────┤
│   데이터 접근 계층       │  (저장소, ORM)
├─────────────────────────┤
│   데이터베이스           │  (PostgreSQL, MongoDB)
└─────────────────────────┘
```

### 예제(Java Spring Boot)

```java
// 프레젠테이션 계층: 컨트롤러
@RestController
@RequestMapping("/users")
public class UserController {
    @Autowired
    private UserService userService;

    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        return ResponseEntity.ok(user);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody UserDTO dto) {
        User user = userService.createUser(dto);
        return ResponseEntity.status(201).body(user);
    }
}

// 비즈니스 로직 계층: 서비스
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private EmailService emailService;

    public User createUser(UserDTO dto) {
        // 비즈니스 논리
        if (userRepository.existsByEmail(dto.getEmail())) {
            throw new UserAlreadyExistsException();
        }

        User user = new User(dto.getName(), dto.getEmail());
        userRepository.save(user);

        emailService.sendWelcomeEmail(user.getEmail());
        return user;
    }

    public User findById(Long id) {
        return userRepository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }
}

// 데이터 접근 계층: 저장소
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    boolean existsByEmail(String email);
}
```

### 엄격한 vs 느슨한 계층화

**엄격한 계층화**: 각 계층은 바로 아래 계층만 호출
```
프레젠테이션 → 비즈니스 로직 → 데이터 접근
```

**느슨한 계층화**: 계층이 수준을 건너뛸 수 있음
```
프레젠테이션 → 데이터 접근 (간단한 읽기의 경우, 비즈니스 로직 우회)
```

**트레이드오프**: 엄격한 계층화는 분리를 강제하지만 보일러플레이트 추가. 느슨한 계층화는 실용적이지만 논리가 프레젠테이션으로 누출될 수 있음.

### 장점

- **관심사 분리**: 명확한 책임
- **테스트 가능성**: 계층을 독립적으로 모킹
- **교체 가능성**: 데이터 계층 교체(SQL → NoSQL) 비즈니스 논리를 건드리지 않고

### 단점

- **성능**: 추가 계층이 오버헤드 추가
- **경직성**: 부자연스러울 때도 논리를 계층에 강제

## 육각형 아키텍처(포트와 어댑터)

**육각형 아키텍처**(Alistair Cockburn, 2005)는 **도메인**을 중심에 두고, **포트**(인터페이스)와 **어댑터**(구현)로 둘러쌉니다.

### 구조

```
       ┌───────────────────┐
       │    어댑터         │
       │  (HTTP, CLI, DB)  │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │      포트         │
       │   (인터페이스)    │
       └─────────┬─────────┘
                 │
       ┌─────────▼─────────┐
       │  도메인/코어       │
       │ (비즈니스 로직)   │
       └───────────────────┘
```

**핵심 아이디어**: 도메인 논리는 인프라(데이터베이스, 프레임워크)에 의존하지 않음. 인프라가 도메인에 의존.

### 예제(Python)

**도메인(핵심 비즈니스 논리)**:
```python
# domain/user.py
class User:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

    def change_email(self, new_email):
        if '@' not in new_email:
            raise ValueError('Invalid email')
        self.email = new_email
```

**포트(인터페이스)**:
```python
# ports/user_repository.py
from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id):
        pass

    @abstractmethod
    def save(self, user):
        pass
```

**어댑터(구현)**:
```python
# adapters/postgres_user_repository.py
from ports.user_repository import UserRepository
from domain.user import User

class PostgresUserRepository(UserRepository):
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_id(self, user_id):
        row = self.db.execute('SELECT * FROM users WHERE id = ?', user_id)
        return User(row['id'], row['name'], row['email'])

    def save(self, user):
        self.db.execute('UPDATE users SET name = ?, email = ? WHERE id = ?',
                        user.name, user.email, user.id)
```

**유스케이스(애플리케이션 서비스)**:
```python
# use_cases/change_email.py
class ChangeEmailUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def execute(self, user_id, new_email):
        user = self.user_repository.find_by_id(user_id)
        user.change_email(new_email)
        self.user_repository.save(user)
```

**어댑터(HTTP)**:
```python
# adapters/flask_app.py
from flask import Flask, request, jsonify
from use_cases.change_email import ChangeEmailUseCase
from adapters.postgres_user_repository import PostgresUserRepository

app = Flask(__name__)
user_repo = PostgresUserRepository(db_connection)

@app.route('/users/<int:user_id>/email', methods=['PUT'])
def change_email(user_id):
    use_case = ChangeEmailUseCase(user_repo)
    use_case.execute(user_id, request.json['email'])
    return jsonify({'status': 'ok'})
```

### 의존성 규칙

**의존성은 안쪽을 가리킴**: 외부 계층이 내부 계층에 의존, 결코 반대가 아님.

```
어댑터(HTTP, DB) → 포트 → 도메인
```

**장점**: 도메인 논리는 인프라 없이 **테스트 가능**:
```python
# 데이터베이스 없이 테스트
class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.users = {}

    def find_by_id(self, user_id):
        return self.users[user_id]

    def save(self, user):
        self.users[user.id] = user

def test_change_email():
    repo = InMemoryUserRepository()
    repo.save(User(1, 'John', 'john@example.com'))

    use_case = ChangeEmailUseCase(repo)
    use_case.execute(1, 'newemail@example.com')

    user = repo.find_by_id(1)
    assert user.email == 'newemail@example.com'
```

## 클린 아키텍처

**클린 아키텍처**(Robert C. Martin, 2012)는 육각형 아키텍처를 명시적 계층으로 일반화합니다:

```
┌──────────────────────────────────────┐
│  프레임워크 & 드라이버 (DB, Web, UI) │  (가장 바깥)
├──────────────────────────────────────┤
│  인터페이스 어댑터 (컨트롤러)        │
├──────────────────────────────────────┤
│  유스케이스 (애플리케이션 로직)      │
├──────────────────────────────────────┤
│  엔티티 (비즈니스 로직)              │  (가장 안쪽)
└──────────────────────────────────────┘
```

### 계층

1. **엔티티**: 핵심 비즈니스 규칙(User, Order)
2. **유스케이스**: 애플리케이션별 논리(CreateUser, PlaceOrder)
3. **인터페이스 어댑터**: 유스케이스를 위한 데이터 변환(컨트롤러, 프레젠터)
4. **프레임워크 & 드라이버**: 외부 도구(웹 프레임워크, 데이터베이스, UI)

### 의존성 규칙

**의존성은 안쪽만을 가리킴**. 내부 계층은 외부 계층에 대해 아무것도 모름.

```cpp
// C++ 예제: 엔티티(가장 안쪽)
// entities/Order.h
class Order {
private:
    int id_;
    double total_;
    std::vector<Item> items_;
public:
    Order(int id, double total) : id_(id), total_(total) {}

    void addItem(const Item& item) {
        items_.push_back(item);
        total_ += item.price;
    }

    double getTotal() const { return total_; }
};

// 유스케이스
// use_cases/PlaceOrder.h
class PlaceOrder {
private:
    OrderRepository* repository_;  // 인터페이스(포트)
public:
    PlaceOrder(OrderRepository* repo) : repository_(repo) {}

    void execute(int userId, const std::vector<Item>& items) {
        Order order(generateId(), 0);
        for (const auto& item : items) {
            order.addItem(item);
        }
        repository_->save(order);
    }
};

// 인터페이스 어댑터
// adapters/HttpController.cpp
void OrderController::placeOrder(HttpRequest& req, HttpResponse& res) {
    auto items = parseItems(req.body);
    PlaceOrder useCase(orderRepository);
    useCase.execute(req.userId, items);
    res.send(201, "Order placed");
}
```

### 독립성

클린 아키텍처는 다음으로부터의 독립성을 달성합니다:
- **프레임워크**: Rails, Spring 등에 종속되지 않음
- **UI**: 논리 변경 없이 웹 UI를 CLI로 교체
- **데이터베이스**: SQL을 NoSQL로 교체
- **외부 기관**: 서드파티 API는 인터페이스 뒤에

## 이벤트 기반 아키텍처

**이벤트 기반 아키텍처(EDA)**는 **이벤트**(상태 변경)를 사용하여 구성 요소 간 통신을 트리거합니다.

### 구성 요소

- **이벤트**: 발생한 일(`UserRegistered`, `OrderPlaced`)
- **생산자**: 이벤트 발행
- **이벤트 버스**: 이벤트를 소비자로 라우팅
- **소비자**: 이벤트에 반응

### 예제(이벤트를 사용한 마이크로서비스)

```javascript
// RabbitMQ를 사용한 Node.js
const amqp = require('amqplib');

// 생산자: 사용자 서비스
class UserService {
    async registerUser(data) {
        const user = await db.createUser(data);

        // 이벤트 발행
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();
        channel.publish('events', 'user.registered', Buffer.from(JSON.stringify({
            userId: user.id,
            email: user.email,
        })));

        return user;
    }
}

// 소비자: 이메일 서비스
class EmailService {
    async start() {
        const connection = await amqp.connect('amqp://localhost');
        const channel = await connection.createChannel();
        channel.consume('user.registered', (msg) => {
            const event = JSON.parse(msg.content.toString());
            this.sendWelcomeEmail(event.email);
            channel.ack(msg);
        });
    }

    sendWelcomeEmail(email) {
        console.log(`Sending welcome email to ${email}`);
    }
}
```

**분리**: 사용자 서비스는 이메일 서비스에 대해 알지 못함. 이벤트를 통해 통신.

### 이벤트 소싱

**이벤트 소싱(Event Sourcing)**은 현재 상태가 아닌 **이벤트 시퀀스로 상태**를 저장합니다.

```python
# 전통적: 현재 상태 저장
user = { 'id': 1, 'email': 'new@example.com', 'name': 'John' }
db.save(user)

# 이벤트 소싱: 이벤트 저장
events = [
    {'type': 'UserCreated', 'data': {'name': 'John', 'email': 'old@example.com'}},
    {'type': 'EmailChanged', 'data': {'email': 'new@example.com'}},
]
db.save_events(events)

# 이벤트 재생으로 상태 재구축
def get_user_state(user_id):
    events = db.get_events(user_id)
    user = {}
    for event in events:
        if event['type'] == 'UserCreated':
            user = event['data']
        elif event['type'] == 'EmailChanged':
            user['email'] = event['data']['email']
    return user
```

**장점**: 완전한 감사 추적, 시간 여행, 디버깅을 위한 재생
**단점**: 복잡성, 최종 일관성

### CQRS(명령 쿼리 책임 분리)

**CQRS**는 **쓰기**(명령)와 **읽기**(쿼리)를 분리합니다.

```
명령(쓰기) → 쓰기 모델 → 이벤트 저장소
                                      ↓
                                  이벤트 버스
                                      ↓
                                  읽기 모델(쿼리에 최적화)
```

**예제**:
- **쓰기 모델**: 일관성을 위한 정규화된 SQL
- **읽기 모델**: 빠른 검색을 위한 비정규화된 Elasticsearch

## MVC, MVP, MVVM

UI 코드를 조직하기 위한 패턴.

### MVC(Model-View-Controller)

```
사용자 입력 → 컨트롤러 → 모델
                 ↓          ↓
              뷰 ←────────┘
```

**모델**: 비즈니스 논리, 데이터
**뷰**: UI 렌더링
**컨트롤러**: 입력 처리, 모델 업데이트

```python
# Flask MVC
@app.route('/users/<int:user_id>')
def show_user(user_id):  # 컨트롤러
    user = User.query.get(user_id)  # 모델
    return render_template('user.html', user=user)  # 뷰
```

### MVP(Model-View-Presenter)

```
사용자 입력 → 프레젠터 ↔ 모델
                 ↕
              뷰
```

뷰는 수동적; 프레젠터가 모든 논리 처리.

### MVVM(Model-View-ViewModel)

```
사용자 입력 → 뷰모델 ↔ 모델
               ↕ (데이터 바인딩)
            뷰
```

**뷰모델**은 뷰를 위한 데이터와 명령 노출. 데이터 바인딩을 가진 프레임워크(React, Vue, Angular)에서 인기.

## 아키텍처 결정 기록(ADR)

**ADR**은 중요한 아키텍처 결정을 문서화합니다.

### 템플릿

```markdown
# ADR-001: 마이크로서비스 아키텍처 사용

## 상태
승인됨

## 컨텍스트
사용자 기반이 증가함에 따라 독립적으로 확장해야 합니다.
현재 모노리스는 작은 변경에도 전체 앱을 배포하여 다운타임 발생.
팀이 30명의 개발자로 성장하여 병합 충돌 발생.

## 결정
다음 서비스로 마이크로서비스 아키텍처를 채택할 것입니다:
- 사용자 서비스
- 주문 서비스
- 결제 서비스
- 알림 서비스

각 서비스는 자체 데이터베이스를 가집니다.
REST API와 RabbitMQ 이벤트를 통한 통신.

## 결과
### 긍정적
- 독립적 확장
- 팀 자율성
- 더 빠른 배포

### 부정적
- 복잡성 증가
- 서비스 메시, 분산 추적 필요
- 최종 일관성 과제

## 고려된 대안
- **모듈화된 모노리스**: 모노리스 유지하되 모듈 경계 강제
  - 거부: 확장이나 배포 문제 해결 못함
```

## 품질 속성

아키텍처는 **품질 속성**(비기능적 요구사항)을 최적화합니다:

- **성능**: 응답 시간, 처리량
- **확장성**: 증가하는 부하 처리
- **가용성**: 가동 시간 백분율(99.9% = 연간 8.76시간 다운타임)
- **유지보수성**: 코드 변경 용이성
- **보안**: 공격에 대한 저항
- **테스트 가능성**: 테스트 용이성

**트레이드오프**: 모든 속성을 최적화할 수 없습니다. 비즈니스 요구에 따라 우선순위 지정.

## CAP 정리

분산 시스템에서 **최대 두 개**만 가질 수 있습니다:

- **일관성(Consistency)**: 모든 노드가 동시에 동일한 데이터를 봄
- **가용성(Availability)**: 모든 요청이 응답을 받음(성공 또는 실패)
- **분할 내성(Partition Tolerance)**: 네트워크 분할에도 시스템 계속

```
     일관성
        /   \
       /     \
   CA /       \ CP
     /         \
    /           \
가용성—분할 내성
         AP
```

**예제**:
- **CA**: 전통적 RDBMS(PostgreSQL) — 일관되고 가용하지만, 분할 내성 없음
- **CP**: MongoDB, HBase — 일관되고 분할 내성 있지만, 분할 중 사용 불가능할 수 있음
- **AP**: Cassandra, DynamoDB — 가용하고 분할 내성 있지만, 최종 일관성

**실제로**: 네트워크 분할은 발생하므로, CP 또는 AP 선택.

## 연습 문제

### 연습 문제 1: 시나리오에 따른 아키텍처 선택

각 시나리오에 대해 아키텍처(모노리스, 마이크로서비스, 서버리스)를 추천하고 정당화하세요:

1. 작업 관리 앱을 위한 MVP를 구축하는 스타트업(3명 개발자, 6개월 타임라인)
2. 100명의 개발자, 하루 100만 사용자를 가진 전자상거래 회사, 결제를 독립적으로 확장해야 함
3. 데이터 처리를 위한 내부 도구(월별 배치 작업 실행, 2명 개발자)

### 연습 문제 2: 계층화 아키텍처 설계

다음을 가진 블로그 플랫폼을 위한 3-tier 계층화 아키텍처를 설계하세요:
- 사용자 인증
- 게시물 생성, 편집, 삭제
- 댓글
- 검색

계층과 각 계층에 속하는 구성 요소를 나열하세요.

### 연습 문제 3: 육각형 아키텍처로 리팩토링

이 긴밀하게 결합된 코드를 육각형 아키텍처로 리팩토링하세요:

```python
from flask import Flask, request
import psycopg2

app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    conn = psycopg2.connect("dbname=mydb user=postgres")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)",
                   (request.json['name'], request.json['email']))
    conn.commit()
    return {'status': 'ok'}
```

정의하세요:
1. 도메인 엔티티
2. 포트(인터페이스)
3. 어댑터(구현)
4. 유스케이스

### 연습 문제 4: 아키텍처 결정 문서화

다음 결정 중 하나에 대한 ADR을 작성하세요:
1. 사용자 데이터에 MongoDB 대신 PostgreSQL 선택
2. REST 대신 GraphQL 채택
3. 서버 사이드 렌더링 vs 클라이언트 사이드 렌더링 사용

### 연습 문제 5: CAP 트레이드오프 분석

회사가 실시간 협업 문서 편집기(Google Docs와 같은)를 구축하고 있습니다. 사용자는 서로의 편집을 즉시 보아야 하고, 일부 서버에 접근할 수 없어도 시스템이 작동해야 합니다.

CAP 중 무엇을 우선시하시겠습니까? CP 또는 AP를 선택하시겠습니까? 논리와 잠재적 트레이드오프를 설명하세요.

## 요약

소프트웨어 아키텍처는 비즈니스 목표를 달성하기 위한 **의도적인 트레이드오프**를 만드는 것입니다:

- **모노리스**: 단순, 빠른 구축, 수직 확장 — 여기서 시작
- **마이크로서비스**: 복잡, 수평 확장, 팀 자율성 — 필요할 때 진화
- **계층화 아키텍처**: 관심사 분리, 테스트 가능성
- **육각형/클린 아키텍처**: 도메인 독립성, 테스트 가능성, 유연성
- **이벤트 기반**: 분리, 확장성, 최종 일관성
- **MVC/MVP/MVVM**: UI 조직 패턴
- **ADR**: 결정과 트레이드오프 문서화
- **품질 속성**: 중요한 것 우선순위(성능, 확장성, 유지보수성)
- **CAP 정리**: 분산 시스템에서 모든 것을 가질 수 없음

**황금 규칙**: 요구사항을 충족하는 가장 간단한 아키텍처를 선택하세요. 나중에 언제든 진화시킬 수 있습니다.

## 내비게이션

[← 이전: 버전 관리 워크플로우](14_Version_Control_Workflows.md) | [다음: 개발자 관행 →](16_Developer_Practices.md)
