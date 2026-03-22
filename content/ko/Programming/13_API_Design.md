# API 설계 원칙

> **토픽**: Programming
> **레슨**: 13 of 16
> **선수 지식**: 객체 지향 프로그래밍, HTTP 기초, JSON/XML 형식
> **목표**: 라이브러리, 모듈, 웹 서비스를 위한 명확하고 일관된 유지보수 가능한 API 설계 방법 학습

## 소개

**API(Application Programming Interface)**는 소프트웨어 구성 요소가 통신하는 방법을 정의하는 계약입니다. 라이브러리 함수, REST 엔드포인트 또는 시스템 인터페이스를 설계하든, 좋은 API 설계는 소프트웨어를 더 쉽게 사용하고, 유지보수하고, 발전시킬 수 있게 만듭니다.

잘못된 API 설계는 다음을 초래합니다:
- 인터페이스를 잘못 사용하는 혼란스러운 사용자
- 작은 변경에도 깨지는 취약한 코드
- 불명확한 문서로 인한 지원 부담
- 소비자의 통합 어려움

훌륭한 API 설계는 보이지 않습니다—사용자는 마찰 없이 목표를 달성합니다.

## API란 무엇인가?

API는 구현 세부 사항을 숨기면서 기능을 노출하는 **추상화(abstraction)**입니다:

```python
# 나쁨: 구현 세부사항 노출
user_data = database.execute_raw_sql("SELECT * FROM users WHERE id = ?", user_id)

# 좋음: API를 통한 추상화
user = user_service.get_user_by_id(user_id)
```

### API의 유형

1. **라이브러리/모듈 API**: 코드 라이브러리의 함수와 클래스
2. **웹 API**: HTTP 기반 서비스(REST, GraphQL, gRPC)
3. **운영체제 API**: 시스템 콜, 파일 I/O, 프로세스 관리
4. **하드웨어 API**: 디바이스 드라이버, 펌웨어 인터페이스

이 레슨은 가장 일반적인 설계 과제를 대표하는 웹 API와 라이브러리 API에 주로 초점을 맞춥니다.

## REST API 설계

**REST(Representational State Transfer)**는 HTTP 메서드와 URI를 사용하여 리소스를 표현하는 웹 API의 아키텍처 스타일입니다.

### 핵심 원칙

#### 1. 리소스와 URI

리소스는 **명사(noun)**이지, 동사(verb)가 아닙니다. URI는 리소스를 식별하고, HTTP 메서드는 작업을 나타냅니다.

**좋은 URI 설계**:
```
GET    /users              # 사용자 목록
GET    /users/123          # 사용자 123 조회
POST   /users              # 사용자 생성
PUT    /users/123          # 사용자 123 업데이트
DELETE /users/123          # 사용자 123 삭제
GET    /users/123/orders   # 사용자 123의 주문 조회
```

**나쁜 URI 설계**:
```
GET    /getUsers           # URI에 동사
POST   /createUser         # URI에 동사
GET    /users/delete/123   # URI 경로에 액션
GET    /user_orders?id=123 # 일관성 없는 네이밍
```

**네이밍 규칙**:
- 컬렉션에는 **복수 명사** 사용(`/users`, `/user` 아님)
- **소문자** 사용, 하이픈으로 구분(`/order-items`, `/OrderItems` 아님)
- 경로 세그먼트로 **계층 구조** 표현(`/users/123/addresses/456`)

#### 2. HTTP 메서드

HTTP 메서드를 의미론적으로 사용:

| 메서드 | 목적 | 멱등성? | 안전성? |
|--------|---------|-------------|-------|
| GET | 리소스 조회 | Yes | Yes |
| POST | 리소스 생성 | No | No |
| PUT | 리소스 교체 | Yes | No |
| PATCH | 부분 업데이트 | No | No |
| DELETE | 리소스 제거 | Yes | No |

**멱등성(Idempotent)**: 동일한 요청을 여러 번 해도 한 번의 요청과 같은 효과
**안전성(Safe)**: 요청이 서버 상태를 변경하지 않음

```javascript
// Express.js 예제
const express = require('express');
const app = express();

app.get('/users/:id', (req, res) => {
  const user = userService.findById(req.params.id);
  res.json(user);
});

app.post('/users', (req, res) => {
  const user = userService.create(req.body);
  res.status(201).json(user);
});

app.put('/users/:id', (req, res) => {
  const user = userService.replace(req.params.id, req.body);
  res.json(user);
});

app.patch('/users/:id', (req, res) => {
  const user = userService.update(req.params.id, req.body);
  res.json(user);
});

app.delete('/users/:id', (req, res) => {
  userService.delete(req.params.id);
  res.status(204).send();
});
```

#### 3. HTTP 상태 코드

결과를 전달하기 위해 적절한 상태 코드 사용:

**2xx 성공**:
- `200 OK`: 표준 성공 응답
- `201 Created`: 리소스 생성됨(POST)
- `204 No Content`: 응답 본문 없이 성공(DELETE)

**4xx 클라이언트 오류**:
- `400 Bad Request`: 유효하지 않은 입력
- `401 Unauthorized`: 인증 필요
- `403 Forbidden`: 인증되었지만 권한 없음
- `404 Not Found`: 리소스가 존재하지 않음
- `409 Conflict`: 리소스 상태 충돌
- `422 Unprocessable Entity`: 유효성 검증 오류

**5xx 서버 오류**:
- `500 Internal Server Error`: 일반 서버 오류
- `503 Service Unavailable`: 일시적 중단

```python
# Flask 예제
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({'error': 'Email is required'}), 400

    if user_exists(data['email']):
        return jsonify({'error': 'User already exists'}), 409

    user = create_user_in_db(data)
    return jsonify(user), 201

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = find_user(user_id)

    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify(user), 200
```

#### 4. 페이지네이션

컬렉션의 경우, 압도적인 응답을 방지하기 위해 페이지네이션 구현:

**오프셋 기반 페이지네이션**:
```
GET /users?offset=20&limit=10

Response:
{
  "data": [...],
  "pagination": {
    "offset": 20,
    "limit": 10,
    "total": 150
  }
}
```

**커서 기반 페이지네이션**(실시간 데이터에 더 적합):
```
GET /users?cursor=eyJpZCI6MTIzfQ==&limit=10

Response:
{
  "data": [...],
  "next_cursor": "eyJpZCI6MTMzfQ=="
}
```

#### 5. 필터링, 정렬, 검색

유연한 데이터 조회를 위한 쿼리 매개변수 제공:

```
GET /users?role=admin&sort=-created_at&search=john
```

```java
// Java Spring Boot 예제
@RestController
@RequestMapping("/users")
public class UserController {

    @GetMapping
    public ResponseEntity<Page<User>> getUsers(
            @RequestParam(required = false) String role,
            @RequestParam(required = false) String search,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            @RequestParam(defaultValue = "id") String sortBy,
            @RequestParam(defaultValue = "ASC") Sort.Direction direction
    ) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        Page<User> users = userService.findUsers(role, search, pageable);
        return ResponseEntity.ok(users);
    }
}
```

#### 6. HATEOAS

**HATEOAS(Hypermedia As The Engine Of Application State)**는 응답에 관련 리소스에 대한 링크를 포함합니다:

```json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "_links": {
    "self": { "href": "/users/123" },
    "orders": { "href": "/users/123/orders" },
    "addresses": { "href": "/users/123/addresses" }
  }
}
```

이는 API를 자체 문서화하고 발견 가능하게 만듭니다.

## RPC 스타일 API

**RPC(Remote Procedure Call)** API는 리소스가 아닌 함수 호출로 작업을 모델링합니다.

### gRPC

gRPC는 효율적인 바이너리 직렬화를 위해 **프로토콜 버퍼(Protocol Buffers)**를 사용합니다:

```protobuf
// user.proto
syntax = "proto3";

service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc CreateUser (CreateUserRequest) returns (User);
  rpc ListUsers (ListUsersRequest) returns (ListUsersResponse);
}

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}

message GetUserRequest {
  int32 id = 1;
}
```

```python
# Python gRPC 서버
import grpc
from concurrent import futures
import user_pb2
import user_pb2_grpc

class UserService(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        user = find_user(request.id)
        if not user:
            context.abort(grpc.StatusCode.NOT_FOUND, 'User not found')
        return user_pb2.User(id=user.id, name=user.name, email=user.email)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
user_pb2_grpc.add_UserServiceServicer_to_server(UserService(), server)
server.add_insecure_port('[::]:50051')
server.start()
```

### RPC vs REST 언제 사용할까

**RPC(gRPC)를 사용할 때**:
- 고성능과 낮은 지연이 중요한 경우
- 양방향 스트리밍이 필요한 경우
- 내부 마이크로서비스 통신
- 타입 안전성이 중요한 경우

**REST를 사용할 때**:
- 공개 API
- 광범위한 클라이언트 호환성(브라우저, 모바일)
- 사람이 읽을 수 있는 디버깅
- 캐싱과 표준 HTTP 툴링

## GraphQL

**GraphQL**은 클라이언트가 필요한 데이터를 정확히 요청할 수 있게 하는 쿼리 언어입니다.

### 스키마 정의

```graphql
type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]!
}

type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
}

type Mutation {
  createUser(name: String!, email: String!): User!
  updateUser(id: ID!, name: String, email: String): User!
}
```

### 쿼리와 뮤테이션

```javascript
// 클라이언트 쿼리 - 필요한 필드만 요청
const query = `
  query GetUser($id: ID!) {
    user(id: $id) {
      name
      email
      orders {
        total
      }
    }
  }
`;

// 서버 리졸버(Node.js)
const resolvers = {
  Query: {
    user: (parent, { id }, context) => {
      return context.db.findUserById(id);
    },
  },
  User: {
    orders: (user, args, context) => {
      return context.db.findOrdersByUserId(user.id);
    },
  },
  Mutation: {
    createUser: (parent, { name, email }, context) => {
      return context.db.createUser({ name, email });
    },
  },
};
```

### N+1 문제

GraphQL은 중첩 필드를 해결할 때 성능 문제를 일으킬 수 있습니다:

```graphql
query {
  users {
    name
    orders {  # N+1: 사용자당 하나의 쿼리
      total
    }
  }
}
```

**해결책: DataLoader**는 요청을 배치 처리하고 캐시합니다:

```javascript
const DataLoader = require('dataloader');

const orderLoader = new DataLoader(async (userIds) => {
  const orders = await db.findOrdersByUserIds(userIds);
  return userIds.map(id => orders.filter(o => o.userId === id));
});

const resolvers = {
  User: {
    orders: (user, args, { orderLoader }) => {
      return orderLoader.load(user.id);
    },
  },
};
```

### GraphQL이 빛나는 경우

**GraphQL을 사용할 때**:
- 클라이언트가 유연한 데이터 가져오기를 필요로 할 때(모바일, 웹 대시보드)
- 과다 가져오기(over-fetching)나 부족 가져오기(under-fetching)를 피하고 싶을 때
- 백엔드 변경 없이 빠른 프론트엔드 반복

**GraphQL을 피해야 할 때**:
- 간단한 CRUD 작업(REST가 더 간단)
- 파일 업로드(GraphQL은 파일 처리가 취약)
- 캐싱이 중요한 경우(HTTP 캐싱이 REST에서 더 잘 작동)

## API 설계 원칙

### 1. 일관성

일관성은 인지 부하를 줄입니다. 규칙을 정하고 따르세요:

**네이밍 규칙**:
```
# 일관성 있음
GET /users
GET /orders
GET /products

# 일관성 없음
GET /users
GET /getOrders
GET /product_list
```

**응답 형식**:
```json
// 일관된 성공 응답
{
  "data": { ... },
  "meta": { ... }
}

// 일관된 오류 응답
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Email is required",
    "fields": { "email": "This field is required" }
  }
}
```

### 2. 최소 놀람의 원칙

API는 사용자가 예상하는 대로 동작해야 합니다. 놀라운 동작을 피하세요:

```python
# 놀라움: 메서드 이름이 읽기 전용을 암시하지만 상태를 변경
def get_user_or_create(email):
    user = find_user(email)
    if not user:
        user = create_user(email)  # 놀람!
    return user

# 명확함: 이름이 가능한 변경을 나타냄
def find_or_create_user(email):
    user = find_user(email)
    if not user:
        user = create_user(email)
    return user
```

### 3. 멱등성

멱등성 작업은 부작용 없이 안전하게 재시도할 수 있습니다:

```python
# 멱등성: PUT은 전체 리소스를 교체
PUT /users/123
{ "name": "John", "email": "john@example.com" }

# 비멱등성: PATCH는 증가시킬 수 있음
PATCH /users/123
{ "login_count": 5 }  # 재시도하면 어떻게 될까?

# 더 나음: 서버 제어 증가
POST /users/123/login
```

### 4. 하위 호환성

기존 클라이언트를 절대 깨뜨리지 마세요. API를 신중하게 발전시키세요:

**안전한 변경**:
- 선택적 필드 추가
- 새 엔드포인트 추가
- 필드 폐기(유예 기간 포함)

**파괴적 변경**:
- 필드 제거
- 필드 타입 변경
- 필드 이름 변경
- 유효성 검증 규칙 변경

```javascript
// 하위 호환: 새로운 선택적 필드
{
  "name": "John",
  "email": "john@example.com",
  "phone": "555-1234"  // 새로운, 선택적
}

// 파괴적: 필드 제거
{
  "name": "John"
  // "email" 제거 - 클라이언트가 깨짐!
}
```

## 버전 관리 전략

파괴적 변경이 필요할 때, API를 버전 관리하세요:

### 1. URL 버전 관리
```
GET /v1/users
GET /v2/users
```
**장점**: 명확하고, 라우팅하기 쉬움
**단점**: URL 변경, 캐싱 문제

### 2. 헤더 버전 관리
```
GET /users
Accept: application/vnd.myapi.v2+json
```
**장점**: 깨끗한 URL
**단점**: 가시성 낮음, 브라우저에서 테스트하기 어려움

### 3. 쿼리 매개변수
```
GET /users?version=2
```
**장점**: 테스트하기 쉬움
**단점**: 버전 관리와 필터링이 혼재

**권장사항**: 주요 버전에는 URL 버전 관리, 사소한 변경에는 기능 플래그 사용.

## 인증과 권한 부여

### API 키
```
GET /users
X-API-Key: abc123xyz
```
간단하지만 보안이 약함. 낮은 위험도의 내부 API에 사용.

### OAuth 2.0
```
GET /users
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```
위임된 권한 부여를 위한 업계 표준. 서드파티 통합에 사용.

### JWT(JSON Web Tokens)
```javascript
// Node.js JWT 예제
const jwt = require('jsonwebtoken');

// 토큰 생성
const token = jwt.sign(
  { userId: 123, role: 'admin' },
  process.env.SECRET_KEY,
  { expiresIn: '1h' }
);

// 토큰 검증
function authenticate(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).send('Unauthorized');

  try {
    const payload = jwt.verify(token, process.env.SECRET_KEY);
    req.user = payload;
    next();
  } catch (err) {
    res.status(403).send('Invalid token');
  }
}
```

## 속도 제한과 스로틀링

API를 남용으로부터 보호하세요:

```python
# Python Flask 속도 제한
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/search')
@limiter.limit("10 per minute")
def search():
    return perform_expensive_search()
```

**응답 헤더**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1614556800
```

제한 초과 시 `429 Too Many Requests` 반환.

## 오류 응답

일관되고 유용한 오류 메시지 제공:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Email is required"
      },
      {
        "field": "age",
        "message": "Age must be at least 18"
      }
    ],
    "request_id": "req_abc123"
  }
}
```

```cpp
// C++ 라이브러리 API 오류 처리
class ApiException : public std::exception {
private:
    std::string message_;
    int code_;
public:
    ApiException(int code, const std::string& msg)
        : code_(code), message_(msg) {}

    int code() const { return code_; }
    const char* what() const noexcept override {
        return message_.c_str();
    }
};

class UserService {
public:
    User getUser(int id) {
        if (id < 0) {
            throw ApiException(400, "Invalid user ID");
        }
        auto user = db.findById(id);
        if (!user) {
            throw ApiException(404, "User not found");
        }
        return user;
    }
};
```

## 문서화

훌륭한 API는 훌륭한 문서를 가지고 있습니다.

### OpenAPI(Swagger)

```yaml
# openapi.yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
paths:
  /users/{userId}:
    get:
      summary: Get user by ID
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
```

**장점**:
- 자동 생성된 인터랙티브 문서
- 클라이언트 SDK 생성
- 유효성 검증

### 필수 문서 요소

1. **시작하기**: 첫 API 호출을 위한 빠른 예제
2. **인증**: 자격 증명을 얻고 사용하는 방법
3. **속도 제한**: 스로틀링 규칙과 헤더
4. **오류 코드**: 예제가 포함된 포괄적 목록
5. **SDK**: 인기 있는 언어용 클라이언트 라이브러리
6. **변경 로그**: 버전 히스토리와 마이그레이션 가이드

## 내부 API 설계(라이브러리와 모듈)

### 최소 표면적

필요한 것만 노출:

```java
// 나쁨: 내부 사항 노출
public class UserService {
    public Database db;  // 노출됨!
    public void validateEmail(String email) { ... }  // 공개이지만 내부용
}

// 좋음: 최소 인터페이스
public class UserService {
    private Database db;

    public User getUser(int id) { ... }
    public User createUser(UserData data) { ... }

    private void validateEmail(String email) { ... }  // 비공개
}
```

### 캡슐화

구현 세부사항 숨기기:

```python
# 나쁨: 구현 노출
class ShoppingCart:
    def __init__(self):
        self.items = []  # 내부 리스트에 직접 접근

    def add_item(self, item):
        self.items.append(item)

cart = ShoppingCart()
cart.items.append(invalid_item)  # 유효성 검증 우회 가능!

# 좋음: 캡슐화됨
class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, item):
        self._validate_item(item)
        self._items.append(item)

    def get_items(self):
        return self._items.copy()  # 참조가 아닌 복사본 반환
```

### 유창한 인터페이스와 빌더 패턴

사용하기 즐거운 API 만들기:

```javascript
// 유창한 인터페이스
const query = new QueryBuilder()
  .select('name', 'email')
  .from('users')
  .where('age', '>', 18)
  .orderBy('name')
  .limit(10)
  .build();

// 빌더 패턴
const user = new UserBuilder()
  .withName('John Doe')
  .withEmail('john@example.com')
  .withRole('admin')
  .build();
```

### 일관된 네이밍과 매개변수 순서

```python
# 일관성: 리소스 먼저, 그 다음 옵션
user_service.get_user(user_id, include_deleted=False)
order_service.get_order(order_id, include_items=True)

# 일관성 없음
user_service.get_user(include_deleted=False, user_id=123)
order_service.find_order(123, True)  # True가 무엇을 의미하는가?
```

## 연습 문제

### 연습 문제 1: 서점을 위한 REST API 설계

다음 요구사항을 가진 온라인 서점을 위한 REST API를 설계하세요:
- 책 탐색(장르, 저자별 필터링)
- 책 검색
- 사용자 계정과 인증
- 장바구니
- 주문 배치
- 주문 내역

제공할 것:
1. HTTP 메서드가 포함된 엔드포인트 목록
2. 샘플 요청/응답 본문
3. 각 엔드포인트의 상태 코드
4. 페이지네이션 전략

### 연습 문제 2: API 검토

이 API 설계를 검토하고 좋은 API 설계 원칙의 위반 사항을 식별하세요:

```
GET /getBook?id=123
POST /createBook?title=NewBook&author=JohnDoe
GET /updateBook?id=123&title=UpdatedTitle
DELETE /books/remove/123
GET /books?page=1&limit=1000
POST /users/login  (자격 증명이 틀려도 200 반환)
```

무엇이 잘못되었는지 나열하고 개선안을 제안하세요.

### 연습 문제 3: 버전 관리 전략

현재 API에 다음 엔드포인트가 있습니다:
```
GET /users/123
Response: { "name": "John", "email": "john@example.com" }
```

파괴적 변경을 추가해야 합니다: `name`을 `first_name`과 `last_name`으로 분할. 다음을 수행하는 버전 관리 전략을 설계하세요:
1. 기존 클라이언트를 깨뜨리지 않음
2. 새 클라이언트가 개선된 구조를 사용할 수 있게 함
3. 마이그레이션 경로 제공

### 연습 문제 4: 오류 처리 설계

다음을 처리해야 하는 뱅킹 API를 위한 포괄적인 오류 응답 형식을 설계하세요:
- 유효성 검증 오류(여러 필드)
- 잔액 부족
- 계정 잠김
- 속도 제한
- 서버 오류

각 오류 유형에 대한 JSON 예제를 제공하세요.

### 연습 문제 5: 라이브러리 API 설계

다음을 지원하는 선호하는 언어로 SQL 쿼리 빌더를 위한 유창한 API를 설계하세요:
- 여러 컬럼으로 SELECT
- 테이블 이름으로 FROM
- 여러 조건으로 WHERE(AND/OR)
- JOIN 작업
- ORDER BY
- LIMIT과 OFFSET

사용 예제:
```
query.select('id', 'name')
     .from('users')
     .where('age > ?', 18)
     .orderBy('name')
     .limit(10)
```

## 요약

좋은 API 설계는 사용자에 대한 공감입니다:

- **REST**: 리소스 지향, HTTP 의미론 사용, 공개 API에 최적
- **RPC/gRPC**: 액션 지향, 효율적, 내부 마이크로서비스에 최적
- **GraphQL**: 유연한 쿼리, 데이터 중심 클라이언트 애플리케이션에 최적
- **일관성**: 예측 가능한 네이밍, 구조, 동작
- **멱등성**: PUT과 DELETE의 안전한 재시도
- **버전 관리**: 클라이언트를 깨뜨리지 않고 변경 관리
- **문서화**: 예제를 통해 API를 자명하게 만들기
- **오류 처리**: 명확하고 실행 가능한 오류 메시지

기억하세요: API는 제품입니다. API 소비자를 고객으로 대하세요—그들의 경험에 투자하세요.

## 내비게이션

[← 이전: 성능 최적화](12_Performance_Optimization.md) | [다음: 버전 관리 워크플로우 →](14_Version_Control_Workflows.md)
